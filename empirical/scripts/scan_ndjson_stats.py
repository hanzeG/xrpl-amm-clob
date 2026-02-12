#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

getcontext().prec = 28

DROPS_PER_XRP = Decimal("1000000")


def read_ndjson(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{path}: JSON decode error at line {i}: {e}") from e
    return rows


def pct(values: List[Decimal], p: float) -> Decimal:
    if not values:
        return Decimal(0)
    xs = sorted(values)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


def pct_int(values: List[int], p: float) -> int:
    if not values:
        return 0
    xs = sorted(values)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


def safe_dec(x: Any) -> Optional[Decimal]:
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError):
        return None


def drops_to_xrp(drops: Any) -> Optional[Decimal]:
    d = safe_dec(drops)
    if d is None:
        return None
    return d / DROPS_PER_XRP


def parse_iou_value(obj: Any) -> Optional[Decimal]:
    # {"currency": "...", "issuer": "...", "value": "..."}
    if not isinstance(obj, dict):
        return None
    return safe_dec(obj.get("value"))


def offer_price_xrp_per_rusd_raw(offer: Dict[str, Any]) -> Optional[Decimal]:
    """
    Price in XRP per rUSD computed ONLY from raw TakerGets/TakerPays (ignore funded).
    """
    tg = offer.get("TakerGets")
    tp = offer.get("TakerPays")

    # Case A: TakerGets = XRP drops, TakerPays = rUSD IOU
    if isinstance(tg, (str, int)) and isinstance(tp, dict):
        xrp = drops_to_xrp(tg)
        rusd = parse_iou_value(tp)
        if xrp is None or rusd is None or rusd == 0:
            return None
        p = xrp / rusd
        return p if p > 0 else None

    # Case B: TakerGets = rUSD IOU, TakerPays = XRP drops
    if isinstance(tg, dict) and isinstance(tp, (str, int)):
        rusd = parse_iou_value(tg)
        xrp = drops_to_xrp(tp)
        if xrp is None or rusd is None or rusd == 0:
            return None
        p = xrp / rusd
        return p if p > 0 else None

    return None


def offer_amounts_xrp_rusd_executable(offer: Dict[str, Any]) -> Optional[Tuple[Decimal, Decimal]]:
    """
    Executable amounts (xrp_amt, rusd_amt). Prefer funded fields if present; otherwise use raw.
    """
    tg = offer.get("TakerGets")
    tp = offer.get("TakerPays")

    tg_f = offer.get("taker_gets_funded")
    tp_f = offer.get("taker_pays_funded")

    # Case A: raw TakerGets is XRP, raw TakerPays is rUSD
    if isinstance(tg, (str, int)) and isinstance(tp, dict):
        xrp = drops_to_xrp(tg_f) if tg_f is not None else drops_to_xrp(tg)
        rusd = parse_iou_value(tp_f) if tp_f is not None else parse_iou_value(tp)
        if xrp is None or rusd is None:
            return None
        return (xrp, rusd)

    # Case B: raw TakerGets is rUSD, raw TakerPays is XRP
    if isinstance(tg, dict) and isinstance(tp, (str, int)):
        rusd = parse_iou_value(tg_f) if tg_f is not None else parse_iou_value(tg)
        xrp = drops_to_xrp(tp_f) if tp_f is not None else drops_to_xrp(tp)
        if xrp is None or rusd is None:
            return None
        return (xrp, rusd)

    return None

@dataclass
class SideBookStats:
    offers_count: int
    parsed_offers: int
    funded_fields_present: int
    zero_funded_like: int
    unique_accounts: int
    prices: List[Decimal]  # xrp per rusd
    best_price: Optional[Decimal]  # meaning depends on side (min for asks / max for bids)
    top_depth: Dict[int, Tuple[Decimal, Decimal]]  # N -> (xrp_sum, rusd_sum)
    price_level_count: int


def summarise_rows_basic(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    offers = [int(r.get("offers_count", 0)) for r in rows]
    has_hash = [r.get("ledger_hash") is not None for r in rows]

    return {
        "n_rows": len(rows),
        "offers_min": min(offers) if offers else 0,
        "offers_mean": float(mean(offers)) if offers else 0.0,
        "offers_max": max(offers) if offers else 0,
        "offers_p50": pct_int(offers, 50),
        "offers_p90": pct_int(offers, 90),
        "offers_p99": pct_int(offers, 99),
        "ledger_hash_missing": sum(1 for x in has_hash if not x),
    }


def ledger_index_set(rows: List[Dict[str, Any]]) -> Tuple[Counter, set]:
    c = Counter()
    s = set()
    for r in rows:
        li = r.get("ledger_index")
        if li is None:
            continue
        c[int(li)] += 1
        s.add(int(li))
    return c, s


def book_stats_for_one_ledger(row: Dict[str, Any], side: str, topNs=(1, 5, 10, 20, 50)) -> SideBookStats:
    offers = row.get("offers", []) or []
    offers_count = int(row.get("offers_count", len(offers)))

    accounts = set()
    prices: List[Decimal] = []
    funded_fields_present = 0
    zero_funded_like = 0

    parsed = []
    for off in offers:
        if isinstance(off, dict) and "Account" in off:
            accounts.add(off["Account"])
        if "taker_gets_funded" in off or "taker_pays_funded" in off:
            funded_fields_present += 1

        p = offer_price_xrp_per_rusd_raw(off)
        if p is None:
            continue

        amt = offer_amounts_xrp_rusd_executable(off)
        if amt is None:
            continue
        xrp_amt, rusd_amt = amt

        # treat (xrp==0 or rusd==0) as zero-funded-like (not executable)
        if xrp_amt == 0 or rusd_amt == 0:
            zero_funded_like += 1

        prices.append(p)
        parsed.append((p, xrp_amt, rusd_amt))

    # sort for depth calc
    # bids: higher price better (more XRP per rUSD you get when selling rUSD)
    # asks: lower price better (less XRP per rUSD you pay when buying rUSD)
    if side == "bid":
        parsed.sort(key=lambda t: t[0], reverse=True)
        best_price = parsed[0][0] if parsed else None
    else:
        parsed.sort(key=lambda t: t[0])
        best_price = parsed[0][0] if parsed else None

    # count unique price levels (rough, by exact decimal string)
    price_level_count = len({str(p) for p in (t[0] for t in parsed)})

    # depth at top N offers (not levels)
    top_depth: Dict[int, Tuple[Decimal, Decimal]] = {}
    for N in topNs:
        x_sum = sum((t[1] for t in parsed[:N]), Decimal(0))
        r_sum = sum((t[2] for t in parsed[:N]), Decimal(0))
        top_depth[N] = (x_sum, r_sum)

    return SideBookStats(
        offers_count=offers_count,
        parsed_offers=len(parsed),
        funded_fields_present=funded_fields_present,
        zero_funded_like=zero_funded_like,
        unique_accounts=len(accounts),
        prices=prices,
        best_price=best_price,
        top_depth=top_depth,
        price_level_count=price_level_count,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--getsxrp", required=True, help="book_rusd_xrp_getsXRP.ndjson")
    ap.add_argument("--getsrusd", required=True, help="book_rusd_xrp_getsrUSD.ndjson")
    ap.add_argument("--topn", type=int, default=10, help="Top-N offers for depth reporting (e.g. 10)")
    ap.add_argument("--show_examples", type=int, default=5, help="Show N ledgers with widest spread")
    args = ap.parse_args()

    a = read_ndjson(args.getsxrp)   # taker_gets = XRP => this side is BID for rUSD (you sell rUSD, receive XRP)
    b = read_ndjson(args.getsrusd)  # taker_gets = rUSD => this side is ASK for rUSD (you buy rUSD, pay XRP)

    # --------------------------
    # Partition A: existing/basic integrity metrics
    # --------------------------
    sa = summarise_rows_basic(a)
    sb = summarise_rows_basic(b)

    ca, seta = ledger_index_set(a)
    cb, setb = ledger_index_set(b)

    only_a = sorted(seta - setb)
    only_b = sorted(setb - seta)
    dup_a = sum(1 for _, v in ca.items() if v > 1)
    dup_b = sum(1 for _, v in cb.items() if v > 1)

    common = sorted(seta & setb)

    print("\n=== Partition A: dataset / alignment ===")
    print(f"getsXRP file   : {args.getsxrp}")
    print(f"getsrUSD file  : {args.getsrusd}")
    print(f"rows (getsXRP) : {sa['n_rows']}")
    print(f"rows (getsrUSD): {sb['n_rows']}")
    print(f"common ledgers : {len(common)}")
    print(f"duplicates (getsXRP/getsrUSD): {dup_a} / {dup_b}")
    print(f"missing ledgers (only in getsXRP/getsrUSD): {len(only_a)} / {len(only_b)}")
    print(f"ledger_hash missing (getsXRP/getsrUSD): {sa['ledger_hash_missing']} / {sb['ledger_hash_missing']}")
    print(f"offers_count min/mean/max (getsXRP): {sa['offers_min']} / {sa['offers_mean']:.2f} / {sa['offers_max']}")
    print(f"offers_count min/mean/max (getsrUSD): {sb['offers_min']} / {sb['offers_mean']:.2f} / {sb['offers_max']}")

    # --------------------------
    # Partition B: offer-level market metrics
    # --------------------------
    spreads: List[Decimal] = []
    spreads_bps: List[Decimal] = []
    mids: List[Decimal] = []
    best_bids: List[Decimal] = []
    best_asks: List[Decimal] = []

    bid_parsed_rates: List[Decimal] = []
    ask_parsed_rates: List[Decimal] = []

    bid_price_levels: List[int] = []
    ask_price_levels: List[int] = []

    bid_unique_accounts: List[int] = []
    ask_unique_accounts: List[int] = []

    bid_funded_share: List[Decimal] = []
    ask_funded_share: List[Decimal] = []

    bid_zero_funded_share: List[Decimal] = []
    ask_zero_funded_share: List[Decimal] = []

    # depth at topN
    topN = args.topn
    bid_depth_x: List[Decimal] = []
    bid_depth_r: List[Decimal] = []
    ask_depth_x: List[Decimal] = []
    ask_depth_r: List[Decimal] = []

    # examples: keep widest spreads
    examples = []

    # index rows by ledger for quick lookup
    a_by = {int(r["ledger_index"]): r for r in a if "ledger_index" in r}
    b_by = {int(r["ledger_index"]): r for r in b if "ledger_index" in r}

    for li in common:
        ra = a_by.get(li)
        rb = b_by.get(li)
        if ra is None or rb is None:
            continue

        bid = book_stats_for_one_ledger(ra, side="bid", topNs=(topN,))
        ask = book_stats_for_one_ledger(rb, side="ask", topNs=(topN,))

        # store parse coverage
        if bid.parsed_offers > 0:
            bid_parsed_rates.append(Decimal(bid.parsed_offers) / Decimal(max(1, bid.offers_count)))
        else:
            bid_parsed_rates.append(Decimal(0))

        if ask.parsed_offers > 0:
            ask_parsed_rates.append(Decimal(ask.parsed_offers) / Decimal(max(1, ask.offers_count)))
        else:
            ask_parsed_rates.append(Decimal(0))

        bid_price_levels.append(bid.price_level_count)
        ask_price_levels.append(ask.price_level_count)

        bid_unique_accounts.append(bid.unique_accounts)
        ask_unique_accounts.append(ask.unique_accounts)

        bid_funded_share.append(Decimal(bid.funded_fields_present) / Decimal(max(1, bid.offers_count)))
        ask_funded_share.append(Decimal(ask.funded_fields_present) / Decimal(max(1, ask.offers_count)))

        bid_zero_funded_share.append(Decimal(bid.zero_funded_like) / Decimal(max(1, bid.parsed_offers)))
        ask_zero_funded_share.append(Decimal(ask.zero_funded_like) / Decimal(max(1, ask.parsed_offers)))

        # best bid/ask and spread
        if bid.best_price is not None:
            best_bids.append(bid.best_price)
        if ask.best_price is not None:
            best_asks.append(ask.best_price)

        if bid.best_price is None or ask.best_price is None:
            continue

        best_bid = bid.best_price
        best_ask = ask.best_price

        mid = (best_bid + best_ask) / Decimal(2)
        spread = best_ask - best_bid  # XRP per rUSD
        mids.append(mid)
        spreads.append(spread)

        # bps relative to mid
        if mid != 0:
            spreads_bps.append(spread / mid * Decimal(10000))

        # depth at topN offers
        bx, br = bid.top_depth[topN]
        ax, ar = ask.top_depth[topN]
        bid_depth_x.append(bx)
        bid_depth_r.append(br)
        ask_depth_x.append(ax)
        ask_depth_r.append(ar)

        examples.append((spread, li, best_bid, best_ask, mid, bx, br, ax, ar))

    examples.sort(key=lambda t: t[0], reverse=True)

    def fmt(d: Decimal, places: int = 12) -> str:
        return f"{d:.{places}f}"

    print("\n=== Partition B: offer-level market metrics (XRP per rUSD) ===")

    # parse coverage
    print("\n--- Parsing / coverage ---")
    print(f"parsed_offers/offer_count (bid side) p50/p90/p99: "
          f"{fmt(pct(bid_parsed_rates, 50), 4)} / {fmt(pct(bid_parsed_rates, 90), 4)} / {fmt(pct(bid_parsed_rates, 99), 4)}")
    print(f"parsed_offers/offer_count (ask side) p50/p90/p99: "
          f"{fmt(pct(ask_parsed_rates, 50), 4)} / {fmt(pct(ask_parsed_rates, 90), 4)} / {fmt(pct(ask_parsed_rates, 99), 4)}")

    # best bid/ask
    if best_bids and best_asks:
        print("\n--- Best prices ---")
        print(f"best_bid (XRP/rUSD) p50/p90/p99: {fmt(pct(best_bids, 50), 12)} / {fmt(pct(best_bids, 90), 12)} / {fmt(pct(best_bids, 99), 12)}")
        print(f"best_ask (XRP/rUSD) p50/p90/p99: {fmt(pct(best_asks, 50), 12)} / {fmt(pct(best_asks, 90), 12)} / {fmt(pct(best_asks, 99), 12)}")

    # spread
    if spreads:
        print("\n--- Spread (best_ask - best_bid) ---")
        print(f"spread (XRP/rUSD) min/p50/p90/p99/max: "
              f"{fmt(min(spreads), 12)} / {fmt(pct(spreads, 50), 12)} / {fmt(pct(spreads, 90), 12)} / {fmt(pct(spreads, 99), 12)} / {fmt(max(spreads), 12)}")
        if spreads_bps:
            print(f"spread (bps, vs mid) min/p50/p90/p99/max: "
                  f"{fmt(min(spreads_bps), 4)} / {fmt(pct(spreads_bps, 50), 4)} / {fmt(pct(spreads_bps, 90), 4)} / {fmt(pct(spreads_bps, 99), 4)} / {fmt(max(spreads_bps), 4)}")

    # depth at topN
    if bid_depth_x and ask_depth_x:
        print(f"\n--- Depth at top {topN} offers (not price-level aggregated) ---")
        print(f"bid side: XRP sum p50/p90/p99 = {fmt(pct(bid_depth_x, 50), 6)} / {fmt(pct(bid_depth_x, 90), 6)} / {fmt(pct(bid_depth_x, 99), 6)}")
        print(f"bid side: rUSD sum p50/p90/p99 = {fmt(pct(bid_depth_r, 50), 6)} / {fmt(pct(bid_depth_r, 90), 6)} / {fmt(pct(bid_depth_r, 99), 6)}")
        print(f"ask side: XRP sum p50/p90/p99 = {fmt(pct(ask_depth_x, 50), 6)} / {fmt(pct(ask_depth_x, 90), 6)} / {fmt(pct(ask_depth_x, 99), 6)}")
        print(f"ask side: rUSD sum p50/p90/p99 = {fmt(pct(ask_depth_r, 50), 6)} / {fmt(pct(ask_depth_r, 90), 6)} / {fmt(pct(ask_depth_r, 99), 6)}")

    # structure metrics
    if bid_price_levels and ask_price_levels:
        print("\n--- Book structure (rough) ---")
        print(f"price levels (bid) p50/p90/p99: {pct_int(bid_price_levels, 50)} / {pct_int(bid_price_levels, 90)} / {pct_int(bid_price_levels, 99)}")
        print(f"price levels (ask) p50/p90/p99: {pct_int(ask_price_levels, 50)} / {pct_int(ask_price_levels, 90)} / {pct_int(ask_price_levels, 99)}")

    if bid_unique_accounts and ask_unique_accounts:
        print("\n--- Participation ---")
        print(f"unique accounts in returned offers (bid) p50/p90/p99: {pct_int(bid_unique_accounts, 50)} / {pct_int(bid_unique_accounts, 90)} / {pct_int(bid_unique_accounts, 99)}")
        print(f"unique accounts in returned offers (ask) p50/p90/p99: {pct_int(ask_unique_accounts, 50)} / {pct_int(ask_unique_accounts, 90)} / {pct_int(ask_unique_accounts, 99)}")

    # funded fields stats
    if bid_funded_share and ask_funded_share:
        print("\n--- Funded-field presence (how often taker_*_funded appears) ---")
        print(f"funded-field share (bid) p50/p90/p99: {fmt(pct(bid_funded_share, 50), 4)} / {fmt(pct(bid_funded_share, 90), 4)} / {fmt(pct(bid_funded_share, 99), 4)}")
        print(f"funded-field share (ask) p50/p90/p99: {fmt(pct(ask_funded_share, 50), 4)} / {fmt(pct(ask_funded_share, 90), 4)} / {fmt(pct(ask_funded_share, 99), 4)}")

    if bid_zero_funded_share and ask_zero_funded_share:
        print("\n--- Zero-funded-like share among parsed offers (xrp==0 or rusd==0) ---")
        print(f"zero-funded share (bid) p50/p90/p99: {fmt(pct(bid_zero_funded_share, 50), 4)} / {fmt(pct(bid_zero_funded_share, 90), 4)} / {fmt(pct(bid_zero_funded_share, 99), 4)}")
        print(f"zero-funded share (ask) p50/p90/p99: {fmt(pct(ask_zero_funded_share, 50), 4)} / {fmt(pct(ask_zero_funded_share, 90), 4)} / {fmt(pct(ask_zero_funded_share, 99), 4)}")

    # examples
    if examples and args.show_examples > 0:
        print(f"\n--- Examples: widest {args.show_examples} spreads ---")
        for spread, li, bb, ba, mid, bx, br, ax, ar in examples[:args.show_examples]:
            print(
                f"ledger {li}: spread={fmt(spread, 12)} XRP/rUSD, "
                f"bid={fmt(bb, 12)}, ask={fmt(ba, 12)}, mid={fmt(mid, 12)}; "
                f"top{topN} bid_depth: {fmt(bx, 6)} XRP / {fmt(br, 6)} rUSD; "
                f"top{topN} ask_depth: {fmt(ax, 6)} XRP / {fmt(ar, 6)} rUSD"
            )


if __name__ == "__main__":
    main()