# empirical/scripts/compare_rusd_xrp_100894647_vs_model.py
# Compare model vs real execution for rUSD-XRP in ledger 100894647 (2025-12-15 snapshot).
# - Pre CLOB: ledger 100894646 book_offers (two directions, 200 offers each)
# - Pre AMM: from amm_swaps *_before + trading_fee from amm_fees
# - Model: RouterQuoteView.preview_out (tries multiple ladder encodings; picks minimal total_in)
# - Real: AMM leg from amm_swaps + CLOB legs from clob_tx
# - If CLOB depth exhausted or quote becomes partial => raise

import json
import os
from decimal import Decimal, ROUND_FLOOR, getcontext
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import SparkSession, functions as F

from xrpl_router.amm import AMM
from xrpl_router.book_step import RouterQuoteView
from xrpl_router.core import Quality, IOUAmount, XRPAmount
from xrpl_router.core.datatypes import Segment
from xrpl_router.core.constants import XRP_QUANTUM, IOU_QUANTUM
from xrpl_router.core.fmt import amount_to_decimal, quantize_down
from xrpl_router.core.exc import InsufficientLiquidityError

getcontext().prec = 40

LEDGER_INDEX = 100894647
ROOT = "artifacts/snapshots/legacy_root/ledger/ledger_100894647_20251215"

AMM_SWAPS_PARQUET = f"{ROOT}/amm_swaps/parquet"
AMM_FEES_PARQUET  = f"{ROOT}/amm_fees/parquet"
CLOB_TX_PARQUET   = f"{ROOT}/clob_tx/parquet"

BOOK_GETS_RUSD_JSON = f"{ROOT}/100894646_offers_getsrUSD.json"
BOOK_GETS_XRP_JSON  = f"{ROOT}/100894646_offers_getsXRP.json"


RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"

PRINT_DP: Optional[int] = 6


def fmt(v: Any, dp: Optional[int] = None) -> str:
    """Format numbers for printing only; never use in computations."""
    if v is None:
        return "None"

    # Default behaviour: if PRINT_DP is not defined or is None, print full precision.
    if dp is None:
        try:
            dp = PRINT_DP  # type: ignore[name-defined]
        except Exception:
            dp = None

    try:
        d = v if isinstance(v, Decimal) else Decimal(str(v))
    except Exception:
        return str(v)

    if dp is None:
        return str(d)

    q = Decimal(1).scaleb(-int(dp))  # 10^-dp
    try:
        return str(d.quantize(q))
    except Exception:
        return str(d)


# Debug toggles
DEBUG_QUOTE = bool(int(os.environ.get("DEBUG_QUOTE", "0")))
DEBUG_BOOK  = bool(int(os.environ.get("DEBUG_BOOK", "0")))


def _pp(obj: Any) -> None:
    try:
        import pprint
        pprint.pprint(obj)
    except Exception:
        print(obj)


def _tier_debug(tiers: List[Tuple[Quality, Any, Any, Any]], in_cur: str, out_cur: str, n: int = 10) -> None:
    """Print first N tiers with implied price and XRPL-provided quality (if present)."""
    print(f"\n[tier_debug] {pretty_cur(in_cur)}->{pretty_cur(out_cur)} showing first {min(n, len(tiers))} tiers")
    for i, (q, out_amt, in_amt, xrpl_q_raw) in enumerate(tiers[:n]):
        out_d = amount_to_decimal(out_amt)
        in_d = amount_to_decimal(in_amt)
        try:
            q_m, q_e = q.mantissa_exponent()
        except Exception:
            q_m, q_e = (None, None)
        px_in_per_out = (in_d / out_d) if out_d != 0 else None
        px_out_per_in = (out_d / in_d) if in_d != 0 else None
        print(
            f"  tier[{i}] out={out_d} {pretty_cur(out_cur)} in={in_d} {pretty_cur(in_cur)} "
            f"px_in_per_out={px_in_per_out} px_out_per_in={px_out_per_in} "
            f"q=(m={q_m}, e={q_e}) xrpl_quality={xrpl_q_raw}"
        )


def _dump_sample_offers(offers: List[Dict[str, Any]], in_cur: str, out_cur: str, n: int = 5) -> None:
    print(f"\n[offer_sample] Showing {min(n, len(offers))} raw offers (pre-filter) for {pretty_cur(in_cur)}->{pretty_cur(out_cur)}")
    for i, o in enumerate(offers[:n]):
        tg = o.get("TakerGets")
        tp = o.get("TakerPays")
        q  = o.get("quality")
        tgf = o.get("taker_gets_funded")
        tpf = o.get("taker_pays_funded")
        of  = o.get("owner_funds")
        print(f"  offer[{i}] quality={q} owner_funds={of}")
        print(f"    TakerGets={tg}")
        print(f"    TakerPays={tp}")
        if tgf is not None or tpf is not None:
            print(f"    taker_gets_funded={tgf}")
            print(f"    taker_pays_funded={tpf}")


def _dump_tier_mapping(tiers: List[Tuple[Quality, Any, Any, Any]], in_cur: str, out_cur: str, n: int = 10) -> None:
    print(f"\n[tier_mapping] First {min(n, len(tiers))} tiers with decimalised amounts for {pretty_cur(in_cur)}->{pretty_cur(out_cur)}")
    for i, (q, out_amt, in_amt, xrpl_q_raw) in enumerate(tiers[:n]):
        out_d = amount_to_decimal(out_amt)
        in_d = amount_to_decimal(in_amt)
        try:
            q_m, q_e = q.mantissa_exponent()
        except Exception:
            q_m, q_e = (None, None)
        print(f"  tier[{i}] out_d={out_d} {pretty_cur(out_cur)} in_d={in_d} {pretty_cur(in_cur)} q=(m={q_m},e={q_e}) xrpl_quality={xrpl_q_raw}")


def assert_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")


def D(x: Any) -> Decimal:
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


def pretty_cur(c: str) -> str:
    return "rUSD" if c == RUSD_HEX else c


def is_xrp(cur: str) -> bool:
    return cur == XRP


def parse_amount(a: Any) -> Tuple[str, Optional[str], Decimal]:
    """
    XRPL JSON amount:
      - XRP: string of drops
      - IOU: {currency, issuer, value}
    Returns value in units (XRP not drops).
    """
    if isinstance(a, str):
        return (XRP, None, Decimal(a) * XRP_QUANTUM)  # drops -> XRP
    if isinstance(a, dict):
        return (a.get("currency"), a.get("issuer"), Decimal(str(a.get("value"))))
    raise ValueError(f"Unrecognised amount: {a!r}")


def load_book_offers(path: str) -> List[Dict[str, Any]]:
    """
    Supports both:
      (A) wrapped-RPC: {"result": {"offers": [...]}}
      (B) saved format: {"offers": [...], "offers_count": ..., "ledger_index": ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    offers = obj.get("offers", None)
    if offers is None:
        offers = obj.get("result", {}).get("offers", [])

    if not offers:
        raise ValueError(f"No offers in {path}")
    return offers


def amt_floor(cur: str, v: Decimal):
    """Convert Decimal units to Amount on-grid (floor)."""
    if is_xrp(cur):
        drops = int((v / XRP_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return XRPAmount(value=drops)
    units = int((v / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
    return IOUAmount.from_components(units, -15)


def build_tiers_from_offers(
    offers: List[Dict[str, Any]],
    in_cur: str,
    out_cur: str,
) -> List[Tuple[Quality, Any, Any, Any]]:
    """
    Return canonical tiers as tuples:
      (quality, out_max_amt, in_at_out_max_amt)
    """
    tiers: List[Tuple[Quality, Any, Any, Any]] = []

    for o in offers:
        # Raw amounts
        gets_cur, _, gets_val = parse_amount(o.get("TakerGets"))
        pays_cur, _, pays_val = parse_amount(o.get("TakerPays"))

        # Prefer funded amounts when present (XRPL may include unfunded offers)
        if "taker_gets_funded" in o:
            try:
                fg_cur, _, fg_val = parse_amount(o.get("taker_gets_funded"))
                if fg_cur == gets_cur and fg_val is not None:
                    gets_val = fg_val
            except Exception:
                pass
        if "taker_pays_funded" in o:
            try:
                fp_cur, _, fp_val = parse_amount(o.get("taker_pays_funded"))
                if fp_cur == pays_cur and fp_val is not None:
                    pays_val = fp_val
            except Exception:
                pass

        # If owner_funds is present and zero, skip early
        try:
            if "owner_funds" in o and D(o.get("owner_funds")) <= 0:
                continue
        except Exception:
            pass

        xrpl_q_raw = o.get("quality")

        # Align offer into IN -> OUT (OUT received, IN paid)
        if gets_cur == out_cur and pays_cur == in_cur:
            out_max_dec = gets_val
            in_need_dec = pays_val
        elif pays_cur == out_cur and gets_cur == in_cur:
            out_max_dec = pays_val
            in_need_dec = gets_val
        else:
            continue

        if out_max_dec <= 0 or in_need_dec <= 0:
            continue

        out_q = XRP_QUANTUM if is_xrp(out_cur) else IOU_QUANTUM
        in_q  = XRP_QUANTUM if is_xrp(in_cur)  else IOU_QUANTUM
        out_max_dec = quantize_down(out_max_dec, out_q)
        in_need_dec = quantize_down(in_need_dec, in_q)

        out_amt = amt_floor(out_cur, out_max_dec)
        in_amt  = amt_floor(in_cur,  in_need_dec)

        if amount_to_decimal(out_amt) <= 0 or amount_to_decimal(in_amt) <= 0:
            continue

        q = Quality.from_amounts(out_amt, in_amt)
        # Keep xrpl_q_raw for debugging (as a 4th element, ignored by router ladder encodings)
        tiers.append((q, out_amt, in_amt, xrpl_q_raw))

    if not tiers:
        raise RuntimeError("No direction-aligned offers found for this IN/OUT.")

    tiers.sort(key=lambda t: t[0], reverse=True)  # best -> worst
    return tiers


def sum_out_capacity(tiers: List[Tuple[Quality, Any, Any, Any]]) -> Decimal:
    return sum((amount_to_decimal(t[1]) for t in tiers), start=Decimal("0"))


def ladder_encodings(tiers: List[Tuple[Quality, Any, Any, Any]]) -> List[List[Any]]:
    """
    Produce several shapes to satisfy RouterQuoteView/BookStep expectations.
    We will try them and pick the best valid quote (minimal total_in).
    """
    dict_style = [{"quality": q, "out_max": out_amt, "in_at_out_max": in_amt} for (q, out_amt, in_amt, _) in tiers]
    tup_style1 = [(q, out_amt, in_amt) for (q, out_amt, in_amt, _) in tiers]
    tup_style2 = [(q, in_amt, out_amt) for (q, out_amt, in_amt, _) in tiers]  # some impls use (q, in, out)
    return [dict_style, tup_style1, tup_style2]


# Helper to convert tiers -> Segments for RouterQuoteView/BookStep
def tiers_to_segments(
    tiers: List[Tuple[Quality, Any, Any, Any]],
    src: str = "CLOB",
) -> List[Segment]:
    """Convert tier tuples into Segment objects required by RouterQuoteView/BookStep."""
    segs: List[Segment] = []
    for i, (q, out_amt, in_amt, xrpl_q_raw) in enumerate(tiers):
        # Use a stable source_id to help debugging/leg attribution
        source_id = None
        try:
            source_id = f"tier:{i}"
        except Exception:
            source_id = None
        segs.append(
            Segment(
                src=src,
                quality=q,
                out_max=out_amt,
                in_at_out_max=in_amt,
                raw_quality=q,
                source_id=source_id,
            )
        )
    return segs



def run_best_quote(
    amm: AMM,
    tiers: List[Tuple[Quality, Any, Any, Any]],
    target_out_amt: Any,
) -> Tuple[Dict[str, Any], List[Any]]:
    """
    Quote exact-out using RouterQuoteView with proper Segment objects for CLOB.
    """
    segs = tiers_to_segments(tiers, src="CLOB")
    view = RouterQuoteView(lambda: segs, amm=amm)
    q = view.preview_out(target_out_amt)
    # Return segs as ladder_used for debugging parity with previous code.
    return q, segs



def avg_price_xrp_per_rusd(in_cur: str, out_cur: str, total_in: Decimal, total_out: Decimal) -> Optional[Decimal]:
    """
    Return XRP per rUSD if direction is XRP->rUSD; otherwise None (kept simple).
    """
    if total_out == 0:
        return None
    if in_cur == XRP and out_cur == RUSD_HEX:
        return total_in / total_out
    return None


# Summarise model slices by source and totals

def summarise_model_slices(quote: Dict[str, Any]) -> Tuple[int, int, Decimal, Decimal, Decimal, Decimal, List[str]]:
    """Return (n_clob, n_amm, clob_in, clob_out, amm_in, amm_out, path_list) from quote slices."""
    n_clob = 0
    n_amm = 0
    clob_in = Decimal("0")
    clob_out = Decimal("0")
    amm_in = Decimal("0")
    amm_out = Decimal("0")
    path: List[str] = []

    for s in quote.get("slices", []) or []:
        if not isinstance(s, dict):
            continue
        src = str(s.get("src") or s.get("kind") or "?")
        path.append(src)
        in_take = amount_to_decimal(s.get("in_take")) if s.get("in_take") is not None else Decimal("0")
        out_take = amount_to_decimal(s.get("out_take")) if s.get("out_take") is not None else Decimal("0")

        if "CLOB" in src or "BOOK" in src or "ORDER" in src:
            n_clob += 1
            clob_in += in_take
            clob_out += out_take
        elif "AMM" in src:
            n_amm += 1
            amm_in += in_take
            amm_out += out_take

    return n_clob, n_amm, clob_in, clob_out, amm_in, amm_out, path


# Summarise real CLOB legs (for concise per-tx print)
def summarise_real_clob_legs(
    legs: List[Any],
    in_cur: str,
    out_cur: str,
    max_items: int = 4,
) -> List[Tuple[Decimal, Decimal, Decimal]]:
    """Return list of (in_amt, out_amt, px_in_per_out) per real CLOB leg aligned to IN->OUT."""
    out: List[Tuple[Decimal, Decimal, Decimal]] = []
    for r in legs[:max_items]:
        base_cur = r["base_currency"]
        counter_cur = r["counter_currency"]
        base_amt = D(r["base_amount"])
        counter_amt = D(r["counter_amount"])
        in_amt: Optional[Decimal] = None
        out_amt: Optional[Decimal] = None
        if base_cur == out_cur and counter_cur == in_cur:
            in_amt = counter_amt
            out_amt = base_amt
        elif counter_cur == out_cur and base_cur == in_cur:
            in_amt = base_amt
            out_amt = counter_amt
        if in_amt is None or out_amt is None or out_amt == 0:
            continue
        price = in_amt / out_amt
        out.append((in_amt, out_amt, price))
    return out


def summarise_model_clob_slices(
    quote: Dict[str, Any],
    max_items: int = 4,
) -> List[Tuple[Decimal, Decimal, Decimal]]:
    """Return list of (in_take, out_take, px_in_per_out) for CLOB slices, limited to max_items."""
    out: List[Tuple[Decimal, Decimal, Decimal]] = []
    for s in quote.get("slices", []) or []:
        if not isinstance(s, dict):
            continue
        src = str(s.get("src") or s.get("kind") or "")
        if ("CLOB" not in src) and ("BOOK" not in src) and ("ORDER" not in src):
            continue
        in_take = amount_to_decimal(s.get("in_take")) if s.get("in_take") is not None else Decimal("0")
        out_take = amount_to_decimal(s.get("out_take")) if s.get("out_take") is not None else Decimal("0")
        price = (in_take / out_take) if out_take != 0 else Decimal("0")
        out.append((in_take, out_take, price))
        if len(out) >= max_items:
            break
    return out


def is_non_decreasing(seq: List[Decimal], tol: Decimal = Decimal("0")) -> bool:
    """Return True if seq[i] <= seq[i+1] + tol for all i (allowing tiny numerical tolerance)."""
    for i in range(len(seq) - 1):
        if seq[i] > seq[i + 1] + tol:
            return False
    return True


def main() -> None:
    for p in [AMM_SWAPS_PARQUET, AMM_FEES_PARQUET, CLOB_TX_PARQUET, BOOK_GETS_RUSD_JSON, BOOK_GETS_XRP_JSON]:
        assert_exists(p)

    spark = SparkSession.builder.appName("compare_rusd_xrp_100894647_vs_model").getOrCreate()

    amm_swaps = spark.read.parquet(AMM_SWAPS_PARQUET).filter(F.col("ledger_index") == F.lit(LEDGER_INDEX))
    amm_fees  = spark.read.parquet(AMM_FEES_PARQUET).filter(F.col("ledger_index") == F.lit(LEDGER_INDEX))
    clob_tx   = spark.read.parquet(CLOB_TX_PARQUET)

    swaps = (
        amm_swaps.filter(
            (
                (F.col("amm_asset_currency") == F.lit(RUSD_HEX)) & (F.col("amm_asset2_currency") == F.lit(XRP))
            ) | (
                (F.col("amm_asset_currency") == F.lit(XRP)) & (F.col("amm_asset2_currency") == F.lit(RUSD_HEX))
            )
        )
        .orderBy(F.col("transaction_index").asc())
        .collect()
    )

    if not swaps:
        raise RuntimeError(f"No rUSD-XRP swaps found in ledger {LEDGER_INDEX} under {AMM_SWAPS_PARQUET}")

    book_gets_rusd = load_book_offers(BOOK_GETS_RUSD_JSON)
    book_gets_xrp  = load_book_offers(BOOK_GETS_XRP_JSON)

    print(f"\n=== Ledger {LEDGER_INDEX} rUSD-XRP: model vs real ===")
    print(f"pre CLOB offers: gets_rUSD={len(book_gets_rusd)} gets_XRP={len(book_gets_xrp)}")
    print(f"tx count (amm_swaps rows): {len(swaps)}")

    # Ledger-level aggregates (across all swaps)
    agg_real_in = Decimal("0")
    agg_real_out = Decimal("0")
    agg_model_in = Decimal("0")
    agg_model_out = Decimal("0")

    agg_real_amm_in = Decimal("0")
    agg_real_amm_out = Decimal("0")
    agg_real_clob_in = Decimal("0")
    agg_real_clob_out = Decimal("0")

    agg_model_amm_in = Decimal("0")
    agg_model_amm_out = Decimal("0")
    agg_model_clob_in = Decimal("0")
    agg_model_clob_out = Decimal("0")

    agg_real_clob_legs = 0
    agg_model_clob_slices = 0
    agg_model_amm_slices = 0

    tx_count = 0
    tx_real_clob_count = 0
    tx_model_clob_count = 0

    for tx in swaps:
        tx_hash = tx["transaction_hash"]
        tx_index = int(tx["transaction_index"]) if tx["transaction_index"] is not None else None

        in_cur = tx["asset_in_currency"]
        out_cur = tx["asset_out_currency"]

        # AMM pre reserves
        side1_cur = tx["amm_asset_currency"]
        side2_cur = tx["amm_asset2_currency"]
        side1_pre = D(tx["amm_asset_balance_before"])
        side2_pre = D(tx["amm_asset2_balance_before"])

        if in_cur == side1_cur and out_cur == side2_cur:
            x_reserve, y_reserve = side1_pre, side2_pre
        elif in_cur == side2_cur and out_cur == side1_cur:
            x_reserve, y_reserve = side2_pre, side1_pre
        else:
            raise RuntimeError("Swap currencies do not match AMM pool sides.")

        # trading_fee
        fee_rows = (
            amm_fees.filter(
                (F.col("transaction_hash") == F.lit(tx_hash)) &
                (F.col("amm_account") == F.lit(tx["amm_account"]))
            )
            .select("trading_fee")
            .limit(1)
            .collect()
        )
        trading_fee = D(fee_rows[0]["trading_fee"]) if fee_rows else Decimal("0")

        # Real legs
        amm_in_real  = D(tx["asset_in_value"])
        amm_out_real = D(tx["asset_out_value"])

        legs = (
            clob_tx.filter(F.col("tx_hash") == F.lit(tx_hash))
                  .select("base_currency", "counter_currency", "base_amount", "counter_amount")
                  .collect()
        )
        clob_in_sum = Decimal("0")
        clob_out_sum = Decimal("0")
        for r in legs:
            base_cur = r["base_currency"]
            counter_cur = r["counter_currency"]
            base_amt = D(r["base_amount"])
            counter_amt = D(r["counter_amount"])

            if base_cur == out_cur and counter_cur == in_cur:
                clob_out_sum += base_amt
                clob_in_sum  += counter_amt
            elif counter_cur == out_cur and base_cur == in_cur:
                clob_out_sum += counter_amt
                clob_in_sum  += base_amt

        real_total_out = amm_out_real + clob_out_sum
        real_total_in  = amm_in_real  + clob_in_sum

        # Build tiers from both saved books; pick the one that yields tiers for this direction.
        tiers: Optional[List[Tuple[Quality, Any, Any, Any]]] = None
        last_err: Optional[Exception] = None
        for book in (book_gets_rusd, book_gets_xrp):
            try:
                tiers = build_tiers_from_offers(book, in_cur, out_cur)
                break
            except Exception as e:
                last_err = e
                continue
        if tiers is None:
            raise RuntimeError(f"Failed to build CLOB tiers for {pretty_cur(in_cur)}->{pretty_cur(out_cur)}: {last_err}")

        cap_out = sum_out_capacity(tiers)

        if DEBUG_BOOK and len(legs) > 0:
            # Show raw offers from both saved books to diagnose direction/quality parsing
            _dump_sample_offers(book_gets_rusd, in_cur, out_cur, n=3)
            _dump_sample_offers(book_gets_xrp, in_cur, out_cur, n=3)
            # Show how tiers were constructed
            _dump_tier_mapping(tiers, in_cur, out_cur, n=10)
            _tier_debug(tiers, in_cur, out_cur, n=10)

        # Exact-out target: match real total out
        target_out = real_total_out
        target_out_amt = amt_floor(out_cur, target_out)

        # Hard check: if real used CLOB but captured book can't possibly supply that out, raise.
        if len(legs) > 0 and target_out > cap_out:
            raise RuntimeError(
                f"CLOB depth exhausted (pre-check): target_out={target_out} > cap_out={cap_out} tx={tx_hash}"
            )

        amm = AMM(x_reserve, y_reserve, trading_fee, x_is_xrp=is_xrp(in_cur), y_is_xrp=is_xrp(out_cur))

        quote, ladder_used = run_best_quote(amm=amm, tiers=tiers, target_out_amt=target_out_amt)

        if DEBUG_QUOTE and len(legs) > 0:
            print("\n[quote_debug] full quote object:")
            _pp(quote)
            print("\n[quote_debug] slices type:", type(quote.get("slices")), "len=", len(quote.get("slices", [])) if hasattr(quote.get("slices", []), "__len__") else None)
            print("[quote_debug] ladder_used_type:", type(ladder_used))
            try:
                print("[quote_debug] ladder_used_first3:")
                # Segments may not be directly serialisable; print key fields.
                for s in ladder_used[:3]:
                    try:
                        print({
                            "src": s.src,
                            "out_max": amount_to_decimal(s.out_max),
                            "in_at_out_max": amount_to_decimal(s.in_at_out_max),
                            "quality_m_e": getattr(s.quality, "mantissa_exponent", lambda: (None, None))(),
                            "source_id": getattr(s, "source_id", None),
                        })
                    except Exception:
                        print(str(s))
            except Exception:
                print("[quote_debug] ladder_used_first3: <unavailable>")

        summary = quote.get("summary", {})
        if summary.get("is_partial"):
            raise RuntimeError(f"Model returned PARTIAL quote (treat as depth exhausted): tx={tx_hash}")

        pred_in  = amount_to_decimal(summary["total_in"])
        pred_out = amount_to_decimal(summary["total_out"])

        # Concise path + totals comparison
        n_clob_m, n_amm_m, clob_in_m, clob_out_m, amm_in_m, amm_out_m, path_m = summarise_model_slices(quote)
        # Update ledger-level aggregates
        tx_count += 1
        agg_real_in += real_total_in
        agg_real_out += real_total_out
        agg_model_in += pred_in
        agg_model_out += pred_out

        agg_real_amm_in += amm_in_real
        agg_real_amm_out += amm_out_real
        agg_real_clob_in += clob_in_sum
        agg_real_clob_out += clob_out_sum

        agg_model_amm_in += amm_in_m
        agg_model_amm_out += amm_out_m
        agg_model_clob_in += clob_in_m
        agg_model_clob_out += clob_out_m

        agg_real_clob_legs += len(legs)
        agg_model_clob_slices += n_clob_m
        agg_model_amm_slices += n_amm_m

        if len(legs) > 0:
            tx_real_clob_count += 1
        if n_clob_m > 0:
            tx_model_clob_count += 1

        # Try to detect sources for debug (unchanged)
        used_sources: List[str] = []
        for s in quote.get("slices", []) or []:
            if isinstance(s, dict):
                if s.get("src"):
                    used_sources.append(str(s.get("src")))
                elif s.get("kind"):
                    used_sources.append(str(s.get("kind")))
            else:
                try:
                    used_sources.append(str(getattr(s, "src")))  # type: ignore[attr-defined]
                except Exception:
                    try:
                        used_sources.append(str(getattr(s, "kind")))  # type: ignore[attr-defined]
                    except Exception:
                        used_sources.append(str(s))
        if DEBUG_QUOTE and len(legs) > 0:
            print("[quote_debug] used_sources_raw:", used_sources)

        used_amm = any("AMM" in x for x in used_sources)
        used_clob = any(("CLOB" in x) or ("BOOK" in x) or ("ORDER" in x) for x in used_sources)

        # Average prices (XRP per rUSD) for XRP->rUSD
        real_avg = avg_price_xrp_per_rusd(in_cur, out_cur, real_total_in, real_total_out)
        pred_avg = avg_price_xrp_per_rusd(in_cur, out_cur, pred_in, pred_out)

        print("\n---")
        print(f"tx {tx_index} | {tx_hash}")
        print(f"dir: {pretty_cur(in_cur)} -> {pretty_cur(out_cur)}")

        # REAL
        real_path = "AMM" if len(legs) == 0 else f"AMM + CLOB({len(legs)} legs)"
        print(f"REAL  path: {real_path}")
        print(f"REAL  AMM : in={fmt(amm_in_real)} {pretty_cur(in_cur)}  out={fmt(amm_out_real)} {pretty_cur(out_cur)}")
        if len(legs) > 0:
            print(f"REAL  CLOB: in={fmt(clob_in_sum)} {pretty_cur(in_cur)}  out={fmt(clob_out_sum)} {pretty_cur(out_cur)}")
        # Print concise summary of real CLOB legs if present
        if len(legs) > 0:
            real_leg_rows = summarise_real_clob_legs(legs, in_cur, out_cur, max_items=8)
            if real_leg_rows:
                print("REAL  legs:")
                for i, (inp, outp, price) in enumerate(real_leg_rows):
                    print(f"  - #{i+1} in={fmt(inp)} {pretty_cur(in_cur)} out={fmt(outp)} {pretty_cur(out_cur)} price(in/out)={fmt(price)}")
                pxs = [price for (_, _, price) in real_leg_rows]
                verdict = "best→worse" if is_non_decreasing(pxs, tol=Decimal('1e-18')) else "NOT monotone"
                print(f"REAL  legs order: {verdict}")
        print(f"REAL  TOT : in={fmt(real_total_in)} {pretty_cur(in_cur)}  out={fmt(real_total_out)} {pretty_cur(out_cur)}")
        if real_avg is not None:
            print(f"REAL  avg_px: {fmt(real_avg)} XRP per rUSD")

        # MODEL
        model_path = " + ".join(path_m) if path_m else ("CLOB" if used_clob else "AMM")
        print(f"MODEL path: {model_path}")
        if n_amm_m > 0:
            print(f"MODEL AMM : in={fmt(amm_in_m)} {pretty_cur(in_cur)}  out={fmt(amm_out_m)} {pretty_cur(out_cur)} ({n_amm_m} slices)")
        if n_clob_m > 0:
            print(f"MODEL CLOB: in={fmt(clob_in_m)} {pretty_cur(in_cur)}  out={fmt(clob_out_m)} {pretty_cur(out_cur)} ({n_clob_m} slices)")
        if n_clob_m > 0:
            model_clob_rows = summarise_model_clob_slices(quote, max_items=8)
            if model_clob_rows:
                print("MODEL CLOB slices:")
                for i, (inp, outp, price) in enumerate(model_clob_rows):
                    print(f"  - #{i+1} in={fmt(inp)} {pretty_cur(in_cur)} out={fmt(outp)} {pretty_cur(out_cur)} price(in/out)={fmt(price)}")
                pxs = [price for (_, _, price) in model_clob_rows]
                verdict = "best→worse" if is_non_decreasing(pxs, tol=Decimal('1e-18')) else "NOT monotone"
                print(f"MODEL CLOB order: {verdict}")
        print(f"MODEL TOT : in={fmt(pred_in)} {pretty_cur(in_cur)}  out={fmt(pred_out)} {pretty_cur(out_cur)}")
        if pred_avg is not None:
            print(f"MODEL avg_px: {fmt(pred_avg)} XRP per rUSD")

        # DIFF
        print(f"DIFF  in : {fmt(pred_in - real_total_in)} {pretty_cur(in_cur)}")
        print(f"DIFF  out: {fmt(pred_out - real_total_out)} {pretty_cur(out_cur)}")
        if (real_avg is not None) and (pred_avg is not None):
            print(f"DIFF  avg_px: {fmt(pred_avg - real_avg)} XRP per rUSD")

        # If real used CLOB but model still doesn't, fail loudly (this is what you care about)
        if len(legs) > 0 and (not used_clob):
            raise RuntimeError(
                f"REAL used CLOB (legs={len(legs)}) but MODEL path has no CLOB slices. tx={tx_hash}"
            )

    print("\n=== Ledger-level summary (rUSD-XRP executions) ===")
    print(f"txs analysed: {tx_count} | REAL paths: AMM-only={tx_count - tx_real_clob_count}, AMM+CLOB={tx_real_clob_count} | MODEL paths: AMM-only={tx_count - tx_model_clob_count}, AMM+CLOB={tx_model_clob_count}")
    print(f"REAL  AMM : in={fmt(agg_real_amm_in)} XRP out={fmt(agg_real_amm_out)} rUSD")
    print(f"REAL  BOOK: in={fmt(agg_real_clob_in)} XRP out={fmt(agg_real_clob_out)} rUSD (legs={agg_real_clob_legs})")
    print(f"REAL  TOT : in={fmt(agg_real_in)} XRP out={fmt(agg_real_out)} rUSD")

    print(f"MODEL AMM : in={fmt(agg_model_amm_in)} XRP out={fmt(agg_model_amm_out)} rUSD (slices={agg_model_amm_slices})")
    print(f"MODEL BOOK: in={fmt(agg_model_clob_in)} XRP out={fmt(agg_model_clob_out)} rUSD (slices={agg_model_clob_slices})")
    print(f"MODEL TOT : in={fmt(agg_model_in)} XRP out={fmt(agg_model_out)} rUSD")

    print(f"DIFF  in : {fmt(agg_model_in - agg_real_in)} XRP")
    print(f"DIFF  out: {fmt(agg_model_out - agg_real_out)} rUSD")

    # Average price across ledger (XRP per rUSD)
    if agg_real_out != 0:
        real_avg_ledger = agg_real_in / agg_real_out
        model_avg_ledger = agg_model_in / agg_model_out if agg_model_out != 0 else None
        print(f"REAL  avg_px: {fmt(real_avg_ledger)} XRP per rUSD")
        if model_avg_ledger is not None:
            print(f"MODEL avg_px: {fmt(model_avg_ledger)} XRP per rUSD")
            print(f"DIFF  avg_px: {fmt(model_avg_ledger - real_avg_ledger)} XRP per rUSD")

    spark.stop()


if __name__ == "__main__":
    main()
