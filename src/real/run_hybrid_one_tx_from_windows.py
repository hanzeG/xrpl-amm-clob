#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run ONE real hybrid transaction using REAL CLOB executed legs as ladder tiers.

Pipeline:
1) Load AMM swaps window parquet.
2) Load CLOB offers_fact_tx window parquet.
3) (Optional) Load AMM fees window parquet for trading_fee.
4) Pick TX_HASH (default to your chosen hybrid example).
5) Read AMM leg row for this tx.
6) Read all CLOB executed legs for this tx.
7) Build REAL ladder tiers directly from CLOB legs (no make_ladder, no decay).
8) Build AMM from pre-trade reserves + trading_fee (ex-ante).
9) Run RouterQuoteView.preview_out(target_out) in HYBRID mode.
10) Print pre state, model prediction (prices only), real-world approx, errors.

Notes:
- We do NOT attempt to reconstruct the full ledger LOB. Ladder == realised path tiers.
- Full-tx "real" totals are approximated as AMM-leg + sum(CLOB legs).
"""

import argparse
import json
from decimal import Decimal, ROUND_FLOOR, getcontext
from typing import List, Dict, Any, Optional

from pyspark.sql import SparkSession, functions as F

from xrpl_router.core import Quality, IOUAmount, XRPAmount
from xrpl_router.core.constants import XRP_QUANTUM, IOU_QUANTUM
from xrpl_router.core.fmt import (
    amount_to_decimal,
    quality_rate_to_decimal,
    quality_price_to_decimal,
    fmt_dec,
    quantize_down,
)
from xrpl_router.amm import AMM
from xrpl_router.book_step import RouterQuoteView

getcontext().prec = 28

# ---------------- Defaults ----------------
AMM_PATH  = "data/output/amm_rusd_xrp_20251201"
CLOB_PATH = "data/output/clob_rusd_xrp_20251201"
FEES_PATH = "data/output/amm_fees_20251201"   # if missing, fallback fee used

DEFAULT_TX_HASH = "EE0AE95CEB8C2FBE6B10F54670D2340BF6883F129E5AF26661402FA1578B0BB2"

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--amm", default=AMM_PATH, help="AMM swaps parquet path")
    p.add_argument("--clob", default=CLOB_PATH, help="CLOB offers parquet path")
    p.add_argument("--fees", default=FEES_PATH, help="AMM fees parquet path")
    p.add_argument("--tx_hash", default=DEFAULT_TX_HASH, help="Hybrid tx_hash to run")
    p.add_argument("--fee_fallback", default="0.003", help="Fallback trading fee if not found")
    return p.parse_args()


def to_dec(x) -> Decimal:
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


def pretty_cur(c: str) -> str:
    return "rUSD" if c == RUSD_HEX else c


def _is_xrp(currency: str) -> bool:
    return currency == XRP


def _amt_floor(d: Decimal, is_xrp: bool):
    """Decimal -> Amount floored to XRPL quantum."""
    if is_xrp:
        drops = int((d / XRP_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return XRPAmount(value=drops)
    units = int((d / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
    return IOUAmount.from_components(units, -15)


def _fmt_dec(d: Decimal, is_xrp: bool) -> str:
    q = XRP_QUANTUM if is_xrp else IOU_QUANTUM
    places = 6 if is_xrp else 15
    return f"{quantize_down(d, q):.{places}f}"


def build_real_ladder_from_clob_legs(
    clob_legs: List[Any],
    router_in_cur: str,
    router_out_cur: str,
    in_is_xrp: bool,
    out_is_xrp: bool,
) -> List[Dict[str, Any]]:
    """
    Build ladder tiers directly from REAL executed CLOB legs.

    Each tier corresponds to one executed leg aligned with router direction.
    """
    tiers: List[Dict[str, Any]] = []

    for r in clob_legs:
        base_cur = r["base_currency"]
        counter_cur = r["counter_currency"]
        base_amt = to_dec(r["base_amount"])
        counter_amt = to_dec(r["counter_amount"])

        # Align to router direction (IN -> OUT)
        if base_cur == router_out_cur and counter_cur == router_in_cur:
            out_dec, in_dec = base_amt, counter_amt
        elif counter_cur == router_out_cur and base_cur == router_in_cur:
            out_dec, in_dec = counter_amt, base_amt
        else:
            continue  # ignore non-aligned legs

        if out_dec <= 0 or in_dec <= 0:
            continue

        out_amt = _amt_floor(out_dec, out_is_xrp)
        in_amt = _amt_floor(in_dec, in_is_xrp)
        q = Quality.from_amounts(out_amt, in_amt)

        tiers.append({
            "quality": q,
            "out_max": out_amt,
            "in_at_out_max": in_amt,
            "src": "CLOB_REAL",
            "meta": {
                "close_time": str(r["close_time"]),
                "price_raw": float(r["price"]) if r["price"] is not None else None,
            }
        })

    # Sort best -> worst by quality rate (matches tier ordering convention)
    tiers.sort(key=lambda t: float(quality_rate_to_decimal(t["quality"])))
    return tiers


def main():
    args = parse_args()
    TX_HASH = args.tx_hash

    spark = SparkSession.builder.appName("run_hybrid_one_tx_from_windows").getOrCreate()

    amm = spark.read.parquet(args.amm)
    clob = spark.read.parquet(args.clob)

    try:
        fees = spark.read.parquet(args.fees)
    except Exception:
        fees = None

    # -----------------------------
    # 1) Load AMM leg (required)
    # -----------------------------
    amm_row = (
        amm.filter(F.col("transaction_hash") == TX_HASH)
           .limit(1)
           .collect()
    )
    if not amm_row:
        raise RuntimeError(f"TX not found in AMM swaps window: {TX_HASH}")
    amm_tx = amm_row[0]

    amm_in_cur  = amm_tx["asset_in_currency"]
    amm_out_cur = amm_tx["asset_out_currency"]
    amm_in_val  = to_dec(amm_tx["asset_in_value"])
    amm_out_val = to_dec(amm_tx["asset_out_value"])

    # Pool sides + pre reserves
    side1_cur = amm_tx["amm_asset_currency"]
    side2_cur = amm_tx["amm_asset2_currency"]
    side1_pre = to_dec(amm_tx["amm_asset_balance_before"])
    side2_pre = to_dec(amm_tx["amm_asset2_balance_before"])

    # Map reserves to routing convention x=IN, y=OUT
    if amm_in_cur == side1_cur and amm_out_cur == side2_cur:
        x_reserve, y_reserve = side1_pre, side2_pre
        x_cur, y_cur = side1_cur, side2_cur
    elif amm_in_cur == side2_cur and amm_out_cur == side1_cur:
        x_reserve, y_reserve = side2_pre, side1_pre
        x_cur, y_cur = side2_cur, side1_cur
    else:
        raise RuntimeError("AMM leg currencies do not match pool sides.")

    in_is_xrp = _is_xrp(amm_in_cur)
    out_is_xrp = _is_xrp(amm_out_cur)

    # -----------------------------
    # 2) Load REAL CLOB executed legs for same tx
    # -----------------------------
    clob_legs = (
        clob.filter(F.col("tx_hash") == TX_HASH)
            .orderBy(F.col("close_time").asc())
            .select("close_time", "price", "base_currency", "counter_currency",
                    "base_amount", "counter_amount")
            .collect()
    )
    if not clob_legs:
        raise RuntimeError(f"No CLOB legs for this tx in CLOB window: {TX_HASH}")

    # -----------------------------
    # 3) Build REAL ladder
    # -----------------------------
    ladder = build_real_ladder_from_clob_legs(
        clob_legs,
        router_in_cur=amm_in_cur,
        router_out_cur=amm_out_cur,
        in_is_xrp=in_is_xrp,
        out_is_xrp=out_is_xrp,
    )
    if not ladder:
        raise RuntimeError("No direction-aligned CLOB legs to build ladder.")

    # -----------------------------
    # 4) Trading fee (ex-ante) from fees window
    # -----------------------------
    fee_fallback = Decimal(str(args.fee_fallback))
    fee_dec = fee_fallback
    fee_src = "fallback_const"

    if fees is not None:
        fee_row = (
            fees.filter(
                (F.col("transaction_hash") == TX_HASH) &
                (F.col("amm_account") == amm_tx["amm_account"])
            )
            .select("trading_fee")
            .limit(1)
            .collect()
        )
        if fee_row and fee_row[0]["trading_fee"] is not None:
            fee_dec = to_dec(fee_row[0]["trading_fee"])
            fee_src = "trading_fee (ex-ante)"

    # -----------------------------
    # 5) Build AMM + pre spot price
    # -----------------------------
    amm_obj = AMM(
        x_reserve, y_reserve, fee_dec,
        x_is_xrp=in_is_xrp,
        y_is_xrp=out_is_xrp
    )
    view_pre = RouterQuoteView(lambda: ladder, amm=amm_obj)
    snap_pre = view_pre.snapshot()["amm"]
    pre_spot_price = quality_price_to_decimal(snap_pre["spq_quality"])

    # -----------------------------
    # 6) Approx real full-tx totals (AMM leg + sum CLOB legs in router direction)
    # -----------------------------
    clob_out_sum = Decimal("0")
    clob_in_sum  = Decimal("0")

    for r in clob_legs:
        base_cur = r["base_currency"]
        counter_cur = r["counter_currency"]
        base_amt = to_dec(r["base_amount"])
        counter_amt = to_dec(r["counter_amount"])

        if base_cur == amm_out_cur and counter_cur == amm_in_cur:
            clob_out_sum += base_amt
            clob_in_sum  += counter_amt
        elif counter_cur == amm_out_cur and base_cur == amm_in_cur:
            clob_out_sum += counter_amt
            clob_in_sum  += base_amt
        else:
            continue

    total_out_real_approx = amm_out_val + clob_out_sum
    total_in_real_approx  = amm_in_val  + clob_in_sum
    real_avg_price = (total_in_real_approx / total_out_real_approx) if total_out_real_approx != 0 else None

    # -----------------------------
    # 7) Run HYBRID model on target_out = approx real total_out
    # -----------------------------
    target_out = total_out_real_approx
    quote = view_pre.preview_out(_amt_floor(target_out, out_is_xrp))

    summary = quote["summary"]
    pred_out = amount_to_decimal(summary["total_out"])
    pred_in  = amount_to_decimal(summary["total_in"])
    pred_avg_price = (pred_in / pred_out) if pred_out != 0 else None

    # post spot price from last AMM slice
    post_spq = None
    for t in quote["slices"]:
        if t.get("src") == "AMM" and t.get("post_spq") is not None:
            post_spq = t["post_spq"]
    post_spot_price = quality_price_to_decimal(post_spq) if post_spq is not None else None

    # -----------------------------
    # 8) PRINT (price-only, similar style)
    # -----------------------------
    in_p = pretty_cur(amm_in_cur)
    out_p = pretty_cur(amm_out_cur)

    print("\n=== Pre state (hybrid) ===")
    print(f"tx_hash        : {TX_HASH}")
    print(f"router_in/out  : {in_p} -> {out_p}")
    print(f"pre_reserves   : x={_fmt_dec(x_reserve, in_is_xrp)} {in_p}, "
          f"y={_fmt_dec(y_reserve, out_is_xrp)} {out_p}, fee={fmt_dec(fee_dec, places=6)} (src={fee_src})")
    print(f"pre_spot_price : {pre_spot_price} ({in_p} per {out_p})")
    print(f"target_out     : {_fmt_dec(target_out, out_is_xrp)} {out_p}")

    print("\n=== Real CLOB ladder (executed legs) ===")
    print(f"tiers = {len(ladder)}")
    for i, t in enumerate(ladder, 1):
        out_d = amount_to_decimal(t["out_max"])
        in_d  = amount_to_decimal(t["in_at_out_max"])
        leg_price = (in_d / out_d) if out_d != 0 else None
        print(f"T{i:02d}: out={out_d} {out_p}, in={in_d} {in_p}, leg_price={leg_price} ({in_p} per {out_p})")

    print("\n=== Model prediction (HYBRID) ===")
    print(f"pred_total_out      : {_fmt_dec(pred_out, out_is_xrp)} {out_p}")
    print(f"pred_total_in       : {_fmt_dec(pred_in, in_is_xrp)} {in_p}")
    if pred_avg_price is not None:
        print(f"pred_avg_price(in/out): {pred_avg_price} ({in_p} per {out_p})")
    if post_spot_price is not None:
        print(f"post_spot_price     : {post_spot_price} ({in_p} per {out_p})")

    print("\n=== Real-world approx (full-tx) ===")
    print(f"real_total_out_approx : {_fmt_dec(total_out_real_approx, out_is_xrp)} {out_p}")
    print(f"real_total_in_approx  : {_fmt_dec(total_in_real_approx, in_is_xrp)} {in_p}")
    if real_avg_price is not None:
        print(f"real_avg_price(in/out): {real_avg_price} ({in_p} per {out_p})")

    print("\n=== Errors (approx full-tx) ===")
    print(f"out_error (pred-real): {pred_out - total_out_real_approx}")
    print(f"in_error  (pred-real): {pred_in  - total_in_real_approx}")
    if pred_avg_price is not None and real_avg_price is not None:
        print(f"avg_price_error: {pred_avg_price - real_avg_price}")

    spark.stop()


if __name__ == "__main__":
    main()