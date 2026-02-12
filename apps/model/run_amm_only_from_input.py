#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Cost fields available in our extracted tables but not yet modelled/validated
# -----------------------------------------------------------------------------
# 1) AMM discounted_fee (fact_amm_fees):
#    - Effective fee after discount; model predicts with trading_fee only
#      unless ex-post override is used.
#
# 2) AMM trading_fee_value (fact_amm_fees):
#    - Real fee amount charged per swap; not yet matched against model fee_paid.
#
# 3) Network transaction_fee (fact_amm_swaps):
#    - On-ledger base fee in drops; only added as an ex-ante proxy in evaluation,
#      not modelled/validated inside routing mechanics.
# -----------------------------------------------------------------------------

"""
Run AMM-only routing model on a real-world snapshot input produced by
build_amm_only_input_from_window.py, using production xrpl_router APIs.

Printing policy (AMM-only):
1) Pre AMM reserves + pre spot price + target_out.
2) Model predicted metrics + model post state (price-related only).
3) Real-world metrics + real post state (if available).
4) Errors.

Notes:
- Quality metrics are intentionally omitted.
- AMM slices are omitted for AMM-only runs.
- rUSD hex currency code is displayed as 'rUSD'.
"""

import argparse
import json
from decimal import Decimal, ROUND_FLOOR, getcontext
from typing import Optional

from xrpl_router.core.fmt import (
    amount_to_decimal,
    fmt_dec,
    quantize_down,
)
from xrpl_router.core import IOUAmount, XRPAmount
from xrpl_router.core.constants import XRP_QUANTUM, IOU_QUANTUM
from xrpl_router.amm import AMM
from xrpl_router.book_step import RouterQuoteView

getcontext().prec = 28

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to model input JSON")
    return p.parse_args()


def pretty_cur(c: str) -> str:
    return "rUSD" if c == RUSD_HEX else c


def _is_xrp(currency: str) -> bool:
    return currency == XRP


def _amt_floor_out(d: Decimal, out_is_xrp: bool):
    """Decimal -> Amount floored to XRPL quantum (mirrors unit tests)."""
    if out_is_xrp:
        drops = int((d / XRP_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return XRPAmount(value=drops)
    else:
        units = int((d / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return IOUAmount.from_components(units, -15)


def _fmt_dec(d: Decimal, is_xrp: bool) -> str:
    q = XRP_QUANTUM if is_xrp else IOU_QUANTUM
    places = 6 if is_xrp else 15
    return f"{quantize_down(d, q):.{places}f}"


def main():
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        inp = json.load(f)

    # -----------------------------
    # 0) Ex-ante network fee approximation (from build script)
    # -----------------------------
    wm = inp.get("window_meta", {})
    net_fee_xrp = Decimal(wm.get("net_fee_approx_xrp", "0"))

    # -----------------------------
    # 1) Read AMM snapshot + trade
    # -----------------------------
    amm_in = inp["amm"]
    trade_in = inp["trade"]
    real_in = inp.get("real_world", {})

    x_reserve = Decimal(amm_in["x_reserve"])
    y_reserve = Decimal(amm_in["y_reserve"])
    fee = Decimal(amm_in["fee"])

    in_currency = trade_in["in_currency"]
    out_currency = trade_in["out_currency"]
    target_out = Decimal(trade_in["target_out"])

    IN_IS_XRP = _is_xrp(in_currency)
    OUT_IS_XRP = _is_xrp(out_currency)

    # Routing convention: x == IN side, y == OUT side
    amm = AMM(
        x_reserve, y_reserve, fee,
        x_is_xrp=IN_IS_XRP,
        y_is_xrp=OUT_IS_XRP
    )

    in_cur_p = pretty_cur(in_currency)
    out_cur_p = pretty_cur(out_currency)

    # -----------------------------
    # 2) Pre-state (spot price)
    #    Use reserves ratio (in/out), not SPQ price-view.
    # -----------------------------
    pre_spot_price = (x_reserve / y_reserve) if y_reserve != 0 else None

    # -----------------------------
    # 3) Run AMM-only quote
    # -----------------------------
    view = RouterQuoteView(lambda: [], amm=amm)
    quote = view.preview_out(_amt_floor_out(target_out, OUT_IS_XRP))

    summary = quote["summary"]
    pred_total_out = summary["total_out"]
    pred_total_in = summary["total_in"]

    pred_out_dec = amount_to_decimal(pred_total_out)
    pred_in_dec = amount_to_decimal(pred_total_in)

    # All-in (incl. ex-ante network fee proxy)
    pred_in_all = pred_in_dec + net_fee_xrp
    pred_avg_price_inout_all: Optional[Decimal] = (
        (pred_in_all / pred_out_dec) if pred_out_dec != 0 else None
    )

    # Post-state inputs from last AMM slice
    dx_eff_dec = None
    fee_paid_dec = None
    slip_dec = None

    for t in quote["slices"]:
        if t.get("src") != "AMM":
            continue
        if t.get("dx_eff") is not None:
            dx_eff_dec = amount_to_decimal(t["dx_eff"])
        if t.get("fee_paid") is not None:
            fee_paid_dec = amount_to_decimal(t["fee_paid"])
        if t.get("slippage_price_premium") is not None:
            slip_dec = Decimal(str(t["slippage_price_premium"]))

    # -----------------------------
    # Compute model post spot price from implied post reserves (in/out).
    # -----------------------------
    post_spot_price = None
    if dx_eff_dec is not None and pred_out_dec is not None:
        x_post_model = x_reserve + dx_eff_dec
        y_post_model = y_reserve - pred_out_dec
        if y_post_model != 0:
            post_spot_price = x_post_model / y_post_model

    # -----------------------------
    # 4) PRINT: Pre reserves + target_out
    # -----------------------------
    print("\n=== Pre AMM state ===")
    print(
        f"pre_reserves   : x={_fmt_dec(x_reserve, IN_IS_XRP)} {in_cur_p}, "
        f"y={_fmt_dec(y_reserve, OUT_IS_XRP)} {out_cur_p}, fee={fmt_dec(fee, places=6)}"
    )
    if pre_spot_price is not None:
        print(f"pre_spot_price : {fmt_dec(pre_spot_price, places=15)} ({in_cur_p} per {out_cur_p})")
    else:
        print(f"pre_spot_price : N/A ({in_cur_p} per {out_cur_p})")
    print(f"target_out     : {_fmt_dec(target_out, OUT_IS_XRP)} {out_cur_p}")
    print(f"net_fee_approx : {net_fee_xrp} XRP (ex-ante proxy)")

    # -----------------------------
    # 5) PRINT: Model predicted metrics + post state (ALL-IN)
    # -----------------------------
    print("\n=== Model (AMM-only, all-in) ===")
    print(f"pred_total_in      : {_fmt_dec(pred_in_all, IN_IS_XRP)} {in_cur_p}")
    if pred_avg_price_inout_all is not None:
        print(f"pred_avg_price(in/out): {pred_avg_price_inout_all} ({in_cur_p} per {out_cur_p})")

    if post_spot_price is not None:
        print(f"post_spot_price    : {fmt_dec(post_spot_price, places=15)} ({in_cur_p} per {out_cur_p})")
    if dx_eff_dec is not None or fee_paid_dec is not None or slip_dec is not None:
        print(
            f"dx_eff / fee / slip: "
            f"dx_eff={_fmt_dec(dx_eff_dec, IN_IS_XRP) if dx_eff_dec is not None else 'N/A'} {in_cur_p}, "
            f"fee_paid={_fmt_dec(fee_paid_dec, IN_IS_XRP) if fee_paid_dec is not None else 'N/A'} {in_cur_p}, "
            f"slip={fmt_dec(slip_dec, places=15) if slip_dec is not None else 'N/A'}"
        )

    # -----------------------------
    # 6) PRINT: Real-world metrics + post state (ALL-IN)
    # -----------------------------
    if real_in:
        real_in_dec = Decimal(real_in["asset_in_value"])
        real_out_dec = Decimal(real_in["asset_out_value"])

        real_in_all = real_in_dec + net_fee_xrp
        real_avg_price_all = (real_in_all / real_out_dec) if real_out_dec != 0 else None

        print("\n=== Real-world (anchor trade, all-in) ===")
        print(f"real_total_in      : {_fmt_dec(real_in_all, IN_IS_XRP)} {in_cur_p}")
        if real_avg_price_all is not None:
            print(f"real_avg_price(in/out): {real_avg_price_all} ({in_cur_p} per {out_cur_p})")

        real_post_spot = None
        if real_in.get("post_spot_price_inout"):
            real_post_spot = Decimal(real_in["post_spot_price_inout"])
        print(f"real_post_spot_price: {real_post_spot if real_post_spot is not None else 'N/A'} ({in_cur_p} per {out_cur_p})")

        # -----------------------------
        # 7) Errors (ALL-IN)
        # -----------------------------
        print("\n=== Errors ===")
        in_err = pred_in_all - real_in_all
        out_err = pred_out_dec - real_out_dec
        print(f"in_error  (pred-real): {in_err}")
        print(f"out_error (pred-real): {out_err}")
        if real_avg_price_all is not None and pred_avg_price_inout_all is not None:
            print(f"avg_price_error: {pred_avg_price_inout_all - real_avg_price_all}")
        if real_post_spot is not None and post_spot_price is not None:
            print(f"post_spot_price_error: {post_spot_price - real_post_spot}")


if __name__ == "__main__":
    main()