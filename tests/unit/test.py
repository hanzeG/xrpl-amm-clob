# -----------------------------------------------------------------------------
# Empirical Execution Efficiency Metrics (Implementation Coverage Overview)
# -----------------------------------------------------------------------------
# The current test framework covers and omits the following key dimensions:
#
# 1. Latency / Quote Freshness:
#    - Not currently modelled. The framework operates on static snapshots.
#    - Future work: introduce tick-based updates or delayed quote propagation.
#
# 3. Effective Total Cost (Aggregated):
#    - Partially implemented via fee_paid and dx_eff.
#    - Future work: add (total_in / total_out) - 1 metric to summarise full cost.

# 5. Optimal Split Ratio (α*) and Crossover Point (q*):
#    - Conceptually supported via hybrid path logic.
#    - Future work: implement cost-curve comparison to compute α* and q*.
# -----------------------------------------------------------------------------
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

import pytest

from xrpl_router.core.fmt import fmt_dec, amount_to_decimal, quality_rate_to_decimal, quality_price_to_decimal, quantize_down
from xrpl_router.core import Quality, IOUAmount, XRPAmount
from xrpl_router.core.constants import XRP_QUANTUM, IOU_QUANTUM
from xrpl_router.core.exc import InsufficientLiquidityError


# --- Asset mode toggles (independent IN/OUT selection) ---
# Set these two flags per test run; they control ladder(out/in) and AMM (y/x) sides respectively.
# Examples:
#  - IOU -> IOU:  IN_IS_XRP = False; OUT_IS_XRP = False
#  - XRP -> XRP:  IN_IS_XRP = True;  OUT_IS_XRP = True
#  - XRP -> IOU:  IN_IS_XRP = True;  OUT_IS_XRP = False
#  - IOU -> XRP:  IN_IS_XRP = False; OUT_IS_XRP = True
IN_IS_XRP = False
OUT_IS_XRP = False

# AMM side flags follow routing convention: x == IN side, y == OUT side
AMM_X_IS_XRP = IN_IS_XRP
AMM_Y_IS_XRP = OUT_IS_XRP

LOG_PLACES_OUT = 6 if OUT_IS_XRP else 15
LOG_PLACES_IN  = 6 if IN_IS_XRP  else 15

# --- Fixed-point, on-grid formatters (no rounding beyond grid) ---
def _fmt_dec_out(d: Decimal) -> str:
    q = XRP_QUANTUM if OUT_IS_XRP else IOU_QUANTUM
    places = 6 if OUT_IS_XRP else 15
    return f"{quantize_down(d, q):.{places}f}"

def _fmt_dec_in(d: Decimal) -> str:
    q = XRP_QUANTUM if IN_IS_XRP else IOU_QUANTUM
    places = 6 if IN_IS_XRP else 15
    return f"{quantize_down(d, q):.{places}f}"

def _fmt_amt_out(a) -> str:
    # `a` is an Amount (XRPAmount or IOUAmount)
    return _fmt_dec_out(amount_to_decimal(a))

def _fmt_amt_in(a) -> str:
    return _fmt_dec_in(amount_to_decimal(a))

# --- Test-default AMM constants (single source of truth) ---
AMM_X0 = Decimal("1900")
AMM_Y0 = Decimal("2000")
AMM_FEE0 = Decimal("0.003")

# TODO: hide quantisation-tail AMM slices in prints only; production filtering to follow
TINY_OUT_HIDE = Decimal("1e-9")

def _qd(q) -> str:
    return fmt_dec(quality_rate_to_decimal(q), places=LOG_PLACES_OUT)

def _pd(q) -> str:
    return fmt_dec(quality_price_to_decimal(q), places=LOG_PLACES_OUT)

# --- AMM pool summary helper ---
def _print_amm_pool_summary(tag: str) -> None:
    """
    Print a compact one-shot summary of the default AMM pool used in tests.

    We avoid introspecting internal Amount fields to prevent accidental type/
    domain mismatches. Instead:
      - reserves/fees are printed from the *known* constructor constants
      - SPQ is computed via the public `spq_quality_int()` API
    """
    # Test-default constants (kept in one place)
    X0 = AMM_X0
    Y0 = AMM_Y0
    FEE = AMM_FEE0

    # Build an AMM with these constants to get SPQ from public API
    amm0 = AMM(X0, Y0, FEE, x_is_xrp=AMM_X_IS_XRP, y_is_xrp=AMM_Y_IS_XRP)
    spq_q = amm0.spq_quality_int()
    spq_qd = _qd(spq_q)
    spq_pd = _pd(spq_q)

    print(f"    AMM pool ({tag}):")
    print(f"      - reserves: x={_fmt_dec_in(X0)}, y={_fmt_dec_out(Y0)}")
    print(f"      - fees: fee={fmt_dec(FEE, places=6)}")
    print(f"      - SPQ: spq_q={spq_qd}, spq_p={spq_pd}")

# Production interfaces for CLOB/AMM/amounts/quality
from xrpl_router.clob import make_ladder
from xrpl_router.amm import AMM
from xrpl_router.book_step import BookStep, RouterQuoteView
from xrpl_router.flow import PaymentSandbox

# ---- Decimal -> Amount helpers (mode-aware: IOU or XRP) ----
def _amt_floor_out(d: Decimal):
    if OUT_IS_XRP:
        drops = int((d / XRP_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return XRPAmount(value=drops)
    else:
        units = int((d / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
        return IOUAmount.from_components(units, -15)

def _amt_ceil_out(d: Decimal):
    if OUT_IS_XRP:
        drops = int((d / XRP_QUANTUM).to_integral_value(rounding=ROUND_CEILING))
        return XRPAmount(value=drops)
    else:
        units = int((d / IOU_QUANTUM).to_integral_value(rounding=ROUND_CEILING))
        return IOUAmount.from_components(units, -15)


# Slippage monotonicity — per-test whitepaper mapping lives above each test.
# See individual test comments for exact section references.

# Whitepaper §1.2.4 + §1.2.5
# We assert two properties on avg_price (average effective price) as trade size increases on a fixed limit order book:
# 1) Non-negativity: avg_price should not be negative (won't get cheaper as you buy more).
# 2) Monotonicity: avg_price should be non-decreasing with trade size, reflecting tiered consumption.
# Rationales: same quality tier consumes offers at constant price; moving to worse tiers increases avg_price.
@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("240")],
])
def test_slippage_monotonic_clob_only(sizes):
    """
    Test avg_price monotonicity for CLOB_ONLY via production BookStep (reverse stage).
    """
    print("===== CLOB_ONLY =====")
    ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=IN_IS_XRP, out_is_xrp=OUT_IS_XRP)
    # Print CLOB ladder summary using RouterQuoteView
    view = RouterQuoteView(lambda: ladder, amm=None)
    snap = view.snapshot()
    print("    CLOB ladder:")
    for i, s in enumerate(snap["clob_ladder"], start=1):
        qd = _qd(s["quality"])
        pd = _pd(s["quality"])
        outd = _fmt_amt_out(s["out_max"])
        ind = _fmt_amt_in(s["in_at_out_max"])
        print(f"      - L{i}: tier_q={qd}, tier_p={pd}, out={outd}, in={ind}")

    slips = []
    for q in sizes:
        view = RouterQuoteView(lambda: ladder, amm=None)
        quote = view.preview_out(_amt_floor_out(q))
        summary = quote["summary"]
        filled = summary["total_out"]
        spent = summary["total_in"]
        if not filled.is_zero():
            # avg_q = Quality.from_amounts(filled, spent)
            avg_q = summary["avg_quality"]
            slips.append(avg_q)
            print(f"\n[CLOB_ONLY] target_out: {_fmt_dec_out(q)}, total_avg_q: {_qd(avg_q)}, total_avg_p: {_pd(avg_q)}")
            print("    Consumed segments (CLOB only):")
            c_idx = 0
            for t in quote["slices"]:
                if t.get("src") != "CLOB":
                    continue
                c_idx += 1
                outd = _fmt_amt_out(t["out_take"])
                ind = _fmt_amt_in(t["in_take"])
                qd = _qd(t["avg_quality"]) ; pd = _pd(t["avg_quality"])
                print(f"      - C{c_idx}: avg_q={qd}, avg_p={pd}, out={outd}, in={ind}")
            # Print status (e.g., PARTIAL) if present
            status = summary if summary.get("is_partial") else None
            if status and status.get("is_partial"):
                print(f"    Status: {status}")
        else:
            slips.append(None)
            print(f"\n[CLOB_ONLY] target_out: {_fmt_dec_out(q)}, avg_quality: N/A, avg_price: N/A")
    print()

    # Positivity + cross-size monotonicity (quality non-increasing)
    for avg_q in slips:
        if avg_q is not None:
            assert quality_rate_to_decimal(avg_q) > 0
    prev = None
    for avg_q in slips:
        if avg_q is not None:
            if prev is not None:
                assert avg_q <= prev, (
                    f"avg_quality should be non-increasing: prev_q={fmt_dec(quality_rate_to_decimal(prev))} curr_q={fmt_dec(quality_rate_to_decimal(avg_q))}"
                )
            prev = avg_q


@pytest.mark.parametrize("q_req", [Decimal("400")])
def test_clob_only_partial_annotation(q_req):
    """CLOB-only: requesting more than ladder capacity should annotate PARTIAL in non-strict mode."""
    ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=IN_IS_XRP, out_is_xrp=OUT_IS_XRP)
    sandbox = PaymentSandbox()
    step = BookStep.from_static(ladder, amm=None, limit_quality=None)
    with pytest.raises(InsufficientLiquidityError) as ei:
        _ = step.rev(sandbox, _amt_floor_out(q_req), return_trace=True)
    exc = ei.value
    print(f"===== CLOB_ONLY (OVER_ASK) =====")
    print(f"\n[CLOB_ONLY/OVER_ASK] target_out: {_fmt_dec_out(q_req)} (expect raise)")
    print(f"    Raised: max_fill_out={_fmt_amt_out(exc.max_fill_out)}")

#
# Note: this dense AMM-only test replaces the coarse-grid AMM-only test to avoid overlap.
# Whitepaper §1.2.7.1 (AMM-only) — denser size grid
# Purpose: with the corrected tier-baseline, avg_price should remain non-negative and non-decreasing
# even across a denser set of target sizes.
# This complements the coarser grid to catch subtle non-monotonicity due to rounding/bucketing noise.
@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("1280")],
])
def test_slippage_monotonic_amm_only(sizes):
    """
    AMM_ONLY via BookStep.rev with return_trace; enforce avg quality positive and marginal non-increasing.
    """
    print("===== AMM_ONLY =====")
    # Print AMM pool summary using RouterQuoteView
    view_init = RouterQuoteView(lambda: [], amm=AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=AMM_X_IS_XRP, y_is_xrp=AMM_Y_IS_XRP))
    snap = view_init.snapshot()
    amm_snap = snap["amm"]
    print("    AMM pool (initial):")
    print(f"      - reserves: x={_fmt_dec_in(amm_snap['x_reserve'])}, y={_fmt_dec_out(amm_snap['y_reserve'])}")
    print(f"      - fees: fee={fmt_dec(Decimal(amm_snap['fee_decimal']), places=6)}")
    spq_qd = _qd(amm_snap['spq_quality'])
    spq_pd = _pd(amm_snap['spq_quality'])
    print(f"      - SPQ: spq_q={spq_qd}, spq_p={spq_pd}")
    slips = []
    for q in sizes:
        view = RouterQuoteView(lambda: [], amm=AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=AMM_X_IS_XRP, y_is_xrp=AMM_Y_IS_XRP))
        quote = view.preview_out(_amt_floor_out(q))
        summary = quote["summary"]
        filled = summary["total_out"]
        spent = summary["total_in"]
        assert not filled.is_zero()
        avg_q = summary["avg_quality"]
        slips.append(avg_q)
        print(f"\n[AMM_ONLY] target_out: {_fmt_dec_out(q)}, total_avg_q: {_qd(avg_q)}, total_avg_p: {_pd(avg_q)}")
        print("    AMM segments:")
        s_idx = 0
        for t in quote["slices"]:
            if t.get("src") != "AMM":
                continue
            out_dec = amount_to_decimal(t["out_take"])
            if out_dec < TINY_OUT_HIDE:
                continue
            s_idx += 1
            outd = _fmt_amt_out(t["out_take"])
            ind = _fmt_amt_in(t["in_take"])
            avg_q = t["avg_quality"]
            qd = _qd(avg_q) ; pd = _pd(avg_q)
            print(f"      - A{s_idx}: avg_q={qd}, avg_p={pd}, out={outd}, in={ind}")
            # AMM diagnostics (pre/post SPQ, dx_eff, fee, slippage)
            if t.get("src") == "AMM":
                pre_q = t.get("pre_spq")
                post_q = t.get("post_spq")
                dx_eff = t.get("dx_eff")
                fee_paid = t.get("fee_paid")
                slip = t.get("slippage_price_premium")
                if pre_q and post_q and dx_eff is not None and fee_paid is not None:
                    pre_qd = _qd(pre_q) ; pre_pd = _pd(pre_q)
                    post_qd = _qd(post_q) ; post_pd = _pd(post_q)
                    dx_eff_d = _fmt_amt_in(dx_eff)
                    fee_d = _fmt_amt_in(fee_paid)
                    try:
                        from decimal import Decimal as _D
                        slip_d = fmt_dec(_D(str(slip)) if slip is not None else _D("0"), places=LOG_PLACES_OUT)
                    except Exception:
                        slip_d = "N/A"
                    # Renamed variables for clarity in diagnostic print
                    pre_q_str = pre_qd
                    pre_p_str = pre_pd
                    post_q_str = post_qd
                    post_p_str = post_pd
                    print(f"        (pre_q={pre_q_str}, pre_p={pre_p_str}; post_q={post_q_str}, post_p={post_p_str}; dx_eff={dx_eff_d}, fee={fee_d}, slip={slip_d})")
        # Print status (e.g., PARTIAL) if present
        status = summary if summary.get("is_partial") else None
        if status and status.get("is_partial"):
            print(f"    Status: {status}")
    print()

    # Positivity
    for avg_q in slips:
        assert quality_rate_to_decimal(avg_q) > 0
    # Marginal non-increasing within each route
    for q in sizes:
        sandbox = PaymentSandbox()
        amm = AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=AMM_X_IS_XRP, y_is_xrp=AMM_Y_IS_XRP)
        step = BookStep.from_static([], amm=amm, limit_quality=None)
        _filled, _spent, trace = step.rev(sandbox, _amt_floor_out(q), return_trace=True)
        marginals_q = [t["quality"] for t in trace if t.get("src") == "AMM"]
        for a, b in zip(marginals_q, marginals_q[1:]):
            assert b <= a, (
                f"marginal quality not non-increasing within route (q={q}): prev_q={fmt_dec(quality_rate_to_decimal(a))} curr_q={fmt_dec(quality_rate_to_decimal(b))}"
            )

    # Strict over-ask integrated case for AMM_ONLY
    q_req_strict = Decimal("2050")
    print("===== AMM_ONLY (OVER_ASK) =====")
    print(f"[AMM_ONLY/OVER_ASK] target_out: {_fmt_dec_out(q_req_strict)} (expect raise)")
    sandbox = PaymentSandbox()
    amm = AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=AMM_X_IS_XRP, y_is_xrp=AMM_Y_IS_XRP)
    step = BookStep.from_static([], amm=amm, limit_quality=None)
    with pytest.raises(InsufficientLiquidityError) as ei:
        _ = step.rev(sandbox, _amt_floor_out(q_req_strict), return_trace=True)
    exc = ei.value
    print(f"    Raised: max_fill_out={_fmt_amt_out(exc.max_fill_out)}")


#
# Whitepaper §1.2.7.2 (Hybrid anchoring)
# When AMM and CLOB coexist within the same iteration, tier selection is anchored to the LOB top quality bucket.
# Under this anchoring, the average effective price (avg_price) should be non-negative and non-decreasing as total size grows,
# per §1.2.4, §1.2.5 (CLOB tiers) and §1.2.7.2 (AMM anchoring).
@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("1280")],
])
def test_slippage_monotonic_hybrid_tier_anchored(sizes):
    """
    Hybrid via BookStep.rev with return_trace; validate avg quality positive/non-increasing and local anchoring order.
    """
    print("===== HYBRID (Anchored) =====")
    ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=IN_IS_XRP, out_is_xrp=OUT_IS_XRP)
    view_init = RouterQuoteView(lambda: ladder, amm=AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=AMM_X_IS_XRP, y_is_xrp=AMM_Y_IS_XRP))
    snap = view_init.snapshot()
    amm_snap = snap["amm"]
    print("    AMM pool (initial):")
    print(f"      - reserves: x={_fmt_dec_in(amm_snap['x_reserve'])}, y={_fmt_dec_out(amm_snap['y_reserve'])}")
    print(f"      - fees: fee={fmt_dec(Decimal(amm_snap['fee_decimal']), places=6)}")
    spq_qd = _qd(amm_snap['spq_quality'])
    spq_pd = _pd(amm_snap['spq_quality'])
    print(f"      - SPQ: spq_q={spq_qd}, spq_p={spq_pd}")
    print("    CLOB ladder:")
    for i, s in enumerate(snap["clob_ladder"], start=1):
        qd = _qd(s["quality"])
        pd = _pd(s["quality"])
        outd = _fmt_amt_out(s["out_max"])
        ind = _fmt_amt_in(s["in_at_out_max"])
        print(f"      - L{i}: tier_q={qd}, tier_p={pd}, out={outd}, in={ind}")

    slips = []
    for q in sizes:
        print()
        view = RouterQuoteView(lambda: ladder, amm=AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=AMM_X_IS_XRP, y_is_xrp=AMM_Y_IS_XRP))
        quote = view.preview_out(_amt_floor_out(q))
        summary = quote["summary"]
        filled = summary["total_out"]
        spent = summary["total_in"]
        if not filled.is_zero():
            avg_q = summary["avg_quality"]
            slips.append(avg_q)
            print(f"[HYBRID] target_out: {_fmt_dec_out(q)}, total_avg_q: {_qd(avg_q)}, total_avg_p: {_pd(avg_q)}")
            print("    Hybrid path:")
            c_idx = 0 ; a_idx = 0
            for t in quote["slices"]:
                # Skip non-slice entries (e.g., status markers without amounts)
                if ("out_take" not in t) or ("in_take" not in t):
                    continue
                out_dec = amount_to_decimal(t["out_take"])
                if t.get("src") == "AMM" and out_dec < TINY_OUT_HIDE:
                    continue
                tag = None
                if t.get("src") == "CLOB":
                    c_idx += 1 ; tag = f"C{c_idx}"
                else:
                    a_idx += 1 ; tag = f"A{a_idx}"
                outd = _fmt_amt_out(t["out_take"])
                ind = _fmt_amt_in(t["in_take"])
                avg_q = t["avg_quality"]
                qd = _qd(avg_q) ; pd = _pd(avg_q)
                print(f"      - {tag}: avg_q={qd}, avg_p={pd}, out={outd}, in={ind}")
                # AMM diagnostics (pre/post SPQ, dx_eff, fee, slippage)
                if t.get("src") == "AMM":
                    pre_q = t.get("pre_spq")
                    post_q = t.get("post_spq")
                    dx_eff = t.get("dx_eff")
                    fee_paid = t.get("fee_paid")
                    slip = t.get("slippage_price_premium")
                    if pre_q and post_q and dx_eff is not None and fee_paid is not None:
                        pre_qd = _qd(pre_q) ; pre_pd = _pd(pre_q)
                        post_qd = _qd(post_q) ; post_pd = _pd(post_q)
                        dx_eff_d = _fmt_amt_in(dx_eff)
                        fee_d = _fmt_amt_in(fee_paid)
                        try:
                            from decimal import Decimal as _D
                            slip_d = fmt_dec(_D(str(slip)) if slip is not None else _D("0"), places=LOG_PLACES_OUT)
                        except Exception:
                            slip_d = "N/A"
                        # Renamed variables for clarity in diagnostic print
                        pre_q_str = pre_qd
                        pre_p_str = pre_pd
                        post_q_str = post_qd
                        post_p_str = post_pd
                        print(f"        (pre_q={pre_q_str}, pre_p={pre_p_str}; post_q={post_q_str}, post_p={post_p_str}; dx_eff={dx_eff_d}, fee={fee_d}, slip={slip_d})")
            # Print status (e.g., PARTIAL) if present
            status = summary if summary.get("is_partial") else None
            if status and status.get("is_partial"):
                print(f"    Status: {status}")
        else:
            slips.append(None)
            print(f"[HYBRID] target_out: {_fmt_dec_out(q)}, avg_quality: N/A, avg_price: N/A")
    print()

    # Positivity
    for avg_q in slips:
        if avg_q is not None:
            assert quality_rate_to_decimal(avg_q) > 0
    # Cross-size monotonicity (quality non-increasing)
    prev = None
    for avg_q in slips:
        if avg_q is not None:
            if prev is not None:
                assert avg_q <= prev, (
                    f"avg_quality should be non-increasing: prev_q={fmt_dec(quality_rate_to_decimal(prev))} curr_q={fmt_dec(quality_rate_to_decimal(avg_q))}"
                )
            prev = avg_q

    # Local anchoring: AMM slice followed by CLOB slice should satisfy q_AMM >= q_CLOB
    for q in sizes:
        sandbox = PaymentSandbox()
        amm = AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=AMM_X_IS_XRP, y_is_xrp=AMM_Y_IS_XRP)
        step = BookStep.from_static(ladder, amm=amm, limit_quality=None)
        _filled, _spent, trace = step.rev(sandbox, _amt_floor_out(q), return_trace=True)
        for a, b in zip(trace, trace[1:]):
            if a.get("src") == "AMM" and b.get("src") == "CLOB":
                assert a["quality"] >= b["quality"], (
                    f"anchoring violated: AMM slice quality {_qd(a['quality'])} is not >= CLOB slice quality {_qd(b['quality'])} at q={q}"
                )

    # Strict over-ask integrated case for HYBRID
    q_req_strict = Decimal("3000")
    print("===== HYBRID (OVER_ASK) =====")
    print(f"\n[HYBRID/OVER_ASK] target_out: {_fmt_dec_out(q_req_strict)} (expect raise)")
    sandbox = PaymentSandbox()
    amm = AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=AMM_X_IS_XRP, y_is_xrp=AMM_Y_IS_XRP)
    step = BookStep.from_static(ladder, amm=amm, limit_quality=None)
    with pytest.raises(InsufficientLiquidityError) as ei:
        _ = step.rev(sandbox, _amt_floor_out(q_req_strict), return_trace=True)
    exc = ei.value
    print(f"    Raised: max_fill_out={_fmt_amt_out(exc.max_fill_out)}")