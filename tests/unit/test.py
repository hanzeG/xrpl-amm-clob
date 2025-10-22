from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

import pytest

from xrpl_router.core.fmt import fmt_dec, amount_to_decimal, quality_rate_to_decimal, quality_price_to_decimal
from xrpl_router.core import Quality, IOUAmount, IOU_QUANTUM
from xrpl_router.core.exc import InsufficientLiquidityError

LOG_PLACES = 6

# --- Test-default AMM constants (single source of truth) ---
AMM_X0 = Decimal("1950")
AMM_Y0 = Decimal("2000")
AMM_FEE0 = Decimal("0.003")
AMM_TR_IN0 = Decimal("0")
AMM_TR_OUT0 = Decimal("0")

# TODO: hide quantisation-tail AMM slices in prints only; production filtering to follow
TINY_OUT_HIDE = Decimal("1e-9")

def _qd(q) -> str:
    return fmt_dec(quality_rate_to_decimal(q), places=LOG_PLACES)

def _pd(q) -> str:
    return fmt_dec(quality_price_to_decimal(q), places=LOG_PLACES)

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
    TR_IN = AMM_TR_IN0
    TR_OUT = AMM_TR_OUT0

    # Build an AMM with these constants to get SPQ from public API
    amm0 = AMM(X0, Y0, FEE, x_is_xrp=False, y_is_xrp=False, tr_in=TR_IN, tr_out=TR_OUT)
    spq_q = amm0.spq_quality_int()
    spq_qd = _qd(spq_q)
    spq_pd = _pd(spq_q)

    print(f"    AMM pool ({tag}):")
    print(f"      - reserves: x={fmt_dec(X0, places=LOG_PLACES)}, y={fmt_dec(Y0, places=LOG_PLACES)}")
    print(f"      - fees: fee={fmt_dec(FEE, places=6)}, tr_in={fmt_dec(TR_IN, places=6)}, tr_out={fmt_dec(TR_OUT, places=6)}")
    print(f"      - SPQ: q={spq_qd}, p={spq_pd}")

# Production interfaces for CLOB/AMM/amounts/quality
from xrpl_router.clob import make_ladder
from xrpl_router.amm import AMM
from xrpl_router.book_step import BookStep
from xrpl_router.flow import PaymentSandbox
# ---- Decimal -> IOUAmount helpers (tests assume IOU legs) ----
def _iou_floor(d: Decimal) -> IOUAmount:
    units = int((d / IOU_QUANTUM).to_integral_value(rounding=ROUND_FLOOR))
    return IOUAmount.from_components(units, -15)

def _iou_ceil(d: Decimal) -> IOUAmount:
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
    [Decimal("20"), Decimal("40"), Decimal("80"), Decimal("160"), Decimal("240"), Decimal("400")],  # expanded sizes
])
def test_slippage_monotonic_clob_only(sizes):
    """
    Test avg_price monotonicity for CLOB_ONLY via production BookStep (reverse stage).
    """
    print("===== CLOB_ONLY =====")
    ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=False, out_is_xrp=False)
    # Print CLOB ladder summary
    print("    CLOB ladder:")
    for i, s in enumerate(ladder, start=1):
        qd = _qd(s.quality)
        pd = _pd(s.quality)
        outd = fmt_dec(amount_to_decimal(s.out_max), places=LOG_PLACES)
        ind = fmt_dec(amount_to_decimal(s.in_at_out_max), places=LOG_PLACES)
        print(f"      - L{i}: q={qd}, p={pd}, out={outd}, in={ind}")

    slips = []
    for q in sizes:
        sandbox = PaymentSandbox()
        step = BookStep.from_static(ladder, amm=None, limit_quality=None)
        filled, spent, trace = step.rev(sandbox, _iou_floor(q), return_trace=True)
        if not filled.is_zero():
            avg_q = Quality.from_amounts(filled, spent)
            slips.append(avg_q)
            print(f"\n[CLOB_ONLY] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: {_qd(avg_q)}, avg_price: {_pd(avg_q)}")
            print("    Consumed segments (CLOB only):")
            c_idx = 0
            for t in trace:
                if t.get("src") != "CLOB":
                    continue
                c_idx += 1
                outd = fmt_dec(amount_to_decimal(t["take_out"]), places=LOG_PLACES)
                ind = fmt_dec(amount_to_decimal(t["take_in"]), places=LOG_PLACES)
                qd = _qd(t["quality"]) ; pd = _pd(t["quality"]) 
                print(f"      - C{c_idx}: q={qd}, p={pd}, out={outd}, in={ind}")
            # Print status (e.g., PARTIAL) if present
            status = next((t for t in reversed(trace) if t.get("status")), None)
            if status:
                print(f"    Status: {status}")
        else:
            slips.append(None)
            print(f"\n[CLOB_ONLY] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: N/A, avg_price: N/A")
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
    ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=False, out_is_xrp=False)
    sandbox = PaymentSandbox()
    step = BookStep.from_static(ladder, amm=None, limit_quality=None)
    filled, spent, trace = step.rev(sandbox, _iou_floor(q_req), return_trace=True, require_full_fill=False)
    # Capacity is 6 * 40 = 240
    assert amount_to_decimal(filled) == Decimal("240")
    # Trace must include PARTIAL marker at the end
    assert any(t.get("status") == "PARTIAL" for t in trace), "Expected PARTIAL status in trace for over-capacity CLOB-only request"
    # Optional: check fill_ratio ~= 0.6
    status = next((t for t in reversed(trace) if t.get("status") == "PARTIAL"), None)
    if status and "fill_ratio" in status:
        assert abs(Decimal(str(status["fill_ratio"])) - Decimal("0.6")) < Decimal("1e-12")


@pytest.mark.parametrize("q_req", [Decimal("400")])
def test_clob_only_strict_raises(q_req):
    """CLOB-only: requesting more than ladder capacity should raise in strict mode."""
    ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=False, out_is_xrp=False)
    sandbox = PaymentSandbox()
    step = BookStep.from_static(ladder, amm=None, limit_quality=None)
    with pytest.raises(InsufficientLiquidityError) as ei:
        _ = step.rev(sandbox, _iou_floor(q_req), return_trace=True, require_full_fill=True)
    # Exception should report max_fill_out = 240
    exc = ei.value
    assert amount_to_decimal(exc.max_fill_out) == Decimal("240")

#
# Note: this dense AMM-only test replaces the coarse-grid AMM-only test to avoid overlap.
# Whitepaper §1.2.7.1 (AMM-only) — denser size grid
# Purpose: with the corrected tier-baseline, avg_price should remain non-negative and non-decreasing
# even across a denser set of target sizes.
# This complements the coarser grid to catch subtle non-monotonicity due to rounding/bucketing noise.
@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("80"), Decimal("320"), Decimal("1280"), Decimal("5000")],
])
def test_slippage_monotonic_amm_only_dense(sizes):
    """
    AMM_ONLY via BookStep.rev with return_trace; enforce avg quality positive and marginal non-increasing.
    """
    print("===== AMM_ONLY/DENSE =====")
    _print_amm_pool_summary("initial")
    slips = []
    for q in sizes:
        sandbox = PaymentSandbox()
        amm = AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=False, y_is_xrp=False, tr_in=AMM_TR_IN0, tr_out=AMM_TR_OUT0)
        step = BookStep.from_static([], amm=amm, limit_quality=None)
        filled, spent, trace = step.rev(sandbox, _iou_floor(q), return_trace=True)
        assert not filled.is_zero()
        avg_q = Quality.from_amounts(filled, spent)
        slips.append(avg_q)
        print(f"\n[AMM_ONLY/DENSE] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: {_qd(avg_q)}, avg_price: {_pd(avg_q)}")
        print("    AMM segments:")
        s_idx = 0
        for t in trace:
            if t.get("src") != "AMM":
                continue
            out_dec = amount_to_decimal(t["take_out"]) 
            if out_dec < TINY_OUT_HIDE:
                continue
            s_idx += 1
            outd = fmt_dec(out_dec, places=LOG_PLACES)
            ind = fmt_dec(amount_to_decimal(t["take_in"]), places=LOG_PLACES)
            qd = _qd(t["quality"]) ; pd = _pd(t["quality"]) 
            print(f"      - A{s_idx}: q={qd}, p={pd}, out={outd}, in={ind}")
        # Print status (e.g., PARTIAL) if present
        status = next((t for t in reversed(trace) if t.get("status")), None)
        if status:
            print(f"    Status: {status}")
    print()

    # Positivity
    for avg_q in slips:
        assert quality_rate_to_decimal(avg_q) > 0
    # Marginal non-increasing within each route
    for q in sizes:
        sandbox = PaymentSandbox()
        amm = AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=False, y_is_xrp=False, tr_in=AMM_TR_IN0, tr_out=AMM_TR_OUT0)
        step = BookStep.from_static([], amm=amm, limit_quality=None)
        _filled, _spent, trace = step.rev(sandbox, _iou_floor(q), return_trace=True)
        marginals_q = [t["quality"] for t in trace if t.get("src") == "AMM"]
        for a, b in zip(marginals_q, marginals_q[1:]):
            assert b <= a, (
                f"marginal quality not non-increasing within route (q={q}): prev_q={fmt_dec(quality_rate_to_decimal(a))} curr_q={fmt_dec(quality_rate_to_decimal(b))}"
            )


#
# Whitepaper §1.2.7.2 (Hybrid anchoring)
# When AMM and CLOB coexist within the same iteration, tier selection is anchored to the LOB top quality bucket.
# Under this anchoring, the average effective price (avg_price) should be non-negative and non-decreasing as total size grows,
# per §1.2.4, §1.2.5 (CLOB tiers) and §1.2.7.2 (AMM anchoring).
@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("80"), Decimal("320"), Decimal("1280"), Decimal("5000")],
])
def test_slippage_monotonic_hybrid_tier_anchored(sizes):
    """
    Hybrid via BookStep.rev with return_trace; validate avg quality positive/non-increasing and local anchoring order.
    """
    print("===== HYBRID (Anchored) =====")
    _print_amm_pool_summary("initial")
    ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=False, out_is_xrp=False)
    # Print CLOB ladder summary
    print("    CLOB ladder:")
    for i, s in enumerate(ladder, start=1):
        qd = _qd(s.quality)
        pd = _pd(s.quality)
        outd = fmt_dec(amount_to_decimal(s.out_max), places=LOG_PLACES)
        ind = fmt_dec(amount_to_decimal(s.in_at_out_max), places=LOG_PLACES)
        print(f"      - L{i}: q={qd}, p={pd}, out={outd}, in={ind}")

    slips = []
    for q in sizes:
        print()
        sandbox = PaymentSandbox()
        amm = AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=False, y_is_xrp=False, tr_in=AMM_TR_IN0, tr_out=AMM_TR_OUT0)
        step = BookStep.from_static(ladder, amm=amm, limit_quality=None)
        filled, spent, trace = step.rev(sandbox, _iou_floor(q), return_trace=True)
        if not filled.is_zero():
            avg_q = Quality.from_amounts(filled, spent)
            slips.append(avg_q)
            print(f"[HYBRID] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: {_qd(avg_q)}, avg_price: {_pd(avg_q)}")
            print("    Hybrid path:")
            c_idx = 0 ; a_idx = 0
            for t in trace:
                # Skip non-slice entries (e.g., status markers without amounts)
                if ("take_out" not in t) or ("take_in" not in t):
                    continue
                out_dec = amount_to_decimal(t["take_out"]) 
                if t.get("src") == "AMM" and out_dec < TINY_OUT_HIDE:
                    continue
                tag = None
                if t.get("src") == "CLOB":
                    c_idx += 1 ; tag = f"C{c_idx}"
                else:
                    a_idx += 1 ; tag = f"A{a_idx}"
                outd = fmt_dec(out_dec, places=LOG_PLACES)
                ind = fmt_dec(amount_to_decimal(t["take_in"]), places=LOG_PLACES)
                qd = _qd(t["quality"]) ; pd = _pd(t["quality"]) 
                print(f"      - {tag}: src={t['src']}, q={qd}, p={pd}, out={outd}, in={ind}")
            # Print status (e.g., PARTIAL) if present
            status = next((t for t in reversed(trace) if t.get("status")), None)
            if status:
                print(f"    Status: {status}")
        else:
            slips.append(None)
            print(f"[HYBRID] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: N/A, avg_price: N/A")
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
        amm = AMM(AMM_X0, AMM_Y0, AMM_FEE0, x_is_xrp=False, y_is_xrp=False, tr_in=AMM_TR_IN0, tr_out=AMM_TR_OUT0)
        step = BookStep.from_static(ladder, amm=amm, limit_quality=None)
        _filled, _spent, trace = step.rev(sandbox, _iou_floor(q), return_trace=True)
        for a, b in zip(trace, trace[1:]):
            if a.get("src") == "AMM" and b.get("src") == "CLOB":
                assert a["quality"] >= b["quality"], (
                    f"anchoring violated: AMM slice quality {_qd(a['quality'])} is not >= CLOB slice quality {_qd(b['quality'])} at q={q}"
                )
