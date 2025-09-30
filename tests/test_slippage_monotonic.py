from decimal import Decimal

# Numerical tolerance for Decimal comparisons (covers rounding/bucketing noise)
EPS_ABS = Decimal("1e-12")
EPS_REL = Decimal("1e-12")

def geq_with_tol(x: Decimal, y: Decimal) -> bool:
    return x + max(EPS_ABS, y.copy_abs() * EPS_REL) >= y

import pytest

from xrpl_router.exec_modes import run_trade_mode, ExecutionMode


# ---------------------------
# Slippage monotonicity (hidden cost) — whitepaper alignment
# ---------------------------
# Expectation: as target_out increases on a fixed book/curve, the average price slippage
# (vs. baseline bucket) should be non-negative and non-decreasing.


@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("80"), Decimal("160")],  # fits within default CLOB depth (6*40=240)
])
def test_slippage_monotonic_clob_only(clob_segments_default, sizes):
    slips = []
    for q in sizes:
        res = run_trade_mode(
            ExecutionMode.CLOB_ONLY,
            target_out=q,
            segments=clob_segments_default,
        )
        er = res.report
        assert er is not None
        # basic sanity: some fill (book should have enough depth for these sizes)
        assert er.total_out > 0
        # slippage is price_effective - baseline_price (1/quality_bucket), cf. router impl
        slips.append(er.slippage_price_avg)

    # Non-negative and non-decreasing (within numerical tolerance)
    for s in slips:
        assert geq_with_tol(s, Decimal(0))
    for a, b in zip(slips, slips[1:]):
        assert geq_with_tol(b, a)


@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("60"), Decimal("120")],
])
def test_slippage_monotonic_amm_only(clob_segments_default, amm_curve_default, amm_ctx_multipath, sizes):
    # segments are ignored in AMM_ONLY, but we pass fixture for signature stability
    slips = []
    for q in sizes:
        res = run_trade_mode(
            ExecutionMode.AMM_ONLY,
            target_out=q,
            segments=clob_segments_default,
            amm_curve=amm_curve_default,
            amm_context=amm_ctx_multipath,
        )
        er = res.report
        assert er is not None
        assert er.total_out > 0
        slips.append(er.slippage_price_avg)

    # Non-negative and non-decreasing (within numerical tolerance)
    for s in slips:
        assert geq_with_tol(s, Decimal(0))
    for a, b in zip(slips, slips[1:]):
        assert geq_with_tol(b, a)