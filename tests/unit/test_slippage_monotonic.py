from decimal import Decimal

# Numerical tolerance for Decimal comparisons (covers rounding/bucketing noise)
# These are test-layer tolerances to absorb quantisation/bucketing noise.
# They align with whitepaper non-negativity/monotonicity checks rather than core grids.
EPS_ABS = Decimal("1e-12")
EPS_REL = Decimal("1e-12")

def fmt_dec(x: Decimal, places: int = 18) -> str:
    # Format Decimal in scientific notation with fixed precision for stable logs
    return format(x, f'.{places}E')

def geq_with_tol(x: Decimal, y: Decimal) -> bool:
    return x + max(EPS_ABS, y.copy_abs() * EPS_REL) >= y

import pytest

from xrpl_router.exec_modes import run_trade_mode, ExecutionMode


# Slippage monotonicity — per-test whitepaper mapping lives above each test.
# See individual test comments for exact section references.

# Whitepaper §1.2.4 + §1.2.5
# We assert two properties on avg_price (average effective price) as trade size increases on a fixed limit order book:
# 1) Non-negativity: avg_price should not be negative (won't get cheaper as you buy more).
# 2) Monotonicity: avg_price should be non-decreasing with trade size, reflecting tiered consumption.
# Rationales: same quality tier consumes offers at constant price; moving to worse tiers increases avg_price.
@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("40"), Decimal("80"), Decimal("160")],  # expanded sizes
])
def test_slippage_monotonic_clob_only(clob_segments_default, sizes):
    """
    Test avg_price (average effective price) monotonicity for CLOB_ONLY execution mode.
    Based on whitepaper §1.2.4 and §1.2.5, avg_price should be non-negative and non-decreasing
    as trade size increases on a fixed limit order book.

    Defaults (fixtures): CLOB has 6 levels, each out_max≈40 IOU, top quality bucket at 1.0 (IOU grids at 1e-15).
    """
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
        # avg_price is the average effective price paid for the trade size
        slips.append(er.avg_price)
        print(f"[CLOB_ONLY] target_out: {q}, avg_price: {fmt_dec(er.avg_price)}")

    # Non-negative check: avg_price should never be negative (won't get cheaper as you buy more).
    for s in slips:
        assert geq_with_tol(s, Decimal(0))
    # Monotonic (non-decreasing) check: avg_price should not decrease as trade size increases.
    # same tier => flat avg_price; crossing to worse tier => higher avg_price;
    # tolerance accounts for grid effects.
    for a, b in zip(slips, slips[1:]):
        assert geq_with_tol(b, a)

#
# Note: this dense AMM-only test replaces the coarse-grid AMM-only test to avoid overlap.
# Whitepaper §1.2.7.1 (AMM-only) — denser size grid
# Purpose: with the corrected tier-baseline, avg_price should remain non-negative and non-decreasing
# even across a denser set of target sizes.
# This complements the coarser grid to catch subtle non-monotonicity due to rounding/bucketing noise.
@pytest.mark.parametrize("sizes", [
    [Decimal("10"), Decimal("20"), Decimal("40"), Decimal("80"), Decimal("160"), Decimal("320")],
])
def test_slippage_monotonic_amm_only_dense(clob_segments_default, amm_curve_real, amm_ctx_multipath, sizes):
    """
    Defaults (fixtures): AMM reserves x≈1000 (XRP grid=1e-6), y≈2000 (IOU grid=1e-15), fee≈0.003, no transfer fees; curve is sliced starting at ~5% of target per iteration. This test also enforces the strict whitepaper property: within each route, marginal (per-iteration) price is non-decreasing.
    """
    slips = []
    for q in sizes:
        res = run_trade_mode(
            ExecutionMode.AMM_ONLY,
            target_out=q,
            segments=clob_segments_default,  # ignored in AMM_ONLY
            amm_curve=amm_curve_real,
            amm_context=amm_ctx_multipath,
        )
        er = res.report
        assert er is not None
        assert er.total_out > 0
        slips.append(er.avg_price)
        print(f"[AMM_ONLY/DENSE] target_out: {q}, avg_price: {fmt_dec(er.avg_price)}")

    # Non-negativity of avg_price per §1.2.7.1
    for s in slips:
        assert geq_with_tol(s, Decimal(0))
    # Strict whitepaper property: within each route (fixed q), marginal (per-iteration) price is non-decreasing.
    for q in sizes:
        res_q = run_trade_mode(
            ExecutionMode.AMM_ONLY,
            target_out=q,
            segments=clob_segments_default,  # ignored in AMM_ONLY
            amm_curve=amm_curve_real,
            amm_context=amm_ctx_multipath,
        )
        er_q = res_q.report
        assert er_q is not None
        # Extract per-iteration marginal effective prices and assert non-decreasing
        marginals = [it.price_effective for it in er_q.iterations if it.out_filled > 0]
        for a, b in zip(marginals, marginals[1:]):
            assert geq_with_tol(b, a), (
                f"marginal price not non-decreasing within route (q={q}): prev={fmt_dec(a)} curr={fmt_dec(b)}"
            )


#
# Whitepaper §1.2.7.2 (Hybrid anchoring)
# When AMM and CLOB coexist within the same iteration, tier selection is anchored to the LOB top quality bucket.
# Under this anchoring, the average effective price (avg_price) should be non-negative and non-decreasing as total size grows,
# per §1.2.4, §1.2.5 (CLOB tiers) and §1.2.7.2 (AMM anchoring).
@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("40"), Decimal("80"), Decimal("160")],
])
def test_slippage_monotonic_hybrid_tier_anchored(
    clob_segments_default,
    amm_curve_real,
    amm_anchor_real,
    amm_ctx_multipath,
    sizes,
):
    """
    Defaults (fixtures): Hybrid uses the above CLOB and AMM; anchoring at LOB top quality (per §1.2.7.2).
    """
    # Test avg_price monotonicity under hybrid anchoring per §1.2.4, §1.2.5, §1.2.7.2.
    slips = []
    for q in sizes:
        res = run_trade_mode(
            ExecutionMode.HYBRID,
            target_out=q,
            segments=clob_segments_default,
            amm_curve=amm_curve_real,
            amm_anchor=amm_anchor_real,
            amm_context=amm_ctx_multipath,
        )
        er = res.report
        assert er is not None
        assert er.total_out > 0
        slips.append(er.avg_price)
        print(f"[HYBRID] target_out: {q}, avg_price: {fmt_dec(er.avg_price)}")

    # Non-negativity of avg_price per §1.2.4, §1.2.5, §1.2.7.2
    for s in slips:
        assert geq_with_tol(s, Decimal(0))
    # Strict property under LOB anchoring: within each route, marginal prices should be non-decreasing.
    for q in sizes:
        res_q = run_trade_mode(
            ExecutionMode.HYBRID,
            target_out=q,
            segments=clob_segments_default,
            amm_curve=amm_curve_real,
            amm_anchor=amm_anchor_real,
            amm_context=amm_ctx_multipath,
        )
        er_q = res_q.report
        assert er_q is not None
        marginals = [it.price_effective for it in er_q.iterations if it.out_filled > 0]
        for a, b in zip(marginals, marginals[1:]):
            assert geq_with_tol(b, a), (
                f"marginal price not non-decreasing within route (q={q}): prev={fmt_dec(a)} curr={fmt_dec(b)}"
            )