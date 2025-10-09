from decimal import Decimal

import pytest

from xrpl_router.research import run_trade_mode, ExecutionMode
from xrpl_router.research.efficiency_scan import (
    analyze_alpha_scan,
    summarize_alpha_scan,
    find_crossover_q,
    summarize_crossover,
)


# ---------------------------
# α* and q* sanity checks (whitepaper-aligned, non-exhaustive)
# ---------------------------


def _tolerance_for(price: Decimal) -> Decimal:
    """Absolute tolerance combining fixed and relative components."""
    abs_eps = Decimal("1e-8")
    rel = (price.copy_abs() * Decimal("1e-4"))
    return abs_eps if rel < abs_eps else rel


def test_alpha_star_sanity_basic(
    clob_segments_default,
    amm_curve_default,
    amm_anchor_default,
    amm_ctx_multipath,
    amm_instance_default,
):
    target = Decimal("100")

    # Run alpha scan with moderate granularity
    scan = analyze_alpha_scan(
        target_out=target,
        segments=clob_segments_default,
        amm_curve=amm_curve_default,
        amm_anchor=amm_anchor_default,
        amm_context=amm_ctx_multipath,
        step=Decimal("0.25"),
        amm_for_fees=amm_instance_default,
    )

    # 1) alpha_star in [0,1]
    assert scan.alpha_star >= 0 and scan.alpha_star <= 1

    # 2) price at alpha* should be no worse than endpoints (allow tolerance)
    row = summarize_alpha_scan(scan)
    p_star = row["avg_price"]

    # endpoints
    clob = run_trade_mode(
        ExecutionMode.CLOB_ONLY,
        target_out=target,
        segments=clob_segments_default,
    )
    amm = run_trade_mode(
        ExecutionMode.AMM_ONLY,
        target_out=target,
        segments=clob_segments_default,
        amm_curve=amm_curve_default,
        amm_context=amm_ctx_multipath,
    )
    e0 = clob.report.avg_price if clob.report is not None else Decimal("Infinity")
    e1 = amm.report.avg_price if amm.report is not None else Decimal("Infinity")

    tol = _tolerance_for(min(e0, e1))
    assert p_star <= min(e0, e1) + tol

    # 3) feasibility: some fill should occur at alpha*
    assert row["total_out"] > 0

    # 4) fee totals are non-negative (they may be zero if alpha*=0)
    assert row["fee_pool_total"] >= 0
    assert row["fee_tr_in_total"] >= 0
    assert row["fee_tr_out_total"] >= 0


def test_qstar_sanity_basic(
    clob_segments_default,
    amm_curve_default,
    amm_anchor_default,
    amm_ctx_multipath,
    amm_instance_default,
):
    # Find crossover with defaults (coarse+refine are configured via ROUTING_CFG)
    res = find_crossover_q(
        segments=clob_segments_default,
        amm_anchor=amm_anchor_default,
        amm_curve=amm_curve_default,
        amm_context=amm_ctx_multipath,
        amm_for_fees=amm_instance_default,
        coarse_steps=8, refine_iters=10,
    )

    # Either we have q* or a best_abs representative
    assert (res.q_star is not None) or (res.best_abs is not None)

    if res.q_star is not None:
        # choose the sample closest to q*
        feasible = [s for s in res.samples if s.feasible_amm and s.feasible_clob]
        cand = min(feasible or res.samples, key=lambda s: abs(s.q - res.q_star))
        # near the intersection, prices should be close within tolerance
        avg = (cand.price_amm + cand.price_clob) / 2
        tol = _tolerance_for(avg)
        assert abs(cand.price_amm - cand.price_clob) <= tol
        # feasibility
        assert cand.res_amm.report is None or cand.res_amm.report.total_out >= 0
        assert cand.res_clob.report is None or cand.res_clob.report.total_out >= 0
        # fees non-negative on AMM leg
        if cand.res_amm.report is not None:
            assert cand.res_amm.report.fee_pool_total >= 0
            assert cand.res_amm.report.fee_tr_in_total >= 0
            assert cand.res_amm.report.fee_tr_out_total >= 0
    else:
        # No explicit q*: best_abs must minimise |price_amm - price_clob|
        diffs = [abs(s.price_amm - s.price_clob) for s in res.samples]
        best = res.best_abs
        assert abs(best.price_amm - best.price_clob) == min(diffs)
        # fees non-negative on AMM leg
        if best.res_amm.report is not None:
            assert best.res_amm.report.fee_pool_total >= 0
            assert best.res_amm.report.fee_tr_in_total >= 0
            assert best.res_amm.report.fee_tr_out_total >= 0