from decimal import Decimal

import pytest

from xrpl_router.exec_modes import run_trade_mode, ExecutionMode
from xrpl_router.efficiency_scan import hybrid_flow


# ---------------------------
# Budget constraint & limiting-step replay (whitepaper §1.3.2)
# ---------------------------


def test_send_max_caps_total_in_clob_only(clob_segments_default):
    """CLOB-only path should honour send_max and never overshoot.

    We first run a baseline to estimate cost for the same target_out, then tighten the
    budget to 60% of that cost. Expect: total_in ≤ send_max and a lower filled ratio.
    """
    target = Decimal("80")

    # Baseline without budget
    base = run_trade_mode(
        ExecutionMode.CLOB_ONLY,
        target_out=target,
        segments=clob_segments_default,
    )
    er0 = base.report
    assert er0 is not None
    assert er0.total_out > 0

    # Tight budget (60% of baseline spend)
    send_max = (er0.total_in * Decimal("0.60"))
    capped = run_trade_mode(
        ExecutionMode.CLOB_ONLY,
        target_out=target,
        segments=clob_segments_default,
        send_max=send_max,
    )
    er1 = capped.report
    assert er1 is not None

    # Budget respected
    assert er1.total_in <= send_max
    # Fill cannot exceed target; with tighter budget it should be strictly lower than baseline
    assert er1.total_out <= target
    assert er1.filled_ratio <= er0.filled_ratio
    # If budget binds, we expect a strictly lower fill (allow equality if book is very cheap)
    assert er1.total_out <= er0.total_out


def test_send_max_caps_total_in_hybrid(clob_segments_default, amm_curve_default, amm_anchor_default, amm_ctx_multipath):
    """Hybrid path should also honour send_max: never spend above send_max, and fill ratio drops.

    This exercises reverse→forward limiting-step replay across mixed (CLOB+AMM) routes.
    """
    target = Decimal("100")

    # Use a shallower CLOB to keep hybrid replay fast under tight budgets
    from .conftest import make_clob_segments
    clob_shallow = make_clob_segments(depth=4, top_quality=Decimal("1.00"), qty_per_level=Decimal("30"), decay=Decimal("0.99"))

    # Baseline without budget
    base = hybrid_flow(
        target_out=target,
        clob_segments=clob_shallow,
        amm_curve=amm_curve_default,
        amm_anchor=amm_anchor_default,
        amm_context=amm_ctx_multipath,
    )
    assert base.total_out > 0

    # Tight budget (70% of baseline spend)
    send_max = (base.total_in * Decimal("0.70"))
    capped = hybrid_flow(
        target_out=target,
        clob_segments=clob_shallow,
        amm_curve=amm_curve_default,
        amm_anchor=amm_anchor_default,
        amm_context=amm_ctx_multipath,
        send_max=send_max,
    )

    # Budget respected
    assert capped.total_in <= send_max
    # Fill cannot exceed target; with tighter budget it should be lower or equal
    assert capped.total_out <= target
    assert capped.total_out <= base.total_out