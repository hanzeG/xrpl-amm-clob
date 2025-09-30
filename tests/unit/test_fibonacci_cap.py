

from decimal import Decimal

import pytest

from xrpl_router.efficiency_scan import hybrid_flow
from xrpl_router.amm_context import AMMContext
from xrpl_router.core import Segment

from xrpl_router.clob import make_ladder, normalise_segments
from xrpl_router.amm import amm_curve_from_linear, amm_anchor_from_discount


# ---------------------------
# Fibonacci cap & iteration bound (whitepaper §1.2.7.3)
# ---------------------------


def test_fib_not_advance_when_amm_unused():
    """If AMM is not used in a multi-path iteration, Fibonacci cap must not advance.

    Scenario: CLOB top is strictly better and has enough depth; AMM anchor/curve are worse,
    so hybrid execution fills entirely from CLOB. The AMMContext should initialise the
    Fibonacci base (lazy) but must NOT advance to the next term; also used-iters stays 0.
    """
    ctx = AMMContext(False)
    ctx.set_multi_path(True)

    # CLOB: high quality, enough to cover the target fully
    clob = normalise_segments(make_ladder(depth=3, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.99")))
    # AMM: slightly worse than book, so it won't be selected
    amm_curve = amm_curve_from_linear(base_quality=Decimal("0.98"), slope=Decimal("0.0"), seg_out=Decimal("20"))
    amm_anchor = amm_anchor_from_discount(discount=Decimal("0.98"))

    # Run once: full fill from CLOB
    res = hybrid_flow(
        target_out=Decimal("40"),
        clob_segments=clob,
        amm_curve=amm_curve,
        amm_anchor=amm_anchor,
        amm_context=ctx,
        amm_for_fees=None,
    )

    # After run: Fibonacci should be initialised (prev==curr==base), not advanced; used iterations = 0
    assert ctx._fib_inited is True
    assert ctx._fib_prev == ctx._fib_curr  # no advance
    assert ctx._amm_used_iters == 0


def test_fib_advances_when_amm_used():
    """When AMM participates, Fibonacci cap advances from base to (2*base).

    We first run a CLOB-only fill to capture the base (prev==curr). Then force AMM-only
    so at least one AMM iteration is consumed; expect curr becomes 2*base and used_iters > 0.
    """
    ctx = AMMContext(False)
    ctx.set_multi_path(True)

    # Step 1: CLOB-only behaviour to initialise base without advancing
    clob = normalise_segments(make_ladder(depth=3, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.99")))
    amm_curve_worse = amm_curve_from_linear(base_quality=Decimal("0.98"), slope=Decimal("0.0"), seg_out=Decimal("20"))
    amm_anchor_worse = amm_anchor_from_discount(discount=Decimal("0.98"))

    hybrid_flow(
        target_out=Decimal("40"),
        clob_segments=clob,
        amm_curve=amm_curve_worse,
        amm_anchor=amm_anchor_worse,
        amm_context=ctx,
    )
    base = ctx._fib_curr

    # Step 2: Force AMM-only so AMM is used and Fibonacci advances
    amm_curve = amm_curve_from_linear(base_quality=Decimal("0.99"), slope=Decimal("0.0"), seg_out=Decimal("10"))
    hybrid_flow(
        target_out=Decimal("30"),
        clob_segments=[],
        amm_curve=amm_curve,
        amm_anchor=None,
        amm_context=ctx,
    )

    assert ctx._amm_used_iters > 0
    # After first AMM participation: prev should equal base, curr should be base+base (=2*base)
    assert ctx._fib_prev == base
    assert ctx._fib_curr == base + base


def test_amm_iteration_cap_stops_at_30():
    """AMM usage count is capped (documented as 30). After reaching the cap, further
    runs should not increase the counter, and AMM contributes nothing in AMM-only mode.
    """
    ctx = AMMContext(False)
    ctx.set_multi_path(True)

    amm_curve = amm_curve_from_linear(base_quality=Decimal("0.99"), slope=Decimal("0.0"), seg_out=Decimal("5"))

    # Drive AMM participation repeatedly until we hit the cap (safety loop ≤ 200)
    loops = 0
    while ctx._amm_used_iters < 30 and loops < 200:
        hybrid_flow(
            target_out=Decimal("10"),
            clob_segments=[],
            amm_curve=amm_curve,
            amm_anchor=None,
            amm_context=ctx,
        )
        loops += 1

    assert ctx._amm_used_iters == 30

    # One more run: counter should not increase; in AMM-only mode with cap reached, expect no progress
    before = ctx._amm_used_iters
    res = hybrid_flow(
        target_out=Decimal("10"),
        clob_segments=[],
        amm_curve=amm_curve,
        amm_anchor=None,
        amm_context=ctx,
    )
    after = ctx._amm_used_iters
    assert after == before
    # If AMM is blocked, total_out should be zero in AMM-only setting (no CLOB to fill)
    assert res.total_out == 0