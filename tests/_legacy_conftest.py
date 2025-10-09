from __future__ import annotations
from decimal import Decimal
from typing import Callable, Iterable, List, Optional, Tuple

import pytest

# Import project primitives
from xrpl_router.core import Segment
from xrpl_router.amm_context import AMMContext
from xrpl_router.clob import make_ladder, normalise_segments
from xrpl_router.amm import AMM, amm_curve_from_linear, amm_anchor_from_discount


# -----------------------------
# Test helpers (pure functions)
# -----------------------------

class FakeAMM:
    """Minimal AMM stub exposing preview_fees_for_fill for fee-metering tests.

    - pool_fee_bps: pool fee in basis points (e.g., 30 = 0.30%).
    - tr_in_bps / tr_out_bps: issuer transfer fees on input/output legs in bps.
    """

    def __init__(self, pool_fee_bps: int = 30, tr_in_bps: int = 0, tr_out_bps: int = 0) -> None:
        self.pool_fee = Decimal(pool_fee_bps) / Decimal(10000)
        self.tr_in = Decimal(tr_in_bps) / Decimal(10000)
        self.tr_out = Decimal(tr_out_bps) / Decimal(10000)

    # Match the signature expected by efficiency_scan wrappers
    def preview_fees_for_fill(self, dx_gross, dy_net):
        dx = Decimal(dx_gross).copy_abs()
        dy = Decimal(dy_net).copy_abs()
        fee_tr_in = dx * self.tr_in if dx > 0 else Decimal(0)
        dx_after_tf = dx - fee_tr_in
        fee_pool = dx_after_tf * self.pool_fee if dx_after_tf > 0 else Decimal(0)
        # For dy, compute gross needed so user receives dy (simulate issuer fee on output)
        fee_tr_out = dy * self.tr_out if dy > 0 else Decimal(0)
        return (fee_pool, fee_tr_in, fee_tr_out)


# -----------------------------
# Pytest fixtures
# -----------------------------

@pytest.fixture()
def amm_ctx_multipath() -> AMMContext:
    ctx = AMMContext(False)
    ctx.setMultiPath(True)
    return ctx


@pytest.fixture()
def clob_segments_default() -> List[Segment]:
    return normalise_segments(make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985")))


@pytest.fixture()
def amm_curve_default() -> Callable[[Decimal], Iterable[Segment]]:
    return amm_curve_from_linear(base_quality=Decimal("0.99"), slope=Decimal("0.00005"), seg_out=Decimal("20"))


@pytest.fixture()
def amm_anchor_default() -> Callable[[Decimal, Decimal], Optional[Segment]]:
    return amm_anchor_from_discount(discount=Decimal("0.995"))


@pytest.fixture()
def amm_instance_default() -> FakeAMM:
    return FakeAMM(pool_fee_bps=30, tr_in_bps=0, tr_out_bps=20)


@pytest.fixture()
def amm_instance_real_factory():
    def factory(
        x: Decimal = Decimal("1000"),
        y: Decimal = Decimal("2000"),
        fee: Decimal = Decimal("0.003"),
        x_is_xrp: bool = True,
        y_is_xrp: bool = False,
        tr_in: Decimal = Decimal("0"),
        tr_out: Decimal = Decimal("0"),
    ):
        return AMM(x_reserve=x, y_reserve=y, fee=fee, x_is_xrp=x_is_xrp, y_is_xrp=y_is_xrp, tr_in=tr_in, tr_out=tr_out)
    return factory


@pytest.fixture()
def amm_curve_real(amm_instance_real_factory) -> Callable[[Decimal], Iterable[Segment]]:
    def provider(target_out: Decimal) -> Iterable[Segment]:
        amm = amm_instance_real_factory()
        segs = list(amm.segments_for_out(target_out, max_segments=30, start_fraction=Decimal("5e-2")))
        if not segs:
            q0 = amm.spq()
            # Use a slightly worse-than-SPQ threshold to guarantee feasibility
            for factor in (Decimal("0.999"), Decimal("0.99"), Decimal("0.95")):
                anchor = amm.synthetic_segment_for_quality(q0 * factor, max_out_cap=target_out)
                if anchor is not None and anchor.out_max > 0:
                    return [anchor]
            # Robust fallback: try to construct a single segment from a conservative AMM swap
            dy = min(target_out, amm.y * Decimal("0.10"))
            dx = amm.swap_in_given_out(dy)
            if dx > 0 and dy > 0:
                seg = Segment(
                    src="AMM",
                    quality=(dy / dx),
                    out_max=dy,
                    in_at_out_max=dx,
                    in_is_xrp=amm.x_is_xrp,
                    out_is_xrp=amm.y_is_xrp,
                )
                return [seg]
            return []
        return segs
    return provider


@pytest.fixture()
def amm_anchor_real(amm_instance_real_factory) -> Callable[[Decimal, Decimal], Optional[Segment]]:
    def anchor(q_threshold: Decimal, max_out_cap: Decimal) -> Optional[Segment]:
        amm = amm_instance_real_factory()
        return amm.synthetic_segment_for_quality(q_threshold, max_out_cap=max_out_cap)
    return anchor
