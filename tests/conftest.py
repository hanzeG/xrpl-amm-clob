import math
from decimal import Decimal
from typing import Callable, Iterable, List, Optional, Tuple

import pytest

# Import project primitives
from xrpl_router.core import Segment, IOU_QUANTUM, round_in_min
from xrpl_router.amm_context import AMMContext


# -----------------------------
# Test helpers (pure functions)
# -----------------------------

def make_clob_segments(
    *,
    depth: int = 5,
    top_quality: Decimal = Decimal("1.00"),
    qty_per_level: Decimal = Decimal("50"),
    decay: Decimal = Decimal("0.98"),
) -> List[Segment]:
    """Create a synthetic CLOB ladder with monotonically degrading quality.

    quality = OUT/IN (higher is better / cheaper). Each level reduces quality by `decay` factor.
    Each level provides `qty_per_level` OUT units.
    """
    segs: List[Segment] = []
    q = top_quality
    for _ in range(depth):
        out_max = qty_per_level
        # IN needed = OUT / quality
        in_need = (out_max / q) if q > 0 else Decimal("Infinity")
        in_need = round_in_min(in_need, is_xrp=False)
        segs.append(
            Segment(
                src="CLOB",
                quality=q,
                out_max=out_max,
                in_at_out_max=in_need,
                in_is_xrp=False,
                out_is_xrp=False,
            )
        )
        q = (q * decay)
        if q <= 0:
            break
    return segs


def amm_curve_stub_factory(
    *,
    base_quality: Decimal = Decimal("0.99"),
    slope: Decimal = Decimal("0.0001"),
    seg_out: Decimal = Decimal("25"),
) -> Callable[[Decimal], Iterable[Segment]]:
    """Return an `amm_curve(target_out)` function producing one or more AMM segments.

    The instantaneous quality starts around `base_quality` and degrades slightly with
    requested OUT via a simple linear model (only for testing determinism).
    """
    def amm_curve(target_out: Decimal) -> Iterable[Segment]:
        # Degrade quality mildly with target size; keep positive
        q_inst = base_quality - (slope * max(target_out, Decimal(0)))
        if q_inst <= Decimal("0.000001"):
            return []
        # Provide a few fixed-size slices so router can iterate
        remain = target_out
        segs: List[Segment] = []
        while remain > 0:
            take = min(seg_out, remain)
            in_need = round_in_min(take / q_inst, is_xrp=False)
            segs.append(
                Segment(
                    src="AMM",
                    quality=q_inst,
                    out_max=take,
                    in_at_out_max=in_need,
                    in_is_xrp=False,
                    out_is_xrp=False,
                )
            )
            remain -= take
        return segs

    return amm_curve


def amm_anchor_stub_factory(
    *,
    discount: Decimal = Decimal("0.995"),
) -> Callable[[Decimal, Decimal], Optional[Segment]]:
    """Return an `amm_anchor(q_lob_top, need)` that proposes a single AMM slice
    anchored to the current CLOB top quality with a small discount (worse than top).
    """
    def amm_anchor(q_lob_top: Decimal, need: Decimal) -> Optional[Segment]:
        if q_lob_top <= 0 or need <= 0:
            return None
        q = q_lob_top * discount  # slightly worse than the book top
        take = min(need, Decimal("50"))
        in_need = round_in_min(take / q, is_xrp=False)
        return Segment(
            src="AMM",
            quality=q,
            out_max=take,
            in_at_out_max=in_need,
            in_is_xrp=False,
            out_is_xrp=False,
        )

    return amm_anchor


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
    ctx.set_multi_path(True)
    return ctx


@pytest.fixture()
def clob_segments_default() -> List[Segment]:
    return make_clob_segments(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"))


@pytest.fixture()
def amm_curve_default() -> Callable[[Decimal], Iterable[Segment]]:
    return amm_curve_stub_factory(base_quality=Decimal("0.99"), slope=Decimal("0.00005"), seg_out=Decimal("20"))


@pytest.fixture()
def amm_anchor_default() -> Callable[[Decimal, Decimal], Optional[Segment]]:
    return amm_anchor_stub_factory(discount=Decimal("0.995"))


@pytest.fixture()
def amm_instance_default() -> FakeAMM:
    return FakeAMM(pool_fee_bps=30, tr_in_bps=0, tr_out_bps=20)
