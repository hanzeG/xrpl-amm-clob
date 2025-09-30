"""CLOB → segments: turn price levels into homogeneous quote slices.
Includes optional issuer transfer fees (IOU-only): tr_in on IN, tr_out on OUT.
Quality is user-view: OUT_net / IN_gross.

Helpers (public):
- make_ladder(depth, top_quality, qty_per_level, decay, ...): canonical geometric ladder
- from_levels([(quality, out_max), ...], ...): explicit levels to segments
- normalise_segments(segments): snap amounts & bucket quality, sort by quality desc
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Optional, Tuple

from .core import (
    Segment,
    to_decimal,
    calc_quality,
    round_in_min,
    round_out_max,
    quality_bucket,
)


@dataclass(frozen=True)
class ClobLevel:
    """One price level: price = IN per 1 OUT; out_liquidity = max OUT."""
    price_in_per_out: Decimal
    out_liquidity: Decimal

    @staticmethod
    def from_numbers(price_in_per_out: float | str | Decimal,
                     out_liquidity: float | str | Decimal) -> "ClobLevel":
        """Build a level from plain numbers/strings."""
        return ClobLevel(
            price_in_per_out=to_decimal(price_in_per_out),
            out_liquidity=to_decimal(out_liquidity),
        )


class Clob:
    """Central Limit Order Book that emits segments for the router."""

    def __init__(
        self,
        levels: List[ClobLevel],
        *,
        in_is_xrp: bool,
        out_is_xrp: bool,
        tr_in: Decimal | str | float = Decimal("0"),
        tr_out: Decimal | str | float = Decimal("0"),
    ):
        self.levels = levels
        self.in_is_xrp = in_is_xrp
        self.out_is_xrp = out_is_xrp
        # Transfer fee rates (apply only when side is IOU)
        tri = to_decimal(tr_in)
        tro = to_decimal(tr_out)
        if tri < 0 or tri >= 1:
            raise ValueError("tr_in must satisfy 0 ≤ tr_in < 1")
        if tro < 0 or tro >= 1:
            raise ValueError("tr_out must satisfy 0 ≤ tr_out < 1")
        self.tr_in = Decimal("0") if in_is_xrp else tri
        self.tr_out = Decimal("0") if out_is_xrp else tro

    def segments(self) -> List[Segment]:
        """Convert price levels to CLOB segments (quality desc)."""
        segs: List[Segment] = []
        for lvl in self.levels:
            # Snap to amount grids and apply transfer fees.
            out_gross = round_out_max(lvl.out_liquidity, is_xrp=self.out_is_xrp)
            out_net = round_out_max(
                out_gross * (Decimal(1) - self.tr_out), is_xrp=self.out_is_xrp
            )
            if out_net <= 0:
                continue
            in_req_at_price = lvl.price_in_per_out * out_gross
            if self.tr_in > 0:
                in_req_at_price = in_req_at_price / (Decimal(1) - self.tr_in)
            in_gross = round_in_min(in_req_at_price, is_xrp=self.in_is_xrp)
            if in_gross <= 0:
                continue
            # Use quality bucket for tier grouping.
            q = quality_bucket(calc_quality(out_net, in_gross))
            if q <= 0:
                continue
            segs.append(Segment(
                src="CLOB",
                quality=q,
                out_max=out_net,
                in_at_out_max=in_gross,
                in_is_xrp=self.in_is_xrp,
                out_is_xrp=self.out_is_xrp,
            ))

        # Highest quality first.
        segs.sort(key=lambda s: s.quality, reverse=True)
        return segs


# -----------------------------
# Public helpers: unified CLOB construction & normalisation
# -----------------------------

def make_ladder(
    *,
    depth: int,
    top_quality: Decimal,
    qty_per_level: Decimal,
    decay: Decimal,
    in_is_xrp: bool = False,
    out_is_xrp: bool = False,
    tr_in: Decimal | str | float = Decimal("0"),
    tr_out: Decimal | str | float = Decimal("0"),
) -> List[Segment]:
    """Create a geometric-quality ladder and emit CLOB segments.

    Args:
        depth: number of levels (>=1)
        top_quality: quality at the best level (OUT/IN)
        qty_per_level: OUT liquidity per level (before transfer fees)
        decay: multiplicative factor (0<decay<1) applied to quality per level
        in_is_xrp/out_is_xrp: amount grids for rounding
        tr_in/tr_out: issuer transfer fees (bps in decimal; ignored if the side is XRP)

    Returns:
        List[Segment] sorted by quality desc, quality bucketed & amounts rounded.
    """
    if depth <= 0:
        return []
    if top_quality <= 0:
        return []
    if qty_per_level <= 0:
        return []
    if decay <= 0 or decay >= 1:
        raise ValueError("decay must be in (0,1)")

    levels: List[ClobLevel] = []
    q = top_quality
    for _ in range(depth):
        price = (Decimal(1) / q)
        levels.append(ClobLevel(price_in_per_out=price, out_liquidity=qty_per_level))
        q = q * decay
        if q <= 0:
            break
    clob = Clob(levels, in_is_xrp=in_is_xrp, out_is_xrp=out_is_xrp, tr_in=tr_in, tr_out=tr_out)
    return clob.segments()


def from_levels(
    levels: Iterable[Tuple[Decimal, Decimal]],
    *,
    in_is_xrp: bool = False,
    out_is_xrp: bool = False,
    tr_in: Decimal | str | float = Decimal("0"),
    tr_out: Decimal | str | float = Decimal("0"),
) -> List[Segment]:
    """Create CLOB segments from explicit (quality, out_max) pairs.

    Quality is OUT/IN (>0). We convert to price (IN/OUT) internally. Amounts and quality
    are snapped to grids/buckets via `segments()`.
    """
    lvls: List[ClobLevel] = []
    for qual, out_max in levels:
        if qual is None or out_max is None:
            continue
        if qual <= 0 or out_max <= 0:
            continue
        price = (Decimal(1) / qual)
        lvls.append(ClobLevel(price_in_per_out=price, out_liquidity=out_max))
    if not lvls:
        return []
    clob = Clob(lvls, in_is_xrp=in_is_xrp, out_is_xrp=out_is_xrp, tr_in=tr_in, tr_out=tr_out)
    return clob.segments()


def normalise_segments(segs: Iterable[Segment]) -> List[Segment]:
    """Normalise pre-built CLOB segments to canonical form.

    - Drop non-positive amounts/qualities
    - Recompute bucketed quality and rounded amounts on declared grids
    - Sort by quality desc
    """
    out: List[Segment] = []
    for s in segs:
        if s.out_max <= 0:
            continue
        if s.in_at_out_max <= 0:
            continue
        if s.quality <= 0:
            continue
        # Re-snap amounts to grids
        out_max = round_out_max(s.out_max, is_xrp=s.out_is_xrp)
        in_need = round_in_min(s.in_at_out_max, is_xrp=s.in_is_xrp)
        if out_max <= 0 or in_need <= 0:
            continue
        q = quality_bucket(calc_quality(out_max, in_need))
        if q <= 0:
            continue
        out.append(Segment(
            src="CLOB",
            quality=q,
            out_max=out_max,
            in_at_out_max=in_need,
            in_is_xrp=s.in_is_xrp,
            out_is_xrp=s.out_is_xrp,
        ))
    out.sort(key=lambda z: z.quality, reverse=True)
    return out