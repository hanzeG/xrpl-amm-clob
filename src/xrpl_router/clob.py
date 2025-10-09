"""CLOB → segments: turn price levels into homogeneous quote slices.
Issuer transfer fees (IOU-only) are supported via tr_in (on IN) and tr_out (on OUT),
but **OUT-side fees are not deducted** at segment construction time (taker-view OUT).
Quality is taker-view: OUT / IN_gross. OUT-side issuer fee is accounted during
execution (ownerGives), per whitepaper §1.3.2.4.

Helpers (public):
- make_ladder(depth, top_quality, qty_per_level, decay, ...): canonical geometric ladder
- from_levels([(quality, out_max), ...], ...): explicit levels to segments
- normalise_segments(segments): integer-domain filter & sort by quality desc

All functions return taker-view segments; issuer OUT-side fees are deferred to execution-time (BookStep).
"""

# NOTE:
#   This module performs Decimal-based rounding only at the I/O boundary (price and liquidity inputs).
#   All subsequent computations, sorting, and quality comparisons are performed in the integer domain
#   via STAmount and Quality. This matches XRPL's ledger semantics where all amounts are integer-scaled.

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List

from .core.datatypes import Segment
from .core.amounts import STAmount, round_in_min, round_out_max
from .core.quality import Quality

# Local Decimal helpers (I/O boundary only)
DecimalLike = Decimal | int | str

def to_decimal(x: DecimalLike) -> Decimal:
    # Convert a numeric-like value to Decimal for I/O boundary calculations
    return x if isinstance(x, Decimal) else Decimal(str(x))


@dataclass(frozen=True)
class ClobLevel:
    """One price level: price = IN per 1 OUT; out_liquidity = max OUT."""
    price_in_per_out: Decimal
    out_liquidity: Decimal

    @staticmethod
    def from_numbers(price_in_per_out: DecimalLike,
                     out_liquidity: DecimalLike) -> "ClobLevel":
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
        tr_in: DecimalLike = Decimal("0"),
        tr_out: DecimalLike = Decimal("0"),
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
            if out_gross <= 0:
                continue
            gross_in_needed = lvl.price_in_per_out * out_gross
            if self.tr_in > 0:
                gross_in_needed = gross_in_needed / (Decimal(1) - self.tr_in)
            in_gross = round_in_min(gross_in_needed, is_xrp=self.in_is_xrp)
            if in_gross <= 0:
                continue
            # Convert to integer-domain amounts.
            out_st = STAmount.from_decimal(out_gross)
            in_st = STAmount.from_decimal(in_gross)
            # Compute integer-domain quality and validate positivity.
            q = Quality.from_amounts(offer_out=out_st, offer_in=in_st)
            if q.rate.sign <= 0:
                continue
            segs.append(Segment(
                src="CLOB",
                quality=q,
                out_max=out_st,
                in_at_out_max=in_st,
                in_is_xrp=self.in_is_xrp,
                out_is_xrp=self.out_is_xrp,
                raw_quality=q,
                source_id=None,
            ))

        # Sort by integer-domain quality (highest first).
        segs.sort(key=lambda s: (s.quality.rate.exponent, s.quality.rate.mantissa), reverse=True)
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
    tr_in: DecimalLike = Decimal("0"),
    tr_out: DecimalLike = Decimal("0"),
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
        List[Segment] sorted by quality desc; amounts are rounded to grids.

    All emitted segments are taker-view; OUT-side issuer fees are applied later during execution.
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
    levels: Iterable[tuple[Decimal, Decimal]],
    *,
    in_is_xrp: bool = False,
    out_is_xrp: bool = False,
    tr_in: DecimalLike = Decimal("0"),
    tr_out: DecimalLike = Decimal("0"),
) -> List[Segment]:
    """Create CLOB segments from explicit (quality, out_max) pairs.

    Quality is OUT/IN (>0). We convert to price (IN/OUT) internally. Amounts and quality
    are snapped to grids/buckets via `segments()`.

    All emitted segments are taker-view; OUT-side issuer fees are applied later during execution.
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
    """Normalize pre-built CLOB segments (integer domain).

    This function operates purely in the integer domain:
    - Drop non-positive amounts/qualities
    - Preserve amounts as given (assumed already snapped to appropriate grids)
    - Sort by quality (higher is better)
    """
    out: List[Segment] = []
    for s in segs:
        if s.out_max.is_zero():
            continue
        if s.in_at_out_max.is_zero():
            continue
        q = s.quality if s.quality.rate.sign > 0 else s.implied_quality()
        if q.rate.sign <= 0:
            continue
        out.append(Segment(
            src=s.src,
            quality=q,
            out_max=s.out_max,
            in_at_out_max=s.in_at_out_max,
            in_is_xrp=s.in_is_xrp,
            out_is_xrp=s.out_is_xrp,
            raw_quality=s.raw_quality or q,
            source_id=getattr(s, "source_id", None),
        ))
    # Sort by numeric rate: (exponent, mantissa) descending
    out.sort(key=lambda z: (z.quality.rate.exponent, z.quality.rate.mantissa), reverse=True)
    return out