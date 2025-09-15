"""CLOB → segments: turn price levels into homogeneous quote slices."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List

from .core import (
    Segment,
    to_decimal,
    calc_quality,
    round_in_min,
    round_out_max,
)


@dataclass(frozen=True)
class ClobLevel:
    """One price level: price = IN per 1 OUT; out_liquidity = max OUT."""
    price_in_per_out: Decimal
    out_liquidity: Decimal

    @staticmethod
    def from_numbers(price_in_per_out: float | str | Decimal,
                     out_liquidity: float | str | Decimal) -> "ClobLevel":
        """Helper to build a level from plain numbers/strings."""
        return ClobLevel(
            price_in_per_out=to_decimal(price_in_per_out),
            out_liquidity=to_decimal(out_liquidity),
        )


class Clob:
    """Central Limit Order Book that emits segments for the router."""

    def __init__(self, levels: List[ClobLevel], *, in_is_xrp: bool, out_is_xrp: bool):
        self.levels = levels
        self.in_is_xrp = in_is_xrp
        self.out_is_xrp = out_is_xrp

    def segments(self) -> List[Segment]:
        """Convert price levels to sorted CLOB segments (quality desc)."""
        segs: List[Segment] = []
        for lvl in self.levels:
            # Amounts snapped to the correct grids.
            out_max = round_out_max(lvl.out_liquidity, is_xrp=self.out_is_xrp)
            if out_max <= 0:
                continue
            in_at_out_max = round_in_min(lvl.price_in_per_out * out_max, is_xrp=self.in_is_xrp)
            if in_at_out_max <= 0:
                continue
            q = calc_quality(out_max, in_at_out_max)
            if q <= 0:
                continue
            segs.append(Segment(src="CLOB", quality=q, out_max=out_max, in_at_out_max=in_at_out_max))

        # Highest quality first (cheapest IN per OUT).
        segs.sort(key=lambda s: s.quality, reverse=True)
        return segs