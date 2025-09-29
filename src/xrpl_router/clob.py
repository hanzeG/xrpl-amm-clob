"""CLOB → segments: turn price levels into homogeneous quote slices.
Includes optional issuer transfer fees (IOU-only): tr_in on IN, tr_out on OUT.
Quality is user-view: OUT_net / IN_gross.
"""
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