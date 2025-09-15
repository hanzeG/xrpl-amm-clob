"""AMM base swaps (constant product, fee on input).

Rules adopted (impact routing results):
- Fee is a pool parameter (≤1%) and is deducted on the input side.
- Given dx→dy: dx_eff = dx*(1-fee); dy = y*dx_eff/(x+dx_eff); dy is floored to amount grid.
- Given dy→dx: dx_eff = dy*x/(y-dy); dx = dx_eff/(1-fee); dx is ceiled to amount grid.
- OUT/IN quality is computed conservatively (quantised down) elsewhere when needed.
- Reject dy ≥ y (cannot drain the pool side).
"""
from __future__ import annotations

from decimal import Decimal
from typing import List

from .core import (
    to_decimal,
    clamp_nonneg,
    quantize_up,
    round_in_min,
    round_out_max,
    quantize_quality,
    XRP_QUANTUM,
    IOU_QUANTUM,
    Segment,
    calc_quality,
)


class AMM:
    """AMM(X,Y) with constant product math and input-side fee.

    Orientation: inputs are X, outputs are Y for these methods.
    Amount grids: X uses drops if x_is_xrp, else 1e-15; Y likewise.
    """

    def __init__(self,
                 x_reserve: Decimal | str | float,
                 y_reserve: Decimal | str | float,
                 fee: Decimal | str | float,
                 *,
                 x_is_xrp: bool,
                 y_is_xrp: bool) -> None:
        self.x = to_decimal(x_reserve)
        self.y = to_decimal(y_reserve)
        self.fee = to_decimal(fee)
        self.x_is_xrp = x_is_xrp
        self.y_is_xrp = y_is_xrp

    # --- Diagnostics ---
    def spq(self) -> Decimal:
        """Small-trade quality ≈ (y/x)*(1-fee), quantised to quality grid."""
        if self.x <= 0 or self.y <= 0:
            return Decimal(0)
        return quantize_quality((self.y / self.x) * (Decimal(1) - self.fee))

    # --- Swaps ---
    def swap_out_given_in(self, dx: Decimal | str | float) -> Decimal:
        """Max OUT for a given IN budget (floored to OUT grid)."""
        dx = round_out_max(clamp_nonneg(to_decimal(dx)), is_xrp=self.x_is_xrp)
        if dx <= 0 or self.x <= 0 or self.y <= 0:
            return Decimal(0)
        dx_eff = dx * (Decimal(1) - self.fee)
        dy_theory = (self.y * dx_eff) / (self.x + dx_eff)
        dy = round_out_max(dy_theory, is_xrp=self.y_is_xrp)
        if dy < 0:
            return Decimal(0)
        return dy

    def swap_in_given_out(self, dy: Decimal | str | float) -> Decimal:
        """Min IN to obtain target OUT (ceiled to IN grid); rejects dy ≥ y."""
        # Round requested OUT up to ensure deliverable and conservative input calc.
        y_quant = XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM
        dy_req = quantize_up(clamp_nonneg(to_decimal(dy)), y_quant)
        if dy_req <= 0:
            return Decimal(0)
        if dy_req >= self.y:
            raise ValueError("requested OUT >= pool reserve")
        dx_eff = (dy_req * self.x) / (self.y - dy_req)
        dx = dx_eff / (Decimal(1) - self.fee)
        # Ceil to input grid so we do not underfund the swap.
        return round_in_min(dx, is_xrp=self.x_is_xrp)

    # --- Whitepaper-style synthetic quote (anchor to target quality) ---
    def synthetic_segment_for_quality(self,
                                      q_threshold: Decimal | str | float,
                                      *,
                                      max_out_cap: Decimal | str | float | None = None
                                      ) -> Segment | None:
        """Produce a single AMM slice whose average quality ~= q_threshold.

        q_threshold is typically the LOB top-tier quality for the current iteration.
        We solve for dy so that dy/dx ≈ q_threshold:
            q_slice = (1 - fee) * (y - dy) / x  =>  dy_anchor = y - (q*x)/(1-fee)
        Then floor dy to OUT grid, ceil dx to IN grid.
        """
        if self.x <= 0 or self.y <= 0:
            return None
        q = clamp_nonneg(to_decimal(q_threshold))
        if q <= 0:
            return None

        one_minus_fee = (Decimal(1) - self.fee)
        if one_minus_fee <= 0:
            return None

        # Theoretical dy that anchors average quality to q.
        dy_theory = self.y - (q * self.x) / one_minus_fee

        # Apply optional cap and floor to amount grid to avoid overpaying.
        y_quant = XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM
        if max_out_cap is not None:
            cap = clamp_nonneg(to_decimal(max_out_cap))
            if cap > 0 and dy_theory > cap:
                dy_theory = cap

        dy = round_out_max(dy_theory, is_xrp=self.y_is_xrp)
        if dy <= 0 or dy >= self.y:
            return None

        # Compute required IN (ceil to grid inside).
        dx = self.swap_in_given_out(dy)
        if dx <= 0:
            return None

        q_slice = calc_quality(dy, dx)
        if q_slice <= 0:
            return None

        return Segment(src="AMM", quality=q_slice, out_max=dy, in_at_out_max=dx)

    # --- Segmentation ---
    def segments_for_out(self,
                         target_out: Decimal | str | float,
                         *,
                         max_segments: int = 30,
                         start_fraction: Decimal = Decimal("1e-4")) -> List[Segment]:
        """Discretise AMM curve into fixed-quality slices up to target OUT.

        Uses Fibonacci growth for slice sizes to cover scale efficiently.
        Each slice records deliverable OUT (quantised up) and required IN (ceiled),
        and its quality is computed conservatively.
        """
        # Normalise inputs
        remaining = clamp_nonneg(to_decimal(target_out))
        if remaining <= 0 or self.y <= 0 or self.x <= 0:
            return []

        y_quant = XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM

        # Base slice size, ensure at least one quantum
        base = self.y * start_fraction
        if base <= 0:
            return []
        base = quantize_up(base, y_quant)
        if base <= 0:
            base = y_quant

        segs: List[Segment] = []
        # Fibonacci sequence for scaling slices
        f_prev, f_curr = 1, 1

        while remaining > 0 and len(segs) < max_segments:
            # Proposed slice before grid quantisation
            proposed = base * f_curr
            # Do not exceed remaining target
            proposed = proposed if proposed <= remaining else remaining

            # Quantise requested OUT upward to ensure deliverable on amount grid
            out_req = quantize_up(proposed, y_quant)
            if out_req <= 0:
                break
            # Guard pool boundary
            if out_req >= self.y:
                break

            # Compute required IN (already ceiled inside)
            in_need = self.swap_in_given_out(out_req)
            if in_need <= 0:
                break

            q = calc_quality(out_req, in_need)
            if q <= 0:
                break

            segs.append(Segment(src="AMM", quality=q, out_max=out_req, in_at_out_max=in_need))

            # Update remaining and fibonacci step
            remaining = remaining - out_req
            if remaining <= 0:
                break
            f_prev, f_curr = f_curr, f_prev + f_curr

        return segs