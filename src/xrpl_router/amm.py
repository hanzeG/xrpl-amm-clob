"""AMM base swaps (constant product, fee on input).

Rules adopted (impact routing results):
- Fee is a pool parameter (≤1%) and is deducted on the input side.
- Given dx→dy: dx_eff = dx*(1-fee); dy = y*dx_eff/(x+dx_eff); dy is floored to amount grid.
- Given dy→dx: dx_eff = dy*x/(y-dy); dx = dx_eff/(1-fee); dx is ceiled to amount grid.
- OUT/IN quality is computed conservatively (quantised down) elsewhere when needed.
- Reject dy ≥ y (cannot drain the pool side).
- Optional issuer transfer fees (IOU-only): tr_in on X (payer side), tr_out on Y (receiver side).
  User quality is defined as OUT_net / IN_gross.
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
                 y_is_xrp: bool,
                 tr_in: Decimal | str | float = Decimal("0"),
                 tr_out: Decimal | str | float = Decimal("0")) -> None:
        self.x = to_decimal(x_reserve)
        self.y = to_decimal(y_reserve)
        self.fee = to_decimal(fee)
        self.x_is_xrp = x_is_xrp
        self.y_is_xrp = y_is_xrp
        # Clamp pool fee and transfer fees
        if self.fee < 0 or self.fee >= 1:
            raise ValueError("fee must satisfy 0 ≤ fee < 1")
        tri = to_decimal(tr_in)
        tro = to_decimal(tr_out)
        if tri < 0 or tri >= 1:
            raise ValueError("tr_in must satisfy 0 ≤ tr_in < 1")
        if tro < 0 or tro >= 1:
            raise ValueError("tr_out must satisfy 0 ≤ tr_out < 1")
        # Transfer fee applies only to IOU sides
        self.tr_in = Decimal("0") if x_is_xrp else tri
        self.tr_out = Decimal("0") if y_is_xrp else tro

    # --- Diagnostics ---
    def spq(self) -> Decimal:
        """Small-trade quality ≈ (y/x)*(1-fee), quantised to quality grid."""
        if self.x <= 0 or self.y <= 0:
            return Decimal(0)
        return quantize_quality((self.y / self.x) * (Decimal(1) - self.fee))

    # --- Swaps ---
    def swap_out_given_in(self, dx: Decimal | str | float) -> Decimal:
        """Max OUT for a given IN budget (floored to OUT grid)."""
        dx_gross = round_out_max(clamp_nonneg(to_decimal(dx)), is_xrp=self.x_is_xrp)
        if dx_gross <= 0 or self.x <= 0 or self.y <= 0:
            return Decimal(0)
        # Apply issuer transfer fee on input (pool receives less if IOU)
        dx_after_tf = dx_gross * (Decimal(1) - self.tr_in)
        # Apply pool fee on the input side (effective for curve)
        dx_eff = dx_after_tf * (Decimal(1) - self.fee)
        if dx_eff <= 0:
            return Decimal(0)
        # Curve output (gross out from pool before receiver-side transfer fee)
        dy_gross_theory = (self.y * dx_eff) / (self.x + dx_eff)
        dy_gross = round_out_max(dy_gross_theory, is_xrp=self.y_is_xrp)
        if dy_gross <= 0:
            return Decimal(0)
        # Receiver-side issuer transfer fee (user receives less if IOU)
        dy_net = dy_gross * (Decimal(1) - self.tr_out)
        dy_net = round_out_max(dy_net, is_xrp=self.y_is_xrp)
        if dy_net < 0:
            return Decimal(0)
        return dy_net

    def swap_in_given_out(self, dy: Decimal | str | float) -> Decimal:
        """Min IN to obtain target OUT (ceiled to IN grid); rejects dy ≥ y."""
        # dy is net-to-user OUT; quantise on OUT grid up (conservative)
        y_quant = XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM
        dy_net = quantize_up(clamp_nonneg(to_decimal(dy)), y_quant)
        if dy_net <= 0:
            return Decimal(0)
        # Convert to gross OUT that must leave the pool (issuer fee on receiver side)
        if self.tr_out > 0:
            one_minus_tr_out = (Decimal(1) - self.tr_out)
            if one_minus_tr_out <= 0:
                return Decimal(0)
            dy_gross = quantize_up(dy_net / one_minus_tr_out, y_quant)
        else:
            dy_gross = dy_net
        if dy_gross >= self.y:
            raise ValueError("requested OUT >= pool reserve (gross)")
        # Effective input needed at curve level
        dx_eff = (dy_gross * self.x) / (self.y - dy_gross)
        # Undo pool fee (input-side)
        one_minus_fee = (Decimal(1) - self.fee)
        if one_minus_fee <= 0:
            return Decimal(0)
        dx_after_tf = dx_eff / one_minus_fee
        # Undo issuer transfer fee on input (payer side): compute gross IN user must provide
        if self.tr_in > 0:
            one_minus_tr_in = (Decimal(1) - self.tr_in)
            if one_minus_tr_in <= 0:
                return Decimal(0)
            dx_gross = dx_after_tf / one_minus_tr_in
        else:
            dx_gross = dx_after_tf
        return round_in_min(dx_gross, is_xrp=self.x_is_xrp)

    # --- Whitepaper-style synthetic quote (anchor to target quality) ---
    def synthetic_segment_for_quality(self,
                                      q_threshold: Decimal | str | float,
                                      *,
                                      max_out_cap: Decimal | str | float | None = None
                                      ) -> Segment | None:
        """Produce a single AMM slice whose average quality ~= q_threshold."""
        if self.x <= 0 or self.y <= 0:
            return None
        q = clamp_nonneg(to_decimal(q_threshold))
        if q <= 0:
            return None

        one_minus_fee = (Decimal(1) - self.fee)
        if one_minus_fee <= 0:
            return None

        dy_theory = self.y - (q * self.x) / one_minus_fee

        y_quant = XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM
        if max_out_cap is not None:
            cap = clamp_nonneg(to_decimal(max_out_cap))
            if cap > 0 and dy_theory > cap:
                dy_theory = cap

        dy = round_out_max(dy_theory, is_xrp=self.y_is_xrp)
        if dy <= 0 or dy >= self.y:
            return None

        # Anchor average quality to q (user view: OUT_net / IN_gross)
        dy_net = dy
        # First guess gross IN from anchored quality
        dx_gross = round_in_min(dy_net / q, is_xrp=self.x_is_xrp)
        if dx_gross <= 0:
            return None
        # Check feasibility against full fee model; if not enough, reduce dy_net
        dy_check = self.swap_out_given_in(dx_gross)
        if dy_check <= 0:
            return None
        if dy_check < dy_net:
            dy_net = dy_check
            dx_gross = round_in_min(dy_net / q, is_xrp=self.x_is_xrp)
            if dx_gross <= 0:
                return None
        q_slice = calc_quality(dy_net, dx_gross)
        if q_slice <= 0:
            return None

        return Segment(
            src="AMM",
            quality=q_slice,
            out_max=dy_net,
            in_at_out_max=dx_gross,
            in_is_xrp=self.x_is_xrp,
            out_is_xrp=self.y_is_xrp,
        )

    # --- Segmentation (standalone; not used by anchored routing) ---
    def segments_for_out(self,
                         target_out: Decimal | str | float,
                         *,
                         max_segments: int = 30,
                         start_fraction: Decimal = Decimal("1e-4")) -> List[Segment]:
        """Discretise AMM curve into fixed-quality slices up to target OUT."""
        remaining = clamp_nonneg(to_decimal(target_out))
        if remaining <= 0 or self.y <= 0 or self.x <= 0:
            return []

        y_quant = XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM
        base = self.y * start_fraction
        if base <= 0:
            return []
        base = quantize_up(base, y_quant)
        if base <= 0:
            base = y_quant

        segs: List[Segment] = []
        f_prev, f_curr = 1, 1

        while remaining > 0 and len(segs) < max_segments:
            proposed = base * f_curr
            proposed = proposed if proposed <= remaining else remaining

            out_req = quantize_up(proposed, y_quant)
            if out_req <= 0:
                break
            if out_req >= self.y:
                break

            in_need = self.swap_in_given_out(out_req)
            if in_need <= 0:
                break

            q = calc_quality(out_req, in_need)
            if q <= 0:
                break

            segs.append(Segment(
                src="AMM",
                quality=q,
                out_max=out_req,
                in_at_out_max=in_need,
                in_is_xrp=self.x_is_xrp,
                out_is_xrp=self.y_is_xrp,
            ))

            remaining = remaining - out_req
            if remaining <= 0:
                break
            f_prev, f_curr = f_curr, f_prev + f_curr

        return segs

    # --- State update after a filled amount (for next-iteration anchoring) ---
    def apply_fill(self, dx: Decimal | str | float, dy: Decimal | str | float) -> None:
        """Apply an executed swap to pool reserves: x += dx, y -= dy.

        `dx` is the actual IN paid by taker (already on grid in normal flows);
        `dy` is the actual OUT delivered to taker (on grid).
        Rounds to amount grids defensively and rejects draining Y.
        """
        dx_gross = round_in_min(clamp_nonneg(to_decimal(dx)), is_xrp=self.x_is_xrp)
        dy_net = round_out_max(clamp_nonneg(to_decimal(dy)), is_xrp=self.y_is_xrp)
        if dx_gross < 0 or dy_net < 0:
            return
        # Pool-side deltas
        dx_to_pool = dx_gross * (Decimal(1) - self.tr_in)
        dy_from_pool = dy_net
        if self.tr_out > 0:
            one_minus_tr_out = (Decimal(1) - self.tr_out)
            if one_minus_tr_out <= 0:
                return
            # Pool must send more so that user receives dy_net
            dy_from_pool = quantize_up(dy_net / one_minus_tr_out, XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM)
        # Quantise deltas to grids (defensive)
        dx_to_pool = round_out_max(dx_to_pool, is_xrp=self.x_is_xrp)
        dy_from_pool = round_out_max(dy_from_pool, is_xrp=self.y_is_xrp)
        if dy_from_pool >= self.y:
            raise ValueError("apply_fill would drain Y reserve (gross)")
        # Update reserves
        self.x = self.x + dx_to_pool
        self.y = self.y - dy_from_pool