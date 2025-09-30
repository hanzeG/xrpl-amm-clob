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
from typing import List, Iterable, Optional, Callable

from .core import (
    to_decimal,
    clamp_nonneg,
    quantize_up,
    round_in_min,
    round_out_max,
    quantize_quality,
    quality_bucket,
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

    def preview_fees_for_fill(self, dx_gross: Decimal | str | float, dy_net: Decimal | str | float) -> tuple[Decimal, Decimal, Decimal]:
        """Preview fee breakdown for a hypothetical fill without mutating state.
        Inputs are interpreted as non-negative magnitudes; direction/sign is handled by callers.
        Returns (fee_pool, fee_tr_in, fee_tr_out) on the *user* grids, using the same rounding
        conventions as the swap/apply methods where applicable.
        """
        dx_g = round_out_max(clamp_nonneg(to_decimal(dx_gross)), is_xrp=self.x_is_xrp)
        dy_n = round_out_max(clamp_nonneg(to_decimal(dy_net)), is_xrp=self.y_is_xrp)
        fee_tr_in = Decimal(0)
        fee_pool = Decimal(0)
        fee_tr_out = Decimal(0)
        if dx_g > 0:
            fee_tr_in = dx_g * (self.tr_in if self.tr_in > 0 else Decimal(0))
            dx_after_tf = dx_g - fee_tr_in
            fee_pool = dx_after_tf * (self.fee if self.fee > 0 else Decimal(0))
            # keep on IN grid (no extra quantisation to avoid bias in preview)
        # Note: do not re-quantise to avoid bias in preview; use user grids once.
        if dy_n > 0 and self.tr_out > 0:
            one_minus_tr_out = (Decimal(1) - self.tr_out)
            if one_minus_tr_out > 0:
                # pool must send gross so user receives dy_n
                dy_gross = quantize_up(dy_n / one_minus_tr_out, XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM)
                fee_tr_out = dy_gross - dy_n
        return (fee_pool, fee_tr_in, fee_tr_out)

    # --- Swaps ---
    def swap_out_given_in(self, dx: Decimal | str | float) -> Decimal:
        """Max OUT for a given IN budget: IN budget is rounded down to the ledger grid; OUT is rounded down."""
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
            # For stateful callers, we keep an explicit exception on gross-out drain.
            # Pure preview paths elsewhere return 0 on infeasible; do not change semantics here.
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
        """Produce a single AMM slice whose *average* quality ~= q_threshold
        and whose *post-trade SPQ* (spot price quality) would remain >= q_threshold.

        This method implements the whitepaper's anchoring rule when AMM and CLOB
        coexist in the same book/tier (1.2.7.2):
        - If AMM SPQ is better than the CLOB top quality, generate a synthetic slice
          such that *after consuming that slice* the AMM's new SPQ is still at least
          as good as the CLOB top quality. This keeps path ordering stable.

        The method performs a conservative feasibility check using the same fee and
        grid policies as `apply_fill`, but without mutating the pool. If the initial
        anchored slice would push SPQ below the threshold, it shrinks the OUT size by
        bisection until the post-trade SPQ >= threshold (or returns None if no
        feasible positive slice exists).
        """
        # Basic guards
        if self.x <= 0 or self.y <= 0:
            return None
        q = clamp_nonneg(to_decimal(q_threshold))
        if q <= 0:
            return None

        # Local helpers mirroring `apply_fill` without mutating state
        one = Decimal("1")
        x_quant = XRP_QUANTUM if self.x_is_xrp else IOU_QUANTUM
        y_quant = XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM
        one_minus_fee = (one - self.fee)
        if one_minus_fee <= 0:
            return None
        one_minus_tr_in = (one - self.tr_in) if self.tr_in > 0 else one
        one_minus_tr_out = (one - self.tr_out) if self.tr_out > 0 else one
        if one_minus_tr_in <= 0 or one_minus_tr_out <= 0:
            return None

        def preview_new_spq(dx_gross: Decimal, dy_net: Decimal) -> Decimal:
            """Compute SPQ after hypothetically applying (dx_gross, dy_net)."""
            # Pool-side deltas as in `apply_fill`
            dx_to_pool = dx_gross * (one_minus_tr_in if self.tr_in > 0 else one)
            dy_from_pool = dy_net
            if self.tr_out > 0:
                # Gross OUT that must leave the pool so user receives dy_net
                dy_from_pool = quantize_up(dy_net / one_minus_tr_out, y_quant)
            # Quantise deltas conservatively (same policy as apply_fill)
            dx_to_pool = quantize_up(dx_to_pool, x_quant)
            # dy_from_pool already on OUT grid (floored earlier) or quantized up above when tr_out>0
            if dy_from_pool >= self.y:
                return Decimal(0)  # would drain the pool
            x_new = self.x + dx_to_pool
            y_new = self.y - dy_from_pool
            if x_new <= 0 or y_new <= 0:
                return Decimal(0)
            # New SPQ ≈ (y/x)*(1-fee), quantised down
            return quantize_quality((y_new / x_new) * one_minus_fee)

        # Step 1: initial anchored guess — follow current implementation logic.
        # Compute a tentative dy from target quality and optional cap.
        # We start with a theoretical dy that would (roughly) keep SPQ near q,
        # then we will verify/adjust it via preview_new_spq.
        dy_theory = self.y - (q * self.x) / one_minus_fee
        if max_out_cap is not None:
            cap = clamp_nonneg(to_decimal(max_out_cap))
            if cap > 0 and dy_theory > cap:
                dy_theory = cap
        dy_net = round_out_max(dy_theory, is_xrp=self.y_is_xrp)
        if dy_net <= 0 or dy_net >= self.y:
            return None

        # First-pass gross IN from anchored average quality
        dx_gross = round_in_min(dy_net / q, is_xrp=self.x_is_xrp)
        if dx_gross <= 0:
            return None
        # Feasibility check against full fee model on the *current* reserves
        dy_check = self.swap_out_given_in(dx_gross)
        if dy_check <= 0:
            return None
        if dy_check < dy_net:
            dy_net = dy_check
            dx_gross = round_in_min(dy_net / q, is_xrp=self.x_is_xrp)
            if dx_gross <= 0:
                return None

        # Step 2: enforce post-trade SPQ >= threshold via monotone shrink (bisection)
        spq_after = preview_new_spq(dx_gross, dy_net)
        if spq_after < q:
            # Binary search on dy_net in (0, dy_net] while maintaining anchored average quality
            lo = Decimal(0)
            hi = dy_net
            # Limit iterations to avoid pathological loops
            for _ in range(40):
                if hi - lo <= y_quant:
                    break
                mid = round_out_max((lo + hi) / 2, is_xrp=self.y_is_xrp)
                if mid <= 0:
                    break
                dx_mid = round_in_min(mid / q, is_xrp=self.x_is_xrp)
                if dx_mid <= 0:
                    # Too small; move lower bound up to avoid zero-in
                    lo = mid
                    continue
                spq_mid = preview_new_spq(dx_mid, mid)
                if spq_mid >= q:
                    # Feasible; try larger
                    dy_net = mid
                    dx_gross = dx_mid
                    lo = mid
                else:
                    # Infeasible; shrink
                    hi = mid
            # Final feasibility check
            spq_after = preview_new_spq(dx_gross, dy_net)
            if spq_after < q or dy_net <= 0:
                return None

        # Finalise slice quality conservatively
        q_slice = calc_quality(dy_net, dx_gross)
        q_slice = quality_bucket(q_slice)
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
        """Discretise AMM curve into slices up to target OUT using *shadow reserves*.

        Within one router iteration, successive slices are priced on a virtual
        state (x_hat, y_hat) updated after each slice (virtual writeback).
        This makes later slices strictly worse on average even if out sizes
        repeat (e.g., 20, 20, 40, ...). Real pool state is not mutated here;
        the router applies the actual writeback after the iteration.
        """
        remaining = clamp_nonneg(to_decimal(target_out))
        if remaining <= 0 or self.y <= 0 or self.x <= 0:
            return []

        y_quant = XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM
        x_quant = XRP_QUANTUM if self.x_is_xrp else IOU_QUANTUM

        base = self.y * start_fraction
        if base <= 0:
            return []
        base = quantize_up(base, y_quant)
        if base <= 0:
            base = y_quant

        # Shadow reserves (local); do not touch real pool state here
        x_hat = self.x
        y_hat = self.y

        one = Decimal("1")
        one_minus_fee = (one - self.fee)
        one_minus_tr_in = (one - self.tr_in) if self.tr_in > 0 else one
        one_minus_tr_out = (one - self.tr_out) if self.tr_out > 0 else one
        if one_minus_fee <= 0 or one_minus_tr_in <= 0 or one_minus_tr_out <= 0:
            return []

        def _swap_in_given_out_on(x_: Decimal, y_: Decimal, dy_net: Decimal) -> Decimal:
            # dy_net is net-to-user OUT on the OUT grid
            dy_net = quantize_up(clamp_nonneg(to_decimal(dy_net)), y_quant)
            if dy_net <= 0:
                return Decimal(0)
            # gross out from pool (issuer fee on receiver side if any)
            dy_gross = quantize_up(dy_net / one_minus_tr_out, y_quant) if self.tr_out > 0 else dy_net
            if dy_gross >= y_:
                return Decimal(0)
            # curve math on shadow state
            dx_eff = (dy_gross * x_) / (y_ - dy_gross)
            if dx_eff <= 0:
                return Decimal(0)
            # undo pool fee and issuer in-fee to get user gross IN
            dx_after_tf = dx_eff / one_minus_fee
            dx_gross = dx_after_tf / one_minus_tr_in if self.tr_in > 0 else dx_after_tf
            return round_in_min(dx_gross, is_xrp=self.x_is_xrp)

        def _swap_out_given_in_on(x_: Decimal, y_: Decimal, dx_gross: Decimal) -> Decimal:
            dx_gross = round_out_max(clamp_nonneg(to_decimal(dx_gross)), is_xrp=self.x_is_xrp)
            if dx_gross <= 0 or x_ <= 0 or y_ <= 0:
                return Decimal(0)
            dx_after_tf = dx_gross * (one_minus_tr_in if self.tr_in > 0 else one)
            dx_eff = dx_after_tf * one_minus_fee
            if dx_eff <= 0:
                return Decimal(0)
            dy_gross_theory = (y_ * dx_eff) / (x_ + dx_eff)
            dy_gross = round_out_max(dy_gross_theory, is_xrp=self.y_is_xrp)
            dy_net = dy_gross * (one_minus_tr_out if self.tr_out > 0 else one)
            dy_net = round_out_max(dy_net, is_xrp=self.y_is_xrp)
            return dy_net

        segs: List[Segment] = []
        f_prev, f_curr = 1, 1

        while remaining > 0 and len(segs) < max_segments:
            proposed = base * f_curr
            if proposed > remaining:
                proposed = remaining

            # Request OUT on the grid for this slice from the *shadow* state
            out_req = quantize_up(proposed, y_quant)
            if out_req <= 0:
                break
            if out_req >= y_hat:
                break  # avoid draining shadow pool side

            # Compute IN needed on shadow state
            in_need = _swap_in_given_out_on(x_hat, y_hat, out_req)
            if in_need <= 0:
                break

            # Recompute obtainable OUT from that IN on the same shadow state to
            # ensure feasibility after grid/fee effects
            dy_net = _swap_out_given_in_on(x_hat, y_hat, in_need)
            if dy_net <= 0:
                break
            if dy_net > out_req:
                # Cap to requested size for consistency
                dy_net = out_req

            q = calc_quality(dy_net, in_need)
            q = quality_bucket(q)
            if q <= 0:
                break

            segs.append(Segment(
                src="AMM",
                quality=q,
                out_max=dy_net,
                in_at_out_max=in_need,
                in_is_xrp=self.x_is_xrp,
                out_is_xrp=self.y_is_xrp,
            ))

            # Virtual writeback to shadow reserves (pool-side deltas)
            dx_to_pool = in_need * (one_minus_tr_in if self.tr_in > 0 else one)
            if self.tr_out > 0:
                # Gross OUT that must leave the pool so user receives dy_net: quantise up as in apply_fill
                dy_from_pool = quantize_up(dy_net / one_minus_tr_out, y_quant)
            else:
                # No issuer fee on OUT: keep dy_net as-is (already on OUT grid from swap)
                dy_from_pool = dy_net
            # Mirror apply_fill: avoid under-credit/debit on the shadow state
            dx_to_pool = quantize_up(dx_to_pool, x_quant)
            # dy_from_pool is already on correct grid; do not round it down
            if dy_from_pool >= y_hat:
                break
            x_hat = x_hat + dx_to_pool
            y_hat = y_hat - dy_from_pool

            remaining = remaining - dy_net
            if remaining <= 0:
                break

            # Fibonacci growth of slice sizes
            f_prev, f_curr = f_curr, f_prev + f_curr

        # Fallback: if no segments were constructed (e.g., tiny target or adverse quantisation),
        # attempt to produce a single feasible slice from the *current* state, per whitepaper
        # AMM-only bootstrap intuition (non-improving, strictly positive amounts on grids).
        if not segs and remaining > 0:
            # Start from a conservative OUT try: min(remaining, base). Ensure it's on the OUT grid and < y.
            dy_try = base if base <= remaining else remaining
            dy_try = quantize_up(dy_try, y_quant)
            if dy_try <= 0:
                dy_try = y_quant
            if dy_try > 0 and dy_try < self.y:
                # Compute required IN on current reserves with full fee/grid policy
                dx_need = _swap_in_given_out_on(self.x, self.y, dy_try)
                if dx_need > 0:
                    # Compute achievable OUT back from that IN on current reserves
                    dy_got = _swap_out_given_in_on(self.x, self.y, dx_need)
                    if dy_got > 0:
                        # Cap to requested size for consistency
                        if dy_got > dy_try:
                            dy_got = dy_try
                        q_fb = calc_quality(dy_got, dx_need)
                        q_fb = quality_bucket(q_fb)
                        if q_fb > 0:
                            segs.append(Segment(
                                src="AMM",
                                quality=q_fb,
                                out_max=dy_got,
                                in_at_out_max=dx_need,
                                in_is_xrp=self.x_is_xrp,
                                out_is_xrp=self.y_is_xrp,
                            ))

        return segs

    # --- State update after a filled amount (for next-iteration anchoring) ---
    def apply_fill(self, dx: Decimal | str | float, dy: Decimal | str | float) -> None:
        """Apply an executed swap to pool reserves: x += dx, y -= dy.

        `dx` is the actual IN paid by taker (already on grid in normal flows);
        `dy` is the actual OUT delivered to taker (on grid).
        Rounds to amount grids defensively and rejects draining Y.
        Pool deltas are quantised once with non-undercredit/undebit policy to avoid bias.
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
        # Quantise deltas to ledger grids with *non-undercredit/undebit* policy:
        # - Credit to pool (dx_to_pool): quantise UP to avoid under-crediting x-reserve.
        # - Debit from pool (dy_from_pool):
        #     * if tr_out>0, we've already quantised UP the gross (so user receives dy_net);
        #     * if tr_out==0, dy_net is already on the OUT grid from swap; do not re-round.
        x_quant = XRP_QUANTUM if self.x_is_xrp else IOU_QUANTUM
        y_quant = XRP_QUANTUM if self.y_is_xrp else IOU_QUANTUM
        dx_to_pool = quantize_up(dx_to_pool, x_quant)
        if self.tr_out == 0:
            # dy_from_pool == dy_net already floored to OUT grid in swap; keep as-is.
            pass
        else:
            # already quantized up above; keep as-is.
            pass
        # Guard against draining the Y reserve on gross-out after applying transfer fees.
        if dy_from_pool >= self.y:
            raise ValueError("apply_fill would drain Y reserve (gross)")
        # Update reserves
        self.x = self.x + dx_to_pool
        self.y = self.y - dy_from_pool

# -----------------------------
# Public helpers: unified AMM factories for experiments/tests
# -----------------------------

def amm_curve_from_linear(
    base_quality: Decimal,
    slope: Decimal,
    seg_out: Decimal,
    *,
    in_is_xrp: bool = False,
    out_is_xrp: bool = False,
) -> Callable[[Decimal], Iterable[Segment]]:
    """Return an `amm_curve(target_out)` producing AMM segments with mildly degrading quality.

    The instantaneous quality is `q_inst = base_quality - slope * max(target_out, 0)`
    (clamped to >0). The curve emits fixed-size OUT slices of `seg_out` until it reaches
    the requested `target_out`, with amounts snapped to the declared grids.

    This is a lightweight convenience for simulations; it does *not* mutate pool state.
    """
    if base_quality <= 0:
        raise ValueError("base_quality must be > 0")
    if slope < 0:
        raise ValueError("slope must be ≥ 0")
    if seg_out <= 0:
        raise ValueError("seg_out must be > 0")

    def amm_curve(target_out: Decimal) -> Iterable[Segment]:
        tgt = clamp_nonneg(to_decimal(target_out))
        if tgt <= 0:
            return []
        # Degrade quality with target size (but keep positive)
        q_inst = base_quality - (slope * tgt)
        if q_inst <= Decimal("1e-18"):
            return []
        out_chunk = round_out_max(seg_out, is_xrp=out_is_xrp)
        if out_chunk <= 0:
            return []
        remain = tgt
        segs: List[Segment] = []
        while remain > 0:
            take = remain if remain <= out_chunk else out_chunk
            take = round_out_max(take, is_xrp=out_is_xrp)
            if take <= 0:
                break
            # IN needed computed from quality; on IN grid
            in_need = round_in_min(take / q_inst, is_xrp=in_is_xrp)
            if in_need <= 0:
                break
            q = quality_bucket(calc_quality(take, in_need))
            if q <= 0:
                break
            segs.append(
                Segment(
                    src="AMM",
                    quality=q,
                    out_max=take,
                    in_at_out_max=in_need,
                    in_is_xrp=in_is_xrp,
                    out_is_xrp=out_is_xrp,
                )
            )
            remain -= take
        return segs

    return amm_curve


def amm_anchor_from_discount(
    discount: Decimal,
    *,
    cap_out: Optional[Decimal] = Decimal("50"),
    in_is_xrp: bool = False,
    out_is_xrp: bool = False,
) -> Callable[[Decimal, Decimal], Optional[Segment]]:
    """Return an `amm_anchor(q_lob_top, need)` that proposes a single anchored AMM slice.

    Given the current CLOB top quality `q_lob_top`, the anchored slice uses
    `q = q_lob_top * discount` (i.e., slightly worse than the book top) and
    size `take = min(need, cap_out)`; amounts are snapped to grids.

    This mirrors the whitepaper's anchoring intuition for AMM vs CLOB coexistence.
    """
    if discount <= 0:
        raise ValueError("discount must be > 0")

    def amm_anchor(q_lob_top: Decimal, need: Decimal) -> Optional[Segment]:
        q_top = clamp_nonneg(to_decimal(q_lob_top))
        req = clamp_nonneg(to_decimal(need))
        if q_top <= 0 or req <= 0:
            return None
        q = q_top * discount
        take = req
        if cap_out is not None and cap_out > 0 and take > cap_out:
            take = cap_out
        take = round_out_max(take, is_xrp=out_is_xrp)
        if take <= 0:
            return None
        in_need = round_in_min(take / q, is_xrp=in_is_xrp)
        if in_need <= 0:
            return None
        q_slice = quality_bucket(calc_quality(take, in_need))
        if q_slice <= 0:
            return None
        return Segment(
            src="AMM",
            quality=q_slice,
            out_max=take,
            in_at_out_max=in_need,
            in_is_xrp=in_is_xrp,
            out_is_xrp=out_is_xrp,
        )

    return amm_anchor