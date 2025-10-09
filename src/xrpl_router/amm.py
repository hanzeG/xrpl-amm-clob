"""
AMM base swaps (constant product, fee on input) — **pool math only**.

This module contains AMM maths used by the execution engine; production callers use the integer-domain wrappers:
- Pool fee (≤1%) is deducted on the *input* side.
- swap_out_given_in / swap_in_given_out implement constant‑product with issuer transfer fees.
- apply_fill updates reserves using non‑undercredit/undebit quantisation.
- Small‑trade spot price quality (SPQ) ≈ (y/x)*(1‑fee), quantised down.

Whitepaper alignment:
- OUT‑side issuer fee is **not** deducted from taker OUT at *segment construction*; it is
  accounted as ownerGives during execution (§1.3.2.4). This module’s swap methods already
  preview/handle issuer fees consistently.
- Strategy for AMM vs CLOB coexistence (anchoring) and multi‑path Fibonacci slicing is handled
  in `amm_liquidity.py` per §1.2.7.1–1.2.7.3. Helpers left here are for research only.

Optional issuer transfer fees (IOU‑only): tr_in on X (payer side), tr_out on Y (receiver side).
User quality is defined as OUT_net / IN_gross.

Research helpers below are for simulation only and are not part of the production routing path.
"""
from __future__ import annotations

from decimal import Decimal
from decimal import ROUND_FLOOR, ROUND_CEILING
from typing import List, Iterable, Optional, Callable
from .core.amounts import round_in_min, round_out_max, quantize_up
from .core.fmt import XRP_QUANTUM, IOU_QUANTUM
from .core.datatypes import Segment
from .core.amounts import STAmount
from .core.quality import Quality

# NOTE: Router-facing interfaces are integer-domain (STAmount, Quality, Segment).
# Decimal pool math is retained for preview/research helpers; production swaps use
# the pure integer-domain wrappers: swap_out_given_in_st / swap_in_given_out_st / apply_fill_st.

# Local helpers for Decimal use
DecimalLike = Decimal | int | str

def to_decimal(x: DecimalLike) -> Decimal:
    """Local bridge: normalise numeric-like to Decimal (I/O boundary only)."""
    return x if isinstance(x, Decimal) else Decimal(str(x))

def clamp_nonneg(x: DecimalLike) -> Decimal:
    """Local helper: clamp to non-negative Decimal (I/O boundary)."""
    d = to_decimal(x)
    return d if d > 0 else Decimal(0)

def quantize_quality(x: Decimal) -> Decimal:
    """Quantise quality for display/anchoring previews (Decimal-only)."""
    try:
        return x.quantize(Decimal("1e-12"))
    except Exception:
        return Decimal(0)

class AMM:
    """AMM(X,Y) with constant product math and input-side fee (integer-domain wrappers available).

    Orientation: inputs are X, outputs are Y for these methods.
    Amount grids: X uses drops if x_is_xrp, else 1e-15; Y likewise.
    """

    def __init__(self,
                 x_reserve: DecimalLike,
                 y_reserve: DecimalLike,
                 fee: DecimalLike,
                 *,
                 x_is_xrp: bool,
                 y_is_xrp: bool,
                 tr_in: DecimalLike = Decimal("0"),
                 tr_out: DecimalLike = Decimal("0")) -> None:
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

        # Integer-domain fee representation (parts-per-billion) for pure-integer curve math
        self._SCALE = 10**9
        def _ppb(x: Decimal) -> int:
            # clamp and round to nearest integer
            if x <= 0:
                return 0
            if x >= 1:
                return self._SCALE
            return int((x * self._SCALE).to_integral_value(rounding=ROUND_FLOOR))
        self._fee_num = _ppb(self.fee)
        self._fee_den = self._SCALE
        self._keep_fee_num = self._fee_den - self._fee_num  # (1 - fee)
        self._tr_in_num = _ppb(self.tr_in)
        self._tr_in_den = self._SCALE
        self._keep_tr_in_num = self._tr_in_den - self._tr_in_num  # (1 - tr_in)
        self._tr_out_num = _ppb(self.tr_out)
        self._tr_out_den = self._SCALE
        self._keep_tr_out_num = self._tr_out_den - self._tr_out_num  # (1 - tr_out)

    # --- Diagnostics ---
    def spq(self) -> Decimal:
        """Small-trade quality ≈ (y/x)*(1-fee), quantised to quality grid."""
        if self.x <= 0 or self.y <= 0:
            return Decimal(0)
        return quantize_quality((self.y / self.x) * (Decimal(1) - self.fee))

    def preview_fees_for_fill(self, dx_gross: DecimalLike, dy_net: DecimalLike) -> tuple[Decimal, Decimal, Decimal]:
        """Preview fee breakdown for a hypothetical fill without mutating state.
        This is a Decimal preview helper; integer-domain execution uses the *_st methods.
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
    def swap_out_given_in(self, dx: DecimalLike) -> Decimal:
        """[PREVIEW/RESEARCH] Max OUT for a given IN budget (Decimal path).
        IN is rounded down to the ledger grid; OUT is rounded down. Production callers
        must use `swap_out_given_in_st` (integer-domain)."""
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

    def swap_in_given_out(self, dy: DecimalLike) -> Decimal:
        """[PREVIEW/RESEARCH] Min IN to obtain target OUT (Decimal path; ceiled to IN grid).
        Rejects dy ≥ y (gross-out). Production callers must use `swap_in_given_out_st` (integer-domain)."""
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

    # --- Integer grid helpers ---
    def _quant(self, is_xrp: bool) -> Decimal:
        return XRP_QUANTUM if is_xrp else IOU_QUANTUM

    def _grid_exp(self, is_xrp: bool) -> int:
        # Both XRP and IOU use 1e-15 quantum at STAmount boundary
        return -15


    def _to_units_floor(self, amt_dec: Decimal, is_xrp: bool) -> int:
        q = self._quant(is_xrp)
        if amt_dec <= 0:
            return 0
        return int((amt_dec / q).to_integral_value(rounding=ROUND_FLOOR))

    def _to_units_ceil(self, amt_dec: Decimal, is_xrp: bool) -> int:
        q = self._quant(is_xrp)
        if amt_dec <= 0:
            return 0
        return int((amt_dec / q).to_integral_value(rounding=ROUND_CEILING))

    def _st_to_units_floor(self, amt: STAmount) -> int:
        """Convert STAmount to integer grid units (floor), avoiding Decimal.
        One unit corresponds to 10^(-15) in STAmount. Negative or zero returns 0.
        """
        if amt is None or amt.is_zero() or amt.sign <= 0:
            return 0
        k = amt.exponent + 15
        if k >= 0:
            return amt.mantissa * (10 ** k)
        div = 10 ** (-k)
        return amt.mantissa // div

    def _st_to_units_ceil(self, amt: STAmount) -> int:
        """Convert STAmount to integer grid units (ceil), avoiding Decimal.
        One unit corresponds to 10^(-15) in STAmount. Negative or zero returns 0.
        """
        if amt is None or amt.is_zero() or amt.sign <= 0:
            return 0
        k = amt.exponent + 15
        if k >= 0:
            return amt.mantissa * (10 ** k)
        div = 10 ** (-k)
        return (amt.mantissa + div - 1) // div

    def _from_units(self, units: int, is_xrp: bool) -> STAmount:
        exp = self._grid_exp(is_xrp)
        # 1 unit == 10^(-15)
        return STAmount.from_components(units, exp, 1)

    def _reserve_units(self) -> tuple[int, int]:
        """Return current reserves as integer grid units (floor)."""
        return (
            self._to_units_floor(self.x, self.x_is_xrp),
            self._to_units_floor(self.y, self.y_is_xrp),
        )

    def spq_quality_int(self) -> Quality:
        """Integer-domain SPQ (small-trade quality) as a Quality object.
        Computed on reserve units with input-side pool fee applied; no Decimal involved.
        """
        x_u, y_u = self._reserve_units()
        if x_u <= 0 or y_u <= 0 or self._keep_fee_num <= 0:
            # Zero quality if reserves/fee invalid
            return Quality.from_amounts(STAmount.zero(), STAmount.from_components(1, -15, 1))
        # SPQ ≈ (y/x) * (1 - fee) ⇒ numerator scaled by keep_fee_num, denominator by fee_den
        out_num_units = y_u * self._keep_fee_num
        in_den_units = x_u * self._fee_den
        out_amt = self._from_units(out_num_units, self.y_is_xrp)
        in_amt = self._from_units(in_den_units, self.x_is_xrp)
        return Quality.from_amounts(out_amt, in_amt)

    @staticmethod
    def _mul_keep_floor(value_units: int, keep_num: int, keep_den: int) -> int:
        # floor(value * (keep_num/keep_den))
        if value_units <= 0 or keep_num <= 0:
            return 0
        return (value_units * keep_num) // keep_den

    @staticmethod
    def _mul_keep_ceil(value_units: int, keep_num: int, keep_den: int) -> int:
        # ceil(value * (keep_num/keep_den))
        if value_units <= 0:
            return 0
        num = value_units * keep_num
        return (num + keep_den - 1) // keep_den

    @staticmethod
    def _ceil_div(num: int, den: int) -> int:
        if num <= 0:
            return 0
        return (num + den - 1) // den

    # --- Integer-domain convenience wrappers (non-breaking) ---
    def swap_out_given_in_st(self, dx: STAmount) -> STAmount:
        """Integer-domain wrapper: return OUT (net to user) for a given IN (gross),
        both as STAmount. Uses pure integer-domain curve math.
        """
        if dx is None or dx.is_zero():
            return STAmount.zero()
        # Input units on X grid
        dx_units = self._st_to_units_floor(dx)
        if dx_units <= 0:
            return STAmount.zero()
        x_units, y_units = self._reserve_units()
        if x_units <= 0 or y_units <= 0:
            return STAmount.zero()
        # Apply transfer-in and pool fee (floor to avoid over-crediting)
        dx_after_tf = self._mul_keep_floor(dx_units, self._keep_tr_in_num, self._tr_in_den)
        dx_eff = self._mul_keep_floor(dx_after_tf, self._keep_fee_num, self._fee_den)
        if dx_eff <= 0:
            return STAmount.zero()
        # Constant-product out (gross from pool), floored
        denom = x_units + dx_eff
        dy_gross_units = (y_units * dx_eff) // denom
        if dy_gross_units <= 0:
            return STAmount.zero()
        # Receiver-side transfer-out: net to user, floored
        dy_net_units = self._mul_keep_floor(dy_gross_units, self._keep_tr_out_num, self._tr_out_den)
        if dy_net_units <= 0:
            return STAmount.zero()
        return self._from_units(dy_net_units, self.y_is_xrp)

    def swap_in_given_out_st(self, dy: STAmount) -> STAmount:
        """Integer-domain wrapper: return minimal IN (gross) as STAmount to obtain
        target OUT (net to user) dy. Uses pure integer-domain curve math.
        """
        if dy is None or dy.is_zero():
            return STAmount.zero()
        # Target OUT units on Y grid (ceil for conservatism)
        dy_net_units = self._st_to_units_ceil(dy)
        if dy_net_units <= 0:
            return STAmount.zero()
        x_units, y_units = self._reserve_units()
        if x_units <= 0 or y_units <= 0:
            return STAmount.zero()
        # Convert to gross-out units (ceil so user surely receives dy)
        if self._tr_out_num > 0:
            dy_gross_units = self._ceil_div(dy_net_units * self._tr_out_den, self._keep_tr_out_num)
        else:
            dy_gross_units = dy_net_units
        if dy_gross_units >= y_units:
            return STAmount.zero()
        # Effective input units needed at curve level (ceil)
        num = dy_gross_units * x_units
        den = y_units - dy_gross_units
        dx_eff_units = self._ceil_div(num, den)
        if dx_eff_units <= 0:
            return STAmount.zero()
        # Undo pool fee and transfer-in (ceil)
        if self._keep_fee_num <= 0:
            return STAmount.zero()
        dx_after_tf_units = self._ceil_div(dx_eff_units * self._fee_den, self._keep_fee_num)
        if self._keep_tr_in_num <= 0:
            return STAmount.zero()
        dx_gross_units = self._ceil_div(dx_after_tf_units * self._tr_in_den, self._keep_tr_in_num)
        if dx_gross_units <= 0:
            return STAmount.zero()
        return self._from_units(dx_gross_units, self.x_is_xrp)

    def apply_fill_st(self, dx: STAmount, dy: STAmount) -> None:
        """Integer-domain wrapper for apply_fill: updates reserves using STAmount inputs.
        Uses pure integer-domain grid math for pool state update.
        """
        if dx is None or dy is None:
            return
        if dx.is_zero() and dy.is_zero():
            return
        dx_units = self._st_to_units_floor(dx)
        dy_units = self._st_to_units_floor(dy)
        if dx_units < 0 or dy_units < 0:
            return
        # Pool-side deltas in units (quantise up for credit, up for gross-out if tr_out>0)
        dx_to_pool_units = self._mul_keep_ceil(dx_units, self._keep_tr_in_num, self._tr_in_den)
        if self._tr_out_num > 0:
            dy_from_pool_units = self._ceil_div(dy_units * self._tr_out_den, self._keep_tr_out_num)
        else:
            dy_from_pool_units = dy_units
        # Guard against draining
        x_units, y_units = self._reserve_units()
        if dy_from_pool_units >= y_units:
            raise ValueError("apply_fill would drain Y reserve (gross)")
        # Convert back to Decimal reserves using quanta
        x_q = self._quant(self.x_is_xrp)
        y_q = self._quant(self.y_is_xrp)
        self.x = self.x + (Decimal(dx_to_pool_units) * x_q)
        self.y = self.y - (Decimal(dy_from_pool_units) * y_q)

    # --- Whitepaper-style synthetic quote (anchor to target quality) ---
    def synthetic_segment_for_quality(self,
                                      q_threshold: Quality,
                                      *,
                                      max_out_cap: STAmount | None = None
                                      ) -> Segment | None:
        """
        Produce a single AMM slice whose average quality ≥ q_threshold and whose
        post-trade SPQ remains ≥ q_threshold. Fully integer-domain implementation.
        """
        # Reserves in integer units
        x_u, y_u = self._reserve_units()
        if x_u <= 0 or y_u <= 0:
            return None

        # Anchor condition: current SPQ must be at least as good as threshold
        spq_now = self.spq_quality_int()
        if not (spq_now.rate >= q_threshold.rate):
            return None

        # OUT cap in units (optional)
        cap_units = None
        if max_out_cap is not None and (not max_out_cap.is_zero()):
            cap_units = self._st_to_units_floor(max_out_cap)
            if cap_units <= 0:
                cap_units = None

        # Helper: given dy_net_units, compute dx_gross_units and feasibility flags
        def compute_dx_for_dy(dy_net_units: int) -> tuple[int, bool, Quality | None]:
            if dy_net_units <= 0:
                return 0, False, None
            # Gross out from pool so user receives dy_net
            if self._tr_out_num > 0:
                dy_gross_units = self._ceil_div(dy_net_units * self._tr_out_den, self._keep_tr_out_num)
            else:
                dy_gross_units = dy_net_units
            if dy_gross_units <= 0 or dy_gross_units >= y_u:
                return 0, False, None
            # Effective input at curve level
            num = dy_gross_units * x_u
            den = y_u - dy_gross_units
            dx_eff_units = self._ceil_div(num, den)
            if dx_eff_units <= 0:
                return 0, False, None
            # Undo pool fee and transfer-in to obtain gross IN from user
            if self._keep_fee_num <= 0 or self._keep_tr_in_num <= 0:
                return 0, False, None
            dx_after_tf_units = self._ceil_div(dx_eff_units * self._fee_den, self._keep_fee_num)
            dx_gross_units = self._ceil_div(dx_after_tf_units * self._tr_in_den, self._keep_tr_in_num)
            if dx_gross_units <= 0:
                return 0, False, None
            # Average slice quality (OUT/IN) in integer domain
            out_amt = self._from_units(dy_net_units, self.y_is_xrp)
            in_amt = self._from_units(dx_gross_units, self.x_is_xrp)
            q_slice = Quality.from_amounts(out_amt, in_amt)
            ok_quality = (q_slice.rate.sign > 0 and q_slice.rate >= q_threshold.rate)
            return dx_gross_units, ok_quality, q_slice

        # Helper: SPQ after applying (dx_gross_units, dy_net_units) must remain ≥ threshold
        def spq_after_ok(dx_gross_units: int, dy_net_units: int) -> bool:
            # Pool-side deltas
            dx_to_pool_units = self._mul_keep_ceil(dx_gross_units, self._keep_tr_in_num, self._tr_in_den)
            if self._tr_out_num > 0:
                dy_from_pool_units = self._ceil_div(dy_net_units * self._tr_out_den, self._keep_tr_out_num)
            else:
                dy_from_pool_units = dy_net_units
            if dy_from_pool_units >= y_u:
                return False
            x_new = x_u + dx_to_pool_units
            y_new = y_u - dy_from_pool_units
            if x_new <= 0 or y_new <= 0:
                return False
            # SPQ' ≈ (y_new/x_new)*(1-fee) ≥ q_threshold
            out_num_units = y_new * self._keep_fee_num
            in_den_units = x_new * self._fee_den
            out_amt = self._from_units(out_num_units, self.y_is_xrp)
            in_amt = self._from_units(in_den_units, self.x_is_xrp)
            spq_next = Quality.from_amounts(out_amt, in_amt)
            return spq_next.rate >= q_threshold.rate

        # Binary search on dy_net_units in [1, hi]
        hi = y_u - 1
        if cap_units is not None and cap_units < hi:
            hi = cap_units
        if hi <= 0:
            return None
        lo = 0
        best_dy = 0
        best_dx = 0
        best_q: Quality | None = None

        # Upward-biased bisection to find the largest feasible dy
        for _ in range(50):
            if lo >= hi:
                break
            mid = (lo + hi + 1) // 2
            dx_mid, ok_q, q_mid = compute_dx_for_dy(mid)
            if ok_q and dx_mid > 0 and spq_after_ok(dx_mid, mid):
                best_dy, best_dx, best_q = mid, dx_mid, q_mid
                lo = mid
            else:
                hi = mid - 1
        # Final check at lo (when loop exits with lo==hi)
        if best_dy == 0 and lo > 0:
            dx_lo, ok_q, q_lo = compute_dx_for_dy(lo)
            if ok_q and dx_lo > 0 and spq_after_ok(dx_lo, lo):
                best_dy, best_dx, best_q = lo, dx_lo, q_lo

        if best_dy <= 0 or best_dx <= 0 or best_q is None:
            return None

        # Construct integer-domain segment
        out_st = self._from_units(best_dy, self.y_is_xrp)
        in_st = self._from_units(best_dx, self.x_is_xrp)
        q_slice = Quality.from_amounts(out_st, in_st)
        if q_slice.rate.sign <= 0 or not (q_slice.rate >= q_threshold.rate):
            return None
        return Segment(
            src="AMM",
            quality=q_slice,
            out_max=out_st,
            in_at_out_max=in_st,
            in_is_xrp=self.x_is_xrp,
            out_is_xrp=self.y_is_xrp,
        )

    # --- Segmentation (standalone; not used by anchored routing) ---
    # NOTE: Research utility (non‑whitepaper). Not used by anchored routing; kept for experiments.
    def segments_for_out(self,
                        target_out: DecimalLike,
                        *,
                        max_segments: int = 30,
                        start_fraction: DecimalLike = Decimal("1e-4")) -> List[Segment]:
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

            out_st = STAmount.from_decimal(dy_net)
            in_st = STAmount.from_decimal(in_need)
            q = Quality.from_amounts(out_st, in_st)
            if q.rate.sign <= 0:
                break

            segs.append(Segment(
                src="AMM",
                quality=q,
                out_max=out_st,
                in_at_out_max=in_st,
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
                        out_st = STAmount.from_decimal(dy_got)
                        in_st = STAmount.from_decimal(dx_need)
                        q_fb = Quality.from_amounts(out_st, in_st)
                        if q_fb.rate.sign > 0:
                            segs.append(Segment(
                                src="AMM",
                                quality=q_fb,
                                out_max=out_st,
                                in_at_out_max=in_st,
                                in_is_xrp=self.x_is_xrp,
                                out_is_xrp=self.y_is_xrp,
                            ))

        return segs

    # --- State update after a filled amount (for next-iteration anchoring) ---
    def apply_fill(self, dx: DecimalLike, dy: DecimalLike) -> None:
        """[PREVIEW/RESEARCH] Apply an executed swap to pool reserves (Decimal path): x += dx, y -= dy.

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

# --- Research helpers (non‑whitepaper) ---
def amm_curve_from_linear(
    base_quality: DecimalLike,
    slope: DecimalLike,
    seg_out: DecimalLike,
    *,
    in_is_xrp: bool = False,
    out_is_xrp: bool = False,
) -> Callable[[Decimal], Iterable[Segment]]:
    """Research helper (non‑whitepaper). Return an `amm_curve(target_out)` producing AMM segments with mildly degrading quality.
    All returned values are integer-domain (STAmount, Quality, Segment).
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
            in_need = round_in_min(take / q_inst, is_xrp=in_is_xrp)
            if in_need <= 0:
                break
            out_st = STAmount.from_decimal(take)
            in_st = STAmount.from_decimal(in_need)
            q = Quality.from_amounts(out_st, in_st)
            if q.rate.sign <= 0:
                break
            segs.append(
                Segment(
                    src="AMM",
                    quality=q,
                    out_max=out_st,
                    in_at_out_max=in_st,
                    in_is_xrp=in_is_xrp,
                    out_is_xrp=out_is_xrp,
                )
            )
            remain -= take
        return segs

    return amm_curve


# --- Research helpers (non‑whitepaper) ---
def amm_anchor_from_discount(
    discount: DecimalLike,
    *,
    cap_out: Optional[DecimalLike] = Decimal("50"),
    in_is_xrp: bool = False,
    out_is_xrp: bool = False,
) -> Callable[[Decimal, Decimal], Optional[Segment]]:
    """Research helper (non‑whitepaper). Return an `amm_anchor(q_lob_top, need)` that proposes a single anchored AMM slice.
    All returned values are integer-domain (STAmount, Quality, Segment).
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
        out_st = STAmount.from_decimal(take)
        in_st = STAmount.from_decimal(in_need)
        q_slice = Quality.from_amounts(out_st, in_st)
        if q_slice.rate.sign <= 0:
            return None
        return Segment(
            src="AMM",
            quality=q_slice,
            out_max=out_st,
            in_at_out_max=in_st,
            in_is_xrp=in_is_xrp,
            out_is_xrp=out_is_xrp,
        )

    return amm_anchor