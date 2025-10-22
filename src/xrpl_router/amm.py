"""
AMM base swaps (constant product, fee on input) — **pool math only**.

This module contains AMM maths used by the execution engine; production callers use the integer-domain wrappers.
Pool fee (≤1%) is deducted on the *input* side.
"""
from __future__ import annotations

# --- Fraction (rational) helpers for integer-only quality math ---
from fractions import Fraction

from decimal import Decimal
from decimal import ROUND_FLOOR, ROUND_CEILING
# --- Fraction (rational) helpers for integer-only quality math ---

def _spq_frac_from_units(x_u: int, y_u: int, keep_fee_num: int, fee_den: int) -> Fraction:
    if x_u <= 0 or y_u <= 0 or keep_fee_num <= 0 or fee_den <= 0:
        return Fraction(0, 1)
    return Fraction(y_u * keep_fee_num, x_u * fee_den)

# --- AMM over-ask exception (for non-drain net capacity) ---
class AMMOverAsk(Exception):
    """Raised when requested OUT exceeds AMM's non-drain net capacity.

    Carries the maximum deliverable OUT (net) and the required IN for that cap.
    """
    def __init__(self, max_out: "Amount", dx_for_max: "Amount"):
        super().__init__("AMM request exceeds capacity without draining reserves")
        self.max_out = max_out
        self.dx_for_max = dx_for_max

from typing import List, Iterable, Optional
from .core import Amount, XRPAmount, IOUAmount, Quality
from .core.fmt import XRP_QUANTUM, IOU_QUANTUM

from .core.datatypes import Segment

# --- Debug utilities (toggleable) ---
DEBUG_AMM = False

def _dbg(msg: str) -> None:
    if DEBUG_AMM:
        print(f"[AMM] {msg}")

# NOTE: Router-facing interfaces are integer-domain (Amount, Quality, Segment).
# Decimal pool math is retained for preview/research helpers; production swaps use
# the pure integer-domain wrappers: swap_out_given_in_st / swap_in_given_out_st / apply_fill_st.

# Local helpers for Decimal use
DecimalLike = Decimal | int | str

def to_decimal(x: DecimalLike) -> Decimal:
    """Local bridge: normalise numeric-like to Decimal (I/O boundary only)."""
    return x if isinstance(x, Decimal) else Decimal(str(x))

class AMM:

    # --- Capacity helpers (integer-domain) ---
    def max_out_net_cap_st(self) -> Amount:
        """Return the maximum OUT (net to user) allowed by non-drain rule (leave ≥1 gross unit).
        Computed on integer grids and converted back to Amount.
        """
        x_u, y_u = self._reserve_units()
        if x_u <= 0 or y_u <= 1:
            # cannot take anything if empty or would drain
            return XRPAmount(0) if self.y_is_xrp else IOUAmount.zero()
        # leave at least 1 gross unit in pool (non-drain rule)
        dy_gross_max = y_u - 1
        # apply transfer-out to get net-to-user capacity
        if self._tr_out_num > 0:
            dy_net_cap_units = (dy_gross_max * self._keep_tr_out_num) // self._tr_out_den
        else:
            dy_net_cap_units = dy_gross_max
        if dy_net_cap_units <= 0:
            return XRPAmount(0) if self.y_is_xrp else IOUAmount.zero()
        return self._from_units(dy_net_cap_units, self.y_is_xrp)

    def dx_for_out_st(self, dy: Amount, *, raise_on_overask: bool = False) -> tuple[Amount, bool]:
        """Return (dx_needed, capped) in integer-domain for the requested OUT (net to user).

        If the request exceeds non-drain capacity, either:
          - when raise_on_overask=True: raise AMMOverAsk(max_out, dx_for_max)
          - else: cap at max_out and return dx for that cap with capped=True.
        """
        # validate request
        if dy is None or (isinstance(dy, XRPAmount) and dy.value == 0) or (isinstance(dy, IOUAmount) and dy.mantissa == 0):
            z = XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
            return z, False
        # units for requested OUT (ceil to guarantee delivery)
        dy_net_units_req = self._amt_to_units_ceil(dy)
        if dy_net_units_req <= 0:
            z = XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
            return z, False
        # current reserves in units
        x_u, y_u = self._reserve_units()
        if x_u <= 0 or y_u <= 0:
            z = XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
            return z, False
        # compute non-drain net capacity in units
        if y_u <= 1:
            dy_net_cap_units = 0
        else:
            dy_gross_max = y_u - 1
            if self._tr_out_num > 0:
                dy_net_cap_units = (dy_gross_max * self._keep_tr_out_num) // self._tr_out_den
            else:
                dy_net_cap_units = dy_gross_max
        if dy_net_cap_units <= 0:
            z = XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
            return z, False
        # check over-ask and cap if needed
        capped = False
        dy_net_units = dy_net_units_req
        if dy_net_units_req > dy_net_cap_units:
            capped = True
            dy_net_units = dy_net_cap_units

        # compute dx for dy_net_units (integer math, identical to swap_in_given_out_st core)
        # gross-out units from pool so user receives dy_net_units
        if self._tr_out_num > 0:
            dy_gross_units = self._ceil_div(dy_net_units * self._tr_out_den, self._keep_tr_out_num)
        else:
            dy_gross_units = dy_net_units
        # guard: still must respect non-drain rule
        if dy_gross_units >= y_u:
            # should not happen due to cap, clamp one unit below
            dy_gross_units = y_u - 1
        # effective input at curve level
        num = dy_gross_units * x_u
        den = y_u - dy_gross_units
        dx_eff_units = self._ceil_div(num, den)
        if dx_eff_units <= 0:
            z = XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
            return z, capped
        # undo pool fee and transfer-in
        if self._keep_fee_num <= 0 or self._keep_tr_in_num <= 0:
            z = XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
            return z, capped
        dx_after_tf_units = self._ceil_div(dx_eff_units * self._fee_den, self._keep_fee_num)
        dx_gross_units = self._ceil_div(dx_after_tf_units * self._tr_in_den, self._keep_tr_in_num)
        if dx_gross_units <= 0:
            z = XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
            return z, capped
        dx_amt = self._from_units(dx_gross_units, self.x_is_xrp)

        if raise_on_overask and capped:
            max_out_amt = self._from_units(dy_net_units, self.y_is_xrp)
            raise AMMOverAsk(max_out_amt, dx_amt)
        return dx_amt, capped
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

    # --- Integer grid helpers ---
    def _quant(self, is_xrp: bool) -> Decimal:
        return XRP_QUANTUM if is_xrp else IOU_QUANTUM



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

    def _amt_to_units_floor(self, amt: Amount) -> int:
        """Convert Amount to integer grid units (floor), avoiding Decimal.
        XRP units are drops; IOU units correspond to 10^(-15).
        """
        if amt is None:
            return 0
        if isinstance(amt, XRPAmount):
            return amt.value if amt.value > 0 else 0
        # IOUAmount path
        m = amt.mantissa
        e = amt.exponent
        if m <= 0:
            return 0
        k = e + 15
        if k >= 0:
            return m * (10 ** k)
        div = 10 ** (-k)
        return m // div

    def _amt_to_units_ceil(self, amt: Amount) -> int:
        """Convert Amount to integer grid units (ceil), avoiding Decimal.
        XRP units are drops; IOU units correspond to 10^(-15).
        """
        if amt is None:
            return 0
        if isinstance(amt, XRPAmount):
            return amt.value if amt.value > 0 else 0
        # IOUAmount path
        m = amt.mantissa
        e = amt.exponent
        if m <= 0:
            return 0
        k = e + 15
        if k >= 0:
            return m * (10 ** k)
        div = 10 ** (-k)
        return (m + div - 1) // div

    def _from_units(self, units: int, is_xrp: bool) -> Amount:
        # 1 unit == 1 drop for XRP; 1 unit == 10^(-15) for IOU
        if units <= 0:
            return XRPAmount(0) if is_xrp else IOUAmount.zero()
        return XRPAmount(units) if is_xrp else IOUAmount.from_components(units, -15)

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
            return Quality.from_amounts(XRPAmount(0), IOUAmount.from_components(1, 0))
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

    def clone(self) -> "AMM":
        """Shallow clone of the AMM state for shadow routing in reverse."""
        return AMM(self.x, self.y, self.fee,
                   x_is_xrp=self.x_is_xrp, y_is_xrp=self.y_is_xrp,
                   tr_in=self.tr_in, tr_out=self.tr_out)
    
    # --- Integer-domain convenience wrappers (non-breaking) ---
    def swap_out_given_in_st(self, dx: Amount) -> Amount:
        """Integer-domain wrapper: return OUT (net to user) for a given IN (gross),
        both as Amount. Uses pure integer-domain curve math.
        """
        if dx is None or (isinstance(dx, XRPAmount) and dx.value == 0) or (isinstance(dx, IOUAmount) and dx.mantissa == 0):
            return XRPAmount(0) if self.y_is_xrp else IOUAmount.zero()
        # Input units on X grid
        dx_units = self._amt_to_units_floor(dx)
        if dx_units <= 0:
            return XRPAmount(0) if self.y_is_xrp else IOUAmount.zero()
        x_units, y_units = self._reserve_units()
        if x_units <= 0 or y_units <= 0:
            return XRPAmount(0) if self.y_is_xrp else IOUAmount.zero()
        # Apply transfer-in and pool fee (floor to avoid over-crediting)
        dx_after_tf = self._mul_keep_floor(dx_units, self._keep_tr_in_num, self._tr_in_den)
        dx_eff = self._mul_keep_floor(dx_after_tf, self._keep_fee_num, self._fee_den)
        if dx_eff <= 0:
            return XRPAmount(0) if self.y_is_xrp else IOUAmount.zero()
        # Constant-product out (gross from pool), floored
        denom = x_units + dx_eff
        dy_gross_units = (y_units * dx_eff) // denom
        if dy_gross_units <= 0:
            return XRPAmount(0) if self.y_is_xrp else IOUAmount.zero()
        # Receiver-side transfer-out: net to user, floored
        dy_net_units = self._mul_keep_floor(dy_gross_units, self._keep_tr_out_num, self._tr_out_den)
        if dy_net_units <= 0:
            return XRPAmount(0) if self.y_is_xrp else IOUAmount.zero()
        return self._from_units(dy_net_units, self.y_is_xrp)

    def swap_in_given_out_st(self, dy: Amount) -> Amount:
        """Integer-domain wrapper: return minimal IN (gross) as Amount to obtain
        target OUT (net to user) dy. Uses pure integer-domain curve math.
        """
        if dy is None or (isinstance(dy, XRPAmount) and dy.value == 0) or (isinstance(dy, IOUAmount) and dy.mantissa == 0):
            return XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
        # Target OUT units on Y grid (ceil for conservatism)
        dy_net_units = self._amt_to_units_ceil(dy)
        if dy_net_units <= 0:
            return XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
        x_units, y_units = self._reserve_units()
        if x_units <= 0 or y_units <= 0:
            return XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
        # Convert to gross-out units (ceil so user surely receives dy)
        if self._tr_out_num > 0:
            dy_gross_units = self._ceil_div(dy_net_units * self._tr_out_den, self._keep_tr_out_num)
        else:
            dy_gross_units = dy_net_units
        if dy_gross_units >= y_units:
            return XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
        # Effective input units needed at curve level (ceil)
        num = dy_gross_units * x_units
        den = y_units - dy_gross_units
        dx_eff_units = self._ceil_div(num, den)
        if dx_eff_units <= 0:
            return XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
        # Undo pool fee and transfer-in (ceil)
        if self._keep_fee_num <= 0:
            return XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
        dx_after_tf_units = self._ceil_div(dx_eff_units * self._fee_den, self._keep_fee_num)
        if self._keep_tr_in_num <= 0:
            return XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
        dx_gross_units = self._ceil_div(dx_after_tf_units * self._tr_in_den, self._keep_tr_in_num)
        if dx_gross_units <= 0:
            return XRPAmount(0) if self.x_is_xrp else IOUAmount.zero()
        return self._from_units(dx_gross_units, self.x_is_xrp)

    def apply_fill_st(self, dx: Amount, dy: Amount) -> None:
        """Integer-domain wrapper for apply_fill: updates reserves using Amount inputs.
        Uses pure integer-domain grid math for pool state update.
        """
        if dx is None or dy is None:
            return
        if ((isinstance(dx, XRPAmount) and dx.value == 0) or (isinstance(dx, IOUAmount) and dx.mantissa == 0)) and \
           ((isinstance(dy, XRPAmount) and dy.value == 0) or (isinstance(dy, IOUAmount) and dy.mantissa == 0)):
            return
        prev_spq_frac = self.spq_quality_int().as_fraction()
        # Pre-trade units for adaptive epsilon
        x_units_pre, y_units_pre = self._reserve_units()
        dx_units = self._amt_to_units_floor(dx)
        dy_units = self._amt_to_units_floor(dy)
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
        # Safety: reserves must remain positive and SPQ should not improve after a fill
        if self.x <= 0 or self.y <= 0:
            raise ValueError("apply_fill_st led to non‑positive reserves")
        next_spq_frac = self.spq_quality_int().as_fraction()
        # With correct integer rounding, SPQ must not improve after a fill
        if next_spq_frac > prev_spq_frac:
            raise AssertionError(
                f"AMM SPQ increased after fill: prev={prev_spq_frac}, next={next_spq_frac}")

    # --- Whitepaper-style synthetic quote (anchor to target quality) ---
    def synthetic_segment_for_quality(self,
                                      q_threshold: Quality,
                                      *,
                                      max_out_cap: Amount | None = None
                                      ) -> Segment | None:
        """
        Produce a single AMM slice whose average quality ≥ q_threshold and whose
        post-trade SPQ remains ≥ q_threshold. Fully integer-domain implementation.
        """
        # Reserves in integer units
        x_u, y_u = self._reserve_units()
        _dbg(f"synthetic_segment: reserves_units x_u={x_u}, y_u={y_u}")
        if x_u <= 0 or y_u <= 0:
            return None

        # Anchor condition: current SPQ must be at least as good as threshold
        spq_now = self.spq_quality_int()
        _dbg(f"synthetic_segment: spq_now={spq_now.as_fraction()}")
        if spq_now.is_zero() or q_threshold.is_zero():
            return None
        if not (spq_now >= q_threshold):
            return None

        # Compute fractions
        spq_now_frac = spq_now.as_fraction()
        q_threshold_frac = q_threshold.as_fraction()
        _dbg(f"synthetic_segment: q_threshold={q_threshold_frac}")

        # OUT cap in units (optional)
        cap_units = None
        if max_out_cap is not None and ((isinstance(max_out_cap, XRPAmount) and max_out_cap.value != 0) or (isinstance(max_out_cap, IOUAmount) and max_out_cap.mantissa != 0)):
            cap_units = self._amt_to_units_floor(max_out_cap)
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
            ok_quality = (q_slice.as_fraction() >= q_threshold_frac)
            # Average slice quality must not exceed current SPQ (integer-domain cap)
            if q_slice.as_fraction() > spq_now_frac:
                ok_quality = False
            return dx_gross_units, ok_quality, q_slice

        # Helper: SPQ after applying (dx_gross_units, dy_net_units) must return next SPQ and window result
        def spq_after_eval(dx_gross_units: int, dy_net_units: int) -> tuple[bool, Fraction]:
            """Return (feasible, spq_next_frac). Feasible=false if drains or invalid."""
            # Pool-side deltas
            dx_to_pool_units = self._mul_keep_ceil(dx_gross_units, self._keep_tr_in_num, self._tr_in_den)
            if self._tr_out_num > 0:
                dy_from_pool_units = self._ceil_div(dy_net_units * self._tr_out_den, self._keep_tr_out_num)
            else:
                dy_from_pool_units = dy_net_units
            if dy_from_pool_units >= y_u:
                return False, Fraction(0, 1)
            x_new = x_u + dx_to_pool_units
            y_new = y_u - dy_from_pool_units
            if x_new <= 0 or y_new <= 0:
                return False, Fraction(0, 1)
            # next SPQ as rational: (y_new*keep_fee_num) / (x_new*fee_den)
            spq_next_frac = _spq_frac_from_units(x_new, y_new, self._keep_fee_num, self._fee_den)
            return True, spq_next_frac

        # Binary search on dy_net_units in [1, hi]; maintain best under and best overshoot
        hi = y_u - 1
        if cap_units is not None and cap_units < hi:
            hi = cap_units
        if hi <= 0:
            return None

        best_under = None  # tuple(dy, dx, q_slice, spq_next_frac)
        best_over = None   # tuple(dy, dx, q_slice, spq_next_frac, gap)

        lo = 1
        # Upward-biased bisection-like probing (monotone in dy):
        for _ in range(80):
            if lo > hi:
                break
            mid = (lo + hi + 1) // 2
            dx_mid, ok_q, q_mid = compute_dx_for_dy(mid)
            if not (ok_q and dx_mid > 0):
                hi = mid - 1
                continue
            feasible, spq_next_rate = spq_after_eval(dx_mid, mid)
            if not feasible:
                hi = mid - 1
                continue
            if spq_next_rate <= q_threshold_frac:
                # Candidate lands at/below threshold; keep the one with the **largest** SPQ_next (closest under)
                if (best_under is None) or (spq_next_rate > best_under[3]):
                    best_under = (mid, dx_mid, q_mid, spq_next_rate)
                    _dbg(f"candidate UNDER mid={mid}: spq_next={spq_next_rate}")
                # Try smaller dy to get closer to the boundary from below
                hi = mid - 1
            else:
                # Above threshold: track smallest overshoot gap and try larger dy to push SPQ down
                gap = spq_next_rate - q_threshold_frac
                if (best_over is None) or (gap < best_over[4]):
                    best_over = (mid, dx_mid, q_mid, spq_next_rate, gap)
                _dbg(f"candidate OVER mid={mid}: gap={gap}")
                lo = mid + 1

        _dbg(f"synthetic_segment: search done; best_under={best_under is not None}, best_over={best_over is not None}")
        # Choose best candidate: prefer under; otherwise smallest overshoot
        if best_under is not None:
            best_dy, best_dx, best_q, spq_next_sel = best_under
        elif best_over is not None:
            best_dy, best_dx, best_q, spq_next_sel, _ = best_over
        else:
            return None

        _dbg(f"synthetic_segment: choose dy={best_dy} dx={best_dx} spq_next_sel={spq_next_sel}")

        # Construct integer-domain segment from the selected candidate
        out_st = self._from_units(best_dy, self.y_is_xrp)
        in_st = self._from_units(best_dx, self.x_is_xrp)
        q_slice = Quality.from_amounts(out_st, in_st)
        # Average quality must be at least as good as threshold; allow equality
        if not (q_slice.as_fraction() >= q_threshold_frac):
            return None
        # Slice average must not exceed current SPQ (integer-domain)
        if q_slice.as_fraction() > spq_now_frac:
            raise AssertionError(
                f"Selected AMM slice quality exceeds current SPQ: slice={q_slice.as_fraction()}, spq={spq_now_frac}")
        # Post-trade SPQ should be monotone non-improving and as close to threshold as grid allows
        _ok, spq_next_frac = spq_after_eval(self._amt_to_units_floor(in_st), self._amt_to_units_floor(out_st))
        if not _ok:
            raise AssertionError("Post-trade SPQ evaluation failed (drain or invalid)")
        # Anti-dust: require at least 1 unit on both sides
        if self._amt_to_units_floor(out_st) <= 0 or self._amt_to_units_floor(in_st) <= 0:
            return None
        _dbg(f"synthetic_segment: q_slice={q_slice.as_fraction()} out={out_st} in={in_st}")
        return Segment(
            src="AMM",
            quality=q_slice,
            out_max=out_st,
            in_at_out_max=in_st,
        )

    # --- State update after a filled amount (for next-iteration anchoring) ---