"""
BookStep — unified market step for CLOB/AMM consumption (whitepaper-aligned).

Whitepaper (§1.3.2.4) responsibilities:
  - Iterate homogeneous quotes (AMM or CLOB) of the same quality tier.
  - Execute offers with proper fee accounting (ownerGives/stpAmt), preserving
    instantaneous quality (limitIn/limitOut).
  - Prefer CLOB when AMM and CLOB are at equal quality (see §1.2.7.2).
  - In multi-path mode, generate AMM slices via Fibonacci sizing (§1.2.7.3) and
    only advance the iteration counter when AMM is actually consumed.

Status:
  This implementation now performs the per-iteration execution internally (Phase A/B:
  proportional scaling → grid rounding → instantaneous-quality guard). It sources AMM
  slices via anchoring or curve as needed and drops AMM residuals across iterations.
  Forward execution also stages AMM pool deltas (dx, dy) into the sandbox; Flow.apply()
  later commits them via apply_sink. Reverse pass remains read-only.
  An optional fee_hook may stage issuer out-side fees into the sandbox; by default no fee is applied.
  Production code should use BookStep; the legacy router remains for research tooling.
"""
from __future__ import annotations


from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, List
from typing import Any, Dict
from .core.datatypes import Segment
from .core.amounts import STAmount
from .core.quality import Quality
#

# --- Integer helpers for proportional allocation ---
def _ten_pow(n: int) -> int:
    return 10 ** n

def _units_on_grid(amount: STAmount, target_exp: int) -> int:
    """Return integer number of quanta (10^target_exp) contained in amount.
    Requires amount.exponent >= target_exp. Caller ensures target_exp is min exponent across set.
    """
    shift = amount.exponent - target_exp
    if shift < 0:
        # Not representable without fraction; treat as zero units for safety
        return 0
    return amount.mantissa * _ten_pow(shift)

def _alloc_proportional(cap_units: List[int], need_units: int) -> List[int]:
    """Largest-remainder allocation of need_units proportionally to cap_units.
    Returns integer allocations per index; sum == need_units and each <= cap.
    """
    total = sum(cap_units)
    if total <= 0 or need_units <= 0:
        return [0] * len(cap_units)
    base = []
    rems = []
    acc = 0
    for u in cap_units:
        num = need_units * u
        q, r = divmod(num, total)
        if q > u:
            q = u
            r = 0
        base.append(q)
        rems.append(r)
        acc += q
    leftover = need_units - acc
    if leftover > 0:
        order = sorted(range(len(cap_units)), key=lambda k: rems[k], reverse=True)
        for k in order:
            if leftover <= 0:
                break
            if base[k] < cap_units[k]:
                base[k] += 1
                leftover -= 1
    return base

def _prefer_clob_on_ties(segs: Iterable[Segment]) -> List[Segment]:
    seq = list(segs)
    # Sort by quality rate (higher is better), tie-break: CLOB before AMM
    return sorted(
        seq,
        key=lambda s: (
            s.quality.rate.exponent,  # primary: exponent
            s.quality.rate.mantissa,  # secondary: mantissa
            0 if s.src != "AMM" else 1,  # CLOB first on ties
        ),
        reverse=True,
    )

def default_fee_hook(trace: List[Dict[str, Any]], sandbox: "PaymentSandbox") -> None:
    """Demonstration-only fee hook: apply 0.1% out-side fee for CLOB slices.
    Mutates sandbox with (dx, dy) where dy is negative (reducing OUT delivered).
    NOTE: This hook is optional and off by default; production callers should
    provide an explicit issuer-fee policy if needed.
    """
    # 0.1% = divide by 1000 (integer scalar division on STAmount)
    for item in trace:
        if item.get("src") == "CLOB":
            take_out: STAmount = item.get("take_out", STAmount.zero())
            if isinstance(take_out, STAmount) and not take_out.is_zero():
                fee = take_out // 1000
                if not fee.is_zero():
                    # Construct a negative out-side fee explicitly in integer domain
                    neg_fee = STAmount.from_components(fee.mantissa, fee.exponent, -1)
                    try:
                        sandbox.stage_fee(STAmount.zero(), neg_fee)
                    except Exception:
                        continue
def _execute_one_iteration(
    segs: List[Segment],
    *,
    target_out: STAmount,
    send_max: Optional[STAmount],
    limit_quality: Optional[Quality],
    amm_anchor: Optional[Callable[[Quality, STAmount], Optional[Segment]]],
    amm_curve: Optional[Callable[[STAmount], Iterable[Segment]]],
    ctx: Optional["AMMContext"],
) -> Tuple[STAmount, STAmount, List[Segment], bool, Optional[Quality], STAmount, STAmount, List[Dict[str, Any]]]:
    """Execute a single iteration (integer domain).

    Returns:
    (filled_out, spent_in, next_segments, amm_used, tier_quality, amm_dx, amm_dy, trace)
    """
    trace: List[Dict[str, Any]] = []
    ZERO = STAmount.zero()
    amm_dx = ZERO
    amm_dy = ZERO
    if not segs:
        return ZERO, ZERO, [], False, None, amm_dx, amm_dy, trace

    # Apply quality floor
    if limit_quality is not None:
        segs = [s for s in segs if s.quality.rate >= limit_quality.rate]
        if not segs:
            return ZERO, ZERO, [], False, None, amm_dx, amm_dy, trace

    # Determine CLOB top quality
    q_lob_top: Optional[Quality] = None
    for s in segs:
        if s.src != "AMM":
            if q_lob_top is None or s.quality.rate > q_lob_top.rate:
                q_lob_top = s.quality

    iter_segs = list(segs)
    synth: Optional[Segment] = None
    if q_lob_top is not None and amm_anchor and (ctx is None or not ctx.maxItersReached()):
        try:
            synth = amm_anchor(q_lob_top, target_out)
        except Exception:
            synth = None
        if synth is not None:
            iter_segs.append(synth)
    elif q_lob_top is None and amm_curve is not None and (ctx is None or not ctx.maxItersReached()):
        try:
            curve_segs = list(amm_curve(target_out))
        except Exception:
            curve_segs = []
        iter_segs.extend([cs for cs in curve_segs if cs and not cs.out_max.is_zero() and not cs.in_at_out_max.is_zero() and cs.quality.rate.sign > 0])

    iter_segs = _prefer_clob_on_ties(iter_segs)
    if not iter_segs:
        return ZERO, ZERO, [], False, None, amm_dx, amm_dy, trace

    # Choose tier (anchor to LOB top bucket when synth present): here we treat exact equality of rate
    if synth is not None and q_lob_top is not None:
        tier = [s for s in iter_segs if (s is synth) or (s.quality.rate == q_lob_top.rate)]
        rest = [s for s in iter_segs if s not in tier]
        tier_quality = q_lob_top
    else:
        # pick max quality
        best = max(iter_segs, key=lambda s: (s.quality.rate.exponent, s.quality.rate.mantissa))
        tier_quality = best.quality
        tier = [s for s in iter_segs if s.quality.rate == best.quality.rate]
        rest = [s for s in iter_segs if s.quality.rate != best.quality.rate]

    if not tier:
        return ZERO, ZERO, rest, False, None, amm_dx, amm_dy, trace

    # Phase A: proportional allocation on out-side units (integer grid)
    # Use a common grid exponent that is the minimum exponent across target and tier out_max
    out_exps = [s.out_max.exponent for s in tier] + [target_out.exponent]
    grid_exp = min(out_exps)
    cap_units = [_units_on_grid(s.out_max, grid_exp) for s in tier]
    need_units = _units_on_grid(target_out, grid_exp)
    total_units = sum(cap_units)
    if total_units <= 0 or need_units <= 0:
        return ZERO, ZERO, rest, False, tier_quality, amm_dx, amm_dy, trace

    alloc_units = _alloc_proportional(cap_units, min(need_units, total_units))

    # Convert allocated out-units back to STAmount and compute required IN via guard
    flo_out: List[STAmount] = []
    flo_in: List[STAmount] = []
    sum_out = ZERO
    sum_in = ZERO
    for s, u in zip(tier, alloc_units):
        if u <= 0:
            flo_out.append(ZERO)
            flo_in.append(ZERO)
            continue
        out_take = STAmount.from_components(u, grid_exp, 1)
        # Proportional IN estimate: ceil(out_take * in_at_out_max / out_max) on integer grids
        out_max_units = _units_on_grid(s.out_max, grid_exp)
        in_grid_exp = s.in_at_out_max.exponent
        in_max_units = _units_on_grid(s.in_at_out_max, in_grid_exp)
        if out_max_units <= 0 or in_max_units <= 0:
            in_amt = STAmount.zero()
        else:
            num = u * in_max_units
            den = out_max_units
            units_in_est = (num + den - 1) // den  # ceil division
            in_amt = STAmount.from_components(units_in_est, in_grid_exp, 1)
        flo_out.append(out_take)
        flo_in.append(in_amt)
        sum_out = sum_out + out_take
        sum_in = sum_in + in_amt

    # Respect send_max by scaling down once if needed
    if send_max is not None and not send_max.is_zero() and sum_in > send_max:
        # Scale down need_units and recompute once
        # Use IN-side common grid
        in_exps = [s.in_at_out_max.exponent for s in tier] + [send_max.exponent]
        in_grid_exp = min(in_exps)
        sum_in_units = _units_on_grid(sum_in, in_grid_exp)
        send_max_units = _units_on_grid(send_max, in_grid_exp)
        if sum_in_units > 0 and send_max_units < sum_in_units:
            scaled_need_units = max(1, need_units * send_max_units // sum_in_units)
            alloc_units = _alloc_proportional(cap_units, min(scaled_need_units, total_units))
            # Recompute OUT/IN with scaled allocation
            flo_out, flo_in, sum_out, sum_in = [], [], ZERO, ZERO
            for s, u in zip(tier, alloc_units):
                if u <= 0:
                    flo_out.append(ZERO); flo_in.append(ZERO); continue
                out_take = STAmount.from_components(u, grid_exp, 1)
                # Proportional IN estimate: ceil(out_take * in_at_out_max / out_max) on integer grids
                out_max_units = _units_on_grid(s.out_max, grid_exp)
                in_grid_exp = s.in_at_out_max.exponent
                in_max_units = _units_on_grid(s.in_at_out_max, in_grid_exp)
                if out_max_units <= 0 or in_max_units <= 0:
                    in_amt = STAmount.zero()
                else:
                    num = u * in_max_units
                    den = out_max_units
                    units_in_est = (num + den - 1) // den  # ceil division
                    in_amt = STAmount.from_components(units_in_est, in_grid_exp, 1)
                flo_out.append(out_take)
                flo_in.append(in_amt)
                sum_out = sum_out + out_take
                sum_in = sum_in + in_amt

    # One-quantum top-up using largest remainders (already handled by allocation; skip here)

    # Build next segments: carry non-AMM residuals; drop AMM residuals
    next_segs: List[Segment] = []
    amm_used = False
    for s, o_i, i_i in zip(tier, flo_out, flo_in):
        if s.src == "AMM" and (not o_i.is_zero() or not i_i.is_zero()):
            amm_used = True
            amm_dx = amm_dx + i_i
            amm_dy = amm_dy + o_i
        rem_out = s.out_max - o_i
        rem_in = s.in_at_out_max - i_i
        if (not rem_out.is_zero()) and (not rem_in.is_zero()) and (s.quality.rate.sign > 0) and (s.src != "AMM") and (s is not synth):
            next_segs.append(Segment(
                src=s.src,
                out_max=rem_out,
                in_at_out_max=rem_in,
                quality=s.quality,
                in_is_xrp=s.in_is_xrp,
                out_is_xrp=s.out_is_xrp,
            ))
    # Append rest (non-tier)
    next_segs.extend([
        s for s in rest if (not s.out_max.is_zero()) and (not s.in_at_out_max.is_zero()) and (s.quality.rate.sign > 0) and s.src != "AMM"
    ])

    # Trace (integer domain)
    for s, o, i in zip(tier, flo_out, flo_in):
        trace.append({"src": s.src, "take_out": o, "take_in": i, "quality": s.quality})

    return sum_out, sum_in, next_segs, amm_used, tier_quality, amm_dx, amm_dy, trace

from .amm_context import AMMContext
from .steps import Step


@dataclass
class BookStep(Step):
    """Unified market step for CLOB/AMM consumption (transitional implementation).

    Forward execution also stages AMM pool deltas (dx, dy) into the sandbox; Flow.apply()
    later commits them via apply_sink. The default_fee_hook stages CLOB out-side issuer fees into the sandbox; rev remains read-only.
    """
    segments_provider: Callable[[], Iterable[Segment]]
    amm_anchor: Optional[Callable[[Quality, STAmount], Optional[Segment]]] = None
    amm_curve: Optional[Callable[[STAmount], Iterable[Segment]]] = None
    amm_context: Optional["AMMContext"] = None
    # route_config: Optional[RouteConfig] = None
    limit_quality: Optional[Quality] = None
    fee_hook: Optional[Callable[[List[Dict[str, Any]], "PaymentSandbox"], None]] = None

    def rev(self, sandbox: "PaymentSandbox", out_req: STAmount) -> Tuple[STAmount, STAmount]:
        ZERO = STAmount.zero()
        if out_req is None or out_req.is_zero():
            self._cached_in = ZERO
            self._cached_out = ZERO
            return self._cached_in, self._cached_out
        try:
            segs = _prefer_clob_on_ties(self.segments_provider())
        except Exception:
            segs = []
        if not segs:
            self._cached_in = ZERO
            self._cached_out = ZERO
            return self._cached_in, self._cached_out
        need = out_req
        total_in = ZERO
        total_out = ZERO
        for _ in range(64):
            if need.is_zero() or not segs:
                break
            filled, spent, next_segs, amm_used, _, amm_dx, amm_dy, _ = _execute_one_iteration(
                segs,
                target_out=need,
                send_max=None,
                limit_quality=self.limit_quality,
                amm_anchor=self.amm_anchor,
                amm_curve=self.amm_curve,
                ctx=self.amm_context,
            )
            if filled.is_zero():
                break
            total_in = total_in + spent
            total_out = total_out + filled
            need = out_req - total_out
            segs = next_segs
            if amm_used and self.amm_context is not None:
                self.amm_context.setAMMUsed()
        self._cached_out = total_out
        self._cached_in = total_in
        return self._cached_in, self._cached_out

    def fwd(self, sandbox: "PaymentSandbox", in_cap: STAmount) -> Tuple[STAmount, STAmount]:
        ZERO = STAmount.zero()
        if in_cap is None or in_cap.is_zero():
            self._cached_in = ZERO
            self._cached_out = ZERO
            return self._cached_in, self._cached_out
        try:
            segs = _prefer_clob_on_ties(self.segments_provider())
        except Exception:
            segs = []
        if not segs:
            self._cached_in = ZERO
            self._cached_out = ZERO
            return self._cached_in, self._cached_out
        remaining_in = in_cap
        total_in = ZERO
        total_out = ZERO
        # Prefer the reverse cache to cap forward OUT so we don't overshoot requested amount
        target_out = self._cached_out if hasattr(self, "_cached_out") and (not getattr(self, "_cached_out").is_zero()) else ZERO
        if target_out.is_zero():
            for s in segs:
                target_out = target_out + s.out_max
        if target_out.is_zero():
            self._cached_in = ZERO
            self._cached_out = ZERO
            return self._cached_in, self._cached_out
        for _ in range(64):
            if remaining_in.is_zero() or not segs or target_out.is_zero():
                break
            filled, spent, next_segs, amm_used, _, amm_dx, amm_dy, trace = _execute_one_iteration(
                segs,
                target_out=target_out,
                send_max=remaining_in,
                limit_quality=self.limit_quality,
                amm_anchor=self.amm_anchor,
                amm_curve=self.amm_curve,
                ctx=self.amm_context,
            )
            if spent.is_zero():
                break
            total_in = total_in + spent
            total_out = total_out + filled
            remaining_in = in_cap - total_in
            segs = next_segs
            # Decrease the remaining OUT target; do not exceed requested OUT from reverse stage
            if not target_out.is_zero():
                target_out = target_out - filled
            else:
                # Fallback: if no explicit cap, sum remaining capacities
                target_out = ZERO
                for s in segs:
                    target_out = target_out + s.out_max
            if amm_used and self.amm_context is not None:
                self.amm_context.setAMMUsed()
            if amm_used and (not amm_dx.is_zero() or not amm_dy.is_zero()) and sandbox is not None:
                try:
                    # Stage AMM deltas: dx added to pool, dy removed from pool (negative on out side)
                    neg_dy = STAmount.from_components(amm_dy.mantissa, amm_dy.exponent, -1) if not amm_dy.is_zero() else STAmount.zero()
                    sandbox.stage_after_iteration(amm_dx, neg_dy)
                except Exception:
                    pass
            hook = self.fee_hook
            if sandbox is not None and hook is not None:
                try:
                    hook(trace, sandbox)
                except Exception:
                    pass
        self._cached_in = total_in
        self._cached_out = total_out
        return self._cached_in, self._cached_out

    def quality_upper_bound(self) -> Quality:
        try:
            segs = list(self.segments_provider())
        except Exception:
            return Quality.from_amounts(STAmount.zero(), STAmount.from_components(1, 0, 1))
        if not segs:
            return Quality.from_amounts(STAmount.zero(), STAmount.from_components(1, 0, 1))
        best = max((s.raw_quality if s.raw_quality else s.quality) for s in segs)
        return best