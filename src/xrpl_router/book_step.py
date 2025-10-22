"""
BookStep — unified market step for CLOB/AMM consumption (whitepaper-aligned).

Status:
  Reverse-first, then forward if needed (sendmax/limiting step). Works uniformly for:
  - CLOB-only: consume top tier sequentially order-by-order.
  - AMM-only: synthesize a slice at SPQ and consume.
  - Mixed: if SPQ(AMM) > q_top(CLOB), take anchored AMM first, then CLOB top.
  AMM residuals are dropped between iterations. Forward staging writes AMM (dx, dy)
  into the sandbox; Flow.apply() commits on success. Fee hook is optional and a no-op by default.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, List
from typing import Any, Dict
from decimal import Decimal
from .core.datatypes import Segment
from .core import Amount, Quality, XRPAmount, IOUAmount
from .core.exc import InsufficientLiquidityError, InsufficientBudgetError
from .core.fmt import amount_to_decimal
from .amm import AMM
# --- Zero-like helper for Amount types ---
def _zero_like(a: Amount) -> Amount:
    return XRPAmount(0) if isinstance(a, XRPAmount) else IOUAmount.zero()
#

# --- Integer helpers for proportional allocation ---
def _ten_pow(n: int) -> int:
    return 10 ** n

def _units_on_grid(amount: Amount, target_exp: int) -> int:
    """Return integer number of quanta (10^target_exp) contained in amount.
    XRPAmount only supports target_exp==0; IOUAmount supports any integer exponent grid.
    """
    if isinstance(amount, XRPAmount):
        assert target_exp == 0, "XRP grid is drops (10^0)"
        return amount.value
    # IOUAmount path
    shift = amount.exponent - target_exp
    if shift < 0:
        return 0
    return amount.mantissa * _ten_pow(shift)

# Helper to construct Amount from integer units
def _from_units(units: int, target_exp: int, is_xrp: bool) -> Amount:
    if units <= 0:
        return XRPAmount(0) if is_xrp else IOUAmount.zero()
    if is_xrp:
        assert target_exp == 0, "XRP uses drops grid (10^0)"
        return XRPAmount(units)
    return IOUAmount.from_components(units, target_exp)

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
    # Sort by quality rate (higher is better), tie-break: CLOB before AMM (CLOB wins ties under reverse=True)
    return sorted(
        seq,
        key=lambda s: (
            s.quality.rate.exponent,  # primary: exponent
            s.quality.rate.mantissa,  # secondary: mantissa
            1 if s.src != "AMM" else 0,  # CLOB wins ties (CLOB gets 1, AMM gets 0; so CLOB comes first in reverse=True)
        ),
        reverse=True,
    )

def default_fee_hook(trace: List[Dict[str, Any]], sandbox: "PaymentSandbox") -> None:
    """Demonstration-only fee hook (no-op). Integer domain is non-negative."""
    return
def _execute_one_iteration(
    segs: List[Segment],
    *,
    target_out: Amount,
    send_max: Optional[Amount],
    limit_quality: Optional[Quality],
    amm_anchor: Optional[Callable[[Quality, Amount], Optional[Segment]]],
    amm_spq: Optional[Quality],
    amm_obj: Optional[AMM],
) -> Tuple[Amount, Amount, List[Segment], bool, Optional[Quality], Amount, Amount, List[Dict[str, Any]]]:
    """Execute a single iteration (integer domain).

    Returns:
    (filled_out, spent_in, next_segments, amm_used, tier_quality, amm_dx, amm_dy, trace)
    """
    trace: List[Dict[str, Any]] = []
    amm_dx = _zero_like(target_out)
    amm_dy = _zero_like(target_out)
    # We may still trade via AMM-only even if there are no CLOB segments.
    no_segs = not segs

    # Apply quality floor
    if limit_quality is not None:
        segs = [s for s in segs if s.quality.rate >= limit_quality.rate]
        if not segs:
            # If CLOB is filtered out but AMM SPQ meets the floor, allow AMM-only
            if amm_anchor and (amm_spq is not None) and (amm_spq.rate >= limit_quality.rate) and (not target_out.is_zero()):
                pass  # proceed; AMM-only synthesis below will handle it
            else:
                z = _zero_like(target_out)
                return z, z, [], False, None, amm_dx, amm_dy, trace

    # Determine CLOB top quality (None if no CLOB in this batch)
    q_lob_top: Optional[Quality] = None
    for s in segs:
        if s.src != "AMM":
            if q_lob_top is None or s.quality.rate > q_lob_top.rate:
                q_lob_top = s.quality

    # Try to synthesize an AMM slice anchored to LOB top; do NOT mix it into LOB tier
    synth: Optional[Segment] = None
    if q_lob_top is not None and amm_anchor:
        try:
            synth = amm_anchor(q_lob_top, target_out)
        except Exception:
            synth = None
            
    # If AMM SPQ is strictly better than LOB top quality, we expect an anchored slice
    if (q_lob_top is not None and amm_anchor and (amm_spq is not None)
            and (amm_spq.rate > q_lob_top.rate) and (not target_out.is_zero())):
        assert synth is not None, "Expected anchored AMM slice when SPQ > LOB top; _amm_anchor returned None"

    # AMM slices bundle for AMM-only shadow slicing
    amm_slices: List[Segment] = []

    # AMM-only mode: if no CLOB top, synthesize a single integer-domain slice up to the target
    if q_lob_top is None and amm_obj is not None and (not target_out.is_zero()):
        try:
            dx_needed, capped = amm_obj.dx_for_out_st(target_out, raise_on_overask=False)
            if dx_needed is not None and (not dx_needed.is_zero()):
                out_amt = target_out if not capped else amm_obj.max_out_net_cap_st()
                if out_amt is not None and (not out_amt.is_zero()):
                    q_slice = Quality.from_amounts(out_amt, dx_needed)
                    amm_slices.append(Segment(src="AMM", out_max=out_amt, in_at_out_max=dx_needed, quality=q_slice))
        except Exception:
            amm_slices = []

    # Phase totals
    total_out = _zero_like(target_out)
    if not no_segs:
        total_in = _zero_like(segs[0].in_at_out_max)
    elif synth is not None:
        total_in = _zero_like(synth.in_at_out_max)
    else:
        total_in = _zero_like(target_out)

    # ----------------------
    # Phase A1: AMM first (anchored slice or AMM-only bundle on shadow state)
    # ----------------------
    if (synth is not None or amm_slices) and (not target_out.is_zero()):
        def _take_from_segment(seg: Segment, remaining: Amount) -> Tuple[Amount, Amount]:
            # Determine OUT grid per segment to avoid quantising capacity to zero
            is_xrp_out_local = isinstance(remaining, XRPAmount)
            grid_exp_out_local = 0 if is_xrp_out_local else min(seg.out_max.exponent, remaining.exponent)
            cap_units = _units_on_grid(seg.out_max, grid_exp_out_local)
            need_units = _units_on_grid(remaining, grid_exp_out_local)
            # Guard: capacity should not quantise to zero when remaining > 0
            if need_units > 0:
                assert cap_units > 0, "AMM slice capacity quantised to zero — check grid alignment"
            take_units = min(cap_units, need_units)
            if take_units <= 0:
                return _zero_like(remaining), _zero_like(remaining)
            out_take = _from_units(take_units, grid_exp_out_local, is_xrp_out_local)
            in_is_xrp = isinstance(seg.in_at_out_max, XRPAmount)
            grid_exp_in = 0 if in_is_xrp else seg.in_at_out_max.exponent
            in_cap_units = _units_on_grid(seg.in_at_out_max, grid_exp_in)
            # ceil proportion for IN
            units_in_est = (take_units * in_cap_units + cap_units - 1) // cap_units
            in_amt = _from_units(units_in_est, grid_exp_in, in_is_xrp)
            return out_take, in_amt

        # Single anchored slice
        if synth is not None:
            out_take_amm, in_amm = _take_from_segment(synth, target_out)
            if (not out_take_amm.is_zero()) and (not in_amm.is_zero()):
                total_out = total_out + out_take_amm
                total_in = total_in + in_amm
                amm_dx = amm_dx + in_amm
                amm_dy = amm_dy + out_take_amm
                trace.append({"src": "AMM", "take_out": out_take_amm, "take_in": in_amm, "quality": synth.quality})
                target_out = target_out - out_take_amm
        # AMM-only bundle (shadow-sliced): consume slices in order
        else:
            for seg in amm_slices:
                if target_out.is_zero():
                    break
                out_take_amm, in_amm = _take_from_segment(seg, target_out)
                if (out_take_amm.is_zero()) or (in_amm.is_zero()):
                    continue
                total_out = total_out + out_take_amm
                total_in = total_in + in_amm
                amm_dx = amm_dx + in_amm
                amm_dy = amm_dy + out_take_amm
                trace.append({"src": "AMM", "take_out": out_take_amm, "take_in": in_amm, "quality": seg.quality})
                target_out = target_out - out_take_amm

    # ----------------------
    # Phase A2: LOB top tier (CLOB only, same quality as q_lob_top)
    # ----------------------
    # If there is no CLOB at all, skip to finalization
    lob_tier: List[Segment] = []
    rest: List[Segment] = []
    tier_quality: Optional[Quality] = None

    if q_lob_top is not None and (not target_out.is_zero()):
        # Build LOB top tier (exclude AMM)
        for s in segs:
            if s.src != "AMM" and s.quality.rate == q_lob_top.rate:
                lob_tier.append(s)
            elif s.src != "AMM":
                rest.append(s)
        tier_quality = q_lob_top

        if lob_tier:
            # Sequentially consume top-tier LOB orders: highest quality first, order as provided
            next_segs: List[Segment] = []
            remaining_target = target_out
            budget_left = None
            if send_max is not None and not send_max.is_zero():
                budget_left = send_max - total_in

            for s in lob_tier:
                if remaining_target.is_zero():
                    # Preserve full residual for untouched orders
                    if (not s.out_max.is_zero()) and (not s.in_at_out_max.is_zero()):
                        next_segs.append(s)
                    continue

                # Determine OUT grid for this segment (per-segment grid)
                is_xrp_out = isinstance(remaining_target, XRPAmount)
                grid_exp = 0 if is_xrp_out else min(s.out_max.exponent, remaining_target.exponent)
                cap_units = _units_on_grid(s.out_max, grid_exp)
                need_units = _units_on_grid(remaining_target, grid_exp)
                if cap_units <= 0 or need_units <= 0:
                    # Keep as residual
                    if (not s.out_max.is_zero()) and (not s.in_at_out_max.is_zero()):
                        next_segs.append(s)
                    continue

                take_units = min(cap_units, need_units)

                # Compute IN estimate on this segment's IN grid (ceil proportion)
                in_is_xrp = isinstance(s.in_at_out_max, XRPAmount)
                in_grid_exp = 0 if in_is_xrp else s.in_at_out_max.exponent
                in_max_units = _units_on_grid(s.in_at_out_max, in_grid_exp)
                if in_max_units <= 0:
                    # cannot take from this segment; keep as residual
                    if (not s.out_max.is_zero()) and (not s.in_at_out_max.is_zero()):
                        next_segs.append(s)
                    continue

                # Apply send_max budget only to CLOB portion (AMM already consumed in Phase A1)
                if budget_left is not None and (not budget_left.is_zero()):
                    budget_units = _units_on_grid(budget_left, in_grid_exp)
                    # Ensure ceil(take * in_max / cap) <= budget_units  => take <= floor(budget_units * cap / in_max)
                    max_units_by_budget = (budget_units * cap_units) // in_max_units
                    if max_units_by_budget < take_units:
                        take_units = max_units_by_budget

                if take_units <= 0:
                    # Out of budget or no capacity
                    if (not s.out_max.is_zero()) and (not s.in_at_out_max.is_zero()):
                        next_segs.append(s)
                    break

                out_take = _from_units(take_units, grid_exp, is_xrp_out)
                units_in_est = (take_units * in_max_units + cap_units - 1) // cap_units
                in_amt = _from_units(units_in_est, in_grid_exp, in_is_xrp)

                # Accumulate fills
                total_out = total_out + out_take
                total_in = total_in + in_amt
                trace.append({"src": s.src, "take_out": out_take, "take_in": in_amt, "quality": s.quality})

                # Update remaining target and budget
                remaining_target = remaining_target - out_take
                if budget_left is not None:
                    budget_left = budget_left - in_amt

                # Compute residual for this segment
                rem_out = s.out_max - out_take
                rem_in = s.in_at_out_max - in_amt
                if (not rem_out.is_zero()) and (not rem_in.is_zero()):
                    next_segs.append(Segment(src=s.src, out_max=rem_out, in_at_out_max=rem_in, quality=s.quality))

                # Stop if target satisfied or budget exhausted
                if remaining_target.is_zero() or (budget_left is not None and budget_left.is_zero()):
                    # Append untouched rest (non-top LOB)
                    next_segs.extend([x for x in rest if (not x.out_max.is_zero()) and (not x.in_at_out_max.is_zero())])
                    return total_out, total_in, next_segs, (not amm_dx.is_zero() or not amm_dy.is_zero()), tier_quality, amm_dx, amm_dy, trace

            # Finished iterating top tier; append rest levels
            next_segs.extend([x for x in rest if (not x.out_max.is_zero()) and (not x.in_at_out_max.is_zero())])
            return total_out, total_in, next_segs, (not amm_dx.is_zero() or not amm_dy.is_zero()), tier_quality, amm_dx, amm_dy, trace

    # If no LOB or nothing more to take from LOB, finalize with whatever we consumed (AMM-only or none)
    next_segs = [] if no_segs else [s for s in segs if s.src != "AMM" and (not s.out_max.is_zero()) and (not s.in_at_out_max.is_zero())]
    return total_out, total_in, next_segs, (not amm_dx.is_zero() or not amm_dy.is_zero()), q_lob_top, amm_dx, amm_dy, trace

from .steps import Step


@dataclass
class BookStep(Step):
    """Unified market step for CLOB/AMM consumption (transitional implementation).

    Forward execution also stages AMM pool deltas (dx, dy) into the sandbox; Flow.apply()
    later commits them via apply_sink. The default_fee_hook stages CLOB out-side issuer fees into the sandbox; rev remains read-only.
    """
    segments_provider: Callable[[], Iterable[Segment]]
    amm: Optional[AMM] = None
    # route_config: Optional[RouteConfig] = None
    limit_quality: Optional[Quality] = None
    fee_hook: Optional[Callable[[List[Dict[str, Any]], "PaymentSandbox"], None]] = None

    @classmethod
    def from_static(
        cls,
        segs: List[Segment],
        *,
        amm: Optional[AMM] = None,
        limit_quality: Optional[Quality] = None,
        fee_hook: Optional[Callable[[List[Dict[str, Any]], "PaymentSandbox"], None]] = None,
    ) -> "BookStep":
        """Convenience factory: wrap static CLOB segments into a provider.

        Tests and simple callers can supply a fixed list of segments while keeping the
        production signature intact.
        """
        def _provider() -> Iterable[Segment]:
            return segs
        return cls(segments_provider=_provider, amm=amm, limit_quality=limit_quality, fee_hook=fee_hook)

    def _amm_anchor(self, q: Quality, cap: Amount) -> Optional[Segment]:
        """Synthesize an AMM segment anchored at quality `q` up to `cap` OUT.

        If the AMM requires the anchor to be strictly below its SPQ and `q` equals SPQ,
        relax the threshold by one ULP on the quality rate to allow an infinitesimal-better slice.
        This keeps the behaviour "anchored at SPQ" in AMM-only without inventing non-anchored slices.
        """
        if self.amm is None:
            return None
        # Primary: anchored by quality threshold (as-is)
        try:
            seg = self.amm.synthetic_segment_for_quality(q, max_out_cap=cap)
            if seg is not None and (not seg.out_max.is_zero()) and (not seg.in_at_out_max.is_zero()):
                return seg
        except Exception:
            pass
        # Secondary: relax the anchor by 1 ULP on the quality rate (same exponent, mantissa-1)
        try:
            r = q.rate
            m2 = r.mantissa - 1 if r.mantissa > 1 else r.mantissa
            q_relaxed = Quality(rate=IOUAmount.from_components(m2, r.exponent))
            seg2 = self.amm.synthetic_segment_for_quality(q_relaxed, max_out_cap=cap)
            if seg2 is not None and (not seg2.out_max.is_zero()) and (not seg2.in_at_out_max.is_zero()):
                return seg2
        except Exception:
            pass
        return None

    def rev(
        self,
        sandbox: "PaymentSandbox",
        out_req: Amount,
        return_trace: bool = False,
        require_full_fill: bool = False,
    ) -> Tuple[Amount, Amount] | Tuple[Amount, Amount, List[Dict[str, Any]]]:
        if out_req is None or out_req.is_zero():
            z = _zero_like(out_req if out_req is not None else XRPAmount(0))
            self._cached_in = z
            self._cached_out = z
            if return_trace:
                return self._cached_out, self._cached_in, []
            return self._cached_out, self._cached_in
        try:
            segs = _prefer_clob_on_ties(self.segments_provider())
        except Exception:
            segs = []
        # Prepare a shadow AMM for reverse routing so AMM state advances between iterations
        shadow_amm = self.amm.clone() if self.amm is not None else None
        # --- AMM-only mode: true if no segs and AMM present
        amm_only_mode = (not segs) and (shadow_amm is not None)
        need = out_req
        total_in = _zero_like(out_req)
        total_out = _zero_like(out_req)
        all_trace: List[Dict[str, Any]] = []
        prev_amm_q: Optional[Quality] = None
        for _ in range(64):
            if need.is_zero():
                break
            # Bind anchoring to the shadow AMM so its evolving state is used
            anchor_fn = (lambda q, cap: shadow_amm.synthetic_segment_for_quality(q, max_out_cap=cap)) if shadow_amm is not None else None
            spq_now = shadow_amm.spq_quality_int() if shadow_amm is not None else None
            filled, spent, next_segs, amm_used, _, amm_dx, amm_dy, itr_trace = _execute_one_iteration(
                segs,
                target_out=need,
                send_max=None,
                limit_quality=self.limit_quality,
                amm_anchor=anchor_fn,
                amm_spq=spq_now,
                amm_obj=shadow_amm,
            )
            if itr_trace:
                # Enforce AMM marginal quality non-increasing within this rev()
                for t in itr_trace:
                    if t.get("src") == "AMM":
                        q = t["quality"]
                        if prev_amm_q is not None:
                            assert q <= prev_amm_q, f"AMM quality increased within a single rev(): prev={prev_amm_q.rate} curr={q.rate}"
                        prev_amm_q = q
                all_trace.extend(itr_trace)
            if filled.is_zero():
                break
            total_in = spent if total_in.is_zero() else (total_in + spent)
            total_out = filled if total_out.is_zero() else (total_out + filled)
            need = out_req - total_out
            segs = next_segs
            # Apply AMM deltas to the shadow AMM so the next iteration sees updated reserves
            if amm_used and (shadow_amm is not None) and (not amm_dx.is_zero() or not amm_dy.is_zero()):
                shadow_amm.apply_fill_st(amm_dx, amm_dy)
            # In AMM-only reverse, produce at most one slice to avoid repeated synthetic segments
            if amm_only_mode:
                break
        self._cached_out = total_out
        self._cached_in = total_in
        # Strict correctness: if requested more than we could fill, raise with context
        if require_full_fill and (total_out < out_req):
            # Provide context via exception, including the trace we collected so far
            raise InsufficientLiquidityError(
                requested_out=out_req,
                max_fill_out=total_out,
                filled_out=total_out,
                spent_in=total_in,
                trace=all_trace if return_trace else None,
            )
        # In non-strict mode, annotate partial in the trace for observability
        if (not require_full_fill) and return_trace and (total_out < out_req):
            try:
                req_dec = amount_to_decimal(out_req)
                filled_dec = amount_to_decimal(total_out)
                fill_ratio = float(filled_dec / req_dec) if req_dec != 0 else 0.0
                all_trace.append({
                    "status": "PARTIAL",
                    "requested": req_dec,
                    "filled": filled_dec,
                    "fill_ratio": fill_ratio,
                })
            except Exception:
                # Best-effort; tracing must not break execution
                all_trace.append({"status": "PARTIAL"})
        if return_trace:
            return self._cached_out, self._cached_in, all_trace
        return self._cached_out, self._cached_in

    def fwd(
        self,
        sandbox: "PaymentSandbox",
        in_cap: Amount,
        return_trace: bool = False,
    ) -> Tuple[Amount, Amount] | Tuple[Amount, Amount, List[Dict[str, Any]]]:
        if in_cap is None or in_cap.is_zero():
            z = _zero_like(in_cap if in_cap is not None else XRPAmount(0))
            self._cached_in = z
            self._cached_out = z
            if return_trace:
                return self._cached_out, self._cached_in, []
            return self._cached_out, self._cached_in
        try:
            segs = _prefer_clob_on_ties(self.segments_provider())
        except Exception:
            segs = []
        remaining_in = in_cap
        total_in = _zero_like(in_cap)
        total_out = _zero_like(in_cap)
        all_trace: List[Dict[str, Any]] = []
        # Prefer the reverse cache to cap forward OUT so we don't overshoot requested amount
        target_out = self._cached_out if hasattr(self, "_cached_out") and (not getattr(self, "_cached_out").is_zero()) else _zero_like(in_cap)
        if target_out.is_zero():
            for s in segs:
                target_out = target_out + s.out_max
        # AMM-only fallback: if no CLOB capacity and we have an AMM, estimate a forward OUT cap from in_cap
        if target_out.is_zero() and self.amm is not None:
            try:
                est_out = self.amm.swap_out_given_in_st(in_cap)
                if est_out is not None and (not est_out.is_zero()):
                    target_out = est_out
            except Exception:
                pass
        if target_out.is_zero():
            z = _zero_like(in_cap)
            self._cached_in = z
            self._cached_out = z
            if return_trace:
                return self._cached_out, self._cached_in, []
            return self._cached_out, self._cached_in
        for _ in range(64):
            if remaining_in.is_zero() or target_out.is_zero():
                break
            filled, spent, next_segs, amm_used, _, amm_dx, amm_dy, itr_trace = _execute_one_iteration(
                segs,
                target_out=target_out,
                send_max=remaining_in,
                limit_quality=self.limit_quality,
                amm_anchor=(self._amm_anchor if self.amm is not None else None),
                amm_spq=(self.amm.spq_quality_int() if self.amm is not None else None),
                amm_obj=self.amm,
            )
            if itr_trace:
                all_trace.extend(itr_trace)
            if spent.is_zero():
                break
            if total_in.is_zero():
                total_in = spent
            else:
                total_in = total_in + spent
            if total_out.is_zero():
                total_out = filled
            else:
                total_out = total_out + filled
            remaining_in = in_cap - total_in
            segs = next_segs
            # Decrease the remaining OUT target; do not exceed requested OUT from reverse stage
            if not target_out.is_zero():
                target_out = target_out - filled
            else:
                # Fallback: if no explicit cap, sum remaining capacities
                target_out = _zero_like(target_out)
                for s in segs:
                    target_out = target_out + s.out_max
            if amm_used and (not amm_dx.is_zero() or not amm_dy.is_zero()) and sandbox is not None:
                try:
                    # Stage AMM deltas: dx added to pool, dy removed from pool (non-negative on out side)
                    sandbox.stage_after_iteration(amm_dx, amm_dy)
                except Exception:
                    pass
            hook = self.fee_hook
            if sandbox is not None and hook is not None:
                try:
                    hook(itr_trace, sandbox)
                except Exception:
                    pass
        self._cached_in = total_in
        self._cached_out = total_out
        if return_trace:
            return self._cached_out, self._cached_in, all_trace
        return self._cached_out, self._cached_in

    def quality_upper_bound(self) -> Quality:
        try:
            segs = list(self.segments_provider())
        except Exception:
            return Quality.from_amounts(XRPAmount(0), IOUAmount.from_components(1, 0))
        if not segs:
            return Quality.from_amounts(XRPAmount(0), IOUAmount.from_components(1, 0))
        best = max(s.quality for s in segs)
        return best