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
from .core.exc import InsufficientLiquidityError, InsufficientBudgetError, AMMOverAsk
from .core.fmt import amount_to_decimal
from .amm import AMM
# --- Zero-like helper for Amount types ---
def _zero_like(a: Amount) -> Amount:
    return XRPAmount(0) if isinstance(a, XRPAmount) else IOUAmount.zero()

# Zero helpers bound to AMM x/y domains
def _zero_x_for_amm(amm: Optional["AMM"]) -> Amount:
    if amm is not None:
        return XRPAmount(0) if amm.x_is_xrp else IOUAmount.zero()
    # Fallback: IOU zero (won't be used if AMM is None)
    return IOUAmount.zero()

def _zero_y_for_amm(amm: Optional["AMM"]) -> Amount:
    if amm is not None:
        return XRPAmount(0) if amm.y_is_xrp else IOUAmount.zero()
    # Fallback: IOU zero (won't be used if AMM is None)
    return IOUAmount.zero()

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
    amm_dx = _zero_x_for_amm(amm_obj)
    amm_dy = _zero_y_for_amm(amm_obj)
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
            
        # If AMM SPQ is strictly better than LOB top quality, we *prefer* an anchored slice.
    # However, due to integer grid / fee rounding / cap, an anchored slice may be impossible
    # to synthesise even when SPQ > LOB top. This is not a fatal error: we must fall back
    # to LOB-only for this iteration (or AMM-only later if LOB exhausts), rather than abort.
    if (q_lob_top is not None and amm_anchor and (amm_spq is not None)
            and (amm_spq.rate > q_lob_top.rate) and (not target_out.is_zero())):
        if synth is None:
            trace.append({
                "src": "AMM",
                "event": "anchor_failed",
                "reason": "SPQ > LOB_top but _amm_anchor returned None (integer grid / rounding / cap)",
                "lob_top_quality": q_lob_top,
                "amm_spq": amm_spq,
                "target_out": target_out,
            })
            # Proceed without AMM-first in this iteration; LOB tier will execute below.

    # AMM slices bundle for AMM-only shadow slicing
    amm_slices: List[Segment] = []

    # AMM-only mode: if no CLOB top, synthesize a single integer-domain slice up to the target
    if q_lob_top is None and amm_obj is not None and (not target_out.is_zero()):
        try:
            dx_needed = amm_obj.dx_for_out_st(target_out)
            if dx_needed is not None and (not dx_needed.is_zero()):
                out_amt = target_out
                q_slice = Quality.from_amounts(out_amt, dx_needed)
                amm_slices.append(Segment(src="AMM", out_max=out_amt, in_at_out_max=dx_needed, quality=q_slice))
        except AMMOverAsk:
            # Over-ask: do not craft a partial AMM slice in this iteration
            pass
        except Exception:
            amm_slices = []

    # Phase totals
    total_out = _zero_like(target_out)
    if not no_segs:
        total_in = _zero_like(segs[0].in_at_out_max)
    elif synth is not None:
        total_in = _zero_like(synth.in_at_out_max)
    elif amm_obj is not None:
        total_in = _zero_x_for_amm(amm_obj)
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
                trace.append(
                    {
                        "src": "AMM",
                        "source_id": synth.source_id,
                        "take_out": out_take_amm,
                        "take_in": in_amm,
                        "quality": synth.quality,
                    }
                )
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
                trace.append(
                    {
                        "src": "AMM",
                        "source_id": seg.source_id,
                        "take_out": out_take_amm,
                        "take_in": in_amm,
                        "quality": seg.quality,
                    }
                )
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
                trace.append(
                    {
                        "src": s.src,
                        "source_id": s.source_id,
                        "take_out": out_take,
                        "take_in": in_amt,
                        "quality": s.quality,
                    }
                )

                # Update remaining target and budget
                remaining_target = remaining_target - out_take
                if budget_left is not None:
                    budget_left = budget_left - in_amt

                # Compute residual for this segment
                rem_out = s.out_max - out_take
                rem_in = s.in_at_out_max - in_amt
                if (not rem_out.is_zero()) and (not rem_in.is_zero()):
                    next_segs.append(
                        Segment(
                            src=s.src,
                            out_max=rem_out,
                            in_at_out_max=rem_in,
                            quality=s.quality,
                            raw_quality=s.raw_quality,
                            source_id=s.source_id,
                        )
                    )

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
        # Compute total capacity (CLOB + AMM)
        from .core.fmt import amount_to_decimal
        total_cap = None
        # Sum all CLOB out_max (excluding AMM)
        clob_cap = None
        for s in segs:
            if s.src != "AMM":
                if clob_cap is None:
                    clob_cap = _zero_like(s.out_max)
                clob_cap = clob_cap + s.out_max
        if clob_cap is None:
            clob_cap = _zero_like(out_req)
        total_cap = clob_cap
        # Add AMM non-drain cap if present
        amm_cap = None
        if self.amm is not None:
            # Compute AMM max out by asking for a huge out and catching AMMOverAsk
            try:
                huge_out = self.amm._from_units(self.amm._to_units_floor(self.amm.y, self.amm.y_is_xrp) + 1000, self.amm.y_is_xrp)
                _ = self.amm.dx_for_out_st(huge_out)
            except AMMOverAsk as e:
                amm_cap = e.max_out
            except Exception:
                amm_cap = None
        if amm_cap is not None:
            total_cap = total_cap + amm_cap
        # Compare out_req to total_cap
        req_dec = amount_to_decimal(out_req)
        cap_dec = amount_to_decimal(total_cap)
        if req_dec > cap_dec:
            raise InsufficientLiquidityError(
                requested_out=out_req,
                max_fill_out=total_cap,
                filled_out=_zero_like(out_req),
                spent_in=_zero_like(out_req),
                trace=None,
            )
        # Proceed with normal path
        need = out_req
        total_in = _zero_like(out_req)
        total_out = _zero_like(out_req)
        all_trace: List[Dict[str, Any]] = []
        prev_amm_q: Optional[Quality] = None
        for _ in range(64):
            if need.is_zero():
                break
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
        self._cached_out = total_out
        self._cached_in = total_in
        # If requested more than we could fill, raise with context
        if amount_to_decimal(total_out) < amount_to_decimal(out_req):
            raise InsufficientLiquidityError(
                requested_out=out_req,
                max_fill_out=total_cap,
                filled_out=total_out,
                spent_in=total_in,
                trace=all_trace if return_trace else None,
            )
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


# --- Read-only router-level preview (no side effects) ---
class RouterQuoteView:
    """
    Read-only router-level preview that mirrors BookStep anchoring and integer-domain math,
    without touching sandbox or mutating a live AMM. Suitable for tests/logging/UI.

    This class intentionally reuses existing routing helpers and AMM integer APIs:
      - `_prefer_clob_on_ties` for ordering CLOB segments
      - `_execute_one_iteration` for anchoring and slice assembly per iteration
      - `AMM.synthetic_segment_for_quality`, `dx_for_out_st`, `swap_*_st`, `apply_fill_st` on shadow clones

    Step 1: skeleton only. Methods are declared with signatures and docstrings,
    but raise NotImplementedError. No business logic is added in this patch.
    """

    def __init__(
        self,
        segments_provider: Callable[[], Iterable[Segment]],
        *,
        amm: Optional[AMM] = None,
        limit_quality: Optional[Quality] = None,
    ) -> None:
        """Initialise a read-only view.

        Parameters
        ----------
        segments_provider : Callable[[], Iterable[Segment]]
            Provider returning current CLOB segments.
        amm : Optional[AMM]
            AMM instance to use for preview (a shadow clone will be used internally).
        limit_quality : Optional[Quality]
            Optional minimum acceptable quality threshold for preview filtering.
        """
        self._provider = segments_provider
        self._amm = amm
        self._limit_q = limit_quality

    # ---- Snapshots (covers pool summary + CLOB ladder) ----
    def snapshot(self) -> dict:
        """Return a dict snapshot with AMM state and ordered CLOB ladder."""
        # AMM snapshot
        amm = self._amm
        if amm is not None:
            x_reserve = amm.x
            y_reserve = amm.y
            fee_decimal = float(amm.fee)
            fee_ppb = (amm._keep_fee_num, amm._fee_den)
            spq_quality = amm.spq_quality_int()
            # Compute max_out_cap by asking for a huge OUT, catch AMMOverAsk for cap
            max_out_cap = None
            try:
                # Try to get dx for a huge out (y_reserve+1000)
                huge_out = amm._from_units(amm._to_units_floor(amm.y, amm.y_is_xrp) + 1000, amm.y_is_xrp)
                _ = amm.dx_for_out_st(huge_out)
            except AMMOverAsk as e:
                max_out_cap = e.max_out
            except Exception:
                max_out_cap = None
            amm_snapshot = {
                "x_reserve": x_reserve,
                "y_reserve": y_reserve,
                "fee_decimal": fee_decimal,
                "fee_ppb": fee_ppb,
                "spq_quality": spq_quality,
                "max_out_cap": max_out_cap,
            }
        else:
            amm_snapshot = None
        # CLOB ladder: order via _prefer_clob_on_ties
        try:
            segs = _prefer_clob_on_ties(self._provider())
        except Exception:
            segs = []
        clob_ladder = [
            {"quality": s.quality, "out_max": s.out_max, "in_at_out_max": s.in_at_out_max}
            for s in segs if s.src != "AMM"
        ]
        return {
            "amm": amm_snapshot,
            "clob_ladder": clob_ladder,
        }

    # ---- Quotes (OUT-driven) ----
    def preview_out(self, out_req: Amount) -> dict:
        """Return a read-only quote preview for a requested OUT amount."""
        if out_req is None or out_req.is_zero():
            return {
                "slices": [],
                "anchors": [],
                "summary": {
                    "total_out": out_req,
                    "total_in": out_req,
                    "avg_quality": None,
                    "requested_out": out_req,
                    "is_partial": False,
                    "fill_ratio": 1.0,
                },
            }
        try:
            segs = _prefer_clob_on_ties(self._provider())
        except Exception:
            segs = []
        shadow_amm = self._amm.clone() if self._amm is not None else None
        need = out_req
        # OUT zero follows out_req domain; IN zero follows first CLOB in-side if present, otherwise AMM x-side if present
        total_out = _zero_like(out_req)
        try:
            segs_probe = _prefer_clob_on_ties(self._provider())
        except Exception:
            segs_probe = []
        if segs_probe:
            total_in = _zero_like(segs_probe[0].in_at_out_max)
        elif self._amm is not None:
            total_in = XRPAmount(0) if self._amm.x_is_xrp else IOUAmount.zero()
        else:
            total_in = _zero_like(out_req)
        slices = []
        anchors = []
        prev_amm_q = None
        for _ in range(64):
            if need.is_zero():
                break
            # Bind anchoring to the shadow AMM so its evolving state is used
            anchor_fn = (lambda q, cap: shadow_amm.synthetic_segment_for_quality(q, max_out_cap=cap)) if shadow_amm is not None else None
            spq_now = shadow_amm.spq_quality_int() if shadow_amm is not None else None
            filled, spent, next_segs, amm_used, tier_quality, amm_dx, amm_dy, itr_trace = self._pick_iteration(
                segs, need, shadow_amm
            )
            # Track anchors
            anchors.append({
                "q_lob_top": tier_quality,
                "q_amm_spq_at_iter": spq_now,
            })
            # Convert itr_trace into slices, add AMM diagnostics if needed
            for t in itr_trace:
                rec = {
                    "src": t.get("src"),
                    "source_id": t.get("source_id"),
                    "out_take": t.get("take_out"),
                    "in_take": t.get("take_in"),
                    "avg_quality": t.get("quality"),
                }
                if t.get("src") == "AMM" and shadow_amm is not None:
                    # Compute AMM diagnostics on a disposable clone
                    diag = self._analyse_amm_slice(
                        shadow_amm, t.get("take_in"), t.get("take_out"), t.get("quality")
                    )
                    rec.update(diag)
                slices.append(rec)
            if filled.is_zero():
                break
            total_in = spent if total_in.is_zero() else (total_in + spent)
            total_out = filled if total_out.is_zero() else (total_out + filled)
            need = out_req - total_out
            segs = next_segs
            # Apply AMM deltas to the shadow AMM so the next iteration sees updated reserves
            if amm_used and (shadow_amm is not None) and (not amm_dx.is_zero() or not amm_dy.is_zero()):
                shadow_amm.apply_fill_st(amm_dx, amm_dy)
        # Compute summary
        is_partial = amount_to_decimal(total_out) < amount_to_decimal(out_req)
        fill_ratio = None
        try:
            req_dec = amount_to_decimal(out_req)
            filled_dec = amount_to_decimal(total_out)
            fill_ratio = float(filled_dec / req_dec) if req_dec != 0 else 0.0
        except Exception:
            fill_ratio = None
        avg_quality = None
        if not total_out.is_zero() and not total_in.is_zero():
            avg_quality = Quality.from_amounts(total_out, total_in)
        return {
            "slices": slices,
            "anchors": anchors,
            "summary": {
                "total_out": total_out,
                "total_in": total_in,
                "avg_quality": avg_quality,
                "requested_out": out_req,
                "is_partial": is_partial,
                "fill_ratio": fill_ratio,
            },
        }

    # ---- Quotes (IN-driven, optional; mirrors BookStep.fwd) ----
    def preview_in(self, in_cap: Amount) -> dict:
        """Return a read-only quote preview for a given IN budget (optional path).

        Structure mirrors `preview_out()` and will be implemented in step 3 if needed.
        """
        raise NotImplementedError("RouterQuoteView.preview_in() is not implemented in step 1")

    # ---- Internal helpers (pure; no side effects) ----
    def _pick_iteration(
        self,
        segs: List[Segment],
        need: Amount,
        shadow_amm: Optional[AMM],
    ) -> tuple:
        """Pure selection for one iteration. Wrapper over _execute_one_iteration."""
        anchor_fn = (lambda q, cap: shadow_amm.synthetic_segment_for_quality(q, max_out_cap=cap)) if shadow_amm is not None else None
        spq_now = shadow_amm.spq_quality_int() if shadow_amm is not None else None
        return _execute_one_iteration(
            segs,
            target_out=need,
            send_max=None,
            limit_quality=self._limit_q,
            amm_anchor=anchor_fn,
            amm_spq=spq_now,
            amm_obj=shadow_amm,
        )

    def _analyse_amm_slice(
        self,
        shadow_amm: AMM,
        in_take: Amount,
        out_take: Amount,
        avg_quality: Quality,
    ) -> dict:
        """Compute AMM diagnostics on disposable clones (pre/post SPQ, fees, slippage)."""
        # Compute pre-trade SPQ
        pre_spq = shadow_amm.spq_quality_int()
        # Simulate fill on a disposable clone to get post_spq
        clone = shadow_amm.clone()
        clone.apply_fill_st(in_take, out_take)
        post_spq = clone.spq_quality_int()
        # Fee details: reconstruct dx_eff, fee_paid from gross in_take using integer grid logic
        dx_gross = shadow_amm._amt_to_units_floor(in_take)
        if dx_gross <= 0:
            dx_eff = _zero_like(in_take)
            fee_paid = _zero_like(in_take)
        else:
            dx_eff_units = (dx_gross * shadow_amm._keep_fee_num) // shadow_amm._fee_den
            dx_eff = shadow_amm._from_units(dx_eff_units, shadow_amm.x_is_xrp)
            fee_paid_units = dx_gross - dx_eff_units
            fee_paid = shadow_amm._from_units(fee_paid_units, shadow_amm.x_is_xrp)
        # Average quality as provided; pre_spq price, avg_price for slippage
        # price = IN/OUT (in decimal)
        try:
            from .core.fmt import amount_to_decimal, quality_rate_to_decimal
            p_pre = 1.0 / float(quality_rate_to_decimal(pre_spq)) if not pre_spq.is_zero() else None
            p_avg = 1.0 / float(quality_rate_to_decimal(avg_quality)) if avg_quality is not None and not avg_quality.is_zero() else None
            slippage_price_premium = (p_avg / p_pre - 1.0) if (p_pre and p_avg) else None
        except Exception:
            slippage_price_premium = None
        return {
            "pre_spq": pre_spq,
            "post_spq": post_spq,
            "dx_eff": dx_eff,
            "fee_paid": fee_paid,
            "slippage_price_premium": slippage_price_premium,
        }
