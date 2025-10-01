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
  Additionally, the default_fee_hook stages CLOB out-side issuer fees into the sandbox; rev remains read-only.
  Production code should use BookStep; the legacy router remains for research tooling.
"""
from __future__ import annotations

def default_fee_hook(trace: List[Dict[str, Any]], sandbox: "PaymentSandbox") -> None:
    """Default execution‑time fee accounting per whitepaper §1.3.2.4.

    For CLOB slices, compute issuer out‑side fee and stage it into the sandbox.
    Placeholder logic: apply a flat 0.1% fee on take_out for demonstration.
    """
    FEE_RATE = Decimal("0.001")
    for item in trace:
        if item.get("src") == "CLOB":
            take_out = item.get("take_out", Decimal(0))
            if take_out > 0:
                fee = (take_out * FEE_RATE).quantize(Decimal("1e-12"))
                if fee > 0:
                    try:
                        sandbox.stage_fee(Decimal(0), -fee)
                    except Exception:
                        continue

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterable, Optional, Tuple, List
from typing import Any, Dict

from .core import (
    Segment,
    round_out_max,
    round_in_min,
    quality_bucket,
    XRP_QUANTUM,
    IOU_QUANTUM,
    _guard_inst_quality,
)
# --- Whitepaper-aligned execution skeleton (to be filled in Batch 2) ---
def _execute_one_iteration(
    segs: List[Segment],
    *,
    target_out: Decimal,
    send_max: Optional[Decimal],
    limit_quality: Optional[Decimal],
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]],
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]],
    ctx: Optional[AMMContext],
) -> Tuple[Decimal, Decimal, List[Segment], bool, Optional[Decimal], Decimal, Decimal, List[Dict[str, Any]]]:
    """Execute a **single** whitepaper iteration locally and return its outcome.
    
    Returns:
      (filled_out, spent_in, next_segments, amm_used, tier_bucket, trace)
    Notes:
      - Mirrors `path_builder.route` Phase A/B (proportional scaling + grid rounding + guard).
      - AMM residuals are **not** carried across iterations (re-sourced next round).
      - Caller is responsible for calling `ctx.setAMMUsed()` when `amm_used` is True.
    """
    trace: List[Dict[str, Any]] = []
    amm_dx = Decimal(0)  # input token sent into pool
    amm_dy = Decimal(0)  # output token received from pool
    if not segs:
        return Decimal(0), Decimal(0), [], False, None, amm_dx, amm_dy, trace
    # Apply limit_quality if any
    if limit_quality is not None:
        qmin_bucket = quality_bucket(limit_quality)
        segs = [s for s in segs if quality_bucket(s.quality) >= qmin_bucket]
        if not segs:
            return Decimal(0), Decimal(0), [], False, None, amm_dx, amm_dy, trace
    # Determine LOB top and optionally inject a synthetic AMM anchored slice
    q_lob_top = max((s.quality for s in segs if s.src != "AMM"), default=Decimal(0))
    iter_segs = list(segs)
    synth: Optional[Segment] = None
    if q_lob_top > 0 and amm_anchor and (ctx is None or not ctx.maxItersReached()):
        try:
            synth = amm_anchor(q_lob_top, target_out)
        except TypeError:
            synth = None
        if synth is not None:
            iter_segs.append(synth)
    elif q_lob_top == 0 and amm_curve is not None and (ctx is None or not ctx.maxItersReached()):
        # No CLOB top → AMM self-pricing via curve segments
        try:
            curve_segs = list(amm_curve(target_out))
        except TypeError:
            curve_segs = []
        curve_segs = [cs for cs in curve_segs if cs and cs.out_max > 0 and cs.in_at_out_max > 0 and cs.quality > 0]
        iter_segs.extend(curve_segs)
    iter_segs = _prefer_clob_on_ties(iter_segs)
    if not iter_segs:
        return Decimal(0), Decimal(0), [], False, None, amm_dx, amm_dy, trace
    # Choose tier (anchor to LOB top bucket when synth is present)
    if synth is not None and q_lob_top > 0:
        tier_bucket = quality_bucket(q_lob_top)
        tier = [s for s in iter_segs if (s is synth) or quality_bucket(s.quality) == tier_bucket]
        rest = [s for s in iter_segs if s not in tier]
    else:
        tier_bucket = max(quality_bucket(s.quality) for s in iter_segs)
        tier = [s for s in iter_segs if quality_bucket(s.quality) == tier_bucket]
        rest = [s for s in iter_segs if quality_bucket(s.quality) != tier_bucket]
    if not tier:
        return Decimal(0), Decimal(0), rest, False, None, amm_dx, amm_dy, trace
    # Phase A: proportional ideal takes
    need = target_out
    tier_cap_out = sum(min(s.out_max, need) for s in tier)
    if tier_cap_out <= 0:
        return Decimal(0), Decimal(0), rest, False, tier_bucket, amm_dx, amm_dy, trace
    ratio_out = (need / tier_cap_out) if tier_cap_out > 0 else Decimal(0)
    ideal_out = []
    ideal_in = []
    for s in tier:
        o = (min(s.out_max, need) * ratio_out)
        ideal_out.append(o)
        slice_ratio = (s.in_at_out_max / s.out_max) if s.out_max > 0 else Decimal(0)
        ideal_in.append(o * slice_ratio)
    # Respect send_max at tier level
    sum_ideal_in = sum(ideal_in)
    if send_max is not None and sum_ideal_in > send_max and sum_ideal_in > 0:
        scale = (send_max / sum_ideal_in)
        ideal_out = [x * scale for x in ideal_out]
        ideal_in = [x * scale for x in ideal_in]
    # Phase B: grid rounding + guard + largest-remainder top-up
    flo_out: List[Decimal] = []
    flo_in: List[Decimal] = []
    rema: List[Decimal] = []
    quanta: List[Decimal] = []
    sum_out = Decimal(0)
    sum_in = Decimal(0)
    for s, o, i in zip(tier, ideal_out, ideal_in):
        q_out = XRP_QUANTUM if s.out_is_xrp else IOU_QUANTUM
        out_floor = round_out_max(o, is_xrp=s.out_is_xrp)
        if out_floor > s.out_max:
            out_floor = round_out_max(s.out_max, is_xrp=s.out_is_xrp)
        if out_floor > 0:
            ratio = s.in_at_out_max / s.out_max
            in_ceil = round_in_min(out_floor * ratio, is_xrp=s.in_is_xrp)
            in_ceil = _guard_inst_quality(out_floor, in_ceil, s.quality, in_is_xrp=s.in_is_xrp)
        else:
            in_ceil = Decimal(0)
        flo_out.append(out_floor)
        flo_in.append(in_ceil)
        quanta.append(q_out)
        rem = (o - out_floor) / q_out if q_out > 0 else Decimal(0)
        rema.append(rem)
        sum_out += out_floor
        sum_in += in_ceil
    # One-quantum top-up
    remaining_out_gap = need - sum_out
    if remaining_out_gap > 0:
        order = sorted(range(len(tier)), key=lambda k: rema[k], reverse=True)
        for k in order:
            if remaining_out_gap <= 0:
                break
            s = tier[k]
            q_out = quanta[k]
            if q_out <= 0:
                continue
            cand_out = flo_out[k] + q_out
            if cand_out > s.out_max or cand_out > need:
                continue
            ratio = s.in_at_out_max / s.out_max
            cand_in = round_in_min(cand_out * ratio, is_xrp=s.in_is_xrp)
            cand_in = _guard_inst_quality(cand_out, cand_in, s.quality, in_is_xrp=s.in_is_xrp)
            inst_q = cand_out / cand_in if cand_in > 0 else Decimal(0)
            if inst_q > s.quality:
                continue
            remaining_out_gap -= q_out
            sum_out += q_out
            sum_in += (cand_in - flo_in[k])
            flo_out[k] = cand_out
            flo_in[k] = cand_in
    # Build next segments: carry non-AMM residuals; drop AMM residuals
    next_segs: List[Segment] = []
    amm_used = False
    for s, out_i, in_i in zip(tier, flo_out, flo_in):
        if s.src == "AMM" and (out_i > 0 or in_i > 0):
            amm_used = True
            amm_dx += in_i
            amm_dy += out_i
        rem_out = s.out_max - out_i
        rem_in = s.in_at_out_max - in_i
        if rem_out > 0 and rem_in > 0 and (s.quality > 0) and (s.src != "AMM") and (s is not synth):
            next_segs.append(Segment(
                src=s.src,
                out_max=rem_out,
                in_at_out_max=rem_in,
                quality=s.quality,
                in_is_xrp=s.in_is_xrp,
                out_is_xrp=s.out_is_xrp,
            ))
    # Append rest (non-tier) for the caller to consider in subsequent rounds
    next_segs.extend([s for s in rest if (s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0 and s.src != "AMM")])
    # Trace items (diagnostic only)
    for s, o_raw, i_raw, o, i in zip(tier, ideal_out, ideal_in, flo_out, flo_in):
        trace.append({
            "src": s.src,
            "take_out": o,
            "take_in": i,
            "quality": s.quality,
            "take_out_raw": o_raw,
            "take_in_raw": i_raw,
        })
    return sum_out, sum_in, next_segs, amm_used, tier_bucket, amm_dx, amm_dy, trace

from .path_builder import _prefer_clob_on_ties  # shared tie-break (whitepaper §1.2.7.2)
from .amm_context import AMMContext
from .steps import Step




# --- Whitepaper-aligned execution skeleton (to be filled in Batch 2) ---
def _tier_segments(segs: List[Segment], lob_top_bucket: Optional[Decimal]) -> Tuple[List[Segment], List[Segment], Optional[Decimal]]:
    """Partition segments into (tier, rest) where 'tier' is the single highest-quality bucket.
    If lob_top_bucket is provided (AMM-anchored coexistence), use it as the tier key.
    Returns: (tier, rest, chosen_bucket).
    """
    if not segs:
        return [], [], None
    # Determine the bucket
    if lob_top_bucket is not None and lob_top_bucket > 0:
        bucket = lob_top_bucket
    else:
        bucket = max((s.quality for s in segs), default=Decimal(0))
    tier: List[Segment] = []
    rest: List[Segment] = []
    for s in segs:
        if s.quality == bucket:
            tier.append(s)
        else:
            rest.append(s)
    return tier, rest, bucket if bucket > 0 else None

def _for_each_offer(tier: List[Segment]) -> Iterable[Segment]:
    """Iterate homogeneous quotes (same tier). Placeholder; will host offer objects later."""
    # In Batch 2, this will yield CLOB `Offer` and synthetic `AMMOffer` instances.
    return list(tier)

def _exec_offer(seg: Segment, out_take: Decimal, *, preserve_quality: bool = True) -> Tuple[Decimal, Decimal, Dict[str, Any]]:
    """Execute (simulate) consuming `out_take` from `seg`. Placeholder returning scaled take.
    Returns (in_spent, out_filled, diagnostics).  In Batch 2, this will compute ownerGives/stpAmt
    and enforce issuer fees per §1.3.2.4.  For now, keep proportional scaling (no writeback).
    """
    if out_take <= 0 or seg.out_max <= 0:
        return Decimal(0), Decimal(0), {"src": seg.src, "note": "noop"}
    # keep slice-quality via proportional scaling (multi-path semantics)
    ratio = (seg.in_at_out_max / seg.out_max) if seg.out_max > 0 else Decimal(0)
    in_need = out_take * ratio
    return in_need, out_take, {"src": seg.src, "ratio": ratio}

def _try_amm_anchor(lob_top_q: Decimal, need: Decimal, *, maker: Optional[Callable[[Decimal, Decimal], Optional[Segment]]]) -> Optional[Segment]:
    """Call AMM anchoring callback if provided and SPQ strictly exceeds LOB top (enforced upstream)."""
    if maker is None or lob_top_q <= 0 or need <= 0:
        return None
    try:
        return maker(lob_top_q, need)
    except Exception:
        return None


@dataclass
class BookStep(Step):
    """Unified market step for CLOB/AMM consumption (transitional implementation).

    Forward execution also stages AMM pool deltas (dx, dy) into the sandbox; Flow.apply()
    later commits them via apply_sink. The default_fee_hook stages CLOB out-side issuer fees into the sandbox; rev remains read-only.
    """

    segments_provider: Callable[[], Iterable[Segment]]
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]] = None
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]] = None
    amm_context: Optional[AMMContext] = None
    # route_config: Optional[RouteConfig] = None
    limit_quality: Optional[Decimal] = None
    fee_hook: Optional[Callable[[List[Dict[str, Any]], "PaymentSandbox"], None]] = None

    def rev(self, sandbox: "PaymentSandbox", out_req: Decimal) -> Tuple[Decimal, Decimal]:
        # Reverse: compute required IN for a desired OUT without mutating ledger state.
        if out_req is None or out_req <= 0:
            self._cached_in = Decimal(0)
            self._cached_out = Decimal(0)
            return self._cached_in, self._cached_out
        try:
            segs = _prefer_clob_on_ties(self.segments_provider())
        except Exception:
            segs = []
        if not segs:
            self._cached_in = Decimal(0)
            self._cached_out = Decimal(0)
            return self._cached_in, self._cached_out

        # Internal multi-iteration execution (whitepaper semantics)
        need = out_req
        total_in = Decimal(0)
        total_out = Decimal(0)
        # Safety cap on iterations to avoid infinite loops
        for _ in range(64):
            if need <= 0 or not segs:
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
            if filled <= 0 or (filled == 0 and spent == 0):
                break
            total_in += spent
            total_out += filled
            need = out_req - total_out
            segs = next_segs
            if amm_used and self.amm_context is not None:
                self.amm_context.setAMMUsed()

        self._cached_out = total_out
        self._cached_in = total_in
        return self._cached_in, self._cached_out

    def fwd(self, sandbox: "PaymentSandbox", in_cap: Decimal) -> Tuple[Decimal, Decimal]:
        # Forward: compute achievable OUT given an input cap; ledger writebacks are still handled by higher layers.
        if in_cap is None or in_cap <= 0:
            self._cached_in = Decimal(0)
            self._cached_out = Decimal(0)
            return self._cached_in, self._cached_out
        try:
            segs = _prefer_clob_on_ties(self.segments_provider())
        except Exception:
            segs = []
        if not segs:
            self._cached_in = Decimal(0)
            self._cached_out = Decimal(0)
            return self._cached_in, self._cached_out

        # Internal multi-iteration execution with IN cap honoured via send_max
        remaining_in = in_cap
        total_in = Decimal(0)
        total_out = Decimal(0)
        # Use a generous target_out upper bound (sum of available OUT); executor will scale by send_max
        target_out = sum((s.out_max for s in segs), Decimal(0))
        if target_out <= 0:
            self._cached_in = Decimal(0)
            self._cached_out = Decimal(0)
            return self._cached_in, self._cached_out
        for _ in range(64):
            if remaining_in <= 0 or not segs or target_out <= 0:
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
            if (filled <= 0 and spent <= 0) or spent <= 0:
                break
            total_in += spent
            total_out += filled
            remaining_in = in_cap - total_in
            segs = next_segs
            # Reduce future target_out by what we've already filled to avoid pointless work
            target_out = sum((s.out_max for s in segs), Decimal(0))
            if amm_used and self.amm_context is not None:
                self.amm_context.setAMMUsed()
            if amm_used and (amm_dx > 0 or amm_dy > 0) and sandbox is not None:
                try:
                    sandbox.stage_after_iteration(amm_dx, -amm_dy)
                except Exception:
                    pass

            # Execution‑time fee accounting (CLOB out‑side issuer fees)
            hook = self.fee_hook or default_fee_hook
            if sandbox is not None and hook is not None:
                try:
                    hook(trace, sandbox)
                except Exception:
                    pass

        self._cached_in = total_in
        self._cached_out = total_out
        return self._cached_in, self._cached_out

    def quality_upper_bound(self) -> Decimal:
        # Upper bound is the best available slice quality (prefer raw_quality).
        try:
            segs = list(self.segments_provider())
        except Exception:
            return Decimal(0)
        if not segs:
            return Decimal(0)
        return max((s.raw_quality or s.quality) for s in segs)