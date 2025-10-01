"""Routing interface: merge segments and consume in quality order.

Implements tiered iterations per the whitepaper: each iteration consumes only the
current highest-quality tier (same-quality slices), with proportional scaling on
limits to preserve slice quality. Supports AMM anchoring: if provided, an
`amm_anchor` callback can inject a synthetic AMM segment anchored to the LOB top
quality for the current iteration. Tier selection is anchored to the current LOB top quality when a synthetic AMM slice is present. Supports send_max/deliver_min limits. Supports per-iteration AMM state writeback via an optional after_iteration callback. When no CLOB top is available, the router can fall back to AMM curve segments via an optional amm_curve callback (AMM self-pricing for that iteration). AMM residual slices are not carried across iterations; they are re-sourced after writeback.

The `limit_quality` parameter (optional) enforces a per-iteration quality floor consistent with the whitepaper flow(Strands)/offer crossing semantics. If provided, only segments at or above this quality (bucketed) are considered in each iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Dict, Callable, Optional, Any

from .core import (
    Segment,
    RouteResult,
    round_in_min,
    round_out_max,
    quality_bucket,
    XRP_QUANTUM,
    IOU_QUANTUM,
    ExecutionReport,
    IterationMetrics,
    _guard_inst_quality,
    _sort_by_bucket_stable,
    _apply_quality_floor,
)

from .amm_context import AMMContext


class RouteError(Exception):
    """Raised when routing cannot proceed (e.g., no segments, invalid target)."""


@dataclass(frozen=True)
class RouteConfig:
    """Router configuration.

    preserve_quality_on_limit: if True, partial consumption keeps slice quality
    constant via proportional scaling (multi-path semantics).
    """
    preserve_quality_on_limit: bool = True


# Tie-break helper: prefer CLOB over AMM when both present at equal quality (same tier)
from typing import Iterable, List, Dict
def _prefer_clob_on_ties(segs: Iterable[Segment]) -> List[Segment]:
    """When AMM and CLOB are at equal quality (same tier), prefer CLOB.
    Uses raw_quality if present, else bucketed quality.
    """
    segs_list = [s for s in segs if s and s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0]
    if not segs_list:
        return []
    tiers: Dict[Decimal, List[bool]] = {}
    for s in segs_list:
        key = (s.raw_quality or s.quality)
        flags = tiers.get(key, [False, False])
        if s.src == "CLOB":
            flags[0] = True
        elif s.src == "AMM":
            flags[1] = True
        tiers[key] = flags
    out: List[Segment] = []
    for s in segs_list:
        key = (s.raw_quality or s.quality)
        has_clob, has_amm = tiers.get(key, (False, False))
        if has_clob and has_amm and s.src == "AMM":
            continue
        out.append(s)
    return out


def route(
    target_out: Decimal,
    segments: Iterable[Segment],
    *,
    config: RouteConfig | None = None,
    amm_anchor: Callable[[Decimal, Decimal], Optional[Segment]] | None = None,
    send_max: Optional[Decimal] = None,
    deliver_min: Optional[Decimal] = None,
    limit_quality: Optional[Decimal] = None,
    amm_context: AMMContext | None = None,
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]] = None,
    after_iteration: Optional[Callable[[Decimal, Decimal], None]] = None,
) -> RouteResult:
    """Route a target OUT across segments in descending quality order.

    The optional `limit_quality` enforces a per-iteration quality floor (bucketed), consistent with the whitepaper flow(Strands)/offer crossing semantics.

    Additionally:
    - Tie-break rule (§1.2.7.2): when AMM and CLOB are equal-quality in a tier, CLOB is preferred.
    """
    if target_out is None:
        raise RouteError("target_out is None")
    if target_out <= 0:
        raise RouteError("target_out must be > 0")

    segs: List[Segment] = _prefer_clob_on_ties(segments)

    # Deterministic tie-breaker: remember insertion order for stable secondary sort (by id)
    order_map_id: Dict[int, int] = {}
    def _ensure_order_for_list(lst: List[Segment]) -> None:
        for s in lst:
            sid = id(s)
            if sid not in order_map_id:
                order_map_id[sid] = len(order_map_id)
    _ensure_order_for_list(segs)

    # AMM-only bootstrap: if no CLOB segments were provided, try curve once
    if not segs and amm_curve is not None:
        try:
            curve_boot = list(amm_curve(target_out))
        except TypeError:
            curve_boot = []  # backward-compat: older signature
        segs = _prefer_clob_on_ties(curve_boot)

    if not segs:
        raise RouteError("no usable segments")

    _sort_by_bucket_stable(segs, order_map_id)

    filled_out: Decimal = Decimal(0)
    spent_in: Decimal = Decimal(0)
    usage: Dict[str, Decimal] = {}
    trace: List[Dict[str, Any]] = []

    preserve = True if config is None else config.preserve_quality_on_limit
    need = target_out

    eff_send_max = send_max if (send_max is not None) else None
    eff_deliver_min = deliver_min if (deliver_min is not None) else None

    ctx = amm_context or AMMContext(False)

    # --- Metrics reporting ---
    iter_records: List[IterationMetrics] = []
    iter_idx: int = 0
    limit_floor = limit_quality  # keep original value for reporting

    # Compute quality floor bucket if limit_quality is provided
    if limit_quality is not None:
        qmin_bucket = quality_bucket(limit_quality)
    else:
        qmin_bucket = None

    # (No-improvement ceiling/floor tracking removed: whitepaper semantics only)

    while need > 0:
        # If no segments are available at the start of an iteration, try sourcing from AMM curve
        if (not segs) and (amm_curve is not None):
            try:
                refilled = list(amm_curve(need))
            except TypeError:
                refilled = []
            segs = _prefer_clob_on_ties(refilled)
            _ensure_order_for_list(segs)
            # If still empty after attempting to refill, break (nothing more to route)
            if not segs:
                break
        elif not segs:
            # No segments and no AMM curve to source from
            break

        # Anchoring per iteration
        q_lob_top = max((s.quality for s in segs if s.src != "AMM"), default=Decimal(0))
        curve_mode = False

        synth: Optional[Segment] = None
        iter_segs = list(segs)

        if q_lob_top > 0 and amm_anchor:
            if not ctx.maxItersReached():
                synth = amm_anchor(q_lob_top, need)
                if synth is not None:
                    iter_segs.append(synth)
        elif q_lob_top == 0 and amm_curve is not None:
            # No CLOB top → AMM self-pricing via curve segments
            curve_mode = True
            curve_segs = []
            if not ctx.maxItersReached():
                try:
                    curve_segs = list(amm_curve(need))
                except TypeError:
                    curve_segs = []  # backward-compat signature
                # Filter invalid segments
                curve_segs = [cs for cs in curve_segs if cs and cs.out_max > 0 and cs.in_at_out_max > 0 and cs.quality > 0]
            iter_segs.extend(curve_segs)

        _ensure_order_for_list(iter_segs)

        # Enforce "prefer CLOB on ties" before sorting/tiering
        iter_segs = _prefer_clob_on_ties(iter_segs)

        # No pre-filtering by "quality ceiling": we enforce no-improvement via an effective-price floor
        # applied after grid rounding at the iteration level (see below).

        # Apply per-iteration quality floor if requested
        if qmin_bucket is not None:
            iter_segs = _apply_quality_floor(iter_segs, qmin_bucket)

        if not iter_segs:
            # Drop any AMM segments (none should remain), and if amm_curve is available and segs is empty, try refill once
            segs = [s for s in segs if s.src != "AMM"]
            if amm_curve is not None and not segs:
                try:
                    refilled = list(amm_curve(need))
                except TypeError:
                    refilled = []
                segs = _prefer_clob_on_ties(refilled)
                _ensure_order_for_list(segs)
                # Apply the same filter to the refilled segments
                if qmin_bucket is not None:
                    segs = _apply_quality_floor(segs, qmin_bucket)
                iter_segs = list(segs)
                if not iter_segs:
                    break
            else:
                break

        # Sort candidates after injecting synth/curve segments to ensure proper tiering
        _sort_by_bucket_stable(iter_segs, order_map_id)

        iter_amm_in: Decimal = Decimal(0)
        iter_amm_out: Decimal = Decimal(0)

        tier: List[Segment] = []
        rest: List[Segment] = []
        # Record the tier bucket used this iteration for reporting
        if synth is not None and q_lob_top > 0 and not curve_mode:
            current_tier_bucket = quality_bucket(q_lob_top)
            # Anchor the tier to LOB top quality bucket: include synth and all segments in that bucket
            tier_bucket = quality_bucket(q_lob_top)
            for s in iter_segs:
                if (s is synth) or (quality_bucket(s.quality) == tier_bucket):
                    tier.append(s)
                else:
                    rest.append(s)
        else:
            current_tier_bucket = max(quality_bucket(s.quality) for s in iter_segs) if iter_segs else Decimal(0)
            # Default: take the highest-quality bucket tier
            max_bucket = max(quality_bucket(s.quality) for s in iter_segs)
            for s in iter_segs:
                (tier if quality_bucket(s.quality) == max_bucket else rest).append(s)

        new_tier: List[Segment] = []

        if need > 0 and tier:
            # Phase A: proportional ideal takes using slice-specific average ratio
            budget_scaled = False
            tier_cap_out = sum(min(s.out_max, need) for s in tier)
            if tier_cap_out <= 0:
                # Nothing usable in this tier
                # Drop any AMM segments before next iteration to avoid stale AMM pricing
                segs = [s for s in rest if (s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0 and s.src != "AMM")]
                continue

            ratio_out = (need / tier_cap_out) if tier_cap_out > 0 else Decimal(0)
            ideal_out = []  # per-slice ideal (pre-quantisation)
            ideal_in = []   # per-slice ideal IN using slice-specific average ratio (captures AMM virtual writeback)
            for s in tier:
                out_prop = (min(s.out_max, need) * ratio_out)
                ideal_out.append(out_prop)
                slice_ratio = (s.in_at_out_max / s.out_max) if s.out_max > 0 else Decimal(0)
                ideal_in.append(out_prop * slice_ratio)

            # Note: Phase A's ideal_in is used only for checking whether tier-level scaling is needed under send_max.
            # Actual per-slice IN is recomputed in Phase B from slice-specific ratio (in_at_out_max/out_max).

            # Respect send_max by uniform scaling at tier-level (preserves slice quality)
            if eff_send_max is not None:
                remaining_in_budget = eff_send_max - spent_in
                if remaining_in_budget <= 0:
                    need = Decimal(0)
                    segs = rest
                    continue
                sum_ideal_in = sum(ideal_in)
                if sum_ideal_in > remaining_in_budget and sum_ideal_in > 0:
                    scale = (remaining_in_budget / sum_ideal_in)
                    ideal_out = [x * scale for x in ideal_out]
                    ideal_in = [x * scale for x in ideal_in]
                    budget_scaled = True

            # Phase B: grid rounding and simple largest-remainder top-up (≤1 quantum per slice)
            flo_out = []
            flo_in = []
            rema = []  # fractional remainder of OUT (normalised by quantum)
            quanta = []
            sum_out = Decimal(0)
            sum_in = Decimal(0)
            for s, o, i in zip(tier, ideal_out, ideal_in):
                q_out = XRP_QUANTUM if s.out_is_xrp else IOU_QUANTUM
                q_in = XRP_QUANTUM if s.in_is_xrp else IOU_QUANTUM
                out_floor = round_out_max(o, is_xrp=s.out_is_xrp)
                # Bound by capacity
                if out_floor > s.out_max:
                    out_floor = round_out_max(s.out_max, is_xrp=s.out_is_xrp)
                # Use slice-specific ratio from precomputed segment (captures AMM virtual writeback):
                # ratio = in_at_out_max / out_max
                if out_floor > 0:
                    ratio = s.in_at_out_max / s.out_max
                    in_ceil = round_in_min(out_floor * ratio, is_xrp=s.in_is_xrp)
                    # Enforce instantaneous quality not to exceed the slice quality
                    in_ceil = _guard_inst_quality(out_floor, in_ceil, s.quality, in_is_xrp=s.in_is_xrp)
                else:
                    in_ceil = Decimal(0)
                flo_out.append(out_floor)
                flo_in.append(in_ceil)
                quanta.append(q_out)
                # remainder normalized by quantum to compare slices with different grids
                rem = (o - out_floor) / q_out if q_out > 0 else Decimal(0)
                rema.append(rem)
                sum_out += out_floor
                sum_in += in_ceil

            # If we are short on OUT vs need, try topping up by at most one quantum per slice
            need_goal = need
            remaining_out_gap = need_goal - sum_out
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
                    if cand_out > s.out_max or cand_out > need_goal:
                        continue
                    # Compute IN impact and check budget + quality guard
                    ratio = s.in_at_out_max / s.out_max
                    cand_in = round_in_min(cand_out * ratio, is_xrp=s.in_is_xrp)
                    add_in = cand_in - flo_in[k]
                    if eff_send_max is not None and (spent_in + sum_in + add_in) > eff_send_max:
                        continue
                    cand_in = _guard_inst_quality(cand_out, cand_in, s.quality, in_is_xrp=s.in_is_xrp)
                    inst_q = cand_out / cand_in if cand_in > 0 else Decimal(0)
                    if inst_q > s.quality:
                        continue
                    # Accept top-up
                    remaining_out_gap -= q_out
                    sum_out += q_out
                    sum_in += add_in
                    flo_out[k] = cand_out
                    flo_in[k] = cand_in

            # Per-iteration totals for reporting
            iter_out_sum = sum(flo_out)
            iter_in_sum = sum(flo_in)
            iter_price_eff = (iter_in_sum / iter_out_sum) if iter_out_sum > 0 else Decimal(0)


            # Baseline price for slippage at this tier, aligned to ledger rounding (computed AFTER enforcing floor)
            if current_tier_bucket > 0 and iter_out_sum > 0:
                in_is_xrp_tier = any(s.in_is_xrp for s in tier)
                baseline_in_guarded = round_in_min(iter_out_sum / current_tier_bucket, is_xrp=in_is_xrp_tier)
                baseline_price = baseline_in_guarded / iter_out_sum
                slippage_price = iter_price_eff - baseline_price
            else:
                baseline_price = Decimal(0)
                slippage_price = Decimal(0)
            # Fee components (pool fee and issuer transfer fees) are AMM-specific and require
            # pool parameters; keep placeholders here. They can be populated by higher-level
            # wrappers that know the concrete AMM instance.
            fee_pool = Decimal(0)
            fee_tr_in = Decimal(0)
            fee_tr_out = Decimal(0)

            # Apply batched fills for this tier
            # Also record Phase-A ideal (pre-grid) amounts for diagnostics: take_out_raw/take_in_raw/inst_q_raw
            for s, out_i, in_i, o_raw, i_raw in zip(tier, flo_out, flo_in, ideal_out, ideal_in):
                if out_i <= 0 or in_i <= 0:
                    # Keep the original segment for future use if it still has capacity
                    if s.out_max > 0 and s.in_at_out_max > 0:
                        new_tier.append(s)
                    continue

                if s.src == "AMM":
                    iter_amm_in += in_i
                    iter_amm_out += out_i

                filled_out += out_i
                spent_in += in_i
                usage[s.src] = usage.get(s.src, Decimal(0)) + out_i
                inst_q_raw = (o_raw / i_raw) if i_raw > 0 else Decimal(0)
                trace.append({
                    "src": s.src,
                    "take_out": out_i,          # post-grid OUT (ledger)
                    "take_in": in_i,            # post-grid IN (ledger)
                    "quality": s.quality,       # slice quality (bucketed elsewhere in prints)
                    # diagnostics (pre-grid, Phase A):
                    "take_out_raw": o_raw,      # ideal OUT before rounding to grid
                    "take_in_raw": i_raw,       # ideal IN before rounding to grid
                    "inst_q_raw": inst_q_raw,   # ideal instantaneous quality (no bucket)
                })

                # Residual capacity (carry to next round only for non-AMM; synthetic never carried)
                #
                # Rationale:
                # - Whitepaper iteration semantics require that after each round's fills are applied,
                #   the AMM state (x,y) is updated and *then* AMM pricing/segments are recomputed.
                # - If we were to carry an AMM residual slice across rounds, it would retain an
                #   obsolete quality computed from the pre-writeback reserves, causing the next
                #   round to compare/route against stale AMM pricing (e.g., inst_q seemingly
                #   exceeding the newly printed SPQ). To avoid that, AMM residuals are dropped and
                #   the AMM (if needed) will be re-sourced via `amm_anchor`/`amm_curve` using the
                #   updated reserves.
                rem_out = s.out_max - out_i
                rem_in = s.in_at_out_max - in_i
                if rem_out > 0 and rem_in > 0 and (s.quality > 0) and (s.src != "AMM") and (s is not synth):
                    new_seg = Segment(
                        src=s.src,
                        out_max=rem_out,
                        in_at_out_max=rem_in,
                        quality=s.quality,
                        in_is_xrp=s.in_is_xrp,
                        out_is_xrp=s.out_is_xrp,
                    )
                    _ensure_order_for_list([new_seg])
                    new_tier.append(new_seg)

            need = target_out - filled_out
            if need <= 0:
                # Write back new segments before recording metrics (unify order)
                segs = [
                    s for s in (new_tier + rest)
                    if (s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0 and s.src != "AMM")
                ]
                _ensure_order_for_list(segs)
                _sort_by_bucket_stable(segs, order_map_id)
                if after_iteration is not None and (iter_amm_in > 0 or iter_amm_out > 0):
                    if iter_amm_in > 0 or iter_amm_out > 0:
                        ctx.setAMMUsed()
                    after_iteration(iter_amm_in, iter_amm_out)
                # Append per-iteration metrics (complete fill)
                iter_records.append(IterationMetrics(
                    iter_index=iter_idx,
                    tier_quality=current_tier_bucket,
                    out_filled=iter_out_sum,
                    in_spent=iter_in_sum,
                    price_effective=iter_price_eff,
                    amm_used=bool(iter_amm_in > 0 or iter_amm_out > 0),
                    budget_limited=budget_scaled,
                    limit_quality_floor=limit_floor,
                    fee_pool=fee_pool,
                    fee_tr_in=fee_tr_in,
                    fee_tr_out=fee_tr_out,
                    slippage_price=slippage_price,
                ))
                iter_idx += 1
                break
        else:
            # Nothing to do in this tier
            new_tier = tier

        segs = [
            s for s in (new_tier + rest)
            if (s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0 and s.src != "AMM")
        ]
        _ensure_order_for_list(segs)
        _sort_by_bucket_stable(segs, order_map_id)
        if after_iteration is not None and (iter_amm_in > 0 or iter_amm_out > 0):
            if iter_amm_in > 0 or iter_amm_out > 0:
                ctx.setAMMUsed()
            after_iteration(iter_amm_in, iter_amm_out)
        # Append per-iteration metrics (normal continue)
        if need > 0 and tier:
            iter_records.append(IterationMetrics(
                iter_index=iter_idx,
                tier_quality=current_tier_bucket,
                out_filled=iter_out_sum,
                in_spent=iter_in_sum,
                price_effective=iter_price_eff,
                amm_used=bool(iter_amm_in > 0 or iter_amm_out > 0),
                budget_limited=budget_scaled,
                limit_quality_floor=limit_floor,
                fee_pool=fee_pool,
                fee_tr_in=fee_tr_in,
                fee_tr_out=fee_tr_out,
                slippage_price=slippage_price,
            ))
            iter_idx += 1
        need = target_out - filled_out

        if eff_send_max is not None and spent_in >= eff_send_max:
            break

    if eff_deliver_min is not None and filled_out < eff_deliver_min:
        raise RouteError("deliver_min not met")

    avg_quality = (filled_out / spent_in) if spent_in > 0 else Decimal(0)

    # --- Build ExecutionReport for diagnostics ---
    avg_price = (spent_in / filled_out) if filled_out > 0 else Decimal(0)
    filled_ratio = (filled_out / target_out) if target_out > 0 else Decimal(0)
    in_budget_ratio = (spent_in / eff_send_max) if (eff_send_max is not None and eff_send_max > 0) else None
    fee_pool_total = sum((it.fee_pool for it in iter_records), Decimal(0))
    fee_tr_in_total = sum((it.fee_tr_in for it in iter_records), Decimal(0))
    fee_tr_out_total = sum((it.fee_tr_out for it in iter_records), Decimal(0))
    # OUT-weighted average slippage (whitepaper semantics: aggregate over filled quantity)
    slip_num = sum((it.slippage_price * it.out_filled) for it in iter_records)
    slip_den = sum((it.out_filled) for it in iter_records)
    slippage_price_avg = (slip_num / slip_den) if slip_den > 0 else Decimal(0)
    report = ExecutionReport(
        iterations=iter_records,
        total_out=filled_out,
        total_in=spent_in,
        avg_price=avg_price,
        avg_quality=avg_quality,
        filled_ratio=filled_ratio,
        in_budget_ratio=in_budget_ratio,
        fee_pool_total=fee_pool_total,
        fee_tr_in_total=fee_tr_in_total,
        fee_tr_out_total=fee_tr_out_total,
        slippage_price_avg=slippage_price_avg,
    )

    return RouteResult(
        filled_out=filled_out,
        spent_in=spent_in,
        avg_quality=avg_quality,
        usage=usage,
        trace=trace,
        report=report,
    )