"""Routing interface: merge segments and consume in quality order.

Implements tiered iterations per the whitepaper: each iteration consumes only the
current highest-quality tier (same-quality slices), with proportional scaling on
limits to preserve slice quality. Supports AMM anchoring: if provided, an
`amm_anchor` callback can inject a synthetic AMM segment anchored to the LOB top
quality for the current iteration. Tier selection is anchored to the current LOB top quality when a synthetic AMM slice is present. Supports send_max/deliver_min limits. Supports per-iteration AMM state writeback via an optional after_iteration callback. When no CLOB top is available, the router can fall back to AMM curve segments via an optional amm_curve callback (AMM self-pricing for that iteration).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Dict, Callable, Optional

from .core import Segment, RouteResult, quantize_down, round_in_min, round_out_max


class RouteError(Exception):
    """Raised when routing cannot proceed (e.g., no segments, invalid target)."""


@dataclass(frozen=True)
class RouteConfig:
    """Router configuration.

    preserve_quality_on_limit: if True, partial consumption keeps slice quality
    constant via proportional scaling (multi-path semantics).
    """
    preserve_quality_on_limit: bool = True


def route(target_out: Decimal,
          segments: Iterable[Segment],
          *,
          config: RouteConfig | None = None,
          amm_anchor: Callable[[Decimal, Decimal], Optional[Segment]] | None = None,
          send_max: Optional[Decimal] = None,
          deliver_min: Optional[Decimal] = None,
          amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]] = None,
          after_iteration: Optional[Callable[[Decimal, Decimal], None]] = None,
          ) -> RouteResult:
    """Route a target OUT across segments in descending quality order.

    Parameters
    ----------
    target_out : Decimal
        The desired output amount to route.
    segments : Iterable[Segment]
        The segments to route through.
    config : RouteConfig | None, optional
        Router configuration options.
    amm_anchor : callable | None, optional
        Function to inject synthetic AMM segment anchored to LOB top quality.
    send_max : Decimal | None, optional
        Maximum input amount allowed.
    deliver_min : Decimal | None, optional
        Minimum output amount required.
    amm_curve : callable | None, optional
        When no CLOB top quality exists in an iteration, this callback is used to supply AMM curve segments directly (self-pricing). It receives (need) and must return Iterable[Segment].
    after_iteration : callable | None, optional
        If provided, called at the end of each iteration with (sum_in_AMM, sum_out_AMM) actually executed in that iteration; use to update AMM reserves.
    """
    if target_out is None:
        raise RouteError("target_out is None")
    if target_out <= 0:
        raise RouteError("target_out must be > 0")

    segs: List[Segment] = [
        s for s in segments
        if (s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0)
    ]
    if not segs:
        raise RouteError("no usable segments")

    segs.sort(key=lambda s: s.quality, reverse=True)

    filled_out: Decimal = Decimal(0)
    spent_in: Decimal = Decimal(0)
    usage: Dict[str, Decimal] = {}
    trace: List[Dict[str, Decimal]] = []

    preserve = True if config is None else config.preserve_quality_on_limit
    need = target_out

    eff_send_max = send_max if (send_max is not None) else None
    eff_deliver_min = deliver_min if (deliver_min is not None) else None

    while need > 0 and segs:
        # Anchoring per iteration
        q_lob_top = max((s.quality for s in segs if s.src != "AMM"), default=Decimal(0))
        curve_mode = False

        synth: Optional[Segment] = None
        iter_segs = list(segs)

        if q_lob_top > 0 and amm_anchor:
            synth = amm_anchor(q_lob_top, need)
            if synth is not None:
                iter_segs.append(synth)
        elif q_lob_top == 0 and amm_curve is not None:
            # No CLOB top in this iteration → AMM self-pricing via curve segments
            curve_mode = True
            try:
                curve_segs = list(amm_curve(need))
            except TypeError:
                # Backward compatibility: allow amm_curve to be defined as Callable[[Decimal], Iterable[Segment]]
                curve_segs = []
            # Filter invalid segments
            curve_segs = [cs for cs in curve_segs if cs and cs.out_max > 0 and cs.in_at_out_max > 0 and cs.quality > 0]
            iter_segs.extend(curve_segs)

        if not iter_segs:
            break

        # Sort candidates after injecting synth/curve segments to ensure proper tiering
        iter_segs.sort(key=lambda s: s.quality, reverse=True)

        iter_amm_in: Decimal = Decimal(0)
        iter_amm_out: Decimal = Decimal(0)

        tier: List[Segment] = []
        rest: List[Segment] = []
        if synth is not None and q_lob_top > 0 and not curve_mode:
            # Anchor the tier to LOB top quality: include synth and all quotes at that tier
            tier_quality = q_lob_top
            for s in iter_segs:
                if (s is synth) or (s.quality == tier_quality):
                    tier.append(s)
                else:
                    rest.append(s)
        else:
            # Default: take the highest-quality tier
            max_quality = iter_segs[0].quality
            for s in iter_segs:
                (tier if s.quality == max_quality else rest).append(s)

        new_tier: List[Segment] = []
        for s in tier:
            if need <= 0:
                new_tier.append(s)
                continue

            # Plan proportional take
            take_out_prop = s.out_max if s.out_max <= need else need
            if take_out_prop <= 0:
                new_tier.append(s)
                continue

            ratio = take_out_prop / s.out_max
            take_in_prop = s.in_at_out_max * ratio

            # Double-sided rounding to amount grids
            take_out = round_out_max(take_out_prop, is_xrp=s.out_is_xrp)
            take_in = round_in_min(take_in_prop, is_xrp=s.in_is_xrp)

            if take_out <= 0 or take_in <= 0:
                new_tier.append(s if s is synth else s)  # no change
                continue

            # Enforce instantaneous quality <= slice quality
            inst_q = take_out / take_in
            if inst_q > s.quality:
                # Reduce OUT to match slice quality (then floor again)
                take_out = round_out_max(take_in * s.quality, is_xrp=s.out_is_xrp)
                if take_out <= 0:
                    new_tier.append(s if s is synth else s)
                    continue

            # Enforce send_max if present (fit budget while keeping quality <= slice quality)
            if eff_send_max is not None:
                remaining_in_budget = eff_send_max - spent_in
                if remaining_in_budget <= 0:
                    need = Decimal(0)
                    break
                if take_in > remaining_in_budget:
                    # Forward recompute under IN budget: choose IN on its grid, derive OUT at anchored quality,
                    # then cap OUT by remaining need and segment capacity, and recompute IN from OUT.
                    allowed_in = round_in_min(remaining_in_budget, is_xrp=s.in_is_xrp)
                    if allowed_in <= 0:
                        need = Decimal(0)
                        break
                    # First pass OUT by quality from allowed IN
                    out_by_quality = round_out_max(allowed_in * s.quality, is_xrp=s.out_is_xrp)
                    # Cap by remaining need and segment capacity
                    capped_out = out_by_quality
                    if capped_out > need:
                        capped_out = round_out_max(need, is_xrp=s.out_is_xrp)
                    if capped_out > s.out_max:
                        capped_out = round_out_max(s.out_max, is_xrp=s.out_is_xrp)
                    if capped_out <= 0:
                        need = Decimal(0)
                        break
                    # Recompute IN from capped OUT at anchored quality, ceil to grid; ensure within budget
                    recomputed_in = round_in_min(capped_out / s.quality, is_xrp=s.in_is_xrp)
                    if recomputed_in > remaining_in_budget:
                        # Final safeguard: shrink OUT to fit budget exactly at anchored quality
                        capped_out = round_out_max(remaining_in_budget * s.quality, is_xrp=s.out_is_xrp)
                        if capped_out <= 0:
                            need = Decimal(0)
                            break
                        recomputed_in = round_in_min(capped_out / s.quality, is_xrp=s.in_is_xrp)
                        if recomputed_in <= 0:
                            need = Decimal(0)
                            break
                    take_in = recomputed_in
                    take_out = capped_out

            if s.src == "AMM":
                iter_amm_in += take_in
                iter_amm_out += take_out

            # Apply
            filled_out += take_out
            spent_in += take_in
            usage[s.src] = usage.get(s.src, Decimal(0)) + take_out
            trace.append({
                "src": s.src,
                "take_out": take_out,
                "take_in": take_in,
                "quality": s.quality,
            })

            # Residual capacity (synthetic not carried to next round)
            remaining_out = s.out_max - take_out
            remaining_in_at_out_max = s.in_at_out_max - take_in
            if remaining_out > 0 and remaining_in_at_out_max > 0 and s is not synth:
                new_seg = Segment(
                    src=s.src,
                    out_max=remaining_out,
                    in_at_out_max=remaining_in_at_out_max,
                    quality=s.quality,
                    in_is_xrp=s.in_is_xrp,
                    out_is_xrp=s.out_is_xrp,
                )
                new_tier.append(new_seg)

            need = target_out - filled_out
            if need <= 0:
                break

        segs = new_tier + rest
        segs = [s for s in segs if s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0]
        segs.sort(key=lambda s: s.quality, reverse=True)
        # Update AMM pool for next iteration
        if after_iteration is not None and (iter_amm_in > 0 or iter_amm_out > 0):
            after_iteration(iter_amm_in, iter_amm_out)
        need = target_out - filled_out

        if eff_send_max is not None and spent_in >= eff_send_max:
            break

    if eff_deliver_min is not None and filled_out < eff_deliver_min:
        raise RouteError("deliver_min not met")

    avg_quality = (filled_out / spent_in) if spent_in > 0 else Decimal(0)
    return RouteResult(
        filled_out=filled_out,
        spent_in=spent_in,
        avg_quality=avg_quality,
        usage=usage,
        trace=trace,
    )