"""Routing interface: merge segments and consume in quality order.

Implements tiered iterations per the whitepaper: each iteration consumes only the
current highest-quality tier (same-quality slices), with proportional scaling on
limits to preserve slice quality. Supports AMM anchoring: if provided, an
`amm_anchor` callback can inject a synthetic AMM segment anchored to the LOB top
quality for the current iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Dict, Callable, Optional

from .core import Segment, RouteResult, quantize_down


# Optional: small error type to make failures explicit.
class RouteError(Exception):
    """Raised when routing cannot proceed (e.g., no segments, invalid target)."""


@dataclass(frozen=True)
class RouteConfig:
    """Router configuration knobs.

    preserve_quality_on_limit: keep slice quality constant when partially
    consuming a segment (multi-path semantics). If False, single-path semantics
    may recompute via source curve (added in later milestones).
    """
    preserve_quality_on_limit: bool = True


def route(target_out: Decimal,
          segments: Iterable[Segment],
          *,
          config: RouteConfig | None = None,
          amm_anchor: Callable[[Decimal, Decimal], Optional[Segment]] | None = None
          ) -> RouteResult:
    """Route a target OUT across segments in descending quality order.

    Parameters
    ----------
    target_out: Decimal
        Desired OUT amount to fill (must be > 0 after quantisation).
    segments: Iterable[Segment]
        Homogeneous quote slices (from CLOB/AMM) to consider.
    config: RouteConfig | None
        Router behaviour switches; default preserves slice quality on limits.
    amm_anchor: callable or None
        If provided, called once per iteration with (q_lob_top, need) and may
        return a synthetic AMM segment anchored to q_lob_top.

    Returns
    -------
    RouteResult
        Totals and step-by-step trace of the consumption.

    Notes
    -----
    - Each iteration consumes only the current max-quality tier.
    - Synthetic AMM segment (if returned) is used only in that iteration.
    - When partially consuming a segment, behaviour depends on `config`.
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

    while need > 0 and segs:
        # --- Optional AMM anchoring ---
        q_lob_top = max((s.quality for s in segs if s.src != "AMM"), default=Decimal(0))
        synth: Optional[Segment] = None
        if amm_anchor and q_lob_top > 0:
            synth = amm_anchor(q_lob_top, need)

        # Collect candidates for this iteration
        iter_segs = list(segs)
        if synth is not None:
            iter_segs.append(synth)

        if not iter_segs:
            break

        max_quality = iter_segs[0].quality
        tier: List[Segment] = []
        rest: List[Segment] = []
        for s in iter_segs:
            if s.quality == max_quality:
                tier.append(s)
            else:
                rest.append(s)

        new_tier: List[Segment] = []
        for s in tier:
            if filled_out >= target_out:
                new_tier.append(s)
                continue

            take_out = s.out_max if s.out_max <= need else need
            if take_out <= 0:
                new_tier.append(s)
                continue

            if preserve:
                ratio = take_out / s.out_max
                take_in = quantize_down(s.in_at_out_max * ratio)
            else:
                ratio = take_out / s.out_max
                take_in = quantize_down(s.in_at_out_max * ratio)

            filled_out += take_out
            spent_in += take_in
            usage[s.src] = usage.get(s.src, Decimal(0)) + take_out
            trace.append({
                "src": s.src,
                "take_out": take_out,
                "take_in": take_in,
                "quality": s.quality,
            })

            remaining_out = s.out_max - take_out
            remaining_in_at_out_max = s.in_at_out_max - take_in
            if remaining_out > 0 and remaining_in_at_out_max > 0 and s is not synth:
                new_seg = Segment(
                    src=s.src,
                    out_max=remaining_out,
                    in_at_out_max=remaining_in_at_out_max,
                    quality=s.quality
                )
                new_tier.append(new_seg)

            need = target_out - filled_out
            if filled_out >= target_out:
                break

        segs = new_tier + rest
        segs = [s for s in segs if s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0]
        segs.sort(key=lambda s: s.quality, reverse=True)
        need = target_out - filled_out

    avg_quality = (filled_out / spent_in) if spent_in > 0 else Decimal(0)

    return RouteResult(
        filled_out=filled_out,
        spent_in=spent_in,
        avg_quality=avg_quality,
        usage=usage,
        trace=trace,
    )