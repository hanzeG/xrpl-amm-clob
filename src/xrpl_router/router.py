

"""Routing interface: merge segments and consume in quality order.

This module defines the public API and minimal types for the router stage (M1).
Implementation will be added incrementally in subsequent steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Dict

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
          config: RouteConfig | None = None) -> RouteResult:
    """Route a target OUT across segments in descending quality order.

    Parameters
    ----------
    target_out: Decimal
        Desired OUT amount to fill (must be > 0 after quantisation).
    segments: Iterable[Segment]
        Homogeneous quote slices (from CLOB/AMM) to consider.
    config: RouteConfig | None
        Router behaviour switches; default preserves slice quality on limits.

    Returns
    -------
    RouteResult
        Totals and step-by-step trace of the consumption.

    Notes
    -----
    - One iteration consumes at most one segment (quality tier).
    - When partially consuming a segment, behaviour depends on `config`.
    - Quality sorting is stable; ties keep input order.
    """
    # --- Validate inputs ---
    if target_out is None:
        raise RouteError("target_out is None")
    # Quantisation happens at segment construction; here we only need positivity.
    if target_out <= 0:
        raise RouteError("target_out must be > 0")

    # Materialise and filter segments with positive capacity and price.
    segs: List[Segment] = [
        s for s in segments
        if (s.out_max > 0 and s.in_at_out_max > 0 and s.quality > 0)
    ]
    if not segs:
        raise RouteError("no usable segments")

    # Sort by quality (descending); stable for ties.
    segs.sort(key=lambda s: s.quality, reverse=True)

    # Init accumulators.
    filled_out: Decimal = Decimal(0)
    spent_in: Decimal = Decimal(0)
    usage: Dict[str, Decimal] = {}
    trace: List[Dict[str, Decimal]] = []

    preserve = True if config is None else config.preserve_quality_on_limit

    for s in segs:
        if filled_out >= target_out:
            break
        need = target_out - filled_out
        if need <= 0:
            break

        take_out = s.out_max if s.out_max <= need else need
        if take_out <= 0:
            continue

        if preserve:
            # Proportional take keeps slice quality constant.
            ratio = take_out / s.out_max
            take_in = quantize_down(s.in_at_out_max * ratio)
        else:
            # Placeholder for single-path semantics (curve recompute to be added later).
            ratio = take_out / s.out_max
            take_in = quantize_down(s.in_at_out_max * ratio)

        # Accumulate
        filled_out += take_out
        spent_in += take_in
        usage[s.src] = usage.get(s.src, Decimal(0)) + take_out
        trace.append({
            "src": s.src,           # type: ignore[typeddict-item]
            "take_out": take_out,
            "take_in": take_in,
            "quality": s.quality,
        })

        if filled_out >= target_out:
            break

    avg_quality = (filled_out / spent_in) if spent_in > 0 else Decimal(0)

    return RouteResult(
        filled_out=filled_out,
        spent_in=spent_in,
        avg_quality=avg_quality,
        usage=usage,
        trace=trace,
    )