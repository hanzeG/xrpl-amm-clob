from __future__ import annotations

from enum import Enum
from decimal import Decimal
from typing import Iterable, Optional, Callable, List

from .core import Segment, RouteResult
from .path_builder import route, RouteConfig
from .amm_context import AMMContext


class ExecutionMode(Enum):
    """Execution modes for apples-to-apples efficiency comparison."""
    CLOB_ONLY = "clob_only"
    AMM_ONLY = "amm_only"
    HYBRID = "hybrid"


def run_trade_mode(
    mode: ExecutionMode,
    *,
    target_out: Decimal,
    segments: Iterable[Segment] | List[Segment],
    send_max: Optional[Decimal] = None,
    deliver_min: Optional[Decimal] = None,
    limit_quality: Optional[Decimal] = None,
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[Segment]]] = None,
    amm_curve: Optional[Callable[[Decimal], Iterable[Segment]]] = None,
    amm_context: Optional[AMMContext] = None,
    apply_sink: Optional[Callable[[Decimal, Decimal], None]] = None,
) -> RouteResult:
    """Run a single trade under a chosen mode.

    Modes:
      - CLOB_ONLY: disable AMM sources (no anchor, no curve). Uses provided CLOB segments.
      - AMM_ONLY: ignore CLOB segments; allow AMM via anchor/curve (usually curve-only when no CLOB top).
        For AMM_ONLY, the `apply_sink` argument is forwarded as `after_iteration` to `route()` to enable
        fee metering or other side-effects.
      - HYBRID: allow both CLOB segments and AMM sources.
    """
    segs_list = list(segments)

    if mode is ExecutionMode.CLOB_ONLY:
        # Use only CLOB segments; disable AMM hooks.
        return route(
            target_out=target_out,
            segments=segs_list,
            config=RouteConfig(preserve_quality_on_limit=True),
            send_max=send_max,
            deliver_min=deliver_min,
            limit_quality=limit_quality,
            amm_anchor=None,
            amm_curve=None,
            amm_context=amm_context,
            after_iteration=None,
        )

    if mode is ExecutionMode.AMM_ONLY:
        # Ignore CLOB segments; pass empty list so router bootstraps via amm_curve/anchor.
        return route(
            target_out=target_out,
            segments=[],
            config=RouteConfig(preserve_quality_on_limit=True),
            send_max=send_max,
            deliver_min=deliver_min,
            limit_quality=limit_quality,
            amm_anchor=amm_anchor,
            amm_curve=amm_curve,
            amm_context=amm_context,
            after_iteration=apply_sink,
        )

    # HYBRID (default): allow both CLOB and AMM sources.
    return route(
        target_out=target_out,
        segments=segs_list,
        config=RouteConfig(preserve_quality_on_limit=True),
        send_max=send_max,
        deliver_min=deliver_min,
        limit_quality=limit_quality,
        amm_anchor=amm_anchor,
        amm_curve=amm_curve,
        amm_context=amm_context,
        after_iteration=None,
    )