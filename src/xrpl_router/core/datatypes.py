"""
Core datatypes used by the router, aligned with XRPL semantics.

These datatypes are intentionally minimal and immutable (where appropriate)
so that ordering and execution logic can remain deterministic and testable.

Notes:
- Amounts use `STAmount` (IOU fixed-point). XRP amounts are handled as integer drops by callers or bridged into `STAmount` for quality.
- `Quality` wraps an STAmount-encoded rate (offerOut / offerIn), higher is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from .amounts import STAmount
from .quality import Quality


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """A single executable segment/offer with a quoted slice quality.

    Fields:
    - id: stable insertion index for deterministic tie-breaking.
    - quality: quoted/bucketed quality (higher is better).
    - out_max: the maximum OUT the segment can deliver for the current slice.
    - in_at_out_max: the IN required (ceiled) at `out_max`.
    - in_is_xrp / out_is_xrp: flags for caller-side bridging decisions.

    The pair (in_at_out_max, out_max) must be non-zero and of compatible types
    when used in arithmetic. XRP-side IN/OUT handling is performed by callers
    that know the asset context.
    """

    id: int
    quality: Quality
    out_max: STAmount
    in_at_out_max: STAmount
    in_is_xrp: bool = False
    out_is_xrp: bool = False

    def is_usable(self) -> bool:
        """Return True if the segment can be taken for a positive slice."""
        return (not self.out_max.is_zero()) and (not self.in_at_out_max.is_zero())

    def implied_quality(self) -> Quality:
        """Compute implied quality from the (in, out) pair at `out_max`.

        This mirrors the definition used for ranking: offerOut / offerIn.
        """
        return Quality.from_amounts(self.out_max, self.in_at_out_max)


# ---------------------------------------------------------------------------
# Execution report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fill:
    """A single executed slice on a given segment.

    The pair (in_paid, out_taken) should reflect the final amounts after
    guardrails and rounding have been applied.
    """

    segment_id: int
    out_taken: STAmount
    in_paid: STAmount


@dataclass
class ExecutionReport:
    """A collection of executed fills for a route evaluation step.

    Aggregation (e.g., summing totals across heterogeneous assets) is not
    performed here; callers may compute per-asset totals as needed.
    """

    fills: List[Fill] = field(default_factory=list)

    def add(self, fill: Fill) -> None:
        self.fills.append(fill)

    def extend(self, fills: Iterable[Fill]) -> None:
        self.fills.extend(fills)

    def is_empty(self) -> bool:
        return not self.fills


# ---------------------------------------------------------------------------
# Route result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteResult:
    """Output of ordering + selection: the ordered segments ready for consume.

    This structure carries only ordering; execution state lives in
    `ExecutionReport`. Keeping these separate simplifies testing.
    """

    ordered: List[Segment]


__all__ = [
    "Segment",
    "Fill",
    "ExecutionReport",
    "RouteResult",
]
