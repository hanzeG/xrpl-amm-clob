"""
Core datatypes used by the router, aligned with XRPL semantics.

These datatypes are intentionally minimal and immutable (where appropriate)
so that ordering and execution logic can remain deterministic and testable.

Notes:
- Amounts use XRPAmount (drops) and IOUAmount (IOU fixed-point). No STAmount bridging.
- `Quality` wraps an IOUAmount-encoded rate (offerOut / offerIn), higher is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Literal

from .amounts import XRPAmount, IOUAmount, Amount
from .quality import Quality


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """Router segment on a single slice/source (integer domain).

    Fields:
    - src: source type tag ("AMM" or "CLOB"). Used for tie-breaks and AMM-residual rules.
    - quality: quoted OUT/IN quality (higher is better). Integer-domain Quality.
    - out_max: maximum OUT available on this slice (Amount = XRPAmount or IOUAmount).
    - in_at_out_max: IN required to exhaust out_max at quoted quality (Amount = XRPAmount or IOUAmount).
    - raw_quality: optional raw quality as originated by the source (often same as quality).
    - source_id: optional identifier for bookkeeping/debugging.
    """

    src: Literal["AMM", "CLOB"]
    quality: Quality
    out_max: Amount
    in_at_out_max: Amount
    raw_quality: Optional[Quality] = None
    source_id: Optional[str] = None

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
    out_taken: Amount
    in_paid: Amount


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
