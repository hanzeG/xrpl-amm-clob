

"""
XRPL Router Core
================

Unified exports for core primitives and utilities aligned with rippled semantics.
This module re-exports stable APIs from submodules for convenient imports.

Numerical convention: IOU STAmount encodes values as mantissa * 10^exponent with
mantissa in [1e15, 1e16-1]. XRP uses integer drops at the I/O boundary.
"""

# Integer-domain constants (rippled-aligned)
from .constants import (
    ST_MANTISSA_DIGITS,
    ST_MANTISSA_MIN,
    ST_MANTISSA_MAX,
    ST_EXP_MIN,
    ST_EXP_MAX,
    DROPS_PER_XRP,
)

# Decimal formatting helpers (non-core arithmetic)
from .fmt import (
    DEFAULT_DECIMAL_PRECISION,
    XRP_QUANTUM,
    IOU_QUANTUM,
    QUALITY_QUANTUM,
    DEFAULT_QUANTUM,
    fmt_dec,
)

# Amount primitives and bridges
from .amounts import (
    STAmount,
    st_from_drops,
)

# Quality (price-like ratio)
from .quality import (
    Quality,
)

# Ordering utilities: guardrails, floors, stable sort
from .ordering import (
    guard_instant_quality,
    guard_instant_quality_xrp,
    apply_quality_floor,
    stable_sort_by_quality,
    prepare_and_order,
)

# Core datatypes for routing/execution
from .datatypes import (
    Segment,
    Fill,
    ExecutionReport,
    RouteResult,
)

__all__ = [
    # constants
    "ST_MANTISSA_DIGITS",
    "ST_MANTISSA_MIN",
    "ST_MANTISSA_MAX",
    "ST_EXP_MIN",
    "ST_EXP_MAX",
    "DROPS_PER_XRP",
    # fmt
    "DEFAULT_DECIMAL_PRECISION",
    "XRP_QUANTUM",
    "IOU_QUANTUM",
    "QUALITY_QUANTUM",
    "DEFAULT_QUANTUM",
    "fmt_dec",
    # amounts
    "STAmount",
    "st_from_drops",
    # quality
    "Quality",
    # ordering
    "guard_instant_quality",
    "guard_instant_quality_xrp",
    "apply_quality_floor",
    "stable_sort_by_quality",
    "prepare_and_order",
    # datatypes
    "Segment",
    "Fill",
    "ExecutionReport",
    "RouteResult",
]