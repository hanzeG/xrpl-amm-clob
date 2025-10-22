"""
XRPL Router Core
==================

Unified exports for integer-domain primitives and utilities aligned with rippled semantics.
All arithmetic follows XRPL-style normalisation and rounding rules.
Decimal helpers are provided *only* for I/O formatting.

Core exposes XRPAmount (drops) and IOUAmount (fixed-point) as public API.
"""

# NOTE:
#   The `core` package defines the integer-domain primitives and arithmetic used
#   across the router. All computations are performed on XRPAmount and IOUAmount objects
#   following XRPL ledger rules for mantissa/exponent scaling, rounding, and normalization.
#   Decimal functions exist only for I/O formatting and display.

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
    fmt_dec,
    round_in_min,
    round_out_max,
)

# Decimal quanta (formatting helpers)
from .constants import (
    XRP_QUANTUM,
    IOU_QUANTUM,
    QUALITY_QUANTUM,
    DEFAULT_QUANTUM,
)

# Amount primitives and bridges
from .amounts import (
    Amount,
    XRPAmount,
    IOUAmount,
    drops_from_xrp_in,
    drops_from_xrp_out,
    xrp_from_drops,
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

# Core exceptions
from .exc import AmountDomainError, NormalisationError, InvariantViolation

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
    "Amount",
    "XRPAmount",
    "IOUAmount",
    "drops_from_xrp_in",
    "drops_from_xrp_out",
    "xrp_from_drops",
    "round_in_min",
    "round_out_max",
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
    # exceptions
    "AmountDomainError",
    "NormalisationError",
    "InvariantViolation",
]