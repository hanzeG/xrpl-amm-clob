"""
Formatting helpers and Decimal-based grids (non-core arithmetic).

Core arithmetic uses integer/rational types. Decimal here is only for
formatting and convenience (e.g., tests, logs, display).
"""

from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_UP
from typing import Any

from .exc import AmountDomainError
from .constants import XRP_QUANTUM, IOU_QUANTUM, QUALITY_QUANTUM, DEFAULT_QUANTUM
from .amounts import XRPAmount, IOUAmount, Amount
from .quality import Quality

# Debug printing control (formatting layer)
DEBUG_FMT = False

def _dbg(msg: str) -> None:
    if DEBUG_FMT:
        print(msg)


# ---------------------------------------------------------------------------
# Global Decimal precision (formatting only)
# ---------------------------------------------------------------------------

#: Default global precision (number of significant digits) for Decimal-based
#: formatting. This does not affect core arithmetic which uses integers.
DEFAULT_DECIMAL_PRECISION: int = 28
getcontext().prec = DEFAULT_DECIMAL_PRECISION


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_dec(x: Decimal, places: int = 18) -> str:
    """Format a Decimal in scientific notation with fixed fractional digits.

    The output is stable for logs and tests, e.g.:
      Decimal('1')        -> '1.000000000000000000E+0'
      Decimal('1e-9')     -> '1.000000000000000000E-9'
      Decimal('123456')   -> '1.234560000000000000E+5'
    """
    return format(x, f".{places}E")


# ---------------------------------------------------------------------------
# Decimal quantisation helpers (I/O only)
# ---------------------------------------------------------------------------

def quantize_down(x: Decimal, quantum: Decimal) -> Decimal:
    """Quantise down to the grid (won't give more OUT)."""
    if x < 0:
        raise AmountDomainError("negative input not allowed for quantize_down")
    if x == 0:
        return Decimal("0")
    q = (x / quantum).to_integral_value(rounding=ROUND_DOWN)
    return q * quantum


def quantize_up(x: Decimal, quantum: Decimal) -> Decimal:
    """Quantise up to the grid (won't pay less IN)."""
    if x < 0:
        raise AmountDomainError("negative input not allowed for quantize_up")
    if x == 0:
        return Decimal("0")
    q = (x / quantum).to_integral_value(rounding=ROUND_UP)
    return q * quantum


def round_out_max(x: Decimal, *, is_xrp: bool) -> Decimal:
    """Align OUT amounts down to asset grid (Decimal I/O convenience)."""
    return quantize_down(x, XRP_QUANTUM if is_xrp else IOU_QUANTUM)


def round_in_min(x: Decimal, *, is_xrp: bool) -> Decimal:
    """Align IN amounts up to asset grid (Decimal I/O convenience)."""
    return quantize_up(x, XRP_QUANTUM if is_xrp else IOU_QUANTUM)


# ---------------------------------------------------------------------------
# Logging/display conversion helpers for Amounts and Quality
# ---------------------------------------------------------------------------

def amount_to_decimal(a: Amount) -> Decimal:
    """Convert an amount into a Decimal for logging/printing only.

    Supports XRPAmount (drops→XRP) and IOUAmount (mantissa×10^exponent).
    """
    if a is None:
        raise AmountDomainError("amount_to_decimal(): received None")
    if isinstance(a, XRPAmount):
        if a.value < 0:
            raise AmountDomainError("amount_to_decimal(): XRPAmount drops must be >= 0")
        return Decimal(a.value) * XRP_QUANTUM
    if isinstance(a, IOUAmount):
        if a.mantissa < 0:
            raise AmountDomainError("amount_to_decimal(): mantissa cannot be negative in non-negative domain")
        if a.mantissa == 0:
            return Decimal(0)
        _dbg(f"amount_to_decimal: m={a.mantissa}, e={a.exponent}")
        return Decimal(a.mantissa) * (Decimal(10) ** a.exponent)
    raise AmountDomainError("amount_to_decimal(): unsupported amount type")

def quality_rate_to_decimal(q: Quality) -> Decimal:
    """Return Decimal form of taker quality (OUT/IN). Strictly requires Quality."""
    if not isinstance(q, Quality):
        raise AmountDomainError("quality_rate_to_decimal(): expected Quality")
    r = q.rate
    m = r.mantissa
    e = r.exponent
    if m <= 0:
        raise AmountDomainError("quality_rate_to_decimal(): non-positive quality not allowed")
    _dbg(f"quality_rate_to_decimal: m={m}, e={e}")
    return Decimal(m) * (Decimal(10) ** e)

def quality_price_to_decimal(q: Quality) -> Decimal:
    """Return Decimal price (IN/OUT) as the reciprocal of Quality.rate, for display only."""
    if not isinstance(q, Quality):
        raise AmountDomainError("quality_price_to_decimal(): expected Quality")
    r = quality_rate_to_decimal(q)
    # r must be > 0 per quality_rate_to_decimal
    return Decimal(1) / r


__all__ = [
    "DEFAULT_DECIMAL_PRECISION",
    "XRP_QUANTUM",
    "IOU_QUANTUM",
    "QUALITY_QUANTUM",
    "DEFAULT_QUANTUM",
    "fmt_dec",
    "quantize_down",
    "quantize_up",
    "round_out_max",
    "round_in_min",
    "amount_to_decimal",
    "quality_rate_to_decimal",
    "quality_price_to_decimal",
]