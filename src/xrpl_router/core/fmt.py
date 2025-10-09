"""
Formatting helpers and Decimal-based grids (non-core arithmetic).

Core arithmetic uses integer/rational types. Decimal here is only for
formatting and convenience (e.g., tests, logs, display).
"""

from decimal import Decimal, getcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .amounts import STAmount  # pragma: no cover
    from .quality import Quality   # pragma: no cover

# ---------------------------------------------------------------------------
# Global Decimal precision (formatting only)
# ---------------------------------------------------------------------------

#: Default global precision (number of significant digits) for Decimal-based
#: formatting. This does not affect core arithmetic which uses integers.
DEFAULT_DECIMAL_PRECISION: int = 28
getcontext().prec = DEFAULT_DECIMAL_PRECISION


# ---------------------------------------------------------------------------
# Quantisation grids (formatting helpers)
# ---------------------------------------------------------------------------

#: Minimum quantisation step for XRP values (1 drop = 1e-6 XRP).
XRP_QUANTUM: Decimal = Decimal("1e-6")

#: Minimum quantisation step for IOU (issued token) values.
IOU_QUANTUM: Decimal = Decimal("1e-15")

#: Minimum quantisation step for quality (OUT/IN rate) values.
QUALITY_QUANTUM: Decimal = Decimal("1e-15")

#: Default quantum to use when asset type is unknown.
DEFAULT_QUANTUM: Decimal = IOU_QUANTUM


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
# Logging/display conversion helpers for STAmount and Quality
# ---------------------------------------------------------------------------

def amount_to_decimal(a: Any) -> Decimal:
    """Convert an integer-domain STAmount into a Decimal for logging/printing only."""
    if a is None or getattr(a, "mantissa", 0) == 0 or getattr(a, "sign", 1) <= 0:
        return Decimal(0)
    m = getattr(a, "mantissa", 0)
    e = getattr(a, "exponent", 0)
    s = getattr(a, "sign", 1)
    val = Decimal(m) * (Decimal(10) ** e)
    return val if s >= 0 else -val

def quality_rate_to_decimal(q: Any) -> Decimal:
    """Return Decimal form of taker quality (OUT/IN)."""
    if q is None or getattr(q, "rate", None) is None:
        return Decimal(0)
    r = q.rate
    m = getattr(r, "mantissa", None)
    e = getattr(r, "exponent", None)
    if m is None or e is None:
        try:
            return Decimal(str(r))
        except Exception:
            return Decimal(0)
    return Decimal(m) * (Decimal(10) ** e)

def quality_price_to_decimal(q: Any) -> Decimal:
    """Return Decimal price (IN/OUT) as the reciprocal of Quality.rate, for display only."""
    r = quality_rate_to_decimal(q)
    return Decimal(0) if r <= 0 else (Decimal(1) / r)


__all__ = [
    "DEFAULT_DECIMAL_PRECISION",
    "XRP_QUANTUM",
    "IOU_QUANTUM",
    "QUALITY_QUANTUM",
    "DEFAULT_QUANTUM",
    "fmt_dec",
    "amount_to_decimal",
    "quality_rate_to_decimal",
    "quality_price_to_decimal",
]