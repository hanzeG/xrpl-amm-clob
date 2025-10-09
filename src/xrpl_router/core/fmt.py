"""
Formatting helpers and Decimal-based grids (non-core arithmetic).

Core arithmetic uses integer/rational types. Decimal here is only for
formatting and convenience (e.g., tests, logs, display).
"""

from decimal import Decimal, getcontext

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


__all__ = [
    "DEFAULT_DECIMAL_PRECISION",
    "XRP_QUANTUM",
    "IOU_QUANTUM",
    "QUALITY_QUANTUM",
    "DEFAULT_QUANTUM",
    "fmt_dec",
]