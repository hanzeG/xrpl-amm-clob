"""
XRPL Router Core Constants (integer domain)
==========================================

Only rippled-aligned integer constants live here. All fixed-point/formatting
helpers that rely on Decimal are moved to `fmt.py`.
"""

# ---------------------------------------------------------------------------
# STAmount (IOU) canonical ranges and XRP integer bridge
# ---------------------------------------------------------------------------

#: XRPL STAmount mantissa uses 16 significant digits (see IOUAmount.cpp).
ST_MANTISSA_DIGITS: int = 16
ST_MANTISSA_MIN: int = 10 ** (ST_MANTISSA_DIGITS - 1)   # 1e15
ST_MANTISSA_MAX: int = (10 ** ST_MANTISSA_DIGITS) - 1   # 9.999...e15

#: Allowed exponent range for STAmount (power of 10).
ST_EXP_MIN: int = -96
ST_EXP_MAX: int = 80

#: Integer bridge: number of drops per 1 XRP.
DROPS_PER_XRP: int = 1_000_000


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

__all__ = [
    "ST_MANTISSA_DIGITS",
    "ST_MANTISSA_MIN",
    "ST_MANTISSA_MAX",
    "ST_EXP_MIN",
    "ST_EXP_MAX",
    "DROPS_PER_XRP",
]