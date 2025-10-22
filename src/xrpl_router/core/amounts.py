"""
Amount primitives: XRPAmount (integer drops) and IOUAmount (fixed-point mantissa/exponent).

- XRPAmount: integers in drops at IO boundary; Decimal only for display.
- IOUAmount: integer fixed-point with ~16 significant digits and bounded exponent.
- Non-negative domain: all amounts are ≥ 0; negative values are rejected at input.
- Rounding semantics: IN rounds up, OUT rounds down; IOU step = 10^exponent.

Core behaviours are aligned with XRPL's IOUAmount/STAmount model (mantissa/exponent
normalisation and bounded exponents). STAmount is no longer part of the public API in core.

# Alignment notes:
# - IOU normalization and bounds follow rippled IOUAmount.cpp (minMantissa=1e15, maxMantissa=1e16-1, minExponent=-96, maxExponent=80).
# - Division rounding semantics mirror STAmount::divRoundImpl (roundUp = away-from-zero, roundDown = toward-zero),
#   but final integer rounding is computed in integer domain without Decimal round-trips.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Tuple, Union
from dataclasses import dataclass

from .constants import (
    ST_MANTISSA_MIN,
    ST_MANTISSA_MAX,
    ST_EXP_MIN,
    ST_EXP_MAX,
    XRP_QUANTUM,
)

# Import core exceptions
from .exc import AmountDomainError, NormalisationError, InvariantViolation

# Debug printing control
DEBUG_AMOUNTS = False

def _dbg(msg: str) -> None:
    if DEBUG_AMOUNTS:
        print(msg)


# ----------------------------
# Integer rounding helpers (centralised)
# ----------------------------

def _ceil_div(a: int, b: int) -> int:
    if a < 0 or b <= 0:
        raise AmountDomainError("_ceil_div expects a>=0 and b>0")
    return 0 if a == 0 else -(-a // b)


def _floor_div(a: int, b: int) -> int:
    if a < 0 or b <= 0:
        raise AmountDomainError("_floor_div expects a>=0 and b>0")
    return a // b




# ----------------------------
# XRP primitive (integer drops)
# ----------------------------

@dataclass(frozen=True)
class XRPAmount:
    """Native XRP amount in integer drops (non-negative domain)."""
    value: int

    def __post_init__(self):
        if self.value < 0:
            raise AmountDomainError("XRPAmount must be >= 0 drops")

    def is_zero(self) -> bool:
        return self.value == 0

    def to_decimal(self) -> Decimal:
        return Decimal(self.value) * XRP_QUANTUM

    # Basic arithmetic in integer domain
    def __add__(self, other: "XRPAmount") -> "XRPAmount":
        if not isinstance(other, XRPAmount):
            raise AmountDomainError("XRPAmount arithmetic requires XRPAmount operands")
        return XRPAmount(self.value + other.value)

    def __sub__(self, other: "XRPAmount") -> "XRPAmount":
        if not isinstance(other, XRPAmount):
            raise AmountDomainError("XRPAmount arithmetic requires XRPAmount operands")
        if self.value < other.value:
            raise InvariantViolation("XRP subtraction underflow")
        return XRPAmount(self.value - other.value)

    def mul_by_scalar(self, k: int) -> "XRPAmount":
        if k < 0:
            raise AmountDomainError(f"negative scalar not allowed: k={k}")
        if k == 0 or self.value == 0:
            return XRPAmount(0)
        return XRPAmount(self.value * k)

    def div_by_scalar_down(self, k: int) -> "XRPAmount":
        if k == 0:
            raise ZeroDivisionError("division by zero scalar")
        if k < 0:
            raise AmountDomainError(f"negative scalar not allowed: k={k}")
        return XRPAmount(_floor_div(self.value, k))

    def div_by_scalar_up(self, k: int) -> "XRPAmount":
        if k == 0:
            raise ZeroDivisionError("division by zero scalar")
        if k < 0:
            raise AmountDomainError(f"negative scalar not allowed: k={k}")
        return XRPAmount(_ceil_div(self.value, k))


# ----------------------------
# IOUAmount (integer fixed-point)
# ----------------------------

def _normalize(m: int, e: int) -> Tuple[int, int]:
    """Normalise mantissa/exponent to canonical range (non-negative domain).

    - Mantissa in [ST_MANTISSA_MIN, ST_MANTISSA_MAX]
    - Exponent within [ST_EXP_MIN, ST_EXP_MAX]
    - Zero is canonicalised to (0, 0)
    """
    if m == 0:
        return 0, 0
    if m < 0:
        raise AmountDomainError("mantissa must be >= 0")

    # Scale mantissa into canonical range by adjusting exponent
    while m != 0 and m < ST_MANTISSA_MIN and e > ST_EXP_MIN:
        m *= 10
        e -= 1
    if m != 0 and m < ST_MANTISSA_MIN and e == ST_EXP_MIN:
        return 0, 0

    while m > ST_MANTISSA_MAX:
        m //= 10
        e += 1
        if e > ST_EXP_MAX:
            raise NormalisationError(f"IOUAmount exponent overflow (m={m}, e={e})")

    if e < ST_EXP_MIN or e > ST_EXP_MAX:
        raise NormalisationError(f"IOUAmount exponent out of bounds (m={m}, e={e})")

    return m, e


def _ten_pow(n: int) -> int:
    """Return 10**n for n >= 0 (internal helper)."""
    if n < 0:
        raise ValueError("_ten_pow expects non-negative exponent")
    return 10 ** n


@dataclass(frozen=True)
class IOUAmount:
    """Fixed-point IOU amount: mantissa * 10^exponent (non-negative domain)."""
    mantissa: int
    exponent: int

    # ------------- constructors -------------

    @staticmethod
    def zero() -> "IOUAmount":
        return IOUAmount(0, 0)

    @classmethod
    def min_positive(cls) -> "IOUAmount":
        """Smallest positive representable IOU amount.
        Mirrors IOUAmount::minPositiveAmount() -> (minMantissa, minExponent).
        """
        return cls.from_components(ST_MANTISSA_MIN, ST_EXP_MIN)

    @classmethod
    def from_components(cls, mantissa: int, exponent: int) -> "IOUAmount":
        m, e = _normalize(mantissa, exponent)
        return cls(m, e)

    @classmethod
    def from_decimal(cls, x: Decimal) -> "IOUAmount":
        """Bridge from Decimal (non-negative only). Result is normalised; directionless.
        Use round_in_min/round_out_max at IO boundaries to enforce payment semantics.
        """
        if x.is_nan() or x.is_infinite():
            raise AmountDomainError("invalid Decimal for IOUAmount")
        if x < 0:
            raise AmountDomainError("negative Decimal not allowed for IOUAmount")
        if x == 0:
            return cls.zero()

        x_abs = x
        tup = x_abs.normalize().as_tuple()
        digits = int("".join(str(d) for d in tup.digits)) if tup.digits else 0
        exp = tup.exponent
        if digits == 0:
            return cls.zero()
        m, e = _normalize(digits, exp)
        return cls(m, e)

    # ------------- conversions -------------

    def to_decimal(self) -> Decimal:
        """Decimal representation of canonical value (mantissa * 10^exponent), for logs/printing only.
        This is not an I/O rounding view; use round_out_max/round_in_min at boundaries.
        """
        if self.mantissa == 0:
            return Decimal("0")
        return Decimal(self.mantissa) * (Decimal(10) ** self.exponent)

    # ------------- predicates -------------

    def is_zero(self) -> bool:
        return self.mantissa == 0

    # ------------- comparisons (integer domain) -------------

    def _cmp_core(self, other: "IOUAmount") -> int:
        # Compare magnitudes only; domain is non-negative.
        if self.is_zero() and other.is_zero():
            return 0
        e1, e2 = self.exponent, other.exponent
        m1, m2 = self.mantissa, other.mantissa
        if e1 == e2:
            return (m1 > m2) - (m1 < m2)
        elif e1 > e2:
            k = _ten_pow(e1 - e2)
            return (m1 * k > m2) - (m1 * k < m2)
        else:
            k = _ten_pow(e2 - e1)
            return (m1 > m2 * k) - (m1 < m2 * k)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IOUAmount):
            return NotImplemented
        return self._cmp_core(other) == 0

    def __lt__(self, other: "IOUAmount") -> bool:
        if not isinstance(other, IOUAmount):
            return NotImplemented
        return self._cmp_core(other) < 0

    def __le__(self, other: "IOUAmount") -> bool:
        if not isinstance(other, IOUAmount):
            return NotImplemented
        return self._cmp_core(other) <= 0

    def __gt__(self, other: "IOUAmount") -> bool:
        if not isinstance(other, IOUAmount):
            return NotImplemented
        return self._cmp_core(other) > 0

    def __ge__(self, other: "IOUAmount") -> bool:
        if not isinstance(other, IOUAmount):
            return NotImplemented
        return self._cmp_core(other) >= 0

    # ------------- arithmetic (integer domain) -------------

    def _with(self, m: int, e: int) -> "IOUAmount":
        m2, e2 = _normalize(m, e)
        return IOUAmount(m2, e2)

    def _add_sub(self, other: "IOUAmount", sign_other: int) -> "IOUAmount":
        # Align to the smaller exponent to avoid fractions.
        e = min(self.exponent, other.exponent)
        d1 = self.exponent - e
        d2 = other.exponent - e
        m1 = self.mantissa * _ten_pow(d1)
        m2 = other.mantissa * _ten_pow(d2)

        if sign_other == +1:
            m = m1 + m2
        else:  # subtraction
            if m1 < m2:
                raise InvariantViolation("subtraction underflow would produce negative amount")
            m = m1 - m2

        if m == 0:
            return IOUAmount.zero()
        return self._with(m, e)

    def __add__(self, other: "IOUAmount") -> "IOUAmount":
        if not isinstance(other, IOUAmount):
            raise AmountDomainError("IOUAmount arithmetic requires IOUAmount operands")
        return self._add_sub(other, +1)

    def __sub__(self, other: "IOUAmount") -> "IOUAmount":
        if not isinstance(other, IOUAmount):
            raise AmountDomainError("IOUAmount arithmetic requires IOUAmount operands")
        return self._add_sub(other, -1)

    def mul_by_scalar(self, k: int) -> "IOUAmount":
        """Multiply by integer scalar; non-negative-only semantics."""
        if k < 0:
            raise AmountDomainError(f"negative scalar not allowed: k={k}")
        if k == 0 or self.is_zero():
            return IOUAmount.zero()
        return self._with(self.mantissa * k, self.exponent)

    def div_by_scalar_down(self, k: int) -> "IOUAmount":
        """Divide by integer scalar, rounding toward zero (DOWN-style). Non-negative-only.
        Returns a normalised IOUAmount (mantissa/exponent), preserving XRPL fixed-point semantics.
        """
        if self.is_zero():
            _dbg("div_down: early zero input")
            return IOUAmount.zero()
        if k == 0:
            raise ZeroDivisionError("division by zero scalar")
        if k < 0:
            raise AmountDomainError(f"negative scalar not allowed: k={k}")
        den = k
        m = self.mantissa
        e = self.exponent
        _dbg(f"div_down: start m={m}, e={e}, den={den}")
        # Scale up mantissa if needed to avoid a zero quotient, respecting ST_EXP_MIN.
        while m != 0 and m < den and e > ST_EXP_MIN:
            m *= 10
            e -= 1
            _dbg(f"div_down: scale m={m}, e={e}")
        q, r = divmod(m, den)
        _dbg(f"div_down: q={q}, r={r}, e={e}")
        if q == 0:
            _dbg("div_down: q==0 -> return zero")
            return IOUAmount.zero()
        # Return normalised fixed-point (q, e); do not force exponent to 0.
        return IOUAmount.from_components(q, e)

    def div_by_scalar_up(self, k: int) -> "IOUAmount":
        """Divide by integer scalar, rounding away from zero (UP-style). Non-negative-only.
        Returns a normalised IOUAmount (mantissa/exponent), preserving XRPL fixed-point semantics.
        """
        if self.is_zero():
            _dbg("div_up: early zero input")
            return IOUAmount.zero()
        if k == 0:
            raise ZeroDivisionError("division by zero scalar")
        if k < 0:
            raise AmountDomainError(f"negative scalar not allowed: k={k}")
        den = k
        m = self.mantissa
        e = self.exponent
        _dbg(f"div_up: start m={m}, e={e}, den={den}")
        # Scale up mantissa if needed to avoid a zero quotient, respecting ST_EXP_MIN.
        while m != 0 and m < den and e > ST_EXP_MIN:
            m *= 10
            e -= 1
            _dbg(f"div_up: scale m={m}, e={e}")
        q, r = divmod(m, den)
        if r != 0:
            q += 1
        _dbg(f"div_up: q={q} (post-round), r={r}, e={e}")
        if q == 0:
            _dbg("div_up: q==0 -> return zero")
            return IOUAmount.zero()
        # Return normalised fixed-point (q, e); do not force exponent to 0.
        return IOUAmount.from_components(q, e)

# Unified amount type alias for signatures
Amount = Union["XRPAmount", "IOUAmount"]

#
# ----------------------------
# XRP Decimal bridges (I/O only)
# ----------------------------

def xrp_from_drops(d: int) -> Decimal:
    """Return Decimal XRP from integer drops (I/O/display only)."""
    if not isinstance(d, int):
        raise AmountDomainError("xrp_from_drops: drops must be int")
    if d < 0:
        raise AmountDomainError("xrp_from_drops: drops must be >= 0")
    return Decimal(d) * XRP_QUANTUM


def drops_from_xrp_out(x: Decimal) -> int:
    """OUT-path: floor XRP Decimal to whole drops (won't give more OUT)."""
    if x.is_nan() or x.is_infinite():
        raise AmountDomainError("drops_from_xrp_out: invalid Decimal")
    if x < 0:
        raise AmountDomainError("drops_from_xrp_out: negative not allowed")
    q = (x / XRP_QUANTUM).to_integral_value(rounding=ROUND_DOWN)
    return int(q)


def drops_from_xrp_in(x: Decimal) -> int:
    """IN-path: ceil XRP Decimal to whole drops (won't pay less IN)."""
    if x.is_nan() or x.is_infinite():
        raise AmountDomainError("drops_from_xrp_in: invalid Decimal")
    if x < 0:
        raise AmountDomainError("drops_from_xrp_in: negative not allowed")
    q = (x / XRP_QUANTUM).to_integral_value(rounding=ROUND_UP)
    return int(q)


__all__ = [
    "xrp_from_drops",
    "drops_from_xrp_out",
    "drops_from_xrp_in",
    "XRPAmount",
    "IOUAmount",
    "Amount",
]
