"""
Amount primitives: XRP drops bridge and IOU STAmount (mantissa/exponent/sign).

- XRP: integers in drops at IO boundary; Decimal only for display.
- IOU: integer fixed-point with ~16 significant digits and bounded exponent.
- Rounding semantics: IN rounds up, OUT rounds down; IOU step = 10^exponent.

Core behaviours are aligned with XRPL's IOUAmount/STAmount model (mantissa/exponent
normalisation and bounded exponents). This module avoids Decimal in STAmount math.

# Alignment notes:
# - IOU normalization and bounds follow rippled IOUAmount.cpp (minMantissa=1e15, maxMantissa=1e16-1, minExponent=-96, maxExponent=80).
# - Division rounding semantics mirror STAmount::divRoundImpl (roundUp = away-from-zero, roundDown = toward-zero).
"""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Tuple
from dataclasses import dataclass

from .constants import (
    ST_MANTISSA_MIN,
    ST_MANTISSA_MAX,
    ST_EXP_MIN,
    ST_EXP_MAX,
)
from .fmt import (
    XRP_QUANTUM,
    IOU_QUANTUM,  # Decimal I/O helpers only; core math does not rely on it
)


# ----------------------------
# Decimal quantisation helpers (I/O only)
# ----------------------------

def quantize_down(x: Decimal, quantum: Decimal) -> Decimal:
    """Quantise down to the grid (won't give more OUT)."""
    if x <= 0:
        return Decimal("0")
    q = (x / quantum).to_integral_value(rounding=ROUND_DOWN)
    return q * quantum


def quantize_up(x: Decimal, quantum: Decimal) -> Decimal:
    """Quantise up to the grid (won't pay less IN)."""
    if x <= 0:
        return Decimal("0")
    q = (x / quantum).to_integral_value(rounding=ROUND_UP)
    return q * quantum


def round_out_max(x: Decimal, *, is_xrp: bool) -> Decimal:
    """Align OUT amounts down to asset grid (Decimal I/O convenience)."""
    return quantize_down(x, XRP_QUANTUM if is_xrp else IOU_QUANTUM)


def round_in_min(x: Decimal, *, is_xrp: bool) -> Decimal:
    """Align IN amounts up to asset grid (Decimal I/O convenience)."""
    return quantize_up(x, XRP_QUANTUM if is_xrp else IOU_QUANTUM)


# ----------------------------
# XRP ↔ drops bridge (IO)
# ----------------------------

def xrp_from_drops(drops: int) -> Decimal:
    """Convert integer drops to XRP Decimal (display/interop)."""
    if drops <= 0:
        return Decimal("0")
    return Decimal(drops) * XRP_QUANTUM


def drops_from_xrp_out(xrp: Decimal) -> int:
    """OUT path: floor XRP to drops (never overpay OUT)."""
    x = round_out_max(xrp, is_xrp=True)
    return int((x / XRP_QUANTUM).to_integral_value(rounding=ROUND_DOWN))


def drops_from_xrp_in(xrp: Decimal) -> int:
    """IN path: ceil XRP to drops (never underpay IN)."""
    x = round_in_min(xrp, is_xrp=True)
    return int((x / XRP_QUANTUM).to_integral_value(rounding=ROUND_UP))


# ----------------------------
# Bridge native drops to STAmount for ratio computations
# ----------------------------

def st_from_drops(drops: int) -> STAmount:
    """Bridge native drops to STAmount for ratio computations.

    Drops < 0 are clamped to 0. Constructs (mantissa=d, exponent=0, sign=+1)
    and lets normalisation canonicalise the representation.
    """
    d = drops if drops > 0 else 0
    if d == 0:
        return STAmount.zero()
    return STAmount.from_components(d, 0, 1)


# ----------------------------
# IOU STAmount (integer fixed-point)
# ----------------------------

def _is_zero_triplet(m: int, e: int, s: int) -> bool:
    return m == 0 or s == 0


def _normalize(m: int, e: int, s: int) -> Tuple[int, int, int]:
    """Normalise mantissa/exponent/sign to canonical range.

    - Mantissa in [ST_MANTISSA_MIN, ST_MANTISSA_MAX]
    - Exponent within [ST_EXP_MIN, ST_EXP_MAX]
    - Zero is canonicalised to (0, 0, 0)
    - Sign is {-1, 0, 1}

    Aligns with rippled IOUAmount.cpp normalize(): if exponent hits min and mantissa is still below range, canonicalise to zero.
    
    Upper-bound overflow is canonicalised to zero here (rippled would throw); this is an intentional divergence for robustness.
    """
    if _is_zero_triplet(m, e, s):
        return 0, 0, 0

    # push sign into s; keep mantissa non-negative
    if m < 0:
        m = -m
        s = -s

    # Scale mantissa into canonical range by adjusting exponent.
    while m != 0 and m < ST_MANTISSA_MIN and e > ST_EXP_MIN:
        m *= 10
        e -= 1
    # If we've hit the exponent lower bound and mantissa is still too small, canonical zero.
    if m != 0 and m < ST_MANTISSA_MIN and e == ST_EXP_MIN:
        return 0, 0, 0
    while m > ST_MANTISSA_MAX:
        m //= 10
        e += 1
        # Upper-bound overflow: canonicalises to zero (rippled throws on overflow).
        if e > ST_EXP_MAX:
            return 0, 0, 0

    # If exponent out of bounds, canonical zero.
    if e < ST_EXP_MIN or e > ST_EXP_MAX:
        return 0, 0, 0

    return m, e, 1 if s > 0 else -1


def _ten_pow(n: int) -> int:
    """Return 10**n for n >= 0 (internal helper)."""
    if n < 0:
        raise ValueError("_ten_pow expects non-negative exponent")
    return 10 ** n


@dataclass(frozen=True)
class STAmount:
    """Fixed-point IOU amount: sign * mantissa * 10^exponent.

    Arithmetic and comparisons are performed in integer domain.
    """

    mantissa: int
    exponent: int
    sign: int = 1  # {-1, 0, 1}

    # ------------- constructors -------------

    @staticmethod
    def zero() -> "STAmount":
        return STAmount(0, 0, 0)

    @classmethod
    def min_positive(cls) -> "STAmount":
        """Smallest positive representable IOU amount.
        Mirrors IOUAmount::minPositiveAmount() -> (minMantissa, minExponent).
        """
        return cls.from_components(ST_MANTISSA_MIN, ST_EXP_MIN, 1)

    @classmethod
    def from_components(cls, mantissa: int, exponent: int, sign: int = 1) -> "STAmount":
        m, e, s = _normalize(mantissa, exponent, sign)
        return cls(m, e, s)

    @classmethod
    def from_decimal(cls, x: Decimal) -> "STAmount":
        """Bridge from Decimal. Result is normalised; directionless (no IO rounding).
        Use round_in_min/round_out_max at IO boundaries to enforce payment semantics.
        """
        if x.is_nan() or x.is_infinite() or x == 0:
            return cls.zero()

        s = -1 if x < 0 else 1
        x_abs = -x if x < 0 else x

        tup = x_abs.normalize().as_tuple()
        digits = int("".join(str(d) for d in tup.digits)) if tup.digits else 0
        exp = tup.exponent  # Decimal exponent for: digits * 10^exp
        if digits == 0:
            return cls.zero()

        # Map to (mantissa, exponent) then normalise to canonical range.
        m, e = digits, exp
        m, e, s = _normalize(m, e, s)
        return cls(m, e, s)

    # ------------- conversions -------------

    def to_decimal(self) -> Decimal:
        """Decimal representation of canonical value (mantissa * 10^exponent), for logs/printing only.
        This is not an I/O rounding view; use round_out_max/round_in_min at boundaries.
        """
        if self.sign == 0 or self.mantissa == 0:
            return Decimal("0")
        value = Decimal(self.mantissa) * (Decimal(10) ** self.exponent)
        return value if self.sign > 0 else -value

    # ------------- predicates -------------

    def is_zero(self) -> bool:
        return self.sign == 0 or self.mantissa == 0

    # ------------- comparisons (integer domain) -------------

    def _cmp_core(self, other: "STAmount") -> int:
        if self.is_zero() and other.is_zero():
            return 0
        if self.sign != other.sign:
            return -1 if self.sign < other.sign else 1

        # Same sign and non-zero
        # Compare by scaling mantissas to a common exponent without Decimal.
        e1, e2 = self.exponent, other.exponent
        m1, m2 = self.mantissa, other.mantissa
        if e1 == e2:
            cmp_mag = (m1 > m2) - (m1 < m2)
        elif e1 > e2:
            # compare m1 * 10^(e1-e2) ? m2
            k = _ten_pow(e1 - e2)
            cmp_mag = (m1 * k > m2) - (m1 * k < m2)
        else:
            # compare m1 ? m2 * 10^(e2-e1)
            k = _ten_pow(e2 - e1)
            cmp_mag = (m1 > m2 * k) - (m1 < m2 * k)

        return cmp_mag if self.sign > 0 else -cmp_mag

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, STAmount):
            return NotImplemented
        return self._cmp_core(other) == 0

    def __lt__(self, other: "STAmount") -> bool:
        return self._cmp_core(other) < 0

    def __le__(self, other: "STAmount") -> bool:
        return self._cmp_core(other) <= 0

    def __gt__(self, other: "STAmount") -> bool:
        return self._cmp_core(other) > 0

    def __ge__(self, other: "STAmount") -> bool:
        return self._cmp_core(other) >= 0

    # ------------- arithmetic (integer domain) -------------

    def _with(self, m: int, e: int, s: int) -> "STAmount":
        m2, e2, s2 = _normalize(m, e, s)
        return STAmount(m2, e2, s2)

    def _add_sub(self, other: "STAmount", sign_other: int) -> "STAmount":
        # Align to the smaller exponent to avoid fractions.
        e = min(self.exponent, other.exponent)
        d1 = self.exponent - e
        d2 = other.exponent - e
        m1 = self.mantissa * _ten_pow(d1)
        m2 = other.mantissa * _ten_pow(d2) * sign_other
        m = self.sign * m1 + other.sign * m2

        if m == 0:
            return STAmount.zero()
        s = 1 if m > 0 else -1
        return self._with(abs(m), e, s)

    def __add__(self, other: "STAmount") -> "STAmount":
        return self._add_sub(other, +1)

    def __sub__(self, other: "STAmount") -> "STAmount":
        return self._add_sub(other, -1)

    def mul_by_scalar(self, k: int) -> "STAmount":
        """Multiply by integer scalar; result normalised."""
        if k == 0 or self.is_zero():
            return STAmount.zero()
        s = self.sign if k > 0 else -self.sign
        return self._with(abs(self.mantissa) * abs(k), self.exponent, s)

    def div_by_scalar_down(self, k: int) -> "STAmount":
        """Divide by integer scalar, rounding toward zero in value (DOWN-style).
        Implements exponent-aware scaling: if mantissa < |k|, scale mantissa ×=10 and exponent -=1
        until division is representable or exponent hits ST_EXP_MIN. Then truncate toward zero.
        """
        if k == 0 or self.is_zero():
            return STAmount.zero()
        den = abs(k)
        s = 1 if (self.sign > 0) == (k > 0) else -1
        m = abs(self.mantissa)
        e = self.exponent

        # Scale up mantissa (by powers of 10) while we can, to keep precision
        while m != 0 and m < den and e > ST_EXP_MIN:
            m *= 10
            e -= 1

        q, r = divmod(m, den)
        # Toward zero: never bump q
        if q == 0:
            # Result rounds to zero toward 0
            return STAmount.zero()
        return self._with(q, e, s)

    def div_by_scalar_up(self, k: int) -> "STAmount":
        """Divide by integer scalar, rounding away from zero in value (UP-style).
        Implements exponent-aware scaling: if mantissa < |k|, scale mantissa ×=10 and exponent -=1
        until division is representable or exponent hits ST_EXP_MIN. Then round away from zero.
        """
        if k == 0 or self.is_zero():
            return STAmount.zero()
        den = abs(k)
        s = 1 if (self.sign > 0) == (k > 0) else -1
        m = abs(self.mantissa)
        e = self.exponent

        # Scale up mantissa (by powers of 10) while we can, to keep precision
        while m != 0 and m < den and e > ST_EXP_MIN:
            m *= 10
            e -= 1

        q, r = divmod(m, den)
        # Away from zero: bump if there's a remainder
        if r != 0:
            q += 1

        if q == 0:
            # If still zero after rounding away-from-zero, return the smallest magnitude
            if s > 0:
                return STAmount.min_positive()
            else:
                # Mirror XRPL: negative min positive when rounding away from zero on negative
                mp = STAmount.min_positive()
                return STAmount.from_components(mp.mantissa, mp.exponent, -1)

        return self._with(q, e, s)


__all__ = [
    "quantize_down",
    "quantize_up",
    "round_out_max",
    "round_in_min",
    "xrp_from_drops",
    "drops_from_xrp_out",
    "drops_from_xrp_in",
    "st_from_drops",
    "STAmount",
]
