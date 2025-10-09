"""
Quality (price-like ratio) utilities aligned with XRPL semantics.

Alignment notes:
- Rate/quality is computed as offerOut / offerIn (higher is better),
  following rippled's `getRate(offerOut, offerIn)`.
- The internal representation uses an STAmount rate produced by a
  division that mirrors rippled's `divide()` path (scale by 10^17,
  integer divide, small offset, then normalise) – see STAmount logic
  around `divide()` in rippled.
- No Decimal is used in core math; everything is integer/normalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from .amounts import STAmount



def _ten_pow(n: int) -> int:
    if n < 0:
        raise ValueError("_ten_pow expects non-negative exponent")
    return 10 ** n


def _divide_like_rippled(num: STAmount, den: STAmount) -> STAmount:
    """Compute an STAmount ratio num/den using rippled-like scaling.

    Mirrors the structure of rippled `divide(offerOut, offerIn, noIssue())`:
    - Work on mantissas, scale numerator by 10^17, then integer-divide by
      denominator mantissa.
    - Add a small positive offset before normalisation (rippled adds +5),
      then set exponent as (num.exp - den.exp - 17).
    - Result sign is positive for valid offers; we treat negatives as zero.

    This function assumes both inputs are already canonical STAmounts.
    """
    if den.is_zero():
        return STAmount.zero()
    if num.is_zero():
        return STAmount.zero()

    num_m, num_e, num_s = num.mantissa, num.exponent, num.sign
    den_m, den_e, den_s = den.mantissa, den.exponent, den.sign

    # Only positive rates are meaningful for ordering; if any sign is negative,
    # treat as zero here (align with "worthless" semantics in ordering).
    if num_s <= 0 or den_s <= 0:
        return STAmount.zero()

    # Scale and divide: floor((num_m * 10^17) / den_m) + small offset
    scaled = num_m * _ten_pow(17)
    q = scaled // den_m
    q += 5  # small positive nudge (see rippled divide implementation)

    exp = num_e - den_e - 17
    # Construct and let STAmount normalise (mantissa/exponent bounds).
    return STAmount.from_components(q, exp, 1)


@dataclass(frozen=True)
class Quality:
    """Quality wrapper around an STAmount rate (higher is better).

    The internal `rate` is an STAmount representing offerOut/offerIn.
    Comparisons delegate to STAmount ordering.
    """

    rate: STAmount

    @staticmethod
    def zero() -> "Quality":
        return Quality(STAmount.zero())

    @classmethod
    def from_amounts(cls, offer_out: STAmount, offer_in: STAmount) -> "Quality":
        """Build quality as offerOut / offerIn.

        Aligns with rippled `getRate(offerOut, offerIn)` which uses
        `divide(offerOut, offerIn, noIssue())` internally.
        """
        r = _divide_like_rippled(offer_out, offer_in)
        return cls(r)

    # Ordering: higher rate is better → invert natural __lt__ on STAmount.
    def __lt__(self, other: "Quality") -> bool:  # type: ignore[override]
        return self.rate < other.rate

    def __le__(self, other: "Quality") -> bool:  # type: ignore[override]
        return self.rate <= other.rate

    def __gt__(self, other: "Quality") -> bool:  # type: ignore[override]
        return self.rate > other.rate

    def __ge__(self, other: "Quality") -> bool:  # type: ignore[override]
        return self.rate >= other.rate

    def is_zero(self) -> bool:
        return self.rate.is_zero()

    # Optional helpers
    def mantissa_exponent(self) -> tuple[int, int]:
        """Return (mantissa, exponent) of the underlying rate."""
        return self.rate.mantissa, self.rate.exponent

    def as_fraction(self) -> Fraction:
        """Return rate as an exact Fraction (sign assumed non-negative)."""
        if self.rate.is_zero():
            return Fraction(0, 1)
        m, e = self.rate.mantissa, self.rate.exponent
        if e >= 0:
            return Fraction(m * _ten_pow(e), 1)
        else:
            return Fraction(m, _ten_pow(-e))


__all__ = [
    "Quality",
]
