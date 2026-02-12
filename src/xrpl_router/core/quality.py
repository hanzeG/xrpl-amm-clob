"""
Quality (price-like ratio) utilities aligned with XRPL semantics (non-negative domain).

Alignment notes:
- Rate/quality is computed as offerOut / offerIn (higher is better), following rippled's getRate.
- Core math is integer-only: scale by 10^17, integer divide, small +5 nudge, then normalise.
- Inputs must be strictly positive XRPAmount or IOUAmount; zero/negative inputs are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .amounts import IOUAmount, XRPAmount, Amount
from .exc import AmountDomainError

# Rippled-style scaling constant for quality computation (scale by 10^17, then +5 nudge).
TEN_POW_17 = 10 ** 17

# Debug printing control
DEBUG_QUALITY = False

def _dbg(msg: str) -> None:
    if DEBUG_QUALITY:
        print(msg)


# --- New helper: SPQ as Fraction ---
def spq_frac_from_units(x_u: int, y_u: int, keep_fee_num: int, fee_den: int) -> Fraction:
    """Return SPQ ≈ (y/x) * (keep_fee_num/fee_den) as an exact Fraction.
    If any inputs are non-positive, returns 0.
    """
    if x_u <= 0 or y_u <= 0 or keep_fee_num <= 0 or fee_den <= 0:
        return Fraction(0, 1)
    return Fraction(y_u * keep_fee_num, x_u * fee_den)


def _divide_like_rippled(num: Amount, den: Amount) -> IOUAmount:
    """Compute a fixed-point IOUAmount ratio num/den using rippled-like scaling.

    - Work on mantissas, scale numerator by 10^17, integer-divide by denominator mantissa.
    - Add a small positive offset (+5), then set exponent as (num.exp - den.exp - 17).
    - Inputs must be strictly positive.
    """
    # Extract (mantissa, exponent) from amounts
    if isinstance(num, XRPAmount):
        if num.value <= 0:
            raise AmountDomainError("quality division requires strictly positive inputs")
        num_m, num_e = num.value, 0
    else:
        if num.mantissa <= 0:
            raise AmountDomainError("quality division requires strictly positive inputs")
        num_m, num_e = num.mantissa, num.exponent

    if isinstance(den, XRPAmount):
        if den.value <= 0:
            raise AmountDomainError("quality division requires strictly positive inputs")
        den_m, den_e = den.value, 0
    else:
        if den.mantissa <= 0:
            raise AmountDomainError("quality division requires strictly positive inputs")
        den_m, den_e = den.mantissa, den.exponent

    _dbg(f"quality.divide: num_m={num_m}, num_e={num_e}, den_m={den_m}, den_e={den_e}")
    scaled = num_m * TEN_POW_17
    q = scaled // den_m
    q += 5  # small positive nudge (see rippled divide implementation)
    exp = num_e - den_e - 17
    _dbg(f"quality.divide: q={q}, exp={exp}")
    return IOUAmount.from_components(q, exp)


@dataclass(frozen=True)
class Quality:
    """Quality wrapper around an IOUAmount rate (higher is better).

    The internal `rate` is an IOUAmount representing offerOut/offerIn.
    Comparisons delegate to IOUAmount ordering.
    """

    rate: IOUAmount

    @staticmethod
    def zero() -> "Quality":
        return Quality(IOUAmount.zero())

    @classmethod
    def from_amounts(cls, offer_out: Amount, offer_in: Amount) -> "Quality":
        """Build quality as offerOut / offerIn.

        Aligns with rippled `getRate(offerOut, offerIn)` which uses
        `divide(offerOut, offerIn, noIssue())` internally.
        """
        # Validate inputs non-zero
        if isinstance(offer_out, XRPAmount):
            if offer_out.value == 0:
                raise AmountDomainError("Quality requires offer_out>0 and offer_in>0")
        else:
            if offer_out.is_zero():
                raise AmountDomainError("Quality requires offer_out>0 and offer_in>0")

        if isinstance(offer_in, XRPAmount):
            if offer_in.value == 0:
                raise AmountDomainError("Quality requires offer_out>0 and offer_in>0")
        else:
            if offer_in.is_zero():
                raise AmountDomainError("Quality requires offer_out>0 and offer_in>0")

        r = _divide_like_rippled(offer_out, offer_in)
        return cls(r)

    # Ordering: higher rate is better; comparisons delegate directly to IOUAmount (no inversion).
    def __lt__(self, other: "Quality") -> bool:  # type: ignore[override]
        if not isinstance(other, Quality):
            return NotImplemented
        return self.rate < other.rate

    def __le__(self, other: "Quality") -> bool:  # type: ignore[override]
        if not isinstance(other, Quality):
            return NotImplemented
        return self.rate <= other.rate

    def __gt__(self, other: "Quality") -> bool:  # type: ignore[override]
        if not isinstance(other, Quality):
            return NotImplemented
        return self.rate > other.rate

    def __ge__(self, other: "Quality") -> bool:  # type: ignore[override]
        if not isinstance(other, Quality):
            return NotImplemented
        return self.rate >= other.rate

    def is_zero(self) -> bool:
        return self.rate.is_zero()

    # Optional helpers
    def mantissa_exponent(self) -> tuple[int, int]:
        """Return (mantissa, exponent) of the underlying rate."""
        return self.rate.mantissa, self.rate.exponent

    def as_fraction(self) -> Fraction:
        """Return rate as an exact Fraction (non-negative domain)."""
        if self.rate.is_zero():
            return Fraction(0, 1)
        m, e = self.rate.mantissa, self.rate.exponent
        if e >= 0:
            return Fraction(m * (10 ** e), 1)
        else:
            return Fraction(m, 10 ** (-e))


def quality_eps_for_state_frac(x_u: int, y_u: int, keep_fee_num: int, fee_den: int) -> Fraction:
    """Return ε ≈ keep_fee * (1/x + 1/y) as an exact Fraction.
    = (keep_fee_num/fee_den) * ((x_u + y_u)/(x_u*y_u)) in the non-negative domain.
    If any inputs are non-positive, returns 0.
    """
    if x_u <= 0 or y_u <= 0 or keep_fee_num <= 0 or fee_den <= 0:
        return Fraction(0, 1)
    return Fraction(keep_fee_num * (x_u + y_u), fee_den * x_u * y_u)


__all__ = [
    "Quality",
    "quality_eps_for_state_frac",
    "spq_frac_from_units",
]
