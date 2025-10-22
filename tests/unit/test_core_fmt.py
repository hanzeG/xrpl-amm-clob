

import pytest
from decimal import Decimal

from xrpl_router.core.amounts import STAmount, AmountDomainError
from xrpl_router.core.quality import Quality
from xrpl_router.core.fmt import (
    amount_to_decimal,
    quality_rate_to_decimal,
    quality_price_to_decimal,
    fmt_dec,
)


def _amt(x: str) -> STAmount:
    return STAmount.from_decimal(Decimal(x))


# -----------------------------
# amount_to_decimal
# -----------------------------

def test_amount_to_decimal_normal_and_zero():
    print("[amount_to_decimal] normal: 1.2345 and zero")
    a = _amt("1.2345")
    z = STAmount.zero()
    print("to_decimal(1.2345) ->", amount_to_decimal(a))
    print("to_decimal(0) ->", amount_to_decimal(z))
    assert amount_to_decimal(a) == Decimal("1.2345")
    assert amount_to_decimal(z) == Decimal("0")


def test_amount_to_decimal_none_and_malformed_raise():
    print("[amount_to_decimal] None and malformed object should raise AmountDomainError")
    class Malformed:
        exponent = -15  # missing mantissa
    with pytest.raises(AmountDomainError):
        amount_to_decimal(None)  # type: ignore[arg-type]
    with pytest.raises(AmountDomainError):
        amount_to_decimal(Malformed())  # type: ignore[arg-type]


def test_amount_to_decimal_negative_mantissa_raises():
    print("[amount_to_decimal] negative mantissa should raise AmountDomainError")
    class FakeAmt:
        mantissa = -1
        exponent = 0
    with pytest.raises(AmountDomainError):
        amount_to_decimal(FakeAmt())  # type: ignore[arg-type]


# -----------------------------
# quality_rate_to_decimal / quality_price_to_decimal
# -----------------------------

def test_quality_rate_and_price_decimal_basic():
    print("[quality_decimals] rate=2 -> price=0.5")
    q = Quality.from_amounts(_amt("2"), _amt("1"))
    rate_dec = quality_rate_to_decimal(q)
    price_dec = quality_price_to_decimal(q)
    print("rate ->", rate_dec, "; price ->", price_dec)
    assert rate_dec == Decimal("2")
    assert price_dec == Decimal("0.5")


def test_quality_rate_to_decimal_missing_or_nonpositive_raises():
    print("[quality_rate_to_decimal] missing rate or non-positive should raise AmountDomainError")
    class NoRate:
        pass
    class NonPositiveRate:
        class R:
            mantissa = 0
            exponent = 0
        rate = R()
    with pytest.raises(AmountDomainError):
        quality_rate_to_decimal(NoRate())  # type: ignore[arg-type]
    with pytest.raises(AmountDomainError):
        quality_rate_to_decimal(NonPositiveRate())  # type: ignore[arg-type]


# -----------------------------
# fmt_dec stability
# -----------------------------

def test_fmt_dec_scientific_formatting():
    print("[fmt_dec] check scientific formatting stability")
    s1 = fmt_dec(Decimal("1"))
    s2 = fmt_dec(Decimal("123456"))
    print("fmt_dec(1) ->", s1)
    print("fmt_dec(123456) ->", s2)
    assert s1 == "1.000000000000000000E+0"
    assert s2 == "1.234560000000000000E+5"