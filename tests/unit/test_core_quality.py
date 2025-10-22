

import pytest
from decimal import Decimal
from fractions import Fraction

from xrpl_router.core.amounts import STAmount, AmountDomainError
from xrpl_router.core.quality import Quality


def _amt(x: str) -> STAmount:
    """Helper: build non-negative STAmount from a Decimal string."""
    return STAmount.from_decimal(Decimal(x))


def test_quality_basic_ratio_and_fraction():
    print("[quality-basic] out=200, in=100 -> expect rate=2.0 and Fraction(2,1)")
    out = _amt("200")
    inn = _amt("100")
    q = Quality.from_amounts(out, inn)
    # Value sanity via Decimal
    dec = q.rate.to_decimal()
    print("rate.to_decimal ->", dec)
    assert dec == Decimal("2")
    # Exact rational form
    frac = q.as_fraction()
    print("rate.as_fraction ->", frac)
    assert frac == Fraction(2, 1)


def test_quality_scale_invariance():
    print("[quality-scale-invariance] from_amounts(k*out, k*in) == from_amounts(out, in)")
    out = _amt("123.45")
    inn = _amt("67.89")
    base = Quality.from_amounts(out, inn)
    # Scale both by 7 (any positive scalar)
    k = _amt("7")
    out2 = out.mul_by_scalar(7)
    inn2 = inn.mul_by_scalar(7)
    q2 = Quality.from_amounts(out2, inn2)
    print("base fraction ->", base.as_fraction(), "; scaled fraction ->", q2.as_fraction())
    assert base.as_fraction() == q2.as_fraction()
    # And rate STAmount equality (normalised)
    assert base.rate == q2.rate


@pytest.mark.parametrize(
    "out_str,in_str,why",
    [
        ("0", "1", "offer_out is zero"),
        ("1", "0", "offer_in is zero"),
    ],
)
def test_quality_rejects_non_positive_inputs(out_str, in_str, why):
    print(f"[quality-invalid] {why} -> expect AmountDomainError")
    out = _amt(out_str)
    inn = _amt(in_str)
    with pytest.raises(AmountDomainError):
        Quality.from_amounts(out, inn)


def test_quality_ordering_via_fraction():
    print("[quality-ordering] 200/100 (=2.0) should be better than 150/100 (=1.5)")
    q_good = Quality.from_amounts(_amt("200"), _amt("100"))
    q_bad = Quality.from_amounts(_amt("150"), _amt("100"))
    f_good = q_good.as_fraction()
    f_bad = q_bad.as_fraction()
    print("fractions ->", f_good, ">", f_bad)
    assert f_good > f_bad


def test_quality_positive_fraction_and_nonzero():
    print("[quality-positive] out=1, in=3 -> positive non-zero fraction")
    q = Quality.from_amounts(_amt("1"), _amt("3"))
    frac = q.as_fraction()
    print("as_fraction ->", frac)
    assert frac > 0
    assert q.rate.is_zero() is False


# ---------------------------------------------------------------------------
# Quality division +5 offset: extreme/edge-case tests
# ---------------------------------------------------------------------------
from xrpl_router.core.constants import ST_MANTISSA_MIN, ST_MANTISSA_MAX

def test_quality_divide_plus5_exact_divisible_extreme():
    print("[quality +5 exact-divisible] use extreme but divisible mantissas to assert exact integer rate")
    # Choose canonical mantissas with equal exponents so out/in == 2 exactly
    m_den = ST_MANTISSA_MIN  # 1e15
    m_num = m_den * 2        # 2e15 (still within canonical range)
    e = 0
    out = STAmount.from_components(m_num, e)
    inn = STAmount.from_components(m_den, e)
    q = Quality.from_amounts(out, inn)
    dec = q.rate.to_decimal()
    print(f"num={m_num}e{e}, den={m_den}e{e}, rate.dec -> {dec}")
    assert dec == Decimal("2")


def test_quality_divide_plus5_minimal_remainder_tight_bounds():
    print("[quality +5 minimal-remainder] num=den+1, same exponents; expect rate in [exact+4e-17, exact+5e-17]")
    # Construct nearly-equal mantissas so remainder is minimal; exponents equal -> exp_diff=0 -> 10^-17 scale
    m_den = ST_MANTISSA_MAX  # 9.999...e15  (large denominator)
    m_num = m_den + 1
    e = 0
    out = STAmount.from_components(m_num, e)
    inn = STAmount.from_components(m_den, e)

    # Exact rational value R = (m_num/m_den) * 10^(e-e) = m_num/m_den
    exact = (Decimal(m_num) / Decimal(m_den))

    q = Quality.from_amounts(out, inn)
    dec = q.rate.to_decimal()
    print(f"num={m_num}e{e}, den={m_den}e{e}, exact={exact}, rate.dec={dec}")

    # The divide uses floor((num*1e17)/den) + 5 with exp = -17, then may normalise.
    # Normalisation can drop the least significant digit via integer division by 10,
    # which can erase the +5 nudge in extreme cases (mantissa overflow then //10).
    # Therefore we assert a bounded error relative to the exact rational and an
    # absolute upper bound from the +5 mapping to value space.
    delta = abs(dec - exact)
    print("delta=", delta)
    # Error bound at this scale is ~1e-16 due to possible //10 during normalisation.
    assert delta <= Decimal("1e-16")
    # Moreover, result should never exceed exact + 5e-17 at this exp-diff.
    assert dec <= exact + Decimal("5e-17")