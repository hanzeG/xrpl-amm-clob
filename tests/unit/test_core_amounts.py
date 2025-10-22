

import math
import pytest
from decimal import Decimal

from xrpl_router.core.amounts import (
    STAmount,
    quantize_down,
    quantize_up,
    xrp_from_drops,
    st_from_drops,
    IOU_QUANTUM,
    XRP_QUANTUM,
    AmountDomainError,
    NormalisationError,
    InvariantViolation,
)
from xrpl_router.core.constants import ST_EXP_MIN, ST_EXP_MAX, ST_MANTISSA_MIN


# -----------------------------
# Quantisation & I/O helpers
# -----------------------------


import pytest

@pytest.mark.parametrize(
    "fn,name,val",
    [
        (quantize_down, "quantize_down", Decimal("-1.23")),
        (quantize_up,   "quantize_up",   Decimal("-0.01")),
    ],
)
def test_quantize_negative_raises(fn, name, val):
    print(f"[{name}-negative] Input={val}, expect AmountDomainError (no clamping to 0)")
    with pytest.raises(AmountDomainError):
        fn(val, IOU_QUANTUM)


def test_quantize_zero_ok():
    print("[quantize-zero] Input=0 IOU, expect 0 for both down/up")
    assert quantize_down(Decimal("0"), IOU_QUANTUM) == Decimal("0")
    assert quantize_up(Decimal("0"), IOU_QUANTUM) == Decimal("0")
    print("quantize_down(0) ->", quantize_down(Decimal("0"), IOU_QUANTUM))
    print("quantize_up(0) ->", quantize_up(Decimal("0"), IOU_QUANTUM))



@pytest.mark.parametrize(
    "call,name",
    [
        (lambda: xrp_from_drops(-1), "xrp_from_drops(-1)"),
        (lambda: st_from_drops(-10), "st_from_drops(-10)"),
        (lambda: STAmount.from_decimal(Decimal("-0.0001")), "STAmount.from_decimal(-0.0001)"),
    ],
)
def test_negative_inputs_rejected(call, name):
    print(f"[negative-inputs] {name} -> expect AmountDomainError")
    with pytest.raises(AmountDomainError):
        call()


# -----------------------------
# STAmount construction & normalisation
# -----------------------------



def test_from_decimal_zero_and_positive():
    print("[STAmount.from_decimal-zero/positive] Inputs: 0 and 123.45; expect zero and exact round-trip")
    z = STAmount.from_decimal(Decimal("0"))
    assert z.is_zero()
    a = STAmount.from_decimal(Decimal("123.45"))
    # round-trip to decimal must be non-negative and close
    assert a.to_decimal() == Decimal("123.45").normalize()
    print("round-trip check: to_decimal(123.45) ->", STAmount.from_decimal(Decimal("123.45")).to_decimal())


def test_normalise_overflow_raises():
    print("[normalize-overflow] Construct with too_big_mantissa near ST_EXP_MAX; expect NormalisationError (no silent zero)")
    # Choose a very large mantissa that will push exponent above ST_EXP_MAX when normalised
    # Start near max exponent already so the loop must overflow.
    too_big_mantissa = 10 ** 40
    with pytest.raises(NormalisationError):
        STAmount.from_components(too_big_mantissa, ST_EXP_MAX)


def test_underflow_at_min_exp_canonical_zero():
    print("[normalize-underflow->zero] Mantissa below ST_MANTISSA_MIN at ST_EXP_MIN; expect canonical zero")
    below_min = max(1, ST_MANTISSA_MIN // 10)
    print("below_min mantissa:", below_min, "exp:", ST_EXP_MIN)
    a = STAmount.from_components(below_min, ST_EXP_MIN)
    assert a.is_zero()


# -----------------------------
# Arithmetic (non-negative domain)
# -----------------------------

def test_subtraction_underflow_raises():
    print("[sub-underflow] 10 - 11, expect InvariantViolation (no negative amounts)")
    a = STAmount.from_decimal(Decimal("10"))
    b = STAmount.from_decimal(Decimal("11"))
    with pytest.raises(InvariantViolation):
        _ = a - b


def test_addition_and_to_decimal_non_negative():
    print("[add-nonnegative] 1.2 + 3.4 -> expect 4.6, non-negative")
    a = STAmount.from_decimal(Decimal("1.2"))
    b = STAmount.from_decimal(Decimal("3.4"))
    c = a + b
    cd = c.to_decimal()
    assert cd == Decimal("4.6")
    assert cd >= 0
    print("sum to_decimal ->", cd)


def test_mul_by_scalar_negative_raises_and_zero_scalar():
    print("[mul-scalar] Input=5; expect negative scalar to raise, zero scalar to yield zero")
    a = STAmount.from_decimal(Decimal("5"))
    with pytest.raises(AmountDomainError):
        _ = a.mul_by_scalar(-3)
    z = a.mul_by_scalar(0)
    assert z.is_zero()
    print("5 * 0 ->", z.to_decimal())



@pytest.mark.parametrize("method", ["down", "up"])
def test_division_by_zero_raises(method):
    print(f"[div-by-zero] a=10; method={method}; expect ZeroDivisionError")
    a = STAmount.from_decimal(Decimal("10"))
    with pytest.raises(ZeroDivisionError):
        getattr(a, f"div_by_scalar_{method}")(0)


def test_division_rounding_down_up():
    print("[div-scalar rounding] a=10; expect 10/3 down=3, up=4 (value-domain rounding)")
    a = STAmount.from_decimal(Decimal("10"))
    d_down = a.div_by_scalar_down(3)
    d_up = a.div_by_scalar_up(3)
    print("a.to_decimal ->", a.to_decimal())
    print("down: (10/3) ->", d_down.to_decimal(), "expected 3")
    print("up:   (10/3) ->", d_up.to_decimal(), "expected 4")
    assert d_down.to_decimal() == Decimal("3")
    assert d_up.to_decimal() == Decimal("4")


@pytest.mark.parametrize("method", ["down", "up"])
def test_division_negative_scalar_raises(method):
    print(f"[div-negative-scalar] a=7; method={method}; expect AmountDomainError")
    a = STAmount.from_decimal(Decimal("7"))
    with pytest.raises(AmountDomainError):
        getattr(a, f"div_by_scalar_{method}")(-2)


# -----------------------------
# XRP conversions sanity
# -----------------------------

def test_xrp_from_drops_zero_and_positive():
    print("[xrp_from_drops] 0 drops -> 0 XRP; 1 drop -> 1e-6 XRP")
    assert xrp_from_drops(0) == Decimal("0")
    assert xrp_from_drops(1) == XRP_QUANTUM  # 1 drop = 1e-6 XRP
    print("xrp_from_drops(1) ->", xrp_from_drops(1))