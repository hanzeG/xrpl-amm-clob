import pytest
from decimal import Decimal

from xrpl_router.core.amounts import STAmount, AmountDomainError
from xrpl_router.core.quality import Quality
from xrpl_router.core.ordering import (
    guard_instant_quality_xrp,
    guard_instant_quality,
    apply_quality_floor,
    stable_sort_by_quality,
    prepare_and_order,
    _ceil_div,
)


def _amt(x: str) -> STAmount:
    return STAmount.from_decimal(Decimal(x))


# -----------------------------
# Guardrails: XRP (drops)
# -----------------------------

def test_guard_instant_quality_xrp_increase_when_implied_too_good():
    print("[guard_xrp-increase] out=2000, in_drops=900, quoted rate=2.0 -> expect new in_drops=1000")
    out_take = _amt("2000")
    q = Quality.from_amounts(_amt("2"), _amt("1"))  # rate = 2.0
    new_drops = guard_instant_quality_xrp(out_take, in_drops=900, slice_quality=q)
    print("new_drops ->", new_drops)
    assert new_drops == 1000


def test_guard_instant_quality_xrp_no_change_when_already_ok():
    print("[guard_xrp-nochange] out=2000, in_drops=1000, quoted rate=2.0 -> expect unchanged 1000")
    out_take = _amt("2000")
    q = Quality.from_amounts(_amt("2"), _amt("1"))
    new_drops = guard_instant_quality_xrp(out_take, in_drops=1000, slice_quality=q)
    print("new_drops ->", new_drops)
    assert new_drops == 1000


def test_guard_instant_quality_xrp_zero_out_returns_input():
    print("[guard_xrp-zero_out] out=0, in_drops=777 -> expect unchanged 777")
    out_take = _amt("0")
    q = Quality.from_amounts(_amt("2"), _amt("1"))
    new_drops = guard_instant_quality_xrp(out_take, in_drops=777, slice_quality=q)
    print("new_drops ->", new_drops)
    assert new_drops == 777


def test_guard_instant_quality_xrp_negative_drops_raises():
    print("[guard_xrp-negative] in_drops=-1 -> expect AmountDomainError")
    out_take = _amt("10")
    q = Quality.from_amounts(_amt("2"), _amt("1"))
    with pytest.raises(AmountDomainError):
        guard_instant_quality_xrp(out_take, in_drops=-1, slice_quality=q)


# -----------------------------
# Guardrails: generic (IOU/any asset)
# -----------------------------

def test_guard_instant_quality_increase_when_implied_too_good():
    print("[guard_generic-increase] out=9, in=4, quoted rate=2.0 -> expect new in=4.5")
    out_take = _amt("9")
    in_ceiled = _amt("4")
    q = Quality.from_amounts(_amt("2"), _amt("1"))
    new_in = guard_instant_quality(out_take, in_ceiled, q)
    print("new_in ->", new_in.to_decimal())
    assert new_in.to_decimal() == Decimal("4.5")
    assert (Decimal("9") / new_in.to_decimal()) == Decimal("2")


def test_guard_instant_quality_no_change_when_already_ok():
    print("[guard_generic-nochange] out=8, in=4, quoted rate=2.0 -> expect unchanged 4")
    out_take = _amt("8")
    in_ceiled = _amt("4")
    q = Quality.from_amounts(_amt("2"), _amt("1"))
    new_in = guard_instant_quality(out_take, in_ceiled, q)
    print("new_in ->", new_in.to_decimal())
    assert new_in == in_ceiled


def test_guard_instant_quality_zero_cases():
    print("[guard_generic-zero] out=0 -> return input; in=0 -> return input")
    q = Quality.from_amounts(_amt("2"), _amt("1"))
    # out_take zero -> unchanged
    new_in_a = guard_instant_quality(_amt("0"), _amt("123"), q)
    print("out=0: new_in ->", new_in_a.to_decimal())
    assert new_in_a.to_decimal() == Decimal("123")
    # in_ceiled zero -> unchanged (zero)
    new_in_b = guard_instant_quality(_amt("10"), _amt("0"), q)
    print("in=0: new_in ->", new_in_b.to_decimal())
    assert new_in_b.to_decimal() == Decimal("0")


# -----------------------------
# Sorting and filtering utilities
# -----------------------------
class Quote:
    def __init__(self, name: str, q: Quality, idx: int):
        self.name = name
        self.quality = q
        self.idx = idx
    def __repr__(self) -> str:
        return f"Quote({self.name}, q={self.quality.rate.to_decimal()}, idx={self.idx})"


def _q(rate_str: str) -> Quality:
    r = STAmount.from_decimal(Decimal(rate_str))
    one = STAmount.from_decimal(Decimal("1"))
    return Quality.from_amounts(r, one)


def test_apply_quality_floor_filters_low_quality():
    print("[floor] keep rate >= 2.0; items A=2.2, B=1.8, C=2.0 -> expect [A, C]")
    items = [
        Quote("A", _q("2.2"), 0),
        Quote("B", _q("1.8"), 1),
        Quote("C", _q("2.0"), 2),
    ]
    kept = apply_quality_floor(items, qmin=_q("2.0"), get_quality=lambda x: x.quality)
    print("kept ->", kept)
    assert [x.name for x in kept] == ["A", "C"]


def test_stable_sort_by_quality_desc_then_index():
    print("[stable-sort] sort by rate desc; A=2.2, B=2.0(idx=1), C=2.0(idx=2) -> expect [A,B,C]")
    items = [
        Quote("A", _q("2.2"), 0),
        Quote("B", _q("2.0"), 1),
        Quote("C", _q("2.0"), 2),
    ]
    ordered = stable_sort_by_quality(items, get_quality=lambda x: x.quality, get_index=lambda x: x.idx)
    print("ordered ->", ordered)
    assert [x.name for x in ordered] == ["A", "B", "C"]


def test_prepare_and_order_floor_then_sort():
    print("[prepare-and-order] floor=2.0 then sort; A=2.2, B=1.8, C=2.0 -> expect [A, C]")
    items = [
        Quote("A", _q("2.2"), 0),
        Quote("B", _q("1.8"), 1),
        Quote("C", _q("2.0"), 2),
    ]
    result = prepare_and_order(items, qmin=_q("2.0"), get_quality=lambda x: x.quality, get_index=lambda x: x.idx)
    print("result ->", result)
    assert [x.name for x in result] == ["A", "C"]


# -----------------------------
# Ceil division helper
# -----------------------------

def test_ceil_div_basics_and_errors():
    print("[ceil-div] 10/3 -> 4; 12/4 -> 3; 0/5 -> 0; b<=0 -> ValueError")
    assert _ceil_div(10, 3) == 4
    assert _ceil_div(12, 4) == 3
    assert _ceil_div(0, 5) == 0
    with pytest.raises(ValueError):
        _ceil_div(1, 0)
    with pytest.raises(ValueError):
        _ceil_div(1, -2)
