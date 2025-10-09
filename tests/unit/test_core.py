
from decimal import Decimal, getcontext
getcontext().prec = 30

def _rate_to_decimal(rate) -> Decimal:
    """Convert STAmount rate (mantissa/exponent) to Decimal (mantissa * 10^exponent)."""
    return Decimal(rate.mantissa) * (Decimal(10) ** rate.exponent)

def _fmt_q(d: Decimal) -> str:
    """Compact decimal string for human checking (trim trailing zeros)."""
    return format(d, '.12g')

def _banner(title: str):
    print("\n" + "="*12 + f" {title} " + "="*12)

print("\n[DEBUG] Starting core import and constants baseline tests...")

# Stage 1: import self-check
def test_sanity_imports():
    """Verify that core submodules import without circular errors."""
    from xrpl_router.core import (
        STAmount,
        Quality,
        guard_instant_quality,
        guard_instant_quality_xrp,
        Segment,
        Fill,
        ExecutionReport,
        RouteResult,
        ST_MANTISSA_DIGITS,
        ST_EXP_MIN,
        ST_EXP_MAX,
        DROPS_PER_XRP,
    )

    _banner("SANITY: imports")
    print("[DEBUG] Successfully imported all core symbols.")
    # Simple sanity type checks
    assert isinstance(ST_MANTISSA_DIGITS, int)
    assert callable(guard_instant_quality)
    assert callable(guard_instant_quality_xrp)
    assert hasattr(STAmount, "__name__") or hasattr(STAmount, "__class__")
    assert hasattr(Quality, "__name__") or hasattr(Quality, "__class__")


# Stage 2: constants baseline tests
def test_sanity_constants_alignment():
    """Check integer constants match XRPL-aligned values."""
    from xrpl_router.core import (
        ST_MANTISSA_DIGITS,
        ST_EXP_MIN,
        ST_EXP_MAX,
        DROPS_PER_XRP,
    )

    _banner("SANITY: constants baseline")
    print(f"[DEBUG] ST_MANTISSA_DIGITS={ST_MANTISSA_DIGITS}, EXP_RANGE=({ST_EXP_MIN},{ST_EXP_MAX}), DROPS_PER_XRP={DROPS_PER_XRP}")

    assert ST_MANTISSA_DIGITS == 16
    assert ST_EXP_MIN == -96
    assert ST_EXP_MAX == 80
    assert DROPS_PER_XRP == 1_000_000

    # Derived checks

    assert 10 ** (ST_MANTISSA_DIGITS - 1) == 1_000_000_000_000_000
    assert (10 ** ST_MANTISSA_DIGITS) - 1 == 9_999_999_999_999_999


# Stage 3: amounts (normalisation, min_positive, add/sub, drops bridge)

def test_amounts_normalize_overflow_and_scale_up():
    from xrpl_router.core import STAmount, ST_MANTISSA_MIN, ST_MANTISSA_MAX

    _banner("AMOUNTS: normalisation")
    # Overflow mantissa -> expect exponent to increase by 1 and mantissa reduced by 10
    a = STAmount.from_components(ST_MANTISSA_MAX * 10, 0, 1)
    print(
        "[DEBUG] normalization (overflow)\n"
        "  case     : -- overflow mantissa (expect exponent+1) --\n"
        "  input    : m=ST_MANTISSA_MAX*10, e=0\n"
        f"  expected : m≈{ST_MANTISSA_MAX}, e≈1, s=1\n"
        f"  actual   : m={a.mantissa}, e={a.exponent}, s={a.sign}"
    )
    assert a.mantissa == ST_MANTISSA_MAX
    assert a.exponent == 1
    assert a.sign == 1

    # Underflow (can scale up): below min mantissa at exp=0 should scale mantissa*10, exponent-1
    b = STAmount.from_components(ST_MANTISSA_MIN // 10, 0, 1)
    print(
        "[DEBUG] normalization (scale-up)\n"
        "  case     : -- scale-up underflow (expect exponent-1) --\n"
        "  input    : m=ST_MANTISSA_MIN/10, e=0\n"
        f"  expected : m≈{ST_MANTISSA_MIN}, e≈-1, s=1\n"
        f"  actual   : m={b.mantissa}, e={b.exponent}, s={b.sign}"
    )
    assert b.mantissa == ST_MANTISSA_MIN
    assert b.exponent == -1
    assert b.sign == 1


def test_amounts_underflow_to_zero_and_min_positive():
    from xrpl_router.core import STAmount, ST_MANTISSA_MIN, ST_EXP_MIN

    _banner("AMOUNTS: underflow & min_positive")
    # Too small at the lowest exponent -> normalises to zero
    z = STAmount.from_components(1, ST_EXP_MIN, 1)
    print(
        "[DEBUG] underflow to zero\n"
        "  case     : -- underflow to zero at min exponent --\n"
        f"  input    : m=1, e={ST_EXP_MIN}\n"
        "  expected : zero\n"
        f"  actual   : zero? {z.is_zero()} (m={z.mantissa}, e={z.exponent}, s={z.sign})"
    )
    assert z.is_zero()

    # Min positive IOU matches (minMantissa, minExponent)
    mp = STAmount.min_positive()
    print(
        "[DEBUG] min_positive IOU\n"
        "  case     : -- min positive IOU value --\n"
        f"  expected : (m={ST_MANTISSA_MIN}, e={ST_EXP_MIN}, s=1)\n"
        f"  actual   : (m={mp.mantissa}, e={mp.exponent}, s={mp.sign})"
    )
    assert mp.mantissa == ST_MANTISSA_MIN
    assert mp.exponent == ST_EXP_MIN
    assert mp.sign == 1


def test_amounts_add_sub_basic():
    from xrpl_router.core import STAmount, ST_MANTISSA_MIN

    _banner("AMOUNTS: add/sub basics")
    a = STAmount.from_components(2 * ST_MANTISSA_MIN, -10, 1)
    b = STAmount.from_components(3 * ST_MANTISSA_MIN, -10, 1)

    s = a + b
    d = b - a

    print(
        "[DEBUG] addition test\n"
        "  case     : -- addition --\n"
        f"  inputs   : A=(m={2*ST_MANTISSA_MIN}, e=-10), B=(m={3*ST_MANTISSA_MIN}, e=-10)\n"
        f"  expected : sum=(m={5*ST_MANTISSA_MIN}, e=-10)\n"
        f"  actual   : sum=(m={s.mantissa}, e={s.exponent}, s={s.sign})"
    )
    print(
        "[DEBUG] subtraction test\n"
        "  case     : -- subtraction (B - A) --\n"
        f"  inputs   : B-A\n"
        f"  expected : diff=(m={1*ST_MANTISSA_MIN}, e=-10)\n"
        f"  actual   : diff=(m={d.mantissa}, e={d.exponent}, s={d.sign})"
    )

    assert s.mantissa == 5 * ST_MANTISSA_MIN and s.exponent == -10 and s.sign == 1
    assert d.mantissa == 1 * ST_MANTISSA_MIN and d.exponent == -10 and d.sign == 1


def test_amounts_drops_bridge():
    from xrpl_router.core import st_from_drops, STAmount

    _banner("AMOUNTS: drops bridge")
    neg = st_from_drops(-5)
    one = st_from_drops(1)
    big = st_from_drops(1_000_000)  # 1 XRP

    print(
        "[DEBUG] drops→STAmount bridge\n"
        "  case     : -- negative→zero, one drop, one XRP --\n"
        "  inputs   : -5 drops, 1 drop, 1_000_000 drops (1 XRP)\n"
        f"  outputs  : neg=(m={neg.mantissa}, e={neg.exponent}, zero? {neg.is_zero()}); "
        f"one=(m={one.mantissa}, e={one.exponent}); "
        f"one_xrp=(m={big.mantissa}, e={big.exponent})"
    )

    assert isinstance(neg, STAmount) and neg.is_zero()
    assert isinstance(one, STAmount) and not one.is_zero()
    assert isinstance(big, STAmount) and not big.is_zero()


# --- Additional test: Amounts scalar mul/div ---
def test_amounts_scalar_mul_div():
    from xrpl_router.core import STAmount, ST_MANTISSA_MIN

    _banner("AMOUNTS: scalar mul/div")
    # Base value: +5e15 @ -10
    a = STAmount.from_components(5 * ST_MANTISSA_MIN, -10, 1)

    # --- div_by_scalar_down (toward zero) ---
    down = a.div_by_scalar_down(3)
    print(
        "[DEBUG] amounts scalar div (DOWN)\n"
        "  case     : -- divide DOWN (toward zero) --\n"
        "  input   : (m=5e15, e=-10, s=+1) / k=3\n"
        f"  result  : (m={down.mantissa}, e={down.exponent}, s={down.sign})"
    )
    # 5/3 toward zero -> 1 remainder -> m=1666666666666666@-10
    assert down.mantissa == 1666666666666666 and down.exponent == -10 and down.sign == 1

    # Negative: -5e15 @ -10
    a_neg = STAmount.from_components(5 * ST_MANTISSA_MIN, -10, -1)
    down_neg = a_neg.div_by_scalar_down(3)
    print(
        "[DEBUG] amounts scalar div (DOWN, negative)\n"
        "  case     : -- divide DOWN (negative) --\n"
        "  input   : (m=5e15, e=-10, s=-1) / k=3\n"
        f"  result  : (m={down_neg.mantissa}, e={down_neg.exponent}, s={down_neg.sign})"
    )
    # -5/3 toward zero -> -1
    assert down_neg.mantissa == 1666666666666666 and down_neg.exponent == -10 and down_neg.sign == -1

    # --- div_by_scalar_up (away from zero) ---
    up = a.div_by_scalar_up(3)
    print(
        "[DEBUG] amounts scalar div (UP)\n"
        "  case     : -- divide UP (away from zero) --\n"
        "  input   : (m=5e15, e=-10, s=+1) / k=3\n"
        f"  result  : (m={up.mantissa}, e={up.exponent}, s={up.sign})"
    )
    # 5/3 away from zero -> 1666666666666667
    assert up.mantissa == 1666666666666667 and up.exponent == -10 and up.sign == 1

    up_neg = a_neg.div_by_scalar_up(3)
    print(
        "[DEBUG] amounts scalar div (UP, negative)\n"
        "  case     : -- divide UP (negative) --\n"
        "  input   : (m=5e15, e=-10, s=-1) / k=3\n"
        f"  result  : (m={up_neg.mantissa}, e={up_neg.exponent}, s={up_neg.sign})"
    )
    # -5/3 away from zero -> -1666666666666667
    assert up_neg.mantissa == 1666666666666667 and up_neg.exponent == -10 and up_neg.sign == -1

    # --- div_by_scalar_up representable case (e.g., 1/2) ---
    # Make mantissa < k but still representable after exponent scaling: 1 / (2*1e15) -> 0.5.
    tiny = STAmount.from_components(1 * ST_MANTISSA_MIN, 0, 1)
    up_tiny = tiny.div_by_scalar_up(2 * ST_MANTISSA_MIN)
    print(
        "[DEBUG] amounts scalar div (UP, representable 1/2 case)\n"
        "  case     : -- divide UP, representable 1/2 --\n"
        f"  input   : (m={tiny.mantissa}, e={tiny.exponent}, s={tiny.sign}) / k={2*ST_MANTISSA_MIN}\n"
        f"  result  : (m={up_tiny.mantissa}, e={up_tiny.exponent}, s={up_tiny.sign})"
    )
    expect = STAmount.from_components(5 * ST_MANTISSA_MIN, -16, 1)
    assert up_tiny == expect

    # --- mul_by_scalar ---
    mul2 = tiny.mul_by_scalar(2)
    print(
        "[DEBUG] amounts scalar mul (+2)\n"
        "  case     : -- multiply by +2 --\n"
        f"  input   : (m={tiny.mantissa}, e={tiny.exponent}, s={tiny.sign}) * 2\n"
        f"  result  : (m={mul2.mantissa}, e={mul2.exponent}, s={mul2.sign})"
    )
    assert mul2.mantissa == 2 * ST_MANTISSA_MIN and mul2.exponent == 0 and mul2.sign == 1

    mul_neg3 = tiny.mul_by_scalar(-3)
    print(
        "[DEBUG] amounts scalar mul (-3)\n"
        "  case     : -- multiply by -3 --\n"
        f"  input   : (m={tiny.mantissa}, e={tiny.exponent}, s={tiny.sign}) * (-3)\n"
        f"  result  : (m={mul_neg3.mantissa}, e={mul_neg3.exponent}, s={mul_neg3.sign})"
    )
    assert mul_neg3.mantissa == 3 * ST_MANTISSA_MIN and mul_neg3.exponent == 0 and mul_neg3.sign == -1

    zero = tiny.mul_by_scalar(0)
    print(
        "[DEBUG] amounts scalar mul (zero)\n"
        "  case     : -- multiply by 0 (zero) --\n"
        f"  input   : (m={tiny.mantissa}, e={tiny.exponent}, s={tiny.sign}) * 0\n"
        f"  result  : (m={zero.mantissa}, e={zero.exponent}, s={zero.sign})"
    )
    assert zero.is_zero()


# Stage 4: quality (ratio construction, ordering, proportional equality)

def _mk_iou(n: int):
    """Helper: build IOU STAmount representing integer n (in 10^0 units).
    Uses mantissa scaling at exponent 0: (mantissa = n * 1e15, exponent=0).
    """
    from xrpl_router.core import STAmount, ST_MANTISSA_MIN
    return STAmount.from_components(ST_MANTISSA_MIN * n, 0, 1)


def test_quality_basic_ordering():
    from xrpl_router.core import Quality

    _banner("QUALITY: construction & ordering")
    in1 = _mk_iou(1)
    out2 = _mk_iou(2)
    out3 = _mk_iou(3)
    in2 = _mk_iou(2)

    q_1_over_2 = Quality.from_amounts(offer_out=out2, offer_in=in1)  # 2.0 (OUT/IN)
    q_2_over_2 = Quality.from_amounts(offer_out=out2, offer_in=in2)  # 1.0 (OUT/IN)
    q_1_over_3 = Quality.from_amounts(offer_out=out3, offer_in=in1)  # 3.0 (OUT/IN)

    print(
        "[DEBUG] quality basic (OUT/IN, higher is better)\n"
        "  case     : -- three offers, compare OUT/IN --\n"
        "  offers   : (out/in) = 2/1, 2/2, 3/1\n"
        f"  computed : [2/1={_fmt_q(_rate_to_decimal(q_1_over_2.rate))}, "
        f"2/2={_fmt_q(_rate_to_decimal(q_2_over_2.rate))}, "
        f"3/1={_fmt_q(_rate_to_decimal(q_1_over_3.rate))}]\n"
        "  expect   : 3/1 > 2/1 > 2/2"
    )

    # Higher is better
    assert q_1_over_2.rate > q_2_over_2.rate
    assert q_1_over_3.rate > q_1_over_2.rate


def test_quality_proportional_equality():
    from xrpl_router.core import Quality

    _banner("QUALITY: proportional equality")
    in1 = _mk_iou(1)
    out2 = _mk_iou(2)

    in10 = _mk_iou(10)
    out20 = _mk_iou(20)

    q_small = Quality.from_amounts(offer_out=out2, offer_in=in1)
    q_big = Quality.from_amounts(offer_out=out20, offer_in=in10)

    print(
        "[DEBUG] proportional equality\n"
        "  case     : -- scale-invariant ratio --\n"
        "  offers   : 2/1 vs 20/10\n"
        f"  computed : 2/1={_fmt_q(_rate_to_decimal(q_small.rate))}, "
        f"20/10={_fmt_q(_rate_to_decimal(q_big.rate))}\n"
        "  expect   : equal"
    )

    # Ratios 1/2 and 10/20 are equal
    assert q_small.rate == q_big.rate


def test_quality_monotonicity_in_and_out():
    from xrpl_router.core import Quality

    _banner("QUALITY: monotonicity in/out")
    out_fixed = _mk_iou(5)
    in_small = _mk_iou(1)
    in_large = _mk_iou(3)

    q_small_in = Quality.from_amounts(offer_out=out_fixed, offer_in=in_small)  # better
    q_large_in = Quality.from_amounts(offer_out=out_fixed, offer_in=in_large)  # worse

    print(
        "[DEBUG] quality monotonicity (IN side)\n"
        "  case     : -- vary IN (fixed OUT) --\n"
        f"  out=5, in=1 → {_fmt_q(_rate_to_decimal(q_small_in.rate))}; "
        f"in=3 → {_fmt_q(_rate_to_decimal(q_large_in.rate))}\n"
        "  expect: left (smaller in) > right (larger in)"
    )

    assert q_small_in.rate > q_large_in.rate

    in_fixed = _mk_iou(5)
    out_small = _mk_iou(1)
    out_large = _mk_iou(3)

    q_small_out = Quality.from_amounts(offer_out=out_small, offer_in=in_fixed)  # worse
    q_large_out = Quality.from_amounts(offer_out=out_large, offer_in=in_fixed)  # better

    print(
        "[DEBUG] quality monotonicity (OUT side)\n"
        "  case     : -- vary OUT (fixed IN) --\n"
        f"  in=5, out=1 → {_fmt_q(_rate_to_decimal(q_small_out.rate))}; "
        f"out=3 → {_fmt_q(_rate_to_decimal(q_large_out.rate))}\n"
        "  expect: right (larger out) > left (smaller out)"
    )

    assert q_large_out.rate > q_small_out.rate


# Stage 5: ordering (guards, floor, stable sort)

def _mk_quality(out_n: int, in_n: int):
    """Helper: construct Quality from integer OUT/IN using IOU amounts at exp=0."""
    from xrpl_router.core import Quality, STAmount, ST_MANTISSA_MIN
    out_amt = STAmount.from_components(ST_MANTISSA_MIN * out_n, 0, 1)
    in_amt = STAmount.from_components(ST_MANTISSA_MIN * in_n, 0, 1)
    return Quality.from_amounts(offer_out=out_amt, offer_in=in_amt)


def test_ordering_guard_xrp_bumps_to_bound():
    from xrpl_router.core import Quality, st_from_drops

    _banner("ORDERING: XRP guardrail")
    # Quoted: out=2 IOU, in=100_000_000 drops (100 XRP) => quality = 2 / 100 = 0.02
    out_take = _mk_iou(2)
    quoted_in = st_from_drops(100_000_000)
    slice_quality = Quality.from_amounts(offer_out=out_take, offer_in=quoted_in)

    # Implied too good: in=99_999_999 drops → quality = 2 / 99.999999 > 0.02 → must bump by +1 drop
    from xrpl_router.core import guard_instant_quality_xrp
    bumped = guard_instant_quality_xrp(out_take=out_take, in_drops=99_999_999, slice_quality=slice_quality)

    from xrpl_router.core import DROPS_PER_XRP
    quoted_q = Decimal(2) / (Decimal(100_000_000) / Decimal(DROPS_PER_XRP))
    implied_start_q = Decimal(2) / (Decimal(99_999_999) / Decimal(DROPS_PER_XRP))
    print(
        "[DEBUG] XRP guard\n"
        "  case     : -- bump to boundary by +1 drop --\n"
        "  scenario : quoted vs implied must not exceed quoted (higher is better)\n"
        f"  quoted   : out=2 IOU, in=100_000_000 drops → q={_fmt_q(quoted_q)}\n"
        f"  implied  : out=2 IOU, in=99_999_999 drops → q={_fmt_q(implied_start_q)} (too favourable)\n"
        "  expect   : bump in by +1 drop → 100_000_000"
    )
    assert bumped == 100_000_000


def test_ordering_quality_floor_keeps_ge():
    from xrpl_router.core import apply_quality_floor

    _banner("ORDERING: quality floor")
    # Items as (id, out, in). Higher is better (out/in).
    items = [
        (0, 1, 1),  # q=1.0
        (1, 3, 2),  # q=1.5
        (2, 4, 1),  # q=4.0
        (3, 1, 3),  # q=0.333..
    ]

    def get_quality(t):
        _, out_n, in_n = t
        return _mk_quality(out_n, in_n)

    # Floor at q=1.5 should keep ids 1 and 2 (>= 1.5)
    q_floor = _mk_quality(3, 2)
    kept = apply_quality_floor(items, qmin=q_floor, get_quality=get_quality)

    print(
        "[DEBUG] quality floor filter\n"
        "  case     : -- keep >= floor, filter others --\n"
        f"  floor    : q={_fmt_q(Decimal(3)/Decimal(2))}\n"
        "  kept     : " + ", ".join([f"id={i}, q={_fmt_q(Decimal(o)/Decimal(i_))}" for (i, o, i_) in kept])
    )
    kept_ids = [i for (i, _, _) in kept]
    assert kept_ids == [1, 2]


def test_stable_sort_by_quality_desc():
    from xrpl_router.core import stable_sort_by_quality

    _banner("ORDERING: stable sort by quality")
    # Items as (idx, out, in)
    items = [
        (2, 2, 1),   # q=2.0
        (0, 1, 1),   # q=1.0
        (1, 2, 1),   # q=2.0 (tie with idx=2; stable: keep original order among ties by index key)
        (3, 3, 2),   # q=1.5
    ]

    def get_quality(t):
        _, out_n, in_n = t
        return _mk_quality(out_n, in_n)

    def get_index(t):
        idx, _, _ = t
        return idx

    sorted_items = stable_sort_by_quality(items, get_quality=get_quality, get_index=get_index)

    # Expect descending by quality (higher first): q=2.0 (idx 1,2), then q=1.5 (idx 3), then q=1.0 (idx 0)
    print(
        "[DEBUG] stable sort\n"
        "  case     : -- desc by quality, tie-break by index --\n"
        "  rule     : sort by quality desc (higher is better), tie by id asc\n"
        "  order    : " + ", ".join([f"id={i}, q={_fmt_q(Decimal(o)/Decimal(i_))}" for (i, o, i_) in sorted_items])
    )
    assert [i for (i, _, _) in sorted_items] == [1, 2, 3, 0]


# --- Additional test: Ordering IOU guard cases ---
def test_ordering_guard_iou_cases():
    from xrpl_router.core import Quality, STAmount, ST_MANTISSA_MIN, guard_instant_quality

    _banner("ORDERING: IOU guardrail (closed-form)")
    def iou(n: int) -> STAmount:
        return STAmount.from_components(ST_MANTISSA_MIN * n, 0, 1)

    out_take = iou(2)

    # Quoted IN = 100 units → q = 2/100 = 0.02
    quoted_in = iou(100)
    q_quoted = Quality.from_amounts(offer_out=out_take, offer_in=quoted_in)

    # Case 1: trigger (implied better than quoted) → bump IN upward to boundary
    implied_in_start = iou(99)  # 2/99 > 2/100
    bumped = guard_instant_quality(out_take, implied_in_start, q_quoted)

    print(
        "[DEBUG] IOU guard (trigger)\n"
        "  case     : -- trigger (bump to boundary) --\n"
        "  quoted  : out=2, in=100 → q=2/100=0.02\n"
        "  implied : out=2, in=99  → q=2/99  > quoted (too favourable)\n"
        f"  bumped  : in → {bumped.mantissa//ST_MANTISSA_MIN} (units)"
    )
    assert bumped == iou(100)

    # Case 2: no trigger (equal to quoted)
    kept   = guard_instant_quality(out_take, quoted_in,        q_quoted)
    print(
        "[DEBUG] IOU guard (no trigger)\n"
        "  case     : -- equal to quoted (no bump) --\n"
        "  implied : equals quoted\n"
        f"  kept    : in → {kept.mantissa//ST_MANTISSA_MIN} (units)"
    )
    assert kept == quoted_in

    # Case 3: already at boundary (not improved by bumping)
    implied_in_low = iou(101)  # 2/101 < 2/100 (worse than quoted) → should keep
    kept2  = guard_instant_quality(out_take, implied_in_low,   q_quoted)
    print(
        "[DEBUG] IOU guard (already worse)\n"
        "  case     : -- worse than quoted (keep) --\n"
        "  implied : worse or equal to quoted\n"
        f"  kept    : in → {kept2.mantissa//ST_MANTISSA_MIN} (units)"
    )
    assert kept2 == implied_in_low

# --- Additional test: ORDERING: XRP guardrail (equal & worse) ---
def test_ordering_guard_xrp_equal_and_worse_cases():
    from xrpl_router.core import Quality, st_from_drops, guard_instant_quality_xrp
    _banner("ORDERING: XRP guardrail (equal & worse)")

    out_take = _mk_iou(2)
    quoted_in = st_from_drops(100_000_000)  # 100 XRP
    q = Quality.from_amounts(offer_out=out_take, offer_in=quoted_in)

    # Equal: no bump
    kept_eq = guard_instant_quality_xrp(out_take, 100_000_000, q)
    print(
        "[DEBUG] XRP guard (equal)\n"
        "  case     : -- implied equals quoted (no bump) --\n"
        "  quoted   : out=2 IOU, in=100_000_000 drops\n"
        f"  kept     : {kept_eq} drops"
    )
    assert kept_eq == 100_000_000

    # Worse: larger IN, should keep
    kept_worse = guard_instant_quality_xrp(out_take, 101_000_000, q)
    print(
        "[DEBUG] XRP guard (worse)\n"
        "  case     : -- implied worse than quoted (keep) --\n"
        f"  in_start : 101_000_000 drops\n"
        f"  kept     : {kept_worse} drops"
    )
    assert kept_worse == 101_000_000

# --- Additional test: ORDERING: prepare (floor) + stable sort pipeline ---
def test_prepare_and_order_pipeline_minimal():
    from xrpl_router.core import prepare_and_order
    _banner("ORDERING: prepare (floor) + stable sort pipeline")

    # items: (idx, out, in)
    items = [
        (5, 1, 2),   # q=0.5
        (2, 4, 2),   # q=2.0
        (3, 3, 2),   # q=1.5
        (1, 2, 2),   # q=1.0
        (4, 6, 3),   # q=2.0 (tie with idx=2)
    ]

    def get_q(t):
        _, o, i = t
        return _mk_quality(o, i)

    def get_idx(t):
        i, _, _ = t
        return i

    q_floor = _mk_quality(3, 2)  # 1.5
    ordered = prepare_and_order(items, qmin=q_floor, get_quality=get_q, get_index=get_idx)
    print(
        "[DEBUG] prepare_and_order pipeline\n"
        "  case     : -- filter by floor then sort desc, tie by index --\n"
        f"  input    : {items}\n"
        f"  floor    : q=1.5\n"
        f"  output   : {ordered}"
    )
    assert [i for (i, _, _) in ordered] == [2, 4, 3]


# --- Edge case: IOU guard with zero out_take should keep input ---
def test_ordering_guard_iou_zero_out_keeps():
    from xrpl_router.core import STAmount, ST_MANTISSA_MIN, Quality, guard_instant_quality
    _banner("ORDERING: IOU guard (zero out_take)")

    # Build zero out_take by subtracting equal amounts
    a = STAmount.from_components(ST_MANTISSA_MIN, 0, 1)
    zero_out = a - a
    in_start = STAmount.from_components(50 * ST_MANTISSA_MIN, 0, 1)
    quoted_in = STAmount.from_components(100 * ST_MANTISSA_MIN, 0, 1)
    q = Quality.from_amounts(offer_out=zero_out, offer_in=quoted_in)

    kept = guard_instant_quality(zero_out, in_start, q)
    print(
        "[DEBUG] IOU guard (zero out_take)\n"
        "  case     : -- zero out implies no change --\n"
        f"  in_start : 50 units\n"
        f"  kept     : mantissa={kept.mantissa}, exponent={kept.exponent}"
    )
    assert kept == in_start


# --- Edge case: XRP guard with zero out_take should keep drops ---
def test_ordering_guard_xrp_zero_out_keeps():
    from xrpl_router.core import STAmount, ST_MANTISSA_MIN, st_from_drops, Quality, guard_instant_quality_xrp
    _banner("ORDERING: XRP guard (zero out_take)")

    a = STAmount.from_components(ST_MANTISSA_MIN, 0, 1)
    zero_out = a - a
    quoted_in = st_from_drops(100_000_000)
    q = Quality.from_amounts(offer_out=zero_out, offer_in=quoted_in)

    kept = guard_instant_quality_xrp(zero_out, 50, q)
    print(
        "[DEBUG] XRP guard (zero out_take)\n"
        "  case     : -- zero out implies no bump --\n"
        f"  in_start : 50 drops\n"
        f"  kept     : {kept} drops"
    )
    assert kept == 50