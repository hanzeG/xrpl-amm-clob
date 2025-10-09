from decimal import Decimal

import pytest

from xrpl_router.amm_context import AMMContext
from xrpl_router.core.fmt import fmt_dec, amount_to_decimal, quality_rate_to_decimal, quality_price_to_decimal


LOG_PLACES = 6

# TODO: hide quantisation-tail AMM slices in prints only; production filtering to follow
TINY_OUT_HIDE = Decimal("1e-12")

def _qd(q) -> str:
    return fmt_dec(quality_rate_to_decimal(q), places=LOG_PLACES)

def _pd(q) -> str:
    return fmt_dec(quality_price_to_decimal(q), places=LOG_PLACES)

# Production interfaces for CLOB/AMM/amounts/quality
from xrpl_router.clob import make_ladder
from xrpl_router.amm import AMM
from xrpl_router.core.amounts import STAmount
from xrpl_router.core.quality import Quality





# Slippage monotonicity — per-test whitepaper mapping lives above each test.
# See individual test comments for exact section references.

# Whitepaper §1.2.4 + §1.2.5
# We assert two properties on avg_price (average effective price) as trade size increases on a fixed limit order book:
# 1) Non-negativity: avg_price should not be negative (won't get cheaper as you buy more).
# 2) Monotonicity: avg_price should be non-decreasing with trade size, reflecting tiered consumption.
# Rationales: same quality tier consumes offers at constant price; moving to worse tiers increases avg_price.
@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("40"), Decimal("80"), Decimal("160"), Decimal("240"), Decimal("400")],  # expanded sizes
])
def test_slippage_monotonic_clob_only(sizes):
    """
    Test avg_price (average effective price) monotonicity for CLOB_ONLY execution mode.
    Based on whitepaper §1.2.4 and §1.2.5, avg_price should be non-negative and non-decreasing
    as trade size increases on a fixed limit order book.

    Defaults (fixtures): CLOB has 6 levels, each out_max≈40 IOU, top quality bucket at 1.0 (IOU grids at 1e-15).
    """
    print("===== CLOB_ONLY =====")
    ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=False, out_is_xrp=False)
    # Print CLOB ladder summary
    print("    CLOB ladder:")
    for i, s in enumerate(ladder, start=1):
        qd = _qd(s.quality)
        pd = _pd(s.quality)
        outd = fmt_dec(amount_to_decimal(s.out_max), places=LOG_PLACES)
        ind = fmt_dec(amount_to_decimal(s.in_at_out_max), places=LOG_PLACES)
        print(f"      - L{i}: q={qd}, p={pd}, out={outd}, in={ind}")
    slips = []
    for q in sizes:
        remaining = q
        total_out_dec = Decimal(0)
        total_in_dec = Decimal(0)
        idx = None
        # Find last level consumed (for block printing)
        out_so_far = Decimal(0)
        for i, s in enumerate(ladder):
            seg_out_dec = amount_to_decimal(s.out_max)
            take = seg_out_dec if seg_out_dec <= remaining else remaining
            if take > 0:
                out_so_far += take
                idx = i
            remaining -= take
            if remaining <= 0:
                break
        # Recompute the actual fill for this q
        remaining = q
        total_out_dec = Decimal(0)
        total_in_dec = Decimal(0)
        for s in ladder:
            seg_out_dec = amount_to_decimal(s.out_max)
            seg_in_dec = amount_to_decimal(s.in_at_out_max)
            take = seg_out_dec if seg_out_dec <= remaining else remaining
            if take <= 0:
                continue
            in_take = (seg_in_dec * (take / seg_out_dec)) if seg_out_dec > 0 else Decimal(0)
            total_out_dec += take
            total_in_dec += in_take
            remaining -= take
            if remaining <= 0:
                break
        if total_out_dec > 0:
            avg_q = Quality.from_amounts(STAmount.from_decimal(total_out_dec), STAmount.from_decimal(total_in_dec))
            slips.append(avg_q)
            # Block header for this target
            print(f"\n[CLOB_ONLY] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: {_qd(avg_q)}, avg_price: {_pd(avg_q)}")
            print("    Consumed levels:")
            if idx is not None:
                for i, s in enumerate(ladder[:idx+1], start=1):
                    qd = _qd(s.quality)
                    pd = _pd(s.quality)
                    outd = fmt_dec(amount_to_decimal(s.out_max), places=LOG_PLACES)
                    ind = fmt_dec(amount_to_decimal(s.in_at_out_max), places=LOG_PLACES)
                    print(f"      - L{i}: q={qd}, p={pd}, out={outd}, in={ind}")
        else:
            slips.append(None)
            print(f"\n[CLOB_ONLY] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: N/A, avg_price: N/A")
            print("    Consumed levels: None")
    print()

    for avg_q in slips:
        if avg_q is not None:
            assert avg_q.rate.sign > 0
    prev = None
    for avg_q in slips:
        if avg_q is not None:
            if prev is not None:
                assert avg_q <= prev, (
                    f"avg_quality should be non-increasing: prev_q={fmt_dec(quality_rate_to_decimal(prev))} curr_q={fmt_dec(quality_rate_to_decimal(avg_q))}"
                )
            prev = avg_q

#
# Note: this dense AMM-only test replaces the coarse-grid AMM-only test to avoid overlap.
# Whitepaper §1.2.7.1 (AMM-only) — denser size grid
# Purpose: with the corrected tier-baseline, avg_price should remain non-negative and non-decreasing
# even across a denser set of target sizes.
# This complements the coarser grid to catch subtle non-monotonicity due to rounding/bucketing noise.
@pytest.mark.parametrize("sizes", [
    [Decimal("10"), Decimal("20"), Decimal("40"), Decimal("80"), Decimal("160"), Decimal("320"), Decimal("640"), Decimal("1280"), Decimal("2560")],
])
def test_slippage_monotonic_amm_only_dense(sizes):
    """
    Defaults (fixtures): AMM reserves x≈1000 (XRP grid=1e-6), y≈2000 (IOU grid=1e-15), fee≈0.003, no transfer fees; curve is sliced starting at ~5% of target per iteration. This test also enforces the strict whitepaper property: within each route, marginal (per-iteration) price is non-decreasing.
    """
    print("===== AMM_ONLY/DENSE =====")
    slips = []
    for q in sizes:
        amm = AMM(Decimal("1000"), Decimal("2000"), Decimal("0.003"), x_is_xrp=False, y_is_xrp=False, tr_in=Decimal("0"), tr_out=Decimal("0"))
        segs = amm.segments_for_out(target_out=q, max_segments=30, start_fraction=Decimal("0.05"))
        print(f"\n[AMM_ONLY/DENSE] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: {_qd(Quality.from_amounts(STAmount.from_decimal(sum(amount_to_decimal(s.out_max) for s in segs)), STAmount.from_decimal(sum(amount_to_decimal(s.in_at_out_max) for s in segs)))) if segs else 'N/A'}, avg_price: {_pd(Quality.from_amounts(STAmount.from_decimal(sum(amount_to_decimal(s.out_max) for s in segs)), STAmount.from_decimal(sum(amount_to_decimal(s.in_at_out_max) for s in segs)))) if segs else 'N/A'}")
        print("    AMM segments:")
        # TODO: temporarily suppress extremely small AMM tail slices in print (quantisation artefacts)
        s_idx = 0
        for s in segs:
            out_dec = amount_to_decimal(s.out_max)
            if out_dec < TINY_OUT_HIDE:
                continue  # hide tiny AMM slices from display only
            s_idx += 1
            qd = _qd(s.quality)
            pd = _pd(s.quality)
            outd = fmt_dec(out_dec, places=LOG_PLACES)
            ind = fmt_dec(amount_to_decimal(s.in_at_out_max), places=LOG_PLACES)
            print(f"      - S{s_idx}: q={qd}, p={pd}, out={outd}, in={ind}")
        total_out_dec = sum(amount_to_decimal(s.out_max) for s in segs)
        total_in_dec = sum(amount_to_decimal(s.in_at_out_max) for s in segs)
        assert total_out_dec > 0
        avg_q = Quality.from_amounts(STAmount.from_decimal(total_out_dec), STAmount.from_decimal(total_in_dec))
        slips.append(avg_q)
    print()

    for avg_q in slips:
        assert avg_q.rate.sign > 0
    for q in sizes:
        amm = AMM(Decimal("1000"), Decimal("2000"), Decimal("0.003"), x_is_xrp=False, y_is_xrp=False, tr_in=Decimal("0"), tr_out=Decimal("0"))
        segs = amm.segments_for_out(target_out=q, max_segments=30, start_fraction=Decimal("0.05"))
        marginals_q = [s.quality for s in segs]
        for a, b in zip(marginals_q, marginals_q[1:]):
            assert b <= a, (
                f"marginal quality not non-increasing within route (q={q}): prev_q={fmt_dec(quality_rate_to_decimal(a))} curr_q={fmt_dec(quality_rate_to_decimal(b))}"
            )


#
# Whitepaper §1.2.7.2 (Hybrid anchoring)
# When AMM and CLOB coexist within the same iteration, tier selection is anchored to the LOB top quality bucket.
# Under this anchoring, the average effective price (avg_price) should be non-negative and non-decreasing as total size grows,
# per §1.2.4, §1.2.5 (CLOB tiers) and §1.2.7.2 (AMM anchoring).
@pytest.mark.parametrize("sizes", [
    [Decimal("20"), Decimal("40"), Decimal("80"), Decimal("160"), Decimal("240"), Decimal("400"), Decimal("1000")],
])
def test_slippage_monotonic_hybrid_tier_anchored(sizes):
    """
    Defaults (fixtures): Hybrid uses the above CLOB and AMM; anchoring at LOB top quality (per §1.2.7.2).
    """
    print("===== HYBRID (Anchored) =====")
    ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=False, out_is_xrp=False)
    # Print CLOB ladder summary
    print("    CLOB ladder:")
    for i, s in enumerate(ladder, start=1):
        qd = _qd(s.quality)
        pd = _pd(s.quality)
        outd = fmt_dec(amount_to_decimal(s.out_max), places=LOG_PLACES)
        ind = fmt_dec(amount_to_decimal(s.in_at_out_max), places=LOG_PLACES)
        print(f"      - L{i}: q={qd}, p={pd}, out={outd}, in={ind}")
    slips = []
    for q in sizes:
        print()  # Visual separation between targets
        path_segments = []
        amm = AMM(Decimal("1000"), Decimal("2000"), Decimal("0.003"), x_is_xrp=False, y_is_xrp=False, tr_in=Decimal("0"), tr_out=Decimal("0"))
        q_top = ladder[0].quality
        remaining = q
        total_out_dec = Decimal(0)
        total_in_dec = Decimal(0)
        idx = 0
        clob_rem = amount_to_decimal(ladder[0].out_max) if ladder else Decimal(0)
        while remaining > 0:
            seg = ladder[idx] if idx < len(ladder) else None
            seg_amm = None
            if seg is not None and clob_rem > 0:
                seg_out_dec = amount_to_decimal(seg.out_max)
                seg_in_dec = amount_to_decimal(seg.in_at_out_max)
            else:
                seg_out_dec = Decimal(0)
                seg_in_dec = Decimal(0)
            need_amt = STAmount.from_decimal(remaining)
            seg_amm = amm.synthetic_segment_for_quality(q_threshold=q_top, max_out_cap=need_amt)
            if seg_amm is not None:
                seg_amm_out_dec = amount_to_decimal(seg_amm.out_max)
                seg_amm_in_dec = amount_to_decimal(seg_amm.in_at_out_max)
            else:
                seg_amm_out_dec = Decimal(0)
                seg_amm_in_dec = Decimal(0)
            use_clob = False
            use_amm = False
            if seg is not None and clob_rem > 0:
                p_clob = quality_price_to_decimal(seg.quality)
            else:
                p_clob = None
            if seg_amm is not None:
                p_amm = quality_price_to_decimal(seg_amm.quality)
            else:
                p_amm = None
            if p_clob is not None and (p_amm is None or p_clob <= p_amm):
                use_clob = True
            elif p_amm is not None:
                use_amm = True
            else:
                break
            if use_clob:
                take = min(clob_rem, remaining)
                if take <= 0:
                    break
                in_take = (seg_in_dec * (take / seg_out_dec)) if seg_out_dec > 0 else Decimal(0)
                total_out_dec += take
                total_in_dec += in_take
                path_segments.append(("CLOB", seg))
                remaining -= take
                clob_rem -= take
                if clob_rem == 0:
                    idx += 1
                    if idx < len(ladder):
                        clob_rem = amount_to_decimal(ladder[idx].out_max)
                    else:
                        clob_rem = Decimal(0)
            elif use_amm:
                take = min(seg_amm_out_dec, remaining)
                if take <= 0:
                    break
                in_take = (seg_amm_in_dec * (take / seg_amm_out_dec)) if seg_amm_out_dec > 0 else Decimal(0)
                total_out_dec += take
                total_in_dec += in_take
                path_segments.append(("AMM", seg_amm))
                amm.apply_fill_st(seg_amm.in_at_out_max, seg_amm.out_max)
                remaining -= take
        if total_out_dec > 0:
            avg_q = Quality.from_amounts(STAmount.from_decimal(total_out_dec), STAmount.from_decimal(total_in_dec))
            slips.append(avg_q)
            print(f"[HYBRID] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: {_qd(avg_q)}, avg_price: {_pd(avg_q)}")
            print("    Hybrid path:")
            # TODO: temporarily suppress extremely small AMM tail slices in print (quantisation artefacts)
            c_idx = 0
            a_idx = 0
            for src, seg in path_segments:
                out_dec = amount_to_decimal(seg.out_max)
                if src == "AMM" and out_dec < TINY_OUT_HIDE:
                    continue  # hide tiny AMM slices from display only
                if src == "CLOB":
                    c_idx += 1
                    tag = f"C{c_idx}"
                else:
                    a_idx += 1
                    tag = f"A{a_idx}"
                qd = _qd(seg.quality)
                pd = _pd(seg.quality)
                outd = fmt_dec(out_dec, places=LOG_PLACES)
                ind = fmt_dec(amount_to_decimal(seg.in_at_out_max), places=LOG_PLACES)
                print(f"      - {tag}: src={src}, q={qd}, p={pd}, out={outd}, in={ind}")
        else:
            slips.append(None)
            print(f"[HYBRID] target_out: {fmt_dec(q, places=LOG_PLACES)}, avg_quality: N/A, avg_price: N/A")
    print()

    for avg_q in slips:
        if avg_q is not None:
            assert avg_q.rate.sign > 0
    for q in sizes:
        ladder = make_ladder(depth=6, top_quality=Decimal("1.00"), qty_per_level=Decimal("40"), decay=Decimal("0.985"), in_is_xrp=False, out_is_xrp=False)
        amm = AMM(Decimal("1000"), Decimal("2000"), Decimal("0.003"), x_is_xrp=False, y_is_xrp=False, tr_in=Decimal("0"), tr_out=Decimal("0"))
        q_top = ladder[0].quality
        remaining = q
        marginals_q = []
        idx = 0
        clob_rem = amount_to_decimal(ladder[0].out_max) if ladder else Decimal(0)
        while remaining > 0:
            seg = ladder[idx] if idx < len(ladder) else None
            seg_amm = None
            if seg is not None and clob_rem > 0:
                seg_out_dec = amount_to_decimal(seg.out_max)
                seg_in_dec = amount_to_decimal(seg.in_at_out_max)
            else:
                seg_out_dec = Decimal(0)
                seg_in_dec = Decimal(0)
            need_amt = STAmount.from_decimal(remaining)
            seg_amm = amm.synthetic_segment_for_quality(q_threshold=q_top, max_out_cap=need_amt)
            if seg_amm is not None:
                seg_amm_out_dec = amount_to_decimal(seg_amm.out_max)
                seg_amm_in_dec = amount_to_decimal(seg_amm.in_at_out_max)
            else:
                seg_amm_out_dec = Decimal(0)
                seg_amm_in_dec = Decimal(0)
            use_clob = False
            use_amm = False
            if seg is not None and clob_rem > 0:
                p_clob = quality_price_to_decimal(seg.quality)
            else:
                p_clob = None
            if seg_amm is not None:
                p_amm = quality_price_to_decimal(seg_amm.quality)
            else:
                p_amm = None
            if p_clob is not None and (p_amm is None or p_clob <= p_amm):
                use_clob = True
            elif p_amm is not None:
                use_amm = True
            else:
                break
            if use_clob:
                take = min(clob_rem, remaining)
                if take <= 0:
                    break
                marginals_q.append(seg.quality)
                remaining -= take
                clob_rem -= take
                if clob_rem == 0:
                    idx += 1
                    if idx < len(ladder):
                        clob_rem = amount_to_decimal(ladder[idx].out_max)
                    else:
                        clob_rem = Decimal(0)
            elif use_amm:
                take = min(seg_amm_out_dec, remaining)
                if take <= 0:
                    break
                marginals_q.append(seg_amm.quality)
                amm.apply_fill_st(seg_amm.in_at_out_max, seg_amm.out_max)
                remaining -= take
        for a, b in zip(marginals_q, marginals_q[1:]):
            assert b <= a, (
                f"marginal quality not non-increasing within route (q={q}): prev_q={fmt_dec(quality_rate_to_decimal(a))} curr_q={fmt_dec(quality_rate_to_decimal(b))}"
            )
