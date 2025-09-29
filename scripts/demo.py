"""Comprehensive demo: CLOB↔AMM routing under whitepaper semantics (anchoring, tiers, limits).

Scenarios covered:
S1a) CLOB-only, single-tier (AMM absent)
S1b) AMM-only, multi-iteration (curve self-pricing)
S2a) CLOB≈AMM (AMM slightly better): same-iteration equal-quality batching

Multi-iteration & tier switching:
S3a) Demand exceeds top-tier capacity → second iteration (CLOB-only)
S3b) AMM slightly worse than CLOB top → round1 CLOB-only, round2 AMM/next tier
S3c) Large demand + multi-level CLOB → consecutive tier switching
"""
from __future__ import annotations

from decimal import Decimal
from typing import List, Optional, Callable
import argparse
import sys

from xrpl_router.clob import ClobLevel, Clob
from xrpl_router.amm import AMM
from xrpl_router.router import route, RouteError
from xrpl_router.core import quality_bucket

# ---------- pretty printers ----------

def brief_book(levels: List[ClobLevel]) -> str:
    if not levels:
        return "CLOB: (empty)"
    parts = [f"CLOB[{i+1}]: price={lvl.price_in_per_out} IN/OUT, out_max={lvl.out_liquidity}" for i, lvl in enumerate(levels)]
    return "; ".join(parts)


def print_result(title: str, filled_out: Decimal, spent_in: Decimal, avg_q: Decimal, trace, iter_logs, *, show_steps: bool = True, compact: bool = False):
    """Pretty printer with optional compaction.

    - If compact=True: only prints consumption summary and totals (no steps, no writebacks).
    - If show_steps=False: hides per-step lines but still prints AMM writebacks.
    - Default: prints steps and writebacks.
    """
    print(f"\n=== {title} ===")

    # Derive tier bucket and anchoring from the steps
    tier_bucket = quality_bucket(trace[0]['quality']) if trace else Decimal('0')
    has_clob = any(s['src'] == 'CLOB' for s in trace)
    has_amm = any(s['src'] == 'AMM' for s in trace)
    anchoring = has_clob and has_amm

    # Aggregates by source
    agg = {'CLOB': {'out': Decimal(0), 'in': Decimal(0)}, 'AMM': {'out': Decimal(0), 'in': Decimal(0)}}
    for s in trace:
        so, si = s['take_out'], s['take_in']
        agg[s['src']]['out'] += so
        agg[s['src']]['in'] += si

    print("- Consumed (by source):")
    if has_clob:
        print(f"  • CLOB: OUT={agg['CLOB']['out']}, IN={agg['CLOB']['in']}")
    if has_amm:
        print(f"  • AMM : OUT={agg['AMM']['out']}, IN={agg['AMM']['in']}")

    if not compact:
        if show_steps:
            print("Steps (iteration process):")
        wb_idx = 0
        amm_out_acc = Decimal('0')
        tol = Decimal('1e-18')  # tiny tolerance for grid quirks
        if show_steps and trace:
            for s in trace:
                so, si = s['take_out'], s['take_in']
                # Prefer diagnostics from router (Phase A, pre-grid) if present
                inst_q_raw = s.get('inst_q_raw')
                if inst_q_raw is None:
                    inst_q_raw = (so / si) if si > 0 else Decimal('0')
                inst_q_bucket = quality_bucket(inst_q_raw)
                print(f"  • {s['src']} out={so} in={si} (inst_q_raw={inst_q_raw}, bucketed={inst_q_bucket})")

                # Only AMM steps contribute to AMM writeback accounting
                if s['src'] == 'AMM' and wb_idx < len(iter_logs):
                    amm_out_acc += so
                    target_out_this_iter = iter_logs[wb_idx][1]  # sum_out for this iteration
                    # When we've matched (within tolerance) the AMM OUT for this iteration, print its writeback
                    if amm_out_acc + tol >= target_out_this_iter:
                        sum_in, sum_out, new_spq = iter_logs[wb_idx]
                        print(f"- AMM writeback → +{sum_in} IN, -{sum_out} OUT; new SPQ={new_spq}")
                        wb_idx += 1
                        amm_out_acc = Decimal('0')

        # If steps are hidden, still print writebacks (unless compact)
        if not show_steps and not compact:
            for (sum_in, sum_out, new_spq) in iter_logs:
                print(f"- AMM writeback → +{sum_in} IN, -{sum_out} OUT; new SPQ={new_spq}")

    print("\nTotals")
    print(f"- OUT={filled_out}  IN={spent_in}  avg_q={avg_q}")


# ---------- scenario runner ----------

def run_scenario(
    title: str,
    *,
    clob_levels: List[ClobLevel],
    amm: Optional[AMM],
    target_out: Decimal,
    send_max: Optional[Decimal] = None,
    deliver_min: Optional[Decimal] = None,
    in_is_xrp: bool = False,
    out_is_xrp: bool = False,
    segments_start_fraction: Optional[Decimal] = None,
    min_quality: Optional[Decimal] = None,
    show_steps: bool = True,
    compact: bool = False,
):
    print("\n" + "=" * 80)
    print(f"Scenario: {title}")
    qmin_bucket: Optional[Decimal] = None
    if min_quality is not None:
        qmin_bucket = quality_bucket(min_quality)
    # Build CLOB segments
    if clob_levels is not None:
        book = Clob(clob_levels, in_is_xrp=in_is_xrp, out_is_xrp=out_is_xrp)
        segs = book.segments()
    else:
        segs = []
    # Apply min_quality filter to initial CLOB segments if provided
    if qmin_bucket is not None and segs:
        segs = [s for s in segs if quality_bucket(s.quality) >= qmin_bucket]
    print("Market")
    print("- " + brief_book(clob_levels))
    if amm is not None:
        print(f"- AMM : X={amm.x} Y={amm.y}  SPQ={amm.spq()} (OUT/IN)")
    else:
        print("- AMM : (none)")
    if qmin_bucket is not None:
        print(f"- min_quality (bucketed) = {qmin_bucket}")

    iter_logs: List[tuple] = []
    # Anchoring & writeback callbacks
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[object]]] = None
    iter_no = {"n": 0}

    if amm is not None:
        def _anchor(q_threshold: Decimal, need: Decimal):
            seg = amm.synthetic_segment_for_quality(q_threshold, max_out_cap=need)
            if seg is None:
                return None
            if qmin_bucket is not None and quality_bucket(seg.quality) < qmin_bucket:
                return None
            return seg
        amm_anchor = _anchor

        if segments_start_fraction is None:
            def _curve(need: Decimal):
                cs = list(amm.segments_for_out(need))
                if qmin_bucket is not None:
                    cs = [s for s in cs if quality_bucket(s.quality) >= qmin_bucket]
                return cs
        else:
            ssf = segments_start_fraction
            def _curve(need: Decimal):
                cs = list(amm.segments_for_out(need, start_fraction=ssf))
                if qmin_bucket is not None:
                    cs = [s for s in cs if quality_bucket(s.quality) >= qmin_bucket]
                return cs
        amm_curve = _curve

        def after_iter(sum_in: Decimal, sum_out: Decimal) -> None:
            iter_no["n"] += 1
            amm.apply_fill(sum_in, sum_out)
            iter_logs.append((sum_in, sum_out, amm.spq()))
    else:
        def after_iter(sum_in: Decimal, sum_out: Decimal) -> None:
            pass

        def amm_curve(need: Decimal):
            return []

    # Route
    try:
        res = route(
            target_out,
            segs,
            amm_anchor=amm_anchor,
            amm_curve=amm_curve,
            send_max=send_max,
            deliver_min=deliver_min,
            after_iteration=after_iter,
        )
        print_result("Route Result", res.filled_out, res.spent_in, res.avg_quality, res.trace, iter_logs, show_steps=show_steps, compact=compact)
    except RouteError as e:
        print("\n=== Route Result ===")
        print(f"Routing failed: {e}")
        print(f"(target_out={target_out}, send_max={send_max}, deliver_min={deliver_min})")


# ---------- build common fixtures ----------

def mk_levels(ioi1: str, qty1: str, ioi2: Optional[str] = None, qty2: Optional[str] = None) -> List[ClobLevel]:
    levels = [ClobLevel.from_numbers(ioi1, qty1)]
    if ioi2 is not None and qty2 is not None:
        levels.append(ClobLevel.from_numbers(ioi2, qty2))
    return levels


#
# -------- scenario registry helpers --------
class Scenario:
    def __init__(self, sid: str, fn: Callable[[], None]):
        self.sid = sid
        self.fn = fn

scenarios: List[Scenario] = []

def add(sid: str, fn: Callable[[], None]) -> None:
    scenarios.append(Scenario(sid, fn))

# ---------- run scenarios ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XRPL AMM/CLOB routing demo")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated scenario ids to run (e.g., S1b,S3a)")
    parser.add_argument("--skip", type=str, default=None, help="Comma-separated scenario ids to skip")
    parser.add_argument("--no-steps", action="store_true", help="Hide per-step lines; still show AMM writebacks")
    parser.add_argument("--compact", action="store_true", help="Compact output: show only consumption and totals (no steps, no writebacks)")
    args = parser.parse_args(sys.argv[1:])

    show_steps = not args.no_steps and (not args.compact)
    compact = bool(args.compact)

    # --------------- Register scenarios ---------------
    # S1a
    levels_S1a = mk_levels("1.0000", "80")
    add("S1a", lambda: run_scenario(
        "S1a) CLOB-only, single-tier (target_out=50)",
        clob_levels=levels_S1a,
        amm=None,
        target_out=Decimal("50"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S1b
    amm_S1b = AMM(x_reserve="200000", y_reserve="200000", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S1b", lambda: run_scenario(
        "S1b) AMM-only, multi-iteration (target_out=400)",
        clob_levels=[],
        amm=amm_S1b,
        target_out=Decimal("400"),
        show_steps=show_steps,
        compact=compact,
    ))


    # S2a
    levels_S2a = mk_levels("1.0000", "60", "1.0100", "100")
    amm_S2a = AMM(x_reserve="10000", y_reserve="10040", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S2a", lambda: run_scenario(
        "S2a) CLOB≈AMM, same-iteration batching (target_out=65)",
        clob_levels=levels_S2a,
        amm=amm_S2a,
        target_out=Decimal("65"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S2b
    # Purpose: Verify **post-trade SPQ constraint** when AMM and CLOB coexist.
    # This tests the anchoring rule from the whitepaper §1.2.7.2:
    #   If AMM SPQ is better than the CLOB top quality, generate a synthetic AMM slice
    #   such that *after consuming that slice* the AMM's new SPQ is still at least as
    #   good as the CLOB top quality; the remainder of the tier then comes from CLOB.
    # What to look for in the output:
    #   - First iteration consumes a small AMM slice + some CLOB at the same tier.
    #   - The printed "AMM writeback" line shows new SPQ ~= top CLOB quality (not worse).
    #   - This ensures tier ordering stability in subsequent iterations.
    levels_S2b = mk_levels("1.0000", "100", "1.0100", "100")
    amm_S2b = AMM(x_reserve="10000", y_reserve="10050", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S2b", lambda: run_scenario(
        "S2b) AMM+CLOB anchoring w/ post-trade SPQ constraint (target_out=70)",
        clob_levels=levels_S2b,
        amm=amm_S2b,
        target_out=Decimal("70"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S3a
    levels_S3a = mk_levels("1.0000", "50", "1.0100", "100")
    add("S3a", lambda: run_scenario(
        "S3a) Demand exceeds top-tier capacity → second iteration (target_out=80)",
        clob_levels=levels_S3a,
        amm=None,
        target_out=Decimal("80"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S3b
    levels_S3b = mk_levels("1.0000", "50", "1.0100", "100")
    amm_S3b = AMM(x_reserve="10000", y_reserve="10020", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S3b", lambda: run_scenario(
        "S3b) AMM slightly worse than CLOB top (target_out=70)",
        clob_levels=levels_S3b,
        amm=amm_S3b,
        target_out=Decimal("70"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S3c
    levels_S3c = [
        ClobLevel.from_numbers("1.0000", "60"),
        ClobLevel.from_numbers("1.0050", "80"),
        ClobLevel.from_numbers("1.0100", "100"),
    ]
    add("S3c", lambda: run_scenario(
        "S3c) Large demand + multi-level CLOB (target_out=180)",
        clob_levels=levels_S3c,
        amm=None,
        target_out=Decimal("180"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S4a
    levels_S4a = mk_levels("1.0100", "60", "1.0200", "100")
    amm_S4a = AMM(x_reserve="10000", y_reserve="10150", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S4a", lambda: run_scenario(
        "S4a) send_max exact binding (target_out=80, send_max=80)",
        clob_levels=levels_S4a,
        amm=amm_S4a,
        target_out=Decimal("80"),
        send_max=Decimal("80"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S4b
    levels_S4b = mk_levels("1.0100", "60", "1.0200", "100")
    amm_S4b = AMM(x_reserve="10000", y_reserve="10150", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S4b", lambda: run_scenario(
        "S4b) deliver_min met (target_out=80, deliver_min=79)",
        clob_levels=levels_S4b,
        amm=amm_S4b,
        target_out=Decimal("80"),
        deliver_min=Decimal("79"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S4c
    levels_S4c = mk_levels("1.0300", "10")
    amm_S4c = AMM(x_reserve="500", y_reserve="520", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S4c", lambda: run_scenario(
        "S4c) deliver_min not met (target_out=200, deliver_min=150, send_max=120)",
        clob_levels=levels_S4c,
        amm=amm_S4c,
        target_out=Decimal("200"),
        send_max=Decimal("120"),
        deliver_min=Decimal("150"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S4d
    amm_S4d = AMM(x_reserve="10000", y_reserve="10000", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S4d", lambda: run_scenario(
        "S4d) target_out < OUT quantum (should trade ~0)",
        clob_levels=[],
        amm=amm_S4d,
        target_out=Decimal("5e-17"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S4e.1 min_quality CLOB filter
    levels_S4e1 = mk_levels("1.0000", "60", "1.0100", "100")
    add("S4e.1", lambda: run_scenario(
        "S4e.1) min_quality keeps only top CLOB (target_out=70, min_q=0.995)",
        clob_levels=levels_S4e1,
        amm=None,
        target_out=Decimal("70"),
        min_quality=Decimal("0.995"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S4e.2 min_quality AMM-only no fill
    amm_S4e2 = AMM(x_reserve="200000", y_reserve="200000", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S4e.2", lambda: run_scenario(
        "S4e.2) min_quality too high (AMM-only) → 0 fill (min_q=1.0)",
        clob_levels=[],
        amm=amm_S4e2,
        target_out=Decimal("80"),
        min_quality=Decimal("1.0"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S5a
    amm_S5a = AMM(x_reserve="10000", y_reserve="10000", fee="0.003", x_is_xrp=False, y_is_xrp=False, tr_in="0.02", tr_out="0.01")
    add("S5a", lambda: run_scenario(
        "S5a) IOU issuer fees tr_in=2%, tr_out=1% (target_out=50)",
        clob_levels=[],
        amm=amm_S5a,
        target_out=Decimal("50"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S5b
    amm_S5b = AMM(x_reserve="5000", y_reserve="5050", fee="0.003", x_is_xrp=True, y_is_xrp=False)
    add("S5b", lambda: run_scenario(
        "S5b) XRP↔IOU mixed grids (target_out=60)",
        clob_levels=mk_levels("1.0000", "50", "1.0050", "80"),
        amm=amm_S5b,
        target_out=Decimal("60"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S5c.1 / S5c.2
    amm_S5c0 = AMM(x_reserve="10000", y_reserve="10040", fee="0.000", x_is_xrp=False, y_is_xrp=False)
    add("S5c.1", lambda: run_scenario(
        "S5c.1) AMM fee=0% (target_out=65)",
        clob_levels=mk_levels("1.0000", "60", "1.0100", "100"),
        amm=amm_S5c0,
        target_out=Decimal("65"),
        show_steps=show_steps,
        compact=compact,
    ))
    amm_S5c1 = AMM(x_reserve="10000", y_reserve="10040", fee="0.010", x_is_xrp=False, y_is_xrp=False)
    add("S5c.2", lambda: run_scenario(
        "S5c.2) AMM fee=1% (target_out=65)",
        clob_levels=mk_levels("1.0000", "60", "1.0100", "100"),
        amm=amm_S5c1,
        target_out=Decimal("65"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S5d
    amm_S5d = AMM(x_reserve="1000", y_reserve="1000", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S5d", lambda: run_scenario(
        "S5d) AMM near exhaustion (target_out≈y)",
        clob_levels=[],
        amm=amm_S5d,
        target_out=Decimal("1000"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S5e.1 / S5e.2
    amm_S5e_iou = AMM(x_reserve="12000", y_reserve="12000", fee="0.003", x_is_xrp=False, y_is_xrp=False, tr_in="0.015", tr_out="0.010")
    add("S5e.1", lambda: run_scenario(
        "S5e.1) IOU↔IOU with issuer fees (target_out=200)",
        clob_levels=[],
        amm=amm_S5e_iou,
        target_out=Decimal("200"),
        show_steps=show_steps,
        compact=compact,
    ))
    amm_S5e_xrp = AMM(x_reserve="12000", y_reserve="12000", fee="0.003", x_is_xrp=True, y_is_xrp=False)
    add("S5e.2", lambda: run_scenario(
        "S5e.2) XRP↔IOU (no issuer fees on XRP side) (target_out=200)",
        clob_levels=[],
        amm=amm_S5e_xrp,
        target_out=Decimal("200"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S6a
    levels_S6a = [
        ClobLevel.from_numbers("1.0000", "70"),
        ClobLevel.from_numbers("1.0000", "50"),
    ]
    add("S6a", lambda: run_scenario(
        "S6a) Same price bucket tie (CLOB only)",
        clob_levels=levels_S6a,
        amm=None,
        target_out=Decimal("60"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S6b.1 / S6b.2
    amm_S6b = AMM(x_reserve="200000", y_reserve="200000", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S6b.1", lambda: run_scenario(
        "S6b.1) Curve with start_fraction=1e-4 (target_out=80)",
        clob_levels=[],
        amm=amm_S6b,
        target_out=Decimal("80"),
        segments_start_fraction=Decimal("1e-4"),
        show_steps=show_steps,
        compact=compact,
    ))
    add("S6b.2", lambda: run_scenario(
        "S6b.2) Curve with start_fraction=5e-4 (target_out=80)",
        clob_levels=[],
        amm=amm_S6b,
        target_out=Decimal("80"),
        segments_start_fraction=Decimal("5e-4"),
        show_steps=show_steps,
        compact=compact,
    ))

    # S7a
    amm_S7a = AMM(x_reserve="100000", y_reserve="100000", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    add("S7a", lambda: run_scenario(
        "S7a) AMM-only, very large target_out (target_out=8000)",
        clob_levels=[],
        amm=amm_S7a,
        target_out=Decimal("8000"),
        show_steps=show_steps,
        compact=compact,
    ))

    # --------------- Filter & run ---------------
    only_set = None
    skip_set = None
    if args.only:
        only_set = set([s.strip() for s in args.only.split(',') if s.strip()])
    if args.skip:
        skip_set = set([s.strip() for s in args.skip.split(',') if s.strip()])

    for sc in scenarios:
        if only_set is not None and sc.sid not in only_set:
            continue
        if skip_set is not None and sc.sid in skip_set:
            continue
        sc.fn()