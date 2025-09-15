"""Comprehensive demo: CLOB↔AMM routing under whitepaper semantics (anchoring, tiers, limits).

Scenarios covered:
A) Both sources, unconstrained (single-tier fill)
B) Both sources, send_max + deliver_min (forward recompute under budget)
C) CLOB strictly better than AMM (AMM contributes 0 in tier)
D) AMM-only (no CLOB levels)
E) Demand exceeds tier-1 capacity → multi-iteration (tier-1 then tier-2)
F) Failure case: deliver_min cannot be met
"""
from __future__ import annotations

from decimal import Decimal
from typing import List, Optional, Callable

from xrpl_router.clob import ClobLevel, Clob
from xrpl_router.amm import AMM
from xrpl_router.router import route, RouteError


# ---------- pretty printers ----------

def brief_book(levels: List[ClobLevel]) -> str:
    if not levels:
        return "CLOB: (empty)"
    parts = [f"CLOB[{i+1}]: price={lvl.price_in_per_out} IN/OUT, out_max={lvl.out_liquidity}" for i, lvl in enumerate(levels)]
    return "; ".join(parts)


def print_result(title: str, filled_out: Decimal, spent_in: Decimal, avg_q: Decimal, trace):
    print(f"\n=== {title} ===")
    print(f"Totals: OUT={filled_out}  IN={spent_in}  avg_q={avg_q}")
    print("Steps (src | out | in | inst_q | tier_q):")
    for s in trace:
        so, si = s['take_out'], s['take_in']
        inst_q = (so / si) if si > 0 else Decimal("0")
        print(f"  {s['src']:>4} | {so} | {si} | {inst_q} | {s['quality']}")


# ---------- scenario runner ----------

def run_scenario(
    title: str,
    *,
    clob_levels: List[ClobLevel],
    amm: Optional[AMM],
    target_out: Decimal,
    send_max: Optional[Decimal] = None,
    deliver_min: Optional[Decimal] = None,
):
    print("\n" + "=" * 80)
    print(f"Scenario: {title}")
    # Build CLOB segments
    if clob_levels is not None:
        book = Clob(clob_levels, in_is_xrp=False, out_is_xrp=False)
        segs = book.segments()
    else:
        segs = []
    print("Market:")
    print("  ", brief_book(clob_levels))
    if amm is not None:
        print(f"  AMM: reserves X={amm.x}, Y={amm.y}, SPQ={amm.spq()}")
    else:
        print("  AMM: (none)")

    # Anchoring & writeback callbacks
    amm_anchor: Optional[Callable[[Decimal, Decimal], Optional[object]]] = None
    iter_no = {"n": 0}

    if amm is not None:
        amm_anchor = lambda q_threshold, need: amm.synthetic_segment_for_quality(
            q_threshold, max_out_cap=need
        )
        amm_curve = lambda need: amm.segments_for_out(need)

        def after_iter(sum_in: Decimal, sum_out: Decimal) -> None:
            iter_no["n"] += 1
            amm.apply_fill(sum_in, sum_out)
            print(f"    [iter {iter_no['n']}] AMM writeback: +{sum_in} IN, -{sum_out} OUT; new SPQ={amm.spq()}")
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
        print_result("Route Result", res.filled_out, res.spent_in, res.avg_quality, res.trace)
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


# ---------- run scenarios ----------
if __name__ == "__main__":
    # A) Both sources, unconstrained (single-tier fill)
    levels_A = mk_levels("1.0100", "60", "1.0200", "100")
    amm_A = AMM(x_reserve="10000", y_reserve="10150", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    run_scenario(
        "A) both sources, unconstrained (target_out=80)",
        clob_levels=levels_A,
        amm=amm_A,
        target_out=Decimal("80"),
    )

    # B) Both sources, send_max + deliver_min
    levels_B = mk_levels("1.0100", "60", "1.0200", "100")
    amm_B = AMM(x_reserve="10000", y_reserve="10150", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    run_scenario(
        "B) both sources with limits (target_out=80, send_max=80, deliver_min=75)",
        clob_levels=levels_B,
        amm=amm_B,
        target_out=Decimal("80"),
        send_max=Decimal("80"),
        deliver_min=Decimal("75"),
    )

    # C) CLOB strictly better than AMM (AMM contributes 0)
    # Make AMM worse: small X, large Y, and/or higher fee to push SPQ below LOB top.
    levels_C = mk_levels("1.0000", "50", "1.0100", "100")  # LOB top quality = 1.0
    amm_C = AMM(x_reserve="8000", y_reserve="12000", fee="0.006", x_is_xrp=False, y_is_xrp=False)  # SPQ < 1.0 expected
    run_scenario(
        "C) CLOB strictly better; AMM anchored slice absent",
        clob_levels=levels_C,
        amm=amm_C,
        target_out=Decimal("70"),
    )

    # D) AMM-only (no CLOB levels)
    levels_D: List[ClobLevel] = []
    amm_D = AMM(x_reserve="9000", y_reserve="9000", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    run_scenario(
        "D) AMM-only (no CLOB)",
        clob_levels=levels_D,
        amm=amm_D,
        target_out=Decimal("100"),
    )

    # E) Demand exceeds tier-1 capacity → multi-iteration
    # LOB top 60 + AMM anchored ≈ 20 < target_out 120 → will enter tier-2 (LOB second level + re-anchored AMM)
    levels_E = mk_levels("1.0100", "60", "1.0200", "100")
    amm_E = AMM(x_reserve="10000", y_reserve="10150", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    run_scenario(
        "E) multi-iteration (target_out=120)",
        clob_levels=levels_E,
        amm=amm_E,
        target_out=Decimal("120"),
    )

    # F) Failure case: deliver_min cannot be met
    levels_F = mk_levels("1.0300", "10")  # very small LOB; AMM also limited
    amm_F = AMM(x_reserve="500", y_reserve="520", fee="0.003", x_is_xrp=False, y_is_xrp=False)
    run_scenario(
        "F) failure: deliver_min not met (target_out=200, deliver_min=150, send_max=120)",
        clob_levels=levels_F,
        amm=amm_F,
        target_out=Decimal("200"),
        send_max=Decimal("120"),
        deliver_min=Decimal("150"),
    )