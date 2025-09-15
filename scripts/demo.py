"""Minimal demo to show CLOB → Segment conversion and AMM anchoring, with optional limits."""

from decimal import Decimal

from xrpl_router.clob import ClobLevel, Clob
from xrpl_router.amm import AMM
from xrpl_router.router import route, RouteError


def print_segments(title, segs):
    print(f"\n=== {title} ===")
    for s in segs:
        print(s)
    print("Legend: quality=OUT/IN; out_max=max deliverable OUT; in_at_out_max=IN needed for out_max; in_is_xrp/out_is_xrp indicate amount grid.")


def print_route_result(title, res):
    print(f"\n=== {title} ===")
    print(f"Filled OUT: {res.filled_out}")
    print(f"Spent IN:   {res.spent_in}")
    print(f"Avg quality: {res.avg_quality}")
    print("Steps:")
    cum_out = Decimal("0")
    cum_in = Decimal("0")
    for i, step in enumerate(res.trace, 1):
        so = step['take_out']
        si = step['take_in']
        inst_q = (so / si) if si > 0 else Decimal("0")
        cum_out += so
        cum_in += si
        print(f"  {i:02d}. {step['src']}: out={so} in={si} inst_q={inst_q} (tier_q={step['quality']}) | cum_out={cum_out} cum_in={cum_in}")
    if res.usage:
        print("Usage summary (OUT by source):", res.usage)


# -----------------------------
# Build CLOB and AMM test data
# -----------------------------
levels = [
    ClobLevel.from_numbers("1.0100", "60"),   # price=1.01 IN per OUT, 60 OUT available
    ClobLevel.from_numbers("1.0200", "100"),  # price=1.02 IN per OUT, 100 OUT available
]
book = Clob(levels, in_is_xrp=False, out_is_xrp=False)
segs = book.segments()
print_segments("CLOB segments", segs)

amm = AMM(
    x_reserve="10000",  # IN side reserve (X)
    y_reserve="10150",  # OUT side reserve (Y)
    fee="0.003",        # 0.3% input-side fee
    x_is_xrp=False,
    y_is_xrp=False,
)
print("\n=== AMM initial state ===")
print(f"Reserves: X={amm.x} Y={amm.y}")
print(f"SPQ:      {amm.spq()}")

# AMM anchoring callback: anchor to LOB top-tier quality each iteration
amm_anchor = lambda q_threshold, need: amm.synthetic_segment_for_quality(
    q_threshold, max_out_cap=need
)

# Track iterations for clarity
iter_no = {"n": 0}

def after_iter(sum_in, sum_out):
    # Apply the iteration's AMM fill back to pool and report new state
    iter_no["n"] += 1
    amm.apply_fill(sum_in, sum_out)
    print(f"[iteration {iter_no['n']}] AMM fill: +{sum_in} IN, -{sum_out} OUT")
    print(f"[iteration {iter_no['n']}] AMM reserves: X={amm.x} Y={amm.y} SPQ={amm.spq()}")

# -------- Scenario A: unconstrained anchored routing --------
print("\n=== Scenario A: Unconstrained (target_out=80) ===")
print("Plan: anchor AMM to LOB top each iteration; no send_max/deliver_min.")

target_out = Decimal("80")
res = route(target_out, segs, amm_anchor=amm_anchor, after_iteration=after_iter)
print_route_result("Anchored Route Result (CLOB + AMM)", res)

print("\n=== AMM state before Scenario B ===")
print(f"Reserves: X={amm.x} Y={amm.y}")
print(f"SPQ:      {amm.spq()}")

# -------- Scenario B: with limits (send_max / deliver_min) --------
print("\n=== Scenario B: With limits (target_out=80, send_max=80, deliver_min=75) ===")
print("Plan: limit total IN to 80; require at least 75 OUT; enforce forward recompute under budget.")

send_max = Decimal("80")
deliver_min = Decimal("75")

try:
    res_limited = route(
        target_out,
        segs,  # CLOB segments only; AMM is injected per-iteration via amm_anchor
        amm_anchor=amm_anchor,
        send_max=send_max,
        deliver_min=deliver_min,
        after_iteration=after_iter,
    )
    print_route_result("Anchored Route with Limits", res_limited)
except RouteError as e:
    print("\n--- Anchored Route with Limits ---")
    print(f"Routing failed: {e}")
    print(f"(send_max={send_max}, deliver_min={deliver_min}, target_out={target_out})")