"""Minimal demo to show CLOB → Segment conversion and AMM anchoring."""

from decimal import Decimal

from xrpl_router.clob import ClobLevel, Clob
from xrpl_router.amm import AMM
from xrpl_router.router import route

# Define a tiny CLOB (IOU/IOU example)
levels = [
    ClobLevel.from_numbers("1.0100", "60"),   # price=1.01 IN per OUT, 60 OUT available
    ClobLevel.from_numbers("1.0200", "100"),  # price=1.02 IN per OUT, 100 OUT available
]

# Build CLOB object (both sides as IOU here)
book = Clob(levels, in_is_xrp=False, out_is_xrp=False)

# Generate CLOB segments
segs = book.segments()

print("--- CLOB segments ---")
for s in segs:
    print(s)

# Build a tiny AMM (IOU/IOU)
amm = AMM(
    x_reserve="10000",  # IN side reserve (X)
    y_reserve="10150",  # OUT side reserve (Y)
    fee="0.003",        # 0.3% input-side fee
    x_is_xrp=False,
    y_is_xrp=False,
)

print("\nAMM SPQ:", amm.spq())

# Define AMM anchoring callback: anchor to LOB top-tier quality each iteration
amm_anchor = lambda q_threshold, need: amm.synthetic_segment_for_quality(
    q_threshold, max_out_cap=need
)

# Route target using CLOB + on-the-fly AMM anchoring
target_out = Decimal("80")
res = route(target_out, segs, amm_anchor=amm_anchor)

print("\n--- Anchored Route Result (CLOB + AMM) ---")
print(f"Filled OUT: {res.filled_out}")
print(f"Spent IN:   {res.spent_in}")
print(f"Avg quality: {res.avg_quality}")
print("Breakdown (per step):")
for step in res.trace:
    print(f"  {step['src']}: took {step['take_out']} OUT for {step['take_in']} IN (q={step['quality']})")