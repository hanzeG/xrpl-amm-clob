"""Minimal demo to show CLOB → Segment conversion."""

from xrpl_router.clob import ClobLevel, Clob
from xrpl_router.amm import AMM

# Define a tiny CLOB (IOU/IOU example)
levels = [
    ClobLevel.from_numbers("1.0100", "60"),   # price=1.01 IN per OUT, 60 OUT available
    ClobLevel.from_numbers("1.0200", "100"),  # price=1.02 IN per OUT, 100 OUT available
]

# Build CLOB object (both sides as IOU here)
book = Clob(levels, in_is_xrp=False, out_is_xrp=False)

# Generate segments
segs = book.segments()

# Print result
for s in segs:
    print(s)

# Build a tiny AMM (IOU/IOU) and generate AMM segments for the same target
amm = AMM(
    x_reserve="10000",  # IN side reserve (X)
    y_reserve="10150",  # OUT side reserve (Y)
    fee="0.003",        # 0.3% input-side fee
    x_is_xrp=False,
    y_is_xrp=False,
)
am_segs = amm.segments_for_out("80")

print("\nAMM SPQ:", amm.spq())
for s in am_segs:
    print(s)

# Merge CLOB + AMM segments and route
all_segs = [*segs, *am_segs]

from decimal import Decimal
from xrpl_router.router import route

target_out = Decimal("80")
res = route(target_out, all_segs)

print("\n--- Mixed Route Result (CLOB + AMM) ---")
print(f"Filled OUT: {res.filled_out}")
print(f"Spent IN:   {res.spent_in}")
print(f"Avg quality: {res.avg_quality}")
print("Breakdown (per step):")
for step in res.trace:
    print(f"  {step['src']}: took {step['take_out']} OUT for {step['take_in']} IN (q={step['quality']})")