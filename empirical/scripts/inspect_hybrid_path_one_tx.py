# empirical/scripts/inspect_hybrid_path_one_tx.py
from pyspark.sql import SparkSession, functions as F
from decimal import Decimal, getcontext

getcontext().prec = 28

AMM_PATH  = "data/output/amm_rusd_xrp_20251201"
CLOB_PATH = "data/output/clob_rusd_xrp_20251201"

TX_HASH = "EE0AE95CEB8C2FBE6B10F54670D2340BF6883F129E5AF26661402FA1578B0BB2"

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"

spark = SparkSession.builder.appName("inspect_hybrid_path_one_tx").getOrCreate()

def to_dec(x):
    if x is None:
        return Decimal("0")
    return Decimal(str(x))

def pretty_cur(c: str) -> str:
    return "rUSD" if c == RUSD_HEX else c

print("\n=== Load windows ===")
amm  = spark.read.parquet(AMM_PATH)
clob = spark.read.parquet(CLOB_PATH)

# -----------------------------
# 0) Quick schema / flag sanity check (so filters actually work)
# -----------------------------
print("\n=== CLOB schema (check fulfilled_by / amm_account availability) ===")
clob.printSchema()

# -----------------------------
# 1) AMM anchor row (this is the AMM leg/slice, not full-tx totals)
# -----------------------------
amm_tx = amm.filter(F.col("transaction_hash") == TX_HASH).limit(1).collect()
if not amm_tx:
    raise RuntimeError(f"TX not found in AMM window: {TX_HASH}")
amm_tx = amm_tx[0]

print("\n=== AMM leg (from fact_amm_swaps) ===")
amm_in_cur  = amm_tx["asset_in_currency"]
amm_out_cur = amm_tx["asset_out_currency"]

amm_in_val  = to_dec(amm_tx["asset_in_value"])
amm_out_val = to_dec(amm_tx["asset_out_value"])

print(f"tx_hash            : {TX_HASH}")
print(f"close_time         : {amm_tx['close_time_datetime']}")
print(f"asset_in           : {amm_in_val} {pretty_cur(amm_in_cur)}")
print(f"asset_out          : {amm_out_val} {pretty_cur(amm_out_cur)}")

# AMM average price for this AMM leg, in router direction (in/out)
amm_avg_price_inout = (amm_in_val / amm_out_val) if amm_out_val != 0 else None
print(f"amm_avg_price      : {amm_avg_price_inout} ({pretty_cur(amm_in_cur)} per {pretty_cur(amm_out_cur)})")

# Pre/post pool states (real-world)
side1_cur = amm_tx["amm_asset_currency"]
side2_cur = amm_tx["amm_asset2_currency"]
side1_pre = to_dec(amm_tx["amm_asset_balance_before"])
side2_pre = to_dec(amm_tx["amm_asset2_balance_before"])
side1_post = to_dec(amm_tx["amm_asset_balance_after"])
side2_post = to_dec(amm_tx["amm_asset2_balance_after"])

print("pool_pre_reserves  : "
      f"{pretty_cur(side1_cur)}={side1_pre}, {pretty_cur(side2_cur)}={side2_pre}")
print("pool_post_reserves : "
      f"{pretty_cur(side1_cur)}={side1_post}, {pretty_cur(side2_cur)}={side2_post}")

# -----------------------------
# 2) CLOB executed legs for same tx
#    Try to exclude AMM-fulfilled legs if the columns exist.
# -----------------------------
clob_tx_df = clob.filter(F.col("tx_hash") == TX_HASH)

# Show distinct markers (helps validate whether filtering is possible)
print("\n=== fulfilled_by / amm_account distinct for this tx ===")
cols_to_show = []
if "fulfilled_by" in clob_tx_df.columns:
    cols_to_show.append(F.col("fulfilled_by").alias("fulfilled_by"))
if "amm_account" in clob_tx_df.columns:
    cols_to_show.append(F.col("amm_account").alias("amm_account"))
if cols_to_show:
    clob_tx_df.select(*cols_to_show).distinct().show(truncate=False)
else:
    print("[WARN] No fulfilled_by or amm_account columns in CLOB parquet; cannot tag AMM-fulfilled legs.")

# Robust filter: if amm_account exists and is non-empty, treat as AMM-fulfilled and drop.
# If no marker columns, fall back to all legs.
if "amm_account" in clob_tx_df.columns:
    clob_pure_df = clob_tx_df.filter(F.coalesce(F.col("amm_account"), F.lit("")) == F.lit(""))
else:
    clob_pure_df = clob_tx_df

# Select fields if present
select_cols = [
    "close_time",
    "price",
    "offer_base_currency", "offer_counter_currency",
    "base_currency", "counter_currency",
    "base_amount", "counter_amount",
]
select_cols = [c for c in select_cols if c in clob_pure_df.columns]

clob_legs = (
    clob_pure_df
        .orderBy(F.col("close_time").asc())
        .select(*select_cols)
        .collect()
)

if not clob_legs:
    print("\n[WARN] No (pure) CLOB legs found for this tx in CLOB window.")
else:
    print("\n=== CLOB executed legs (from offers_fact_tx) ===")
    router_in_cur  = amm_in_cur
    router_out_cur = amm_out_cur

    for i, r in enumerate(clob_legs, start=1):
        ct = r["close_time"]
        price = r["price"] if "price" in r.asDict() else None

        base_cur = r["base_currency"]
        counter_cur = r["counter_currency"]
        base_amt = to_dec(r["base_amount"])
        counter_amt = to_dec(r["counter_amount"])

        # Normalise to router direction for leg-level print:
        # OUT is router_out_cur; IN is router_in_cur.
        if base_cur == router_out_cur and counter_cur == router_in_cur:
            out_amt, in_amt = base_amt, counter_amt
        elif counter_cur == router_out_cur and base_cur == router_in_cur:
            out_amt, in_amt = counter_amt, base_amt
        else:
            out_amt, in_amt = Decimal("0"), Decimal("0")  # not aligned

        leg_price = (in_amt / out_amt) if out_amt != 0 else None

        print(
            f"L{i:02d} close_time={ct} "
            f"out={out_amt} {pretty_cur(router_out_cur)} "
            f"in={in_amt} {pretty_cur(router_in_cur)} "
            f"leg_price={leg_price} ({pretty_cur(router_in_cur)} per {pretty_cur(router_out_cur)})"
        )

# -----------------------------
# 3) Hybrid split summary (approximate full-tx totals)
#    IMPORTANT:
#    - AMM row is only the AMM leg.
#    - CLOB legs here are CLOB-only legs (AMM-fulfilled excluded if possible).
#    - Full tx totals are approximated by summing legs; if markers are missing,
#      totals may be inconsistent and should be treated as upper bounds.
# -----------------------------
router_in_cur = amm_in_cur
router_out_cur = amm_out_cur

clob_out_sum = Decimal("0")
clob_in_sum  = Decimal("0")

for r in clob_legs:
    base_cur = r["base_currency"]
    counter_cur = r["counter_currency"]
    base_amt = to_dec(r["base_amount"])
    counter_amt = to_dec(r["counter_amount"])

    # Accumulate OUT and IN in router direction.
    if base_cur == router_out_cur and counter_cur == router_in_cur:
        clob_out_sum += base_amt
        clob_in_sum  += counter_amt
    elif counter_cur == router_out_cur and base_cur == router_in_cur:
        clob_out_sum += counter_amt
        clob_in_sum  += base_amt
    else:
        pass

total_out_approx = amm_out_val + clob_out_sum
total_in_approx  = amm_in_val  + clob_in_sum

clob_avg_price_inout = (clob_in_sum / clob_out_sum) if clob_out_sum != 0 else None

print("\n=== Hybrid split (approx full-tx, by router direction) ===")
print(f"router_in_currency  : {pretty_cur(router_in_cur)}")
print(f"router_out_currency : {pretty_cur(router_out_cur)}")

print(f"amm_out_leg         : {amm_out_val} {pretty_cur(router_out_cur)}")
print(f"clob_out_leg_sum    : {clob_out_sum} {pretty_cur(router_out_cur)}")
print(f"total_out_approx    : {total_out_approx} {pretty_cur(router_out_cur)}")

print(f"amm_in_leg          : {amm_in_val} {pretty_cur(router_in_cur)}")
print(f"clob_in_leg_sum     : {clob_in_sum} {pretty_cur(router_in_cur)}")
print(f"total_in_approx     : {total_in_approx} {pretty_cur(router_in_cur)}")

print(f"clob_avg_price      : {clob_avg_price_inout} ({pretty_cur(router_in_cur)} per {pretty_cur(router_out_cur)})")

if clob_out_sum > amm_out_val and ("amm_account" not in clob_tx_df.columns and "fulfilled_by" not in clob_tx_df.columns):
    print("[WARN] CLOB legs cannot be filtered for AMM-fulfilled rows (no markers in parquet). "
          "Hybrid totals may be inconsistent; treat CLOB sums as execution segments, not additive OUT.")

spark.stop()
