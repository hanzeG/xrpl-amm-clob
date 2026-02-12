from pyspark.sql import SparkSession, functions as F
import os

# -----------------------------
# Target ledger window (inclusive)
# -----------------------------
LEDGER_START = 100915230
LEDGER_END   = 100916230

# -----------------------------
# Inputs from your Spark exports
# -----------------------------
AMM_PATH  = "data/output/amm_rusd_xrp_100915230_100916230"
CLOB_PATH = "data/output/clob_rusd_xrp_100915230_100916230"

# -----------------------------
# Mapping from rippled:
# a table with columns:
#   tx_hash (string), ledger_index (bigint), transaction_index (bigint)
# You will generate this separately from rippled (ledger+meta).
# -----------------------------
TX_MAP_PATH = "data/output/tx_map_100915230_100916230"  # parquet dir (recommended)

RUSD = "524C555344000000000000000000000000000000"
XRP  = "XRP"

# Output root
OUT_ROOT = "data/output/mixed_timeline_100915230_100916230"
OUT_TIMELINE_PARQUET = os.path.join(OUT_ROOT, "timeline_parquet")
OUT_SUMMARY_PARQUET  = os.path.join(OUT_ROOT, "summary_parquet")
OUT_HOURLY_PARQUET   = os.path.join(OUT_ROOT, "hourly_parquet")

OUT_TIMELINE_CSV = os.path.join(OUT_ROOT, "timeline_csv")
OUT_SUMMARY_CSV  = os.path.join(OUT_ROOT, "summary_csv")
OUT_HOURLY_CSV   = os.path.join(OUT_ROOT, "hourly_csv")

spark = (
    SparkSession.builder
    .appName("mixed_timeline_rusd_xrp_100915230_100916230_taker_consistent")
    .getOrCreate()
)

# =====================================================
# Load windows
# =====================================================
amm  = spark.read.parquet(AMM_PATH)
clob = spark.read.parquet(CLOB_PATH)

# =====================================================
# Load tx mapping (rippled-derived) for strict ordering & 2nd-pass filtering
# =====================================================
# Expect schema: tx_hash, ledger_index, transaction_index
tx_map = spark.read.parquet(TX_MAP_PATH).select(
    F.col("tx_hash").alias("map_tx_hash"),
    F.col("ledger_index").cast("bigint").alias("map_ledger_index"),
    F.col("transaction_index").cast("bigint").alias("map_transaction_index"),
).dropDuplicates(["map_tx_hash"])

# =====================================================
# AMM events (tx-level) — taker in/out is explicit
# =====================================================
amm_events = (
    amm
    .withColumn("event_time", F.col("close_time_datetime"))
    .withColumn("source", F.lit("AMM"))
    .withColumnRenamed("transaction_hash", "tx_hash")
    # Taker direction (pay -> get)
    .withColumn(
        "direction",
        F.when(
            (F.col("asset_in_currency") == XRP) & (F.col("asset_out_currency") == RUSD),
            F.lit("XRP->rUSD")
        ).when(
            (F.col("asset_in_currency") == RUSD) & (F.col("asset_out_currency") == XRP),
            F.lit("rUSD->XRP")
        ).otherwise(F.lit("other"))
    )
    .withColumn("amount_in",  F.col("asset_in_value"))   # taker pays
    .withColumn("amount_out", F.col("asset_out_value"))  # taker gets
    .withColumn(
        "avg_price_in_out",
        F.when(F.col("asset_out_value") != 0,
               F.col("asset_in_value") / F.col("asset_out_value"))
    )
    # Extra fields for debugging / market structure
    .withColumn("maker_account", F.lit(None).cast("string"))
    .withColumn("taker_account", F.col("account"))  # AMM swap initiator
    .withColumn("base_currency", F.lit(None).cast("string"))
    .withColumn("counter_currency", F.lit(None).cast("string"))
    .withColumn("base_amount", F.lit(None).cast("double"))
    .withColumn("counter_amount", F.lit(None).cast("double"))
    .select(
        "event_time",
        "source",
        "tx_hash",
        "ledger_index",
        "transaction_index",
        "direction",
        "amount_in",
        "amount_out",
        "avg_price_in_out",
        F.lit(None).cast("double").alias("clob_price"),
        "maker_account",
        "taker_account",
        "base_currency",
        "counter_currency",
        "base_amount",
        "counter_amount",
    )
)

# =====================================================
# CLOB events (leg-level) — infer taker direction from receivers
# Then join tx_map to add (ledger_index, transaction_index) and do 2nd-pass filter.
# =====================================================
clob_enriched = (
    clob
    .withColumn("event_time", F.col("close_time"))
    .withColumn("source", F.lit("CLOB"))
    .withColumn("maker_account", F.col("account"))
    .withColumn(
        "taker_account",
        F.when(F.col("account") == F.col("base_receiver"), F.col("counter_receiver"))
         .when(F.col("account") == F.col("counter_receiver"), F.col("base_receiver"))
         .otherwise(F.lit(None).cast("string"))
    )
    # taker pays / gets currencies
    .withColumn(
        "taker_pays_currency",
        F.when(F.col("account") == F.col("base_receiver"), F.col("base_currency"))
         .when(F.col("account") == F.col("counter_receiver"), F.col("counter_currency"))
         .otherwise(F.lit(None).cast("string"))
    )
    .withColumn(
        "taker_gets_currency",
        F.when(F.col("account") == F.col("base_receiver"), F.col("counter_currency"))
         .when(F.col("account") == F.col("counter_receiver"), F.col("base_currency"))
         .otherwise(F.lit(None).cast("string"))
    )
    # taker pays / gets amounts (amount_in = pays, amount_out = gets)
    .withColumn(
        "amount_in",
        F.when(F.col("account") == F.col("base_receiver"), F.col("base_amount"))
         .when(F.col("account") == F.col("counter_receiver"), F.col("counter_amount"))
         .otherwise(F.lit(None).cast("double"))
    )
    .withColumn(
        "amount_out",
        F.when(F.col("account") == F.col("base_receiver"), F.col("counter_amount"))
         .when(F.col("account") == F.col("counter_receiver"), F.col("base_amount"))
         .otherwise(F.lit(None).cast("double"))
    )
    # Direction label in XRP/rUSD terms, based on taker pays/gets
    .withColumn(
        "direction",
        F.when(
            (F.col("taker_pays_currency") == XRP) & (F.col("taker_gets_currency") == RUSD),
            F.lit("XRP->rUSD")
        ).when(
            (F.col("taker_pays_currency") == RUSD) & (F.col("taker_gets_currency") == XRP),
            F.lit("rUSD->XRP")
        ).otherwise(F.lit("other"))
    )
)

# Join mapping to add strict chain ordering and do 2nd-pass ledger filtering
clob_with_idx = (
    clob_enriched
    .join(tx_map, clob_enriched["tx_hash"] == tx_map["map_tx_hash"], "left")
    .withColumn("ledger_index", F.col("map_ledger_index"))
    .withColumn("transaction_index", F.col("map_transaction_index"))
    .drop("map_tx_hash", "map_ledger_index", "map_transaction_index")
)

# 2nd-pass filter: keep only rows whose tx maps into the target ledger window
clob_with_idx = clob_with_idx.filter(
    (F.col("ledger_index").isNotNull()) &
    (F.col("ledger_index") >= F.lit(LEDGER_START)) &
    (F.col("ledger_index") <= F.lit(LEDGER_END))
)

clob_events = (
    clob_with_idx
    .select(
        "event_time",
        "source",
        "tx_hash",
        "ledger_index",
        "transaction_index",
        "direction",
        "amount_in",
        "amount_out",
        F.lit(None).cast("double").alias("avg_price_in_out"),
        F.col("price").alias("clob_price"),
        "maker_account",
        "taker_account",
        "base_currency",
        "counter_currency",
        "base_amount",
        "counter_amount",
    )
)

# =====================================================
# Unified strictly-ordered timeline
# =====================================================
timeline = (
    amm_events.unionByName(clob_events)
    .orderBy(
        F.col("ledger_index").asc(),
        F.col("transaction_index").asc(),
        F.col("source").asc(),
        F.col("tx_hash").asc()
    )
    .cache()
)

print("\n=== Unified AMM + CLOB timeline (strict order, first 200 rows) ===")
timeline.show(200, truncate=False)

# =====================================================
# Summary
# =====================================================
summary = timeline.groupBy("source").agg(
    F.count("*").alias("rows"),
    F.countDistinct("tx_hash").alias("unique_tx"),
)

print(f"\n=== Summary in ledger window [{LEDGER_START}, {LEDGER_END}] ===")
summary.show(truncate=False)

# Hourly activity (optional; uses event_time)
hourly = (
    timeline
    .withColumn("hour", F.date_trunc("hour", F.col("event_time")))
    .groupBy("hour", "source")
    .count()
    .orderBy("hour", "source")
)

print("\n=== Hourly activity (AMM + CLOB) ===")
hourly.show(48, truncate=False)

# =====================================================
# Write outputs
# =====================================================
print("\n=== Writing outputs ===")
print(f"OUT_ROOT = {OUT_ROOT}")

timeline.write.mode("overwrite").parquet(OUT_TIMELINE_PARQUET)
summary.write.mode("overwrite").parquet(OUT_SUMMARY_PARQUET)
hourly.write.mode("overwrite").parquet(OUT_HOURLY_PARQUET)

(timeline.coalesce(1)
         .write.mode("overwrite")
         .option("header", True)
         .csv(OUT_TIMELINE_CSV))

(summary.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(OUT_SUMMARY_CSV))

(hourly.coalesce(1)
       .write.mode("overwrite")
       .option("header", True)
       .csv(OUT_HOURLY_CSV))

print("Wrote parquet:")
print(f"  timeline -> {OUT_TIMELINE_PARQUET}")
print(f"  summary  -> {OUT_SUMMARY_PARQUET}")
print(f"  hourly   -> {OUT_HOURLY_PARQUET}")

print("Wrote csv:")
print(f"  timeline -> {OUT_TIMELINE_CSV}")
print(f"  summary  -> {OUT_SUMMARY_CSV}")
print(f"  hourly   -> {OUT_HOURLY_CSV}")

spark.stop()