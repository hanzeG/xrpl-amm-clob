# empirical/scripts/overview_rusd_xrp_window_20251201.py
from pyspark.sql import SparkSession, functions as F

AMM_PATH = "data/output/amm_rusd_xrp_20251216"
CLOB_PATH = "data/output/clob_rusd_xrp_20251216"

RUSD = "524C555344000000000000000000000000000000"
RUSD_ISSUER = "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De"

spark = (
    SparkSession.builder
    .appName("overview_rusd_xrp_window_20251216")
    .getOrCreate()
)

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# ---------------- AMM overview ----------------
print_header("AMM WINDOW OVERVIEW")

amm = spark.read.parquet(AMM_PATH)

amm_cnt = amm.count()
print(f"rows = {amm_cnt}")

amm_time = amm.agg(
    F.min("close_time_datetime").alias("min_time"),
    F.max("close_time_datetime").alias("max_time")
).collect()[0]
print(f"time range = [{amm_time['min_time']}, {amm_time['max_time']}]")

amm_uniques = amm.agg(
    F.countDistinct("transaction_hash").alias("tx_hash_unique"),
    F.countDistinct("amm_account").alias("amm_pool_unique"),
    F.countDistinct("account").alias("trader_unique")
).collect()[0].asDict()  # Row -> dict
print(
    "unique tx_hash = {tx_hash_unique}, unique amm pools = {amm_pool_unique}, unique traders = {trader_unique}"
    .format(**amm_uniques)
)

amm_dir = (
    amm.withColumn(
        "direction",
        F.when((F.col("asset_in_currency") == "XRP") & (F.col("asset_out_currency") == RUSD), "XRP->rUSD")
         .when((F.col("asset_in_currency") == RUSD) & (F.col("asset_out_currency") == "XRP"), "rUSD->XRP")
         .otherwise("other")
    )
    .groupBy("direction")
    .count()
)
print("direction counts:")
amm_dir.orderBy(F.col("count").desc()).show(truncate=False)

amm_amount_stats = amm.agg(
    F.min("asset_in_value").alias("asset_in_min"),
    F.expr("percentile_approx(asset_in_value, 0.5)").alias("asset_in_p50"),
    F.expr("percentile_approx(asset_in_value, 0.95)").alias("asset_in_p95"),
    F.max("asset_in_value").alias("asset_in_max"),
    F.min("asset_out_value").alias("asset_out_min"),
    F.expr("percentile_approx(asset_out_value, 0.5)").alias("asset_out_p50"),
    F.expr("percentile_approx(asset_out_value, 0.95)").alias("asset_out_p95"),
    F.max("asset_out_value").alias("asset_out_max"),
).collect()[0]
print("asset_in stats (min/p50/p95/max) =",
      amm_amount_stats["asset_in_min"], amm_amount_stats["asset_in_p50"],
      amm_amount_stats["asset_in_p95"], amm_amount_stats["asset_in_max"])
print("asset_out stats (min/p50/p95/max) =",
      amm_amount_stats["asset_out_min"], amm_amount_stats["asset_out_p50"],
      amm_amount_stats["asset_out_p95"], amm_amount_stats["asset_out_max"])

print("hourly activity:")
amm_hourly = (
    amm.withColumn("hour", F.date_trunc("hour", F.col("close_time_datetime")))
       .groupBy("hour")
       .count()
       .orderBy("hour")
)
amm_hourly.show(48, truncate=False)

print("top AMM pools by swaps:")
amm.groupBy("amm_account").count().orderBy(F.col("count").desc()).show(10, truncate=False)

print("top traders by swaps:")
amm.groupBy("account").count().orderBy(F.col("count").desc()).show(10, truncate=False)


# ---------------- CLOB overview ----------------
print_header("CLOB WINDOW OVERVIEW")

clob = spark.read.parquet(CLOB_PATH)

clob_cnt = clob.count()
print(f"rows = {clob_cnt}")

clob_time = clob.agg(
    F.min("close_time").alias("min_time"),
    F.max("close_time").alias("max_time")
).collect()[0]
print(f"time range = [{clob_time['min_time']}, {clob_time['max_time']}]")

clob_uniques = clob.agg(
    F.countDistinct("tx_hash").alias("tx_hash_unique"),
    F.countDistinct("account").alias("maker_unique"),
    F.countDistinct("base_receiver").alias("base_receiver_unique"),
    F.countDistinct("counter_receiver").alias("counter_receiver_unique")
).collect()[0].asDict()  # Row -> dict
print(
    "unique tx_hash = {tx_hash_unique}, unique makers = {maker_unique}, "
    "unique base_receivers = {base_receiver_unique}, unique counter_receivers = {counter_receiver_unique}"
    .format(**clob_uniques)
)

clob_dir = (
    clob.withColumn(
        "direction",
        F.when((F.col("offer_base_currency") == "XRP") & (F.col("offer_counter_currency") == RUSD), "sell XRP for rUSD")
         .when((F.col("offer_base_currency") == RUSD) & (F.col("offer_counter_currency") == "XRP"), "sell rUSD for XRP")
         .otherwise("other")
    )
    .groupBy("direction")
    .count()
)
print("offer direction counts:")
clob_dir.orderBy(F.col("count").desc()).show(truncate=False)

clob_exec_dir = (
    clob.withColumn(
        "exec_direction",
        F.when((F.col("base_currency") == "XRP") & (F.col("counter_currency") == RUSD), "base XRP / counter rUSD")
         .when((F.col("base_currency") == RUSD) & (F.col("counter_currency") == "XRP"), "base rUSD / counter XRP")
         .otherwise("other")
    )
    .groupBy("exec_direction")
    .count()
)
print("executed leg direction counts:")
clob_exec_dir.orderBy(F.col("count").desc()).show(truncate=False)

price_stats = clob.agg(
    F.min("price").alias("price_min"),
    F.expr("percentile_approx(price, 0.5)").alias("price_p50"),
    F.expr("percentile_approx(price, 0.95)").alias("price_p95"),
    F.max("price").alias("price_max"),
).collect()[0]
print("price stats (min/p50/p95/max) =",
      price_stats["price_min"], price_stats["price_p50"],
      price_stats["price_p95"], price_stats["price_max"])

print("hourly activity:")
clob_hourly = (
    clob.withColumn("hour", F.date_trunc("hour", F.col("close_time")))
        .groupBy("hour")
        .count()
        .orderBy("hour")
)
clob_hourly.show(48, truncate=False)

print("top makers by tx:")
clob.groupBy("account").count().orderBy(F.col("count").desc()).show(10, truncate=False)


# ---------------- Duplicate tx hash check ----------------
print_header("AMM vs CLOB DUPLICATE TX HASH CHECK")

# Normalise column names: AMM uses `transaction_hash`, CLOB uses `tx_hash`
amm_tx = amm.select(F.col("transaction_hash").alias("tx_hash")).distinct()
clob_tx = clob.select("tx_hash").distinct()

dup_tx = amm_tx.join(clob_tx, on="tx_hash", how="inner").cache()

dup_cnt = dup_tx.count()
print(f"duplicate tx_hash count = {dup_cnt}")

if dup_cnt > 0:
    print("sample duplicate tx_hash:")
    dup_tx.orderBy("tx_hash").show(50, truncate=False)

    # Optional: how many AMM rows / CLOB rows belong to duplicates
    amm_dup_rows = amm.join(dup_tx, amm.transaction_hash == dup_tx.tx_hash, "inner").count()
    clob_dup_rows = clob.join(dup_tx, "tx_hash", "inner").count()
    print(f"AMM rows with duplicate tx_hash = {amm_dup_rows}")
    print(f"CLOB rows with duplicate tx_hash = {clob_dup_rows}")

spark.stop()
