# empirical/scripts/export_amm_fees_window_20251216.py
from pyspark.sql import SparkSession, functions as F

DATE_MIN_STR = "2025-12-16"
DATE_MAX_STR = "2025-12-17"  # exclusive

AMM_FEES_TABLE  = "data/config.share#ripple-ubri-share.ripplex.fact_amm_fees"
OUT_FEES_WINDOW = "data/output/amm_fees_20251216"

spark = (
    SparkSession.builder
    .appName("export_amm_fees_window_20251216")
    .getOrCreate()
)

d_min = F.to_date(F.lit(DATE_MIN_STR))
d_max = F.to_date(F.lit(DATE_MAX_STR))

fees = (
    spark.read.format("deltaSharing")
    .load(AMM_FEES_TABLE)
)

fees_window = fees.filter(
    (F.col("close_time_date") >= d_min) &
    (F.col("close_time_date") < d_max)
)

cnt = fees_window.count()
print(f"AMM fees rows in window [{DATE_MIN_STR}, {DATE_MAX_STR}) = {cnt}")

(
    fees_window
    .orderBy(F.col("close_time_datetime").asc())
    .write.mode("overwrite")
    .parquet(OUT_FEES_WINDOW)
)

print(f"Fees window written to: {OUT_FEES_WINDOW}")

spark.stop()
