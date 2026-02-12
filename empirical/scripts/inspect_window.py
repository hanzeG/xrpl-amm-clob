# empirical/scripts/inspect_window.py
from pyspark.sql import SparkSession, functions as F
import os, sys

spark = SparkSession.builder.appName("inspect_window_export_csv").getOrCreate()

AMM_PARQUET = "data/output/amm_rusd_xrp_20251201"
CLOB_PARQUET = "data/output/clob_rusd_xrp_20251201"
OUT_DIR = "data/output_csv"

# Optional sampling to avoid accidental huge exports in the future.
# Set SAMPLE_N to None to export full data.
SAMPLE_N = None  # e.g., 1000 for a small preview

def require_path(p):
    if not os.path.exists(p):
        print(f"Path not found: {p}", file=sys.stderr)
        spark.stop()
        sys.exit(1)

require_path(AMM_PARQUET)
require_path(CLOB_PARQUET)
os.makedirs(OUT_DIR, exist_ok=True)

amm = spark.read.parquet(AMM_PARQUET)
clob = spark.read.parquet(CLOB_PARQUET)

# Cache counts to avoid running multiple Spark jobs for count()
amm_cnt = amm.count()
clob_cnt = clob.count()

# AMM has close_time_datetime, CLOB has close_time
amm_sorted = amm.orderBy(F.col("close_time_datetime").asc())
clob_sorted = clob.orderBy(F.col("close_time").asc())

# Optionally export samples only
if SAMPLE_N is not None:
    amm_sorted = amm_sorted.limit(int(SAMPLE_N))
    clob_sorted = clob_sorted.limit(int(SAMPLE_N))

amm_out = os.path.join(OUT_DIR, "amm_rusd_xrp_20251201_csv")
clob_out = os.path.join(OUT_DIR, "clob_rusd_xrp_20251201_csv")

# coalesce(1) is fine for small windows, but avoid it for large exports
(amm_sorted.coalesce(1)
    .write.mode("overwrite")
    .option("header", True)
    .csv(amm_out)
)

(clob_sorted.coalesce(1)
    .write.mode("overwrite")
    .option("header", True)
    .csv(clob_out)
)

print(f"AMM rows = {amm_cnt}, CSV -> {amm_out}")
print(f"CLOB rows = {clob_cnt}, CSV -> {clob_out}")

spark.stop()
