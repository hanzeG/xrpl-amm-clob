from pyspark.sql import SparkSession, functions as F

RUSD = "524C555344000000000000000000000000000000"
RUSD_ISSUER = "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De"

DATE_MIN_STR = "2025-12-16"
DATE_MAX_STR = "2025-12-17"  # exclusive upper bound

AMM_TABLE  = "data/config.share#ripple-ubri-share.ripplex.fact_amm_swaps"
CLOB_TABLE = "data/config.share#ripple-ubri-share.ripplex.offers_fact_tx"

OUT_AMM  = "data/output/amm_rusd_xrp_20251216"
OUT_CLOB = "data/output/clob_rusd_xrp_20251216"

# CSV outputs (ledger-level stats) — ONLY AMM
OUT_AMM_LEDGER_CSV = "data/output_csv/amm_rusd_xrp_20251216_ledgers_csv"

spark = (
    SparkSession.builder
    .appName("export_rusd_xrp_window_20251216")
    .getOrCreate()
)

d_min = F.to_date(F.lit(DATE_MIN_STR))
d_max = F.to_date(F.lit(DATE_MAX_STR))

# =========================
# AMM: fact_amm_swaps
# =========================
amm_swaps = (
    spark.read.format("deltaSharing")
    .load(AMM_TABLE)
    .select(
        "transaction_hash",
        "ledger_index",
        "transaction_index",
        "close_time_datetime",
        "close_time_date",
        "account",
        "destination",
        "transaction_type",
        "payment_currency",
        "payment_issuer",
        "payment_amount",
        "payment_delivermax",
        "payment_delivermin",
        "payment_sendmax_currency",
        "payment_sendmax_issuer",
        "payment_sendmax_amount",
        "offer_base_currency",
        "offer_counter_currency",
        "offer_base_issuer",
        "offer_counter_issuer",
        "offer_base_amount",
        "offer_counter_amount",
        "amm_account",
        "amm_asset_currency",
        "amm_asset_issuer",
        "amm_asset_balance_before",
        "amm_asset_balance_after",
        "amm_asset_balance_delta",
        "amm_asset2_currency",
        "amm_asset2_issuer",
        "amm_asset2_balance_before",
        "amm_asset2_balance_after",
        "amm_asset2_balance_delta",
        "asset_in_currency",
        "asset_in_issuer",
        "asset_in_value",
        "asset_out_currency",
        "asset_out_issuer",
        "asset_out_value",
        "source_tag",
        "destination_tag",
        "memo",
        "transaction_fee",
    )
)

amm_rusd_xrp = amm_swaps.filter(
    (F.col("close_time_date") >= d_min) &
    (F.col("close_time_date") < d_max) &
    (
        (
            (F.col("amm_asset_currency") == RUSD) &
            (F.col("amm_asset_issuer") == RUSD_ISSUER) &
            (F.col("amm_asset2_currency") == "XRP")
        ) |
        (
            (F.col("amm_asset2_currency") == RUSD) &
            (F.col("amm_asset2_issuer") == RUSD_ISSUER) &
            (F.col("amm_asset_currency") == "XRP")
        )
    )
)

amm_cnt = amm_rusd_xrp.count()
print(f"AMM rUSD-XRP swaps in window [{DATE_MIN_STR}, {DATE_MAX_STR}) = {amm_cnt}")

(
    amm_rusd_xrp
    .orderBy(F.col("close_time_datetime").asc())
    .write.mode("overwrite")
    .parquet(OUT_AMM)
)
print(f"AMM data written to: {OUT_AMM}")

# AMM ledger stats CSV (for finding ledgers with AMM activity)
amm_ledgers = (
    amm_rusd_xrp
    .groupBy("ledger_index")
    .agg(
        F.count("*").alias("amm_swap_count"),
        F.min("close_time_datetime").alias("amm_first_time"),
        F.max("close_time_datetime").alias("amm_last_time"),
        F.min("transaction_index").alias("amm_min_tx_index"),
        F.max("transaction_index").alias("amm_max_tx_index"),
    )
    .orderBy(F.col("amm_swap_count").desc(), F.col("ledger_index").asc())
)

(
    amm_ledgers
    .coalesce(1)  # one csv file
    .write.mode("overwrite")
    .option("header", True)
    .csv(OUT_AMM_LEDGER_CSV)
)
print(f"AMM ledger stats CSV written to: {OUT_AMM_LEDGER_CSV}")

# =========================
# CLOB: offers_fact_tx
# =========================
clob_tx = (
    spark.read.format("deltaSharing")
    .load(CLOB_TABLE)
    .select(
        "close_time",
        "close_time_date",
        "tx_hash",
        "transaction_type",
        "fulfilled_by",
        "account",
        "destination",
        "payment_currency",
        "payment_issuer",
        "payment_amount",
        "offer_base_currency",
        "offer_counter_currency",
        "offer_base_issuer",
        "offer_counter_issuer",
        "offer_base_amount",
        "offer_counter_amount",
        "base_receiver",
        "base_currency",
        "base_issuer",
        "base_amount",
        "counter_receiver",
        "counter_currency",
        "counter_issuer",
        "counter_amount",
        "amm_account",
        "price",
    )
)

clob_rusd_xrp = clob_tx.filter(
    (F.col("close_time_date") >= d_min) &
    (F.col("close_time_date") < d_max) &
    (
        (
            (F.col("offer_base_currency") == RUSD) &
            (F.col("offer_base_issuer") == RUSD_ISSUER) &
            (F.col("offer_counter_currency") == "XRP")
        ) |
        (
            (F.col("offer_counter_currency") == RUSD) &
            (F.col("offer_counter_issuer") == RUSD_ISSUER) &
            (F.col("offer_base_currency") == "XRP")
        ) |
        (
            (F.col("base_currency") == RUSD) &
            (F.col("base_issuer") == RUSD_ISSUER) &
            (F.col("counter_currency") == "XRP")
        ) |
        (
            (F.col("counter_currency") == RUSD) &
            (F.col("counter_issuer") == RUSD_ISSUER) &
            (F.col("base_currency") == "XRP")
        )
    )
)

clob_cnt = clob_rusd_xrp.count()
print(f"CLOB rUSD-XRP tx in window [{DATE_MIN_STR}, {DATE_MAX_STR}) = {clob_cnt}")

(
    clob_rusd_xrp
    .orderBy(F.col("close_time").asc())
    .write.mode("overwrite")
    .parquet(OUT_CLOB)
)
print(f"CLOB data written to: {OUT_CLOB}")

spark.stop()