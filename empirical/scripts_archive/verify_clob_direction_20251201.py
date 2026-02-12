from pyspark.sql import SparkSession, functions as F

CLOB_PATH = "data/output/clob_rusd_xrp_20251201"
RUSD = "524C555344000000000000000000000000000000"
XRP  = "XRP"

spark = SparkSession.builder.appName("verify_clob_direction_20251201").getOrCreate()
clob = spark.read.parquet(CLOB_PATH)

# ------------------------------------------------------------
# 1) Coverage check:
#    Verify that the maker account is always one of the receivers.
#    If a material number of rows fall into "neither", the receiver-based
#    inference is not applicable for those rows.
# ------------------------------------------------------------
cov = (
    clob.select(
        F.when(F.col("account") == F.col("base_receiver"), F.lit("maker_is_base_receiver"))
         .when(F.col("account") == F.col("counter_receiver"), F.lit("maker_is_counter_receiver"))
         .otherwise(F.lit("neither"))
         .alias("receiver_case")
    )
    .groupBy("receiver_case")
    .count()
)

cov.show(truncate=False)

# ------------------------------------------------------------
# 2) Derive taker pays/gets currencies using receiver relationships:
#    - account is assumed to be the maker (Offer owner)
#    - base_receiver receives base_currency/base_amount
#    - counter_receiver receives counter_currency/counter_amount
#
#    If maker == base_receiver:
#      maker gets base, taker gets counter  => taker pays base, gets counter
#    If maker == counter_receiver:
#      maker gets counter, taker gets base => taker pays counter, gets base
# ------------------------------------------------------------
taker_pays = (
    F.when(F.col("account") == F.col("base_receiver"), F.col("base_currency"))
     .when(F.col("account") == F.col("counter_receiver"), F.col("counter_currency"))
)

taker_gets = (
    F.when(F.col("account") == F.col("base_receiver"), F.col("counter_currency"))
     .when(F.col("account") == F.col("counter_receiver"), F.col("base_currency"))
)

derived_dir = (
    F.when((taker_pays == XRP) & (taker_gets == RUSD), F.lit("XRP->rUSD"))
     .when((taker_pays == RUSD) & (taker_gets == XRP), F.lit("rUSD->XRP"))
     .otherwise(F.lit("other"))
)

check = clob.withColumn("derived_direction", derived_dir)
check.groupBy("derived_direction").count().show(truncate=False)

# ------------------------------------------------------------
# 3) Strong numerical consistency check:
#    Empirically, offers_fact_tx.price is consistent with:
#        price ≈ amount_out / amount_in
#    (i.e., taker-gets per taker-pays), not amount_in/amount_out.
#
#    We therefore validate:
#        abs_err = | (amount_out/amount_in) - price |
#    for rows where the derived direction is XRP<->rUSD.
# ------------------------------------------------------------
df = (
    check
    .withColumn(
        "amount_in",  # taker pays
        F.when(F.col("account") == F.col("base_receiver"), F.col("base_amount"))
         .when(F.col("account") == F.col("counter_receiver"), F.col("counter_amount"))
    )
    .withColumn(
        "amount_out",  # taker gets
        F.when(F.col("account") == F.col("base_receiver"), F.col("counter_amount"))
         .when(F.col("account") == F.col("counter_receiver"), F.col("base_amount"))
    )
    .withColumn(
        "ratio_out_in",
        F.when(F.col("amount_in") != 0, F.col("amount_out") / F.col("amount_in"))
    )
    .withColumn("abs_err", F.abs(F.col("ratio_out_in") - F.col("price")))
    .filter(F.col("derived_direction").isin("XRP->rUSD", "rUSD->XRP"))
)

df.agg(
    F.count("*").alias("n"),
    F.expr("percentile_approx(abs_err, 0.5)").alias("abs_err_p50"),
    F.expr("percentile_approx(abs_err, 0.95)").alias("abs_err_p95"),
    F.max("abs_err").alias("abs_err_max"),
).show(truncate=False)

spark.stop()