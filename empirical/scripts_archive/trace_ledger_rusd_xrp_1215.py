import argparse
import os
from pyspark.sql import SparkSession, functions as F

RUSD = "524C555344000000000000000000000000000000"
RUSD_ISSUER = "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De"
XRP = "XRP"


def write_csv_onefile(df, out_dir: str):
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(out_dir)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=int, required=True)
    ap.add_argument("--amm-swaps", required=True)
    ap.add_argument("--amm-fees", required=True)
    ap.add_argument("--clob", required=True)  # 仍保留参数，但不再用 ndjson 过滤 ledger
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ledger = int(args.ledger)

    spark = (
        SparkSession.builder
        .appName(f"trace_ledger_rusd_xrp_no_rpc_{ledger}")
        .getOrCreate()
    )

    os.makedirs(args.out, exist_ok=True)

    # -----------------------
    # AMM swaps (strict)
    # -----------------------
    amm_swaps_raw = spark.read.parquet(args.amm_swaps)

    amm_pair_filter = (
        (
            (F.col("amm_asset_currency") == RUSD) &
            (F.col("amm_asset_issuer") == RUSD_ISSUER) &
            (F.col("amm_asset2_currency") == XRP)
        ) |
        (
            (F.col("amm_asset2_currency") == RUSD) &
            (F.col("amm_asset2_issuer") == RUSD_ISSUER) &
            (F.col("amm_asset_currency") == XRP)
        )
    )

    amm_swaps = (
        amm_swaps_raw
        .filter((F.col("ledger_index") == F.lit(ledger)) & amm_pair_filter)
        .withColumnRenamed("transaction_hash", "tx_hash")
        .withColumn("source", F.lit("AMM_SWAP"))
        .withColumn("event_time", F.col("close_time_datetime"))
        .withColumn("maker_account", F.lit(None).cast("string"))
        .withColumn("taker_account", F.col("account"))
        .withColumn("clob_price", F.lit(None).cast("double"))
        .withColumn("base_currency", F.lit(None).cast("string"))
        .withColumn("counter_currency", F.lit(None).cast("string"))
        .withColumn("base_amount", F.lit(None).cast("double"))
        .withColumn("counter_amount", F.lit(None).cast("double"))
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
        .withColumn("amount_in",  F.col("asset_in_value").cast("double"))
        .withColumn("amount_out", F.col("asset_out_value").cast("double"))
        .withColumn(
            "avg_price_in_out",
            F.when(
                F.col("asset_out_value").cast("double") != 0,
                F.col("asset_in_value").cast("double") / F.col("asset_out_value").cast("double")
            )
        )
        .withColumn("amm_fee_value", F.lit(None).cast("double"))
        .withColumn("amm_fee_currency", F.lit(None).cast("string"))
        .select(
            "event_time",
            "source",
            "tx_hash",
            F.col("ledger_index").cast("bigint").alias("ledger_index"),
            F.col("transaction_index").cast("bigint").alias("transaction_index"),
            "direction",
            "amount_in",
            "amount_out",
            "avg_price_in_out",
            "clob_price",
            "maker_account",
            "taker_account",
            "base_currency",
            "counter_currency",
            "base_amount",
            "counter_amount",
            "amm_account",
            F.col("transaction_fee").cast("double").alias("transaction_fee"),
            "amm_fee_value",
            "amm_fee_currency",
        )
    )

    # -----------------------
    # AMM fees (strict; tie to pools seen in swaps)
    # -----------------------
    amm_accounts = amm_swaps.select("amm_account").dropna(subset=["amm_account"]).dropDuplicates()

    amm_fees_raw = spark.read.parquet(args.amm_fees)

    amm_fees = (
        amm_fees_raw
        .filter(F.col("ledger_index") == F.lit(ledger))
        .join(amm_accounts, on="amm_account", how="inner")
        .withColumnRenamed("transaction_hash", "tx_hash")
        .withColumn("source", F.lit("AMM_FEE"))
        .withColumn("event_time", F.col("close_time_datetime"))
        .withColumn("direction", F.lit(None).cast("string"))
        .withColumn("amount_in", F.lit(None).cast("double"))
        .withColumn("amount_out", F.lit(None).cast("double"))
        .withColumn("avg_price_in_out", F.lit(None).cast("double"))
        .withColumn("clob_price", F.lit(None).cast("double"))
        .withColumn("maker_account", F.lit(None).cast("string"))
        .withColumn("taker_account", F.col("account"))
        .withColumn("base_currency", F.lit(None).cast("string"))
        .withColumn("counter_currency", F.lit(None).cast("string"))
        .withColumn("base_amount", F.lit(None).cast("double"))
        .withColumn("counter_amount", F.lit(None).cast("double"))
        .withColumn("transaction_fee", F.lit(None).cast("double"))
        .withColumn("amm_fee_value", F.col("trading_fee_value").cast("double"))
        .withColumn("amm_fee_currency", F.col("trading_fee_currency").cast("string"))
        .select(
            "event_time",
            "source",
            "tx_hash",
            F.col("ledger_index").cast("bigint").alias("ledger_index"),
            F.col("transaction_index").cast("bigint").alias("transaction_index"),
            "direction",
            "amount_in",
            "amount_out",
            "avg_price_in_out",
            "clob_price",
            "maker_account",
            "taker_account",
            "base_currency",
            "counter_currency",
            "base_amount",
            "counter_amount",
            "amm_account",
            "transaction_fee",
            "amm_fee_value",
            "amm_fee_currency",
        )
    )

    amm_strict = (
        amm_swaps.unionByName(amm_fees)
        .orderBy(
            F.col("ledger_index").asc(),
            F.col("transaction_index").asc(),
            F.col("source").asc(),
            F.col("tx_hash").asc(),
        )
    )

    # -----------------------
    # CLOB: 不再用 ndjson；由于 clob parquet 没有 ledger_index，无法严格过滤到单一 ledger
    # 这里直接跳过（输出空表），避免给你造成“混入别的 ledger 执行”的错觉
    # -----------------------
    clob_schema = amm_strict.schema
    clob_empty = spark.createDataFrame([], schema=clob_schema)

    clob_ordered = clob_empty

    combined = (
        amm_strict.unionByName(clob_ordered)
        .orderBy(
            F.col("ledger_index").asc(),
            F.col("transaction_index").asc_nulls_last(),
            F.col("source").asc(),
            F.col("tx_hash").asc(),
        )
    )

    print(f"\n=== Ledger {ledger} counts ===")
    print("AMM swaps:", amm_swaps.count())
    print("AMM fees :", amm_fees.count())
    print("CLOB     :", 0)

    out_amm = os.path.join(args.out, f"ledger_{ledger}_amm_strict_parquet")
    out_clob = os.path.join(args.out, f"ledger_{ledger}_clob_parquet")
    out_comb = os.path.join(args.out, f"ledger_{ledger}_combined_parquet")

    out_amm_csv = os.path.join(args.out, f"ledger_{ledger}_amm_strict_csv")
    out_clob_csv = os.path.join(args.out, f"ledger_{ledger}_clob_csv")
    out_comb_csv = os.path.join(args.out, f"ledger_{ledger}_combined_csv")

    amm_strict.write.mode("overwrite").parquet(out_amm)
    clob_ordered.write.mode("overwrite").parquet(out_clob)
    combined.write.mode("overwrite").parquet(out_comb)

    write_csv_onefile(amm_strict, out_amm_csv)
    write_csv_onefile(clob_ordered, out_clob_csv)
    write_csv_onefile(combined, out_comb_csv)

    print("\nWrote parquet:")
    print(" ", out_amm)
    print(" ", out_clob)
    print(" ", out_comb)

    print("\nWrote csv:")
    print(" ", out_amm_csv)
    print(" ", out_clob_csv)
    print(" ", out_comb_csv)

    spark.stop()


if __name__ == "__main__":
    main()