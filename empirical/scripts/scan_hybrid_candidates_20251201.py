# empirical/scripts/scan_hybrid_candidates_20251201.py
# -*- coding: utf-8 -*-

"""
Scan hybrid (AMM+CLOB) candidate transactions from real-window data.

Goal:
- Find tx_hash that appear in BOTH AMM swaps and CLOB executed legs ("hybrid").
- For each hybrid tx, compute simple trade-level diagnostics to help pick an anchor:
    * clob_leg_count: number of CLOB execution rows for that tx
    * price monotonicity proxy within tx
    * AMM trade size (asset_out_value) to avoid extremes
    * time (T) from AMM side
- Rank and show top candidates.

Inputs:
- AMM swaps parquet window (fact_amm_swaps filtered for rUSD/XRP)
- CLOB parquet window (offers_fact_tx filtered for rUSD/XRP)
Defaults match your 2025-12-01 window.

Output:
- Print top candidates to console.
- Optionally write full candidate table to parquet/csv.

Usage:
    spark-submit --properties-file data/spark.properties empirical/scripts/scan_hybrid_candidates_20251201.py
    spark-submit ... empirical/scripts/scan_hybrid_candidates_20251201.py --top_n 30 --out data/output/hybrid_candidates_20251201
"""

import argparse
from pyspark.sql import SparkSession, functions as F, Window


DEFAULT_AMM_PATH = "data/output/amm_rusd_xrp_20251201"
DEFAULT_CLOB_PATH = "data/output/clob_rusd_xrp_20251201"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--amm", default=DEFAULT_AMM_PATH, help="AMM swaps parquet path")
    p.add_argument("--clob", default=DEFAULT_CLOB_PATH, help="CLOB tx parquet path")
    p.add_argument("--top_n", type=int, default=20, help="How many top candidates to display")
    p.add_argument("--out", default=None, help="Optional output dir (parquet). If set, writes full table")
    p.add_argument("--out_csv", default=None, help="Optional output dir (csv). If set, writes full table")
    return p.parse_args()


def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .appName("scan_hybrid_candidates_20251201")
        .getOrCreate()
    )

    amm = spark.read.parquet(args.amm)
    clob = spark.read.parquet(args.clob)

    # -----------------------------
    # 1) Build hybrid tx set (dup hashes)
    # -----------------------------
    amm_tx = amm.select(F.col("transaction_hash").alias("tx_hash")).distinct()
    clob_tx = clob.select("tx_hash").distinct()
    dup_tx = amm_tx.join(clob_tx, on="tx_hash", how="inner").cache()

    # -----------------------------
    # 2) Join AMM+dup for anchor-side fields
    #    We keep only AMM rows in dup set.
    # -----------------------------
    amm_hybrid = (
        amm.join(dup_tx, amm.transaction_hash == dup_tx.tx_hash, "inner")
           .select(
               F.col("transaction_hash").alias("tx_hash"),
               "ledger_index",
               "transaction_index",
               "close_time_datetime",
               "asset_in_currency",
               "asset_out_currency",
               "asset_in_value",
               "asset_out_value",
               "amm_account",
               "amm_asset_currency",
               "amm_asset2_currency",
           )
           .cache()
    )

    # If by any chance AMM has multiple rows per tx_hash (rare), keep the earliest by tx index.
    w_amm = Window.partitionBy("tx_hash").orderBy(F.col("transaction_index").asc())
    amm_hybrid_1 = (
        amm_hybrid.withColumn("rn_amm", F.row_number().over(w_amm))
                  .filter(F.col("rn_amm") == 1)
                  .drop("rn_amm")
                  .cache()
    )

    # -----------------------------
    # 3) Compute CLOB execution diagnostics per tx_hash
    # -----------------------------
    clob_hybrid = (
        clob.join(dup_tx, on="tx_hash", how="inner")
            .select(
                "tx_hash",
                "close_time",
                "price",
                "base_currency",
                "counter_currency",
                "base_amount",
                "counter_amount",
            )
            .cache()
    )

    # leg count and price stats
    clob_agg = (
        clob_hybrid.groupBy("tx_hash")
                   .agg(
                       F.count("*").alias("clob_leg_count"),
                       F.min("price").alias("clob_price_min"),
                       F.expr("percentile_approx(price, 0.5)").alias("clob_price_p50"),
                       F.max("price").alias("clob_price_max"),
                       F.sum("base_amount").alias("clob_base_sum"),
                       F.sum("counter_amount").alias("clob_counter_sum"),
                   )
    )

    # -----------------------------
    # 4) Price monotonicity proxy within each tx
    #    Steps:
    #      - sort legs by close_time (then price tie-break)
    #      - compute diff sign of price
    #      - monotone if all diffs >=0 or all <=0
    # -----------------------------
    w_clob = Window.partitionBy("tx_hash").orderBy(F.col("close_time").asc(), F.col("price").asc())
    clob_seq = (
        clob_hybrid
        .withColumn("prev_price", F.lag("price").over(w_clob))
        .withColumn("price_diff", F.col("price") - F.col("prev_price"))
        .withColumn(
            "diff_sign",
            F.when(F.col("price_diff") > 0, F.lit(1))
             .when(F.col("price_diff") < 0, F.lit(-1))
             .otherwise(F.lit(0))
        )
    )

    clob_sign_agg = (
        clob_seq.groupBy("tx_hash")
                .agg(
                    F.min("diff_sign").alias("diff_sign_min"),
                    F.max("diff_sign").alias("diff_sign_max"),
                )
                .withColumn(
                    "price_monotone_flag",
                    F.when(
                        (F.col("diff_sign_min") >= 0) | (F.col("diff_sign_max") <= 0),
                        F.lit(True)
                    ).otherwise(F.lit(False))
                )
    )

    # -----------------------------
    # 5) Merge AMM + CLOB diagnostics into candidate table
    # -----------------------------
    candidates = (
        amm_hybrid_1.join(clob_agg, on="tx_hash", how="left")
                    .join(clob_sign_agg, on="tx_hash", how="left")
                    .cache()
    )

    # -----------------------------
    # 6) Add AMM size percentiles for ranking (avoid extremes)
    # -----------------------------
    size_stats = amm_hybrid_1.agg(
        F.expr("percentile_approx(asset_out_value, 0.5)").alias("out_p50"),
        F.expr("percentile_approx(asset_out_value, 0.8)").alias("out_p80"),
        F.expr("percentile_approx(asset_out_value, 0.2)").alias("out_p20"),
    ).collect()[0]
    out_p50 = float(size_stats["out_p50"])
    out_p80 = float(size_stats["out_p80"])
    out_p20 = float(size_stats["out_p20"])

    # distance to median
    candidates = candidates.withColumn(
        "out_dist_to_p50",
        F.abs(F.col("asset_out_value") - F.lit(out_p50))
    )

    # middle-time proxy: distance to day's median time (seconds)
    # We compute median close_time_datetime among hybrid AMM rows.
    t_stats = amm_hybrid_1.agg(
        F.expr("percentile_approx(unix_timestamp(close_time_datetime), 0.5)").alias("t_p50_unix")
    ).collect()[0]
    t_p50_unix = int(t_stats["t_p50_unix"])

    candidates = candidates.withColumn(
        "t_dist_to_mid_sec",
        F.abs(F.unix_timestamp("close_time_datetime") - F.lit(t_p50_unix))
    )

    # -----------------------------
    # 7) Ranking:
    #    1) more CLOB legs (descending)
    #    2) monotone first
    #    3) closer to AMM size median
    #    4) closer to mid-day
    # -----------------------------
    ranked = (
        candidates.orderBy(
            F.col("clob_leg_count").desc(),
            F.col("price_monotone_flag").desc(),
            F.col("out_dist_to_p50").asc(),
            F.col("t_dist_to_mid_sec").asc(),
        )
    )

    # -----------------------------
    # 8) Console output
    # -----------------------------
    print("\n=== Hybrid candidate scan ===")
    print(f"AMM rows total         : {amm.count()}")
    print(f"CLOB rows total        : {clob.count()}")
    print(f"Hybrid tx count (dup)  : {dup_tx.count()}")
    print(f"Hybrid AMM rows        : {amm_hybrid_1.count()}")
    print(f"AMM asset_out p20/p50/p80 : {out_p20:.6f} / {out_p50:.6f} / {out_p80:.6f}")

    print("\n=== Top hybrid candidates ===")
    ranked.select(
        "tx_hash",
        "close_time_datetime",
        "asset_in_currency",
        "asset_out_currency",
        "asset_in_value",
        "asset_out_value",
        "clob_leg_count",
        "clob_price_min",
        "clob_price_p50",
        "clob_price_max",
        "price_monotone_flag",
        "out_dist_to_p50",
        "t_dist_to_mid_sec",
    ).show(args.top_n, truncate=False)

    # -----------------------------
    # 9) Optional write full table
    # -----------------------------
    if args.out:
        (ranked.coalesce(1)
               .write.mode("overwrite")
               .parquet(args.out))
        print(f"\nWrote full candidate table to parquet: {args.out}")

    if args.out_csv:
        (ranked.coalesce(1)
               .write.mode("overwrite")
               .option("header", True)
               .csv(args.out_csv))
        print(f"\nWrote full candidate table to csv: {args.out_csv}")

    spark.stop()


if __name__ == "__main__":
    main()
