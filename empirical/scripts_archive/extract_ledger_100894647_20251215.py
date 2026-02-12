# empirical/scripts/extract_ledger_100894647_20251215.py
# Extract one ledger from 2025-12-15 outputs. CLOB is filtered by tx hash list.

import os
import json
from pyspark.sql import SparkSession, functions as F

LEDGER_INDEX = 100894647

IN_AMM  = "data/output/amm_rusd_xrp_20251215"
IN_CLOB = "data/output/clob_rusd_xrp_20251215"
IN_FEES = "data/output/amm_fees_20251215"

LEDGER_JSON = "artifacts/snapshots/legacy_root/ledger_100894647_expanded.json"

OUT_ROOT = f"artifacts/snapshots/legacy_root/ledger/ledger_{LEDGER_INDEX}_20251215"

RUSD_HEX = "524C555344000000000000000000000000000000"
RUSD_ISSUER = "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De"


def assert_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing path: {path}")


def load_ledger_tx_hashes(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    txs = obj.get("result", {}).get("ledger", {}).get("transactions", [])
    hashes = []
    for t in txs:
        h = t.get("hash")
        if h:
            hashes.append(h)
    hashes = sorted(set(hashes))
    if not hashes:
        raise ValueError(f"No transaction hashes found in {path}")
    return hashes


def add_readable_currency_cols(df, specs):
    # specs: list of (currency_col, issuer_col, prefix)
    for ccy_col, iss_col, prefix in specs:
        if ccy_col in df.columns:
            df = df.withColumn(
                f"{prefix}_currency_readable",
                F.when(F.col(ccy_col) == "XRP", F.lit("XRP"))
                 .when(F.col(ccy_col) == RUSD_HEX, F.lit("rUSD"))
                 .otherwise(F.col(ccy_col))
            )
        if iss_col and iss_col in df.columns:
            df = df.withColumn(
                f"{prefix}_issuer_readable",
                F.when(F.col(iss_col) == RUSD_ISSUER, F.lit("rUSD_issuer"))
                 .otherwise(F.col(iss_col))
            )
    return df


def write_all_formats(df, name: str, order_cols=None):
    out_dir = os.path.join(OUT_ROOT, name)
    out_csv = os.path.join(out_dir, "csv")
    out_parquet = os.path.join(out_dir, "parquet")
    out_ndjson = os.path.join(out_dir, "ndjson")

    if order_cols:
        df = df.orderBy(*[F.col(c).asc() for c in order_cols if c in df.columns])

    df.write.mode("overwrite").parquet(out_parquet)

    (df.coalesce(1)
       .write.mode("overwrite")
       .option("header", True)
       .csv(out_csv))

    (df.select(F.to_json(F.struct(*[F.col(c) for c in df.columns])).alias("json"))
       .coalesce(1)
       .write.mode("overwrite")
       .text(out_ndjson))

    print(f"[OK] {name}")
    print(f"  Parquet: {out_parquet}")
    print(f"  CSV:     {out_csv}")
    print(f"  NDJSON:  {out_ndjson}")


def distinct_count(df, col):
    if col not in df.columns:
        return None
    return df.select(col).where(F.col(col).isNotNull()).distinct().count()


def main():
    assert_exists(IN_AMM)
    assert_exists(IN_CLOB)
    assert_exists(IN_FEES)
    assert_exists(LEDGER_JSON)

    ledger_tx_hashes = load_ledger_tx_hashes(LEDGER_JSON)
    print(f"Ledger {LEDGER_INDEX} tx hashes loaded: {len(ledger_tx_hashes)}")

    spark = SparkSession.builder.appName("extract_ledger_100894647_20251215").getOrCreate()

    # AMM swaps
    amm = spark.read.parquet(IN_AMM).filter(F.col("ledger_index") == F.lit(LEDGER_INDEX))
    amm = add_readable_currency_cols(amm, [
        ("payment_currency", "payment_issuer", "payment"),
        ("payment_sendmax_currency", "payment_sendmax_issuer", "sendmax"),
        ("offer_base_currency", "offer_base_issuer", "offer_base"),
        ("offer_counter_currency", "offer_counter_issuer", "offer_counter"),
        ("amm_asset_currency", "amm_asset_issuer", "amm_asset1"),
        ("amm_asset2_currency", "amm_asset2_issuer", "amm_asset2"),
        ("asset_in_currency", "asset_in_issuer", "asset_in"),
        ("asset_out_currency", "asset_out_issuer", "asset_out"),
    ])
    print(f"AMM swaps rows @ ledger {LEDGER_INDEX}: {amm.count()}")
    write_all_formats(amm, "amm_swaps", ["close_time_datetime", "transaction_index", "transaction_hash"])

    # CLOB tx (filter by tx hash list)
    clob = spark.read.parquet(IN_CLOB)

    tx_col = None
    for c in ["tx_hash", "transaction_hash", "hash"]:
        if c in clob.columns:
            tx_col = c
            break
    if tx_col is None:
        raise ValueError(f"No tx hash column found in {IN_CLOB}. Columns: {clob.columns}")

    # isin() is fine here because one ledger has limited tx count
    clob = clob.filter(F.col(tx_col).isin(ledger_tx_hashes))

    clob = add_readable_currency_cols(clob, [
        ("payment_currency", "payment_issuer", "payment"),
        ("offer_base_currency", "offer_base_issuer", "offer_base"),
        ("offer_counter_currency", "offer_counter_issuer", "offer_counter"),
        ("base_currency", "base_issuer", "base"),
        ("counter_currency", "counter_issuer", "counter"),
    ])

    print(f"CLOB tx rows matched by tx hash @ ledger {LEDGER_INDEX}: {clob.count()}")
    write_all_formats(clob, "clob_tx", ["close_time", tx_col])

    # AMM fees
    fees = spark.read.parquet(IN_FEES).filter(F.col("ledger_index") == F.lit(LEDGER_INDEX))
    fees = add_readable_currency_cols(fees, [
        ("asset_currency", "asset_issuer", "asset"),
        ("asset2_currency", "asset2_issuer", "asset2"),
    ])
    print(f"AMM fees rows @ ledger {LEDGER_INDEX}: {fees.count()}")
    write_all_formats(fees, "amm_fees", ["close_time_datetime", "transaction_index", "transaction_hash"])

    print("\n=== Summary ===")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"AMM swaps distinct tx: {distinct_count(amm, 'transaction_hash')}")
    print(f"CLOB tx distinct tx:   {distinct_count(clob, tx_col)}")
    print(f"AMM fees distinct tx:  {distinct_count(fees, 'transaction_hash')}")

    spark.stop()


if __name__ == "__main__":
    main()
