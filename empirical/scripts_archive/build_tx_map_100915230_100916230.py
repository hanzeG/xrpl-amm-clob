# empirical/scripts/build_tx_map_100915230_100916230.py
import json
import os
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from pyspark.sql import SparkSession, functions as F

RPC_URL = "http://127.0.0.1:5005"

LEDGER_START = 100915230
LEDGER_END   = 100916230  # inclusive

OUT_DIR = "data/output/tx_map_100915230_100916230"

# IMPORTANT:
# - Do NOT start filenames with "_" (Spark may ignore them).
# - Do NOT write parquet to OUT_DIR directly with mode("overwrite"), otherwise it will wipe tmp_tx_map.ndjson.
TMP_NDJSON   = os.path.join(OUT_DIR, "tmp_tx_map.ndjson")
FAILED_TXT   = os.path.join(OUT_DIR, "_failed_ledgers.txt")
OUT_PARQUET  = os.path.join(OUT_DIR, "parquet")  # write parquet to a subdirectory

# Retry policy
MAX_RETRIES = 6
BASE_SLEEP_SEC = 0.25  # exponential backoff base


def rpc_call(method: str, params: dict) -> dict:
    payload = {"method": method, "params": [params]}
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        RPC_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_ledger(ledger_index: int) -> dict:
    # We want expanded tx objects with metadata so we can read meta.TransactionIndex
    params = {
        "ledger_index": ledger_index,
        "transactions": True,
        "expand": True,
        "binary": False,
    }

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            out = rpc_call("ledger", params)
            result = out.get("result", {})
            if result.get("status") == "success" and "ledger" in result:
                return result["ledger"]

            last_err = out
            time.sleep(BASE_SLEEP_SEC * (2 ** attempt))
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = str(e)
            time.sleep(BASE_SLEEP_SEC * (2 ** attempt))

    raise RuntimeError(
        f"Failed to fetch ledger {ledger_index} after retries. Last error: {last_err}"
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Pull ledgers and write an NDJSON of tx_hash -> (ledger_index, transaction_index)
    n_rows = 0
    failed = []

    with open(TMP_NDJSON, "w", encoding="utf-8") as f:
        for li in range(LEDGER_START, LEDGER_END + 1):
            try:
                ledger = fetch_ledger(li)
            except Exception as e:
                print(f"ledger {li}: ERROR {e}", file=sys.stderr)
                failed.append(li)
                continue

            txs = ledger.get("transactions", [])
            for tx in txs:
                tx_hash = tx.get("hash") or tx.get("tx_hash") or tx.get("tx")
                meta = tx.get("meta") or tx.get("metaData") or {}
                tx_index = meta.get("TransactionIndex")

                # Skip anything without a strict ordering key
                if tx_hash is None or tx_index is None:
                    continue

                rec = {
                    "tx_hash": tx_hash,
                    "ledger_index": int(li),
                    "transaction_index": int(tx_index),
                }
                f.write(json.dumps(rec) + "\n")
                n_rows += 1

            if li % 100 == 0:
                print(f"progress: fetched up to ledger {li}, rows={n_rows}")

    with open(FAILED_TXT, "w", encoding="utf-8") as ff:
        for li in failed:
            ff.write(str(li) + "\n")

    print(f"Wrote temporary ndjson: {TMP_NDJSON}")
    print(f"Failed ledgers (if any): {FAILED_TXT}")
    print(f"Total tx_map rows (before parquet): {n_rows}")

    # 2) Convert NDJSON -> parquet via Spark
    spark = (
        SparkSession.builder
        .appName(f"build_tx_map_{LEDGER_START}_{LEDGER_END}")
        .getOrCreate()
    )

    # Read the NDJSON (one JSON object per line)
    df = spark.read.json(TMP_NDJSON)

    # Ensure types, drop nulls, de-dup by tx_hash, and sort for convenience
    df2 = (
        df.select(
            F.col("tx_hash").cast("string").alias("tx_hash"),
            F.col("ledger_index").cast("bigint").alias("ledger_index"),
            F.col("transaction_index").cast("bigint").alias("transaction_index"),
        )
        .dropna(subset=["tx_hash", "ledger_index", "transaction_index"])
        .dropDuplicates(["tx_hash"])
        .orderBy(F.col("ledger_index").asc(), F.col("transaction_index").asc())
        .cache()  # materialise so later actions don't re-scan the source unexpectedly
    )

    # Force evaluation now (so we fail early if JSON read has issues)
    total = df2.count()
    print(f"Total tx_map rows (after de-dup): {total}")

    print("Schema:")
    df2.printSchema()
    print("Sample:")
    df2.show(20, truncate=False)

    # Write parquet to a subdirectory (do NOT overwrite OUT_DIR, or you'll delete tmp_tx_map.ndjson)
    df2.write.mode("overwrite").parquet(OUT_PARQUET)
    print(f"tx_map parquet written to: {OUT_PARQUET}")

    spark.stop()


if __name__ == "__main__":
    main()
