import argparse
import json
import os
import random
from decimal import Decimal, getcontext

from pyspark.sql import SparkSession, functions as F, Window

getcontext().prec = 28

# ---------------- Defaults ----------------
DEFAULT_AMM_PATH = "data/output/amm_rusd_xrp_20251201"
DEFAULT_CLOB_PATH = "data/output/clob_rusd_xrp_20251201"
DEFAULT_FEES_PATH = "data/output/amm_fees_20251201"

DEFAULT_OUT_JSON = "data/output/model_input_amm_only.json"

RUSD_HEX = "524C555344000000000000000000000000000000"
XRP = "XRP"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--amm", default=DEFAULT_AMM_PATH, help="AMM swaps parquet path")
    p.add_argument("--clob", default=DEFAULT_CLOB_PATH, help="CLOB parquet path")
    p.add_argument("--fees", default=DEFAULT_FEES_PATH, help="AMM fees parquet path (full window)")
    p.add_argument("--seed", type=int, default=7, help="Random seed for anchor selection")
    p.add_argument(
        "--use_discounted_fee",
        action="store_true",
        help="Use discounted_fee (ex-post) instead of trading_fee (ex-ante)"
    )
    p.add_argument("--out_json", default=DEFAULT_OUT_JSON, help="Output JSON path")
    p.add_argument("--export_range", action="store_true", help="Export a ledger_index range to out_dir and exit")
    p.add_argument("--ledger_start", type=int, default=None, help="Start ledger_index (inclusive) for range export")
    p.add_argument("--ledger_end", type=int, default=None, help="End ledger_index (inclusive) for range export")
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for range export (e.g., artifacts/snapshots/legacy_root/ledger_100893230_to_100895416)",
    )
    p.add_argument("--fetch_range", action="store_true", help="Fetch/derive rUSD-XRP datasets for a ledger_index range from source datasets")
    p.add_argument("--src_amm", default=None, help="Source AMM swaps dataset path (parquet)")
    p.add_argument("--src_fees", default=None, help="Source AMM fees dataset path (parquet)")
    p.add_argument("--src_clob", default=None, help="Source CLOB legs dataset path (parquet)")
    p.add_argument("--share_profile", default="data/config.share", help="Delta Sharing profile file")
    p.add_argument("--share", default=None, help="Delta Sharing share name")
    p.add_argument("--schema", default=None, help="Delta Sharing schema name")
    p.add_argument("--table_amm", default="ripplex_fact_amm_swaps", help="Delta Sharing table for AMM swaps")
    p.add_argument("--table_fees", default="ripplex_fact_amm_fees", help="Delta Sharing table for AMM fees")
    p.add_argument("--table_clob", default="ripplex_fact_clob_tx", help="Delta Sharing table for CLOB legs")
    return p.parse_args()


def to_decimal(x) -> Decimal:
    """Safe conversion to Decimal."""
    if x is None:
        return Decimal("0")
    return Decimal(str(x))


def pretty_time(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


def filter_amm_pair(df):
    """Keep only rUSD<->XRP swaps (either direction)."""
    return df.filter(
        ((F.col("asset_in_currency") == RUSD_HEX) & (F.col("asset_out_currency") == XRP)) |
        ((F.col("asset_in_currency") == XRP) & (F.col("asset_out_currency") == RUSD_HEX))
    )


def filter_clob_pair(df):
    """Keep only rUSD/XRP book legs (either base/counter orientation)."""
    return df.filter(
        ((F.col("base_currency") == RUSD_HEX) & (F.col("counter_currency") == XRP)) |
        ((F.col("base_currency") == XRP) & (F.col("counter_currency") == RUSD_HEX))
    )


def main():
    args = parse_args()

    spark = (
        SparkSession.builder
        .appName("build_input_from_window")
        .getOrCreate()
    )

    # -----------------------------
    # Fetch range mode: read from source datasets, filter by ledger_index + pair, write outputs
    # -----------------------------
    if args.fetch_range:
        if args.ledger_start is None or args.ledger_end is None or args.out_dir is None:
            raise RuntimeError("--fetch_range requires --ledger_start, --ledger_end, and --out_dir")
        use_src_paths = (args.src_amm is not None and args.src_fees is not None and args.src_clob is not None)
        use_delta = (args.share is not None and args.schema is not None)
        if not use_src_paths and not use_delta:
            raise RuntimeError(
                "--fetch_range requires either (--src_amm --src_fees --src_clob) OR (--share --schema) for Delta Sharing"
            )
        if args.ledger_end < args.ledger_start:
            raise RuntimeError("ledger_end must be >= ledger_start")

        ensure_dir(args.out_dir)
        lo = int(args.ledger_start)
        hi = int(args.ledger_end)

        if use_src_paths:
            amm_src = spark.read.parquet(args.src_amm)
            fees_src = spark.read.parquet(args.src_fees)
            clob_src = spark.read.parquet(args.src_clob)
        else:
            prof = args.share_profile
            amm_url = f"{prof}#{args.share}.{args.schema}.{args.table_amm}"
            fees_url = f"{prof}#{args.share}.{args.schema}.{args.table_fees}"
            clob_url = f"{prof}#{args.share}.{args.schema}.{args.table_clob}"
            amm_src = spark.read.format("deltaSharing").load(amm_url)
            fees_src = spark.read.format("deltaSharing").load(fees_url)
            clob_src = spark.read.format("deltaSharing").load(clob_url)

        amm_f = filter_amm_pair(amm_src).filter((F.col("ledger_index") >= lo) & (F.col("ledger_index") <= hi)).cache()
        fees_f = fees_src.filter((F.col("ledger_index") >= lo) & (F.col("ledger_index") <= hi)).cache()
        clob_f = filter_clob_pair(clob_src).filter((F.col("ledger_index") >= lo) & (F.col("ledger_index") <= hi)).cache()

        amm_out = os.path.join(args.out_dir, f"amm_rusd_xrp_{lo}_{hi}")
        clob_out = os.path.join(args.out_dir, f"clob_rusd_xrp_{lo}_{hi}")
        fees_out = os.path.join(args.out_dir, f"amm_fees_{lo}_{hi}")

        amm_f.write.mode("overwrite").parquet(amm_out)
        clob_f.write.mode("overwrite").parquet(clob_out)
        fees_f.write.mode("overwrite").parquet(fees_out)

        src_mode = "src_paths" if use_src_paths else "delta_sharing"
        print(f"\n=== Fetched ledger_index range (rUSD-XRP) [{src_mode}] ===")
        print(f"range: {lo}..{hi}")
        print(f"AMM  rows: {amm_f.count()} -> {amm_out}")
        print(f"CLOB rows: {clob_f.count()} -> {clob_out}")
        print(f"FEES rows: {fees_f.count()} -> {fees_out}")

        spark.stop()
        return

    amm = spark.read.parquet(args.amm)
    clob = spark.read.parquet(args.clob)
    fees = spark.read.parquet(args.fees)

    # -----------------------------
    # Range export mode: filter by ledger_index and write parquet outputs
    # -----------------------------
    if args.export_range:
        if args.ledger_start is None or args.ledger_end is None or args.out_dir is None:
            raise RuntimeError("--export_range requires --ledger_start, --ledger_end, and --out_dir")
        if args.ledger_end < args.ledger_start:
            raise RuntimeError("ledger_end must be >= ledger_start")

        ensure_dir(args.out_dir)

        lo = int(args.ledger_start)
        hi = int(args.ledger_end)

        amm_f = amm.filter((F.col("ledger_index") >= lo) & (F.col("ledger_index") <= hi))
        clob_f = clob.filter((F.col("ledger_index") >= lo) & (F.col("ledger_index") <= hi))
        fees_f = fees.filter((F.col("ledger_index") >= lo) & (F.col("ledger_index") <= hi))

        amm_out = os.path.join(args.out_dir, f"amm_rusd_xrp_{lo}_{hi}")
        clob_out = os.path.join(args.out_dir, f"clob_rusd_xrp_{lo}_{hi}")
        fees_out = os.path.join(args.out_dir, f"amm_fees_{lo}_{hi}")

        amm_f.write.mode("overwrite").parquet(amm_out)
        clob_f.write.mode("overwrite").parquet(clob_out)
        fees_f.write.mode("overwrite").parquet(fees_out)

        print("\n=== Exported ledger_index range ===")
        print(f"range: {lo}..{hi}")
        print(f"AMM  rows: {amm_f.count()} -> {amm_out}")
        print(f"CLOB rows: {clob_f.count()} -> {clob_out}")
        print(f"FEES rows: {fees_f.count()} -> {fees_out}")

        spark.stop()
        return

    # -----------------------------
    # 1) Duplicate tx_hash set (hybrid)
    # -----------------------------
    amm_tx = amm.select(F.col("transaction_hash").alias("tx_hash")).distinct()
    clob_tx = clob.select("tx_hash").distinct()
    dup_tx = amm_tx.join(clob_tx, on="tx_hash", how="inner").cache()
    dup_cnt = dup_tx.count()

    # -----------------------------
    # 2) Pure AMM-only swaps
    # -----------------------------
    amm_pure = (
        amm.join(dup_tx, amm.transaction_hash == dup_tx.tx_hash, how="left_anti")
           .cache()
    )
    amm_pure_cnt = amm_pure.count()

    # -----------------------------
    # 3) Anchor selection
    #    Avoid edges: take middle 60% time-ordered swaps then random pick.
    # -----------------------------
    w = Window.orderBy(F.col("close_time_datetime").asc())
    amm_ranked = amm_pure.withColumn("rn", F.row_number().over(w))
    n = amm_pure_cnt

    low = int(n * 0.2) + 1
    high = int(n * 0.8)

    candidates = amm_ranked.filter((F.col("rn") >= low) & (F.col("rn") <= high)).cache()
    cand_cnt = candidates.count()

    random.seed(args.seed)
    pick_rn = random.randint(low, high)

    anchor = candidates.filter(F.col("rn") == pick_rn).collect()
    if not anchor:
        raise RuntimeError("No anchor selected; check window/candidate filtering.")
    anchor = anchor[0]

    # -----------------------------
    # 3.5) Ex-ante network fee approximation (drops)
    #    Use the immediately previous pure AMM-only swap's transaction_fee.
    #    This is forward-looking-safe and does NOT use the anchor's realised fee.
    # -----------------------------
    prev_row = (
        amm_pure
        .filter(F.col("close_time_datetime") < anchor["close_time_datetime"])
        .orderBy(F.col("close_time_datetime").desc())
        .select("transaction_fee")
        .limit(1)
        .collect()
    )

    if prev_row:
        net_fee_approx_drops = Decimal(str(prev_row[0]["transaction_fee"] or 0))
        net_fee_approx_source = "prev_swap_fee"
    else:
        net_fee_approx_drops = Decimal("0")
        net_fee_approx_source = "no_prev_swap"

    net_fee_approx_xrp = net_fee_approx_drops / Decimal("1000000")

    # -----------------------------
    # 4) Determine in/out currencies for anchor
    # -----------------------------
    in_cur = anchor["asset_in_currency"]
    out_cur = anchor["asset_out_currency"]

    # -----------------------------
    # 5) Map reserves to routing convention: x=IN, y=OUT
    #    Also map post-trade reserves to compute real post spot price.
    # -----------------------------
    side1_cur = anchor["amm_asset_currency"]
    side2_cur = anchor["amm_asset2_currency"]

    side1_bal_before = to_decimal(anchor["amm_asset_balance_before"])
    side2_bal_before = to_decimal(anchor["amm_asset2_balance_before"])
    side1_bal_after = to_decimal(anchor["amm_asset_balance_after"])
    side2_bal_after = to_decimal(anchor["amm_asset2_balance_after"])

    if in_cur == side1_cur and out_cur == side2_cur:
        x_currency, y_currency = side1_cur, side2_cur
        x_reserve, y_reserve = side1_bal_before, side2_bal_before
        x_after, y_after = side1_bal_after, side2_bal_after
    elif in_cur == side2_cur and out_cur == side1_cur:
        x_currency, y_currency = side2_cur, side1_cur
        x_reserve, y_reserve = side2_bal_before, side1_bal_before
        x_after, y_after = side2_bal_after, side1_bal_after
    else:
        raise ValueError(
            f"Anchor trade currencies do not match AMM pool sides: "
            f"in={in_cur}, out={out_cur}, side1={side1_cur}, side2={side2_cur}"
        )

    real_post_spot_price_inout = (x_after / y_after) if y_after != 0 else None

    # -----------------------------
    # 6) Match fee for anchor from fees window
    #    Default = trading_fee (ex-ante)
    #    Optional = discounted_fee (ex-post) if --use_discounted_fee
    # -----------------------------
    fee_row = (
        fees.filter(
            (F.col("transaction_hash") == anchor["transaction_hash"]) &
            (F.col("amm_account") == anchor["amm_account"])
        )
        .select("trading_fee", "discounted_fee")
        .limit(1)
        .collect()
    )

    fee_fallback = Decimal("0.003")  # last resort
    trading_fee = None
    discounted_fee = None
    if fee_row:
        trading_fee = fee_row[0]["trading_fee"]
        discounted_fee = fee_row[0]["discounted_fee"]

    if args.use_discounted_fee and discounted_fee is not None:
        fee_dec = to_decimal(discounted_fee)
        fee_src = "discounted_fee (ex-post)"
    elif trading_fee is not None:
        fee_dec = to_decimal(trading_fee)
        fee_src = "trading_fee (ex-ante)"
    elif discounted_fee is not None:
        # discounted exists but trading missing (rare); still take discounted
        fee_dec = to_decimal(discounted_fee)
        fee_src = "discounted_fee (fallback)"
    else:
        fee_dec = fee_fallback
        fee_src = "fallback_const"

    # -----------------------------
    # 7) Build model input dict
    # -----------------------------
    target_out = to_decimal(anchor["asset_out_value"])
    real_in_val = to_decimal(anchor["asset_in_value"])
    real_out_val = to_decimal(anchor["asset_out_value"])
    avg_price_real = (real_in_val / real_out_val) if real_out_val != 0 else None

    model_input = {
        "amm": {
            "x_reserve": str(x_reserve),
            "y_reserve": str(y_reserve),
            "x_currency": x_currency,
            "y_currency": y_currency,
            "fee": str(fee_dec),
            "fee_source": fee_src,
        },
        "trade": {
            "target_out": str(target_out),
            "out_currency": out_cur,
            "in_currency": in_cur,
        },
        "anchor_meta": {
            "T": pretty_time(anchor["close_time_datetime"]),
            "ledger_index": int(anchor["ledger_index"]),
            "transaction_index": int(anchor["transaction_index"]),
            "tx_hash": anchor["transaction_hash"],
            "amm_account": anchor["amm_account"],
        },
        "real_world": {
            "asset_in_value": str(real_in_val),
            "asset_out_value": str(real_out_val),
            "avg_price_real": str(avg_price_real) if avg_price_real is not None else None,
            "post_x_reserve": str(x_after),
            "post_y_reserve": str(y_after),
            "post_spot_price_inout": str(real_post_spot_price_inout) if real_post_spot_price_inout is not None else None,
        },
        "window_meta": {
            "dup_tx_count": int(dup_cnt),
            "pure_amm_count": int(amm_pure_cnt),
            "candidate_count": int(cand_cnt),
            "seed": int(args.seed),
            "use_discounted_fee": bool(args.use_discounted_fee),
            "net_fee_approx_drops": str(net_fee_approx_drops),
            "net_fee_approx_xrp": str(net_fee_approx_xrp),
            "net_fee_approx_source": net_fee_approx_source,
        }
    }

    # -----------------------------
    # 8) Console summary (concise)
    # -----------------------------
    print("\n=== Window summary ===")
    print(f"AMM rows total              : {amm.count()}")
    print(f"CLOB rows total             : {clob.count()}")
    print(f"Duplicate tx_hash (hybrid)  : {dup_cnt}")
    print(f"Pure AMM-only AMM rows      : {amm_pure_cnt}")
    print(f"Feasible pure candidates    : {cand_cnt}")

    print("\n=== Selected anchor (pure AMM-only) ===")
    print(f"T (pre-trade time): {pretty_time(anchor['close_time_datetime'])}")
    print(f"tx_hash           : {anchor['transaction_hash']}")
    print(f"asset_in          : {real_in_val} {in_cur}")
    print(f"asset_out         : {real_out_val} {out_cur}")

    print("\n=== AMM snapshot at T (mapped to x=IN,y=OUT) ===")
    print(f"x_reserve ({x_currency}) = {x_reserve}")
    print(f"y_reserve ({y_currency}) = {y_reserve}")
    print(f"fee = {fee_dec}  (source: {fee_src})")

    print("\n=== Network fee approximation (ex-ante) ===")
    print(f"net_fee_approx   : {net_fee_approx_drops} drops = {net_fee_approx_xrp} XRP "
          f"(source: {net_fee_approx_source})")

    print("\n=== Model input (AMM-only) ===")
    print(json.dumps(model_input, indent=2))

    # -----------------------------
    # 9) Write JSON
    # -----------------------------
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(model_input, f, ensure_ascii=False, indent=2)
    print(f"\nWrote model input to: {args.out_json}")

    spark.stop()


if __name__ == "__main__":
    main()
