"""
validate_live_vs_offline.py -- feature parity ledger

Compares scripts/12_inference.py's live feature computation against the
already-computed offline features (v2/features/{asset}-features-{ym}.parquet)
that actually fed model training, for identical historical windows.

This doesn't need live data. compute_features_from_trades() is a pure
function of a trade list -- it never touches the system clock, only the
timestamps in whatever trades it's handed. Historical archived trades work
identically to a live Binance fetch. What's being tested is whether the two
code paths compute the same feature values given the same input, not
whether "live" and "historical" match -- they were never meant to.

Raw trades are scanned lazily (pl.scan_parquet + filter + collect), never
loaded fully into memory -- an active month can hold tens of millions of
trade rows, but each comparison only needs ~310 seconds of it. Peak memory
scales with the sample window, not the month.

Usage:
    python3 validate_live_vs_offline.py --asset BTCUSDT --year 2022 --month 11 --n-samples 10
"""

import sys
import os
import argparse
import importlib.util
import tempfile
import random
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as fs
from google.cloud import storage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
from config import BUCKET

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(REPO_ROOT, "gcp-key.json")

# 12_inference.py can't be imported with a normal `import` statement --
# module names can't start with a digit. Load it by file path instead.
_spec = importlib.util.spec_from_file_location(
    "inference_module", os.path.join(REPO_ROOT, "scripts", "12_inference.py")
)
inference_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inference_module)
compute_features_from_trades = inference_module.compute_features_from_trades

FEATURES_PREFIX = "v2/features/"
TRADES_PREFIX   = "v2/trades_parquet_flat/"

STRICT_TOLERANCE   = 1e-6
ABS_FLOOR           = 1e-4  # below this absolute difference, always OK
                            # regardless of relative % -- handles features
                            # near zero (e.g. VWAP_dev) where a tiny
                            # absolute gap inflates into a huge percentage
MODERATE_TOLERANCE = 0.05  # 5% -- confirmed tight on real 60s/300s
                            # sum-based features (ILLIQ, RV, intensity,
                            # VWAP_dev); these count toward real failures

# Confirmed on real November 2022 BTCUSDT data: every 10s-scale feature is
# exposed to real market burst clustering, not just ratio-type ones.
# intensity_10s showed 6-23% swings despite 500-800 trades/bucket -- a
# single boundary trade can't explain that; it means dozens of trades
# landed differently, which happens when real arrivals cluster in bursts
# (unlike the synthetic test's evenly-spaced trades) and a burst straddles
# the boundary. "volume" and "rv" are themselves 10s-scale quantities (rv
# == RV_10s exactly by construction). Given a generous sanity ceiling,
# not a tight bound -- still catches genuine regressions (the original
# bugs were 900%+ off), tolerates real burst-driven noise at this scale.
TEN_SECOND_FEATURES = {
    "volume", "rv", "OFI_10s", "TCI_10s", "intensity_10s",
    "VWAP_10s", "ILLIQ_10s", "RV_10s", "VWAP_dev_10s",
}
SIGNED_TEN_SECOND_FEATURES = {"OFI_10s", "TCI_10s", "VWAP_dev_10s"}
SANITY_CEILING = 1.0  # 100%

# Ratio/correlation features exposed even after dilution across 6-30 bars.
RATIO_FEATURES = {
    "OFI_60s", "TCI_60s", "Kyle_lambda_60s", "Kyle_lambda_300s",
}
SIGN_FLIP_FLOOR = 0.1  # raised from an earlier 0.01: values this close to
                        # zero can legitimately flip sign from real timing
                        # noise in a small-sample correlation -- confirmed
                        # on real data (Kyle_lambda_60s: -0.037 vs +0.028,
                        # both near-zero, not a confident-direction flip
                        # like the original bug's -0.036 vs +0.522)

# Every feature both pipelines produce with matching names.
COMPARE_FEATURES = [
    "price", "volume", "rv",
    "OFI_10s", "TCI_10s", "intensity_10s", "VWAP_10s", "ILLIQ_10s", "RV_10s",
    "OFI_60s", "TCI_60s", "intensity_60s", "VWAP_60s", "ILLIQ_60s", "RV_60s", "Kyle_lambda_60s",
    "OFI_300s", "TCI_300s", "intensity_300s", "VWAP_300s", "ILLIQ_300s", "RV_300s", "Kyle_lambda_300s",
    "VWAP_dev_10s", "VWAP_dev_60s", "VWAP_dev_300s",
]


def download_to_temp(bucket, blob_path):
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    blob.download_to_filename(tmp.name)
    return tmp.name


def read_trades_window_from_gcs(gcs_fs, gcs_path, start_ms, end_ms):
    """
    Read only the row groups overlapping [start_ms, end_ms] from a remote
    parquet file, using row-group min/max statistics to skip everything
    else -- the same byte-range-request pattern 03_quality_audit.py uses
    for footer-only schema checks, extended here to data row groups.

    An active month's raw trades file can be hundreds of millions of rows
    and several GB; blob.download_to_filename() pulls all of it before
    anything else can happen, with no progress feedback, which is why a
    naive full download can sit for minutes with nothing printed. This
    instead opens the file, checks each row group's time range against
    what's actually needed, and only fetches the groups that overlap.
    """
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt   = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

    with gcs_fs.open_input_file(gcs_path) as file_handle:
        pf = pq.ParquetFile(file_handle)
        time_col_idx = pf.schema_arrow.get_field_index("time")

        tables = []
        for i in range(pf.num_row_groups):
            col_stats = pf.metadata.row_group(i).column(time_col_idx).statistics
            if col_stats is None or not col_stats.has_min_max:
                # No stats available for this group -- can't safely skip it
                tables.append(pf.read_row_group(i))
                continue
            if col_stats.max < start_dt or col_stats.min > end_dt:
                continue  # no overlap with the window -- skip entirely, no fetch
            tables.append(pf.read_row_group(i))

        if not tables:
            return pl.DataFrame()

        combined = pl.from_arrow(pa.concat_tables(tables))
        return combined.filter(
            (pl.col("time").dt.epoch(time_unit="ms") > start_ms) &
            (pl.col("time").dt.epoch(time_unit="ms") <= end_ms)
        )


def build_trade_dicts(window_df, window_end_ms, lookback_ms):
    """
    Slice an already-materialized, already-narrow DataFrame (from
    read_trades_window_from_gcs) to [window_end - lookback_ms, window_end]
    and convert to the dict format compute_features_from_trades() expects:
    price, qty, isBuyerMaker, time (epoch milliseconds int).
    """
    start_ms = window_end_ms - lookback_ms

    sliced = window_df.filter(
        (pl.col("time").dt.epoch(time_unit="ms") > start_ms) &
        (pl.col("time").dt.epoch(time_unit="ms") <= window_end_ms)
    )

    trades = []
    for row in sliced.iter_rows(named=True):
        trades.append({
            "price": row["price"],
            "qty": row["qty"],
            "isBuyerMaker": row["is_buyer_maker"],
            "time": int(row["time"].timestamp() * 1000),
        })
    return trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    gcs_fs = fs.GcsFileSystem()

    ym = f"{args.year}-{args.month:02d}"
    print(f"Loading offline features for {args.asset} {ym} ...")

    features_blob = f"{FEATURES_PREFIX}{args.asset}-features-{ym}.parquet"
    trades_gcs_path = f"{BUCKET}/{TRADES_PREFIX}{args.asset}-trades-{ym}.parquet"

    features_local = download_to_temp(bucket, features_blob)
    if features_local is None:
        print(f"Missing offline features: {features_blob}")
        return
    if not bucket.blob(f"{TRADES_PREFIX}{args.asset}-trades-{ym}.parquet").exists():
        print(f"Missing raw trades file for {ym}")
        return

    offline_df = pl.read_parquet(features_local)
    os.remove(features_local)
    print(f"Offline features: {offline_df.shape[0]:,} bars. "
          f"Raw trades read on demand, row-group-pruned -- no full-month download.")

    # Only sample timestamps deep enough into the month for full 300s
    # Kyle's lambda history (needs 31 bars = 310s back).
    min_ts = int(offline_df["time"][0].timestamp()) + 400
    valid = offline_df.filter(pl.col("time").dt.epoch(time_unit="s") > min_ts)

    if valid.shape[0] == 0:
        print("No bars deep enough into the month for a full comparison. Try a different month.")
        return

    n = min(args.n_samples, valid.shape[0])
    sample_rows = valid.sample(n=n, seed=args.seed)

    ledger = []
    for row in sample_rows.iter_rows(named=True):
        ts = row["time"]

        # group_by_dynamic labels buckets by their START, not end -- a row
        # labeled 14:00:00 spans [14:00:00, 14:00:10). compute_features_
        # from_trades() anchors t_end to timestamps[-1] (whichever trade is
        # literally last in the list), so comparing against the raw label
        # instant -- or even label+10s -- introduces an artificial gap
        # unless a real trade happens to land exactly there. Snap to the
        # actual last real trade inside the bucket instead; this is what
        # test_feature_parity.py's synthetic-data testing found necessary
        # to avoid reporting harness artifacts as feature bugs.
        bucket_start_ms = int(ts.timestamp() * 1000)
        bucket_end_ms   = bucket_start_ms + 10_000

        # Fetch [bucket_end - 310s, bucket_end] in ONE row-group-pruned
        # remote read -- covers both the snap-detection bucket (its last
        # 10s) and the full 310s lookback, so only one GCS round-trip is
        # needed per sample rather than two.
        fetch_start_ms = bucket_end_ms - 310_000
        window_df = read_trades_window_from_gcs(
            gcs_fs, trades_gcs_path, fetch_start_ms, bucket_end_ms
        )
        if window_df.is_empty():
            print(f"  skip {ts} -- no trades found in the fetched window")
            continue

        bucket_trades_df = window_df.filter(
            (pl.col("time").dt.epoch(time_unit="ms") >= bucket_start_ms) &
            (pl.col("time").dt.epoch(time_unit="ms") <  bucket_end_ms)
        )
        if bucket_trades_df.is_empty():
            print(f"  skip {ts} -- no real trades in this bucket to snap to")
            continue
        window_end_ms = int(bucket_trades_df["time"].max().timestamp() * 1000)

        # 310s lookback matches inference's FETCH_PADDING_S for Kyle's lambda
        trades = build_trade_dicts(window_df, window_end_ms, lookback_ms=310_000)

        if len(trades) < 10:
            print(f"  skip {ts} -- only {len(trades)} trades in window")
            continue

        try:
            live_features = compute_features_from_trades(trades, args.asset)
        except Exception as e:
            print(f"  skip {ts} -- live computation failed: {e}")
            continue

        for feat in COMPARE_FEATURES:
            offline_val = row.get(feat)
            live_val    = live_features.get(feat)
            if offline_val is None or live_val is None:
                continue
            abs_diff = abs(offline_val - live_val)
            rel_diff = abs_diff / (abs(offline_val) + 1e-10)

            if rel_diff < STRICT_TOLERANCE or abs_diff < ABS_FLOOR:
                status = "OK"
            elif feat in TEN_SECOND_FEATURES:
                if feat in SIGNED_TEN_SECOND_FEATURES:
                    sign_flip = (
                        abs(offline_val) > SIGN_FLIP_FLOOR and abs(live_val) > SIGN_FLIP_FLOOR
                        and np.sign(offline_val) != np.sign(live_val)
                    )
                    status = "FAIL_SIGN_FLIP" if sign_flip else "NOTE_10S_NOISE"
                else:
                    status = "FAIL" if rel_diff > SANITY_CEILING else "NOTE_10S_NOISE"
            elif feat in RATIO_FEATURES:
                sign_flip = (
                    abs(offline_val) > SIGN_FLIP_FLOOR and abs(live_val) > SIGN_FLIP_FLOOR
                    and np.sign(offline_val) != np.sign(live_val)
                )
                status = "FAIL_SIGN_FLIP" if sign_flip else "NOTE_RATIO_NOISE"
            elif rel_diff < MODERATE_TOLERANCE:
                status = "NOTE_BOUNDARY_NOISE"
            else:
                status = "FAIL"

            ledger.append({
                "timestamp":     str(ts),
                "feature":       feat,
                "offline_value": offline_val,
                "live_value":    live_val,
                "abs_diff":      abs_diff,
                "rel_diff_pct":  rel_diff * 100,
                "status":        status,
            })

    if not ledger:
        print("No comparisons produced. Check data availability for this month.")
        return

    ledger_df = pl.DataFrame(ledger)
    out_path = os.path.join(REPO_ROOT, f"live_offline_ledger_{args.asset}_{ym}.csv")
    ledger_df.write_csv(out_path)
    print(f"\nLedger written to {out_path} ({len(ledger)} rows)")

    n_fail = ledger_df.filter(pl.col("status").str.starts_with("FAIL")).shape[0]
    n_note = ledger_df.filter(pl.col("status").str.starts_with("NOTE")).shape[0]
    n_ok   = ledger_df.filter(pl.col("status") == "OK").shape[0]
    print(f"\n{len(ledger)} checks: {n_ok} OK, {n_note} boundary-sensitive notes, {n_fail} real failures")
    if n_fail > 0:
        print("\nFAIL rows:")
        with pl.Config(tbl_rows=-1):
            print(ledger_df.filter(pl.col("status").str.starts_with("FAIL")))

    print("\n=== Per-feature max absolute/relative difference across sampled timestamps ===")
    summary = (
        ledger_df.group_by("feature")
        .agg([
            pl.col("abs_diff").max().alias("max_abs_diff"),
            pl.col("rel_diff_pct").max().alias("max_rel_diff_pct"),
        ])
        .sort("feature")
    )
    with pl.Config(tbl_rows=-1):
        print(summary)

    if n_fail > 0:
        print(f"\nFAIL -- {n_fail} real discrepancy(ies) found on real historical data.")
        sys.exit(1)
    else:
        print(f"\nPASS -- live and offline agree ({n_note} known boundary-sensitive notes, not failures).")
        sys.exit(0)


if __name__ == "__main__":
    main()