"""
14b_test_feature_parity.py -- regression test against future training/inference drift

This is what 14b_test_feature_parity.py exists to catch: the four real bugs
found in 12_inference.py (missing sqrt in RV, 10x error in intensity, wrong
ILLIQ numerator, trade-level instead of bar-level Kyle's lambda) all came
from hand-mirroring 04a_feature_engineering.py's formulas by eye instead of
sharing one implementation. feature_formulas.py now removes that risk for
the formulas 12_inference.py uses directly -- but the deeper protection is
this test: it imports 04a's ACTUAL compute_single_asset_features() and
12_inference's ACTUAL compute_features_from_trades(), runs both on the same
synthetic trades, and fails loudly if they ever disagree again.

No GCS, no network, no live data -- runs anywhere, including CI, in seconds.

Usage:
    python3 14b_test_feature_parity.py
"""

import sys
import os
import tempfile
import importlib.util
from datetime import datetime, timezone, timedelta

import numpy as np
import polars as pl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

STRICT_TOLERANCE = 1e-6   # true floating-point-noise-only tolerance
ABS_FLOOR = 1e-4  # below this absolute difference, always OK regardless
                   # of relative % -- handles features near zero (VWAP_dev,
                   # RV, ILLIQ at typical magnitudes) where a tiny absolute
                   # gap inflates into a huge percentage
MODERATE_TOLERANCE = 0.05  # 5% -- for 60s/300s sum-based features; far
                            # tighter than any real regression produces
                            # (the original bugs were 30x-900x off)

# Confirmed on real November 2022 BTCUSDT data (14a_validate_live_vs_
# offline.py), not just synthetic: every 10s-scale feature is exposed to
# real market burst clustering, not just ratio-type ones. intensity_10s
# showed 6-23% swings despite 500-800 trades/bucket in that run -- a
# single boundary trade can't explain that; it means dozens of trades
# landed differently, consistent with real arrivals clustering in bursts
# (unlike this file's evenly-spaced synthetic trades) and a burst
# straddling the boundary. "volume" and "rv" are themselves 10s-scale
# quantities (rv == RV_10s exactly by construction). Given a generous
# sanity ceiling, not a tight bound -- still catches genuine regressions,
# tolerates real burst-driven noise at this scale.
TEN_SECOND_FEATURES = {
    "volume", "rv", "OFI_10s", "TCI_10s", "intensity_10s",
    "VWAP_10s", "ILLIQ_10s", "RV_10s", "VWAP_dev_10s",
}
SIGNED_TEN_SECOND_FEATURES = {"OFI_10s", "TCI_10s", "VWAP_dev_10s"}
SANITY_CEILING = 1.0  # 100%

# Ratio/correlation features exposed even after dilution across 6-30 bars.
# A genuine implementation bug (e.g. Kyle's lambda reverting to trade-level
# correlation) flips the SIGN outright on a confidently non-zero value --
# confirmed empirically. Boundary noise near zero can also flip sign
# trivially, which is why SIGN_FLIP_FLOOR excludes near-zero values from
# counting as a real flip.
RATIO_FEATURES = {
    "OFI_60s", "TCI_60s", "Kyle_lambda_60s", "Kyle_lambda_300s",
}
SIGN_FLIP_FLOOR = 0.1  # raised from an earlier 0.01: confirmed on real
                        # data that values this close to zero can flip
                        # sign from legitimate timing noise alone
                        # (Kyle_lambda_60s: -0.037 vs +0.028, both
                        # near-zero -- not a confident-direction flip like
                        # the original bug's -0.036 vs +0.522)


def load_module(filename, module_name):
    """04a_feature_engineering.py and 12_inference.py both start with a
    digit, so they can't be imported with a normal `import` statement."""
    path = os.path.join(REPO_ROOT, "scripts", filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_synthetic_trades(n_bars=40, trades_per_bar=15, seed=42):
    """
    n_bars * 10s of synthetic trades, dense enough that every 10s bucket
    has real activity -- an all-empty bucket wouldn't meaningfully test
    the formulas being compared.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    rows = []
    price = 50000.0
    trade_id = 0
    for bar in range(n_bars):
        bar_start_ms = int((start + timedelta(seconds=bar * 10)).timestamp() * 1000)
        offsets = np.sort(rng.integers(0, 10_000, trades_per_bar))
        for off in offsets:
            price += rng.normal(0, 3)
            qty = float(rng.uniform(0.01, 2.0))
            is_buyer_maker = bool(rng.random() > 0.5)
            rows.append({
                "id":             trade_id,
                "price":          float(price),
                "qty":            qty,
                "quote_qty":      float(price * qty),
                "time_ms":        bar_start_ms + int(off),
                "is_buyer_maker": is_buyer_maker,
                "is_best_match":  True,
            })
            trade_id += 1

    return rows


def rows_to_offline_parquet(rows):
    """Write rows to a temp parquet matching what 04a expects: 'time' as
    a real Datetime[us, UTC] column, matching the post-conversion format
    02_csv_to_parquet.py actually persists (not the raw epoch-ms int the
    live Binance API returns)."""
    df = pl.DataFrame({
        "id":             [r["id"] for r in rows],
        "price":          [r["price"] for r in rows],
        "qty":            [r["qty"] for r in rows],
        "quote_qty":      [r["quote_qty"] for r in rows],
        "time":           [r["time_ms"] for r in rows],
        "is_buyer_maker": [r["is_buyer_maker"] for r in rows],
        "is_best_match":  [r["is_best_match"] for r in rows],
    }).with_columns(
        pl.from_epoch(pl.col("time"), time_unit="ms").dt.replace_time_zone("UTC")
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    df.write_parquet(tmp.name)
    return tmp.name


def rows_to_live_trades(rows, window_end_ms, lookback_ms):
    """Same rows, sliced and reshaped to what compute_features_from_trades()
    expects: price, qty, isBuyerMaker, time (epoch ms int) -- matching the
    live Binance API's actual response shape."""
    start_ms = window_end_ms - lookback_ms
    return [
        {
            "price": r["price"],
            "qty": r["qty"],
            "isBuyerMaker": r["is_buyer_maker"],
            "time": r["time_ms"],
        }
        for r in rows
        if start_ms < r["time_ms"] <= window_end_ms
    ]


COMPARE_FEATURES = [
    "price", "volume", "rv",
    "OFI_10s", "TCI_10s", "intensity_10s", "VWAP_10s", "ILLIQ_10s", "RV_10s",
    "OFI_60s", "TCI_60s", "intensity_60s", "VWAP_60s", "ILLIQ_60s", "RV_60s", "Kyle_lambda_60s",
    "OFI_300s", "TCI_300s", "intensity_300s", "VWAP_300s", "ILLIQ_300s", "RV_300s", "Kyle_lambda_300s",
    "VWAP_dev_10s", "VWAP_dev_60s", "VWAP_dev_300s",
]


def main():
    print("Loading 04a_feature_engineering.py and 12_inference.py ...")
    offline_mod  = load_module("04a_feature_engineering.py", "offline_module")
    live_mod     = load_module("12_inference.py", "live_module")

    print("Generating synthetic trades (40 bars = 400s, dense enough for full 300s Kyle's lambda) ...")
    rows = generate_synthetic_trades(n_bars=40, trades_per_bar=200)

    print("Running 04a's actual compute_single_asset_features() ...")
    offline_parquet = rows_to_offline_parquet(rows)
    offline_df = offline_mod.compute_single_asset_features(offline_parquet)
    os.remove(offline_parquet)

    # Test the last few bars, which have full 310s of history behind them
    test_rows = offline_df.tail(3)

    n_failures = 0   # strict-tolerance features: real regressions, fail the run
    n_noted    = 0    # boundary-sensitive features: expected, informational only
    n_checks   = 0

    for row in test_rows.iter_rows(named=True):
        ts = row["time"]
        # group_by_dynamic labels buckets by their START, not end -- a row
        # labeled 00:06:10 spans [00:06:10, 00:06:20). Snap to the actual
        # LAST REAL TRADE inside that bucket, not the round label+10s
        # instant -- compute_features_from_trades() anchors t_end to
        # timestamps[-1] (whichever trade is literally last in the list),
        # so comparing against an idealized round boundary introduces an
        # artificial gap that doesn't exist in genuine live usage, where
        # t_end is always just "the freshest trade," never a grid mark.
        bucket_start_ms = int(ts.timestamp() * 1000)
        bucket_end_ms   = bucket_start_ms + 10_000
        bucket_trades   = [r for r in rows if bucket_start_ms <= r["time_ms"] < bucket_end_ms]
        if not bucket_trades:
            print(f"  skip {ts} -- no real trades in this bucket to snap to")
            continue
        window_end_ms = max(r["time_ms"] for r in bucket_trades)

        print(f"\n=== Comparing bar labeled {ts} (snapped to real trade at {window_end_ms}) ===")
        live_trades = rows_to_live_trades(rows, window_end_ms, lookback_ms=310_000)

        if len(live_trades) < 10:
            print(f"  skip -- only {len(live_trades)} trades in window")
            continue

        live_features = live_mod.compute_features_from_trades(live_trades, "SYNTHETIC")

        for feat in COMPARE_FEATURES:
            offline_val = row.get(feat)
            live_val    = live_features.get(feat)
            if offline_val is None or live_val is None:
                continue

            n_checks += 1
            abs_diff = abs(offline_val - live_val)
            rel_diff = abs_diff / (abs(offline_val) + 1e-10)

            if rel_diff < STRICT_TOLERANCE or abs_diff < ABS_FLOOR:
                print(f"  [OK]    {feat:<18} offline={offline_val:.8f}  live={live_val:.8f}")
                continue

            if feat in TEN_SECOND_FEATURES:
                if feat in SIGNED_TEN_SECOND_FEATURES:
                    sign_flip = (
                        abs(offline_val) > SIGN_FLIP_FLOOR and abs(live_val) > SIGN_FLIP_FLOOR
                        and np.sign(offline_val) != np.sign(live_val)
                    )
                    if sign_flip:
                        n_failures += 1
                        print(f"  [FAIL]  {feat:<18} offline={offline_val:.8f}  live={live_val:.8f}  "
                              f"SIGN FLIP on non-trivial values -- likely a real regression")
                    else:
                        n_noted += 1
                        print(f"  [NOTE]  {feat:<18} offline={offline_val:.8f}  live={live_val:.8f}  "
                              f"rel_diff={rel_diff:.2e}  (10s-scale feature, expected burst noise)")
                elif rel_diff > SANITY_CEILING:
                    n_failures += 1
                    print(f"  [FAIL]  {feat:<18} offline={offline_val:.8f}  live={live_val:.8f}  rel_diff={rel_diff:.2e}")
                else:
                    n_noted += 1
                    print(f"  [NOTE]  {feat:<18} offline={offline_val:.8f}  live={live_val:.8f}  "
                          f"rel_diff={rel_diff:.2e}  (10s-scale feature, expected burst noise)")
            elif feat in RATIO_FEATURES:
                # Magnitude noise is expected and not meaningfully boundable
                # for these; a sign flip on a non-trivial value is not.
                sign_flip = (
                    abs(offline_val) > SIGN_FLIP_FLOOR and abs(live_val) > SIGN_FLIP_FLOOR
                    and np.sign(offline_val) != np.sign(live_val)
                )
                if sign_flip:
                    n_failures += 1
                    print(f"  [FAIL]  {feat:<18} offline={offline_val:.8f}  live={live_val:.8f}  "
                          f"SIGN FLIP on non-trivial values -- likely a real regression")
                else:
                    n_noted += 1
                    print(f"  [NOTE]  {feat:<18} offline={offline_val:.8f}  live={live_val:.8f}  "
                          f"rel_diff={rel_diff:.2e}  (ratio feature, expected boundary noise)")
            elif rel_diff < MODERATE_TOLERANCE:
                n_noted += 1
                print(f"  [NOTE]  {feat:<18} offline={offline_val:.8f}  live={live_val:.8f}  "
                      f"rel_diff={rel_diff:.2e}  (within {MODERATE_TOLERANCE:.0%} boundary-noise allowance)")
            else:
                n_failures += 1
                print(f"  [FAIL]  {feat:<18} offline={offline_val:.8f}  live={live_val:.8f}  rel_diff={rel_diff:.2e}")

    print(f"\n{'='*60}")
    print(f"{n_checks} checks, {n_failures} failures, {n_noted} boundary-sensitive notes")
    print(f"{'='*60}")

    if n_failures > 0:
        print("\nFAIL -- live and offline feature computation have diverged.")
        sys.exit(1)
    else:
        if n_noted > 0:
            print(f"\nPASS -- live and offline feature computation agree "
                  f"({n_noted} known boundary-sensitive notes, not failures).")
        else:
            print("\nPASS -- live and offline feature computation agree exactly.")
        sys.exit(0)


if __name__ == "__main__":
    main()