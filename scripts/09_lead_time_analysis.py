import sys
import os
import io
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from datetime import timedelta
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASSETS, BUCKET, WINDOWS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcp-key.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

# Each crisis event is mapped to the fold whose test window contains it.
#
# Fold-to-window mapping (from 06d_train_production.py):
#   test_window = fold  ->  fold N tests on WINDOWS[N]
#
#   Window 0: 2020-02 -> 2020-07  (COVID-19 crash -- ALWAYS training, excluded)
#   Window 1: 2020-11 -> 2021-05  (May 2021 crash  -- fold 1 test)
#   Window 2: 2021-11 -> 2022-05  (Terra-Luna      -- fold 2 test)
#   Window 3: 2022-11 -> 2023-04  (FTX bankruptcy  -- fold 3 test)
#   Window 4: 2024-07 -> 2024-12  (no crisis event -- fold 4 test)
#
# COVID-19 (March 2020) is excluded: Window 0 is training data in every fold
# and no out-of-sample predictions exist for that period.

CRISIS_EVENTS = [
    {
        "name":       "May 2021 Crash",
        "onset":      "2021-05-19 12:00:00",
        "fold":       1,
        "window_idx": 1,
    },
    {
        "name":       "Terra-Luna Collapse",
        "onset":      "2022-05-09 08:00:00",
        "fold":       2,
        "window_idx": 2,
    },
    {
        "name":       "FTX Bankruptcy",
        "onset":      "2022-11-08 14:00:00",
        "fold":       3,
        "window_idx": 3,
    },
]

# Best seed -- used for all lead-time calculations
BEST_SEED = 42

# Stress probability threshold for the first warning signal.
# Raised from 0.5 to 0.85 — we want the model screaming, not whispering.
PROB_THRESHOLD = 0.85

# Lookback window: scan this many hours before each HMM onset
LOOKBACK_HOURS = 4

# Continuity check: require the signal to hold for this many consecutive bars
# before declaring a warning. One bar spiking above threshold may be noise;
# two consecutive bars (10 minutes at 300s resolution) confirms a sustained
# liquidity drain rather than a transient spike.
MIN_CONSECUTIVE_BARS = 2

# GCS path prefixes (mirrors 06d_train_production.py)
RESULTS_PREFIX = f"v2/results_production/seed_{BEST_SEED}/"
LABELS_PREFIX  = "v2/labels/"

# Output paths
OUTPUT_CSV     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lead_time_results.csv")
OUTPUT_PARQUET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lead_time_results.parquet")


# =============================================================================
# TIME INDEX RECONSTRUCTION
# =============================================================================

def parse_window_months(window_idx):
    """
    Return a list of (year, month) tuples covering the given window.
    Mirrors parse_window_months() in 06d_train_production.py.
    """
    start_ym, end_ym = WINDOWS[window_idx]

    def ym_to_tuple(ym):
        year, month = ym.split("-")
        return int(year), int(month)

    start_year, start_month = ym_to_tuple(start_ym)
    end_year,   end_month   = ym_to_tuple(end_ym)

    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def reconstruct_time_index(bucket, window_idx):
    """
    Reconstruct the ordered time index for a given test window.
    """
    months = parse_window_months(window_idx)
    logger.info(
        f"  Reconstructing time index for window {window_idx} "
        f"({WINDOWS[window_idx][0]} -> {WINDOWS[window_idx][1]}) "
        f"-- {len(months)} months x {len(ASSETS)} assets"
    )

    frames = []
    for asset_id, asset in enumerate(ASSETS):
        asset_frames = []
        for year, month in months:
            path = f"{LABELS_PREFIX}{asset}-labels-{year}-{month:02d}.parquet"
            blob = bucket.blob(path)
            if not blob.exists():
                logger.warning(f"    Missing label file: {path}")
                continue
            buf = io.BytesIO()
            blob.download_to_file(buf)
            buf.seek(0)
            df = pl.read_parquet(buf).select("time")
            asset_frames.append(df)

        if not asset_frames:
            logger.warning(f"    No label files found for {asset} in window {window_idx}")
            continue

        asset_df = pl.concat(asset_frames).with_columns(
            pl.lit(asset_id).alias("asset_id")
        )
        frames.append(asset_df)
        logger.info(f"    {asset}: {len(asset_df):,} rows")

    if not frames:
        raise RuntimeError(
            f"No label files found for window {window_idx}. "
            f"Check GCS path: {LABELS_PREFIX}"
        )

    time_index = pl.concat(frames).sort("time")
    logger.info(f"  Time index reconstructed: {len(time_index):,} rows total")
    return time_index


# =============================================================================
# PREDICTIONS LOADING
# =============================================================================

def load_predictions(bucket, fold):
    """
    Load the pooled binary predictions parquet for a given fold.
    """
    path = f"{RESULTS_PREFIX}pooled/fold_{fold}/predictions_binary.parquet"
    logger.info(f"  Loading predictions from {path}")
    blob = bucket.blob(path)
    if not blob.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    df = pl.read_parquet(buf)
    logger.info(f"  Loaded {len(df):,} rows, columns: {df.columns}")
    return df


# =============================================================================
# LEAD-TIME COMPUTATION
# =============================================================================

def compute_lead_time(event, time_index, predictions):
    """
    Compute the predictive lead time for a single crisis event.
    Requires MIN_CONSECUTIVE_BARS consecutive bars above PROB_THRESHOLD.
    """
    name  = event["name"]
    onset = pl.Series([event["onset"]]).str.to_datetime(
        format="%Y-%m-%d %H:%M:%S", time_unit="us", time_zone="UTC"
    )[0]

    if len(time_index) != len(predictions):
        raise RuntimeError(
            f"[{name}] Row count mismatch: "
            f"time_index={len(time_index):,}, predictions={len(predictions):,}."
        )

    df = predictions.with_columns(time_index["time"])

    if df["time"].dtype != pl.Datetime("us", "UTC"):
        df = df.with_columns(
            pl.col("time").dt.replace_time_zone("UTC")
        )

    window_start = onset - timedelta(hours=LOOKBACK_HOURS)
    window_df    = df.filter(
        (pl.col("time") >= window_start) & (pl.col("time") <= onset)
    )

    if window_df.is_empty():
        logger.warning(f"  [{name}] No rows in lookback window")
        return {
            "event":              name,
            "fold":               event["fold"],
            "hmm_onset_utc":      str(onset)[:19],
            "first_warning_utc":  None,
            "lead_time_minutes":  None,
            "max_prob_in_window": None,
            "n_bars_in_window":   0,
            "status":             "NO_DATA",
        }

    prob_col = "xgb_prob_class1"
    max_prob = round(window_df[prob_col].max(), 4)

    # Continuity check: find first run of MIN_CONSECUTIVE_BARS above threshold
    sorted_df     = window_df.sort("time")
    probs         = sorted_df[prob_col].to_list()
    times         = sorted_df["time"].to_list()
    first_warning = None

    for i in range(len(probs) - MIN_CONSECUTIVE_BARS + 1):
        if all(probs[i + j] > PROB_THRESHOLD for j in range(MIN_CONSECUTIVE_BARS)):
            first_warning = times[i]
            break

    if first_warning is None:
        logger.info(
            f"  [{name}] No sustained warning signal "
            f"({MIN_CONSECUTIVE_BARS} consecutive bars > {PROB_THRESHOLD}) -- "
            f"max P(stress) in window: {max_prob:.4f}"
        )
        return {
            "event":              name,
            "fold":               event["fold"],
            "hmm_onset_utc":      str(onset)[:19],
            "first_warning_utc":  None,
            "lead_time_minutes":  None,
            "max_prob_in_window": max_prob,
            "n_bars_in_window":   len(window_df),
            "status":             "NO_SIGNAL",
        }

    lead_minutes = round(
        (onset - first_warning).total_seconds() / 60, 2
    )

    logger.info(
        f"  [{name}] First warning: {str(first_warning)[:19]}  "
        f"->  Lead time: {lead_minutes:.1f} min"
    )
    return {
        "event":              name,
        "fold":               event["fold"],
        "hmm_onset_utc":      str(onset)[:19],
        "first_warning_utc":  str(first_warning)[:19],
        "lead_time_minutes":  lead_minutes,
        "max_prob_in_window": max_prob,
        "n_bars_in_window":   len(window_df),
        "status":             "OK",
    }


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================

def print_lead_time_table(results):
    print("\n" + "=" * 80)
    print("PREDICTIVE LEAD-TIME ANALYSIS")
    print(f"Model: XGBoost Binary | Seed: {BEST_SEED} | Pooled")
    print(f"Threshold: P(stress) > {PROB_THRESHOLD} | Lookback: {LOOKBACK_HOURS}h pre-onset")
    print(f"Continuity: >= {MIN_CONSECUTIVE_BARS} consecutive bars above threshold")
    print("Note: COVID-19 (Mar 2020) excluded -- Window 0 is training data in all folds")
    print("=" * 80)
    print(
        f"{'Event':<25} {'Fold':<6} {'HMM Onset':<22} "
        f"{'First Warning':<22} {'Lead Time':<12} {'Status'}"
    )
    print("-" * 80)

    lead_times = []
    for r in results:
        lead_str    = f"{r['lead_time_minutes']:.1f} min" if r["lead_time_minutes"] is not None else "-"
        warning_str = r["first_warning_utc"] if r["first_warning_utc"] else "-"
        print(
            f"{r['event']:<25} {r['fold']:<6} {r['hmm_onset_utc']:<22} "
            f"{warning_str:<22} {lead_str:<12} {r['status']}"
        )
        if r["lead_time_minutes"] is not None:
            lead_times.append(r["lead_time_minutes"])

    print("-" * 80)

    if lead_times:
        print(f"\n  Events with signal : {len(lead_times)} / {len(results)}")
        print(f"  Mean lead time     : {np.mean(lead_times):.1f} min")
        print(f"  Median lead time   : {np.median(lead_times):.1f} min")
        print(f"  Min lead time      : {np.min(lead_times):.1f} min")
        print(f"  Max lead time      : {np.max(lead_times):.1f} min")
    else:
        print("\n  No warning signals detected across any events.")

    print("=" * 80)


def print_operational_interpretation(results):
    print("\n" + "=" * 80)
    print("OPERATIONAL INTERPRETATION")
    print("-" * 80)

    for r in results:
        if r["lead_time_minutes"] is None:
            print(f"  {r['event']:<25}  [{r['status']}]")
            continue

        lt = r["lead_time_minutes"]
        if lt < 5:
            note = "Tight -- automated circuit breakers only."
        elif lt < 15:
            note = "Sufficient for automated deleveraging and margin triggers."
        elif lt < 30:
            note = "Sufficient for automated protocols and human-in-the-loop decisions."
        else:
            note = "Ample -- supports full position review and coordinated desk response."

        print(f"  {r['event']:<25}  {lt:.1f} min -> {note}")

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    # --- Skip logic ---
    _marker = "v2/pipeline_markers/09_lead_time_analysis.done"
    if bucket.blob(_marker).exists():
        logger.info("09_lead_time_analysis -- already complete, skipping")
        logger.info(f"  To rerun: gsutil rm gs://{BUCKET}/{_marker}")
        return

    all_results = []

    for event in CRISIS_EVENTS:
        logger.info(f"Processing: {event['name']} (fold {event['fold']})")

        try:
            time_index = reconstruct_time_index(bucket, event["window_idx"])
        except RuntimeError as e:
            logger.error(str(e))
            continue

        try:
            predictions = load_predictions(bucket, event["fold"])
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        result = compute_lead_time(event, time_index, predictions)
        all_results.append(result)

    if not all_results:
        logger.error("No results produced -- check GCS paths and credentials")
        return

    print_lead_time_table(all_results)
    print_operational_interpretation(all_results)

    out_df = pl.DataFrame(all_results)
    out_df.write_csv(OUTPUT_CSV)
    out_df.write_parquet(OUTPUT_PARQUET)
    logger.info(f"Results saved to {OUTPUT_CSV}")
    logger.info(f"Results saved to {OUTPUT_PARQUET}")

    print(f"\n  Output files:")
    print(f"    {OUTPUT_CSV}")
    print(f"    {OUTPUT_PARQUET}")
    print()

    # --- Write done marker ---
    bucket.blob(_marker).upload_from_string(b"")
    logger.info(f"Done marker written: gs://{BUCKET}/{_marker}")


if __name__ == "__main__":
    main()
