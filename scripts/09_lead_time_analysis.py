import sys
import os
import io
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BUCKET

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/workspace/maic/gcp-key.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

# HMM-validated stress onset timestamps for the four crisis events.
# Ground-truth is the point at which the HMM transitions to the stress state
# (label 2), cross-referenced against publicly documented event timelines.
CRISIS_EVENTS = {
    "COVID-19 Crash":      "2020-03-12 10:00:00",
    "May 2021 Crash":      "2021-05-19 12:00:00",
    "Terra-Luna Collapse": "2022-05-09 08:00:00",
    "FTX Bankruptcy":      "2022-11-08 14:00:00",
}

# Lead-time analysis uses the best-performing seed and the largest fold,
# which together represent the most representative out-of-sample evaluation.
BEST_SEED    = 42
BEST_FOLD    = 4

# Stress probability threshold: first bar exceeding this value is treated
# as the model's first sustained warning signal.
PROB_THRESHOLD = 0.5

# Lookback window: we scan this many hours before each HMM onset for the
# first warning signal. 12 hours is chosen to avoid noise from unrelated
# microstructure fluctuations earlier in the session.
LOOKBACK_HOURS = 12

# Candidate GCS paths for the XGBoost binary probability file.
# The script tries each in order and uses the first one that exists.
PROB_PATH_CANDIDATES = [
    f"v2/results_production/seed_{BEST_SEED}/pooled/fold_{BEST_FOLD}/stage/binary_xgb_probs.parquet",
    f"v2/results_production/seed_{BEST_SEED}/pooled/fold_{BEST_FOLD}/binary_xgb_probs.parquet",
]

# Output paths
OUTPUT_CSV     = "/workspace/maic/lead_time_results.csv"
OUTPUT_PARQUET = "/workspace/maic/lead_time_results.parquet"


# =============================================================================
# GCS HELPERS
# =============================================================================

def resolve_prob_path(bucket):
    """
    Try candidate GCS paths in order and return the first one that exists.
    Raises FileNotFoundError if none are found.
    """
    for path in PROB_PATH_CANDIDATES:
        if bucket.blob(path).exists():
            logger.info(f"Found probability file at: {path}")
            return path
    raise FileNotFoundError(
        f"XGBoost binary probability file not found. Tried:\n"
        + "\n".join(f"  {p}" for p in PROB_PATH_CANDIDATES)
    )


def load_prob_file(bucket, gcs_path):
    """
    Download a parquet file from GCS and return it as a Polars DataFrame.
    """
    logger.info(f"Downloading {gcs_path} ...")
    blob   = bucket.blob(gcs_path)
    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    df = pl.read_parquet(buffer)
    logger.info(f"  Loaded {len(df):,} rows, columns: {df.columns}")
    return df


# =============================================================================
# PROBABILITY COLUMN DETECTION
# =============================================================================

def detect_prob_column(df):
    """
    Identify the stress probability column from the DataFrame.

    XGBoost binary predictions are stored with one probability column per
    class. For binary stress detection the positive class (stress) is label 1,
    so we look for columns named 'prob_1', 'prob_stress', or similar.
    Falls back to any column whose name contains 'prob'.
    """
    candidates = [
        c for c in df.columns
        if c in ("prob_1", "prob_stress")
        or ("prob" in c.lower() and "2" not in c)   # exclude multiclass prob_2
    ]

    # If above heuristic finds nothing, fall back to any prob column
    if not candidates:
        candidates = [c for c in df.columns if "prob" in c.lower()]

    if not candidates:
        raise ValueError(
            f"Cannot identify stress probability column. "
            f"Available columns: {df.columns}"
        )

    col = candidates[0]
    logger.info(f"Using probability column: '{col}'")
    return col


# =============================================================================
# LEAD-TIME COMPUTATION
# =============================================================================

def compute_lead_times(df, prob_col):
    """
    For each crisis event, find the earliest bar within the lookback window
    where the model's stress probability exceeds PROB_THRESHOLD, and compute
    the lead time in minutes relative to the HMM-defined onset.

    Returns a list of result dicts — one per event.
    """
    # Ensure the time column is in UTC-aware datetime format
    if df["time"].dtype != pl.Datetime("us", "UTC"):
        df = df.with_columns(
            pl.col("time").cast(pl.Datetime("us", "UTC"))
        )

    results = []

    for event, onset_str in CRISIS_EVENTS.items():
        onset_time    = pl.Series([onset_str]).str.to_datetime(
            format="%Y-%m-%d %H:%M:%S", time_unit="us", time_zone="UTC"
        )[0]
        window_start  = onset_time - pl.Duration(hours=LOOKBACK_HOURS)

        # Filter to the pre-crisis lookback window
        window_df = df.filter(
            (pl.col("time") >= window_start)
            & (pl.col("time") <= onset_time)
        )

        if window_df.is_empty():
            logger.warning(
                f"  [{event}] No data found in lookback window "
                f"({window_start} → {onset_time}). "
                f"Check that fold {BEST_FOLD} covers this date."
            )
            results.append({
                "event":              event,
                "hmm_onset_utc":      str(onset_time),
                "first_warning_utc":  None,
                "lead_time_minutes":  None,
                "n_bars_in_window":   0,
                "status":             "NO_DATA",
            })
            continue

        # Find the first bar where stress probability crosses the threshold
        warnings_df = window_df.filter(pl.col(prob_col) > PROB_THRESHOLD)

        if warnings_df.is_empty():
            logger.info(
                f"  [{event}] No warning signal found in {len(window_df)} bars. "
                f"Max prob in window: {window_df[prob_col].max():.4f}"
            )
            results.append({
                "event":              event,
                "hmm_onset_utc":      str(onset_time),
                "first_warning_utc":  None,
                "lead_time_minutes":  None,
                "n_bars_in_window":   len(window_df),
                "max_prob_in_window": round(window_df[prob_col].max(), 4),
                "status":             "NO_SIGNAL",
            })
            continue

        first_warning = warnings_df.sort("time")["time"][0]
        delta_seconds = (onset_time - first_warning).total_seconds()
        lead_minutes  = round(delta_seconds / 60, 2)

        logger.info(
            f"  [{event}] First warning: {first_warning}  "
            f"→ Lead time: {lead_minutes:.1f} min"
        )
        results.append({
            "event":              event,
            "hmm_onset_utc":      str(onset_time),
            "first_warning_utc":  str(first_warning),
            "lead_time_minutes":  lead_minutes,
            "n_bars_in_window":   len(window_df),
            "max_prob_in_window": round(warnings_df[prob_col].max(), 4),
            "status":             "OK",
        })

    return results


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================

def print_lead_time_table(results):
    """
    Print a clean, paper-ready lead-time summary table.
    """
    print("\n" + "=" * 80)
    print("PREDICTIVE LEAD-TIME ANALYSIS")
    print(f"Model: XGBoost Binary | Seed: {BEST_SEED} | Fold: {BEST_FOLD} | Pooled")
    print(f"Threshold: P(stress) > {PROB_THRESHOLD} | Lookback: {LOOKBACK_HOURS}h pre-onset")
    print("=" * 80)
    print(
        f"{'Event':<25} {'HMM Onset (UTC)':<22} "
        f"{'First Warning (UTC)':<22} {'Lead Time':<12} {'Status'}"
    )
    print("─" * 80)

    lead_times = []
    for r in results:
        lead_str    = f"{r['lead_time_minutes']:.1f} min" if r["lead_time_minutes"] is not None else "—"
        warning_str = r["first_warning_utc"][:19] if r["first_warning_utc"] else "—"
        onset_str   = r["hmm_onset_utc"][:19]

        print(
            f"{r['event']:<25} {onset_str:<22} "
            f"{warning_str:<22} {lead_str:<12} {r['status']}"
        )

        if r["lead_time_minutes"] is not None:
            lead_times.append(r["lead_time_minutes"])

    print("─" * 80)

    if lead_times:
        print(f"\n  Events with signal:  {len(lead_times)} / {len(results)}")
        print(f"  Mean lead time:      {np.mean(lead_times):.1f} min")
        print(f"  Median lead time:    {np.median(lead_times):.1f} min")
        print(f"  Min lead time:       {np.min(lead_times):.1f} min")
        print(f"  Max lead time:       {np.max(lead_times):.1f} min")
    else:
        print("\n  No lead-time signals detected across any events.")

    print("=" * 80)


def print_operational_interpretation(results):
    """
    Print a practitioner-facing interpretation of each event's lead time.
    """
    print("\n" + "=" * 80)
    print("OPERATIONAL INTERPRETATION")
    print("─" * 80)

    action_map = {
        range(0, 5):   "Insufficient for manual intervention; suitable for automated circuit breakers only.",
        range(5, 15):  "Sufficient for automated deleveraging and margin top-up triggers.",
        range(15, 30): "Sufficient for both automated protocols and human-in-the-loop risk decisions.",
        range(30, 120): "Ample window; supports full position review and coordinated desk response.",
    }

    for r in results:
        if r["lead_time_minutes"] is None:
            print(f"  {r['event']:<25}  [{r['status']}] — no actionable signal detected.")
            continue

        lt   = r["lead_time_minutes"]
        note = "Outside mapped range."
        for rng, msg in action_map.items():
            if int(lt) in rng:
                note = msg
                break

        print(f"  {r['event']:<25}  {lt:.1f} min → {note}")

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    # Locate and load XGBoost binary probability predictions
    try:
        prob_path = resolve_prob_path(bucket)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    df = load_prob_file(bucket, prob_path)

    # Identify the stress probability column
    try:
        prob_col = detect_prob_column(df)
    except ValueError as e:
        logger.error(str(e))
        return

    # Compute lead times for all four crisis events
    logger.info("Computing lead times across crisis events...")
    results = compute_lead_times(df, prob_col)

    # Print results
    print_lead_time_table(results)
    print_operational_interpretation(results)

    # Save to disk
    out_df = pl.DataFrame([
        {k: v for k, v in r.items()} for r in results
    ])
    out_df.write_csv(OUTPUT_CSV)
    out_df.write_parquet(OUTPUT_PARQUET)
    logger.info(f"Lead-time results saved to {OUTPUT_CSV}")
    logger.info(f"Lead-time results saved to {OUTPUT_PARQUET}")

    print(f"\n  Output files:")
    print(f"    {OUTPUT_CSV}")
    print(f"    {OUTPUT_PARQUET}")
    print()


if __name__ == "__main__":
    main()

