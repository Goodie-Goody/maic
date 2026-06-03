"""
13a_persistence_baseline.py

Computes a persistence (naive) baseline for comparison against production
XGBoost results. Addresses reviewer concern 6: a binary weighted F1 of 0.97
requires a hard baseline to confirm it reflects genuine predictive skill
rather than class imbalance or label autocorrelation.

WHAT IT DOES:
    For each fold and mode (binary / multiclass), loads the pooled
    predictions parquet (which contains both true labels and predicted
    probabilities), constructs a persistence prediction by shifting true
    labels forward by one bar (i.e. "predict today's label = yesterday's
    label"), and computes standard classification metrics.

    Also computes a majority-class baseline (always predict the most
    frequent class) as a second reference point.

OUTPUTS (repo root):
    persistence_baseline_results.csv   Per-fold, per-mode metrics for
                                       both persistence and majority-class
                                       baselines, alongside XGBoost for
                                       direct comparison.

    Console: formatted comparison table.

WHY THIS MATTERS:
    If stress is highly autocorrelated (bars cluster together), a
    persistence rule can achieve surprisingly high weighted F1 simply
    by inertia. Showing that XGBoost substantially outperforms persistence
    — especially on stress-class F1 and at fold transitions — confirms
    the model is learning structure beyond label momentum.

GCS PATHS (mirrors 09_lead_time_analysis.py and 06d_train_production.py):
    Predictions : v2/results_production/seed_42/pooled/fold_{N}/predictions_binary.parquet
                  v2/results_production/seed_42/pooled/fold_{N}/predictions_multiclass.parquet
    Labels      : v2/labels/{ASSET}-labels-{YEAR}-{MONTH:02d}.parquet

Usage:
    python3 scripts/13a_persistence_baseline.py
"""

import sys
import os
import io
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from sklearn.metrics import (
    f1_score,
    classification_report,
    accuracy_score,
)
from google.cloud import storage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from config import ASSETS, BUCKET, WINDOWS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(REPO_ROOT, "gcp-key.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

BEST_SEED        = 42
N_FOLDS          = 4
MODES            = ["binary", "multiclass"]
RESULTS_PREFIX   = f"v2/results_production/seed_{BEST_SEED}/"
OUTPUT_CSV       = os.path.join(REPO_ROOT, "persistence_baseline_results.csv")

# For reference: XGBoost production results are in production_results.csv
# We pull XGB seed=42 results inline for the comparison table.
PRODUCTION_CSV   = os.path.join(REPO_ROOT, "production_results.csv")

# =============================================================================
# GCS HELPERS
# =============================================================================

def load_parquet_from_gcs(bucket, gcs_path):
    blob = bucket.blob(gcs_path)
    if not blob.exists():
        return None
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return pl.read_parquet(buf)


def load_predictions(bucket, fold, mode):
    path = f"{RESULTS_PREFIX}pooled/fold_{fold}/predictions_{mode}.parquet"
    logger.info(f"  Loading predictions: {path}")
    df = load_parquet_from_gcs(bucket, path)
    if df is None:
        raise FileNotFoundError(f"Predictions not found: {path}")
    logger.info(f"  Loaded {len(df):,} rows | columns: {df.columns}")
    return df


# =============================================================================
# BASELINE COMPUTATION
# =============================================================================

def compute_persistence_baseline(y_true: np.ndarray, mode: str) -> dict:
    """
    Persistence baseline: predict that each bar's label equals the
    previous bar's label. The first bar has no predecessor, so it is
    dropped (one row lost per fold — negligible).
    """
    y_persist = y_true[:-1]   # yesterday's label
    y_eval    = y_true[1:]    # today's true label

    avg = "binary" if mode == "binary" else "weighted"

    f1w  = f1_score(y_eval, y_persist, average="weighted",    zero_division=0)
    f1m  = f1_score(y_eval, y_persist, average="macro",       zero_division=0)
    acc  = accuracy_score(y_eval, y_persist)

    # Stress-class F1 specifically
    classes = np.unique(y_eval)
    if mode == "binary":
        stress_label = 1
    else:
        stress_label = 2   # calm=0, elevated=1, stress=2

    if stress_label in classes:
        f1s = f1_score(
            y_eval, y_persist,
            labels=[stress_label],
            average="macro",
            zero_division=0,
        )
    else:
        f1s = 0.0

    return {
        "f1_weighted": round(f1w, 4),
        "f1_macro":    round(f1m, 4),
        "f1_stress":   round(f1s, 4),
        "accuracy":    round(acc, 4),
    }


def compute_majority_baseline(y_true: np.ndarray, mode: str) -> dict:
    """
    Majority-class baseline: always predict the most frequent class.
    Upper bound for a degenerate classifier.
    """
    classes, counts = np.unique(y_true, return_counts=True)
    majority = classes[np.argmax(counts)]
    y_maj    = np.full_like(y_true, fill_value=majority)

    f1w = f1_score(y_true, y_maj, average="weighted", zero_division=0)
    f1m = f1_score(y_true, y_maj, average="macro",    zero_division=0)
    acc = accuracy_score(y_true, y_maj)

    stress_label = 1 if mode == "binary" else 2
    if stress_label in classes:
        f1s = f1_score(
            y_true, y_maj,
            labels=[stress_label],
            average="macro",
            zero_division=0,
        )
    else:
        f1s = 0.0

    return {
        "f1_weighted": round(f1w, 4),
        "f1_macro":    round(f1m, 4),
        "f1_stress":   round(f1s, 4),
        "accuracy":    round(acc, 4),
    }


def compute_class_prevalence(y_true: np.ndarray, mode: str) -> dict:
    """
    Report class prevalence (% of bars in each class) for the fold.
    This directly answers the reviewer question about whether high F1
    is an artifact of class imbalance.
    """
    total = len(y_true)
    classes, counts = np.unique(y_true, return_counts=True)
    prev = {int(c): round(cnt / total * 100, 2) for c, cnt in zip(classes, counts)}

    if mode == "binary":
        return {
            "pct_not_stress": prev.get(0, 0.0),
            "pct_stress":     prev.get(1, 0.0),
            "pct_elevated":   None,
            "pct_calm":       None,
        }
    else:
        return {
            "pct_calm":     prev.get(0, 0.0),
            "pct_elevated": prev.get(1, 0.0),
            "pct_stress":   prev.get(2, 0.0),
            "pct_not_stress": None,
        }


# =============================================================================
# TRUE LABEL EXTRACTION
# =============================================================================

def extract_true_labels(df: pl.DataFrame, mode: str) -> np.ndarray:
    """
    Extract true labels from predictions parquet.
    The parquet written by 06d_train_production.py includes a 'true_label'
    column. Fall back to inferring from available columns if needed.
    """
    if "true_label" in df.columns:
        return df["true_label"].to_numpy().astype(int)

    # Fallback: for binary, derive from xgb_prob_class1 > 0.5 is NOT the
    # true label. Look for y_true or label columns.
    for candidate in ["y_true", "label", "labels"]:
        if candidate in df.columns:
            return df[candidate].to_numpy().astype(int)

    raise KeyError(
        f"Cannot find true label column in predictions parquet. "
        f"Available columns: {df.columns}"
    )


# =============================================================================
# XGB REFERENCE (from production_results.csv)
# =============================================================================

def load_xgb_reference(fold: int, mode: str) -> dict:
    """
    Load XGBoost seed=42 results from production_results.csv for
    direct inline comparison.
    """
    if not os.path.exists(PRODUCTION_CSV):
        return {}
    df = pl.read_csv(PRODUCTION_CSV)
    row = df.filter(
        (pl.col("model") == "xgb") &
        (pl.col("seed")  == 42)    &
        (pl.col("fold")  == fold)  &
        (pl.col("mode")  == mode)
    )
    if row.is_empty():
        return {}
    r = row.row(0, named=True)
    return {
        "f1_weighted": r.get("f1_weighted_avg", None),
        "f1_macro":    r.get("f1_macro_avg",    None),
        "f1_stress":   r.get("f1_stress",        None),
        "accuracy":    r.get("accuracy",         None),
    }


# =============================================================================
# PRINT TABLE
# =============================================================================

def print_comparison_table(records: list):
    print("\n" + "=" * 90)
    print("PERSISTENCE BASELINE vs XGBoost COMPARISON")
    print(f"Seed: {BEST_SEED} | Pooled | All folds | Binary and Multiclass")
    print("=" * 90)

    header = (
        f"{'Fold':<6} {'Mode':<12} {'Baseline':<22} "
        f"{'F1-W':<8} {'F1-M':<8} {'F1-S':<8} {'Acc':<8} "
        f"{'Stress%':<10}"
    )
    print(header)
    print("-" * 90)

    prev_fold = None
    for r in records:
        if r["fold"] != prev_fold:
            if prev_fold is not None:
                print()
            prev_fold = r["fold"]

        stress_pct = r.get("pct_stress", "-")
        stress_str = f"{stress_pct:.1f}%" if stress_pct is not None else "-"

        f1w = f"{r['f1_weighted']:.4f}" if r["f1_weighted"] is not None else "-"
        f1m = f"{r['f1_macro']:.4f}"    if r["f1_macro"]    is not None else "-"
        f1s = f"{r['f1_stress']:.4f}"   if r["f1_stress"]   is not None else "-"
        acc = f"{r['accuracy']:.4f}"    if r["accuracy"]    is not None else "-"

        print(
            f"{r['fold']:<6} {r['mode']:<12} {r['baseline']:<22} "
            f"{f1w:<8} {f1m:<8} {f1s:<8} {acc:<8} "
            f"{stress_str:<10}"
        )

    print("=" * 90)
    print("\nInterpretation:")
    print("  XGBoost lift over persistence on F1-Stress is the key metric.")
    print("  High stress% (>30%) would suggest high F1 is partly prevalence-driven.")
    print("  XGBoost should substantially exceed persistence on stress-class F1.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    # --- Skip logic ---
    marker = "v2/pipeline_markers/13a_persistence_baseline.done"
    if bucket.blob(marker).exists():
        logger.info("13a_persistence_baseline — already complete, skipping")
        logger.info(f"  To rerun: gsutil rm gs://{BUCKET}/{marker}")
        return

    logger.info("=" * 60)
    logger.info("13a_persistence_baseline.py")
    logger.info("Computing persistence and majority-class baselines")
    logger.info("=" * 60)

    all_records = []

    for fold in range(1, N_FOLDS + 1):
        logger.info(f"\nFold {fold}")

        for mode in MODES:
            logger.info(f"  Mode: {mode}")

            try:
                df = load_predictions(bucket, fold, mode)
            except FileNotFoundError as e:
                logger.error(f"  {e}")
                continue

            try:
                y_true = extract_true_labels(df, mode)
            except KeyError as e:
                logger.error(f"  {e}")
                continue

            prevalence     = compute_class_prevalence(y_true, mode)
            persist_scores = compute_persistence_baseline(y_true, mode)
            majority_scores = compute_majority_baseline(y_true, mode)
            xgb_scores     = load_xgb_reference(fold, mode)

            stress_pct = prevalence.get("pct_stress")

            # Persistence row
            all_records.append({
                "fold":        fold,
                "mode":        mode,
                "baseline":    "persistence",
                "f1_weighted": persist_scores["f1_weighted"],
                "f1_macro":    persist_scores["f1_macro"],
                "f1_stress":   persist_scores["f1_stress"],
                "accuracy":    persist_scores["accuracy"],
                "pct_stress":  stress_pct,
                "pct_calm":    prevalence.get("pct_calm"),
                "pct_elevated":prevalence.get("pct_elevated"),
                "pct_not_stress": prevalence.get("pct_not_stress"),
            })

            # Majority-class row
            all_records.append({
                "fold":        fold,
                "mode":        mode,
                "baseline":    "majority_class",
                "f1_weighted": majority_scores["f1_weighted"],
                "f1_macro":    majority_scores["f1_macro"],
                "f1_stress":   majority_scores["f1_stress"],
                "accuracy":    majority_scores["accuracy"],
                "pct_stress":  stress_pct,
                "pct_calm":    prevalence.get("pct_calm"),
                "pct_elevated":prevalence.get("pct_elevated"),
                "pct_not_stress": prevalence.get("pct_not_stress"),
            })

            # XGBoost reference row
            if xgb_scores:
                all_records.append({
                    "fold":        fold,
                    "mode":        mode,
                    "baseline":    "xgb_seed42",
                    "f1_weighted": xgb_scores.get("f1_weighted"),
                    "f1_macro":    xgb_scores.get("f1_macro"),
                    "f1_stress":   xgb_scores.get("f1_stress"),
                    "accuracy":    xgb_scores.get("accuracy"),
                    "pct_stress":  stress_pct,
                    "pct_calm":    prevalence.get("pct_calm"),
                    "pct_elevated":prevalence.get("pct_elevated"),
                    "pct_not_stress": prevalence.get("pct_not_stress"),
                })

            logger.info(
                f"  Persistence  -> F1-W: {persist_scores['f1_weighted']:.4f} | "
                f"F1-S: {persist_scores['f1_stress']:.4f}"
            )
            logger.info(
                f"  Majority     -> F1-W: {majority_scores['f1_weighted']:.4f} | "
                f"F1-S: {majority_scores['f1_stress']:.4f}"
            )
            if xgb_scores:
                logger.info(
                    f"  XGBoost      -> F1-W: {xgb_scores.get('f1_weighted','?'):.4f} | "
                    f"F1-S: {xgb_scores.get('f1_stress','?'):.4f}"
                )

    if not all_records:
        logger.error("No records produced — check GCS paths and credentials")
        return

    print_comparison_table(all_records)

    out_df = pl.DataFrame(all_records)
    out_df.write_csv(OUTPUT_CSV)
    logger.info(f"Results saved to {OUTPUT_CSV}")

    # Upload to GCS alongside other result files
    gcs_out = "v2/results/persistence_baseline_results.csv"
    bucket.blob(gcs_out).upload_from_filename(OUTPUT_CSV)
    logger.info(f"Uploaded to gs://{BUCKET}/{gcs_out}")

    # Write done marker
    bucket.blob(marker).upload_from_string(b"")
    logger.info(f"Done marker written: gs://{BUCKET}/{marker}")

    logger.info("=" * 60)
    logger.info("13a_persistence_baseline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()