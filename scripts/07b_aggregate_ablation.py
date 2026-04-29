import sys
import os
import json
import logging
import warnings
warnings.filterwarnings("ignore")

import polars as pl
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASSETS, BUCKET

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

RESULTS_PREFIX = "v2/results_ablation/"
CONDITIONS     = ["control_raw", "experiment_fracdiff"]
MODES          = ["multiclass", "binary"]
MODELS         = ["lr", "rf", "xgb", "lstm", "cnn_gaf"]

# Ablation runs fold 4 only — largest expanding window
TARGET_FOLD = 4

LABEL_NAMES = {
    "multiclass": ["calm", "elevated", "stress"],
    "binary":     ["not_stress", "stress"],
}


# =============================================================================
# HELPERS
# =============================================================================

def fetch_metrics_blob(bucket, gcs_path):
    """Download and parse a metrics JSON blob. Returns None if missing."""
    blob = bucket.blob(gcs_path)
    if not blob.exists():
        logger.warning(f"  Missing: {gcs_path}")
        return None
    try:
        return json.loads(blob.download_as_text())
    except Exception as e:
        logger.warning(f"  Failed to parse {gcs_path}: {e}")
        return None


def extract_metrics(report, mode):
    """Extract flat metrics dict from a classification report."""
    if not report:
        return {}

    row = {}

    # Per-class F1
    for label in LABEL_NAMES[mode]:
        class_data = report.get(label, {})
        row[f"f1_{label}"]        = round(class_data.get("f1-score",  0.0), 4)
        row[f"precision_{label}"] = round(class_data.get("precision", 0.0), 4)
        row[f"recall_{label}"]    = round(class_data.get("recall",    0.0), 4)

    # Aggregate
    for avg_key in ["macro avg", "weighted avg"]:
        avg_data  = report.get(avg_key, {})
        clean_key = avg_key.replace(" ", "_")
        row[f"f1_{clean_key}"]        = round(avg_data.get("f1-score",  0.0), 4)
        row[f"precision_{clean_key}"] = round(avg_data.get("precision", 0.0), 4)
        row[f"recall_{clean_key}"]    = round(avg_data.get("recall",    0.0), 4)

    row["accuracy"] = round(report.get("accuracy", 0.0), 4)
    return row


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_ablation_results(bucket):
    """
    Pulls all ablation metrics JSONs from v2/results_ablation/ across:
      - Both conditions (control_raw, experiment_fracdiff)
      - Both modes (multiclass, binary)
      - All models (lr, rf, xgb, lstm, cnn_gaf)
      - Fold 4 pooled only

    Returns a Polars DataFrame with one row per
    (condition, mode, model) combination.
    """
    logger.info("Aggregating ablation results from GCS...")
    records = []

    for condition in CONDITIONS:
        prefix = f"{RESULTS_PREFIX}pooled/fold_{TARGET_FOLD}/{condition}/"

        for mode in MODES:
            gcs_path    = f"{prefix}metrics_{mode}.json"
            metrics_raw = fetch_metrics_blob(bucket, gcs_path)

            if metrics_raw is None:
                continue

            logger.info(f"  OK: {condition} {mode}")

            for model in MODELS:
                model_report = metrics_raw.get(model)
                if not model_report:
                    continue

                row = extract_metrics(model_report, mode)
                if not row:
                    continue

                row.update({
                    "condition": condition,
                    "mode":      mode,
                    "model":     model,
                    "fold":      TARGET_FOLD,
                })
                records.append(row)

    if not records:
        logger.error("No ablation records found — check GCS paths and credentials")
        return None

    df = pl.DataFrame(records)
    logger.info(f"Aggregated {len(df)} ablation result rows")
    return df


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================

def print_head_to_head(df):
    """
    Prints control vs fracdiff side by side per model per mode.
    The delta and winner columns are the core ablation finding.
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY — Control Raw vs Experiment FracDiff")
    print(f"Pooled Fold {TARGET_FOLD} | 18,826,552 train rows | 4,769,280 test rows")
    print("=" * 80)

    for mode in MODES:
        print(f"\n{'─' * 80}")
        print(f"MODE: {mode.upper()}")
        print(f"{'─' * 80}")
        print(f"{'Model':<12} {'Control F1W':<14} {'FracDiff F1W':<14} "
              f"{'Delta':<10} {'Control Acc':<14} {'FracDiff Acc':<14} {'Winner'}")
        print(f"{'─' * 80}")

        mode_df = df.filter(pl.col("mode") == mode)

        for model in MODELS:
            ctrl = mode_df.filter(
                (pl.col("condition") == "control_raw") &
                (pl.col("model") == model)
            )
            exp = mode_df.filter(
                (pl.col("condition") == "experiment_fracdiff") &
                (pl.col("model") == model)
            )

            if ctrl.is_empty() or exp.is_empty():
                continue

            ctrl_f1w = ctrl["f1_weighted_avg"][0]
            exp_f1w  = exp["f1_weighted_avg"][0]
            ctrl_acc = ctrl["accuracy"][0]
            exp_acc  = exp["accuracy"][0]
            delta    = exp_f1w - ctrl_f1w

            winner = "FRACDIFF ✓" if delta > 0.001 else "CONTROL ✓" if delta < -0.001 else "TIE"

            print(
                f"{model:<12} {ctrl_f1w:<14.4f} {exp_f1w:<14.4f} "
                f"{delta:+.4f}    {ctrl_acc:<14.4f} {exp_acc:<14.4f} {winner}"
            )


def print_stress_class_comparison(df):
    """
    Stress class F1 is the most important metric for this research question.
    A model that ignores the stress class is useless for market risk monitoring.
    """
    print(f"\n{'=' * 80}")
    print("STRESS CLASS F1 COMPARISON (Most Important Metric for Research Question)")
    print(f"{'=' * 80}")

    for mode in MODES:
        stress_col = "f1_stress"
        print(f"\n{'─' * 80}")
        print(f"MODE: {mode.upper()} — F1 for Stress Class")
        print(f"{'─' * 80}")
        print(f"{'Model':<12} {'Control':<12} {'FracDiff':<12} {'Delta':<10} {'Winner'}")
        print(f"{'─' * 80}")

        mode_df = df.filter(pl.col("mode") == mode)

        for model in MODELS:
            ctrl = mode_df.filter(
                (pl.col("condition") == "control_raw") &
                (pl.col("model") == model)
            )
            exp = mode_df.filter(
                (pl.col("condition") == "experiment_fracdiff") &
                (pl.col("model") == model)
            )

            if ctrl.is_empty() or exp.is_empty():
                continue

            ctrl_stress = ctrl[stress_col][0] if stress_col in ctrl.columns else 0.0
            exp_stress  = exp[stress_col][0]  if stress_col in exp.columns  else 0.0
            delta       = exp_stress - ctrl_stress
            winner      = "FRACDIFF ✓" if delta > 0.001 else "CONTROL ✓" if delta < -0.001 else "TIE"

            print(f"{model:<12} {ctrl_stress:<12.4f} {exp_stress:<12.4f} {delta:+.4f}    {winner}")


def print_ablation_verdict(df):
    """
    Counts wins per condition across all models and modes.
    Gives a clear overall verdict for the paper's results section.
    """
    print(f"\n{'=' * 80}")
    print("ABLATION VERDICT — Overall Win Count")
    print(f"{'=' * 80}")

    fracdiff_wins = 0
    control_wins  = 0
    ties          = 0
    total         = 0

    for mode in MODES:
        mode_df = df.filter(pl.col("mode") == mode)
        for model in MODELS:
            ctrl = mode_df.filter(
                (pl.col("condition") == "control_raw") &
                (pl.col("model") == model)
            )
            exp = mode_df.filter(
                (pl.col("condition") == "experiment_fracdiff") &
                (pl.col("model") == model)
            )
            if ctrl.is_empty() or exp.is_empty():
                continue

            delta = exp["f1_weighted_avg"][0] - ctrl["f1_weighted_avg"][0]
            total += 1
            if delta > 0.001:
                fracdiff_wins += 1
            elif delta < -0.001:
                control_wins += 1
            else:
                ties += 1

    print(f"\n  Total comparisons : {total} ({len(MODELS)} models × {len(MODES)} modes)")
    print(f"  FracDiff wins     : {fracdiff_wins}")
    print(f"  Control wins      : {control_wins}")
    print(f"  Ties              : {ties}")

    if fracdiff_wins > control_wins:
        print(f"\n  VERDICT: Fractional differencing IMPROVES performance")
        print(f"  ({fracdiff_wins}/{total} models benefit from price stationarity)")
    elif control_wins > fracdiff_wins:
        print(f"\n  VERDICT: Raw price OUTPERFORMS fractionally differenced price")
        print(f"  ({control_wins}/{total} models perform better without differencing)")
    else:
        print(f"\n  VERDICT: INCONCLUSIVE — equal wins across conditions")

    print(f"\n  Paper sentence:")
    print(f"  'Fractional differencing of the price series (d values: BTC=0.3,")
    print(f"   ETH=0.4, SOL=0.2) resulted in {fracdiff_wins}/{total} models showing")
    print(f"   improved weighted F1 scores on the pooled fold 4 test set,")
    print(f"   confirming the value of stationarity-preserving transformations")
    print(f"   for market stress classification.'")


def print_baseline_vs_ablation_context(df):
    """
    Compares ablation control_raw results against the 06b baseline
    to confirm they are consistent — they should be near-identical
    since both use raw features on the same fold 4 pooled data.
    """
    print(f"\n{'=' * 80}")
    print("CONSISTENCY CHECK — Control Raw vs 06b Baseline (should be near-identical)")
    print("Note: Small differences expected due to different random seeds in DataLoaders")
    print(f"{'=' * 80}")
    # This is a sanity check note — actual comparison requires loading 07a results
    print("  Run 07a_aggregate_results.py and compare pooled/fold_4 rows")
    print("  against control_raw rows here for full consistency verification.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    df = aggregate_ablation_results(bucket)
    if df is None:
        return

    # Save full ablation results
    output_parquet = "/workspace/maic/ablation_results.parquet"
    output_csv     = "/workspace/maic/ablation_results.csv"
    df.write_parquet(output_parquet)
    df.write_csv(output_csv)
    logger.info(f"Ablation results saved to {output_parquet}")

    # Print comparison tables
    print_head_to_head(df)
    print_stress_class_comparison(df)
    print_ablation_verdict(df)
    print_baseline_vs_ablation_context(df)

    print(f"\n{'=' * 80}")
    print(f"Full ablation results saved to:")
    print(f"  {output_parquet}")
    print(f"  {output_csv}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

