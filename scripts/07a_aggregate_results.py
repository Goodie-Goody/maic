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

RESULTS_PREFIX   = "v2/results_run1/"
MODES            = ["multiclass", "binary"]
MODELS           = ["lr", "rf", "xgb", "lstm", "cnn_gaf"]
N_FOLDS          = 4

# Label names per mode
LABEL_NAMES = {
    "multiclass": ["calm", "elevated", "stress"],
    "binary":     ["not_stress", "stress"],
}

# =============================================================================
# HELPERS
# =============================================================================

def safe_get(d, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d


def fetch_metrics_blob(bucket, gcs_path):
    """Download and parse a metrics JSON blob. Returns None if missing."""
    blob = bucket.blob(gcs_path)
    if not blob.exists():
        return None
    try:
        return json.loads(blob.download_as_text())
    except Exception as e:
        logger.warning(f"  Failed to parse {gcs_path}: {e}")
        return None


def extract_model_metrics(report, mode, model_name):
    """
    Extract key metrics from a classification report dict.
    Returns a flat dict of metrics for one model/mode/fold/asset combination.
    """
    if report is None:
        return None

    row = {}

    # Per-class F1 scores
    for label in LABEL_NAMES[mode]:
        class_report = report.get(label, {})
        row[f"f1_{label}"]        = round(class_report.get("f1-score",  0.0), 4)
        row[f"precision_{label}"] = round(class_report.get("precision", 0.0), 4)
        row[f"recall_{label}"]    = round(class_report.get("recall",    0.0), 4)

    # Aggregate metrics
    for avg_key in ["macro avg", "weighted avg"]:
        avg_report = report.get(avg_key, {})
        clean_key  = avg_key.replace(" ", "_")
        row[f"f1_{clean_key}"]        = round(avg_report.get("f1-score",  0.0), 4)
        row[f"precision_{clean_key}"] = round(avg_report.get("precision", 0.0), 4)
        row[f"recall_{clean_key}"]    = round(avg_report.get("recall",    0.0), 4)

    # Accuracy
    row["accuracy"] = round(report.get("accuracy", 0.0), 4)

    return row


# =============================================================================
# MAIN AGGREGATION
# =============================================================================

def aggregate_baseline_results(bucket):
    """
    Pulls all metrics JSONs from v2/results_run1/ across:
      - All assets (BTCUSDT, ETHUSDT, SOLUSDT) + pooled
      - All folds (1-4)
      - Both modes (multiclass, binary)
      - All models (lr, rf, xgb, lstm, cnn_gaf)

    Returns a Polars DataFrame with one row per
    (asset, fold, mode, model) combination.
    """
    logger.info("Aggregating baseline results from GCS...")
    records = []

    all_scopes = [(a, False) for a in ASSETS] + [("pooled", True)]

    for asset, is_pooled in all_scopes:
        for fold in range(1, N_FOLDS + 1):
            prefix = (
                f"{RESULTS_PREFIX}pooled/fold_{fold}/"
                if is_pooled
                else f"{RESULTS_PREFIX}{asset}/fold_{fold}/"
            )

            for mode in MODES:
                gcs_path    = f"{prefix}metrics_{mode}.json"
                metrics_raw = fetch_metrics_blob(bucket, gcs_path)

                if metrics_raw is None:
                    logger.warning(f"  Missing: {gcs_path}")
                    continue

                logger.info(f"  OK: {asset} fold {fold} {mode}")

                for model_name in MODELS:
                    model_report = metrics_raw.get(model_name)
                    if model_report is None:
                        continue

                    row = extract_model_metrics(model_report, mode, model_name)
                    if row is None:
                        continue

                    row.update({
                        "asset":    asset,
                        "fold":     fold,
                        "mode":     mode,
                        "model":    model_name,
                        "is_pooled": is_pooled,
                    })
                    records.append(row)

    if not records:
        logger.error("No records found — check GCS paths and credentials")
        return None

    df = pl.DataFrame(records)
    logger.info(f"Aggregated {len(df)} result rows")
    return df


def print_summary_table(df):
    """
    Prints a human-readable summary grouped by asset/mode/model,
    averaged across folds.
    """
    print("\n" + "=" * 80)
    print("BASELINE RESULTS SUMMARY — Averaged Across Folds")
    print("=" * 80)

    for mode in MODES:
        print(f"\n{'─' * 80}")
        print(f"MODE: {mode.upper()}")
        print(f"{'─' * 80}")
        print(f"{'Asset':<12} {'Model':<12} {'Accuracy':<10} {'F1 Weighted':<14} {'F1 Macro':<12} {'Folds'}")
        print(f"{'─' * 80}")

        mode_df = df.filter(pl.col("mode") == mode)

        for asset in ASSETS + ["pooled"]:
            asset_df = mode_df.filter(pl.col("asset") == asset)
            if asset_df.is_empty():
                continue

            for model in MODELS:
                model_df = asset_df.filter(pl.col("model") == model)
                if model_df.is_empty():
                    continue

                n_folds  = len(model_df)
                acc      = model_df["accuracy"].mean()
                f1_w     = model_df["f1_weighted_avg"].mean()
                f1_m     = model_df["f1_macro_avg"].mean()

                print(f"{asset:<12} {model:<12} {acc:<10.4f} {f1_w:<14.4f} {f1_m:<12.4f} {n_folds}")

            print()


def print_fold_detail(df, asset, mode):
    """
    Prints per-fold breakdown for a specific asset and mode.
    Useful for seeing whether performance degrades or improves over time.
    """
    print(f"\n{'=' * 80}")
    print(f"FOLD-LEVEL DETAIL — {asset} — {mode.upper()}")
    print(f"{'=' * 80}")
    print(f"{'Fold':<6} {'Model':<12} {'Accuracy':<10} {'F1 Weighted':<14} {'F1 Macro':<12} {'F1 Stress'}")
    print(f"{'─' * 80}")

    filtered = df.filter(
        (pl.col("asset") == asset) &
        (pl.col("mode")  == mode)
    ).sort(["fold", "model"])

    stress_col = "f1_stress" if mode == "binary" else "f1_stress"

    for row in filtered.iter_rows(named=True):
        stress_f1 = row.get(stress_col, 0.0) or 0.0
        print(
            f"{row['fold']:<6} {row['model']:<12} "
            f"{row['accuracy']:<10.4f} {row['f1_weighted_avg']:<14.4f} "
            f"{row['f1_macro_avg']:<12.4f} {stress_f1:.4f}"
        )


def print_best_models(df):
    """
    Prints the best performing model per asset/mode combination
    based on weighted F1, averaged across folds.
    """
    print(f"\n{'=' * 80}")
    print("BEST MODEL PER ASSET AND MODE (by Weighted F1, avg across folds)")
    print(f"{'=' * 80}")
    print(f"{'Asset':<12} {'Mode':<14} {'Best Model':<12} {'F1 Weighted':<14} {'Accuracy'}")
    print(f"{'─' * 80}")

    for asset in ASSETS + ["pooled"]:
        for mode in MODES:
            subset = df.filter(
                (pl.col("asset") == asset) &
                (pl.col("mode")  == mode)
            )
            if subset.is_empty():
                continue

            avg = subset.group_by("model").agg([
                pl.col("f1_weighted_avg").mean().alias("avg_f1_weighted"),
                pl.col("accuracy").mean().alias("avg_accuracy"),
            ]).sort("avg_f1_weighted", descending=True)

            best    = avg.row(0, named=True)
            print(
                f"{asset:<12} {mode:<14} {best['model']:<12} "
                f"{best['avg_f1_weighted']:<14.4f} {best['avg_accuracy']:.4f}"
            )


def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    # Aggregate all baseline results
    df = aggregate_baseline_results(bucket)
    if df is None:
        return

    # Save full results table locally for further analysis
    output_path = "/workspace/maic/baseline_results.parquet"
    df.write_parquet(output_path)
    logger.info(f"Full results saved to {output_path}")

    # Print summary tables
    print_summary_table(df)
    print_fold_detail(df, "pooled", "multiclass")
    print_fold_detail(df, "pooled", "binary")
    print_best_models(df)

    # Also save a CSV for easy reading
    csv_path = "/workspace/maic/baseline_results.csv"
    df.write_csv(csv_path)
    logger.info(f"CSV saved to {csv_path}")

    print(f"\n{'=' * 80}")
    print(f"Full results saved to:")
    print(f"  {output_path}")
    print(f"  {csv_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

