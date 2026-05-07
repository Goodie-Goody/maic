import sys
import os
import io
import gc
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb
from google.cloud import storage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASSETS, BUCKET, WINDOWS

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

# We use fold 3 as the robustness probe because:
#   - Training windows 0-2 provide enough data for a reliable local HMM fit
#   - Test window 3 (2022-11 → 2023-04) contains the FTX crisis event,
#     the most abrupt structural break in the dataset — the hardest case
#     for label contamination to matter
#   - Fold 3 results are already in production_results.csv for comparison

PROBE_FOLD       = 3
TRAIN_WINDOWS    = [0, 1, 2]   # Windows used to fit the local HMM
TEST_WINDOW      = 3           # Window whose labels we compare

# HMM config — mirrors 05_label_generation.py exactly
HMM_FEATURES = ["RV_300s", "OFI_300s", "Kyle_lambda_300s", "intensity_300s"]
N_STATES     = 3
N_ITER       = 100
N_SEEDS      = 5

# XGBoost config — mirrors 06d_train_production.py
XGB_PARAMS = {
    "n_estimators":    300,
    "max_depth":       6,
    "learning_rate":   0.05,
    "tree_method":     "hist",
    "device":          "cuda",   # falls back to cpu if no GPU
    "eval_metric":     "mlogloss",
    "use_label_encoder": False,
    "random_state":    42,
}

# GCS paths
FEATURES_RAW_PREFIX    = "v2/features/"         # used for HMM fitting
FEATURES_FD_PREFIX     = "v2/features_fracdiff/" # used for XGBoost
LABELS_GLOBAL_PREFIX   = "v2/labels/"

# Global XGBoost fold 3 metrics are loaded dynamically from production_results.csv
# at runtime — see load_global_xgb_metrics() below.

# Output
OUTPUT_DIR = "/workspace/maic"


# =============================================================================
# LOAD GLOBAL BASELINE FROM PRODUCTION RESULTS
# =============================================================================

def load_global_xgb_metrics():
    """
    Pull fold 3, seed 42, binary XGBoost metrics directly from
    production_results.csv — the single source of truth for all
    reported numbers in the paper.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "production_results.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"production_results.csv not found at {csv_path}. "
            f"Run 07c_aggregate_production.py first."
        )

    df = pl.read_csv(csv_path)
    row = df.filter(
        (pl.col("fold")  == PROBE_FOLD) &
        (pl.col("seed")  == 42)         &
        (pl.col("mode")  == "binary")   &
        (pl.col("model") == "xgb")
    )

    if row.is_empty():
        raise RuntimeError(
            f"No production result found for fold={PROBE_FOLD}, seed=42, "
            f"mode=binary, model=xgb in production_results.csv"
        )

    f1w  = float(row["f1_weighted_avg"][0])
    f1s  = float(row["f1_stress"][0])
    logger.info(
        f"Global XGB baseline (fold {PROBE_FOLD}, seed 42, binary): "
        f"F1-W={f1w:.4f}, F1-Stress={f1s:.4f}"
    )
    return f1w, f1s




def parse_window_months(window_idx):
    start_ym, end_ym = WINDOWS[window_idx]
    sy, sm = int(start_ym[:4]), int(start_ym[5:])
    ey, em = int(end_ym[:4]), int(end_ym[5:])
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def load_asset_window(bucket, asset, window_idx, prefix, columns=None):
    """
    Load feature or label parquet files for a single asset and window.
    Returns a sorted Polars DataFrame, or None if no files found.
    """
    months  = parse_window_months(window_idx)
    frames  = []

    # SOLUSDT did not exist before November 2020
    for year, month in months:
        if asset == "SOLUSDT" and (year, month) < (2020, 11):
            continue
        path = f"{prefix}{asset}-{'features' if 'features' in prefix else 'labels'}-{year}-{month:02d}.parquet"
        blob = bucket.blob(path)
        if not blob.exists():
            logger.warning(f"  Missing: {path}")
            continue
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        df = pl.read_parquet(buf, columns=columns)
        frames.append(df)

    if not frames:
        return None
    return pl.concat(frames).sort("time")


def load_pooled_windows(bucket, window_indices, prefix, columns=None):
    """
    Load and pool data across multiple windows and all assets.
    Tags each row with asset_id to mirror the pooled training setup in 06d.
    Returns a single sorted DataFrame.
    """
    frames = []
    for asset_id, asset in enumerate(ASSETS):
        for w in window_indices:
            df = load_asset_window(bucket, asset, w, prefix, columns)
            if df is not None:
                df = df.with_columns(pl.lit(asset_id).alias("asset_id"))
                frames.append(df)
    if not frames:
        raise RuntimeError(f"No data found for windows {window_indices}")
    return pl.concat(frames).sort("time")


# =============================================================================
# HMM FITTING — LOCAL (TRAINING DATA ONLY)
# =============================================================================

def fit_local_hmm(bucket, asset, train_window_indices):
    """
    Fit a 3-state Gaussian HMM on training-window features only for one asset.
    Mirrors fit_hmm() in 05_label_generation.py exactly.
    Returns (model, scaler, state_map).
    """
    logger.info(f"  [{asset}] Loading training features for local HMM fit...")
    frames = []
    for w in train_window_indices:
        df = load_asset_window(
            bucket, asset, w, FEATURES_RAW_PREFIX,
            columns=["time"] + HMM_FEATURES
        )
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No training features found for {asset}")

    train_df = pl.concat(frames).sort("time")
    X        = train_df.select(HMM_FEATURES).to_numpy()
    X        = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model = None
    best_score = -np.inf

    for seed in range(N_SEEDS):
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=N_ITER,
            random_state=seed,
            verbose=False,
        )
        model.fit(X_scaled)
        score = model.score(X_scaled)
        logger.info(
            f"    Seed {seed}: log-likelihood {score:.4f}, "
            f"converged: {model.monitor_.converged}"
        )
        if score > best_score:
            best_score = score
            best_model = model

    logger.info(f"  [{asset}] Best log-likelihood: {best_score:.4f}")

    # Map states by RV mean (lowest=calm, mid=elevated, highest=stress)
    rv_idx    = HMM_FEATURES.index("RV_300s")
    state_rvs = {s: best_model.means_[s][rv_idx] for s in range(N_STATES)}
    sorted_s  = sorted(state_rvs.items(), key=lambda x: x[1])
    state_map = {sorted_s[i][0]: i for i in range(N_STATES)}

    return best_model, scaler, state_map


def predict_local_labels(model, scaler, state_map, bucket, asset, window_idx):
    """
    Apply a fitted HMM to a single window and return (time_series, local_labels).
    """
    df = load_asset_window(
        bucket, asset, window_idx, FEATURES_RAW_PREFIX,
        columns=["time"] + HMM_FEATURES
    )
    if df is None:
        return None, None

    X        = df.select(HMM_FEATURES).to_numpy()
    X        = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)
    raw      = model.predict(X_scaled)
    mapped   = np.array([state_map[s] for s in raw])
    return df["time"], mapped


# =============================================================================
# LABEL COMPARISON
# =============================================================================

def load_global_labels(bucket, asset, window_idx):
    """Load global HMM labels from GCS for a single asset and window."""
    df = load_asset_window(
        bucket, asset, window_idx, LABELS_GLOBAL_PREFIX,
        columns=["time", "label"]
    )
    return df


def compare_labels(global_labels, local_labels, asset):
    """
    Compare global vs local HMM labels element-wise.
    Returns a dict of agreement statistics.
    """
    g = np.array(global_labels)
    l = np.array(local_labels)

    agreement     = (g == l).mean()
    stress_global = (g == 2).mean()
    stress_local  = (l == 2).mean()

    # Per-state agreement
    per_state = {}
    for s, name in [(0, "calm"), (1, "elevated"), (2, "stress")]:
        mask = (g == s)
        if mask.sum() > 0:
            per_state[name] = (g[mask] == l[mask]).mean()
        else:
            per_state[name] = float("nan")

    logger.info(f"  [{asset}] Label agreement: {agreement:.4f}")
    logger.info(f"  [{asset}] Stress % — global: {stress_global:.4f}, local: {stress_local:.4f}")
    for name, val in per_state.items():
        logger.info(f"  [{asset}] Agreement within {name}: {val:.4f}")

    return {
        "asset":          asset,
        "agreement":      round(agreement, 4),
        "stress_global":  round(stress_global, 4),
        "stress_local":   round(stress_local, 4),
        "agree_calm":     round(per_state["calm"], 4),
        "agree_elevated": round(per_state["elevated"], 4),
        "agree_stress":   round(per_state["stress"], 4),
    }


# =============================================================================
# XGBOOST ROBUSTNESS EVALUATION
# =============================================================================

def prepare_xy(feat_df, label_series, binary=True):
    """
    Merge features with labels on time and return X, y arrays.
    Mirrors prepare_xy logic in 06d_train_production.py.
    """
    label_df = pl.DataFrame({"time": feat_df["time"], "label": label_series})
    merged   = feat_df.join(label_df, on="time", how="inner")

    feature_cols = [c for c in merged.columns if c not in ["time", "label", "asset_id"]]
    X = merged.select(feature_cols).to_numpy().astype(np.float32)
    y = merged["label"].to_numpy().astype(np.int32)

    if binary:
        y = (y == 2).astype(np.int32)

    return X, y


def align_and_concat(frames):
    """
    Concatenate a list of DataFrames that may have different column sets.
    Finds the intersection of columns across all frames and selects only those,
    preventing ShapeError when fracdiff feature files differ across assets.
    """
    # Find common columns in insertion order of the first frame
    common = set(frames[0].columns)
    for f in frames[1:]:
        common &= set(f.columns)
    common_ordered = [c for c in frames[0].columns if c in common]
    logger.info(f"  Common columns across all frames: {len(common_ordered)}")
    return pl.concat([f.select(common_ordered) for f in frames])


TRAIN_CHECKPOINT = f"{OUTPUT_DIR}/robustness_train_checkpoint.parquet"
TEST_CHECKPOINT  = f"{OUTPUT_DIR}/robustness_test_checkpoint.parquet"


def run_xgb_robustness(bucket, global_f1w, global_f1s):
    """
    Train XGBoost on fold 3 training data with LOCAL HMM labels,
    evaluate on fold 3 test data with LOCAL labels,
    and compare to the global-label production result.

    Checkpoints train and test DataFrames to disk so the expensive
    HMM fitting and GCS downloads are skipped on restart.
    """
    logger.info("Running XGBoost robustness evaluation...")

    # --- Build or load training set ---
    if os.path.exists(TRAIN_CHECKPOINT):
        logger.info(f"  Loading training checkpoint from {TRAIN_CHECKPOINT}")
        train_df = pl.read_parquet(TRAIN_CHECKPOINT)
    else:
        train_feat_frames = []

        for asset_id, asset in enumerate(ASSETS):
            logger.info(f"  Fitting local HMM for {asset} (training set)...")
            model, scaler, state_map = fit_local_hmm(bucket, asset, TRAIN_WINDOWS)

            for w in TRAIN_WINDOWS:
                feat_df = load_asset_window(bucket, asset, w, FEATURES_FD_PREFIX)
                if feat_df is None:
                    continue
                _, local_labels = predict_local_labels(
                    model, scaler, state_map, bucket, asset, w
                )
                if local_labels is None:
                    continue

                feat_df = feat_df.with_columns(pl.lit(asset_id).alias("asset_id"))

                # Align local labels to fracdiff time index
                raw_df       = load_asset_window(
                    bucket, asset, w, FEATURES_RAW_PREFIX,
                    columns=["time"] + HMM_FEATURES
                )
                label_df     = pl.DataFrame({
                    "time":  raw_df["time"],
                    "label": pl.Series(local_labels),
                })
                feat_df = feat_df.join(label_df, on="time", how="inner")
                train_feat_frames.append(feat_df)

            gc.collect()

        if not train_feat_frames:
            raise RuntimeError("No training data assembled for XGBoost robustness check")

        train_df = align_and_concat(train_feat_frames).sort("time")
        train_df.write_parquet(TRAIN_CHECKPOINT)
        logger.info(f"  Training checkpoint saved to {TRAIN_CHECKPOINT}")

    feature_cols = [c for c in train_df.columns if c not in ["time", "label", "asset_id"]]
    X_train      = train_df.select(feature_cols).to_numpy().astype(np.float32)
    X_train      = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_bin  = (train_df["label"].to_numpy() == 2).astype(np.int32)
    logger.info(f"  Training set: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")

    # --- Build or load test set ---
    if os.path.exists(TEST_CHECKPOINT):
        logger.info(f"  Loading test checkpoint from {TEST_CHECKPOINT}")
        test_df = pl.read_parquet(TEST_CHECKPOINT)
    else:
        test_feat_frames = []

        for asset_id, asset in enumerate(ASSETS):
            feat_df = load_asset_window(bucket, asset, TEST_WINDOW, FEATURES_FD_PREFIX)
            if feat_df is None:
                continue
            feat_df = feat_df.with_columns(pl.lit(asset_id).alias("asset_id"))

            global_label_df = load_global_labels(bucket, asset, TEST_WINDOW)

            logger.info(f"  Fitting local HMM for {asset} (test labels)...")
            model, scaler, state_map = fit_local_hmm(bucket, asset, TRAIN_WINDOWS)
            _, local_lab = predict_local_labels(
                model, scaler, state_map, bucket, asset, TEST_WINDOW
            )
            raw_test_df = load_asset_window(
                bucket, asset, TEST_WINDOW, FEATURES_RAW_PREFIX,
                columns=["time"] + HMM_FEATURES
            )
            local_label_df = pl.DataFrame({
                "time":        raw_test_df["time"],
                "local_label": pl.Series(local_lab),
            })

            merged = feat_df \
                .join(global_label_df.rename({"label": "global_label"}), on="time", how="inner") \
                .join(local_label_df, on="time", how="inner")

            test_feat_frames.append(merged)
            gc.collect()

        if not test_feat_frames:
            raise RuntimeError("No test data assembled")

        test_df = align_and_concat(test_feat_frames).sort("time")
        test_df.write_parquet(TEST_CHECKPOINT)
        logger.info(f"  Test checkpoint saved to {TEST_CHECKPOINT}")

    # Use only the feature cols identified from training data
    feature_cols = [
        c for c in feature_cols
        if c in test_df.columns and c not in ["time", "label", "asset_id",
                                               "global_label", "local_label"]
    ]
    X_test       = test_df.select(feature_cols).to_numpy().astype(np.float32)
    X_test       = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test_glob  = (test_df["global_label"].to_numpy() == 2).astype(np.int32)
    y_test_local = (test_df["local_label"].to_numpy()  == 2).astype(np.int32)
    logger.info(f"  Test set: {X_test.shape[0]:,} rows")

    # --- Train XGBoost on local labels ---
    logger.info("  Training XGBoost on local labels...")
    try:
        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(X_train, y_train_bin)
    except Exception:
        logger.warning("  CUDA unavailable, falling back to CPU")
        params_cpu = {**XGB_PARAMS, "tree_method": "hist", "device": "cpu"}
        clf = xgb.XGBClassifier(**params_cpu)
        clf.fit(X_train, y_train_bin)

    y_pred = clf.predict(X_test)

    # Local model vs local test labels (apples-to-apples)
    f1_local_vs_local  = f1_score(y_test_local, y_pred, average="weighted")
    f1s_local_vs_local = f1_score(y_test_local, y_pred, pos_label=1, average="binary")

    # Local model vs global test labels (cross-check)
    f1_local_vs_global  = f1_score(y_test_glob, y_pred, average="weighted")
    f1s_local_vs_global = f1_score(y_test_glob, y_pred, pos_label=1, average="binary")

    logger.info(f"  XGB (local labels) vs local test  — F1-W: {f1_local_vs_local:.4f}, F1-Stress: {f1s_local_vs_local:.4f}")
    logger.info(f"  XGB (local labels) vs global test — F1-W: {f1_local_vs_global:.4f}, F1-Stress: {f1s_local_vs_global:.4f}")
    logger.info(f"  XGB (global, production)          — F1-W: {global_f1w:.4f}, F1-Stress: {global_f1s:.4f}")

    return {
        "f1w_local_vs_local":    round(f1_local_vs_local,   4),
        "f1s_local_vs_local":    round(f1s_local_vs_local,  4),
        "f1w_local_vs_global":   round(f1_local_vs_global,  4),
        "f1s_local_vs_global":   round(f1s_local_vs_global, 4),
        "f1w_global_production": global_f1w,
        "f1s_global_production": global_f1s,
    }


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================

def print_label_agreement_table(comparison_results):
    print("\n" + "=" * 80)
    print("HMM ROBUSTNESS CHECK — LABEL AGREEMENT (Global vs Local Fit)")
    print(f"Probe: Fold {PROBE_FOLD} | Test Window: {WINDOWS[TEST_WINDOW][0]} → {WINDOWS[TEST_WINDOW][1]}")
    print(f"Local HMM trained on Windows {TRAIN_WINDOWS} only")
    print("=" * 80)
    print(f"{'Asset':<12} {'Overall':<12} {'Calm':<12} {'Elevated':<12} {'Stress':<12} {'Stress%(G)':<12} {'Stress%(L)'}")
    print("-" * 80)

    agreements = []
    for r in comparison_results:
        print(
            f"{r['asset']:<12} {r['agreement']:<12.4f} "
            f"{r['agree_calm']:<12.4f} {r['agree_elevated']:<12.4f} "
            f"{r['agree_stress']:<12.4f} {r['stress_global']:<12.4f} "
            f"{r['stress_local']:.4f}"
        )
        agreements.append(r["agreement"])

    print("-" * 80)
    print(f"{'Mean':<12} {np.mean(agreements):<12.4f}")
    print("=" * 80)
    print()
    print("Interpretation:")
    mean_agree = np.mean(agreements)
    if mean_agree >= 0.90:
        print(f"  Mean agreement {mean_agree:.4f} >= 0.90: global HMM labelling introduces")
        print(f"  minimal look-ahead bias. Local and global labels are substantively equivalent.")
    elif mean_agree >= 0.80:
        print(f"  Mean agreement {mean_agree:.4f} in [0.80, 0.90): modest label differences exist")
        print(f"  but are unlikely to materially affect model training.")
    else:
        print(f"  Mean agreement {mean_agree:.4f} < 0.80: non-trivial label divergence detected.")
        print(f"  Consider per-fold HMM refitting as a robustness extension.")
    print("=" * 80)


def print_xgb_robustness_table(xgb_results):
    print("\n" + "=" * 80)
    print("HMM ROBUSTNESS CHECK — XGBOOST PERFORMANCE COMPARISON")
    print(f"Fold {PROBE_FOLD} | Binary Stress Detection | Seed 42 | Pooled")
    print("=" * 80)
    print(f"{'Condition':<45} {'F1-Weighted':<15} {'F1-Stress'}")
    print("-" * 80)
    print(
        f"{'Global labels (production run)':<45} "
        f"{xgb_results['f1w_global_production']:<15.4f} "
        f"{xgb_results['f1s_global_production']:.4f}"
    )
    print(
        f"{'Local labels vs local test labels':<45} "
        f"{xgb_results['f1w_local_vs_local']:<15.4f} "
        f"{xgb_results['f1s_local_vs_local']:.4f}"
    )
    print(
        f"{'Local labels vs global test labels (cross)':<45} "
        f"{xgb_results['f1w_local_vs_global']:<15.4f} "
        f"{xgb_results['f1s_local_vs_global']:.4f}"
    )
    print("-" * 80)

    delta_f1w = xgb_results["f1w_local_vs_local"] - xgb_results["f1w_global_production"]
    delta_f1s = xgb_results["f1s_local_vs_local"] - xgb_results["f1s_global_production"]

    print(f"\n  Delta F1-Weighted (local - global): {delta_f1w:+.4f}")
    print(f"  Delta F1-Stress   (local - global): {delta_f1s:+.4f}")
    print()

    if abs(delta_f1w) <= 0.02:
        print("  Verdict: Performance difference is within ±0.02 — global HMM labelling")
        print("  does not materially inflate model performance. Look-ahead bias is negligible.")
    else:
        print(f"  Verdict: Performance difference of {delta_f1w:+.4f} exceeds ±0.02 threshold.")
        print("  Consider acknowledging this in the limitations section.")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    # --- Skip logic ---
    _marker = "v2/pipeline_markers/10_hmm_robustness_check.done"
    if bucket.blob(_marker).exists():
        logger.info("10_hmm_robustness_check — already complete, skipping")
        logger.info(f"  To rerun: gsutil rm gs://{BUCKET}/{_marker}")
        return

    logger.info("=" * 60)
    logger.info("HMM ROBUSTNESS CHECK")
    logger.info(f"Probe fold     : {PROBE_FOLD}")
    logger.info(f"Train windows  : {TRAIN_WINDOWS} ({[WINDOWS[w] for w in TRAIN_WINDOWS]})")
    logger.info(f"Test window    : {TEST_WINDOW} ({WINDOWS[TEST_WINDOW]})")
    logger.info(f"HMM features   : {HMM_FEATURES}")
    logger.info(f"HMM states     : {N_STATES}")
    logger.info(f"HMM seeds      : {N_SEEDS}")
    logger.info("=" * 60)

    # --- Step 1: Label comparison (skip if already completed) ---
    AGREEMENT_CSV = f"{OUTPUT_DIR}/hmm_robustness_label_agreement.csv"

    if os.path.exists(AGREEMENT_CSV):
        logger.info(f"STEP 1: Label agreement already computed — loading from {AGREEMENT_CSV}")
        comparison_results = pl.read_csv(AGREEMENT_CSV).to_dicts()
        print_label_agreement_table(comparison_results)
    else:
        logger.info("STEP 1: Comparing global vs local HMM labels on test window")
        comparison_results = []

        for asset in ASSETS:
            logger.info(f"Processing {asset}...")
            model, scaler, state_map = fit_local_hmm(bucket, asset, TRAIN_WINDOWS)
            times, local_labels = predict_local_labels(
                model, scaler, state_map, bucket, asset, TEST_WINDOW
            )
            if local_labels is None:
                logger.warning(f"  No test data for {asset} in window {TEST_WINDOW}")
                continue
            global_df = load_global_labels(bucket, asset, TEST_WINDOW)
            if global_df is None or global_df.is_empty():
                logger.warning(f"  No global labels for {asset} in window {TEST_WINDOW}")
                continue
            local_df = pl.DataFrame({"time": times, "local": pl.Series(local_labels)})
            merged   = global_df.join(local_df, on="time", how="inner")
            result   = compare_labels(
                merged["label"].to_numpy(),
                merged["local"].to_numpy(),
                asset
            )
            comparison_results.append(result)
            gc.collect()

        print_label_agreement_table(comparison_results)
        pl.DataFrame(comparison_results).write_csv(AGREEMENT_CSV)
        logger.info(f"Label agreement saved to {AGREEMENT_CSV}")

    # --- Step 2: XGBoost performance comparison ---
    logger.info("\nSTEP 2: XGBoost performance comparison (local vs global labels)")
    global_f1w, global_f1s = load_global_xgb_metrics()
    xgb_results = run_xgb_robustness(bucket, global_f1w, global_f1s)
    print_xgb_robustness_table(xgb_results)

    # Save XGBoost results
    xgb_df = pl.DataFrame([xgb_results])
    xgb_df.write_csv(f"{OUTPUT_DIR}/hmm_robustness_xgb_comparison.csv")
    logger.info(f"XGBoost comparison saved to {OUTPUT_DIR}/hmm_robustness_xgb_comparison.csv")

    # --- Write done marker ---
    bucket.blob(_marker).upload_from_string(b"")
    logger.info(f"Done marker written: gs://{BUCKET}/{_marker}")

    print(f"\n  Output files:")
    print(f"    {OUTPUT_DIR}/hmm_robustness_label_agreement.csv")
    print(f"    {OUTPUT_DIR}/hmm_robustness_xgb_comparison.csv")
    print()


if __name__ == "__main__":
    main()
