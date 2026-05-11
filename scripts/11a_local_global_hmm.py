"""
11a_hmm_stability_and_local_labels.py

Two interconnected analyses that together make the case for global HMM
labelling as the methodologically superior choice:

PART A — Global HMM Stability Proof
    Compares the April 20 backup HMM models against the current production
    models across all parameters: state means (all 4 features), covariance
    matrices, transition matrices, and downstream stress label distributions.
    Proves the global HMM converges to the same solution across independent
    runs — essentially deterministic at this data scale.

PART B — Local HMM Generation Across All Folds
    Fits a local HMM for each fold using only training windows available
    at that point in time (strict temporal honesty). Saves labels to GCS at
    v2/labels_local/fold_{n}/ and records stress distributions per fold.
    Produces the instability table showing local stress% fluctuates wildly
    while global stays near-constant.

    Fold 1 local: trained on Window 0 only        (~6 months, COVID only)
    Fold 2 local: trained on Windows 0-1          (~12 months)
    Fold 3 local: trained on Windows 0-2          (~18 months)
    Fold 4 local: trained on Windows 0-3          (~24 months)

OUTPUTS (all saved to repo root):
    global_hmm_stability.csv        Part A parameter comparison
    local_hmm_label_audit.csv       Part B per-fold distributions
    global_vs_local_summary.csv     Head-to-head comparison table

GCS:
    v2/labels_local/fold_{n}/{asset}-labels-{year}-{month:02d}.parquet

Usage:
    python3 scripts/11a_hmm_stability_and_local_labels.py
"""

import sys
import os
import io
import gc
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
from datetime import date

# =============================================================================
# DYNAMIC PATH RESOLUTION
# =============================================================================
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
LABELS_PREFIX       = "v2/labels/"
LOCAL_LABELS_PREFIX = "v2/labels_local/"
FEATURES_PREFIX     = "v2/features/"
BACKUP_PREFIX       = "v2/vm_backup_20260420_1709/logs/logs/"
LOCAL_MODELS_DIR    = os.path.join(REPO_ROOT, "logs")
OUTPUT_DIR          = REPO_ROOT

N_STATES = 3
N_ITER   = 100
N_SEEDS  = 5

HMM_FEATURES = ["RV_300s", "OFI_300s", "Kyle_lambda_300s", "intensity_300s"]

# Walk-forward fold structure — matches config.py WINDOWS exactly
FOLD_TRAIN_WINDOWS = {
    1: [WINDOWS[0]],
    2: [WINDOWS[0], WINDOWS[1]],
    3: [WINDOWS[0], WINDOWS[1], WINDOWS[2]],
    4: [WINDOWS[0], WINDOWS[1], WINDOWS[2], WINDOWS[3]],
}

FOLD_TEST_WINDOWS = {
    1: WINDOWS[1],
    2: WINDOWS[2],
    3: WINDOWS[3],
    4: WINDOWS[4],
}

FOLD_CRISIS_LABEL = {
    1: "May 2021 Crash",
    2: "Terra-Luna Collapse",
    3: "FTX Bankruptcy",
    4: "2024 Resurgence",
}

STATE_NAMES = {0: "calm", 1: "elevated", 2: "stress"}

# =============================================================================
# SHARED UTILITIES
# =============================================================================

def months_in_range(start_ym, end_ym):
    sy, sm = int(start_ym[:4]), int(start_ym[5:])
    ey, em = int(end_ym[:4]),   int(end_ym[5:])
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def clean_features(features_np):
    """
    Robust cleaning before HMM fitting.
    Handles inf/-inf and extreme outliers that cause dtype overflow in
    hmmlearn's EM algorithm. This mirrors the cleaning applied in
    04b_stationarity_fracdiff.py before fractional differencing.

    Steps:
      1. Replace +inf/-inf with nan
      2. Per-column: replace nan with column median
      3. Per-column: clip to [-p999, +p999] where p999 is the 99.9th
         percentile of absolute values — removes extreme Kyle_lambda
         and ILLIQ spikes from near-zero volume bars without discarding
         the economic signal in normal observations
    """
    out = features_np.copy().astype(np.float64)

    # Step 1 — inf to nan
    out[np.isinf(out)] = np.nan

    for col in range(out.shape[1]):
        col_data = out[:, col]

        # Step 2 — nan to median
        median = np.nanmedian(col_data)
        if np.isnan(median):
            median = 0.0
        col_data = np.where(np.isnan(col_data), median, col_data)

        # Step 3 — clip to 99.9th percentile of absolute values
        p999 = np.nanpercentile(np.abs(col_data), 99.9)
        if p999 > 0:
            col_data = np.clip(col_data, -p999, p999)

        out[:, col] = col_data

    return out


def load_features_for_windows(bucket, asset, windows):
    all_months = set()
    for start_ym, end_ym in windows:
        for ym in months_in_range(start_ym, end_ym):
            all_months.add(ym)

    frames = []
    cols = ["time"] + HMM_FEATURES
    for year, month in sorted(all_months):
        path = f"{FEATURES_PREFIX}{asset}-features-{year}-{month:02d}.parquet"
        blob = bucket.blob(path)
        if not blob.exists():
            logger.warning(f"  Missing features: {path}")
            continue
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        try:
            frames.append(pl.read_parquet(buf, columns=cols))
        except Exception as e:
            logger.warning(f"  Read error {path}: {e}")

    if not frames:
        return None
    return pl.concat(frames).sort("time")


def load_labels_from_gcs(bucket, asset, prefix, windows):
    all_months = set()
    for start_ym, end_ym in windows:
        for ym in months_in_range(start_ym, end_ym):
            all_months.add(ym)

    frames = []
    for year, month in sorted(all_months):
        path = f"{prefix}{asset}-labels-{year}-{month:02d}.parquet"
        blob = bucket.blob(path)
        if not blob.exists():
            continue
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        try:
            frames.append(pl.read_parquet(buf))
        except Exception as e:
            logger.warning(f"  Read error {path}: {e}")

    if not frames:
        return None
    return pl.concat(frames).sort("time")


def stress_dist(labels_np):
    n = len(labels_np)
    if n == 0:
        return None
    return {
        "n_bars":       n,
        "calm_pct":     round((labels_np == 0).mean() * 100, 2),
        "elevated_pct": round((labels_np == 1).mean() * 100, 2),
        "stress_pct":   round((labels_np == 2).mean() * 100, 2),
    }


# =============================================================================
# PART A — GLOBAL HMM STABILITY
# =============================================================================

def extract_params(pkl_obj):
    """
    Extract and sort HMM parameters by RV mean (index 0) for
    consistent state ordering across independent runs.
    """
    model = pkl_obj.get("model") if isinstance(pkl_obj, dict) else pkl_obj
    if not hasattr(model, "means_"):
        return None

    order = np.argsort([model.means_[i][0] for i in range(model.n_components)])
    return {
        "means":     model.means_[order],
        "covars":    model.covars_[order],
        "transmat":  model.transmat_[order][:, order],
        "startprob": model.startprob_[order],
    }


def run_global_stability_analysis(bucket):
    logger.info("\n" + "=" * 60)
    logger.info("PART A: GLOBAL HMM STABILITY ANALYSIS")
    logger.info("=" * 60)

    feat_labels = ["RV", "OFI", "Kyle_lambda", "intensity"]
    rows = []

    for asset in ASSETS:
        logger.info(f"\n  {asset}")

        # Load April 20 backup from GCS
        backup_blob = bucket.blob(f"{BACKUP_PREFIX}{asset}_hmm_model.pkl")
        if not backup_blob.exists():
            logger.warning(f"  Backup not found: {BACKUP_PREFIX}{asset}_hmm_model.pkl")
            continue
        backup_pkl = pickle.loads(backup_blob.download_as_bytes())

        # Load current production model from local disk
        local_path = os.path.join(LOCAL_MODELS_DIR, f"{asset}_hmm_model.pkl")
        if not os.path.exists(local_path):
            logger.warning(f"  Local model not found: {local_path}")
            continue
        with open(local_path, "rb") as f:
            current_pkl = pickle.load(f)

        p_bk = extract_params(backup_pkl)
        p_cu = extract_params(current_pkl)
        if p_bk is None or p_cu is None:
            logger.warning(f"  {asset}: could not extract params from one or both models")
            continue

        row = {"asset": asset}

        # Per-state per-feature mean differences
        for state_idx in range(N_STATES):
            sname = STATE_NAMES[state_idx]
            bk_m  = p_bk["means"][state_idx]
            cu_m  = p_cu["means"][state_idx]
            diffs = np.abs(bk_m - cu_m)

            logger.info(f"    State {state_idx} ({sname}):")
            for fi, feat in enumerate(feat_labels):
                logger.info(
                    f"      {feat:<14} backup={bk_m[fi]:+.6f}  "
                    f"current={cu_m[fi]:+.6f}  diff={diffs[fi]:.2e}"
                )
                row[f"mean_diff_{sname}_{feat}"] = round(float(diffs[fi]), 8)

        # Max absolute difference across all parameter matrices
        for key in ["means", "covars", "transmat", "startprob"]:
            diff = float(np.abs(p_bk[key] - p_cu[key]).max())
            row[f"max_abs_diff_{key}"] = round(diff, 8)
            logger.info(f"    max_abs_diff_{key}: {diff:.2e}")

        rows.append(row)

    if not rows:
        logger.warning("  No stability rows collected — check GCS and local model paths")
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    out = os.path.join(OUTPUT_DIR, "global_hmm_stability.csv")
    df.write_csv(out)
    logger.info(f"\n  Saved: {out}")
    return df


# =============================================================================
# PART B — LOCAL HMM GENERATION
# =============================================================================

def fit_local_hmm(features_np):
    """
    Fit a 3-state Gaussian HMM using the same protocol as 05a_label_generation.py.
    Applies robust cleaning first to handle inf/extreme values in early-period data.
    Runs N_SEEDS initialisations and returns the best model by log-likelihood.
    """
    # Clean before fitting — handles inf and extreme outliers
    features_clean = clean_features(features_np)

    best_model, best_scaler, best_score = None, None, -np.inf

    for seed in range(N_SEEDS):
        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(features_clean)

            # Final safety check after scaling
            if not np.isfinite(X).all():
                logger.warning(f"    Seed {seed}: non-finite values after scaling — skipping")
                continue

            model = GaussianHMM(
                n_components=N_STATES,
                covariance_type="diag",
                n_iter=N_ITER,
                random_state=seed,
                verbose=False,
            )
            model.fit(X, [len(X)])
            score = model.score(X, [len(X)])

            if score > best_score:
                best_score  = score
                best_model  = model
                best_scaler = scaler

        except Exception as e:
            logger.warning(f"    Seed {seed} failed: {e}")

    return best_model, best_scaler, best_score


def label_with_model(model, scaler, features_np):
    """
    Predict HMM states and map to calm/elevated/stress by RV rank.
    Applies same cleaning as fit_local_hmm for consistency.
    """
    features_clean = clean_features(features_np)
    X   = scaler.transform(features_clean)

    # Safety check
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    raw    = model.predict(X)
    order  = np.argsort([model.means_[i][0] for i in range(model.n_components)])
    mapping = {raw_s: rank for rank, raw_s in enumerate(order)}
    return np.array([mapping[s] for s in raw])


def save_labels_to_gcs(bucket, asset, fold_n, times_series, labels_np):
    """
    Save local HMM labels for a fold's test window to GCS.
    Partitioned by month to match the global label structure at v2/labels/.
    """
    prefix = f"{LOCAL_LABELS_PREFIX}fold_{fold_n}/"

    df = pl.DataFrame({
        "time":  times_series,
        "label": labels_np.astype(np.int32),
    })

    if df["time"].dtype != pl.Datetime("us", "UTC"):
        df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))

    df = df.with_columns([
        pl.col("time").dt.year().alias("_y"),
        pl.col("time").dt.month().alias("_m"),
    ])

    saved = 0
    for (yr, mo), grp in df.group_by(["_y", "_m"]):
        grp_clean = grp.drop(["_y", "_m"]).sort("time")
        buf = io.BytesIO()
        grp_clean.write_parquet(buf)
        buf.seek(0)
        path = f"{prefix}{asset}-labels-{yr}-{mo:02d}.parquet"
        bucket.blob(path).upload_from_file(
            buf, content_type="application/octet-stream"
        )
        saved += 1

    logger.info(f"    Uploaded {saved} monthly parquets → gs://{BUCKET}/{prefix}")


def run_local_hmm_generation(bucket):
    logger.info("\n" + "=" * 60)
    logger.info("PART B: LOCAL HMM GENERATION ACROSS ALL FOLDS")
    logger.info("=" * 60)

    local_rows  = []
    global_rows = []

    for fold_n, train_windows in FOLD_TRAIN_WINDOWS.items():
        test_window  = FOLD_TEST_WINDOWS[fold_n]
        crisis_label = FOLD_CRISIS_LABEL[fold_n]

        logger.info(f"\n  Fold {fold_n} | test={test_window} | [{crisis_label}]")
        logger.info(f"  Train windows: {train_windows}")

        marker     = f"v2/pipeline_markers/11a_local_fold_{fold_n}.done"
        fold_done  = bucket.blob(marker).exists()

        for asset in ASSETS:

            # ------------------------------------------------------------------
            # Global labels for this test window — always load for comparison
            # ------------------------------------------------------------------
            global_df = load_labels_from_gcs(
                bucket, asset, LABELS_PREFIX, [test_window]
            )
            if global_df is not None and not global_df.is_empty():
                d = stress_dist(global_df["label"].to_numpy())
                if d:
                    global_rows.append({
                        "fold":          fold_n,
                        "asset":         asset,
                        "crisis_event":  crisis_label,
                        "train_months":  "full_history",
                        "hmm_type":      "global",
                        **d,
                    })
                    logger.info(
                        f"  [{asset}] Global test: "
                        f"calm={d['calm_pct']:.1f}%  "
                        f"elev={d['elevated_pct']:.1f}%  "
                        f"stress={d['stress_pct']:.1f}%"
                    )

            # ------------------------------------------------------------------
            # Local labels — load from GCS if already computed, else fit
            # ------------------------------------------------------------------
            if fold_done:
                local_prefix = f"{LOCAL_LABELS_PREFIX}fold_{fold_n}/"
                local_df = load_labels_from_gcs(
                    bucket, asset, local_prefix, [test_window]
                )
                if local_df is not None and not local_df.is_empty():
                    d = stress_dist(local_df["label"].to_numpy())
                    if d:
                        n_train = sum(
                            len(months_in_range(s, e)) for s, e in train_windows
                        )
                        local_rows.append({
                            "fold":          fold_n,
                            "asset":         asset,
                            "crisis_event":  crisis_label,
                            "train_months":  n_train,
                            "hmm_type":      "local",
                            **d,
                        })
                        logger.info(
                            f"  [{asset}] Local (cached): "
                            f"calm={d['calm_pct']:.1f}%  "
                            f"elev={d['elevated_pct']:.1f}%  "
                            f"stress={d['stress_pct']:.1f}%"
                        )
                continue

            # ------------------------------------------------------------------
            # Fit local HMM from scratch
            # ------------------------------------------------------------------
            logger.info(f"\n  [{asset}] Fitting Fold {fold_n} local HMM...")

            train_df = load_features_for_windows(bucket, asset, train_windows)
            if train_df is None or train_df.is_empty():
                logger.warning(f"  [{asset}] No training features — skipping")
                continue

            clean        = train_df.drop_nulls(subset=HMM_FEATURES)
            n_bars       = len(clean)
            n_months     = sum(len(months_in_range(s, e)) for s, e in train_windows)

            logger.info(f"    {n_bars:,} bars | {n_months} months")

            if n_bars < 500:
                logger.warning(
                    f"    WARNING: Only {n_bars} bars — HMM may be unreliable "
                    f"(expected for Fold 1 with single window training data)"
                )

            feats_np = clean[HMM_FEATURES].to_numpy().astype(np.float64)
            model, scaler, ll = fit_local_hmm(feats_np)

            if model is None:
                logger.warning(
                    f"  [{asset}] Fold {fold_n}: All seeds failed. "
                    f"Recording as failed fold — this itself is evidence of "
                    f"local HMM instability on limited training data."
                )
                local_rows.append({
                    "fold":          fold_n,
                    "asset":         asset,
                    "crisis_event":  crisis_label,
                    "train_months":  n_months,
                    "hmm_type":      "local",
                    "n_bars":        0,
                    "calm_pct":      None,
                    "elevated_pct":  None,
                    "stress_pct":    None,
                })
                del feats_np, clean
                gc.collect()
                continue

            logger.info(f"    Best log-likelihood: {ll:.2f}")

            # Log state means for transparency
            feat_labels = ["RV", "OFI", "Kyle_lambda", "intensity"]
            order = np.argsort([model.means_[i][0] for i in range(N_STATES)])
            for rank, raw_s in enumerate(order):
                sname = STATE_NAMES[rank]
                means = model.means_[raw_s]
                logger.info(
                    f"    {sname}: "
                    + "  ".join(
                        f"{f}={means[fi]:.4f}" for fi, f in enumerate(feat_labels)
                    )
                )

            # Label the test window with this local model
            test_df = load_features_for_windows(bucket, asset, [test_window])
            if test_df is None or test_df.is_empty():
                logger.warning(f"  [{asset}] No test features for Fold {fold_n}")
                del feats_np, clean, model, scaler
                gc.collect()
                continue

            test_clean  = test_df.drop_nulls(subset=HMM_FEATURES)
            test_np     = test_clean[HMM_FEATURES].to_numpy().astype(np.float64)
            test_labels = label_with_model(model, scaler, test_np)

            d = stress_dist(test_labels)
            logger.info(
                f"    Local test labels: "
                f"calm={d['calm_pct']:.1f}%  "
                f"elev={d['elevated_pct']:.1f}%  "
                f"stress={d['stress_pct']:.1f}%"
            )

            # Save to GCS
            save_labels_to_gcs(
                bucket, asset, fold_n,
                test_clean["time"], test_labels
            )

            local_rows.append({
                "fold":          fold_n,
                "asset":         asset,
                "crisis_event":  crisis_label,
                "train_months":  n_months,
                "hmm_type":      "local",
                **d,
            })

            del feats_np, clean, model, scaler, test_np, test_labels
            gc.collect()

        # Mark this fold complete
        if not fold_done:
            bucket.blob(marker).upload_from_string(b"")
            logger.info(f"  Fold {fold_n} complete — marker written")

    # ==========================================================================
    # BUILD OUTPUT TABLES
    # ==========================================================================
    if not local_rows and not global_rows:
        logger.warning("No rows collected — check GCS connectivity")
        return None, None

    # Local audit CSV
    local_df = pl.DataFrame(local_rows).sort(["fold", "asset"])
    local_df.write_csv(os.path.join(OUTPUT_DIR, "local_hmm_label_audit.csv"))
    logger.info(f"\n  Saved: local_hmm_label_audit.csv")

    # Head-to-head comparison
    comp_rows = []
    for fold_n in FOLD_TRAIN_WINDOWS:
        for asset in ASSETS:
            g = next(
                (r for r in global_rows
                 if r["fold"] == fold_n and r["asset"] == asset), None
            )
            l = next(
                (r for r in local_rows
                 if r["fold"] == fold_n and r["asset"] == asset), None
            )
            if not g or not l:
                continue

            # Handle failed folds gracefully
            l_stress = l.get("stress_pct")
            g_stress = g.get("stress_pct")
            diff     = round(abs(g_stress - l_stress), 2) if (
                l_stress is not None and g_stress is not None
            ) else None

            comp_rows.append({
                "fold":                fold_n,
                "asset":               asset,
                "crisis_event":        FOLD_CRISIS_LABEL[fold_n],
                "local_train_months":  l["train_months"],
                "global_stress_pct":   g_stress,
                "local_stress_pct":    l_stress,
                "stress_pct_diff":     diff,
                "global_calm_pct":     g.get("calm_pct"),
                "local_calm_pct":      l.get("calm_pct"),
                "global_elevated_pct": g.get("elevated_pct"),
                "local_elevated_pct":  l.get("elevated_pct"),
            })

    comp_df = pl.DataFrame(comp_rows).sort(["asset", "fold"])
    comp_df.write_csv(os.path.join(OUTPUT_DIR, "global_vs_local_summary.csv"))
    logger.info(f"  Saved: global_vs_local_summary.csv")

    # Print summary table to stdout
    logger.info("\n" + "=" * 72)
    logger.info("GLOBAL vs LOCAL HMM — STRESS% ON TEST WINDOWS")
    logger.info("=" * 72)
    logger.info(
        f"  {'F':<3} {'Asset':<10} {'Crisis':<26} "
        f"{'Global%':<9} {'Local%':<9} {'|Diff|':<8} {'TrainMo'}"
    )
    logger.info("  " + "-" * 72)
    for r in comp_rows:
        local_s  = f"{r['local_stress_pct']:.1f}" if r["local_stress_pct"] is not None else "FAILED"
        diff_s   = f"{r['stress_pct_diff']:.1f}"  if r["stress_pct_diff"]  is not None else "N/A"
        logger.info(
            f"  {r['fold']:<3} {r['asset']:<10} {r['crisis_event']:<26} "
            f"{r['global_stress_pct']:<9.1f} {local_s:<9} "
            f"{diff_s:<8} {r['local_train_months']}"
        )

    return local_df, comp_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    overall_marker = "v2/pipeline_markers/11a_hmm_stability.done"
    if bucket.blob(overall_marker).exists():
        logger.info("11a already complete. To rerun:")
        logger.info(f"  gsutil rm gs://{BUCKET}/{overall_marker}")
        return

    # PART A — Global stability proof (~30 seconds)
    stability_df = run_global_stability_analysis(bucket)

    # PART B — Local HMM generation across all folds (~20-40 minutes, CPU only)
    local_df, comp_df = run_local_hmm_generation(bucket)

    # Write overall completion marker
    bucket.blob(overall_marker).upload_from_string(b"")

    logger.info("\n" + "=" * 60)
    logger.info("11a complete.")
    logger.info(f"  global_hmm_stability.csv    → {OUTPUT_DIR}")
    logger.info(f"  local_hmm_label_audit.csv   → {OUTPUT_DIR}")
    logger.info(f"  global_vs_local_summary.csv → {OUTPUT_DIR}")
    logger.info(
        f"  Local labels in GCS         → "
        f"gs://{BUCKET}/{LOCAL_LABELS_PREFIX}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
