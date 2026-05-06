"""
08_generate_paper_figures.py

Generates all publication-quality figures for the MAIC paper.
Saves locally and uploads to GCS.

Output structure:
  /workspace/maic/paper_figures/
    baseline/     <- 06b: asset-specific vs pooled comparison
    ablation/     <- 06c: fracdiff impact
    production/   <- 06d: main paper figures (Fig 1-5)
    summary/      <- seed stability summary

GCS: gs://fe-binance-data-2025/v2/paper_figures/{subfolder}/
"""

import os
import sys
import json
import pickle
import numpy as np
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import roc_curve, auc
from google.cloud import storage
import logging

sys.path.insert(0, "/workspace/maic")
from config import ASSETS, BUCKET, WINDOWS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = "/workspace/maic/paper_figures"
DIRS = {
    "baseline":   os.path.join(BASE_DIR, "baseline"),
    "ablation":   os.path.join(BASE_DIR, "ablation"),
    "production": os.path.join(BASE_DIR, "production"),
    "summary":    os.path.join(BASE_DIR, "summary"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

SEED         = 42
PROD_PREFIX  = f"v2/results_production/seed_{SEED}/pooled/fold_4/"
FEAT_PREFIX  = "v2/features_fracdiff/"
LABEL_PREFIX = "v2/labels/"
DATA_CACHE   = "/workspace/maic/data_cache_production"
PROD_CSV     = "/workspace/maic/production_results.csv"
ABL_CSV      = "/workspace/maic/ablation_results.csv"
BASE_CSV     = "/workspace/maic/baseline_results.csv"
GCS_PREFIX   = "v2/paper_figures"

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
})

CLASS_COLORS = {0: "#2166ac", 1: "#f59b42", 2: "#c32f27"}
CLASS_NAMES  = {0: "Calm", 1: "Elevated", 2: "Stress"}

MODEL_COLORS = {
    "lr": "#4dac26", "rf": "#7b3294", "xgb": "#d01c8b",
    "lstm": "#0571b0", "cnn_gaf": "#f4a582",
}
MODEL_LABELS = {
    "lr": "LR", "rf": "RF", "xgb": "XGBoost",
    "lstm": "LSTM", "cnn_gaf": "CNN-GAF",
}
MODEL_LINES = {
    "lr": "--", "rf": "-.", "xgb": "-",
    "lstm": ":", "cnn_gaf": (0, (3, 1, 1, 1)),
}

FEAT_CLEAN = {
    "RV_300s":          "Realised Volatility (300s)",
    "Kyle_lambda_300s": "Kyle's λ (300s)",
    "intensity_300s":   "Trade Intensity (300s)",
    "asset_id":         "Asset Identifier",
    "VWAP_300s":        "VWAP (300s)",
    "OFI_300s":         "Order Flow Imbalance (300s)",
    "RV_60s":           "Realised Volatility (60s)",
    "price":            "Price (frac. diff.)",
    "ILLIQ_300s":       "ILLIQ (300s)",
    "VWAP_60s":         "VWAP (60s)",
    "VWAP_10s":         "VWAP (10s)",
    "TCI_300s":         "Trade Concentration (300s)",
    "intensity_60s":    "Trade Intensity (60s)",
    "ILLIQ_60s":        "ILLIQ (60s)",
    "ILLIQ_10s":        "ILLIQ (10s)",
    "Kyle_lambda_60s":  "Kyle's λ (60s)",
    "TCI_60s":          "Trade Concentration (60s)",
    "volume":           "Volume",
    "rv":               "RV (raw)",
    "intensity_10s":    "Trade Intensity (10s)",
}

gcs_client = storage.Client()
gcs_bucket = gcs_client.bucket(BUCKET)


def gcs_download(gcs_path, local_path, quiet=False):
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    gcs_bucket.blob(gcs_path).download_to_filename(local_path)
    if not quiet:
        log.info(f"  Downloaded: {os.path.basename(gcs_path)}")
    return local_path


def gcs_upload(local_path, subfolder):
    fname    = os.path.basename(local_path)
    gcs_path = f"{GCS_PREFIX}/{subfolder}/{fname}"
    gcs_bucket.blob(gcs_path).upload_from_filename(local_path)
    log.info(f"  Uploaded: {gcs_path}")


def gcs_read_json(gcs_path):
    return json.loads(gcs_bucket.blob(gcs_path).download_as_text())


def save_fig(path, subfolder):
    plt.savefig(path)
    plt.close()
    log.info(f"  Saved: {os.path.basename(path)}")
    gcs_upload(path, subfolder)


def load_model(gcs_path, local_name):
    local = f"{DATA_CACHE}/{local_name}"
    if not os.path.exists(local):
        gcs_download(gcs_path, local)
    with open(local, "rb") as f:
        obj = pickle.load(f)
    return obj.get("model", obj) if isinstance(obj, dict) else obj


def load_fold4_test_data():
    log.info("Loading fold 4 test data...")
    start_ym, end_ym = WINDOWS[4]

    def parse_months(s, e):
        months = []
        sy, sm = int(s[:4]), int(s[5:])
        ey, em = int(e[:4]), int(e[5:])
        y, m = sy, sm
        while (y, m) <= (ey, em):
            months.append((y, m))
            m += 1
            if m > 12:
                m, y = 1, y + 1
        return months

    frames = []
    for asset in ASSETS:
        for year, month in parse_months(start_ym, end_ym):
            lf = f"{DATA_CACHE}/feat_{asset}_{year}_{month:02d}.parquet"
            ll = f"{DATA_CACHE}/label_{asset}_{year}_{month:02d}.parquet"
            if not os.path.exists(lf):
                try:
                    gcs_download(f"{FEAT_PREFIX}{asset}-features-{year}-{month:02d}.parquet", lf, quiet=True)
                except Exception as e:
                    log.warning(f"  Missing feat {asset} {year}-{month:02d}: {e}")
                    continue
            if not os.path.exists(ll):
                try:
                    gcs_download(f"{LABEL_PREFIX}{asset}-labels-{year}-{month:02d}.parquet", ll, quiet=True)
                except Exception as e:
                    log.warning(f"  Missing label {asset} {year}-{month:02d}: {e}")
                    continue
            try:
                df = (pl.read_parquet(lf)
                      .join(pl.read_parquet(ll), on="time", how="inner")
                      .with_columns(pl.lit(asset).alias("asset_id")))
                frames.append(df)
            except Exception as e:
                log.warning(f"  Read error {asset} {year}-{month:02d}: {e}")

    if not frames:
        raise RuntimeError("No data loaded — check GCS paths and cache")

    common_set = set(frames[0].columns)
    for f in frames[1:]:
        common_set &= set(f.columns)
    
    common_cols = [c for c in frames[0].columns if c in common_set]
    
    aligned_frames = []
    for f in frames:
        aligned_frames.append(f.select(common_cols))
        
    df = pl.concat(aligned_frames).sort("time")
    log.info(f"  {df.shape[0]:,} rows processed")
    return df


def prepare_features(df):
    EXCLUDE = {"time", "label", "asset_id"}
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    le = LabelEncoder()
    asset_enc = le.fit_transform(df["asset_id"].to_numpy())
    X = np.column_stack([df.select(feat_cols).to_numpy(), asset_enc])
    feat_names = feat_cols + ["asset_id"]
    y = df["label"].to_numpy()
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    return X_sc, y, feat_names


def plot_shap_multiclass_bar(shap_values, feat_names, title, save_path, subfolder, top_n=12):
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
        mean_abs  = np.array([np.abs(shap_values[c]).mean(0) for c in range(n_classes)])
    elif len(np.shape(shap_values)) == 3:
        n_classes = shap_values.shape[2]
        mean_abs  = np.abs(shap_values).mean(axis=0).T
    else:
        n_classes = 1
        mean_abs  = np.abs(shap_values).mean(0, keepdims=True)

    top_idx    = np.argsort(mean_abs.sum(0))[-top_n:][::-1]
    top_names  = [FEAT_CLEAN.get(feat_names[int(i)], feat_names[int(i)]) for i in top_idx]
    top_values = mean_abs[:, top_idx]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    y_pos = np.arange(len(top_names))
    left  = np.zeros(len(top_names))

    for c in range(n_classes):
        ax.barh(y_pos, top_values[c], left=left,
                color=CLASS_COLORS.get(c, "#888"),
                label=CLASS_NAMES.get(c, f"Class {c}"),
                alpha=0.88, height=0.62)
        left += top_values[c]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean SHAP value (average impact on model output)")
    ax.set_title(title, pad=6)
    if n_classes > 1:
        ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    save_fig(save_path, subfolder)


def plot_shap_binary_bar(shap_values, feat_names, title, save_path, subfolder, top_n=12):
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(np.shape(shap_values)) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
        
    mean_abs = np.abs(sv).mean(0)
    top_idx  = np.argsort(mean_abs)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, mean_abs[top_idx], color=CLASS_COLORS[2], alpha=0.85, height=0.62)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEAT_CLEAN.get(feat_names[int(i)], feat_names[int(i)]) for i in top_idx], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean SHAP value (impact on Stress class prediction)")
    ax.set_title(title, pad=6)
    plt.tight_layout()
    save_fig(save_path, subfolder)


def generate_production_figures(X_sc, y, feat_names):
    log.info("Generating production figures (06d)...")
    prod = DIRS["production"]

    log.info("  Fig 1: SHAP XGBoost multiclass...")
    xgb_mc  = load_model(f"{PROD_PREFIX}models/xgb_multiclass.pkl", "xgb_multiclass.pkl")
    np.random.seed(SEED)
    idx     = np.random.choice(len(X_sc), size=min(2000, len(X_sc)), replace=False)
    X_samp  = X_sc[idx]
    shap_mc = shap.TreeExplainer(xgb_mc).shap_values(X_samp)
    plot_shap_multiclass_bar(
        shap_mc, feat_names,
        "XGBoost Feature Attribution by Regime\n(Fold 4, Pooled, Seed 42)",
        os.path.join(prod, "fig1_shap_xgb_multiclass.png"), "production"
    )

    log.info("  Fig 1b: SHAP XGBoost binary...")
    xgb_bin  = load_model(f"{PROD_PREFIX}models/xgb_binary.pkl", "xgb_binary.pkl")
    shap_bin = shap.TreeExplainer(xgb_bin).shap_values(X_samp)
    plot_shap_binary_bar(
        shap_bin, feat_names,
        "XGBoost Feature Attribution — Binary Stress Detection\n(Fold 4, Pooled, Seed 42)",
        os.path.join(prod, "fig1b_shap_xgb_binary.png"), "production"
    )

    log.info("  Fig 2: Fold progression...")
    df_prod   = pd.read_csv(PROD_CSV)
    fold_prog = df_prod.groupby(["model", "fold", "mode"])["f1_weighted_avg"].mean().reset_index()
    fold_sizes = {1: "3.1M", 2: "8.6M", 3: "14.1M", 4: "18.8M"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mode, title in zip(
        axes,
        ["multiclass", "binary"],
        ["Multiclass (Calm / Elevated / Stress)", "Binary (Stress vs. Non-Stress)"]
    ):
        sub = fold_prog[fold_prog["mode"] == mode]
        for m in ["lr", "rf", "xgb", "lstm", "cnn_gaf"]:
            row = sub[sub["model"] == m].sort_values("fold")
            if row.empty:
                continue
            ax.plot(row["fold"].values, row["f1_weighted_avg"].values,
                    color=MODEL_COLORS[m], linestyle=MODEL_LINES[m],
                    linewidth=1.8, marker="o", markersize=5, label=MODEL_LABELS[m])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([f"F{i}\n({fold_sizes[i]})" for i in [1, 2, 3, 4]], fontsize=8.5)
        ax.set_xlabel("Fold (training rows)")
        ax.set_ylabel("Weighted F1")
        ax.set_title(title, pad=6)
        ax.set_ylim(0.1, 1.02)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("Fold Progression: Weighted F1 vs Training Data Size\n(Averaged across 5 seeds)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(os.path.join(prod, "fig2_fold_progression.png"), "production")

    log.info("  Fig 3: SHAP RF binary...")
    class_counts = np.bincount(y)
    w_map  = {c: len(y) / (len(class_counts) * class_counts[c]) for c in range(len(class_counts))}
    probs  = np.array([w_map[int(c)] for c in y])
    probs /= probs.sum()
    rf_idx = np.random.choice(len(X_sc), size=min(50000, len(X_sc)), replace=True, p=probs)
    surrogate = SklearnRF(n_estimators=50, max_depth=8, random_state=SEED, n_jobs=-1)
    surrogate.fit(X_sc[rf_idx], (y[rf_idx] >= 2).astype(int))
    shap_rf = shap.TreeExplainer(surrogate).shap_values(X_samp)
    plot_shap_binary_bar(
        shap_rf, feat_names,
        "Random Forest Feature Attribution — Binary Stress Detection\n"
        "(Fold 4, Pooled, sklearn Surrogate, Seed 42)",
        os.path.join(prod, "fig3_shap_rf_binary.png"), "production"
    )

    log.info("  Fig 4: LSTM Integrated Gradients...")
    try:
        ig_dict = gcs_read_json(f"{PROD_PREFIX}ig_lstm_multiclass.json")
        feat_ig = list(ig_dict.keys())
        feat_imp = np.array(list(ig_dict.values()), dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4.8))
        top_n   = min(12, len(feat_imp))
        top_idx = np.argsort(feat_imp)[-top_n:][::-1]
        
        ax.barh(np.arange(top_n), feat_imp[top_idx],
                     color=CLASS_COLORS[0], alpha=0.85, height=0.62)
        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels(
            [FEAT_CLEAN.get(feat_ig[i], feat_ig[i]) for i in top_idx], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Integrated Gradient (Globally Aggregated)")
        ax.set_title("LSTM Feature Attribution — Multiclass, Fold 4, Pooled", pad=6, fontsize=11, fontweight="bold")
        plt.tight_layout()
        save_fig(os.path.join(prod, "fig4_ig_lstm_multiclass.png"), "production")
    except Exception as e:
        log.warning(f"  Fig 4 failed: {e}")

    log.info("  Fig 5: ROC curves binary...")
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random (AUC=0.50)")
        for m in ["lr", "rf", "xgb", "lstm", "cnn_gaf"]:
            try:
                pq = f"v2/results_production/seed_{SEED}/pooled/fold_4/stage/binary_{m}_probs.parquet"
                lp = f"{DATA_CACHE}/roc_{m}.parquet"
                if not os.path.exists(lp):
                    gcs_download(pq, lp, quiet=True)
                probs_df   = pl.read_parquet(lp)
                stress_col = next((c for c in probs_df.columns
                                   if "prob_2" in c or "stress" in c.lower()), probs_df.columns[-1])
                y_score = probs_df[stress_col].to_numpy()
                y_true  = (y >= 2).astype(int)[:len(y_score)]
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc     = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=MODEL_COLORS[m], linestyle=MODEL_LINES[m],
                        linewidth=1.8, label=f"{MODEL_LABELS[m]}  (AUC={roc_auc:.3f})")
            except Exception as e:
                log.warning(f"  ROC {m}: {e}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Binary Stress Classification\n(Fold 4, Pooled, Seed 42)", pad=6)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        plt.tight_layout()
        save_fig(os.path.join(prod, "fig5_roc_binary.png"), "production")
    except Exception as e:
        log.warning(f"  Fig 5 failed: {e}")


def generate_ablation_figures():
    log.info("Generating ablation figures (06c)...")
    if not os.path.exists(ABL_CSV):
        log.warning(f"  {ABL_CSV} not found — skipping")
        return
    try:
        df     = pd.read_csv(ABL_CSV)
        models = ["lr", "rf", "xgb", "lstm", "cnn_gaf"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, mode in zip(axes, ["multiclass", "binary"]):
            sub  = df[df["mode"] == mode]
            ctrl = sub[sub["condition"].str.contains("control", case=False)].set_index("model")
            frac = sub[sub["condition"].str.contains("fracdiff|experiment", case=False)].set_index("model")
            x    = np.arange(len(models))
            w    = 0.35
            cv   = [ctrl.loc[m, "f1_stress"] if m in ctrl.index else 0 for m in models]
            fv   = [frac.loc[m, "f1_stress"] if m in frac.index else 0 for m in models]
            ax.bar(x - w/2, cv, w, color="#4393c3", alpha=0.85, label="Raw price (control)")
            ax.bar(x + w/2, fv, w, color="#d6604d", alpha=0.85, label="FracDiff (experiment)")
            for i, (c, f) in enumerate(zip(cv, fv)):
                delta = f - c
                col   = "#2ca02c" if delta > 0.002 else ("#d62728" if delta < -0.002 else "#888")
                ax.text(i, max(c, f) + 0.015, f"{delta:+.3f}",
                        ha="center", fontsize=7.5, color=col, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=9)
            ax.set_ylabel("Stress-Class F1")
            ax.set_ylim(0, 1.12)
            ax.set_title(f"{'Multiclass' if mode=='multiclass' else 'Binary'}", pad=6)
            if mode == "multiclass":
                ax.legend(fontsize=8.5, framealpha=0.9)

        fig.suptitle("Ablation: Impact of Fractional Differencing on Stress-Class F1\n(Fold 4, Pooled)",
                     fontsize=11, fontweight="bold", y=1.01)
        plt.tight_layout()
        save_fig(os.path.join(DIRS["ablation"], "ablation_fracdiff_stress_f1.png"), "ablation")
    except Exception as e:
        log.warning(f"  Ablation figures failed: {e}")


def generate_baseline_figures():
    log.info("Generating baseline figures (06b)...")
    if not os.path.exists(BASE_CSV):
        log.warning(f"  {BASE_CSV} not found — skipping")
        return
    try:
        df   = pd.read_csv(BASE_CSV)
        xgb  = df[df["model"] == "xgb"].copy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for ax, mode in zip(axes, ["multiclass", "binary"]):
            sub = xgb[xgb["mode"] == mode]
            for vals_filter, label, color, ls in [
                (lambda s: s[s["is_pooled"] == True], "Pooled",  "#d01c8b", "-"),
                (lambda s: s[(s["is_pooled"] == False) & (s["asset"] == "BTCUSDT")], "BTC", "#2166ac", "--"),
                (lambda s: s[(s["is_pooled"] == False) & (s["asset"] == "ETHUSDT")], "ETH", "#4dac26", "-."),
                (lambda s: s[(s["is_pooled"] == False) & (s["asset"] == "SOLUSDT")], "SOL", "#f59b42", ":"),
            ]:
                row = vals_filter(sub).groupby("fold")["f1_weighted_avg"].mean()
                if not row.empty:
                    ax.plot(row.index, row.values, color=color, linestyle=ls,
                            linewidth=1.8, marker="o", markersize=5, label=label)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels([f"Fold {i}" for i in [1, 2, 3, 4]], fontsize=8.5)
            ax.set_xlabel("Fold")
            ax.set_ylabel("Weighted F1")
            ax.set_title(f"{'Multiclass' if mode=='multiclass' else 'Binary'}", pad=6)
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
            ax.set_ylim(0.3, 1.02)

        fig.suptitle("XGBoost: Asset-Specific vs Pooled Performance (Baseline)",
                     fontsize=11, fontweight="bold", y=1.01)
        plt.tight_layout()
        save_fig(os.path.join(DIRS["baseline"], "baseline_xgb_pooled_vs_specific.png"), "baseline")
    except Exception as e:
        log.warning(f"  Baseline figures failed: {e}")


def generate_summary_figures():
    log.info("Generating summary figures...")
    if not os.path.exists(PROD_CSV):
        return
    try:
        df     = pd.read_csv(PROD_CSV)
        fold4  = df[df["fold"] == 4]
        models = ["lr", "rf", "xgb", "lstm", "cnn_gaf"]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, mode in zip(axes, ["multiclass", "binary"]):
            sub   = fold4[fold4["mode"] == mode]
            means = [sub[sub["model"] == m]["f1_weighted_avg"].mean() for m in models]
            stds  = [sub[sub["model"] == m]["f1_weighted_avg"].std()  for m in models]
            x     = np.arange(len(models))
            bars  = ax.bar(x, means, 0.55,
                           color=[MODEL_COLORS[m] for m in models],
                           alpha=0.88, yerr=stds, capsize=4,
                           error_kw={"linewidth": 1.2, "ecolor": "#333"})
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        mean + std + 0.02, f"{mean:.3f}",
                        ha="center", fontsize=8, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=9)
            ax.set_ylabel("Weighted F1 (mean ± std, 5 seeds)")
            ax.set_ylim(0, 1.14)
            ax.set_title(f"{'Multiclass' if mode=='multiclass' else 'Binary'}", pad=6)

        fig.suptitle("Production Stability: Seed-Averaged Weighted F1 at Fold 4",
                     fontsize=11, fontweight="bold", y=1.01)
        plt.tight_layout()
        save_fig(os.path.join(DIRS["summary"], "summary_seed_stability.png"), "summary")
    except Exception as e:
        log.warning(f"  Summary figures failed: {e}")


if __name__ == "__main__":
    log.info("MAIC Paper Figure Generator")
    log.info(f"Output root: {BASE_DIR}")

    df_test = load_fold4_test_data()
    X_sc, y, feat_names = prepare_features(df_test)

    generate_production_figures(X_sc, y, feat_names)
    generate_ablation_figures()
    generate_baseline_figures()
    generate_summary_figures()

    log.info("\nAll figures complete.")
    for name, d in DIRS.items():
        files = [f for f in os.listdir(d) if f.endswith(".png")]
        log.info(f"  {name}/  {len(files)} figures")
