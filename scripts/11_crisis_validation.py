"""
11_crisis_validation.py

Formal validation of HMM stress labels against four independently documented
crisis events, plus discovery of "silent" microstructure stress events.
Addresses the methodological concern that HMM labels are self-referential.

Improvements in this version:
  - Fixed Polars .cum_sum() syntax for version compatibility.
  - Added publication-quality plotting with serif-font styling.
  - Fully dynamic path resolution for cross-environment reproducibility.

Usage:
  python3 scripts/11_crisis_validation.py
"""

import sys
import os
import io
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from google.cloud import storage
from datetime import date, timedelta

# =============================================================================
# DYNAMIC PATH RESOLUTION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from config import ASSETS, BUCKET, WINDOWS

# Dynamically map the GCP key relative to the repository root
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(REPO_ROOT, "gcp-key.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG & STYLING
# =============================================================================
LABELS_PREFIX   = "v2/labels/"
FEATURES_PREFIX = "v2/features/"
OUTPUT_DIR      = REPO_ROOT
FIGURE_DIR      = os.path.join(REPO_ROOT, "paper_figures", "validation")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Publication styling to match Script 08
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "figure.dpi":         300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

CRISIS_EVENTS = [
    {"name": "COVID-19 Crash", "short": "COVID", "onset": "2020-03-12", "crisis_end": "2020-03-14", "assets": ["BTCUSDT", "ETHUSDT"]},
    {"name": "May 2021 Crash", "short": "May21", "onset": "2021-05-19", "crisis_end": "2021-05-22", "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
    {"name": "Terra-Luna Collapse", "short": "TerraLuna", "onset": "2022-05-09", "crisis_end": "2022-05-12", "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
    {"name": "FTX Bankruptcy", "short": "FTX", "onset": "2022-11-08", "crisis_end": "2022-11-11", "assets": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
]

PRE_CRISIS_DAYS = 30
POST_CRISIS_START_DAYS = 7
POST_CRISIS_END_DAYS = 21
RV_THRESHOLD_PERCENTILE = 95
SILENT_STRESS_BUFFER_DAYS = 14
SILENT_MIN_DURATION_BARS = 12
SILENT_TOP_N_EVENTS = 5

ASSET_COLORS = {"BTCUSDT": "#f7931a", "ETHUSDT": "#627eea", "SOLUSDT": "#9945ff"}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_for_period(bucket, asset, start_date, end_date, prefix, columns=None):
    start = pl.Series([start_date]).str.to_datetime(format="%Y-%m-%d", time_unit="us", time_zone="UTC")[0]
    end   = pl.Series([end_date]).str.to_datetime(format="%Y-%m-%d", time_unit="us", time_zone="UTC")[0]
    s_date, e_date = date.fromisoformat(start_date), date.fromisoformat(end_date)

    months_needed = set()
    cur = date(s_date.year, s_date.month, 1)
    while cur <= e_date:
        months_needed.add((cur.year, cur.month))
        if cur.month == 12: cur = date(cur.year + 1, 1, 1)
        else: cur = date(cur.year, cur.month + 1, 1)

    frames = []
    file_type = "labels" if "labels" in prefix else "features"
    for year, month in sorted(months_needed):
        path = f"{prefix}{asset}-{file_type}-{year}-{month:02d}.parquet"
        blob = bucket.blob(path)
        if not blob.exists(): continue
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        df = pl.read_parquet(buf, columns=columns)
        frames.append(df)

    if not frames: return None
    df = pl.concat(frames).sort("time")
    if df["time"].dtype != pl.Datetime("us", "UTC"):
        df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))
    return df.filter((pl.col("time") >= start) & (pl.col("time") <= end))

def load_labels_for_period(bucket, asset, start_date, end_date):
    return load_data_for_period(bucket, asset, start_date, end_date, LABELS_PREFIX)

def load_features_for_period(bucket, asset, start_date, end_date):
    return load_data_for_period(bucket, asset, start_date, end_date, FEATURES_PREFIX, columns=["time", "RV_300s"])

def load_full_history(bucket, asset):
    logger.info(f"   [{asset}] Loading complete history for silent event scan...")
    start_ym, end_ym = WINDOWS[0][0], WINDOWS[-1][1]
    start_date, end_date = f"{start_ym}-01", f"{end_ym}-28"
    labels_df = load_data_for_period(bucket, asset, start_date, end_date, LABELS_PREFIX)
    features_df = load_data_for_period(bucket, asset, start_date, end_date, FEATURES_PREFIX, 
                                       columns=["time", "RV_300s", "OFI_300s", "Kyle_lambda_300s", "intensity_300s"])
    if labels_df is None or features_df is None: return None
    return features_df.join(labels_df, on="time", how="inner")

def offset_date(date_str, days):
    return (date.fromisoformat(date_str) + timedelta(days=days)).isoformat()

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_stress_rates(labels_df):
    if labels_df is None or labels_df.is_empty(): return None
    return {
        "n_bars":       len(labels_df),
        "stress_pct":   round((labels_df["label"] == 2).mean() * 100, 2),
        "elevated_pct": round((labels_df["label"] == 1).mean() * 100, 2),
        "calm_pct":     round((labels_df["label"] == 0).mean() * 100, 2),
    }

def proportion_z_test(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0: return None, None, None
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1: return None, None, None
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0: return None, None, None
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return round(z, 3), round(p_val, 6), p_val < 0.01

def compute_rv_threshold_stress(features_df, rv_threshold):
    if features_df is None or features_df.is_empty(): return None
    rv = np.nan_to_num(features_df["RV_300s"].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    return (rv > rv_threshold).astype(int)

def hmm_threshold_agreement(hmm_stress, rv_stress):
    if hmm_stress is None or rv_stress is None: return None
    n = min(len(hmm_stress), len(rv_stress))
    p_o = (hmm_stress[:n] == rv_stress[:n]).mean()
    p_e = (hmm_stress[:n].mean() * rv_stress[:n].mean() + (1 - hmm_stress[:n].mean()) * (1 - rv_stress[:n].mean()))
    kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
    return round(p_o * 100, 2), round(kappa, 3)

# =============================================================================
# TIER 1: KNOWN CRISIS VALIDATION
# =============================================================================

def run_crisis_validation(bucket):
    summary_rows, stats_rows, agreement_rows, crossasset_rows = [], [], [], []

    for event in CRISIS_EVENTS:
        name, onset, crisis_end, assets = event["name"], event["onset"], event["crisis_end"], event["assets"]
        pre_start, pre_end = offset_date(onset, -PRE_CRISIS_DAYS), offset_date(onset, -1)
        post_start, post_end = offset_date(crisis_end, POST_CRISIS_START_DAYS), offset_date(crisis_end, POST_CRISIS_END_DAYS)

        logger.info(f"\nEvent: {name}")
        crisis_labels_by_asset = {}

        for asset in assets:
            pre_l, cri_l, post_l = load_labels_for_period(bucket, asset, pre_start, pre_end), load_labels_for_period(bucket, asset, onset, crisis_end), load_labels_for_period(bucket, asset, post_start, post_end)
            pre_r, cri_r, post_r = compute_stress_rates(pre_l), compute_stress_rates(cri_l), compute_stress_rates(post_l)

            if pre_r is None or cri_r is None: continue
            summary_rows.extend([{"event": name, "asset": asset, "period": "pre_crisis", **pre_r}, {"event": name, "asset": asset, "period": "crisis", **cri_r}])
            if post_r: summary_rows.append({"event": name, "asset": asset, "period": "post_crisis", **post_r})

            z, p_val, sig = proportion_z_test(cri_r["stress_pct"]/100, cri_r["n_bars"], pre_r["stress_pct"]/100, pre_r["n_bars"])
            stats_rows.append({"event": name, "asset": asset, "baseline_stress": pre_r["stress_pct"], "crisis_stress": cri_r["stress_pct"], "elevation_pp": round(cri_r["stress_pct"] - pre_r["stress_pct"], 2), "z_score": z, "p_value": p_val, "significant_p01": sig})

            pre_f, cri_f = load_features_for_period(bucket, asset, pre_start, pre_end), load_features_for_period(bucket, asset, onset, crisis_end)
            if pre_f is not None and cri_f is not None:
                rv_t = np.percentile(np.nan_to_num(pre_f["RV_300s"].to_numpy(), nan=0.0), RV_THRESHOLD_PERCENTILE)
                res = hmm_threshold_agreement((cri_l["label"].to_numpy() == 2).astype(int), compute_rv_threshold_stress(cri_f, rv_t))
                if res:
                    agree_pct, kappa = res
                    agreement_rows.append({"event": name, "asset": asset, "rv_threshold_pct": RV_THRESHOLD_PERCENTILE, "rv_threshold_value": round(rv_t, 6), "agreement_pct": agree_pct, "cohens_kappa": kappa, "hmm_stress_pct": cri_r["stress_pct"], "rv_stress_pct": round(((np.nan_to_num(cri_f["RV_300s"].to_numpy()) > rv_t).mean() * 100), 2)})

            if cri_l is not None: crisis_labels_by_asset[asset] = cri_l

        if len(crisis_labels_by_asset) >= 2:
            common_df = None
            for asset, df in crisis_labels_by_asset.items():
                renamed = df.rename({"label": f"label_{asset}"})
                common_df = renamed if common_df is None else common_df.join(renamed, on="time", how="inner")
            if common_df is not None and len(common_df) > 0:
                asset_cols = [f"label_{a}" for a in crisis_labels_by_asset.keys()]
                all_s = pl.Series(np.all([common_df[c].to_numpy() == 2 for c in asset_cols], axis=0))
                any_s = pl.Series(np.any([common_df[c].to_numpy() == 2 for c in asset_cols], axis=0))
                crossasset_rows.append({"event": name, "assets_covered": ", ".join(crisis_labels_by_asset.keys()), "n_bars_aligned": len(common_df), "all_assets_stress_pct": round(all_s.mean() * 100, 2), "any_asset_stress_pct": round(any_s.mean() * 100, 2)})

    return pl.DataFrame(summary_rows), pl.DataFrame(stats_rows), pl.DataFrame(agreement_rows), pl.DataFrame(crossasset_rows)

# =============================================================================
# TIER 2: SILENT EVENTS ANALYSIS
# =============================================================================

def run_silent_events_analysis(bucket):
    logger.info("\nTIER 2: SILENT MICROSTRUCTURE EVENTS DISCOVERY")
    silent_events = []
    for asset in ASSETS:
        df = load_full_history(bucket, asset)
        if df is None or df.is_empty(): continue
        mask = pl.lit(True)
        for ev in CRISIS_EVENTS:
            ex_s = pl.Series([offset_date(ev["onset"], -SILENT_STRESS_BUFFER_DAYS)]).str.to_datetime(time_unit="us", time_zone="UTC")[0]
            ex_e = pl.Series([offset_date(ev["crisis_end"], SILENT_STRESS_BUFFER_DAYS)]).str.to_datetime(time_unit="us", time_zone="UTC")[0]
            mask = mask & ~((pl.col("time") >= ex_s) & (pl.col("time") <= ex_e))
        
        # FIXED: .cum_sum() used instead of .cumsum()
        runs = df.filter(mask).with_columns((pl.col("label") != pl.col("label").shift()).fill_null(True).cum_sum().alias("run_id"))
        top_events = runs.filter(pl.col("label") == 2).group_by("run_id").agg([
            pl.col("time").first().alias("start_time"), pl.col("time").last().alias("end_time"),
            pl.len().alias("duration_bars"), pl.col("RV_300s").max().alias("max_RV"),
            pl.col("OFI_300s").min().alias("min_OFI"), pl.col("OFI_300s").max().alias("max_OFI")
        ]).filter(pl.col("duration_bars") >= SILENT_MIN_DURATION_BARS).sort("max_RV", descending=True).head(SILENT_TOP_N_EVENTS)
        
        for row in top_events.iter_rows(named=True):
            silent_events.append({"asset": asset, "start_time": str(row["start_time"])[:19], "end_time": str(row["end_time"])[:19], "duration_mins": row['duration_bars'] * 5, "max_RV": round(row["max_RV"], 6), "extreme_OFI": round(min(row["min_OFI"], row["max_OFI"], key=abs), 4)})
    return pl.DataFrame(silent_events)

# =============================================================================
# PLOTTING FUNCTIONS (Styled for Publication)
# =============================================================================

def plot_crisis_stress_rates(summary_df):
    logger.info("  Generating Tier 1 figure: Crisis Stress Rates...")
    plot_df = summary_df.filter(pl.col("period").is_in(["pre_crisis", "crisis"]))
    fig, axes = plt.subplots(1, len(CRISIS_EVENTS), figsize=(14, 5), sharey=True)
    for i, event in enumerate(CRISIS_EVENTS):
        ax, event_data = axes[i], plot_df.filter(pl.col("event") == event["name"])
        if event_data.is_empty(): continue
        pivot = event_data.pivot(values="stress_pct", index="asset", on="period")
        assets, pre, cri = pivot["asset"].to_list(), pivot["pre_crisis"].to_list(), pivot["crisis"].to_list()
        x, width = np.arange(len(assets)), 0.35
        ax.bar(x - width/2, pre, width, label='Pre-Crisis', color='#999999', alpha=0.7)
        ax.bar(x + width/2, cri, width, label='Crisis', color='#c32f27', alpha=0.8)
        ax.set_title(event["short"])
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45)
        if i == 0: ax.set_ylabel("HMM Stress Label Rate (%)")
        if i == len(CRISIS_EVENTS)-1: ax.legend(framealpha=0.85)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig_tier1_crisis_rates.png"))
    plt.close()

def plot_silent_event_intensity(silent_df):
    if silent_df.is_empty(): return
    logger.info("  Generating Tier 2 figure: Silent Event Intensity...")
    plt.figure(figsize=(9, 6))
    for asset in ASSETS:
        ad = silent_df.filter(pl.col("asset") == asset)
        if ad.is_empty(): continue
        plt.scatter(ad["duration_mins"], ad["max_RV"], color=ASSET_COLORS[asset], label=asset, s=80, alpha=0.7, edgecolors='k')
    plt.xscale('log')
    plt.xlabel("Duration (minutes) - Log Scale")
    plt.ylabel("Peak Realised Volatility (Max RV)")
    plt.title("Tier 2: Detected 'Silent' Microstructure Crises")
    plt.legend(framealpha=0.85)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig_tier2_silent_intensity.png"))
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    client, bucket = storage.Client(), storage.Client().bucket(BUCKET)
    _marker = "v2/pipeline_markers/11_crisis_validation.done"
    if bucket.blob(_marker).exists():
        logger.info("11_crisis_validation — already complete, skipping")
        return

    # TIER 1
    summary_df, stats_df, agreement_df, crossasset_df = run_crisis_validation(bucket)
    plot_crisis_stress_rates(summary_df)
    summary_df.write_csv(os.path.join(OUTPUT_DIR, "crisis_validation_summary.csv"))
    stats_df.write_csv(os.path.join(OUTPUT_DIR, "crisis_validation_stats.csv"))
    if not agreement_df.is_empty(): agreement_df.write_csv(os.path.join(OUTPUT_DIR, "crisis_validation_agreement.csv"))

    # TIER 2
    silent_df = run_silent_events_analysis(bucket)
    plot_silent_event_intensity(silent_df)
    if not silent_df.is_empty(): silent_df.write_csv(os.path.join(OUTPUT_DIR, "crisis_validation_silent_events.csv"))

    bucket.blob(_marker).upload_from_string(b"")
    logger.info(f"Validation complete. Outputs saved to {OUTPUT_DIR}/ and {FIGURE_DIR}/")

if __name__ == "__main__":
    main()
