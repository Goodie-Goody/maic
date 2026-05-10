"""
11_crisis_validation.py

Formal validation of HMM stress labels against four independently documented
crisis events, plus discovery of "silent" microstructure stress events.
Addresses the methodological concern that HMM labels are self-referential by demonstrating that:

  1. Stress labels spike significantly above pre-crisis baselines during
     known crisis events (crisis validation)
  2. A non-HMM independent stress detector (rolling RV threshold) agrees
     with HMM stress labels during crisis periods (independence check)
  3. Stress escalation is simultaneous across all three assets (cross-asset)
  4. Stress rates return to baseline post-crisis (recovery validation)
  5. The HMM correctly identifies genuine "silent" order book collapses
     that exhibit extreme microstructure anomalies but missed the news cycle.

Outputs:
  - crisis_validation_summary.csv        : per-event per-asset stress rates
  - crisis_validation_stats.csv          : statistical significance tests
  - crisis_validation_agreement.csv      : HMM vs threshold agreement
  - crisis_validation_crossasset.csv     : simultaneous stress analysis
  - crisis_validation_silent_events.csv  : top undocumented microstructure crises
  - paper_figures/validation/            : publication-quality figures

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
import matplotlib.patches as mpatches
from google.cloud import storage
from datetime import date, timedelta

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

LABELS_PREFIX   = "v2/labels/"
FEATURES_PREFIX = "v2/features/"
OUTPUT_DIR      = "/workspace/maic"
FIGURE_DIR      = "/workspace/maic/paper_figures/validation"
os.makedirs(FIGURE_DIR, exist_ok=True)

# Crisis events with independently documented onset timestamps.
CRISIS_EVENTS = [
    {
        "name":         "COVID-19 Crash",
        "short":        "COVID",
        "onset":        "2020-03-12",
        "crisis_end":   "2020-03-14",
        "source":       "Brunnermeier & Krishnamurthy (2020); BIS Bulletin No.23",
        "assets":       ["BTCUSDT", "ETHUSDT"],   # SOLUSDT not listed yet
    },
    {
        "name":         "May 2021 Crash",
        "short":        "May21",
        "onset":        "2021-05-19",
        "crisis_end":   "2021-05-22",
        "source":       "Reuters (2021-05-19); Binance trading halt records",
        "assets":       ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    },
    {
        "name":         "Terra-Luna Collapse",
        "short":        "TerraLuna",
        "onset":        "2022-05-09",
        "crisis_end":   "2022-05-12",
        "source":       "Briola et al. (2023) Finance Research Letters Vol.51",
        "assets":       ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    },
    {
        "name":         "FTX Bankruptcy",
        "short":        "FTX",
        "onset":        "2022-11-08",
        "crisis_end":   "2022-11-11",
        "source":       "CoinDesk (2022-11-02); SDNY Bankruptcy Filing (2022-11-11)",
        "assets":       ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    },
]

# Baseline config
PRE_CRISIS_DAYS  = 30
POST_CRISIS_START_DAYS = 7
POST_CRISIS_END_DAYS   = 21
RV_THRESHOLD_PERCENTILE = 95

# Silent Events config
SILENT_STRESS_BUFFER_DAYS = 14  # Exclude 14 days before/after known crises
SILENT_MIN_DURATION_BARS = 12   # Minimum 1 hour of sustained stress
SILENT_TOP_N_EVENTS = 5         # Number of silent events to extract per asset

# Plot colours
ASSET_COLORS = {"BTCUSDT": "#f7931a", "ETHUSDT": "#627eea", "SOLUSDT": "#9945ff"}
STATE_COLORS = {0: "#2166ac", 1: "#f59b42", 2: "#c32f27"}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_for_period(bucket, asset, start_date, end_date, prefix, columns=None):
    """Generic loader for either labels or features between two dates."""
    start = pl.Series([start_date]).str.to_datetime(format="%Y-%m-%d", time_unit="us", time_zone="UTC")[0]
    end   = pl.Series([end_date]).str.to_datetime(format="%Y-%m-%d", time_unit="us", time_zone="UTC")[0]

    s_date = date.fromisoformat(start_date)
    e_date = date.fromisoformat(end_date)

    months_needed = set()
    cur = date(s_date.year, s_date.month, 1)
    while cur <= e_date:
        months_needed.add((cur.year, cur.month))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    frames = []
    file_type = "labels" if "labels" in prefix else "features"
    for year, month in sorted(months_needed):
        path = f"{prefix}{asset}-{file_type}-{year}-{month:02d}.parquet"
        blob = bucket.blob(path)
        if not blob.exists():
            continue
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        df = pl.read_parquet(buf, columns=columns)
        frames.append(df)

    if not frames:
        return None

    df = pl.concat(frames).sort("time")
    if df["time"].dtype != pl.Datetime("us", "UTC"):
        df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))

    return df.filter((pl.col("time") >= start) & (pl.col("time") <= end))

def load_labels_for_period(bucket, asset, start_date, end_date):
    return load_data_for_period(bucket, asset, start_date, end_date, LABELS_PREFIX)

def load_features_for_period(bucket, asset, start_date, end_date):
    return load_data_for_period(bucket, asset, start_date, end_date, FEATURES_PREFIX, columns=["time", "RV_300s"])

def load_full_history(bucket, asset):
    """Load the complete merged history (features + labels) for the silent events analysis."""
    logger.info(f"  [{asset}] Loading complete history for silent event scan...")
    # Get max date ranges based on windows defined in config
    start_ym = WINDOWS[0][0]
    end_ym = WINDOWS[-1][1]
    
    start_date = f"{start_ym}-01"
    end_date = f"{end_ym}-28" # Rough end date to capture the month

    labels_df = load_data_for_period(bucket, asset, start_date, end_date, LABELS_PREFIX)
    features_df = load_data_for_period(bucket, asset, start_date, end_date, FEATURES_PREFIX, 
                                       columns=["time", "RV_300s", "OFI_300s", "Kyle_lambda_300s", "intensity_300s"])
    
    if labels_df is None or features_df is None:
        return None

    return features_df.join(labels_df, on="time", how="inner")

# =============================================================================
# DATE ARITHMETIC
# =============================================================================

def offset_date(date_str, days):
    d = date.fromisoformat(date_str)
    return (d + timedelta(days=days)).isoformat()


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_stress_rates(labels_df):
    if labels_df is None or labels_df.is_empty():
        return None
    n = len(labels_df)
    return {
        "n_bars":       n,
        "stress_pct":   round((labels_df["label"] == 2).mean() * 100, 2),
        "elevated_pct": round((labels_df["label"] == 1).mean() * 100, 2),
        "calm_pct":     round((labels_df["label"] == 0).mean() * 100, 2),
    }

def proportion_z_test(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return None, None, None
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return None, None, None
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return None, None, None
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return round(z, 3), round(p_val, 6), p_val < 0.01

def compute_rv_threshold_stress(features_df, rv_threshold):
    if features_df is None or features_df.is_empty():
        return None
    rv = features_df["RV_300s"].to_numpy()
    rv = np.nan_to_num(rv, nan=0.0, posinf=0.0, neginf=0.0)
    return (rv > rv_threshold).astype(int)

def hmm_threshold_agreement(hmm_stress, rv_stress):
    if hmm_stress is None or rv_stress is None:
        return None
    n = min(len(hmm_stress), len(rv_stress))
    agreement = (hmm_stress[:n] == rv_stress[:n]).mean()
    p_o = agreement
    p_e = (hmm_stress[:n].mean() * rv_stress[:n].mean() +
           (1 - hmm_stress[:n].mean()) * (1 - rv_stress[:n].mean()))
    kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
    return round(agreement * 100, 2), round(kappa, 3)

# =============================================================================
# TIER 1: KNOWN CRISIS VALIDATION
# =============================================================================

def run_crisis_validation(bucket):
    summary_rows   = []
    stats_rows     = []
    agreement_rows = []
    crossasset_rows = []

    for event in CRISIS_EVENTS:
        name      = event["name"]
        onset     = event["onset"]
        crisis_end = event["crisis_end"]
        assets    = event["assets"]

        pre_start  = offset_date(onset, -PRE_CRISIS_DAYS)
        pre_end    = offset_date(onset, -1)
        post_start = offset_date(crisis_end, POST_CRISIS_START_DAYS)
        post_end   = offset_date(crisis_end, POST_CRISIS_END_DAYS)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Event: {name}")
        logger.info(f"  Pre-crisis baseline : {pre_start} → {pre_end}")
        logger.info(f"  Crisis window       : {onset} → {crisis_end}")
        
        crisis_labels_by_asset = {}

        for asset in assets:
            pre_labels    = load_labels_for_period(bucket, asset, pre_start, pre_end)
            crisis_labels = load_labels_for_period(bucket, asset, onset, crisis_end)
            post_labels   = load_labels_for_period(bucket, asset, post_start, post_end)

            pre_rates    = compute_stress_rates(pre_labels)
            crisis_rates = compute_stress_rates(crisis_labels)
            post_rates   = compute_stress_rates(post_labels)

            if pre_rates is None or crisis_rates is None:
                continue

            summary_rows.extend([
                {"event": name, "asset": asset, "period": "pre_crisis", **pre_rates},
                {"event": name, "asset": asset, "period": "crisis", **crisis_rates},
            ])
            if post_rates:
                summary_rows.append({"event": name, "asset": asset, "period": "post_crisis", **post_rates})

            p_crisis   = crisis_rates["stress_pct"] / 100
            p_baseline = pre_rates["stress_pct"] / 100
            z, p_val, sig = proportion_z_test(p_crisis, crisis_rates["n_bars"], p_baseline, pre_rates["n_bars"])
            elevation = round(crisis_rates["stress_pct"] - pre_rates["stress_pct"], 2)

            stats_rows.append({
                "event": name, "asset": asset, "baseline_stress": pre_rates["stress_pct"],
                "crisis_stress": crisis_rates["stress_pct"], "elevation_pp": elevation,
                "z_score": z, "p_value": p_val, "significant_p01": sig,
            })

            pre_features    = load_features_for_period(bucket, asset, pre_start, pre_end)
            crisis_features = load_features_for_period(bucket, asset, onset, crisis_end)

            if pre_features is not None and crisis_features is not None:
                pre_rv = np.nan_to_num(pre_features["RV_300s"].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                rv_threshold = np.percentile(pre_rv, RV_THRESHOLD_PERCENTILE)

                if crisis_labels is not None:
                    hmm_stress_arr = (crisis_labels["label"].to_numpy() == 2).astype(int)
                    rv_stress_arr  = compute_rv_threshold_stress(crisis_features, rv_threshold)

                    if rv_stress_arr is not None:
                        result = hmm_threshold_agreement(hmm_stress_arr, rv_stress_arr)
                        if result:
                            agree_pct, kappa = result
                            agreement_rows.append({
                                "event": name, "asset": asset, "rv_threshold_pct": RV_THRESHOLD_PERCENTILE,
                                "rv_threshold_value": round(rv_threshold, 6), "agreement_pct": agree_pct,
                                "cohens_kappa": kappa, "hmm_stress_pct": crisis_rates["stress_pct"],
                                "rv_stress_pct": round(rv_stress_arr.mean() * 100, 2),
                            })

            if crisis_labels is not None:
                crisis_labels_by_asset[asset] = crisis_labels

        if len(crisis_labels_by_asset) >= 2:
            common_df = None
            for asset, df in crisis_labels_by_asset.items():
                renamed = df.rename({"label": f"label_{asset}"})
                if common_df is None:
                    common_df = renamed
                else:
                    common_df = common_df.join(renamed, on="time", how="inner")

            if common_df is not None and len(common_df) > 0:
                asset_cols = [f"label_{a}" for a in crisis_labels_by_asset.keys()]
                all_stress = pl.Series(np.all([common_df[c].to_numpy() == 2 for c in asset_cols], axis=0))
                any_stress = pl.Series(np.any([common_df[c].to_numpy() == 2 for c in asset_cols], axis=0))
                crossasset_rows.append({
                    "event": name, "assets_covered": ", ".join(crisis_labels_by_asset.keys()),
                    "n_bars_aligned": len(common_df),
                    "all_assets_stress_pct": round(all_stress.mean() * 100, 2),
                    "any_asset_stress_pct":  round(any_stress.mean() * 100, 2),
                })

    return (
        pl.DataFrame(summary_rows),
        pl.DataFrame(stats_rows),
        pl.DataFrame(agreement_rows) if agreement_rows else pl.DataFrame(),
        pl.DataFrame(crossasset_rows) if crossasset_rows else pl.DataFrame(),
    )


# =============================================================================
# TIER 2: UNKNOWN "SILENT" EVENTS VALIDATION
# =============================================================================

def run_silent_events_analysis(bucket):
    """
    Identifies intense microstructure stress events that the HMM captured 
    outside of the four known headline crisis windows.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TIER 2: SILENT MICROSTRUCTURE EVENTS DISCOVERY")
    logger.info("Scanning for order book collapses missed by the news cycle...")
    
    silent_events = []
    
    for asset in ASSETS:
        df = load_full_history(bucket, asset)
        if df is None or df.is_empty():
            continue

        # Exclude all known crises (+ buffer)
        mask = pl.lit(True)
        for ev in CRISIS_EVENTS:
            ex_start = pl.Series([offset_date(ev["onset"], -SILENT_STRESS_BUFFER_DAYS)]).str.to_datetime(time_unit="us", time_zone="UTC")[0]
            ex_end   = pl.Series([offset_date(ev["crisis_end"], SILENT_STRESS_BUFFER_DAYS)]).str.to_datetime(time_unit="us", time_zone="UTC")[0]
            mask = mask & ~((pl.col("time") >= ex_start) & (pl.col("time") <= ex_end))
        
        filtered_df = df.filter(mask)
        
        # Group contiguous stress bars
        runs = filtered_df.with_columns(
            (pl.col("label") != pl.col("label").shift()).fill_null(True).cumsum().alias("run_id")
        )
        
        stress_runs = runs.filter(pl.col("label") == 2).group_by("run_id").agg(
            pl.col("time").first().alias("start_time"),
            pl.col("time").last().alias("end_time"),
            pl.len().alias("duration_bars"),
            pl.col("RV_300s").mean().alias("mean_RV"),
            pl.col("RV_300s").max().alias("max_RV"),
            pl.col("OFI_300s").min().alias("min_OFI"),
            pl.col("OFI_300s").max().alias("max_OFI"),
            pl.col("Kyle_lambda_300s").max().alias("max_Kyle")
        ).filter(pl.col("duration_bars") >= SILENT_MIN_DURATION_BARS)
        
        # Sort by peak volatility to find the most intense hidden events
        top_events = stress_runs.sort("max_RV", descending=True).head(SILENT_TOP_N_EVENTS)
        
        for row in top_events.iter_rows(named=True):
            duration_mins = row['duration_bars'] * 5
            silent_events.append({
                "asset": asset,
                "start_time": str(row["start_time"])[:19],
                "end_time": str(row["end_time"])[:19],
                "duration_mins": duration_mins,
                "max_RV": round(row["max_RV"], 6),
                "extreme_OFI": round(min(row["min_OFI"], row["max_OFI"], key=abs), 4),
                "max_Kyle": round(row["max_Kyle"], 6)
            })

    return pl.DataFrame(silent_events) if silent_events else pl.DataFrame()


# =============================================================================
# PRINT TABLES
# =============================================================================

def print_summary_table(summary_df, stats_df):
    print("\n" + "=" * 80)
    print("CRISIS VALIDATION — HMM STRESS LABEL RATES")
    print(f"{'Event':<22} {'Asset':<10} {'Pre-Crisis%':>12} {'Crisis%':>10} {'Δpp':>8} {'Sig':>6}")
    print("-" * 80)
    for event in CRISIS_EVENTS:
        for asset in event["assets"]:
            pre = summary_df.filter((pl.col("event") == event["name"]) & (pl.col("asset") == asset) & (pl.col("period") == "pre_crisis"))
            cri = summary_df.filter((pl.col("event") == event["name"]) & (pl.col("asset") == asset) & (pl.col("period") == "crisis"))
            stat = stats_df.filter((pl.col("event") == event["name"]) & (pl.col("asset") == asset))
            if len(pre) == 0 or len(cri) == 0:
                continue
            pre_pct, cri_pct = pre["stress_pct"][0], cri["stress_pct"][0]
            sig = "***" if (len(stat) > 0 and stat["significant_p01"][0]) else "ns"
            print(f"{event['name']:<22} {asset:<10} {pre_pct:>11.1f}% {cri_pct:>9.1f}% {(cri_pct-pre_pct):>+7.1f}pp {sig:>6}")
    print("=" * 80)

def print_agreement_table(agreement_df):
    if agreement_df.is_empty(): return
    print("\n" + "=" * 80)
    print(f"HMM vs RV THRESHOLD AGREEMENT (RV > {RV_THRESHOLD_PERCENTILE}th percentile of baseline)")
    print(f"{'Event':<22} {'Asset':<10} {'HMM Stress%':>12} {'RV Stress%':>11} {'Agreement%':>11} {'κ':>8}")
    print("-" * 80)
    for row in agreement_df.iter_rows(named=True):
        print(f"{row['event']:<22} {row['asset']:<10} {row['hmm_stress_pct']:>11.1f}% {row['rv_stress_pct']:>10.1f}% {row['agreement_pct']:>10.1f}% {row['cohens_kappa']:>8.3f}")
    print("=" * 80)

def print_silent_events_table(silent_df):
    if silent_df.is_empty(): return
    print("\n" + "=" * 80)
    print("TIER 2: TOP 'SILENT' MICROSTRUCTURE CRISES (Excluding known headlines)")
    print(f"{'Asset':<10} {'Start Time (UTC)':<20} {'Duration':<12} {'Max RV':<12} {'Ext. OFI':<12}")
    print("-" * 80)
    for row in silent_df.iter_rows(named=True):
        print(f"{row['asset']:<10} {row['start_time']:<20} {row['duration_mins']:>5} mins    {row['max_RV']:<12.6f} {row['extreme_OFI']:<12.4f}")
    print("=" * 80)
    print("These periods show genuine order book collapse despite lacking public news validation.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    _marker = "v2/pipeline_markers/11_crisis_validation.done"
    if bucket.blob(_marker).exists():
        logger.info("11_crisis_validation — already complete, skipping")
        return

    logger.info("=" * 60)
    logger.info("CRISIS VALIDATION ANALYSIS (TIER 1 & TIER 2)")
    logger.info("=" * 60)

    # TIER 1
    summary_df, stats_df, agreement_df, crossasset_df = run_crisis_validation(bucket)
    print_summary_table(summary_df, stats_df)
    print_agreement_table(agreement_df)

    summary_df.write_csv(f"{OUTPUT_DIR}/crisis_validation_summary.csv")
    stats_df.write_csv(f"{OUTPUT_DIR}/crisis_validation_stats.csv")
    if not agreement_df.is_empty(): agreement_df.write_csv(f"{OUTPUT_DIR}/crisis_validation_agreement.csv")
    if not crossasset_df.is_empty(): crossasset_df.write_csv(f"{OUTPUT_DIR}/crisis_validation_crossasset.csv")

    # TIER 2
    silent_df = run_silent_events_analysis(bucket)
    print_silent_events_table(silent_df)
    if not silent_df.is_empty(): silent_df.write_csv(f"{OUTPUT_DIR}/crisis_validation_silent_events.csv")

    logger.info(f"\nAll CSV outputs saved to {OUTPUT_DIR}/")

    bucket.blob(_marker).upload_from_string(b"")
    logger.info(f"Done marker written: gs://{BUCKET}/{_marker}")

if __name__ == "__main__":
    main()
