"""
11b_crisis_validation_full.py

Unified, definitive validation script. Supersedes the original
11_crisis_validation.py entirely.

THREE TIERS:

TIER 1 — Known Crisis Validation
    Stress elevation z-tests against four independently documented crisis
    events. Pre/during/post stress rates. Cross-asset simultaneous stress.
    Establishes that HMM stress labels spike significantly during known
    crisis periods — externally anchored by academic literature and news.

TIER 2 — Global vs Local HMM: External Validator Comparison
    Both global and local HMM labels measured against THREE genuinely
    independent validators — none of which were HMM training features:

        Validator 1 — Price Drawdown
            Maximum percentage drawdown from rolling peak within each bar.
            HMM never saw raw price levels — only microstructure features
            derived from trades. Completely independent.

        Validator 2 — Crisis Timestamp Binary
            Four documented crisis events from the academic literature.
            Externally defined, not derived from this dataset at all.
            Binary: bar falls within documented crisis window or not.

        Validator 3 — Cross-Asset Simultaneous Stress
            HMM fitted per-asset independently — never saw joint dynamics.
            Whether stress occurs simultaneously across assets is external
            to any individual asset's HMM.

    For each validator: Cohen's kappa computed for both global and local
    HMM labels. Head-to-head comparison table shows which labelling scheme
    tracks external reality better.

TIER 3 — Silent Event Discovery
    Stress episodes outside known crisis windows — microstructure stress
    with no major public narrative. Demonstrates the framework's forward-
    looking utility beyond retrospective validation.

OUTPUTS:
    crisis_validation_summary.csv           Tier 1 stress rates
    crisis_validation_stats.csv             Tier 1 z-tests
    crisis_validation_crossasset.csv        Tier 1 cross-asset
    global_vs_local_kappa.csv               Tier 2 head-to-head
    crisis_validation_silent_events.csv     Tier 3 silent events
    paper_figures/validation/               Publication figures

NOTE ON FEATURE INDEPENDENCE:
    RV, OFI, Kyle_lambda, and intensity are ALL HMM training features.
    None of them are used as external validators here — doing so would
    be circular. Only price (never seen by HMM), crisis timestamps
    (externally defined), and cross-asset dynamics (fitted independently)
    are used as ground truth references.

Usage:
    python3 scripts/11b_crisis_validation_full.py
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
OUTPUT_DIR          = REPO_ROOT
FIGURE_DIR          = os.path.join(REPO_ROOT, "paper_figures", "validation")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Publication styling
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.dpi":       300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

ASSET_COLORS = {
    "BTCUSDT": "#f7931a",
    "ETHUSDT": "#627eea",
    "SOLUSDT": "#9945ff",
}

# Four independently documented crisis events
# Onset/end dates from academic literature and public record
CRISIS_EVENTS = [
    {
        "name":       "COVID-19 Crash",
        "short":      "COVID",
        "onset":      "2020-03-12",
        "crisis_end": "2020-03-14",
        "assets":     ["BTCUSDT", "ETHUSDT"],
        "fold":       None,   # Window 0 — training data in all folds
        "note":       "Global market crash triggered by pandemic declaration",
    },
    {
        "name":       "May 2021 Crash",
        "short":      "May21",
        "onset":      "2021-05-19",
        "crisis_end": "2021-05-22",
        "assets":     ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "fold":       1,
        "note":       "Sustained multi-hour deterioration, China mining ban fears",
    },
    {
        "name":       "Terra-Luna Collapse",
        "short":      "TerraLuna",
        "onset":      "2022-05-09",
        "crisis_end": "2022-05-12",
        "assets":     ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "fold":       2,
        "note":       "Stablecoin algorithmic failure, DeFi contagion",
    },
    {
        "name":       "FTX Bankruptcy",
        "short":      "FTX",
        "onset":      "2022-11-08",
        "crisis_end": "2022-11-11",
        "assets":     ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "fold":       3,
        "note":       "Exchange insolvency, abrupt structural collapse",
    },
]

# Walk-forward fold test windows — matches config.py WINDOWS
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

PRE_CRISIS_DAYS         = 30
POST_CRISIS_START_DAYS  = 7
POST_CRISIS_END_DAYS    = 21
SILENT_STRESS_BUFFER    = 14
SILENT_MIN_BARS         = 12
SILENT_TOP_N            = 5

# Price drawdown config
DRAWDOWN_ROLLING_WINDOW = 12   # bars (~1 hour rolling peak)
DRAWDOWN_THRESHOLD_PCT  = 95   # percentile of pre-crisis drawdowns

# =============================================================================
# SHARED UTILITIES
# =============================================================================

def offset_date(date_str, days):
    return (date.fromisoformat(date_str) + timedelta(days=days)).isoformat()


def months_in_range(start_ym, end_ym):
    sy, sm = int(start_ym[:4]), int(start_ym[5:])
    ey, em = int(end_ym[:4]),   int(end_ym[5:])
    months, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def load_parquets_by_date(bucket, asset, prefix, start_date, end_date,
                           columns=None):
    """Load parquets for a date range (YYYY-MM-DD strings)."""
    s = date.fromisoformat(start_date)
    e = date.fromisoformat(end_date)

    all_months = set()
    cur = date(s.year, s.month, 1)
    while cur <= e:
        all_months.add((cur.year, cur.month))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)

    file_type = "labels" if "labels" in prefix else "features"
    frames = []
    for year, month in sorted(all_months):
        path = f"{prefix}{asset}-{file_type}-{year}-{month:02d}.parquet"
        blob = bucket.blob(path)
        if not blob.exists():
            continue
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        try:
            df = pl.read_parquet(buf, columns=columns)
            frames.append(df)
        except Exception as ex:
            logger.warning(f"  Read error {path}: {ex}")

    if not frames:
        return None

    df = pl.concat(frames).sort("time")
    if df["time"].dtype != pl.Datetime("us", "UTC"):
        df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))

    start_dt = pl.Series([start_date]).str.to_datetime(
        format="%Y-%m-%d", time_unit="us", time_zone="UTC")[0]
    end_dt = pl.Series([end_date]).str.to_datetime(
        format="%Y-%m-%d", time_unit="us", time_zone="UTC")[0]

    return df.filter(
        (pl.col("time") >= start_dt) & (pl.col("time") <= end_dt)
    )


def load_parquets_by_windows(bucket, asset, prefix, windows, columns=None):
    """Load parquets for a list of (start_ym, end_ym) window tuples."""
    all_months = set()
    for start_ym, end_ym in windows:
        for ym in months_in_range(start_ym, end_ym):
            all_months.add(ym)

    file_type = "labels" if "labels" in prefix else "features"
    frames = []
    for year, month in sorted(all_months):
        path = f"{prefix}{asset}-{file_type}-{year}-{month:02d}.parquet"
        blob = bucket.blob(path)
        if not blob.exists():
            continue
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        try:
            df = pl.read_parquet(buf, columns=columns)
            frames.append(df)
        except Exception as ex:
            logger.warning(f"  Read error {path}: {ex}")

    if not frames:
        return None

    df = pl.concat(frames).sort("time")
    if df["time"].dtype != pl.Datetime("us", "UTC"):
        df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))
    return df


def cohens_kappa(y_true, y_pred):
    """Cohen's kappa between two binary arrays, corrected for chance."""
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n].astype(float), y_pred[:n].astype(float)
    p_o = (yt == yp).mean()
    p_e = yt.mean() * yp.mean() + (1 - yt.mean()) * (1 - yp.mean())
    if p_e == 1.0:
        return 0.0, round(p_o * 100, 2)
    kappa = (p_o - p_e) / (1 - p_e)
    return round(float(kappa), 3), round(float(p_o) * 100, 2)


def kappa_label(k):
    """Landis & Koch (1977) interpretation."""
    if k is None:    return "N/A"
    if k < 0.0:      return "poor"
    if k < 0.20:     return "slight"
    if k < 0.40:     return "fair"
    if k < 0.60:     return "moderate"
    if k < 0.80:     return "substantial"
    return "almost perfect"


def proportion_z_test(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return None, None, None
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool in (0, 1):
        return None, None, None
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        return None, None, None
    z = (p1 - p2) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return round(z, 3), round(p_val, 6), bool(p_val < 0.01)


def stress_rates(labels_df):
    if labels_df is None or labels_df.is_empty():
        return None
    return {
        "n_bars":       len(labels_df),
        "stress_pct":   round((labels_df["label"] == 2).mean() * 100, 2),
        "elevated_pct": round((labels_df["label"] == 1).mean() * 100, 2),
        "calm_pct":     round((labels_df["label"] == 0).mean() * 100, 2),
    }


# =============================================================================
# PRICE DRAWDOWN VALIDATOR
# =============================================================================

def compute_price_drawdown(bucket, asset, start_date, end_date,
                            rolling_window=DRAWDOWN_ROLLING_WINDOW):
    """
    Compute maximum percentage drawdown from a rolling peak for each bar.
    Uses the 'price' column from feature parquets (VWAP or close proxy).

    Returns numpy array of drawdown values (positive = drawdown).
    HMM never saw raw price levels — this is a genuinely independent signal.
    """
    features_df = load_parquets_by_date(
        bucket, asset, FEATURES_PREFIX,
        start_date, end_date,
        columns=["time", "price"]
    )

    if features_df is None or features_df.is_empty():
        return None, None

    prices = features_df["price"].to_numpy().astype(np.float64)
    prices = np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)

    # Forward-fill nan
    mask = np.isnan(prices)
    idx  = np.where(~mask, np.arange(len(prices)), 0)
    np.maximum.accumulate(idx, out=idx)
    prices = prices[idx]

    # Rolling maximum (expanding minimum lookback = rolling_window)
    rolling_peak = np.array([
        np.nanmax(prices[max(0, i - rolling_window):i + 1])
        for i in range(len(prices))
    ])

    drawdown = np.where(
        rolling_peak > 0,
        (rolling_peak - prices) / rolling_peak * 100,
        0.0
    )

    return drawdown, features_df["time"]


# =============================================================================
# TIER 1 — KNOWN CRISIS VALIDATION
# =============================================================================

def run_tier1(bucket):
    logger.info("\n" + "=" * 60)
    logger.info("TIER 1: KNOWN CRISIS VALIDATION")
    logger.info("=" * 60)

    summary_rows   = []
    stats_rows     = []
    crossasset_rows = []

    for event in CRISIS_EVENTS:
        name      = event["name"]
        onset     = event["onset"]
        crisis_end = event["crisis_end"]
        assets    = event["assets"]

        pre_start  = offset_date(onset,      -PRE_CRISIS_DAYS)
        pre_end    = offset_date(onset,      -1)
        post_start = offset_date(crisis_end,  POST_CRISIS_START_DAYS)
        post_end   = offset_date(crisis_end,  POST_CRISIS_END_DAYS)

        logger.info(f"\n  Event: {name}")
        crisis_labels_by_asset = {}

        for asset in assets:
            pre_l  = load_parquets_by_date(
                bucket, asset, LABELS_PREFIX, pre_start, pre_end)
            cri_l  = load_parquets_by_date(
                bucket, asset, LABELS_PREFIX, onset, crisis_end)
            post_l = load_parquets_by_date(
                bucket, asset, LABELS_PREFIX, post_start, post_end)

            pre_r  = stress_rates(pre_l)
            cri_r  = stress_rates(cri_l)
            post_r = stress_rates(post_l)

            if pre_r is None or cri_r is None:
                continue

            summary_rows.extend([
                {"event": name, "asset": asset,
                 "period": "pre_crisis",  **pre_r},
                {"event": name, "asset": asset,
                 "period": "crisis",      **cri_r},
            ])
            if post_r:
                summary_rows.append(
                    {"event": name, "asset": asset,
                     "period": "post_crisis", **post_r}
                )

            z, p_val, sig = proportion_z_test(
                cri_r["stress_pct"] / 100, cri_r["n_bars"],
                pre_r["stress_pct"] / 100, pre_r["n_bars"],
            )
            stats_rows.append({
                "event":           name,
                "asset":           asset,
                "baseline_stress": pre_r["stress_pct"],
                "crisis_stress":   cri_r["stress_pct"],
                "elevation_pp":    round(cri_r["stress_pct"] - pre_r["stress_pct"], 2),
                "z_score":         z,
                "p_value":         p_val,
                "significant_p01": sig,
            })

            logger.info(
                f"    [{asset}] pre={pre_r['stress_pct']:.1f}%  "
                f"crisis={cri_r['stress_pct']:.1f}%  "
                f"z={z}  p={p_val}  sig={sig}"
            )

            if cri_l is not None:
                crisis_labels_by_asset[asset] = cri_l

        # Cross-asset simultaneous stress
        if len(crisis_labels_by_asset) >= 2:
            common_df = None
            for asset, df in crisis_labels_by_asset.items():
                renamed = df.rename({"label": f"label_{asset}"})
                common_df = renamed if common_df is None else \
                    common_df.join(renamed, on="time", how="inner")

            if common_df is not None and len(common_df) > 0:
                asset_cols = [f"label_{a}" for a in crisis_labels_by_asset]
                all_s = np.all(
                    [common_df[c].to_numpy() == 2 for c in asset_cols], axis=0
                )
                any_s = np.any(
                    [common_df[c].to_numpy() == 2 for c in asset_cols], axis=0
                )
                crossasset_rows.append({
                    "event":               name,
                    "assets_covered":      ", ".join(crisis_labels_by_asset),
                    "n_bars_aligned":      len(common_df),
                    "all_assets_stress_pct": round(all_s.mean() * 100, 2),
                    "any_asset_stress_pct":  round(any_s.mean() * 100, 2),
                })
                logger.info(
                    f"    Cross-asset: all={all_s.mean()*100:.1f}%  "
                    f"any={any_s.mean()*100:.1f}%  "
                    f"n_aligned={len(common_df):,}"
                )

    summary_df    = pl.DataFrame(summary_rows)
    stats_df      = pl.DataFrame(stats_rows)
    crossasset_df = pl.DataFrame(crossasset_rows)

    summary_df.write_csv(
        os.path.join(OUTPUT_DIR, "crisis_validation_summary.csv"))
    stats_df.write_csv(
        os.path.join(OUTPUT_DIR, "crisis_validation_stats.csv"))
    crossasset_df.write_csv(
        os.path.join(OUTPUT_DIR, "crisis_validation_crossasset.csv"))

    logger.info("\n  Tier 1 CSVs saved.")
    return summary_df, stats_df, crossasset_df


# =============================================================================
# TIER 2 — GLOBAL vs LOCAL: EXTERNAL VALIDATOR COMPARISON
# =============================================================================

def build_crisis_timestamp_binary(n_bars, times_series, onset, crisis_end):
    """
    Binary array: 1 if bar falls within the documented crisis window.
    Externally defined — not derived from this dataset.
    """
    onset_dt = pl.Series([onset]).str.to_datetime(
        format="%Y-%m-%d", time_unit="us", time_zone="UTC")[0]
    end_dt   = pl.Series([crisis_end]).str.to_datetime(
        format="%Y-%m-%d", time_unit="us", time_zone="UTC")[0]

    times_np = times_series.to_numpy()
    binary   = np.array([
        1 if (onset_dt <= t <= end_dt) else 0
        for t in times_np
    ], dtype=int)
    return binary


def build_drawdown_binary(bucket, asset, test_window, pre_start, pre_end,
                          threshold_percentile=DRAWDOWN_THRESHOLD_PCT):
    """
    Binary stress flag from price drawdown.
    Threshold computed from pre-crisis baseline drawdowns.
    Completely independent of HMM — price levels were never HMM inputs.
    """
    start_date = f"{test_window[0]}-01"
    end_date   = f"{test_window[1]}-28"

    # Compute threshold from pre-crisis baseline
    base_drawdown, _ = compute_price_drawdown(
        bucket, asset, pre_start, pre_end
    )
    if base_drawdown is None or len(base_drawdown) == 0:
        return None, None

    threshold = np.nanpercentile(base_drawdown, threshold_percentile)

    # Compute drawdown for test window
    test_drawdown, test_times = compute_price_drawdown(
        bucket, asset, start_date, end_date
    )
    if test_drawdown is None:
        return None, None

    binary = (test_drawdown > threshold).astype(int)
    return binary, test_times


def run_tier2(bucket):
    logger.info("\n" + "=" * 60)
    logger.info("TIER 2: GLOBAL vs LOCAL HMM — EXTERNAL VALIDATOR COMPARISON")
    logger.info("=" * 60)
    logger.info(
        "  Validators used (all genuinely independent of HMM training):\n"
        "    1. Price drawdown     — HMM never saw raw price levels\n"
        "    2. Crisis timestamps  — externally documented events\n"
        "    3. Cross-asset stress — HMM fitted per-asset independently\n"
        "  NOTE: RV, OFI, Kyle_lambda, intensity excluded — all HMM inputs"
    )

    kappa_rows = []

    for fold_n, test_window in FOLD_TEST_WINDOWS.items():
        crisis_label = FOLD_CRISIS_LABEL[fold_n]

        # Find the crisis event for this fold (if any)
        fold_event = next(
            (e for e in CRISIS_EVENTS if e.get("fold") == fold_n), None
        )

        logger.info(f"\n  Fold {fold_n} | {test_window} | [{crisis_label}]")

        # Pre-crisis baseline dates for drawdown threshold
        pre_start = offset_date(f"{test_window[0]}-01", -PRE_CRISIS_DAYS)
        pre_end   = offset_date(f"{test_window[0]}-01", -1)

        # Collect per-asset cross-asset labels (global) for validator 3
        global_stress_by_asset = {}

        for asset in ASSETS:
            logger.info(f"\n    [{asset}]")

            # ----------------------------------------------------------------
            # Load global labels for test window
            # ----------------------------------------------------------------
            global_df = load_parquets_by_windows(
                bucket, asset, LABELS_PREFIX, [test_window]
            )
            if global_df is None or global_df.is_empty():
                logger.warning(f"    No global labels — skipping")
                continue

            global_stress = (global_df["label"].to_numpy() == 2).astype(int)
            global_stress_by_asset[asset] = global_stress
            n_global = len(global_stress)

            # ----------------------------------------------------------------
            # Load local labels for test window
            # ----------------------------------------------------------------
            local_prefix = f"{LOCAL_LABELS_PREFIX}fold_{fold_n}/"
            local_df = load_parquets_by_windows(
                bucket, asset, local_prefix, [test_window]
            )

            has_local = (
                local_df is not None and
                not local_df.is_empty()
            )
            if has_local:
                local_stress = (
                    local_df["label"].to_numpy() == 2
                ).astype(int)
                n_local = len(local_stress)
            else:
                logger.warning(f"    No local labels for Fold {fold_n}")

            # ----------------------------------------------------------------
            # VALIDATOR 1 — Price Drawdown
            # ----------------------------------------------------------------
            drawdown_binary, drawdown_times = build_drawdown_binary(
                bucket, asset, test_window, pre_start, pre_end
            )

            if drawdown_binary is not None and len(drawdown_binary) > 0:
                # Align with global labels by length (both cover same window)
                n_align = min(n_global, len(drawdown_binary))

                # Global kappa
                kappa_g, agree_g = cohens_kappa(
                    global_stress[:n_align], drawdown_binary[:n_align]
                )
                kappa_rows.append({
                    "fold":            fold_n,
                    "asset":           asset,
                    "crisis_event":    crisis_label,
                    "hmm_type":        "global",
                    "validator":       "price_drawdown",
                    "validator_basis": "Max % drawdown from rolling peak — price never seen by HMM",
                    "hmm_stress_pct":  round(global_stress.mean() * 100, 2),
                    "validator_stress_pct": round(drawdown_binary.mean() * 100, 2),
                    "n_bars":          n_align,
                    "cohens_kappa":    kappa_g,
                    "agreement_pct":   agree_g,
                    "kappa_interp":    kappa_label(kappa_g),
                })
                logger.info(
                    f"    Drawdown validator  — global κ={kappa_g:.3f} "
                    f"({kappa_label(kappa_g)})"
                )

                # Local kappa
                if has_local:
                    n_align_l = min(n_local, len(drawdown_binary))
                    kappa_l, agree_l = cohens_kappa(
                        local_stress[:n_align_l], drawdown_binary[:n_align_l]
                    )
                    kappa_rows.append({
                        "fold":            fold_n,
                        "asset":           asset,
                        "crisis_event":    crisis_label,
                        "hmm_type":        "local",
                        "validator":       "price_drawdown",
                        "validator_basis": "Max % drawdown from rolling peak — price never seen by HMM",
                        "hmm_stress_pct":  round(local_stress.mean() * 100, 2),
                        "validator_stress_pct": round(drawdown_binary.mean() * 100, 2),
                        "n_bars":          n_align_l,
                        "cohens_kappa":    kappa_l,
                        "agreement_pct":   agree_l,
                        "kappa_interp":    kappa_label(kappa_l),
                    })
                    logger.info(
                        f"    Drawdown validator  — local  κ={kappa_l:.3f} "
                        f"({kappa_label(kappa_l)})"
                    )

            # ----------------------------------------------------------------
            # VALIDATOR 2 — Crisis Timestamp Binary
            # Only meaningful for folds that contain a documented crisis
            # ----------------------------------------------------------------
            if fold_event is not None and asset in fold_event["assets"]:
                ts_binary = build_crisis_timestamp_binary(
                    n_global,
                    global_df["time"],
                    fold_event["onset"],
                    fold_event["crisis_end"],
                )

                if ts_binary.sum() > 0:
                    # Global kappa
                    kappa_g, agree_g = cohens_kappa(global_stress, ts_binary)
                    kappa_rows.append({
                        "fold":            fold_n,
                        "asset":           asset,
                        "crisis_event":    crisis_label,
                        "hmm_type":        "global",
                        "validator":       "crisis_timestamp",
                        "validator_basis": f"Documented crisis window: {fold_event['onset']} to {fold_event['crisis_end']}",
                        "hmm_stress_pct":  round(global_stress.mean() * 100, 2),
                        "validator_stress_pct": round(ts_binary.mean() * 100, 2),
                        "n_bars":          n_global,
                        "cohens_kappa":    kappa_g,
                        "agreement_pct":   agree_g,
                        "kappa_interp":    kappa_label(kappa_g),
                    })
                    logger.info(
                        f"    Timestamp validator — global κ={kappa_g:.3f} "
                        f"({kappa_label(kappa_g)})"
                    )

                    # Local kappa
                    if has_local:
                        # Align timestamp binary with local labels
                        if local_df is not None:
                            ts_binary_l = build_crisis_timestamp_binary(
                                n_local,
                                local_df["time"],
                                fold_event["onset"],
                                fold_event["crisis_end"],
                            )
                            kappa_l, agree_l = cohens_kappa(
                                local_stress, ts_binary_l
                            )
                            kappa_rows.append({
                                "fold":            fold_n,
                                "asset":           asset,
                                "crisis_event":    crisis_label,
                                "hmm_type":        "local",
                                "validator":       "crisis_timestamp",
                                "validator_basis": f"Documented crisis window: {fold_event['onset']} to {fold_event['crisis_end']}",
                                "hmm_stress_pct":  round(local_stress.mean() * 100, 2),
                                "validator_stress_pct": round(ts_binary_l.mean() * 100, 2),
                                "n_bars":          n_local,
                                "cohens_kappa":    kappa_l,
                                "agreement_pct":   agree_l,
                                "kappa_interp":    kappa_label(kappa_l),
                            })
                            logger.info(
                                f"    Timestamp validator — local  κ={kappa_l:.3f} "
                                f"({kappa_label(kappa_l)})"
                            )

        # --------------------------------------------------------------------
        # VALIDATOR 3 — Cross-Asset Simultaneous Stress
        # HMM fitted per-asset — never saw joint dynamics
        # --------------------------------------------------------------------
        if len(global_stress_by_asset) >= 2:
            assets_present = list(global_stress_by_asset.keys())
            n_min = min(len(v) for v in global_stress_by_asset.values())

            # Any-asset simultaneous stress as the reference signal
            any_stress = np.any(
                np.stack([v[:n_min] for v in global_stress_by_asset.values()]),
                axis=0
            ).astype(int)

            for asset in assets_present:
                g_stress = global_stress_by_asset[asset][:n_min]
                kappa_g, agree_g = cohens_kappa(g_stress, any_stress)

                kappa_rows.append({
                    "fold":            fold_n,
                    "asset":           asset,
                    "crisis_event":    crisis_label,
                    "hmm_type":        "global",
                    "validator":       "crossasset_simultaneous",
                    "validator_basis": "Any-asset simultaneous stress — HMM fitted per-asset independently",
                    "hmm_stress_pct":  round(g_stress.mean() * 100, 2),
                    "validator_stress_pct": round(any_stress.mean() * 100, 2),
                    "n_bars":          n_min,
                    "cohens_kappa":    kappa_g,
                    "agreement_pct":   agree_g,
                    "kappa_interp":    kappa_label(kappa_g),
                })

                # Local cross-asset
                local_prefix = f"{LOCAL_LABELS_PREFIX}fold_{fold_n}/"
                local_df_ca = load_parquets_by_windows(
                    bucket, asset, local_prefix, [test_window]
                )
                if local_df_ca is not None and not local_df_ca.is_empty():
                    l_stress = (
                        local_df_ca["label"].to_numpy() == 2
                    ).astype(int)
                    n_align = min(len(l_stress), len(any_stress))
                    kappa_l, agree_l = cohens_kappa(
                        l_stress[:n_align], any_stress[:n_align]
                    )
                    kappa_rows.append({
                        "fold":            fold_n,
                        "asset":           asset,
                        "crisis_event":    crisis_label,
                        "hmm_type":        "local",
                        "validator":       "crossasset_simultaneous",
                        "validator_basis": "Any-asset simultaneous stress — HMM fitted per-asset independently",
                        "hmm_stress_pct":  round(l_stress.mean() * 100, 2),
                        "validator_stress_pct": round(any_stress.mean() * 100, 2),
                        "n_bars":          n_align,
                        "cohens_kappa":    kappa_l,
                        "agreement_pct":   agree_l,
                        "kappa_interp":    kappa_label(kappa_l),
                    })

            logger.info(
                f"\n    Cross-asset validator computed for "
                f"{assets_present} (Fold {fold_n})"
            )

    if not kappa_rows:
        logger.warning("  No kappa rows computed — check GCS connectivity")
        return None

    kappa_df = pl.DataFrame(kappa_rows).sort(
        ["fold", "asset", "hmm_type", "validator"]
    )
    kappa_df.write_csv(
        os.path.join(OUTPUT_DIR, "global_vs_local_kappa.csv")
    )
    logger.info(f"\n  Tier 2 CSV saved: global_vs_local_kappa.csv")

    # Print summary
    logger.info("\n" + "=" * 72)
    logger.info("GLOBAL vs LOCAL — KAPPA AGAINST EXTERNAL VALIDATORS")
    logger.info("=" * 72)
    logger.info(
        f"  {'F':<3} {'Asset':<10} {'Validator':<26} "
        f"{'Global κ':<10} {'Local κ':<10} "
        f"{'Global interp':<18} {'Local interp'}"
    )
    logger.info("  " + "-" * 88)

    for fold_n in FOLD_TEST_WINDOWS:
        fold_data = kappa_df.filter(pl.col("fold") == fold_n)
        if fold_data.is_empty():
            continue
        logger.info(
            f"  --- Fold {fold_n}: {FOLD_CRISIS_LABEL[fold_n]} ---"
        )
        for asset in ASSETS:
            for validator in [
                "price_drawdown",
                "crisis_timestamp",
                "crossasset_simultaneous",
            ]:
                g = fold_data.filter(
                    (pl.col("asset") == asset) &
                    (pl.col("hmm_type") == "global") &
                    (pl.col("validator") == validator)
                )
                l = fold_data.filter(
                    (pl.col("asset") == asset) &
                    (pl.col("hmm_type") == "local") &
                    (pl.col("validator") == validator)
                )
                if g.is_empty() and l.is_empty():
                    continue

                g_k = f"{g['cohens_kappa'][0]:.3f}" if len(g) > 0 else "N/A"
                l_k = f"{l['cohens_kappa'][0]:.3f}" if len(l) > 0 else "no data"
                g_i = g["kappa_interp"][0] if len(g) > 0 else "N/A"
                l_i = l["kappa_interp"][0] if len(l) > 0 else "no data"

                logger.info(
                    f"  {fold_n:<3} {asset:<10} {validator:<26} "
                    f"{g_k:<10} {l_k:<10} {g_i:<18} {l_i}"
                )

    return kappa_df


# =============================================================================
# TIER 3 — SILENT EVENT DISCOVERY
# =============================================================================

def run_tier3(bucket):
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: SILENT MICROSTRUCTURE EVENT DISCOVERY")
    logger.info("=" * 60)

    silent_rows = []

    for asset in ASSETS:
        logger.info(f"\n  [{asset}] Scanning full history...")

        start_ym = WINDOWS[0][0]
        end_ym   = WINDOWS[-1][1]

        labels_df = load_parquets_by_windows(
            bucket, asset, LABELS_PREFIX,
            [(start_ym, end_ym)]
        )
        features_df = load_parquets_by_windows(
            bucket, asset, FEATURES_PREFIX,
            [(start_ym, end_ym)],
            columns=["time", "price"]
        )

        if labels_df is None or features_df is None:
            logger.warning(f"  [{asset}] Missing data — skipping")
            continue

        # Join labels with price for context
        df = labels_df.join(
            features_df.select(["time", "price"]),
            on="time", how="left"
        )

        # Exclude known crisis windows + buffer
        mask = pl.lit(True)
        for ev in CRISIS_EVENTS:
            ex_s = pl.Series([offset_date(ev["onset"], -SILENT_STRESS_BUFFER)]
                             ).str.to_datetime(time_unit="us", time_zone="UTC")[0]
            ex_e = pl.Series([offset_date(ev["crisis_end"], SILENT_STRESS_BUFFER)]
                             ).str.to_datetime(time_unit="us", time_zone="UTC")[0]
            mask = mask & ~(
                (pl.col("time") >= ex_s) & (pl.col("time") <= ex_e)
            )

        df_filtered = df.filter(mask)

        # Identify contiguous stress runs
        runs = df_filtered.with_columns(
            (pl.col("label") != pl.col("label").shift())
            .fill_null(True)
            .cum_sum()
            .alias("run_id")
        )

        top_events = (
            runs.filter(pl.col("label") == 2)
            .group_by("run_id")
            .agg([
                pl.col("time").first().alias("start_time"),
                pl.col("time").last().alias("end_time"),
                pl.len().alias("duration_bars"),
                pl.col("price").mean().alias("mean_price"),
            ])
            .filter(pl.col("duration_bars") >= SILENT_MIN_BARS)
            .sort("duration_bars", descending=True)
            .head(SILENT_TOP_N)
        )

        for row in top_events.iter_rows(named=True):
            duration_mins = row["duration_bars"] * 5
            silent_rows.append({
                "asset":          asset,
                "start_time":     str(row["start_time"])[:19],
                "end_time":       str(row["end_time"])[:19],
                "duration_mins":  duration_mins,
                "duration_hours": round(duration_mins / 60, 1),
                "mean_price":     round(row["mean_price"] or 0, 2),
            })
            logger.info(
                f"    {str(row['start_time'])[:16]}  "
                f"duration={duration_mins}min  "
                f"price={row['mean_price']:.2f}"
            )

    if not silent_rows:
        logger.warning("  No silent events found")
        return None

    silent_df = pl.DataFrame(silent_rows).sort(
        ["duration_mins"], descending=True
    )
    silent_df.write_csv(
        os.path.join(OUTPUT_DIR, "crisis_validation_silent_events.csv")
    )
    logger.info(f"\n  Tier 3 CSV saved: crisis_validation_silent_events.csv")
    return silent_df


# =============================================================================
# FIGURES
# =============================================================================

def plot_tier1_stress_rates(summary_df):
    logger.info("  Generating Tier 1 figure: Crisis Stress Rates...")
    plot_df = summary_df.filter(
        pl.col("period").is_in(["pre_crisis", "crisis"])
    )

    fig, axes = plt.subplots(
        1, len(CRISIS_EVENTS), figsize=(14, 5), sharey=True
    )

    for i, event in enumerate(CRISIS_EVENTS):
        ax = axes[i]
        event_data = plot_df.filter(pl.col("event") == event["name"])
        if event_data.is_empty():
            continue

        pivot = event_data.pivot(
            values="stress_pct", index="asset", on="period"
        )
        assets = pivot["asset"].to_list()
        pre    = pivot["pre_crisis"].to_list()
        cri    = pivot["crisis"].to_list()

        x, w = np.arange(len(assets)), 0.35
        ax.bar(x - w/2, pre, w, label="Pre-Crisis",
               color="#999999", alpha=0.7)
        ax.bar(x + w/2, cri, w, label="Crisis",
               color="#c32f27", alpha=0.8)

        ax.set_title(event["short"])
        ax.set_xticks(x)
        ax.set_xticklabels(
            [a.replace("USDT", "") for a in assets], rotation=45
        )
        if i == 0:
            ax.set_ylabel("HMM Stress Label Rate (%)")
        if i == len(CRISIS_EVENTS) - 1:
            ax.legend(framealpha=0.85)

    plt.suptitle(
        "Tier 1: HMM Stress Rate — Pre-Crisis vs Crisis Windows",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "fig_tier1_crisis_rates.png"))
    plt.close()
    logger.info("    Saved: fig_tier1_crisis_rates.png")


def plot_tier2_kappa_comparison(kappa_df):
    logger.info("  Generating Tier 2 figure: Global vs Local Kappa...")

    validators = ["price_drawdown", "crisis_timestamp", "crossasset_simultaneous"]
    validator_labels = {
        "price_drawdown":          "Price\nDrawdown",
        "crisis_timestamp":        "Crisis\nTimestamp",
        "crossasset_simultaneous": "Cross-Asset\nStress",
    }

    folds_with_data = sorted(kappa_df["fold"].unique().to_list())
    n_folds = len(folds_with_data)

    fig, axes = plt.subplots(
        1, n_folds, figsize=(4 * n_folds, 5), sharey=True
    )
    if n_folds == 1:
        axes = [axes]

    for ax_i, fold_n in enumerate(folds_with_data):
        ax = axes[ax_i]
        fold_data = kappa_df.filter(pl.col("fold") == fold_n)

        x = np.arange(len(validators))
        w = 0.35

        global_kappas = []
        local_kappas  = []

        for v in validators:
            g = fold_data.filter(
                (pl.col("validator") == v) &
                (pl.col("hmm_type") == "global")
            )
            l = fold_data.filter(
                (pl.col("validator") == v) &
                (pl.col("hmm_type") == "local")
            )
            # Average across assets
            global_kappas.append(
                float(g["cohens_kappa"].mean()) if len(g) > 0 else 0.0
            )
            local_kappas.append(
                float(l["cohens_kappa"].mean()) if len(l) > 0 else 0.0
            )

        ax.bar(x - w/2, global_kappas, w,
               label="Global HMM", color="#2ecc71", alpha=0.85)
        ax.bar(x + w/2, local_kappas,  w,
               label="Local HMM",  color="#e74c3c", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(
            f"Fold {fold_n}\n{FOLD_CRISIS_LABEL[fold_n]}", fontsize=9
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [validator_labels[v] for v in validators], fontsize=8
        )
        ax.set_ylim(-0.3, 1.0)
        if ax_i == 0:
            ax.set_ylabel("Cohen's Kappa")
        if ax_i == n_folds - 1:
            ax.legend(framealpha=0.85, fontsize=8)

    plt.suptitle(
        "Tier 2: Global vs Local HMM Agreement with External Validators\n"
        "(higher κ = better alignment with independent stress signal)",
        fontsize=11, fontweight="bold", y=1.04
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "fig_tier2_global_vs_local_kappa.png")
    )
    plt.close()
    logger.info("    Saved: fig_tier2_global_vs_local_kappa.png")


def plot_tier3_silent_events(silent_df):
    if silent_df is None or silent_df.is_empty():
        return
    logger.info("  Generating Tier 3 figure: Silent Events...")

    plt.figure(figsize=(9, 6))
    for asset in ASSETS:
        ad = silent_df.filter(pl.col("asset") == asset)
        if ad.is_empty():
            continue
        plt.scatter(
            ad["duration_mins"], ad["mean_price"],
            color=ASSET_COLORS[asset], label=asset,
            s=90, alpha=0.75, edgecolors="k", linewidth=0.5
        )

    plt.xscale("log")
    plt.xlabel("Duration (minutes) — Log Scale")
    plt.ylabel("Mean Price During Event (USDT)")
    plt.title(
        "Tier 3: Detected Silent Microstructure Stress Events\n"
        "(outside known crisis windows, ≥60 min duration)"
    )
    plt.legend(framealpha=0.85)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "fig_tier3_silent_events.png")
    )
    plt.close()
    logger.info("    Saved: fig_tier3_silent_events.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    marker = "v2/pipeline_markers/11b_crisis_validation.done"
    if bucket.blob(marker).exists():
        logger.info("11b already complete. To rerun:")
        logger.info(f"  gsutil rm gs://{BUCKET}/{marker}")
        return

    # TIER 1
    summary_df, stats_df, crossasset_df = run_tier1(bucket)

    # TIER 2
    kappa_df = run_tier2(bucket)

    # TIER 3
    silent_df = run_tier3(bucket)

    # Figures
    logger.info("\nGenerating figures...")
    if summary_df is not None and not summary_df.is_empty():
        plot_tier1_stress_rates(summary_df)
    if kappa_df is not None and not kappa_df.is_empty():
        plot_tier2_kappa_comparison(kappa_df)
    if silent_df is not None and not silent_df.is_empty():
        plot_tier3_silent_events(silent_df)

    bucket.blob(marker).upload_from_string(b"")

    logger.info("\n" + "=" * 60)
    logger.info("11b complete.")
    logger.info(f"  CSVs    → {OUTPUT_DIR}")
    logger.info(f"  Figures → {FIGURE_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
