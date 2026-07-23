"""
12_inference.py  --  live liquidity stress detection

Fetches the last 300s of Binance trades, computes microstructure features,
applies fractional differencing to price, and runs XGBoost.

stress_prob is the probability that current microstructure resembles a stress
regime right now, not a countdown. Lead times of 56-108 min before documented
crisis events arise because order flow deteriorates before price does.
A WARNING fires when stress_prob > 0.85 for 2+ consecutive bars.

Usage:
    python3 scripts/12_inference.py --asset BTCUSDT
    python3 scripts/12_inference.py --asset all --loop
    python3 scripts/12_inference.py --asset BTCUSDT --json

Model: pooled XGBoost binary, Fold 4 Seed 42, F1-W = 0.9706 (28 features).
"""

import sys
import os
import json
import time
import pickle
import shutil
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import requests
from datetime import datetime, timezone

# =============================================================================
# DYNAMIC PATH RESOLUTION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from config import ASSETS, ASSET_D_VALUES, BUCKET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# LOGGING CONFIG
# =============================================================================
LOG_DIR          = os.path.join(REPO_ROOT, "logs", "inference")
INFERENCE_LOG    = os.path.join(LOG_DIR, "inference_log.csv")
OUTCOME_LOG      = os.path.join(LOG_DIR, "outcome_log.csv")
MAX_LOG_SIZE_MB  = 50
OUTCOME_DELAY_S  = 1800   # 30 minutes — check price 30 min after signal
PRICE_DROP_THRESHOLD_PCT = 0.3   # 0.3% drop = TRUE POSITIVE

os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================
BINANCE_BASE        = "https://api.binance.com"
AGG_TRADES_ENDPOINT = "/api/v3/aggTrades"   # time-windowed aggregated trades
KLINES_ENDPOINT     = "/api/v3/klines"
BAR_SECONDS         = 300                   # 5-minute bar — matches training exactly
# Training's Kyle's lambda is a rolling correlation over w consecutive 10s
# bars (w=30 for the 300s version). A rolling window of w bar-to-bar diffs
# needs w+1 underlying bar prices, i.e. 310s of history for the 300s case,
# not 300s. Fetch this extra padding purely to support that -- window_features
# still cuts off at exactly BAR_SECONDS for every other feature.
FETCH_PADDING_S     = 10
# N_TRADES removed: we now use a time window, not a fixed trade count
EPSILON         = 1e-10

DEFAULT_FOLD      = 4
DEFAULT_SEED      = 42
STRESS_THRESHOLD  = 0.85
MIN_CONSECUTIVE   = 2

# GCS path patterns — pooled vs asset-specific (author's private fallback)
MODEL_GCS_PATH_POOLED = (
    "v2/results_production/seed_{seed}/pooled/fold_{fold}/models/xgb_binary.pkl"
)
MODEL_GCS_PATH_ASSET = (
    "v2/results_production/seed_{seed}/{asset}/fold_{fold}/models/xgb_binary.pkl"
)
MODELS_DIR = os.path.join(REPO_ROOT, "logs")

# Public Hugging Face repo holding the model pickles — this is what lets
# anyone without GCP credentials run inference. Update to your actual repo.
HF_REPO_ID = "Goooddy/maic-models"

# Must match COMMON_FEATURES in 06d exactly. "price" is fracdiff'd in place.
# 27 features for asset-specific, 28 for pooled (+ asset_id).
BASE_FEATURES = [
    "price",    # fractionally differenced in place by 04b (d=0.3/0.4/0.2 per asset)
    "volume", "rv",
    "OFI_10s",  "TCI_10s",  "intensity_10s",  "VWAP_10s",  "ILLIQ_10s",  "RV_10s",
    "OFI_60s",  "TCI_60s",  "intensity_60s",  "VWAP_60s",  "ILLIQ_60s",  "RV_60s",
    "Kyle_lambda_60s",
    "OFI_300s", "TCI_300s", "intensity_300s", "VWAP_300s", "ILLIQ_300s",
    "RV_300s",  "Kyle_lambda_300s",
    "VWAP_dev_10s", "VWAP_dev_60s", "VWAP_dev_300s", "CV_dt_10s",
]
POOLED_FEATURES      = BASE_FEATURES + ["asset_id"]
ASSET_SPECIFIC_FEATURES = BASE_FEATURES  # no asset_id

ASSET_ID_MAP = {asset: idx for idx, asset in enumerate(ASSETS)}


def get_model_features(is_pooled=True):
    """Return the correct feature list for pooled vs asset-specific models."""
    return POOLED_FEATURES if is_pooled else ASSET_SPECIFIC_FEATURES

# Rolling stress history for continuity condition
# Buffer size 288 = 24 hours of 5-minute bars — never loses duration context
_stress_history  = {asset: [] for asset in ASSETS}
_stress_duration = {asset: 0  for asset in ASSETS}   # true consecutive bar count

# ANSI colours
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# =============================================================================
# INFERENCE LOGGING + OUTCOME TRACKING
# =============================================================================

def _rotate_log_if_needed(path, max_mb=MAX_LOG_SIZE_MB):
    """
    If log file exceeds max_mb, keep only the most recent half.
    Prevents unbounded disk growth during long --loop runs.
    """
    if not os.path.exists(path):
        return
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < max_mb:
        return
    logger.info(f"  Log rotation: {path} ({size_mb:.1f}MB) — trimming oldest 50%")
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if len(lines) <= 2:
            return
        header = lines[0]
        keep   = lines[len(lines) // 2:]  # keep newest half
        with open(path, "w") as f:
            f.write(header)
            f.writelines(keep)
    except Exception as e:
        logger.warning(f"  Log rotation failed: {e}")


def log_inference(result, raw_price):
    """Append one inference result to inference_log.csv."""
    if "error" in result:
        return
    _rotate_log_if_needed(INFERENCE_LOG)
    header_needed = not os.path.exists(INFERENCE_LOG)
    kf  = result.get("key_features", {})
    row = ",".join([
        str(result["timestamp"]),
        str(result["asset"]),
        f"{raw_price:.4f}",
        f"{result['stress_prob']:.4f}",
        str(result["regime"]),
        str(int(result["warning_active"])),
        str(result["consecutive_bars"]),
        f"{kf.get('OFI_300s', 0):.4f}",
        f"{kf.get('RV_300s', 0):.6f}",
        f"{kf.get('Kyle_lambda_300s', 0):.4f}",
        f"{kf.get('intensity_300s', 0):.4f}",
        str(result.get("n_trades_used", 0)),
        str(result.get("window_seconds", 300)),
    ]) + "\n"
    with open(INFERENCE_LOG, "a") as f:
        if header_needed:
            f.write("timestamp,asset,price,stress_prob,regime,"
                    "warning_active,consecutive_bars,"
                    "OFI_300s,RV_300s,Kyle_lambda_300s,intensity_300s,"
                    "n_trades,window_s\n")
        f.write(row)


def log_pending_outcome(result, raw_price):
    """Record a pending outcome when WARNING fires. Resolved after 30 min."""
    if "error" in result or not result.get("warning_active"):
        return
    _rotate_log_if_needed(OUTCOME_LOG)
    check_time    = datetime.now(timezone.utc).timestamp() + OUTCOME_DELAY_S
    header_needed = not os.path.exists(OUTCOME_LOG)
    row = ",".join([
        str(result["timestamp"]),
        str(result["asset"]),
        f"{raw_price:.4f}",
        f"{result['stress_prob']:.4f}",
        str(result["consecutive_bars"]),
        f"{check_time:.0f}",
        "",   # price_at_check — filled by check_pending_outcomes
        "",   # price_change_pct — filled later
        "",   # outcome — filled later
    ]) + "\n"
    with open(OUTCOME_LOG, "a") as f:
        if header_needed:
            f.write("signal_time,asset,price_at_signal,stress_prob,"
                    "consecutive_bars,check_at_unix,"
                    "min_price_in_window,min_change_pct,outcome\n")
        f.write(row)
def check_pending_outcomes():
    """
    Resolve any warnings whose 30-minute outcome window has elapsed.

    Uses the minimum price over the window rather than the price at exactly
    T+30min -- stress events often dip and recover, and we want to capture that.

    STRESS_CONFIRMED  -- price dropped >= threshold and stayed down
    STRESS_ABSORBED   -- price dipped but recovered
    FALSE_POSITIVE    -- price never moved meaningfully
    PRICE_PUMPED      -- price went up instead
    """
    if not os.path.exists(OUTCOME_LOG):
        return

    now_unix = datetime.now(timezone.utc).timestamp()

    try:
        with open(OUTCOME_LOG, "r") as f:
            lines = f.readlines()
    except Exception:
        return

    if len(lines) <= 1:
        return

    header      = lines[0]
    updated     = [header]
    any_updated = False

    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) < 9:
            updated.append(line)
            continue

        # Already resolved
        if parts[8].strip():
            updated.append(line)
            continue

        check_at = float(parts[5]) if parts[5] else 0
        if check_at == 0 or now_unix < check_at:
            updated.append(line)
            continue

        asset           = parts[1]
        price_at_signal = float(parts[2])
        signal_unix     = check_at - OUTCOME_DELAY_S

        try:
            # Fetch all 1-minute klines over the 30-minute window
            # to find the MINIMUM price (worst drawdown)
            resp = requests.get(
                f"{BINANCE_BASE}{KLINES_ENDPOINT}",
                params={
                    "symbol":    asset,
                    "interval":  "1m",
                    "startTime": int(signal_unix * 1000),
                    "endTime":   int(check_at * 1000),
                    "limit":     35,
                },
                timeout=10
            )
            resp.raise_for_status()
            klines = resp.json()

            if not klines:
                updated.append(line)
                continue

            # Extract low prices across the window — captures intrabar drops
            lows      = [float(k[3]) for k in klines]   # k[3] = low
            closes    = [float(k[4]) for k in klines]    # k[4] = close
            price_min = min(lows)
            price_end = closes[-1]

            min_change_pct = ((price_min - price_at_signal)
                              / price_at_signal) * 100
            end_change_pct = ((price_end - price_at_signal)
                              / price_at_signal) * 100

            # Classification based on minimum price in window
            if min_change_pct <= -PRICE_DROP_THRESHOLD_PCT:
                if end_change_pct <= -PRICE_DROP_THRESHOLD_PCT:
                    outcome = "STRESS_CONFIRMED"     # dropped and stayed down
                else:
                    outcome = "STRESS_ABSORBED"      # dropped but recovered
            elif end_change_pct >= PRICE_DROP_THRESHOLD_PCT:
                outcome = "PRICE_PUMPED"             # went up instead
            else:
                outcome = "FALSE_POSITIVE"           # never moved meaningfully

            parts[6] = f"{price_min:.4f}"   # worst price in window
            parts[7] = f"{min_change_pct:.3f}"
            parts[8] = outcome

            logger.info(
                f"  OUTCOME [{asset}]: signal={price_at_signal:.2f} "
                f"min={price_min:.2f} ({min_change_pct:+.3f}%) "
                f"end={price_end:.2f} ({end_change_pct:+.3f}%) "
                f"= {outcome}"
            )
            any_updated = True

        except Exception as e:
            logger.warning(f"  Outcome check failed for {asset}: {e}")

        updated.append(",".join(parts) + "\n")

    if any_updated:
        with open(OUTCOME_LOG, "w") as f:
            f.writelines(updated)


# =============================================================================
# FRACTIONAL DIFFERENCING
# Exact implementation from 04b_stationarity_fracdiff.py
# =============================================================================

# Training pipeline (04b) uses thres=1e-5, which produces 1,458-3,382 weights
# depending on asset d-value. Live inference only has ~80 one-minute bars of
# price history available, making 1e-5 impossible to satisfy (the fracdiff loop
# never executes and silently outputs NaN -> 0.0).
#
# thres=1e-3 produces 55-73 weights across all three assets — comfortably within
# 80 bars while preserving the meaningful long-memory signal. The ablation study
# (Section 4.2) confirms fracdiff contributes negligibly to XGBoost specifically,
# so the practical impact of this threshold difference is minimal, but correctness
# matters: the feature should carry real signal, not a hardcoded zero.
FRACDIFF_THRES_INFERENCE = 1e-3


def get_weights_ffd(d, thres=1e-5):
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1])


def frac_diff_single(price_series, d, thres=1e-5):
    """
    Apply fixed-width fractional differencing to price_series.
    thres controls weight truncation — lower = more weights = more history needed.
    Training uses 1e-5; inference uses FRACDIFF_THRES_INFERENCE (1e-3).
    """
    weights = get_weights_ffd(d, thres)
    w_len   = len(weights)
    n       = len(price_series)
    result  = np.full(n, np.nan)
    for i in range(w_len - 1, n):
        window = price_series[i - w_len + 1: i + 1]
        if np.any(np.isnan(window)):
            continue
        result[i] = np.dot(weights, window)
    return result

# =============================================================================
# BINANCE DATA FETCHING
# =============================================================================

def fetch_agg_trades_window(symbol, window_seconds=BAR_SECONDS, retries=3):
    """
    Fetch all aggTrades from the last window_seconds. Paginates automatically
    since BTC regularly exceeds Binance's 1,000-record limit per call even in
    quiet markets. Returns dicts with: price, qty, isBuyerMaker, time.
    """
    now_ms   = int(time.time() * 1000)
    start_ms = now_ms - (window_seconds * 1000)

    url        = f"{BINANCE_BASE}{AGG_TRADES_ENDPOINT}"
    all_trades = []
    fetch_from = start_ms

    for attempt in range(retries):
        try:
            # Paginate until we have all trades in the window
            while True:
                params = {
                    "symbol":    symbol,
                    "startTime": fetch_from,
                    "endTime":   now_ms,
                    "limit":     1000,
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                raw = resp.json()

                if not raw:
                    break

                # Normalise to match raw trades format
                batch = [
                    {
                        "price":        t["p"],
                        "qty":          t["q"],
                        "isBuyerMaker": t["m"],
                        "time":         t["T"],
                    }
                    for t in raw
                ]
                all_trades.extend(batch)

                # If we got fewer than 1,000 we have everything
                if len(raw) < 1000:
                    break

                # Otherwise paginate — start from 1ms after last trade
                fetch_from = raw[-1]["T"] + 1

                # Safety: don't fetch beyond our window
                if fetch_from >= now_ms:
                    break

            if not all_trades:
                raise RuntimeError(f"Empty aggTrades response for {symbol}")

            logger.info(
                f"  [{symbol}] Fetched {len(all_trades)} aggTrades "
                f"over last {window_seconds}s window"
                + (" (paginated)" if len(all_trades) >= 1000 else "")
            )
            return all_trades

        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Binance aggTrades API failed: {e}")
            all_trades = []
            fetch_from = start_ms
            time.sleep(2 ** attempt)

    return []


def fetch_price_history(symbol, interval="1m", limit=60, retries=3):
    url    = f"{BINANCE_BASE}{KLINES_ENDPOINT}"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return np.array([float(k[4]) for k in resp.json()])
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Binance klines failed: {e}")
            time.sleep(2 ** attempt)
    return np.array([])

# =============================================================================
# FEATURE ENGINEERING
# Mirrors 04a_feature_engineering.py exactly
# =============================================================================

def build_10s_bars(timestamps, prices, qtys, is_buyer_side, window_end_ms, n_bars):
    """
    Bucket trades into n_bars contiguous 10-second bars ending at
    window_end_ms. Forward-fills price on empty bars, zero-fills volume --
    mirrors 04a's group_by_dynamic + upsample + forward_fill construction.

    Needed specifically for Kyle's lambda: training computes it as a rolling
    correlation over bar-level price diffs and signed volume, not trade-level
    values -- correlation and std don't decompose across bucket boundaries
    the way sums do, so this can't be approximated with trade-level data.

    Returns (bar_prices, bar_v_buy, bar_v_sell), oldest bar first.
    """
    bar_prices = np.full(n_bars, np.nan)
    bar_v_buy  = np.zeros(n_bars)
    bar_v_sell = np.zeros(n_bars)

    bar_start_ms = window_end_ms - n_bars * 10_000
    mask = (timestamps > bar_start_ms) & (timestamps <= window_end_ms)
    if not mask.any():
        return bar_prices, bar_v_buy, bar_v_sell

    ts_w, p_w, q_w, buy_w = timestamps[mask], prices[mask], qtys[mask], is_buyer_side[mask]
    bar_idx = np.clip(((ts_w - bar_start_ms - 1) // 10_000).astype(int), 0, n_bars - 1)

    for i in range(n_bars):
        sel = bar_idx == i
        if sel.any():
            bar_prices[i] = p_w[sel][-1]
            bar_v_buy[i]  = q_w[sel][buy_w[sel]].sum()
            bar_v_sell[i] = q_w[sel][~buy_w[sel]].sum()

    last_valid = None
    for i in range(n_bars):
        if np.isnan(bar_prices[i]):
            bar_prices[i] = last_valid if last_valid is not None else p_w[0]
        else:
            last_valid = bar_prices[i]

    return bar_prices, bar_v_buy, bar_v_sell


def compute_features_from_trades(trades, asset):
    if not trades:
        raise ValueError("Empty trades list")

    prices     = np.array([float(t["price"])           for t in trades])
    qtys       = np.array([float(t["qty"])              for t in trades])
    is_buyer   = np.array([bool(t["isBuyerMaker"])      for t in trades])
    timestamps = np.array([int(t["time"])               for t in trades])

    t_end    = timestamps[-1]
    volume   = qtys.sum()
    def window_features(window_seconds):
        cutoff = t_end - window_seconds * 1000
        mask_w = timestamps >= cutoff
        if mask_w.sum() < 2:
            mask_w = np.ones(len(trades), dtype=bool)

        p_w   = prices[mask_w]
        q_w   = qtys[mask_w]
        buy_w = ~is_buyer[mask_w]

        v_buy_w  = q_w[buy_w].sum()
        v_sell_w = q_w[~buy_w].sum()
        n_buy_w  = buy_w.sum()
        n_sell_w = (~buy_w).sum()
        n_w      = mask_w.sum()
        vol_w    = q_w.sum()

        vwap_w  = (p_w * q_w).sum() / (vol_w + EPSILON)
        lr_w    = np.diff(np.log(p_w + EPSILON))
        # RV must match training: sqrt(sum of squared log returns), not the
        # bare sum. 04a takes sqrt at the 10s-bucket level, then squares/rolls/
        # re-sqrts across the window -- net effect is sqrt(Sum r^2) over the
        # full window, i.e. realized volatility, not realized variance.
        rv_w    = np.sqrt((lr_w ** 2).sum()) if len(lr_w) > 0 else 0.0
        # ILLIQ numerator must be the sum of every trade's |log return| in
        # the window (path variation), not a single first-to-last displacement.
        # A window that whipsaws and ends flat should show high ILLIQ, not zero.
        abs_ret = np.abs(lr_w).sum() if len(lr_w) > 0 else 0.0

        ofi     = (v_buy_w - v_sell_w) / (v_buy_w + v_sell_w + EPSILON)
        tci     = (n_buy_w - n_sell_w) / (n_buy_w + n_sell_w + EPSILON)
        # Training divides trade count by window_seconds directly (w*10 bars
        # * 10s == window_seconds). Dividing by window_seconds/10 instead
        # made this exactly 10x too large.
        intens  = n_w / max(window_seconds, 1)
        illiq   = abs_ret / (vol_w + EPSILON)

        # Kyle's lambda: training computes this as rolling correlation over
        # w consecutive 10-second bars (w=6 for 60s, w=30 for 300s) of
        # bar-level price diffs and signed volume -- not trade-level values.
        # Correlation/std don't decompose across bucket boundaries the way
        # sums do, so this needs genuine bar aggregation, not a trade-level
        # approximation. w+1 bar prices are needed to get w diffs.
        if window_seconds > 10:
            n_bars = window_seconds // 10 + 1
            bar_prices, bar_v_buy, bar_v_sell = build_10s_bars(
                timestamps, prices, qtys, ~is_buyer, t_end, n_bars
            )
            dp_bar = np.diff(bar_prices)
            sv_bar = (bar_v_buy - bar_v_sell)[1:]
            if len(dp_bar) > 1 and dp_bar.std() > EPSILON and sv_bar.std() > EPSILON:
                kyle_lam = (
                    np.corrcoef(dp_bar, sv_bar)[0, 1]
                    * (dp_bar.std() / (sv_bar.std() + EPSILON))
                )
                if not np.isfinite(kyle_lam):
                    kyle_lam = 0.0
            else:
                kyle_lam = 0.0
        else:
            kyle_lam = 0.0

        return dict(ofi=float(ofi), tci=float(tci), intensity=float(intens),
                    vwap=float(vwap_w), illiq=float(illiq), rv=float(rv_w),
                    kyle_lam=float(kyle_lam))

    w10  = window_features(10)
    w60  = window_features(60)
    w300 = window_features(300)

    cp = float(prices[-1])

    if len(timestamps) > 2:
        dt   = np.diff(timestamps).astype(float)
        cv_dt = dt.std() / (dt.mean() + EPSILON)
    else:
        cv_dt = 0.0

    return {
        "price":           cp,
        "volume":          float(volume),
        # rv == RV_10s exactly in training (w=1 case collapses to the same
        # sqrt(sum r^2) formula) -- reuse it rather than recompute over a
        # mismatched trade scope.
        "rv":              w10["rv"],
        "OFI_10s":         w10["ofi"],   "TCI_10s":        w10["tci"],
        "intensity_10s":   w10["intensity"], "VWAP_10s":   w10["vwap"],
        "ILLIQ_10s":       w10["illiq"], "RV_10s":         w10["rv"],
        "OFI_60s":         w60["ofi"],   "TCI_60s":        w60["tci"],
        "intensity_60s":   w60["intensity"], "VWAP_60s":   w60["vwap"],
        "ILLIQ_60s":       w60["illiq"], "RV_60s":         w60["rv"],
        "Kyle_lambda_60s": w60["kyle_lam"],
        "OFI_300s":        w300["ofi"],  "TCI_300s":       w300["tci"],
        "intensity_300s":  w300["intensity"], "VWAP_300s": w300["vwap"],
        "ILLIQ_300s":      w300["illiq"], "RV_300s":       w300["rv"],
        "Kyle_lambda_300s": w300["kyle_lam"],
        "VWAP_dev_10s":    (cp - w10["vwap"])  / (w10["vwap"]  + EPSILON),
        "VWAP_dev_60s":    (cp - w60["vwap"])  / (w60["vwap"]  + EPSILON),
        "VWAP_dev_300s":   (cp - w300["vwap"]) / (w300["vwap"] + EPSILON),
        "CV_dt_10s":       float(cv_dt),
    }


def apply_fracdiff_to_price(features, asset, price_history):
    """
    Overwrites features["price"] with the fractionally differenced value,
    matching what 04b does to the training parquets. Uses asset-specific
    d-values (BTC=0.3, ETH=0.4, SOL=0.2) from config.py.
    """
    d = ASSET_D_VALUES.get(asset, 0.3)

    # Use a relaxed threshold at inference time -- training had years of history,
    # we only have ~80 bars. 1e-3 gives 55-73 weights, which fits comfortably.
    weights_needed = len(get_weights_ffd(d, thres=FRACDIFF_THRES_INFERENCE))

    if len(price_history) < weights_needed:
        logger.warning(
            f"  [{asset}] need {weights_needed} bars for fracdiff, "
            f"got {len(price_history)}. Using raw price."
        )
        return features

    full_series = np.append(price_history, features["price"])
    fd_series   = frac_diff_single(full_series, d, thres=FRACDIFF_THRES_INFERENCE)
    fd_value    = fd_series[-1]

    if not np.isfinite(fd_value):
        logger.warning(f"  [{asset}] fracdiff non-finite, using raw price.")
        features["price"] = float(features["price"])
    else:
        features["price"] = float(fd_value)

    return features

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(fold=DEFAULT_FOLD, seed=DEFAULT_SEED, asset=None):
    """Load model from local cache or GCS. asset=None uses the pooled model."""
    """
    Load model: local cache -> Hugging Face Hub (public) -> GCS (author only).
    Public users hit HF and never need gcp-key.json.
    """
    tag        = asset if asset else "pooled"
    filename   = f"xgb_binary_{tag}_fold{fold}_seed{seed}.pkl"
    local_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(local_path):
        logger.info(f"  Loading model from local cache: {local_path}")
        with open(local_path, "rb") as f:
            return pickle.load(f)

    # Try Hugging Face Hub first -- public repo, no credentials needed.
    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"  Downloading model from Hugging Face: {HF_REPO_ID}/{filename}")
        hf_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copy(hf_path, local_path)
        with open(local_path, "rb") as f:
            return pickle.load(f)
    except ImportError:
        logger.warning("  huggingface_hub not installed -- trying GCS")
    except Exception as e:
        logger.warning(f"  Hugging Face download failed ({e}) -- trying GCS")

    # Fallback: private GCS bucket. Only works with the author's own
    # gcp-key.json -- kept for internal use, not required for public users.
    if asset:
        gcs_path = MODEL_GCS_PATH_ASSET.format(seed=seed, fold=fold, asset=asset)
    else:
        gcs_path = MODEL_GCS_PATH_POOLED.format(seed=seed, fold=fold)

    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
            REPO_ROOT, "gcp-key.json"
        )
        from google.cloud import storage
        blob = storage.Client().bucket(BUCKET).blob(gcs_path)
        if not blob.exists():
            raise FileNotFoundError(f"Model not found at gs://{BUCKET}/{gcs_path}")
        pkl = pickle.loads(blob.download_as_bytes())
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(local_path, "wb") as f:
            pickle.dump(pkl, f)
        logger.info(f"  Cached to {local_path}")
        return pkl
    except Exception as e:
        raise RuntimeError(
            f"Could not load model from local cache, Hugging Face, or GCS.\n"
            f"Last error: {e}\n"
            f"To fix: place the model pickle manually at {local_path}, or "
            f"pip install huggingface_hub and check {HF_REPO_ID} is reachable."
        )

# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(asset, model_pkl, price_history):
    model   = model_pkl["model"]
    scaler  = model_pkl["scaler"]
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        trades = fetch_agg_trades_window(asset, window_seconds=BAR_SECONDS + FETCH_PADDING_S)
        if len(trades) < 10:
            return _err(asset, ts,
                f"Insufficient aggTrades ({len(trades)}) in last "
                f"{BAR_SECONDS}s window — market may be extremely thin")
    except Exception as e:
        return _err(asset, ts, f"Binance API error: {e}")

    try:
        features = compute_features_from_trades(trades, asset)
    except Exception as e:
        return _err(asset, ts, f"Feature error: {e}")

    # Save raw price for display BEFORE fracdiff overwrites it
    raw_price = features["price"]
    features = apply_fracdiff_to_price(features, asset, price_history)
    # Determine if this is a pooled model by checking if scaler
    # was trained on a feature set that includes asset_id
    # (pooled models have one extra feature dimension)
    n_scaler_features = scaler.n_features_in_
    is_pooled = (n_scaler_features == len(POOLED_FEATURES))
    model_features = get_model_features(is_pooled)

    if is_pooled:
        features["asset_id"] = ASSET_ID_MAP[asset]

    for f in model_features:
        if f not in features:
            logger.warning(f"  [{asset}] Missing feature {f} — filling 0")
            features[f] = 0.0

    X = np.array([[features[f] for f in model_features]], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        X_sc   = scaler.transform(X)
        probs  = model.predict_proba(X_sc)[0]
        classes = list(model.classes_)
        if 1 in classes:
            stress_prob = float(probs[classes.index(1)])
        elif 2 in classes:
            stress_prob = float(probs[classes.index(2)])
        else:
            stress_prob = float(probs[-1])
    except Exception as e:
        return _err(asset, ts, f"Inference error: {e}")

    # Continuity condition
    # _stress_history: rolling window for display (last 288 bars = 24h)
    # _stress_duration: true consecutive count, never capped
    hist = _stress_history[asset]
    hist.append(stress_prob)
    if len(hist) > 288:
        hist.pop(0)

    # Update true duration counter
    if stress_prob > STRESS_THRESHOLD:
        _stress_duration[asset] += 1
    else:
        _stress_duration[asset] = 0

    consecutive = _stress_duration[asset]
    warning     = consecutive >= MIN_CONSECUTIVE

    if stress_prob >= STRESS_THRESHOLD:
        regime = "stress"
    elif stress_prob >= 0.5:
        regime = "elevated"
    else:
        regime = "calm"

    result = {
        "timestamp":         ts,
        "asset":             asset,
        "price":             round(raw_price, 4),  # raw price for display
        "stress_prob":       round(stress_prob, 4),
        "regime":            regime,
        "warning_active":    warning,
        "consecutive_bars":  consecutive,
        "alert_triggered":   warning,
        "threshold":         STRESS_THRESHOLD,
        "n_trades_used":     len(trades),
        "window_seconds":    BAR_SECONDS,
        "key_features": {
            "OFI_300s":          round(features.get("OFI_300s", 0), 4),
            "RV_300s":           round(features.get("RV_300s", 0), 6),
            "Kyle_lambda_300s":  round(features.get("Kyle_lambda_300s", 0), 4),
            "intensity_300s":    round(features.get("intensity_300s", 0), 4),
            "price_fracdiff":    round(features.get("price", 0), 6),  # model input
        },
        "interpretation": _interpret(stress_prob, warning, consecutive, features),
    }

    # Log every inference to inference_log.csv
    log_inference(result, raw_price)

    # If warning active, log pending outcome (checked 30 min later)
    if result["warning_active"]:
        log_pending_outcome(result, raw_price)

    return result


def _interpret(prob, warning, consec, feats):
    lines = []
    if prob >= 0.85:
        lines.append(
            f"HIGH STRESS P={prob:.1%}: Order flow and liquidity indicate "
            f"acute microstructure stress."
        )
    elif prob >= 0.5:
        lines.append(f"ELEVATED P={prob:.1%}: Microstructure shows elevated stress.")
    else:
        lines.append(f"CALM P={prob:.1%}: Microstructure consistent with normal conditions.")

    if warning:
        lines.append(
            f"WARNING ACTIVE for {consec} bars (~{consec*5} min). "
            f"Microstructure stress reflects deteriorating LIQUIDITY CONDITIONS "
            f"— elevated sell pressure, thin depth, abnormal trade flow. "
            f"Price impact is NOT guaranteed: liquid markets may absorb stress "
            f"without significant price movement. "
            f"For acute structural events, historical lead time is 1-3 hours "
            f"(Terra-Luna: 108min, FTX: 176min). Monitor for sustained deterioration."
        )

    ofi = feats.get("OFI_300s", 0)
    if abs(ofi) > 0.3:
        d = "sell-side" if ofi < 0 else "buy-side"
        lines.append(f"Extreme {d} OFI={ofi:.3f}.")

    return " | ".join(lines)


def _err(asset, ts, msg):
    logger.error(f"  [{asset}] {msg}")
    return {"timestamp": ts, "asset": asset, "error": msg,
            "stress_prob": None, "regime": "unknown", "warning_active": False}

# =============================================================================
# DISPLAY
# =============================================================================

def print_result(result, as_json=False):
    if as_json:
        print(json.dumps(result, indent=2))
        return

    asset = result["asset"]
    if "error" in result:
        print(f"\n{RED}[{asset}] ERROR: {result['error']}{RESET}")
        return

    prob    = result["stress_prob"]
    regime  = result["regime"]
    warning = result["warning_active"]
    consec  = result["consecutive_bars"]

    c = RED if regime == "stress" else (YELLOW if regime == "elevated" else GREEN)
    alert = (f"\n  {RED}{BOLD}WARNING ACTIVE — {consec} consecutive bars "
             f"(~{consec*5} min){RESET}") if warning else ""

    print(f"""
{BOLD}{'─'*60}{RESET}
{BOLD}{asset}{RESET}  {result['timestamp']}
  Price:        ${result['price']:,.4f}
  Stress prob:  {c}{BOLD}{prob:.1%}{RESET}
  Regime:       {c}{regime.upper()}{RESET}
  Trades used:  {result['n_trades_used']:,} (last {result.get('window_seconds', 300)}s window)
  Note:         Stress = liquidity conditions, NOT a price prediction{alert}

  Key microstructure signals:
    OFI 300s:       {result['key_features']['OFI_300s']:+.4f}
    RV  300s:       {result['key_features']['RV_300s']:.6f}
    Kyle λ 300s:    {result['key_features']['Kyle_lambda_300s']:+.4f}
    Intensity 300s: {result['key_features']['intensity_300s']:.4f}

  {result['interpretation']}
""")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MAIC: Early Warning System — Live Stress Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/12_inference.py --asset BTCUSDT
  python3 scripts/12_inference.py --asset all
  python3 scripts/12_inference.py --asset all --loop
  python3 scripts/12_inference.py --asset BTCUSDT --json
  python3 scripts/12_inference.py --asset ETHUSDT --fold 3 --seed 777
        """
    )
    parser.add_argument("--asset",  default="BTCUSDT",
                        choices=ASSETS + ["all"])
    parser.add_argument("--loop",   action="store_true",
                        help="Run continuously every 5 minutes")
    parser.add_argument("--json",   action="store_true",
                        help="Output as JSON")
    parser.add_argument("--fold",   type=int, default=DEFAULT_FOLD)
    parser.add_argument("--seed",   type=int, default=DEFAULT_SEED)
    parser.add_argument("--asset-specific", action="store_true",
                        help="[DISABLED] Asset-specific models were never trained "
                             "at production (5-seed) rigor -- see note in code.")
    parser.add_argument("--quiet",  action="store_true",
                        help="Suppress INFO logs")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    assets = ASSETS if args.asset == "all" else [args.asset]

    # Pooled is the only production-grade model. 06d_train_production.py runs
    # pooled only across all 5 seeds by design (asset-specific training was
    # scoped to 06b's single-seed exploratory run, not the production sweep).
    # --asset-specific is kept as a flag for visibility but currently errors:
    # there is no 5-seed asset-specific model to load.
    if getattr(args, "asset_specific", False):
        logger.error(
            "--asset-specific is disabled: production training (06d) only "
            "produces pooled models across all 5 seeds. Single-seed "
            "asset-specific models exist under v2/results_run1/{asset}/ from "
            "06b's exploratory run, but aren't production-grade. Use the "
            "pooled model (default) instead."
        )
        sys.exit(1)
    model_asset = None

    logger.info(
        f"Loading {'asset-specific' if model_asset else 'pooled'} model "
        f"(fold={args.fold}, seed={args.seed})..."
    )
    try:
        model_pkl = load_model(fold=args.fold, seed=args.seed, asset=model_asset)
        logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    def run_once():
        # Check any pending outcomes whose 30-min window has elapsed
        check_pending_outcomes()

        for asset in assets:
            logger.info(f"Fetching live data for {asset}...")
            try:
                price_history = fetch_price_history(asset, limit=80)
            except Exception as e:
                logger.warning(f"  [{asset}] Price history unavailable: {e}")
                price_history = np.array([])
            result = run_inference(asset, model_pkl, price_history)
            print_result(result, as_json=args.json)

    if not args.loop:
        run_once()
    else:
        print(f"\n{BOLD}MAIC Early Warning System — Continuous Monitoring{RESET}")
        print(f"Assets: {', '.join(assets)}")
        print(f"Alert: P(stress) > {STRESS_THRESHOLD} for >= {MIN_CONSECUTIVE} bars")
        print("Press Ctrl+C to stop.\n")
        try:
            while True:
                run_once()
                print(f"\n{BOLD}Next update in 300 seconds...{RESET}")
                time.sleep(300)
        except KeyboardInterrupt:
            print(f"\n{BOLD}Monitoring stopped.{RESET}")


if __name__ == "__main__":
    main()