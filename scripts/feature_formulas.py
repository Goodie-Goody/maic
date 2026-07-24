"""
feature_formulas.py -- single source of truth for microstructure feature math

Every formula here is deliberately small and pure: given already-aggregated
inputs, return one feature value. No I/O, no framework dependency, no state.

This exists because 12_inference.py and 04a_feature_engineering.py compute
the same seven features on structurally different data -- a numpy array of
a few hundred trades vs. a polars LazyFrame spanning a month -- and
hand-mirroring the math between the two is exactly how four real bugs
(missing sqrt in RV, 10x error in intensity, wrong ILLIQ numerator, trade-
level instead of bar-level Kyle's lambda) happened in the first place.

12_inference.py imports these functions directly rather than reimplementing
the math -- zero drift risk for the live side. 04a's polars expressions
can't call these per-row without a severe performance cost at month-scale
(polars' map_elements is slow), so training remains vectorized and
separately written. test_feature_parity.py is what keeps the two honest:
it feeds identical synthetic bucket data through both this module and 04a's
actual polars expressions and asserts they agree.
"""

import numpy as np

EPSILON = 1e-8


def ofi(v_buy: float, v_sell: float) -> float:
    """Order flow imbalance: buy volume minus sell volume, normalized to [-1, 1]."""
    return float((v_buy - v_sell) / (v_buy + v_sell + EPSILON))


def tci(n_buy: float, n_sell: float) -> float:
    """Trade concentration index: buy count minus sell count, normalized."""
    return float((n_buy - n_sell) / (n_buy + n_sell + EPSILON))


def intensity(n_total: float, window_seconds: float) -> float:
    """Trades per second over the window. Must divide by window_seconds
    directly -- dividing by window_seconds/10 makes this 10x too large."""
    return float(n_total / max(window_seconds, 1))


def vwap(quote_volume: float, volume: float) -> float:
    """Volume-weighted average price."""
    return float(quote_volume / (volume + EPSILON))


def vwap_deviation(price: float, vwap_value: float) -> float:
    """Current price's deviation from VWAP, normalized."""
    return float((price - vwap_value) / (vwap_value + EPSILON))


def illiq(abs_log_ret_sum: float, volume: float) -> float:
    """
    Amihud-style illiquidity: total absolute-return path variation per
    unit volume. abs_log_ret_sum must be the SUM of every trade's |log
    return| within the window, not a single net displacement -- a window
    that whipsaws and ends flat should show high ILLIQ, not zero.
    """
    return float(abs_log_ret_sum / (volume + EPSILON))


def realized_volatility(sum_sq_log_ret: float) -> float:
    """
    RV = sqrt(sum of squared log returns) -- realized volatility, not
    variance. sum_sq_log_ret must already be a sum of squares; this
    function only takes the final sqrt.
    """
    return float(np.sqrt(max(sum_sq_log_ret, 0.0)))


def clip_log_return(log_ret: np.ndarray, bound: float = 0.5) -> np.ndarray:
    """
    Bound log returns to [-bound, bound] before they feed RV/ILLIQ, guarding
    against a bad print or data glitch producing one absurd return that
    dominates an otherwise-normal window. Matches 04a's clip exactly.
    """
    return np.clip(log_ret, -bound, bound)


def kyle_lambda(dp_bars: np.ndarray, sv_bars: np.ndarray) -> float:
    """
    Price impact coefficient: correlation between bar-to-bar price changes
    and bar-level signed volume, scaled by their relative dispersion.

    MUST be computed on bar-level series (10-second bars), not trade-level
    values -- correlation and standard deviation don't decompose across
    aggregation boundaries the way sums do, so trade-level correlation is a
    genuinely different statistic, not an approximation of the bar-level
    one, and can even carry a different sign.
    """
    dp_bars = np.asarray(dp_bars)
    sv_bars = np.asarray(sv_bars)
    if len(dp_bars) < 2 or dp_bars.std() <= EPSILON or sv_bars.std() <= EPSILON:
        return 0.0
    result = np.corrcoef(dp_bars, sv_bars)[0, 1] * (dp_bars.std() / (sv_bars.std() + EPSILON))
    return float(result) if np.isfinite(result) else 0.0