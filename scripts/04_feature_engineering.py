import sys
import os
import gc
import logging
import tempfile
import psutil

import polars as pl
from google.cloud import storage
from datetime import datetime
from calendar import monthrange

# Ensure config is accessible
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ASSETS, BUCKET, WINDOWS, PARQUET_COMPRESSION
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

INPUT_PREFIX = "v2/trades_parquet_flat/"
OUTPUT_PREFIX = "v2/features/"
EPSILON = 1e-8  # Prevent division by zero

def parse_window_months(start_ym, end_ym):
    months = []
    sy, sm = int(start_ym[:4]), int(start_ym[5:])
    ey, em = int(end_ym[:4]), int(end_ym[5:])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months

def gcs_blob_exists(bucket, blob_path):
    return bucket.blob(blob_path).exists()

def compute_single_asset_features(file_path):
    """Streams gigabytes of raw trades into a tiny, memory-safe 10s feature DataFrame"""
    lf = pl.scan_parquet(file_path)

    # 1. Base transformations (Tick level)
    lf = lf.with_columns([
        pl.when(~pl.col("is_buyer_maker")).then(pl.col("qty")).otherwise(0).alias("v_buy"),
        pl.when(pl.col("is_buyer_maker")).then(pl.col("qty")).otherwise(0).alias("v_sell"),
        pl.when(~pl.col("is_buyer_maker")).then(1).otherwise(0).alias("n_buy"),
        pl.when(pl.col("is_buyer_maker")).then(1).otherwise(0).alias("n_sell"),
        pl.col("price").log().diff().fill_null(0.0).clip(-0.5, 0.5).alias("log_ret"),
        pl.col("time").diff().dt.total_microseconds().alias("dt_micro")
    ])

    # 2. Dynamic 10-second Aggregation
    agg = lf.group_by_dynamic("time", every="10s").agg([
        pl.col("v_buy").sum().alias("v_buy"),
        pl.col("v_sell").sum().alias("v_sell"),
        pl.col("n_buy").sum().alias("n_buy"),
        pl.col("n_sell").sum().alias("n_sell"),
        pl.col("price").last().alias("price"),
        (pl.col("price") * pl.col("qty")).sum().alias("quote_volume"),
        pl.col("qty").sum().alias("volume"),
        (pl.col("log_ret").pow(2).sum()).sqrt().alias("rv"),
        pl.col("dt_micro").mean().alias("mean_dt"),
        pl.col("dt_micro").std().alias("std_dt"),
        pl.col("log_ret").abs().sum().alias("abs_ret_sum")
    ])

    # Execute computation out-of-core and bring the small aggregated frame to memory
    df = agg.collect(engine="streaming")

    # 3. Upsample to enforce strict chronological sequence
    df = df.upsample(time_column="time", every="10s").with_columns([
        pl.col("v_buy").fill_null(0),
        pl.col("v_sell").fill_null(0),
        pl.col("n_buy").fill_null(0),
        pl.col("n_sell").fill_null(0),
        pl.col("volume").fill_null(0),
        pl.col("quote_volume").fill_null(0),
        pl.col("rv").fill_null(0),
        pl.col("abs_ret_sum").fill_null(0),
        pl.col("price").forward_fill(),
    ])

    # Pre-compute bases for Kyle's Lambda at the 10s base level
    df = df.with_columns([
        pl.col("price").diff().fill_null(0.0).alias("dp_10s"),
        (pl.col("v_buy") - pl.col("v_sell")).alias("sv_10s")
    ])

    # 4. Multi-Scale Feature Engineering
    windows = {"10s": 1, "60s": 6, "300s": 30}
    exprs = []

    for name, w in windows.items():
        v_buy_w = pl.col("v_buy").rolling_sum(window_size=w)
        v_sell_w = pl.col("v_sell").rolling_sum(window_size=w)
        n_buy_w = pl.col("n_buy").rolling_sum(window_size=w)
        n_sell_w = pl.col("n_sell").rolling_sum(window_size=w)
        vol_w = pl.col("volume").rolling_sum(window_size=w)
        quote_w = pl.col("quote_volume").rolling_sum(window_size=w)
        abs_ret_w = pl.col("abs_ret_sum").rolling_sum(window_size=w)
        rv_w = (pl.col("rv").pow(2).rolling_sum(window_size=w)).sqrt()

        # Imbalance and Intensity
        exprs.extend([
            ((v_buy_w - v_sell_w) / (v_buy_w + v_sell_w + EPSILON)).alias(f"OFI_{name}"),
            ((n_buy_w - n_sell_w) / (n_buy_w + n_sell_w + EPSILON)).alias(f"TCI_{name}"),
            ((n_buy_w + n_sell_w) / (w * 10)).alias(f"intensity_{name}"),
        ])

        # Impact and Volatility
        vwap_expr = quote_w / (vol_w + EPSILON)
        exprs.extend([
            vwap_expr.alias(f"VWAP_{name}"),
            (abs_ret_w / (vol_w + EPSILON)).alias(f"ILLIQ_{name}"),
            rv_w.alias(f"RV_{name}")
        ])

        # Kyle's Lambda (Requires w > 1 for std dev and correlation to work)
        if w > 1:
            lambda_w = (
                pl.rolling_corr("dp_10s", "sv_10s", window_size=w) * (pl.col("dp_10s").rolling_std(window_size=w) / (pl.col("sv_10s").rolling_std(window_size=w) + EPSILON))
            ).fill_null(0.0).alias(f"Kyle_lambda_{name}")
            exprs.append(lambda_w)

    df = df.with_columns(exprs)

    # Group 4: VWAP Deviation & CV Duration
    df = df.with_columns([
        ((pl.col("price") - pl.col("VWAP_10s")) / (pl.col("VWAP_10s") + EPSILON)).alias("VWAP_dev_10s"),
        ((pl.col("price") - pl.col("VWAP_60s")) / (pl.col("VWAP_60s") + EPSILON)).alias("VWAP_dev_60s"),
        ((pl.col("price") - pl.col("VWAP_300s")) / (pl.col("VWAP_300s") + EPSILON)).alias("VWAP_dev_300s"),
        (pl.col("std_dt") / (pl.col("mean_dt") + EPSILON)).fill_null(0.0).alias("CV_dt_10s")
    ])

    # Drop intermediate base columns to save space
    drop_cols = ["v_buy", "v_sell", "n_buy", "n_sell", "quote_volume", "abs_ret_sum", "mean_dt", "std_dt", "dp_10s", "sv_10s"]
    return df.drop(drop_cols)

def compute_cross_asset_features(target_df, lead_df, lead_name):
    """Computes Group 5 Contagion features dynamically based on a lead asset"""
    df = target_df.join(
        lead_df.select(["time", "OFI_60s", "OFI_300s"]), 
        on="time", 
        how="left", 
        suffix=f"_{lead_name}"
    )

    df = df.with_columns([
        pl.rolling_corr("OFI_60s", f"OFI_60s_{lead_name}", window_size=6).fill_null(0.0).alias(f"OFI_corr_60s_{lead_name}"),
        pl.rolling_corr("OFI_300s", f"OFI_300s_{lead_name}", window_size=30).fill_null(0.0).alias(f"OFI_corr_300s_{lead_name}"),
        pl.rolling_corr("OFI_60s", pl.col(f"OFI_60s_{lead_name}").shift(1), window_size=6).fill_null(0.0).alias(f"Lead_Lag_60s_{lead_name}")
    ])
    
    return df.drop([f"OFI_60s_{lead_name}", f"OFI_300s_{lead_name}"])

def main():
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(BUCKET)

    logger.info("Starting Multi-Scale Feature Engineering Pipeline (v2.2)")
    failed_months = []

    for start_ym, end_ym in WINDOWS:
        for year, month in parse_window_months(start_ym, end_ym):
            logger.info(f"=== Processing Month: {year}-{month:02d} ===")
            frames = {}
            
            # Step 1: Populate frames (Compute from raw OR load existing)
            for asset in ASSETS:
                if asset == "SOLUSDT" and (year, month) < (2020, 11): 
                    continue

                output_blob = f"{OUTPUT_PREFIX}{asset}-features-{year}-{month:02d}.parquet"
                
                # Check if feature file already exists to save compute time
                if gcs_blob_exists(bucket, output_blob):
                    logger.info(f"  {asset} output exists, loading into frames for cross-asset use...")
                    tmp_existing = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                    bucket.blob(output_blob).download_to_filename(tmp_existing.name)
                    frames[asset] = pl.read_parquet(tmp_existing.name)
                    os.remove(tmp_existing.name)
                    continue

                input_blob = f"{INPUT_PREFIX}{asset}-trades-{year}-{month:02d}.parquet"
                
                if not gcs_blob_exists(bucket, input_blob):
                    logger.warning(f"  Missing raw data for {asset}, skipping.")
                    continue
                
                logger.info(f"  Computing single-asset features for {asset}...")
                blob = bucket.blob(input_blob)
                tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                blob.download_to_filename(tmp_raw.name)

                try:
                    df = compute_single_asset_features(tmp_raw.name)
                    frames[asset] = df
                except Exception as e:
                    logger.error(f"  {asset} {year}-{month:02d} failed: {e}")
                    failed_months.append((asset, year, month))
                finally:
                    os.remove(tmp_raw.name)
                    gc.collect()

            if not frames:
                continue

            logger.info(f"  RAM used: {psutil.virtual_memory().percent}%")

            # Step 2: Compute Cross-Asset Features & Upload (Skip upload if output already exists)
            for asset, df in frames.items():
                output_blob = f"{OUTPUT_PREFIX}{asset}-features-{year}-{month:02d}.parquet"

                if gcs_blob_exists(bucket, output_blob):
                    logger.info(f"  {asset} output already exists, skipping upload.")
                    continue
                
                # Apply BTC Contagion to ETH and SOL
                if asset != "BTCUSDT" and "BTCUSDT" in frames:
                    logger.info(f"  Applying BTC Cross-Asset features to {asset}...")
                    df = compute_cross_asset_features(df, frames["BTCUSDT"], "BTC")
                
                # Apply ETH Contagion strictly to SOL
                if asset == "SOLUSDT" and "ETHUSDT" in frames:
                    logger.info(f"  Applying ETH Cross-Asset features to {asset}...")
                    df = compute_cross_asset_features(df, frames["ETHUSDT"], "ETH")
                
                logger.info(f"  {asset} feature shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
                logger.info(f"  Uploading completed feature set for {asset}...")
                
                tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
                df.write_parquet(tmp_out.name, compression=PARQUET_COMPRESSION)
                
                bucket.blob(output_blob).upload_from_filename(tmp_out.name)
                os.remove(tmp_out.name)

    if failed_months:
        logger.error("Failed months:")
        for asset, year, month in failed_months:
            logger.error(f"  {asset} {year}-{month:02d}")
        sys.exit(1)

    logger.info("Feature Engineering Complete.")

if __name__ == "__main__":
    main()

