import os
import tempfile
import logging
import numpy as np
import polars as pl
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.signal import lfilter
from google.cloud import storage
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Config
BUCKET_NAME = "fe-binance-data-2025" # Replace with your exact bucket name if different
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SOURCE_PREFIX = "v2/features/"
DEST_PREFIX = "v2/features_fracdiff/"
P_VALUE_THRESHOLD = 0.05
MAX_SAMPLES_FOR_ADF = 100000  # We don't need millions of rows just to find 'd'

# Columns we suspect need differencing
POTENTIAL_NON_STATIONARY = [
    "price", "volume", 
    "VWAP_10s", "VWAP_60s", "VWAP_300s"
]

def get_weights_ffd(d, thres=1e-5):
    """
    Generates the weights for fractional differencing.
    (López de Prado, 2018)
    """
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w)

def fast_frac_diff(series, d, thres=1e-5):
    """
    Applies fractional differencing using SciPy's fast C-level linear filter.
    This is thousands of times faster than Pandas .rolling().apply()
    """
    # Get weights and pad to match lengths
    w = get_weights_ffd(d, thres)
    
    # Fill NaNs with forward fill, then 0, just to be safe before convolution
    series_filled = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply filter: y = w * x
    diff_series = lfilter(w, [1.0], series_filled)
    return diff_series

def find_min_d(series, thres=1e-5):
    """
    Grid search to find the minimum d in [0.1, 0.9] that passes the ADF test.
    """
    for d in np.arange(0.1, 1.0, 0.1):
        diffed = fast_frac_diff(series, d, thres)
        # Drop the initial window where the filter is warming up
        valid_diffed = diffed[len(get_weights_ffd(d, thres)):] 
        
        if len(valid_diffed) > 100:
            adf_stat, p_value, _, _, _, _ = adfuller(valid_diffed)
            if p_value < P_VALUE_THRESHOLD:
                return round(d, 2)
    return 1.0 # Fallback to integer differencing if memory cannot be preserved

def process_file(bucket, blob, d_map):
    """
    Downloads a single parquet file, applies the mapped d-values, and re-uploads.
    """
    filename = blob.name.split("/")[-1]
    dest_path = f"{DEST_PREFIX}{filename}"
    
    # Check if already processed
    if bucket.blob(dest_path).exists():
        logger.info(f"  Skipping {filename}, already fractionally differenced.")
        return

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    
    try:
        blob.download_to_filename(tmp_in.name)
        df = pl.read_parquet(tmp_in.name)
        
        # Apply FD to the required columns
        for col, d in d_map.items():
            if col in df.columns:
                series = df[col].to_numpy()
                fd_series = fast_frac_diff(series, d)
                df = df.with_columns(pl.Series(col, fd_series))
        
        # Save and upload
        df.write_parquet(tmp_out.name)
        bucket.blob(dest_path).upload_from_filename(tmp_out.name)
        logger.info(f"  Processed & Uploaded: {dest_path}")
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
    finally:
        os.remove(tmp_in.name)
        os.remove(tmp_out.name)

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    
    logger.info("Step 1: Identifying Non-Stationary Columns and calculating minimum 'd'...")
    d_map = {}
    
    # Download a sample file to calculate 'd' (e.g., BTCUSDT first month)
    sample_blobs = list(client.list_blobs(BUCKET_NAME, prefix=f"{SOURCE_PREFIX}BTCUSDT-features-2020", max_results=1))
    if not sample_blobs:
        sample_blobs = list(client.list_blobs(BUCKET_NAME, prefix=f"{SOURCE_PREFIX}BTCUSDT", max_results=1))
        
    sample_blob = sample_blobs[0]
    tmp_sample = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    sample_blob.download_to_filename(tmp_sample.name)
    df_sample = pl.read_parquet(tmp_sample.name).head(MAX_SAMPLES_FOR_ADF)
    os.remove(tmp_sample.name)
    
    for col in POTENTIAL_NON_STATIONARY:
        if col in df_sample.columns:
            series = df_sample[col].to_numpy()
            _, p_value, _, _, _, _ = adfuller(series)
            
            if p_value > P_VALUE_THRESHOLD:
                logger.info(f"  [FAIL] {col} is non-stationary (p={p_value:.4f}). Searching for optimal d...")
                d_opt = find_min_d(series)
                d_map[col] = d_opt
                logger.info(f"  [SUCCESS] Optimal d for {col} = {d_opt}")
            else:
                logger.info(f"  [PASS] {col} is already stationary (p={p_value:.4f}).")

    if not d_map:
        logger.info("No columns require fractional differencing. Exiting.")
        return
        
    logger.info(f"Final Differencing Map: {d_map}")
    logger.info("Step 2: Applying Fractional Differencing across all assets in parallel...")
    
    # Gather all feature files
    all_blobs = list(client.list_blobs(BUCKET_NAME, prefix=SOURCE_PREFIX))
    parquet_blobs = [b for b in all_blobs if b.name.endswith(".parquet")]
    
    # Process files in parallel using CPU cores
    max_workers = os.cpu_count() or 4
    logger.info(f"Spinning up {max_workers} CPU threads for processing...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, bucket, blob, d_map) for blob in parquet_blobs]
        concurrent.futures.wait(futures)

    logger.info("==================================================")
    logger.info("Fractional Differencing Pipeline Complete.")
    logger.info(f"New features available in gs://{BUCKET_NAME}/{DEST_PREFIX}")

if __name__ == "__main__":
    main()
    