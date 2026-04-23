import sys
import os
import tempfile
import logging
import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller
from scipy.signal import lfilter
from google.cloud import storage
import concurrent.futures

# Import pipeline config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ASSETS, BUCKET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Config
SOURCE_PREFIX = "v2/features/"
DEST_PREFIX = "v2/features_fracdiff/"
P_VALUE_THRESHOLD = 0.05
MAX_SAMPLES_FOR_ADF = 100000  # We don't need millions of rows just to find 'd'

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
    """
    w = get_weights_ffd(d, thres)
    # Fill NaNs with 0.0 before convolution
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
        valid_diffed = diffed[len(get_weights_ffd(d, thres)):] 
        
        if len(valid_diffed) > 100:
            adf_stat, p_value, _, _, _, _ = adfuller(valid_diffed)
            if p_value < P_VALUE_THRESHOLD:
                return round(d, 2)
    return 1.0 

def process_file(bucket, blob, asset_d_map):
    """
    Downloads a single parquet file, applies the mapped d-values, and re-uploads.
    If no differencing is needed, executes a fast cloud-to-cloud copy.
    """
    filename = blob.name.split("/")[-1]
    dest_path = f"{DEST_PREFIX}{filename}"
    
    if bucket.blob(dest_path).exists():
        logger.info(f"  Skipping {filename}, already processed.")
        return

    # Fast path: If no non-stationary columns for this asset, just copy it over
    if not asset_d_map:
        bucket.copy_blob(blob, bucket, new_name=dest_path)
        logger.info(f"  Copied without differencing: {filename}")
        return

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    
    try:
        blob.download_to_filename(tmp_in.name)
        df = pl.read_parquet(tmp_in.name)
        
        # Apply FD to the required columns
        for col, d in asset_d_map.items():
            if col in df.columns:
                series = df[col].to_numpy()
                fd_series = fast_frac_diff(series, d)
                
                # Zero out warmup period to avoid garbage values
                warmup = len(get_weights_ffd(d))
                fd_series[:warmup] = 0.0 
                
                df = df.with_columns(pl.Series(col, fd_series))
        
        # Save and upload
        df.write_parquet(tmp_out.name)
        bucket.blob(dest_path).upload_from_filename(tmp_out.name)
        logger.info(f"  Differenced & Uploaded: {filename}")
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
    finally:
        if os.path.exists(tmp_in.name): os.remove(tmp_in.name)
        if os.path.exists(tmp_out.name): os.remove(tmp_out.name)

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    
    logger.info("Step 1: Calculating optimal 'd' per asset dynamically...")
    global_d_maps = {}
    
    for asset in ASSETS:
        logger.info(f"--- Testing Stationarity for {asset} ---")
        
        # Grab one sample file for the specific asset
        sample_blobs = list(client.list_blobs(BUCKET, prefix=f"{SOURCE_PREFIX}{asset}", max_results=1))
        if not sample_blobs:
            logger.warning(f"No files found for {asset}. Skipping.")
            continue
            
        sample_blob = sample_blobs[0]
        tmp_sample = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        sample_blob.download_to_filename(tmp_sample.name)
        
        df_sample = pl.read_parquet(tmp_sample.name).head(MAX_SAMPLES_FOR_ADF)
        os.remove(tmp_sample.name)
        
        # Test all numeric columns dynamically
        cols_to_test = [c for c in df_sample.columns if c != "time" and df_sample[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        
        asset_d_map = {}
        for col in cols_to_test:
            series = df_sample[col].to_numpy()
            
            # Skip completely empty or constant columns
            if np.all(series == series[0]):
                continue
                
            _, p_value, _, _, _, _ = adfuller(series)
            
            if p_value > P_VALUE_THRESHOLD:
                d_opt = find_min_d(series)
                asset_d_map[col] = d_opt
                logger.info(f"  [FAIL] {col} (p={p_value:.4f}) -> Assigned d={d_opt}")
            
        global_d_maps[asset] = asset_d_map
        logger.info(f"  Final Map for {asset}: {asset_d_map if asset_d_map else 'All columns stationary'}")

    logger.info("\nStep 2: Applying Fractional Differencing in parallel...")
    
    all_blobs = list(client.list_blobs(BUCKET, prefix=SOURCE_PREFIX))
    parquet_blobs = [b for b in all_blobs if b.name.endswith(".parquet")]
    
    max_workers = os.cpu_count() or 4
    logger.info(f"Spinning up {max_workers} CPU threads for processing...")
    
    # Helper to map the right asset map to the file
    def process_wrapper(blob):
        asset = next((a for a in ASSETS if a in blob.name), None)
        if asset:
            process_file(bucket, blob, global_d_maps.get(asset, {}))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_wrapper, blob) for blob in parquet_blobs]
        concurrent.futures.wait(futures)

    logger.info("==================================================")
    logger.info("Fractional Differencing Pipeline Complete.")
    logger.info(f"New features available in gs://{BUCKET}/{DEST_PREFIX}")

if __name__ == "__main__":
    main()
    