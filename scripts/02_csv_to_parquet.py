import sys
import os
import gc
import logging
import tempfile
import zipfile

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import pyarrow.compute as pc
from google.cloud import storage
from datetime import datetime, timezone
from calendar import monthrange

# Ensure config is accessible
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ASSETS, BUCKET, WINDOWS, PARQUET_COMPRESSION, SCHEMA,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MONTHLY_ZIP_PREFIX = "raw/zips/monthly/"
DAILY_ZIP_PREFIX = "raw/zips/daily/"
OUTPUT_PREFIX = "v2/trades_parquet_flat/"
MIN_OUTPUT_BYTES = 10 * 1024 * 1024

# 1. Final Parquet Schema (Time is UTC Timestamp)
PYARROW_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("price", pa.float64()),
    ("qty", pa.float64()),
    ("quote_qty", pa.float64()),
    ("time", pa.timestamp("us", tz="UTC")),
    ("is_buyer_maker", pa.bool_()),
    ("is_best_match", pa.bool_()),
])

# 2. Raw CSV Schema (Time is raw int64 milliseconds, no headers in file)
CSV_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("price", pa.float64()),
    ("qty", pa.float64()),
    ("quote_qty", pa.float64()),
    ("time", pa.int64()), 
    ("is_buyer_maker", pa.bool_()),
    ("is_best_match", pa.bool_()),
])

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

def gcs_blob_size(bucket, blob_path):
    blob = bucket.blob(blob_path)
    blob.reload()
    return blob.size

def download_zip_to_disk(bucket, blob_path):
    """Downloads to a temp file on disk to save RAM."""
    blob = bucket.blob(blob_path)
    blob.reload()
    size_mb = blob.size / 1024 / 1024
    logger.info(f"  Downloading {blob_path} ({size_mb:.1f} MB) to disk")
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    blob.download_to_filename(tmp.name)
    return tmp.name

def process_csv_from_zip(zip_path, writer, tag):
    """Processes CSVs using PyArrow's C++ streaming engine for the Fast Route."""
    rows_written = 0
    
    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            logger.warning(f"  {tag} no CSV files found in zip")
            return 0
            
        # ANOMALY FIX: Guard against duplicated files in Binance Zips
        if len(csv_files) > 1:
            logger.warning(f"  {tag} Multiple CSVs in zip ({len(csv_files)}), using only the first: {csv_files[0]}")
            csv_files = csv_files[:1]

        for csv_name in csv_files:
            logger.info(f"  Processing {csv_name} via PyArrow Streaming")
            with zf.open(csv_name) as csv_file:
                reader = pcsv.open_csv(
                    csv_file,
                    read_options=pcsv.ReadOptions(
                        block_size=50*1024*1024,
                        column_names=CSV_SCHEMA.names # Tells PyArrow there are no headers
                    ),
                    convert_options=pcsv.ConvertOptions(
                        column_types=CSV_SCHEMA
                    )
                )

                for batch in reader:
                    table = pa.Table.from_batches([batch])
                    
                    # Convert int64 milliseconds to Parquet timestamp[us, tz=UTC]
                    time_us = pc.multiply(table["time"], 1000)
                    time_col = pc.cast(time_us, pa.timestamp("us", tz="UTC"))
                    table = table.set_column(4, "time", time_col)
                    
                    writer.write_table(table)
                    rows_written += table.num_rows
                
                gc.collect()

    return rows_written

def process_csv_with_timestamp_tracking(zip_path, writer, tag, filter_after=None):
    """Processes CSVs and tracks the max timestamp. Used for the Stitching Route."""
    rows_written = 0
    max_ts = filter_after

    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            return 0, max_ts
            
        # ANOMALY FIX: Guard against duplicated files in Binance Zips
        if len(csv_files) > 1:
            logger.warning(f"  {tag} Multiple CSVs in zip ({len(csv_files)}), using only the first: {csv_files[0]}")
            csv_files = csv_files[:1]
            
        for csv_name in csv_files:
            with zf.open(csv_name) as f:
                reader = pcsv.open_csv(
                    f, 
                    read_options=pcsv.ReadOptions(
                        block_size=50*1024*1024,
                        column_names=CSV_SCHEMA.names # Tells PyArrow there are no headers
                    ),
                    convert_options=pcsv.ConvertOptions(
                        column_types=CSV_SCHEMA
                    )
                )
                for batch in reader:
                    table = pa.Table.from_batches([batch])
                    
                    # Convert int64 milliseconds to Parquet timestamp[us, tz=UTC]
                    time_us = pc.multiply(table["time"], 1000)
                    time_col = pc.cast(time_us, pa.timestamp("us", tz="UTC"))
                    table = table.set_column(4, "time", time_col)
                    
                    if filter_after:
                        mask = pc.greater(table["time"], filter_after)
                        table = table.filter(mask)
                    
                    if table.num_rows > 0:
                        writer.write_table(table)
                        rows_written += table.num_rows
                        
                        current_max = pc.max(table["time"]).as_py()
                        if max_ts is None or (current_max and current_max > max_ts):
                            max_ts = current_max
                            
                gc.collect()
                
    return rows_written, max_ts

def process_month(gcs_client, bucket, asset, year, month):
    tag = f"{asset} {year}-{month:02d}"
    output_blob = f"{OUTPUT_PREFIX}{asset}-trades-{year}-{month:02d}.parquet"

    # 1. Check if already complete
    if gcs_blob_exists(bucket, output_blob):
        if gcs_blob_size(bucket, output_blob) >= MIN_OUTPUT_BYTES:
            logger.info(f"{tag} already converted, skipping")
            return True
        else:
            logger.warning(f"{tag} incomplete, deleting old version")
            bucket.blob(output_blob).delete()

    # 2. Gather source files
    monthly_blob = f"{MONTHLY_ZIP_PREFIX}{asset}-trades-{year}-{month:02d}.zip"
    has_monthly = gcs_blob_exists(bucket, monthly_blob)
    
    daily_blobs = []
    days_in_month = monthrange(year, month)[1]
    for day in range(1, days_in_month + 1):
        db = f"{DAILY_ZIP_PREFIX}{asset}-trades-{year}-{month:02d}-{day:02d}.zip"
        if gcs_blob_exists(bucket, db):
            daily_blobs.append((day, db))

    if not has_monthly and not daily_blobs:
        logger.warning(f"{tag} no source files found")
        return False

    temp_parquet = f"/tmp/{asset}-{year}-{month:02d}.parquet"
    total_written = 0

    # ==========================================
    # THE HARDCODED EXCEPTION TRIGGER
    # ==========================================
    is_anomaly_month = (asset == "BTCUSDT" and year == 2023 and month == 3)

    try:
        with pq.ParquetWriter(temp_parquet, PYARROW_SCHEMA, compression=PARQUET_COMPRESSION) as writer:
            
            if is_anomaly_month:
                logger.info(f"{tag} triggers HARDCODED STITCH exception. Merging monthly and dailies.")
                last_timestamp = None
                
                # Process the partial monthly
                if has_monthly:
                    zip_path = download_zip_to_disk(bucket, monthly_blob)
                    try:
                        written, last_ts = process_csv_with_timestamp_tracking(zip_path, writer, tag)
                        total_written += written
                        last_timestamp = last_ts
                        logger.info(f"  Monthly finished at {last_timestamp}")
                    finally:
                        if os.path.exists(zip_path): os.remove(zip_path)
                
                # Process Dailies, filtering out overlap
                for day, db in sorted(daily_blobs):
                    zip_path = download_zip_to_disk(bucket, db)
                    try:
                        written, last_ts = process_csv_with_timestamp_tracking(zip_path, writer, tag, filter_after=last_timestamp)
                        total_written += written
                        if last_ts: last_timestamp = last_ts
                    finally:
                        if os.path.exists(zip_path): os.remove(zip_path)

            else:
                # ==========================================
                # THE STANDARD FAST ROUTE (99% of months)
                # ==========================================
                logger.info(f"{tag} starting standard conversion")
                
                if has_monthly:
                    zip_path = download_zip_to_disk(bucket, monthly_blob)
                    try:
                        total_written += process_csv_from_zip(zip_path, writer, tag)
                    finally:
                        if os.path.exists(zip_path): os.remove(zip_path)
                else:
                    for day, db in sorted(daily_blobs):
                        zip_path = download_zip_to_disk(bucket, db)
                        try:
                            total_written += process_csv_from_zip(zip_path, writer, tag)
                        finally:
                            if os.path.exists(zip_path): os.remove(zip_path)

        # 3. Final Upload
        if total_written > 0:
            out_blob = bucket.blob(output_blob)
            out_blob.upload_from_filename(temp_parquet, timeout=900)
            logger.info(f"{tag} success: {total_written:,} rows uploaded")
            return True
        return False

    except Exception as e:
        logger.error(f"{tag} failed: {e}")
        return False
    finally:
        if os.path.exists(temp_parquet): os.remove(temp_parquet)
        gc.collect()

def main():
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(BUCKET)

    logger.info("Starting Memory-Optimized Pipeline")
    total = 0
    failed = []

    for start_ym, end_ym in WINDOWS:
        for year, month in parse_window_months(start_ym, end_ym):
            for asset in ASSETS:
                if asset == "SOLUSDT" and (year, month) < (2020, 11): 
                    continue
                
                if process_month(gcs_client, bucket, asset, year, month):
                    total += 1
                else:
                    failed.append((asset, year, month))

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(f"Months converted : {total}")
    logger.info(f"Failures         : {len(failed)}")
    
    if failed:
        logger.error("Failed months:")
        for asset, year, month in failed:
            logger.error(f"  {asset} {year}-{month:02d}")
        exit(1)

if __name__ == "__main__":
    main()

