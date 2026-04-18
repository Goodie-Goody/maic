import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import gc
import re
import logging
import tempfile
import zipfile
from datetime import datetime, timezone
from calendar import monthrange

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from google.cloud import storage

from config import (
    ASSETS,
    BUCKET,
    WINDOWS,
    PARQUET_COMPRESSION,
    SCHEMA,
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
BATCH_SIZE = 500_000

PYARROW_SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("price", pa.float64()),
    ("qty", pa.float64()),
    ("quote_qty", pa.float64()),
    ("time", pa.timestamp("us", tz="UTC")),
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
    blob = bucket.blob(blob_path)
    return blob.exists()


def gcs_blob_size(bucket, blob_path):
    blob = bucket.blob(blob_path)
    blob.reload()
    return blob.size


def detect_time_unit(raw_value):
    try:
        ts = int(raw_value)
        if ts < 1e12:
            return "s"
        elif ts < 1e15:
            return "ms"
        elif ts < 1e18:
            return "us"
        else:
            return "ns"
    except (ValueError, TypeError):
        return "ms"


def convert_timestamp(raw_value, unit):
    ts = int(raw_value)
    if unit == "s":
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    elif unit == "ms":
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    elif unit == "us":
        return datetime.fromtimestamp(ts / 1_000_000, tz=timezone.utc)
    else:
        return datetime.fromtimestamp(ts / 1_000_000_000, tz=timezone.utc)


def read_zip_from_gcs(bucket, blob_path):
    blob = bucket.blob(blob_path)
    blob.reload()
    size_mb = blob.size / 1024 / 1024
    logger.info(f"  Downloading {blob_path} ({size_mb:.1f} MB)")
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return buf


def process_csv_from_zip(zip_buf, writer, seen_ids, tag):
    rows_written = 0
    rows_skipped = 0
    time_unit = None

    with zipfile.ZipFile(zip_buf) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            logger.warning(f"  {tag} no CSV files found in zip")
            return rows_written, rows_skipped

        for csv_name in csv_files:
            logger.info(f"  Processing {csv_name}")
            with zf.open(csv_name) as csv_file:
                batch_ids = []
                batch_prices = []
                batch_qtys = []
                batch_quote_qtys = []
                batch_times = []
                batch_buyer_maker = []
                batch_best_match = []

                header = csv_file.readline()

                for line in csv_file:
                    parts = line.decode().strip().split(",")
                    if len(parts) < 7:
                        continue

                    try:
                        trade_id = int(parts[0])

                        if trade_id in seen_ids:
                            rows_skipped += 1
                            continue

                        if time_unit is None:
                            time_unit = detect_time_unit(parts[4])
                            logger.info(f"  Detected time unit: {time_unit}")

                        ts = convert_timestamp(parts[4], time_unit)

                        batch_ids.append(trade_id)
                        batch_prices.append(float(parts[1]))
                        batch_qtys.append(float(parts[2]))
                        batch_quote_qtys.append(float(parts[3]))
                        batch_times.append(ts)
                        batch_buyer_maker.append(parts[5].strip().lower() == "true")
                        batch_best_match.append(parts[6].strip().lower() == "true")

                        if len(batch_ids) >= BATCH_SIZE:
                            table = pa.table({
                                "id": pa.array(batch_ids, type=pa.int64()),
                                "price": pa.array(batch_prices, type=pa.float64()),
                                "qty": pa.array(batch_qtys, type=pa.float64()),
                                "quote_qty": pa.array(batch_quote_qtys, type=pa.float64()),
                                "time": pa.array(batch_times, type=pa.timestamp("us", tz="UTC")),
                                "is_buyer_maker": pa.array(batch_buyer_maker, type=pa.bool_()),
                                "is_best_match": pa.array(batch_best_match, type=pa.bool_()),
                            })
                            seen_ids.update(batch_ids)
                            writer.write_table(table)
                            rows_written += len(batch_ids)

                            batch_ids.clear()
                            batch_prices.clear()
                            batch_qtys.clear()
                            batch_quote_qtys.clear()
                            batch_times.clear()
                            batch_buyer_maker.clear()
                            batch_best_match.clear()

                            gc.collect()

                    except (ValueError, IndexError):
                        continue

                # flush remaining rows
                if batch_ids:
                    table = pa.table({
                        "id": pa.array(batch_ids, type=pa.int64()),
                        "price": pa.array(batch_prices, type=pa.float64()),
                        "qty": pa.array(batch_qtys, type=pa.float64()),
                        "quote_qty": pa.array(batch_quote_qtys, type=pa.float64()),
                        "time": pa.array(batch_times, type=pa.timestamp("us", tz="UTC")),
                        "is_buyer_maker": pa.array(batch_buyer_maker, type=pa.bool_()),
                        "is_best_match": pa.array(batch_best_match, type=pa.bool_()),
                    })
                    seen_ids.update(batch_ids)
                    writer.write_table(table)
                    rows_written += len(batch_ids)
                    gc.collect()

    return rows_written, rows_skipped


def process_month(gcs_client, bucket, asset, year, month):
    tag = f"{asset} {year}-{month:02d}"
    output_blob = f"{OUTPUT_PREFIX}{asset}-trades-{year}-{month:02d}.parquet"

    if gcs_blob_exists(bucket, output_blob):
        size = gcs_blob_size(bucket, output_blob)
        if size >= MIN_OUTPUT_BYTES:
            logger.info(f"{tag} already converted, skipping")
            return True
        else:
            logger.warning(f"{tag} output exists but incomplete, reconverting")
            bucket.blob(output_blob).delete()

    # collect all source zips for this month
    monthly_blob = f"{MONTHLY_ZIP_PREFIX}{asset}-trades-{year}-{month:02d}.zip"
    daily_blobs = []

    days_in_month = monthrange(year, month)[1]
    for day in range(1, days_in_month + 1):
        daily_blob = f"{DAILY_ZIP_PREFIX}{asset}-trades-{year}-{month:02d}-{day:02d}.zip"
        if gcs_blob_exists(bucket, daily_blob):
            daily_blobs.append((day, daily_blob))

    has_monthly = gcs_blob_exists(bucket, monthly_blob)

    if not has_monthly and not daily_blobs:
        logger.warning(f"{tag} no source files found, skipping")
        return False

    logger.info(f"{tag} starting conversion")
    if has_monthly:
        logger.info(f"  Monthly zip: present")
    logger.info(f"  Daily zips : {len(daily_blobs)} files")

    # write to local temp file to avoid RAM accumulation
    temp_path = f"/tmp/{asset}-{year}-{month:02d}.parquet"
    seen_ids = set()
    total_written = 0
    total_skipped = 0

    try:
        with pq.ParquetWriter(temp_path, PYARROW_SCHEMA, compression=PARQUET_COMPRESSION) as writer:

            # process monthly zip first
            if has_monthly:
                zip_buf = read_zip_from_gcs(bucket, monthly_blob)
                written, skipped = process_csv_from_zip(zip_buf, writer, seen_ids, tag)
                total_written += written
                total_skipped += skipped
                del zip_buf
                gc.collect()
                logger.info(f"  Monthly: {written:,} rows written, {skipped:,} skipped")

            # process daily zips in order
            for day, daily_blob in sorted(daily_blobs):
                zip_buf = read_zip_from_gcs(bucket, daily_blob)
                written, skipped = process_csv_from_zip(zip_buf, writer, seen_ids, tag)
                total_written += written
                total_skipped += skipped
                del zip_buf
                gc.collect()
                logger.info(f"  Day {day:02d}: {written:,} rows written, {skipped:,} skipped")

        logger.info(f"{tag} conversion complete")
        logger.info(f"  Total rows written : {total_written:,}")
        logger.info(f"  Total rows skipped : {total_skipped:,}")

        # upload temp file to GCS
        temp_size_mb = os.path.getsize(temp_path) / 1024 / 1024
        logger.info(f"  Output size        : {temp_size_mb:.1f} MB")
        logger.info(f"  Uploading to gs://{BUCKET}/{output_blob}")

        out_blob = bucket.blob(output_blob)
        with open(temp_path, "rb") as f:
            out_blob.upload_from_file(
                f,
                content_type="application/octet-stream",
                timeout=900
            )

        logger.info(f"{tag} uploaded successfully")
        return True

    except Exception as e:
        logger.error(f"{tag} failed: {e}")
        return False

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            gc.collect()


def main():
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(BUCKET)

    logger.info("Starting CSV to Parquet conversion pipeline")
    logger.info(f"Assets      : {ASSETS}")
    logger.info(f"Windows     : {WINDOWS}")
    logger.info(f"Bucket      : {BUCKET}")
    logger.info(f"Output      : {OUTPUT_PREFIX}")

    total = 0
    failed = []

    for start_ym, end_ym in WINDOWS:
        months = parse_window_months(start_ym, end_ym)
        for year, month in months:
            for asset in ASSETS:
                if asset == "SOLUSDT" and (year, month) < (2020, 11):
                    logger.info(f"SOLUSDT {year}-{month:02d} skipped, pre-listing")
                    continue
                success = process_month(gcs_client, bucket, asset, year, month)
                if success:
                    total += 1
                else:
                    failed.append((asset, year, month))

    logger.info("=" * 60)
    logger.info("Conversion pipeline complete")
    logger.info(f"Months converted : {total}")
    logger.info(f"Failures         : {len(failed)}")

    if failed:
        logger.error("Failed months:")
        for asset, year, month in failed:
            logger.error(f"  {asset} {year}-{month:02d}")
        exit(1)

    logger.info("All months converted successfully")


if __name__ == "__main__":
    main()

