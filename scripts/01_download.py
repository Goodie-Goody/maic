import hashlib
import io
import logging
import os
import sys
import zipfile
from datetime import datetime, timezone
from calendar import monthrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from google.cloud import storage

from config import (
    ASSETS,
    BUCKET,
    WINDOWS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BINANCE_BASE = "https://data.binance.vision/data/spot"
MONTHLY_PATH = "monthly/trades"
DAILY_PATH = "daily/trades"
MONTHLY_ZIP_PREFIX = "raw/zips/monthly/"
DAILY_ZIP_PREFIX = "raw/zips/daily/"


def build_monthly_url(asset, year, month):
    filename = f"{asset}-trades-{year}-{month:02d}.zip"
    url = f"{BINANCE_BASE}/{MONTHLY_PATH}/{asset}/{filename}"
    return url, filename


def build_daily_url(asset, year, month, day):
    filename = f"{asset}-trades-{year}-{month:02d}-{day:02d}.zip"
    url = f"{BINANCE_BASE}/{DAILY_PATH}/{asset}/{filename}"
    return url, filename


def fetch_checksum(url):
    response = requests.get(f"{url}.CHECKSUM", timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Checksum fetch failed for {url}: {response.status_code}"
        )
    return response.text.strip().split()[0]


def download_and_verify(url):
    response = requests.get(url, timeout=120, stream=True)
    if response.status_code == 404:
        return None, "not_found"
    if response.status_code != 200:
        raise RuntimeError(
            f"Download failed for {url}: {response.status_code}"
        )

    buf = io.BytesIO()
    sha256 = hashlib.sha256()
    for chunk in response.iter_content(chunk_size=8192):
        buf.write(chunk)
        sha256.update(chunk)

    expected = fetch_checksum(url)
    actual = sha256.hexdigest()

    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {url}\n"
            f"  Expected : {expected}\n"
            f"  Got      : {actual}"
        )

    buf.seek(0)
    return buf, "ok"


def extract_dates_from_zip(zip_buf):
    dates = set()
    with zipfile.ZipFile(zip_buf) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            with zf.open(name) as f:
                next(f)
                for line in f:
                    parts = line.decode().strip().split(",")
                    if len(parts) < 5:
                        continue
                    try:
                        ts = int(parts[4])
                        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                        dates.add(dt.date())
                    except (ValueError, IndexError):
                        continue
    return dates


def upload_to_gcs(client, buf, destination_blob):
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(destination_blob)
    buf.seek(0)
    blob.upload_from_file(buf, content_type="application/zip", timeout=600)
    logger.info(f"Uploaded to gs://{BUCKET}/{destination_blob}")


def delete_from_gcs(client, blob_path):
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.delete()
        logger.info(f"Deleted gs://{BUCKET}/{blob_path}")


def expected_days(year, month):
    return set(range(1, monthrange(year, month)[1] + 1))


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


def gcs_blob_exists(client, blob_path):
    bucket = client.bucket(BUCKET)
    return bucket.blob(blob_path).exists()


def cleanup_wrong_paths(client):
    wrong_prefix = "raw/trades_parquet_flat/raw/"
    bucket = client.bucket(BUCKET)
    blobs = list(client.list_blobs(BUCKET, prefix=wrong_prefix))
    if not blobs:
        logger.info("No wrongly pathed files found, nothing to clean up")
        return
    logger.info(f"Found {len(blobs)} wrongly pathed files, deleting...")
    for blob in blobs:
        blob.delete()
        logger.info(f"Deleted gs://{BUCKET}/{blob.name}")
    logger.info("Cleanup complete")


def process_month(client, asset, year, month):
    tag = f"{asset} {year}-{month:02d}"
    monthly_blob = f"{MONTHLY_ZIP_PREFIX}{asset}-trades-{year}-{month:02d}.zip"

    if gcs_blob_exists(client, monthly_blob):
        logger.info(f"{tag} monthly zip already in GCS, skipping download")
        return

    url, filename = build_monthly_url(asset, year, month)
    logger.info(f"{tag} downloading monthly file")

    buf, status = download_and_verify(url)

    if status == "not_found":
        logger.warning(f"{tag} monthly file not found on Binance Vision")
        return

    logger.info(f"{tag} checksum verified")

    buf.seek(0)
    zip_buf_copy = io.BytesIO(buf.read())
    buf.seek(0)

    upload_to_gcs(client, buf, monthly_blob)

    logger.info(f"{tag} checking date coverage in monthly file")
    zip_buf_copy.seek(0)
    found_dates = extract_dates_from_zip(zip_buf_copy)
    found_days = {d.day for d in found_dates}
    missing_days = expected_days(year, month) - found_days

    if not missing_days:
        logger.info(f"{tag} all days present in monthly file")
        return

    logger.warning(
        f"{tag} missing {len(missing_days)} days: {sorted(missing_days)}"
    )
    logger.info(f"{tag} downloading individual daily files for missing days")

    for day in sorted(missing_days):
        daily_blob = (
            f"{DAILY_ZIP_PREFIX}{asset}-trades-{year}-{month:02d}-{day:02d}.zip"
        )

        if gcs_blob_exists(client, daily_blob):
            logger.info(f"{tag}-{day:02d} daily zip already in GCS, skipping")
            continue

        daily_url, _ = build_daily_url(asset, year, month, day)
        logger.info(f"{tag}-{day:02d} downloading daily file")

        daily_buf, daily_status = download_and_verify(daily_url)

        if daily_status == "not_found":
            logger.warning(
                f"{tag}-{day:02d} daily file not found on Binance Vision"
            )
            continue

        logger.info(f"{tag}-{day:02d} checksum verified")
        upload_to_gcs(client, daily_buf, daily_blob)


def main():
    client = storage.Client()

    logger.info("Starting download pipeline")
    logger.info(f"Assets  : {ASSETS}")
    logger.info(f"Windows : {WINDOWS}")
    logger.info(f"Bucket  : {BUCKET}")

    logger.info("Cleaning up any wrongly pathed files from previous runs")
    cleanup_wrong_paths(client)

    total = 0
    failed = []

    for start_ym, end_ym in WINDOWS:
        months = parse_window_months(start_ym, end_ym)
        for year, month in months:
            for asset in ASSETS:
                if asset == "SOLUSDT" and (year, month) < (2020, 11):
                    logger.info(
                        f"SOLUSDT {year}-{month:02d} skipped, pre-listing"
                    )
                    continue
                try:
                    process_month(client, asset, year, month)
                    total += 1
                except Exception as e:
                    logger.error(f"{asset} {year}-{month:02d} failed: {e}")
                    failed.append((asset, year, month, str(e)))

    logger.info("=" * 60)
    logger.info("Download pipeline complete")
    logger.info(f"Months processed : {total}")
    logger.info(f"Failures         : {len(failed)}")

    if failed:
        logger.error("Failed months:")
        for asset, year, month, err in failed:
            logger.error(f"  {asset} {year}-{month:02d} : {err}")
        exit(1)

    logger.info("All downloads complete and verified")


if __name__ == "__main__":
    main()

