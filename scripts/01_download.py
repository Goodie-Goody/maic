import hashlib
import io
import logging
import os
import re
import sys
import zipfile
from calendar import monthrange
from datetime import datetime, timezone

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

MONTHLY_PATTERN = re.compile(
    r"^(BTCUSDT|ETHUSDT|SOLUSDT)-trades-(\d{4})-(\d{2})\.zip$"
)
DAILY_PATTERN = re.compile(
    r"^(BTCUSDT|ETHUSDT|SOLUSDT)-trades-(\d{4})-(\d{2})-(\d{2})\.zip$"
)

MIN_MONTHLY_BYTES = 10 * 1024 * 1024
MIN_DAILY_BYTES = 1 * 1024 * 1024


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
    response = requests.get(url, timeout=600, stream=True)
    if response.status_code == 404:
        return None, "not_found"
    if response.status_code != 200:
        raise RuntimeError(
            f"Download failed for {url}: {response.status_code}"
        )

    buf = io.BytesIO()
    sha256 = hashlib.sha256()
    for chunk in response.iter_content(chunk_size=65536):
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
    blob.upload_from_file(buf, content_type="application/zip", timeout=900)
    logger.info(f"Uploaded to gs://{BUCKET}/{destination_blob}")


def gcs_blob_exists(client, blob_path):
    bucket = client.bucket(BUCKET)
    return bucket.blob(blob_path).exists()


def gcs_blob_size(client, blob_path):
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(blob_path)
    blob.reload()
    return blob.size


def delete_blob(client, blob_path):
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(blob_path)
    blob.delete()
    logger.info(f"Deleted gs://{BUCKET}/{blob_path}")


def build_expected_blobs():
    expected = set()
    for start_ym, end_ym in WINDOWS:
        months = parse_window_months(start_ym, end_ym)
        for year, month in months:
            for asset in ASSETS:
                if asset == "SOLUSDT" and (year, month) < (2020, 11):
                    continue
                expected.add(
                    f"{MONTHLY_ZIP_PREFIX}{asset}-trades-{year}-{month:02d}.zip"
                )
    return expected


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


def expected_days(year, month):
    return set(range(1, monthrange(year, month)[1] + 1))


def validate_bucket(client):
    logger.info("Validating bucket contents")
    expected_monthly = build_expected_blobs()
    issues = 0

    # Check monthly prefix
    monthly_blobs = list(client.list_blobs(BUCKET, prefix=MONTHLY_ZIP_PREFIX))
    for blob in monthly_blobs:
        filename = blob.name.split("/")[-1]
        match = MONTHLY_PATTERN.match(filename)

        if not match:
            logger.warning(f"Unexpected file found, deleting: {blob.name}")
            blob.delete()
            issues += 1
            continue

        if blob.size < MIN_MONTHLY_BYTES:
            logger.warning(
                f"Incomplete monthly file ({blob.size} bytes), deleting: {blob.name}"
            )
            blob.delete()
            issues += 1
            continue

    # Check daily prefix
    daily_blobs = list(client.list_blobs(BUCKET, prefix=DAILY_ZIP_PREFIX))
    for blob in daily_blobs:
        filename = blob.name.split("/")[-1]
        match = DAILY_PATTERN.match(filename)

        if not match:
            logger.warning(f"Unexpected file found, deleting: {blob.name}")
            blob.delete()
            issues += 1
            continue

        if blob.size < MIN_DAILY_BYTES:
            logger.warning(
                f"Incomplete daily file ({blob.size} bytes), deleting: {blob.name}"
            )
            blob.delete()
            issues += 1
            continue

    # Check for old wrong-path files
    wrong_blobs = list(
        client.list_blobs(BUCKET, prefix="raw/trades_parquet_flat/raw/")
    )
    for blob in wrong_blobs:
        logger.warning(f"Wrong path file found, deleting: {blob.name}")
        blob.delete()
        issues += 1

    if issues == 0:
        logger.info("Bucket validation passed, no issues found")
    else:
        logger.info(f"Bucket validation complete, resolved {issues} issues")


def process_month(client, asset, year, month):
    tag = f"{asset} {year}-{month:02d}"
    monthly_blob = f"{MONTHLY_ZIP_PREFIX}{asset}-trades-{year}-{month:02d}.zip"

    if gcs_blob_exists(client, monthly_blob):
        size = gcs_blob_size(client, monthly_blob)
        if size >= MIN_MONTHLY_BYTES:
            logger.info(f"{tag} already in GCS and complete, skipping")
            return
        else:
            logger.warning(
                f"{tag} exists in GCS but incomplete ({size} bytes), re-downloading"
            )
            delete_blob(client, monthly_blob)

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
            size = gcs_blob_size(client, daily_blob)
            if size >= MIN_DAILY_BYTES:
                logger.info(f"{tag}-{day:02d} already in GCS and complete, skipping")
                continue
            else:
                logger.warning(
                    f"{tag}-{day:02d} exists but incomplete ({size} bytes), re-downloading"
                )
                delete_blob(client, daily_blob)

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

    validate_bucket(client)

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

