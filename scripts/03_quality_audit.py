import sys
import os
import json
import logging
from datetime import date
from calendar import monthrange

import pyarrow.parquet as pq
from pyarrow import fs
from google.cloud import bigquery
from google.cloud import storage

# Ensure config is accessible
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ASSETS,
    BUCKET,
    WINDOWS,
    BQ_DATASET,
    BQ_TABLE,
    PROJECT_ID,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_PREFIX = "v2/trades_parquet_flat/"
REPORT_PATH = "logs/quality_audit_report.json"

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

def get_expected_months():
    expected = []
    for start_ym, end_ym in WINDOWS:
        for year, month in parse_window_months(start_ym, end_ym):
            for asset in ASSETS:
                if asset == "SOLUSDT" and (year, month) < (2020, 11):
                    continue
                expected.append((asset, year, month))
    return expected

def check_file_inventory(bucket, expected_months):
    logger.info("Checking file inventory")
    results = {}
    missing_files = []
    incomplete_files = []

    for asset, year, month in expected_months:
        blob_path = f"{OUTPUT_PREFIX}{asset}-trades-{year}-{month:02d}.parquet"
        blob = bucket.blob(blob_path)

        if not blob.exists():
            logger.warning(f"Missing: {blob_path}")
            missing_files.append(blob_path)
            continue

        blob.reload()
        size_mb = blob.size / 1024 / 1024

        if blob.size < 10 * 1024 * 1024:
            logger.warning(f"Incomplete ({size_mb:.1f} MB): {blob_path}")
            incomplete_files.append(blob_path)
            continue

        results[f"{asset}-{year}-{month:02d}"] = {
            "size_mb": round(size_mb, 1),
            "status": "present"
        }

    logger.info(f"Files present    : {len(results)}")
    logger.info(f"Files missing    : {len(missing_files)}")
    logger.info(f"Files incomplete : {len(incomplete_files)}")

    return results, missing_files, incomplete_files

def check_schema(expected_months):
    logger.info("Checking schema consistency across ALL files (Footer-only read)")
    schema_issues = []

    expected_fields = {
        "id": "int64",
        "price": "double",
        "qty": "double",
        "quote_qty": "double",
        "time": "timestamp[us, tz=UTC]",
        "is_buyer_maker": "bool",
        "is_best_match": "bool",
    }

    # Initialize the PyArrow GCS File System for byte-range requests
    gcs = fs.GcsFileSystem()

    for asset, year, month in expected_months:
        gcs_path = f"{BUCKET}/{OUTPUT_PREFIX}{asset}-trades-{year}-{month:02d}.parquet"

        try:
            # Reads ONLY the metadata footer, skipping the massive data blocks
            with gcs.open_input_file(gcs_path) as file_handle:
                schema = pq.read_schema(file_handle)

            for field in schema:
                expected_type = expected_fields.get(field.name)
                actual_type = str(field.type)
                if expected_type and actual_type != expected_type:
                    issue = f"{gcs_path}: {field.name} expected {expected_type} got {actual_type}"
                    logger.warning(issue)
                    schema_issues.append(issue)
                    
        except Exception as e:
            issue = f"Failed to read schema for {gcs_path}: {e}"
            logger.warning(issue)
            schema_issues.append(issue)

    if not schema_issues:
        logger.info("Schema consistent across all files")

    return schema_issues

def run_bq_audit(bq_client):
    logger.info("Refreshing BigQuery External Table Cache")
    try:
        # Forces BigQuery to scan the GCS bucket for your newly converted v2 files
        bq_client.query(f"CALL BQ.REFRESH_EXTERNAL_TABLE('{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}')").result()
    except Exception as e:
        logger.debug(f"Cache refresh skipped (likely a native table, not external): {e}")

    logger.info("Running BigQuery audit queries")

    summary_query = f"""
    SELECT
        REGEXP_EXTRACT(_FILE_NAME, r'(BTCUSDT|ETHUSDT|SOLUSDT)') AS asset,
        COUNT(DISTINCT DATE(time)) AS days_present,
        COUNT(*) AS total_trades,
        COUNT(*) - COUNT(DISTINCT id) AS duplicates,
        COUNTIF(price <= 0) AS bad_prices,
        COUNTIF(price > 10000000) AS extreme_prices,
        COUNTIF(qty <= 0) AS bad_quantities,
        COUNTIF(id IS NULL) AS null_ids,
        COUNTIF(time IS NULL) AS null_timestamps,
        MIN(time) AS dataset_start,
        MAX(time) AS dataset_end
    FROM
        `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
    WHERE
        _FILE_NAME LIKE '%v2/trades_parquet_flat%'
        AND time >= TIMESTAMP('2020-01-01')
        AND time < TIMESTAMP('2026-01-01')
    GROUP BY asset
    ORDER BY asset
    """

    logger.info("Running summary audit")
    summary_results = []
    rows = bq_client.query(summary_query).result()
    
    for row in rows:
        result = {
            "asset": row.asset,
            "days_present": row.days_present,
            "total_trades": row.total_trades,
            "duplicates": row.duplicates,
            "bad_prices": row.bad_prices,
            "extreme_prices": row.extreme_prices,
            "bad_quantities": row.bad_quantities,
            "null_ids": row.null_ids,
            "null_timestamps": row.null_timestamps,
            "dataset_start": str(row.dataset_start),
            "dataset_end": str(row.dataset_end),
        }
        summary_results.append(result)
        logger.info(
            f"  {row.asset}: {row.days_present} days, "
            f"{row.total_trades:,} trades, "
            f"{row.duplicates} duplicates, "
            f"{row.bad_prices} bad prices"
        )

    gap_query = f"""
    WITH daily_trades AS (
        SELECT
            REGEXP_EXTRACT(_FILE_NAME, r'(BTCUSDT|ETHUSDT|SOLUSDT)') AS asset,
            DATE(time) AS trade_date,
            COUNT(*) AS trade_count
        FROM
            `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE
            _FILE_NAME LIKE '%v2/trades_parquet_flat%'
            AND time >= TIMESTAMP('2020-01-01')
            AND time < TIMESTAMP('2026-01-01')
        GROUP BY asset, trade_date
    )
    SELECT
        asset,
        trade_date,
        trade_count
    FROM daily_trades
    WHERE trade_count < 1000
    ORDER BY asset, trade_date
    """

    logger.info("Checking for suspiciously low trade count days")
    low_count_days = []
    rows = bq_client.query(gap_query).result()
    for row in rows:
        low_count_days.append({
            "asset": row.asset,
            "date": str(row.trade_date),
            "trade_count": row.trade_count,
        })
        logger.warning(
            f"  Low trade count: {row.asset} {row.trade_date} "
            f"({row.trade_count} trades)"
        )

    if not low_count_days:
        logger.info("No suspiciously low trade count days found")

    return summary_results, low_count_days

def assess_overall_quality(summary_results, missing_files, incomplete_files, schema_issues, low_count_days):
    issues = []

    if missing_files:
        issues.append(f"{len(missing_files)} missing files")
    if incomplete_files:
        issues.append(f"{len(incomplete_files)} incomplete files")
    if schema_issues:
        issues.append(f"{len(schema_issues)} schema inconsistencies")

    for result in summary_results:
        if result["duplicates"] > 0:
            issues.append(f"{result['asset']} has {result['duplicates']} duplicate trade IDs")
        if result["bad_prices"] > 0:
            issues.append(f"{result['asset']} has {result['bad_prices']} bad prices")
        if result["extreme_prices"] > 0:
            issues.append(f"{result['asset']} has {result['extreme_prices']} extreme outlier prices")
        if result["bad_quantities"] > 0:
            issues.append(f"{result['asset']} has {result['bad_quantities']} bad quantities")
        if result["null_ids"] > 0:
            issues.append(f"{result['asset']} has {result['null_ids']} null IDs")
        if result["null_timestamps"] > 0:
            issues.append(f"{result['asset']} has {result['null_timestamps']} null timestamps")

    if low_count_days:
        issues.append(f"{len(low_count_days)} days with suspiciously low trade counts")

    if not issues:
        status = "PASSED"
        score = 100.0
    else:
        status = "FAILED"
        score = max(0.0, 100.0 - (len(issues) * 10))

    return status, score, issues

def main():
    gcs_client = storage.Client()
    bq_client = bigquery.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET)

    logger.info("Starting comprehensive quality audit")
    logger.info(f"Bucket   : {BUCKET}")
    logger.info(f"Prefix   : {OUTPUT_PREFIX}")
    logger.info(f"BQ table : {PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}")

    expected_months = get_expected_months()
    logger.info(f"Expected months: {len(expected_months)}")

    # file inventory check
    file_results, missing_files, incomplete_files = check_file_inventory(
        bucket, expected_months
    )

    # schema check (Updated to use PyArrow GCS FileSystem)
    schema_issues = check_schema(expected_months)

    # bigquery audit
    summary_results, low_count_days = run_bq_audit(bq_client)

    # overall assessment
    status, score, issues = assess_overall_quality(
        summary_results, missing_files, incomplete_files,
        schema_issues, low_count_days
    )

    # build report
    report = {
        "audit_date": str(date.today()),
        "status": status,
        "quality_score": score,
        "expected_months": len(expected_months),
        "files_present": len(file_results),
        "files_missing": len(missing_files),
        "files_incomplete": len(incomplete_files),
        "schema_issues": schema_issues,
        "missing_files": missing_files,
        "incomplete_files": incomplete_files,
        "bq_summary": summary_results,
        "low_count_days": low_count_days,
        "issues": issues,
    }

    os.makedirs("logs", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Audit complete")
    logger.info(f"Status       : {status}")
    logger.info(f"Quality score: {score}%")
    logger.info(f"Report saved : {REPORT_PATH}")

    if issues:
        logger.error("Critical Data Issues found:")
        for issue in issues:
            logger.error(f"  {issue}")
        exit(1)

    logger.info("Dataset passed all quality checks")

if __name__ == "__main__":
    main()

