import sys
import os
import logging

import pyarrow.parquet as pq
from pyarrow import fs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BUCKET, ASSETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURES_PREFIX = "v2/features/"


def main():
    # --- Skip logic ---
    import os as _os
    _os.environ.setdefault(
        "GOOGLE_APPLICATION_CREDENTIALS",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gcp-key.json")
    )
    from google.cloud import storage as _storage
    _client = _storage.Client()
    _bucket = _client.bucket(BUCKET)
    _marker = "v2/pipeline_markers/05b_verify_features.done"
    if _bucket.blob(_marker).exists():
        logger.info("05b_verify_features — already complete, skipping")
        logger.info(f"  To rerun: gsutil rm gs://{BUCKET}/{_marker}")
        return

    gcs = fs.GcsFileSystem()

    base_path = f"{BUCKET}/{FEATURES_PREFIX}"
    file_infos = gcs.get_file_info(fs.FileSelector(base_path, recursive=False))

    parquet_files = [f for f in file_infos if f.base_name.endswith(".parquet")]

    if not parquet_files:
        logger.error("No feature files found in GCS")
        sys.exit(1)

    logger.info(f"Found {len(parquet_files)} feature files")

    total_rows = {asset: 0 for asset in ASSETS}
    file_counts = {asset: 0 for asset in ASSETS}
    schema_reference = {}
    schema_mismatch = False

    for file_info in sorted(parquet_files, key=lambda f: f.base_name):
        filename = file_info.base_name
        asset = filename.split("-")[0]

        if asset not in ASSETS:
            logger.warning(f"Unexpected file: {filename}")
            continue

        try:
            with gcs.open_input_file(file_info.path) as file_handle:
                metadata = pq.read_metadata(file_handle)

                rows = metadata.num_rows
                cols = metadata.num_columns
                col_names = metadata.schema.names

                total_rows[asset] += rows
                file_counts[asset] += 1

                if asset not in schema_reference:
                    schema_reference[asset] = col_names
                    logger.info(
                        f"  {asset} established schema with {cols} columns: {col_names}"
                    )
                elif schema_reference[asset] != col_names:
                    logger.error(f"  SCHEMA MISMATCH in {filename}")
                    logger.error(f"    Expected : {schema_reference[asset]}")
                    logger.error(f"    Got      : {col_names}")
                    schema_mismatch = True

        except Exception as e:
            logger.error(f"  Failed to read metadata for {filename}: {e}")

    logger.info("=" * 60)
    logger.info("Feature Dataset Summary")
    logger.info("=" * 60)

    for asset in ASSETS:
        logger.info(
            f"  {asset}: {file_counts[asset]} files, "
            f"{total_rows[asset]:,} rows, "
            f"{len(schema_reference.get(asset, []))} columns"
        )

    grand_total = sum(total_rows.values())
    logger.info(f"  Grand total : {grand_total:,} rows")
    logger.info(f"  Grand total : {grand_total / 1_000_000:.2f}M observations")
    logger.info("=" * 60)

    if schema_mismatch:
        logger.error("Audit FAILED: Schema inconsistencies detected across files")
        sys.exit(1)

    logger.info("Audit PASSED: All schemas consistent")

    # --- Write done marker ---
    _bucket.blob(_marker).upload_from_string(b"")
    logger.info(f"Done marker written: gs://{BUCKET}/{_marker}")


if __name__ == "__main__":
    main()

