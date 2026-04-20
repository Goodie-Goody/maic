import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BUCKET = os.getenv("GCP_BUCKET")
REGION = os.getenv("GCP_REGION", "us-central1")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

WINDOWS = [
    ("2020-02", "2020-07"),
    ("2020-11", "2021-05"),
    ("2021-11", "2022-05"),
    ("2022-11", "2023-04"),
    ("2024-07", "2024-12"),
]

GCS_PREFIX = "raw/trades_parquet_flat/"

PARQUET_COMPRESSION = "snappy"

SCHEMA = {
    "id": "int64",
    "price": "float64",
    "qty": "float64",
    "quote_qty": "float64",
    "time": "timestamp[us, tz=UTC]",
    "is_buyer_maker": "bool",
    "is_best_match": "bool",
}
