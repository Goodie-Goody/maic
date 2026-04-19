```markdown
# MAIC Data Pipeline — Walkthrough

## Overview

This walkthrough documents the full data engineering pipeline for the paper "Real-time Liquidity Stress Detection in Cryptocurrency Markets Using Trade Flow Analysis and Machine Learning". It covers everything from GCP account setup to production-ready parquet files in GCS, written so that anyone can replicate the pipeline from scratch.

> **Important:** Cloud Shell is used only for provisioning infrastructure — creating buckets, datasets, and VMs. All data processing scripts must be run on the Compute Engine VM, not in Cloud Shell. Cloud Shell lacks the memory and persistent storage required for multi-gigabyte tick data files.

---

## Step 0 — GCP Account Setup

Create a GCP account at [console.cloud.google.com](https://console.cloud.google.com). Enable billing by linking a payment method. Once billing is active, open Cloud Shell by clicking the terminal icon in the top right of the GCP console.

Set your project:

```bash
gcloud config set project your-project-id
```

Create a GCS bucket in `us-central1`:

```bash
gsutil mb -l us-central1 gs://your-bucket-name
```

Create a BigQuery dataset:

```bash
bq mk --dataset --location=us-central1 your-project-id:your-dataset-name
```

Enable the Compute Engine API:

```bash
gcloud services enable compute.googleapis.com
```

---

## Step 1 — GitHub Repository

Create a public repository on [github.com](https://github.com). Initialize it with a README. Then clone it into Cloud Shell:

```bash
cd ~
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

Configure git identity:

```bash
git config --global user.email "your-email"
git config --global user.name "Your Name"
```

GitHub requires a Personal Access Token for pushing via the command line. Generate one at GitHub — Settings — Developer Settings — Personal Access Tokens — Tokens Classic — with `repo` scope.

Configure git to securely cache your token in memory for 10 hours so you are not prompted on every push:

```bash
git config --global credential.helper 'cache --timeout=36000'
```

On your first `git push` you will be prompted for your username and token once. Git will remember it securely for subsequent pushes within the timeout window.

---

## Step 2 — Folder Structure

```bash
mkdir -p scripts docs logs

touch config.py
touch requirements.txt
touch run.sh
touch .env.example
touch scripts/01_download.py
touch scripts/02_csv_to_parquet.py
touch scripts/03_quality_audit.py
touch docs/walkthrough.md
```

The scripts are numbered to make the execution order explicit. Each script is self-contained, reads from `config.py`, logs to `logs/`, and exits with a clear success or failure message.

---

## Step 3 — Environment and Configuration

Create `.env` locally — this file is never pushed to GitHub:

```bash
GCP_PROJECT_ID=your-project-id
GCP_BUCKET=your-bucket-name
GCP_REGION=us-central1
BQ_DATASET=your-bigquery-dataset
BQ_TABLE=your-external-table
```

Create `.gitignore`:

```
.env
__pycache__/
*.pyc
*.log
.venv/
*.parquet
*.csv
.DS_Store
logs/
```

Create `.env.example` which is pushed to GitHub so replicators know what variables are needed:

```bash
GCP_PROJECT_ID=your-project-id-here
GCP_BUCKET=your-bucket-name-here
GCP_REGION=us-central1
BQ_DATASET=your-bigquery-dataset-name
BQ_TABLE=your-external-table-name
```

`config.py` loads from `.env` and defines all pipeline constants in one place. Everything else imports from here:

```python
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
    ("2020-11", "2021-04"),
    ("2021-11", "2022-04"),
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
```

`requirements.txt`:

```
pyarrow
pandas
google-cloud-storage
google-cloud-bigquery
tqdm
python-dotenv
requests
```

---

## Step 4 — BigQuery External Table

Before running the quality audit, create the BigQuery external table pointing at the production parquet files. Run this in Cloud Shell:

```bash
bq query --use_legacy_sql=false --project_id=your-project-id '
CREATE OR REPLACE EXTERNAL TABLE `your-project-id.your-dataset.your-table`
OPTIONS (
  format = "PARQUET",
  uris = ["gs://your-bucket/v2/trades_parquet_flat/*.parquet"]
)
'
```

This only needs to be run once, and again whenever the output prefix changes. After this BigQuery can query all parquet files in the production prefix as a single logical table using the `_FILE_NAME` pseudocolumn for file-level filtering.

---

## Step 5 — First Commit and Push

```bash
git add .
git commit -m "add project config, requirements, and environment setup"
git push
```

---

## Step 6 — Compute Engine VM Setup

All download and conversion scripts must be run on this VM. Do not run them in Cloud Shell.

Create the VM from Cloud Shell:

```bash
gcloud compute instances create pipeline-vm \
  --zone=us-central1-a \
  --machine-type=e2-highcpu-8 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --scopes=storage-full \
  --boot-disk-size=50GB
```

`e2-highcpu-8` provides 8 vCPUs and 8GB RAM. The `--scopes=storage-full` flag grants the VM read/write access to all GCS buckets in the project via the default Compute Engine service account. Running in `us-central1-a` — the same region as the bucket — means all transfers use Google's internal network with no egress costs. Cost is approximately $0.19/hour — delete the VM immediately after the pipeline completes.

SSH in from Cloud Shell:

```bash
gcloud compute ssh pipeline-vm --zone=us-central1-a
```

On the VM, install dependencies and clone the repo:

```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git
git clone https://github.com/your-username/your-repo.git
cd your-repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Authenticate Python GCS access:

```bash
gcloud auth application-default login
```

This opens a browser link. Complete the sign-in and paste the verification code back into the terminal. Credentials are saved to `~/.config/gcloud/application_default_credentials.json` and picked up automatically by the google-cloud-storage library.

Configure git identity and credential cache on the VM:

```bash
git config --global user.email "your-email"
git config --global user.name "Your Name"
git config --global credential.helper 'cache --timeout=36000'
```

Create the `.env` file on the VM:

```bash
nano .env
```

Fill in your actual GCP values and save.

Add swap space to prevent silent OOM failures on large files:

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

Verify swap is active:

```bash
free -h
```

---

## Step 7 — Running the Pipeline

Each script runs in the background using `nohup` so it survives terminal disconnects. All output goes to a dedicated log file. Reconnect at any time and check the log to monitor progress.

### Script 1 — Download

[`scripts/01_download.py`](../scripts/01_download.py)

Downloads monthly trade archives for each asset and window from [Binance Vision](https://data.binance.vision). SHA256 checksum verification happens inline during download — if the checksum does not match the file is discarded immediately. For months where the monthly archive is missing days, the script automatically falls back to downloading individual daily files for those specific days only. The script validates the bucket on startup, removing unexpected or incomplete files, and has full resume capability — files already present in GCS above the minimum size threshold are skipped.

Raw zips land in:
- `raw/zips/monthly/` — complete monthly archives
- `raw/zips/daily/` — daily patches for incomplete months

```bash
mkdir -p logs
nohup python3 scripts/01_download.py > logs/01_download.log 2>&1 &
echo "PID: $!"
tail -f logs/01_download.log
```

Monitor progress:

```bash
# files landed in GCS
gsutil ls gs://your-bucket/raw/zips/monthly/ | wc -l

# any errors or warnings
grep "ERROR\|WARNING" logs/01_download.log
```

### Script 2 — Convert CSV to Parquet

[`scripts/02_csv_to_parquet.py`](../scripts/02_csv_to_parquet.py)

Reads raw CSV zip files from GCS and converts them to parquet using PyArrow's C++ streaming CSV reader. Schema is applied explicitly — no type inference, every column cast correctly, `time` column always written as UTC microsecond timestamp. For months where both a monthly archive and daily patch files exist, the script stitches them together using timestamp-based deduplication — monthly file processed first, daily files appended with overlap filtered by the last timestamp of the preceding file.

Output lands in `v2/trades_parquet_flat/` as one parquet file per asset per month. Files are written to local `/tmp/` first then uploaded to GCS to minimise peak RAM usage.

> **Note:** BTCUSDT March 2023 is handled as a known anomaly — the Binance monthly archive for this month only contains days 1–12. The script detects this and stitches the remaining 19 daily files using timestamp filtering.

```bash
nohup python3 scripts/02_csv_to_parquet.py > logs/02_csv_to_parquet.log 2>&1 &
echo "PID: $!"
tail -f logs/02_csv_to_parquet.log
```

Monitor progress:

```bash
# parquet files in v2
gsutil ls gs://your-bucket/v2/trades_parquet_flat/ | wc -l

# disk space on vm
df -h /tmp
```

### Script 3 — Quality Audit

[`scripts/03_quality_audit.py`](../scripts/03_quality_audit.py)

Runs a three-layer audit against the converted parquet files:

1. **File inventory** — checks every expected parquet file exists in GCS and is above the minimum size threshold
2. **Schema check** — downloads parquet metadata for every file and verifies column names and types match the expected schema exactly
3. **BigQuery audit** — runs SQL queries against the external table checking null IDs, bad prices, extreme outlier prices above $10M, bad quantities, duplicate trade IDs, timestamp validity, and days with suspiciously low trade counts

Produces a structured JSON report saved to `logs/quality_audit_report.json`. Exits with code 1 if any issues are found.

Before running this script, update the BigQuery external table to point at `v2/trades_parquet_flat/` as described in Step 4.

```bash
nohup python3 scripts/03_quality_audit.py > logs/03_quality_audit.log 2>&1 &
echo "PID: $!"
tail -f logs/03_quality_audit.log
```

Check the report:

```bash
cat logs/quality_audit_report.json
```

---

## Step 8 — Delete the VM

Once the pipeline is complete and logs confirm success, delete the VM immediately to stop billing:

```bash
exit
gcloud compute instances delete pipeline-vm --zone=us-central1-a --quiet
```

Verify:

```bash
gcloud compute instances list
```

---

## GCS Path Reference

| Path | Contents |
|------|----------|
| `raw/zips/monthly/` | Raw zip archives from Binance Vision, monthly granularity |
| `raw/zips/daily/` | Raw zip archives for gap patches, daily granularity |
| `v2/trades_parquet_flat/` | Production parquet files, one per asset per month |

---

## Dataset Summary

| Asset | Days Present | Total Trades | Duplicates | Start | End |
|-------|-------------|--------------|------------|-------|-----|
| BTCUSDT | 909 | 2,442,072,262 | 0 | 2020-02-01 | 2024-12-31 |
| ETHUSDT | 909 | 1,080,172,305 | 0 | 2020-02-01 | 2024-12-31 |
| SOLUSDT | 727 | 485,194,970 | 0 | 2020-11-01 | 2024-12-31 |

> SOL starts November 2020 — this is expected, not a data gap. SOL was listed on Binance in November 2020.

---

## Sampling Window Rationale

The dataset covers five deliberate six-month windows rather than a continuous time series. This design choice serves two purposes: it captures distinct market regimes with different liquidity characteristics, and it manages class imbalance by ensuring stress and non-stress periods are represented in roughly comparable proportions across the dataset.

| Window | Period | Market Context |
|--------|--------|---------------|
| 1 | Feb 2020 — Jul 2020 | COVID crash and recovery |
| 2 | Nov 2020 — Apr 2021 | Bull run buildup |
| 3 | Nov 2021 — Apr 2022 | Peak bull market and correction |
| 4 | Nov 2022 — Apr 2023 | Post-FTX recovery |
| 5 | Jul 2024 — Dec 2024 | Post-halving cycle |

---

## Notes for Replicators

- Copy `.env.example` to `.env` and fill in your own GCP details before running anything
- The VM must be in the same region as your GCS bucket for internal network transfers
- `gcloud auth application-default login` is required on the VM before running any Python script that accesses GCS
- Add swap space before running the conversion script — large files like BTCUSDT November 2022 approach the VM RAM ceiling during processing
- All scripts are idempotent — safe to re-run if interrupted, already completed files are skipped
- Scripts 1 and 2 must be run on the VM. Script 3 can be run on the VM or in Cloud Shell since it delegates heavy computation to BigQuery
- BTCUSDT March 2023 is a known data anomaly at source — the Binance monthly archive only contains days 1–12. The pipeline handles this automatically

---

*Last updated: April 19, 2026*
```
