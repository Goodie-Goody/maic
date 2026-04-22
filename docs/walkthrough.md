# MAIC Data Pipeline — Walkthrough

## Overview

This walkthrough documents the full data engineering and machine learning pipeline for the paper "Real-time Liquidity Stress Detection in Cryptocurrency Markets Using Trade Flow Analysis and Machine Learning". It covers everything from GCP account setup to trained models in GCS, written so that anyone can replicate the pipeline from scratch.

> **Important:** Cloud Shell is used only for provisioning infrastructure (creating buckets, datasets, and VMs). All data processing and training scripts must be run on a Compute Engine VM, not in Cloud Shell. Cloud Shell lacks the memory and persistent storage required for multi-gigabyte tick data files and model training.

---

## Step 0 — GCP Account & Infrastructure Setup

Create a GCP account at [console.cloud.google.com](https://console.cloud.google.com). Enable billing by linking a payment method. Once billing is active, open Cloud Shell by clicking the terminal icon in the top right. Cloud Shell is your orchestration terminal — it has `gcloud` and `bq` pre-installed.

Set your project and create your storage and database backend:

```bash
gcloud config set project your-project-id

# Create the GCS bucket in us-central1
gsutil mb -l us-central1 gs://your-bucket-name

# Create the BigQuery dataset
bq mk --dataset --location=us-central1 your-project-id:your-dataset-name

# Enable the Compute Engine API
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

Configure git to securely cache your token in memory for 10 hours:

```bash
git config --global credential.helper 'cache --timeout=36000'
```

On your first `git push` you will be prompted for your username and token once.

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
touch scripts/04_feature_engineering.py
touch scripts/05_label_generation.py
touch scripts/06_train_models.py
touch scripts/verify_features.py
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
*.pkl
.DS_Store
logs/
```

The `logs/` directory is excluded from git because it contains large pkl model files and log outputs. These are backed up to GCS instead (see Step 9).

Create `.env.example` which is pushed to GitHub so replicators know what variables are needed:

```bash
GCP_PROJECT_ID=your-project-id-here
GCP_BUCKET=your-bucket-name-here
GCP_REGION=us-central1
BQ_DATASET=your-bigquery-dataset-name
BQ_TABLE=your-external-table-name
```

`config.py` loads from `.env` and defines all pipeline constants in one place:

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
    ("2020-11", "2021-05"),
    ("2021-11", "2022-05"),
    ("2022-11", "2023-04"),
    ("2024-07", "2024-12"),
]

GCS_PREFIX = "v2/trades_parquet_flat/"
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
polars
google-cloud-storage
google-cloud-bigquery
tqdm
python-dotenv
requests
psutil
hmmlearn
scikit-learn
shap
captum
pyts
matplotlib
xgboost
torch
torchvision
torchaudio
```

---

## Step 4 — Sampling Window Design

The dataset covers five deliberate six-month windows rather than a continuous time series. This design choice serves two purposes: it captures distinct market regimes with different liquidity characteristics, and it manages class imbalance by ensuring stress and non-stress periods are represented in roughly comparable proportions across the dataset.

| Window | Period | Market Context | Key Stress Events |
|--------|--------|---------------|-------------------|
| 1 | Feb 2020 — Jul 2020 | COVID crash and recovery | COVID crash (March 12-13 2020) |
| 2 | Nov 2020 — May 2021 | Bull run and China/Musk crash | May 2021 crash (May 19 2021) |
| 3 | Nov 2021 — May 2022 | Peak bull market and Terra-Luna | Terra-Luna collapse (May 9-12 2022) |
| 4 | Nov 2022 — Apr 2023 | Post-FTX recovery | FTX collapse (November 8-11 2022) |
| 5 | Jul 2024 — Dec 2024 | Post-halving cycle | — |

> **Note on window boundaries:** Windows 2 and 3 were originally designed to end in April, but this was corrected to include May after discovering that two major stress events (May 2021 crash and Terra-Luna collapse) fell outside the original boundaries by a single month. The `validate_against_events` function in Script 5 caught this during label validation.

---

## Step 5 — BigQuery External Table

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

This only needs to be run once, and again whenever the output prefix changes or new files are added to the existing prefix. After this BigQuery can query all parquet files in the production prefix as a single logical table using the `_FILE_NAME` pseudocolumn for file-level filtering.

---

## Step 6 — First Commit and Push

```bash
git add .
git commit -m "add project config, requirements, and environment setup"
git push
```

---

## Step 7 — Pipeline VM Setup (CPU)

The first five scripts run on a CPU-only VM. GPU is not required until the training script (Script 6).

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

`e2-highcpu-8` provides 8 vCPUs and 8GB RAM. The `--scopes=storage-full` flag grants the VM read/write access to all GCS buckets in the project. Running in `us-central1-a` — the same region as the bucket — means all transfers use Google's internal network with no egress costs. Cost is approximately $0.19/hour.

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

Create the `.env` file on the VM:

```bash
nano .env
```

Add swap space to prevent silent OOM failures on large files:

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
free -h
```

---

## Step 8 — Running the Data Pipeline

Each script runs in the background using `nohup` so it survives terminal disconnects.

### Script 1 — Download

[`scripts/01_download.py`](../scripts/01_download.py)

Downloads monthly trade archives for each asset and window from [Binance Vision](https://data.binance.vision). SHA256 checksum verification happens inline during download. For months where the monthly archive is missing days, the script automatically falls back to downloading individual daily files. Full resume capability — files already present in GCS above the minimum size threshold are skipped.

Raw zips land in:
- `raw/zips/monthly/` — complete monthly archives
- `raw/zips/daily/` — daily patches for incomplete months

```bash
mkdir -p logs
nohup python3 scripts/01_download.py > logs/01_download.log 2>&1 &
echo "PID: $!"
tail -f logs/01_download.log
```

### Script 2 — Convert CSV to Parquet

[`scripts/02_csv_to_parquet.py`](../scripts/02_csv_to_parquet.py)

Reads raw CSV zip files from GCS and converts them to parquet using PyArrow's C++ streaming CSV reader. Schema is applied explicitly — no type inference, every column cast correctly, `time` column always written as UTC microsecond timestamp.

**Two known source data anomalies handled automatically:**

1. **BTCUSDT March 2023** — the Binance monthly archive only contains days 1-12. The script detects this and stitches the remaining 19 daily files using timestamp filtering.

2. **Duplicate CSV files in zip archives** — Binance occasionally packages the same CSV file under two different paths within a single zip (observed for May 2021 across all three assets). The script detects this condition, takes only the first CSV, and logs a warning.

```bash
nohup python3 scripts/02_csv_to_parquet.py > logs/02_csv_to_parquet.log 2>&1 &
echo "PID: $!"
tail -f logs/02_csv_to_parquet.log
```

### Script 3 — Quality Audit

[`scripts/03_quality_audit.py`](../scripts/03_quality_audit.py)

Runs a three-layer audit against the converted parquet files:

1. **File inventory** — checks every expected parquet file exists and is above the minimum size threshold
2. **Schema check** — reads parquet metadata for every file and verifies column names and types
3. **BigQuery audit** — runs SQL queries checking null IDs, bad prices, extreme outliers above $10M, bad quantities, duplicate trade IDs, and days with suspiciously low trade counts

Produces a structured JSON report saved to `logs/quality_audit_report.json`. Exits with code 1 if any issues are found.

Before running, ensure the BigQuery external table is refreshed:

```bash
bq query --use_legacy_sql=false --project_id=your-project-id '
CREATE OR REPLACE EXTERNAL TABLE `your-project-id.your-dataset.your-table`
OPTIONS (
  format = "PARQUET",
  uris = ["gs://your-bucket/v2/trades_parquet_flat/*.parquet"]
)
'
```

Then run the audit:

```bash
nohup python3 scripts/03_quality_audit.py > logs/03_quality_audit.log 2>&1 &
echo "PID: $!"
tail -f logs/03_quality_audit.log
cat logs/quality_audit_report.json
```

Expected result:

```
status: PASSED
quality_score: 100.0
BTCUSDT: 971 days, 2,570,682,192 trades, 0 duplicates
ETHUSDT: 971 days, 1,191,086,501 trades, 0 duplicates
SOLUSDT: 789 days,   516,787,062 trades, 0 duplicates
```

### Script 4 — Feature Engineering

[`scripts/04_feature_engineering.py`](../scripts/04_feature_engineering.py)

Computes multi-scale microstructure features from raw tick data using a Polars lazy evaluation pipeline with streaming execution. Raw trades are aggregated into 10-second windows using dynamic grouping and upsampled to produce a strictly regular time series.

Features are computed at three temporal scales — 10 seconds, 60 seconds, and 300 seconds:

| Feature Group | Features | Scales |
|---------------|----------|--------|
| Trade Flow Imbalance | OFI, TCI | 10s, 60s, 300s |
| Trade Intensity | intensity | 10s, 60s, 300s |
| Price Impact | ILLIQ, Kyle_lambda | 10s, 60s, 300s |
| Volatility | RV, VWAP, VWAP_dev | 10s, 60s, 300s |
| Inter-trade Duration | CV_dt | 10s |
| Cross-asset Contagion | OFI_corr, Lead_Lag | 60s, 300s |

Cross-asset contagion features are computed using BTC as the market leader for ETH and SOL, and ETH as an additional lead asset for SOL. This reflects the documented price discovery hierarchy in cryptocurrency markets.

Output lands in `v2/features/` as one parquet file per asset per month.

```bash
nohup python3 scripts/04_feature_engineering.py > logs/04_feature_engineering.log 2>&1 &
echo "PID: $!"
tail -f logs/04_feature_engineering.log
```

### Verify Feature Dataset

[`scripts/verify_features.py`](../scripts/verify_features.py)

Reads only the parquet metadata footer of each feature file via PyArrow byte-range requests. Verifies row counts, column counts, and schema consistency across all files.

```bash
python3 scripts/verify_features.py
```

### Script 5 — Label Generation

[`scripts/05_label_generation.py`](../scripts/05_label_generation.py)

Fits a 3-state Gaussian Hidden Markov Model per asset on four core microstructure features — RV_300s, OFI_300s, Kyle_lambda_300s, and intensity_300s. The HMM identifies latent market regimes unsupervised. States are then labeled by sorting on mean realized volatility: 0 (calm), 1 (elevated), 2 (stress).

**Key design decisions:**

- **StandardScaler** is applied before fitting. The HMM features operate on vastly different scales (OFI bounded in [-1, 1], intensity potentially in hundreds). Without scaling the HMM's covariance estimation would be dominated by high-magnitude features and ignore Order Flow Imbalance entirely.

- **Diagonal covariance** (`covariance_type="diag"`) is used instead of full covariance. Full covariance would require approximately 20 minutes per asset versus 9 minutes for diagonal, and the practical difference in regime detection quality is minimal for this dataset.

- **Multi-seed initialization**. The EM algorithm is sensitive to starting conditions and can get stuck in local minima. The script runs 5 seeds (0 through 4) independently and keeps the model with the highest log likelihood. This is necessary because `hmmlearn` does not support `n_init` in its API.

- **Validation against known events**. After fitting, the labels are validated against four known stress events: COVID crash, May 2021 crash, Terra-Luna collapse, and FTX collapse. The unsupervised model consistently identifies 80-99% of observations within these event windows as State 2 (stress), providing strong evidence that the states are economically meaningful.

- **Model persistence**. Each fitted model is saved to `logs/{ASSET}_hmm_model.pkl` along with its scaler and state mapping. This enables future inference on new data without recomputing the global distribution.

```bash
nohup python3 scripts/05_label_generation.py > logs/05_label_generation.log 2>&1 &
echo "PID: $!"
tail -f logs/05_label_generation.log
```

Expected result:

| Asset | Calm | Elevated | Stress |
|-------|------|----------|--------|
| BTCUSDT | 49.4% | 26.3% | 24.3% |
| ETHUSDT | 50.7% | 25.3% | 24.0% |
| SOLUSDT | 39.2% | 40.1% | 20.7% |

Validation against known events:

| Event | BTC | ETH | SOL |
|-------|-----|-----|-----|
| COVID crash | 86.9% | 88.8% | n/a |
| May 2021 crash | 97.6% | 99.8% | 99.9% |
| Terra-Luna | 47.7% | 69.7% | 71.0% |
| FTX collapse | 81.9% | 69.5% | 95.0% |

---

## Step 9 — Backup Before VM Shutdown

Before spinning down the pipeline VM, back up all logs and pkl files to GCS:

```bash
BACKUP="v2/vm_backup_$(date +%Y%m%d_%H%M)/"

gsutil -m cp -r ~/maic/logs/ gs://your-bucket/${BACKUP}logs/
gsutil -m cp -r ~/maic/scripts/ gs://your-bucket/${BACKUP}scripts/
gsutil cp ~/maic/config.py gs://your-bucket/${BACKUP}
gsutil cp ~/maic/requirements.txt gs://your-bucket/${BACKUP}
gsutil cp ~/maic/.env.example gs://your-bucket/${BACKUP}

gsutil ls gs://your-bucket/${BACKUP}
```

Do not back up `.env` — it contains credentials.

Stop (not delete) the pipeline VM to preserve its disk state for potential future use:

```bash
exit
gcloud compute instances stop pipeline-vm --zone=us-central1-a
```

Stopped VMs incur only disk storage cost (~$2/month for 50GB).

---

## Step 10 — GPU VM Setup (for Script 6)

Script 6 requires a GPU for LSTM and CNN-GAF training. There are three paths depending on your GCP account status.

### Path A — GCP GPU VM (if quota is available)

New GCP accounts often have zero GPU quota. Check:

```bash
gcloud compute project-info describe --project=your-project-id \
  | grep -i "NVIDIA\|GPU"
```

If quota is 0, request an increase via the Quotas page in the console. If the system shows "enter a value between 0 and 0" you are not eligible for self-service quota increase and must contact Google Cloud Sales.

If quota is approved, find the current deep learning VM image family (these get deprecated periodically):

```bash
gcloud compute images list \
  --project=deeplearning-platform-release \
  --filter="family~pytorch" \
  --format="table(family, creationTimestamp)" \
  --sort-by="~creationTimestamp" \
  | head -n 10
```

Create the GPU VM using the most recent image family:

```bash
gcloud compute instances create gpu-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --scopes=storage-full \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE
```

If zone reports `ZONE_RESOURCE_POOL_EXHAUSTED`, try a loop across zones:

```bash
for zone in us-central1-b us-east1-c europe-west4-a us-west1-b; do
  echo "Trying $zone..."
  gcloud compute instances create gpu-vm \
    --zone=$zone \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --scopes=storage-full \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    && echo "SUCCESS in $zone" && break
done
```

SSH in and set up:

```bash
gcloud compute ssh gpu-vm --zone=us-central1-a

# Verify GPU
nvidia-smi

# Clone repo and install
git clone https://github.com/your-username/your-repo.git
cd your-repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

nano .env
gcloud auth application-default login

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Path B — Google Colab Pro (recommended fallback)

If GCP GPU quota is unavailable, Colab Pro provides immediate T4/V100/A100 access for £9.99/month. Sign up at [colab.research.google.com/signup](https://colab.research.google.com/signup).

Create a new notebook. Runtime → Change runtime type → T4 GPU → High-RAM.

Run one cell to verify GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Open the terminal at the bottom left of the Colab screen and proceed exactly as you would on a VM:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
gcloud auth application-default login
gcloud config set project your-project-id

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

nano .env
python3 -c "import torch; print(torch.cuda.is_available())"
```

Colab Pro sessions last up to 24 hours which is sufficient for the full training run.

### Path C — Third-party GPU providers

Vast.ai, Lambda Labs, Paperspace, and RunPod all offer GPU instances with SSH access. GCS access requires `gcloud auth application-default login` which adds some friction but works.

---

## Step 11 — Running the Training Script

### Script 6 — Train Models

[`scripts/06_train_models.py`](../scripts/06_train_models.py)

Trains five models per asset per fold using purged walk-forward cross-validation:

| Model | Library | Interpretability Method |
|-------|---------|------------------------|
| Logistic Regression | scikit-learn | SHAP LinearExplainer |
| Random Forest | scikit-learn | SHAP TreeExplainer |
| XGBoost | xgboost (GPU) | SHAP TreeExplainer |
| LSTM | PyTorch | Integrated Gradients (Captum) |
| CNN-GAF | PyTorch + pyts | Gradient Saliency |

**Key design decisions:**

- **Walk-forward cross-validation** — train on windows 0 to n-1, test on window n. With 5 windows this produces 4 folds.

- **Purging** — the last 30 rows of each training window are dropped to prevent leakage from the 300-second lookback features overlapping into the test set.

- **Three-class primary, binary secondary** — each fold trains both a three-class classifier (calm, elevated, stress) and a binary classifier (stress vs not-stress). The three-class results show methodological completeness; the binary results show practical utility for real-time alerts.

- **Asset-specific and pooled models** — each asset gets its own model set. Additionally, a pooled model trained on all three assets tests the Sirignano and Cont universal price formation hypothesis. The pooled model uses only the 26 features common across all assets plus an asset indicator column.

- **Class weights** — computed dynamically per fold to handle class imbalance. Passed to all models as `class_weight="balanced"` equivalent.

- **XGBoost on GPU** — uses `tree_method="hist"` with `device="cuda"` when available.

- **LSTM architecture** — 2-layer LSTM with 128 hidden units, dropout 0.2, gradient clipping at 1.0, learning rate scheduler reducing on plateau.

- **CNN-GAF architecture** — 2 conv layers (32, 64 channels) with max pooling, 2 fully connected layers (128, n_classes) with dropout 0.3. Uses only the 4 core HMM features to keep image computation tractable.

- **Output structure** — per fold per mode (binary/multiclass): predictions parquet, metrics JSON, confusion matrices PNG, ROC curves PNG, loss curves PNG, SHAP summary plots PNG, saved models (pkl for sklearn/xgb, pt for PyTorch). All uploaded to `gs://your-bucket/v2/results/{asset}/fold_{n}/` after each fold completes.

Run:

```bash
nohup python3 scripts/06_train_models.py > logs/06_train_models.log 2>&1 &
echo "PID: $!"
tail -f logs/06_train_models.log
```

Estimated runtime on T4: 6-8 hours for all assets all folds all modes.

---

## Step 12 — Post-Training Cleanup

Once training completes and results are confirmed in GCS, immediately delete the GPU VM to stop billing:

```bash
exit
gcloud compute instances delete gpu-vm --zone=your-zone --quiet
gcloud compute instances list
```

Back up all training logs and outputs:

```bash
BACKUP="v2/vm_backup_$(date +%Y%m%d_%H%M)_training/"
gsutil -m cp -r ~/maic/logs/ gs://your-bucket/${BACKUP}
```

---

## GCS Path Reference

| Path | Contents |
|------|----------|
| `raw/zips/monthly/` | Raw zip archives from Binance Vision, monthly granularity |
| `raw/zips/daily/` | Raw zip archives for gap patches, daily granularity |
| `v2/trades_parquet_flat/` | Production parquet files, one per asset per month |
| `v2/features/` | Feature parquet files, one per asset per month, 10s windows |
| `v2/labels/` | Label parquet files, one per asset per month, 3-state regimes |
| `v2/results/` | Training outputs — predictions, metrics, plots, models |
| `v2/vm_backup_YYYYMMDD_HHMM/` | Dated VM state backups — logs, scripts, pkl files |

---

## Dataset Summary

| Asset | Days Present | Total Trades | Duplicates | Start | End |
|-------|-------------|--------------|------------|-------|-----|
| BTCUSDT | 971 | 2,570,682,192 | 0 | 2020-02-01 | 2024-12-31 |
| ETHUSDT | 971 | 1,191,086,501 | 0 | 2020-02-01 | 2024-12-31 |
| SOLUSDT | 789 | 516,787,062 | 0 | 2020-11-01 | 2024-12-31 |
| **Total** | | **4,278,555,755** | **0** | | |

> SOL starts November 2020 — this is expected, not a data gap. SOL was listed on Binance in November 2020.

---

## Feature Dataset Summary

| Asset | Files | Rows | Columns |
|-------|-------|------|---------|
| BTCUSDT | 30 | 8,389,440 | 28 |
| ETHUSDT | 30 | 8,389,440 | 31 |
| SOLUSDT | 24 | 6,816,952 | 34 |
| **Total** | **90** | **23,595,832** | — |

BTCUSDT has 28 columns — 26 single-asset features plus `time` and `price`. ETHUSDT has 3 additional BTC cross-asset contagion features. SOLUSDT has 6 additional cross-asset features — 3 from BTC and 3 from ETH.

---

## Notes for Replicators

- Copy `.env.example` to `.env` and fill in your own GCP details before running anything
- The VM must be in the same region as your GCS bucket for internal network transfers
- `gcloud auth application-default login` is required on every VM before running any Python script that accesses GCS
- Add swap space before running the conversion script
- All scripts are idempotent — safe to re-run if interrupted, already completed files are skipped
- Scripts 1 through 5 run on a CPU VM. Script 6 requires GPU (or CPU with long runtime)
- The `validate_against_events` function in Script 5 is critical — it caught the Windows 2/3 boundary error that would have silently excluded two major stress events
- Binance occasionally packages duplicate CSV files inside monthly zips. Script 2 handles this generically
- Deep learning VM image families get deprecated periodically — use the `gcloud compute images list` command in Step 10 to find the current family name
- If GCP GPU quota is unavailable, Colab Pro at £9.99/month is the fastest fallback path

---

*Last updated: April 22, 2026*
