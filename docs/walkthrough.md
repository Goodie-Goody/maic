# MAIC Pipeline — Walkthrough

## Overview

This walkthrough documents the full infrastructure and execution path for the paper *Near Real-Time Liquidity Stress Detection in Cryptocurrency Markets Using Trade Flow Analysis and Machine Learning*. It covers everything from GCP account setup to trained models, paper figures, and analysis outputs.

**Python version:** 3.11+ required. Python 3.11 is recommended — all scripts were developed and validated on 3.11. Python 3.12 may have cuML compatibility issues.

**Compute model:** Data preparation and analysis run on CPU (any machine). Model training requires a CUDA-capable GPU. The pipeline was built and validated on RunPod using an NVIDIA RTX PRO 4500 Blackwell GPU (34GB VRAM).

> **GCS is the single source of truth.** All data, features, labels, models, and results live in your GCS bucket. Compute instances are ephemeral — spin them up, run scripts, spin them down. Everything that matters persists in the cloud.

---

## Step 0 — GCP Account and Infrastructure Setup

Create a GCP account at [console.cloud.google.com](https://console.cloud.google.com). Enable billing. Open Cloud Shell and provision your storage and database backend:

```bash
gcloud config set project your-project-id

# Storage bucket — same region as your compute to avoid egress costs
gsutil mb -l us-central1 gs://your-bucket-name

# BigQuery dataset for the quality audit
bq mk --dataset --location=us-central1 your-project-id:your-dataset-name
```

Create a service account with Storage read/write permissions and download its key as `gcp-key.json`. This file must never be committed to git — it is covered by `.gitignore`.

---

## Step 1 — Repository Setup

Clone the repo and configure your environment:

```bash
git clone https://github.com/Goodie-Goody/maic.git
cd maic

# Create your .env from the template
cp .env.example .env
# Fill in: GCP_PROJECT_ID, GCP_BUCKET, GCP_REGION, BQ_DATASET, BQ_TABLE

# Add your service account key
cp /path/to/your-key.json gcp-key.json
```

Your `.env` file should look like:

```bash
GCP_PROJECT_ID=your-project-id
GCP_BUCKET=your-bucket-name
GCP_REGION=us-central1
BQ_DATASET=your-bigquery-dataset
BQ_TABLE=your-external-table
```

---

## Step 2 — Environment Setup

Run `setup.sh` once per machine. It installs all dependencies and automatically selects the correct PyTorch build for your GPU:

```bash
bash setup.sh
```

What `setup.sh` does:
- Installs all packages from `requirements.txt`
- Detects GPU compute capability via `nvidia-smi`
- Installs **PyTorch nightly cu128** for Blackwell (sm_120) GPUs
- Installs **PyTorch stable cu121** for all other GPUs (Ampere, Ada Lovelace, Hopper)
- Installs cuML for GPU-accelerated Random Forest
- Makes `cpu_pipeline.sh`, `gpu_pipeline.sh`, `cpu_post_gpu.sh`, and `status.sh` executable
- Configures git credentials

After setup, validate that everything is in order before running any computation:

```bash
bash cpu_pipeline.sh --check   # env, imports, GCS bucket
bash gpu_pipeline.sh --check   # GPU, CUDA, cuML, XGBoost GPU, GCS bucket
```

---

## Step 3 — Sampling Window Design

The dataset covers five deliberate six-month windows rather than a continuous time series. This design serves two purposes: it captures distinct market regimes with different liquidity characteristics, and it ensures stress and non-stress periods are represented in comparable proportions.

| Window | Period | Market Context | Key Event |
|--------|--------|---------------|-----------|
| 0 | Feb 2020 – Jul 2020 | COVID crash and recovery | COVID crash (Mar 12 2020) |
| 1 | Nov 2020 – May 2021 | Bull run | May 2021 crash (May 19 2021) |
| 2 | Nov 2021 – May 2022 | Peak bull market and contagion | Terra-Luna collapse (May 9 2022) |
| 3 | Nov 2022 – Apr 2023 | Post-FTX recovery | FTX bankruptcy (Nov 8 2022) |
| 4 | Jul 2024 – Dec 2024 | Post-halving cycle | — |

> **Note on window 0 and the COVID event:** Window 0 is always training data across all folds. No out-of-sample predictions exist for March 2020, so the COVID-19 crash is excluded from the lead-time analysis.

> **Note on SOLUSDT:** Data collection begins November 2020 to align with the asset's Binance listing date. SOLUSDT has no data in Window 0.

> **Note on window boundaries:** Windows 1 and 2 were originally designed to end in April, but corrected to include May after two major stress events (May 2021 crash and Terra-Luna) fell outside the original boundaries by a single month.

---

## Step 4 — BigQuery External Table

Before running the quality audit, create the BigQuery external table pointing at the production parquet files. Run this in Cloud Shell once:

```bash
bq query --use_legacy_sql=false --project_id=your-project-id '
CREATE OR REPLACE EXTERNAL TABLE `your-project-id.your-dataset.your-table`
OPTIONS (
  format = "PARQUET",
  uris = ["gs://your-bucket/v2/trades_parquet_flat/*.parquet"]
)
'
```

---

## Step 5 — Running the Pipeline

The pipeline is split across three bash scripts that enforce correct execution order. Each script uses local `.done` markers and GCS pipeline markers to skip completed stages safely.

### Option A — Foreground (recommended for first run / validation)

Watch output live. Terminal must stay open:

```bash
bash cpu_pipeline.sh      # stages 01–05b
bash gpu_pipeline.sh      # stages 06a–06d
bash cpu_post_gpu.sh      # stages 07a–10
```

### Option B — Background with nohup (recommended for long training runs)

Survives terminal disconnects:

```bash
nohup bash cpu_pipeline.sh  >> logs/cpu_pipeline.log  2>&1 &
# wait for completion, then:
nohup bash gpu_pipeline.sh  >> logs/gpu_pipeline.log  2>&1 &
# wait for completion, then:
nohup bash cpu_post_gpu.sh  >> logs/cpu_post_gpu.log  2>&1 &
```

Monitor progress any time:

```bash
bash status.sh
tail -f logs/gpu_pipeline.log
```

### Pipeline Flags

All scripts support:

```bash
--check      # validate environment only, no computation
--dry-run    # print execution plan, run nothing
--from=N     # resume from stage N
--only=X     # GPU pipeline only: run a single stage (e.g. --only=06d)
```

---

## Step 6 — What Each Script Does

### 01 — Download

Downloads monthly trade archives for each asset and window from [Binance Vision](https://data.binance.vision). SHA256 checksum verification inline. For months where the monthly archive is incomplete, falls back to daily files automatically. Full resume capability — files already in GCS above the minimum size threshold are skipped.

### 02 — CSV to Parquet

Converts raw CSV zip files to Parquet using PyArrow's C++ streaming reader. Schema applied explicitly — no type inference, `time` always written as UTC microsecond timestamp. Two known Binance data anomalies handled automatically:
- **BTCUSDT March 2023** — monthly archive only contains days 1-12; remaining 19 daily files stitched using timestamp filtering
- **Duplicate CSV files in zip archives** — detected and deduplicated with a warning

### 03 — Quality Audit

Three-layer audit:
1. File inventory — every expected parquet exists and exceeds minimum size
2. Schema check — column names and types consistent across all files
3. BigQuery audit — null IDs, bad prices, extreme outliers, duplicate trade IDs, low-count days

Produces a JSON report at `logs/quality_audit_report.json`. Exits with code 1 if any issues found.

Expected result: `status: PASSED, quality_score: 100.0`

### 04a — Feature Engineering

Computes seven microstructure features at three temporal scales (10s, 60s, 300s) using Polars lazy evaluation with streaming execution. Cross-asset contagion features (OFI correlation, lead-lag) computed using BTC as market leader for ETH and SOL, and ETH as an additional lead for SOL. Output: `v2/features/`, one parquet per asset per month.

### 04b — Stationarity and Fractional Differencing

Runs Augmented Dickey-Fuller tests on all features at 5% significance. All seven microstructure features are natively stationary — they measure rates, ratios, and local dynamics rather than price levels. Only the raw price series is non-stationary. Applies fractional differencing with asset-specific minimum differencing orders (BTC: d=0.3, ETH: d=0.4, SOL: d=0.2) to preserve maximum memory content while achieving stationarity. Output: `v2/features_fracdiff/`.

### 05a — Label Generation

Fits a 3-state Gaussian HMM per asset on four features: RV_300s, OFI_300s, Kyle_lambda_300s, intensity_300s. StandardScaler applied before fitting. Diagonal covariance. Five seeds evaluated, best log-likelihood retained. States mapped by RV mean rank: 0=calm, 1=elevated, 2=stress. Labels validated against four known crisis events. Model pickles saved to `logs/` and backed up to GCS. Output: `v2/labels/`, one parquet per asset per month.

### 05b — Feature Verification

Reads only parquet metadata footers via byte-range requests. Verifies row counts, column counts, and schema consistency across all feature files. Fast — no data movement.

### 06a–06d — Model Training

Four training scripts of increasing scope:

| Script | Scope | Output Path |
|--------|-------|-------------|
| 06a | Baseline: asset-specific, folds 1-4 | `v2/results/` |
| 06b | Extended: asset-specific + pooled | `v2/results/` |
| 06c | Ablation: fracdiff vs raw price | `v2/results_ablation/` |
| 06d | Production: 5 seeds × 4 folds × pooled | `v2/results_production/` |

06d is the primary production run. It uses a purged expanding walk-forward scheme with a 30-minute embargo between training and test windows, GPU-accelerated cuML Random Forest and XGBoost, BF16 mixed precision for LSTM, and writes GCS stage markers after every completed fold-model combination for fine-grained resumption.

Estimated runtime on RTX PRO 4500 Blackwell: 06d takes 6–10 hours for 5 seeds × 4 folds.

### 07a–07c — Aggregation

Reads results from GCS and assembles `production_results.csv` and `production_results.parquet` — the single source of truth for all reported metrics. 07c is the most important: it produces the production results used in the paper and loaded dynamically by scripts 08 and 10.

### 08 — Paper Figure Generation

Generates all five publication figures locally and uploads to `v2/paper_figures/` in GCS:
- Fig 1: SHAP XGBoost multiclass feature attribution
- Fig 2: Fold progression learning curves
- Fig 3: SHAP Random Forest binary (surrogate-based)
- Fig 4: LSTM Integrated Gradients
- Fig 5: ROC curves across all five architectures

### 09 — Lead-Time Analysis

Evaluates the framework's operational utility across three crisis events using genuine out-of-sample predictions. Each event is evaluated against the fold whose test window contains it (fold 1 for May 2021, fold 2 for Terra-Luna, fold 3 for FTX). Warning declared at first run of ≥ 2 consecutive 300-second bars with P(stress) > 0.85.

Results: Terra-Luna 108.3 min, FTX 176.0 min, May 2021 ≥ 240 min (lower bound — signal present throughout the 4-hour search window, reflecting the event's gradual character).

### 10 — HMM Robustness Check

Compares globally-fitted HMM labels against locally-fitted HMM labels for Fold 3. Trains a fresh HMM on Windows 0–2 only per asset, applies it to the test window, and compares label agreement. Then trains XGBoost on locally-labelled data and evaluates against both local and global test labels. The cross-evaluation (local model vs global labels) yielding F1-Stress of 0.4802 confirms that local refitting introduces its own calibration bias — global fitting is the methodologically superior choice.

---

## Step 7 — Skip/Resume Logic

Every script in the pipeline checks for a GCS completion marker before doing any work:

```
gs://<bucket>/v2/pipeline_markers/<script_name>.done
```

If the marker exists, the script logs a skip message and exits immediately — no computation, no risk of overwriting results. The marker is written only on successful completion.

To force any script to rerun:

```bash
gsutil rm gs://your-bucket/v2/pipeline_markers/05a_label_generation.done
```

The bash pipeline runners additionally use local `.done` files in `logs/` for stage-level skip logic. Both levels work together: the GCS marker prevents redundant computation even if local markers are absent (e.g. on a fresh compute instance).

---

## Step 8 — Before Shutting Down a Compute Instance

Back up HMM model pickles and logs to GCS before terminating any instance:

```bash
BACKUP="v2/vm_backup_$(date +%Y%m%d_%H%M)/"
gsutil -m cp logs/*.pkl gs://your-bucket/${BACKUP}logs/
gsutil -m cp -r logs/ gs://your-bucket/${BACKUP}logs/
```

Do not back up `gcp-key.json` or `.env` — they contain credentials.

---

## GCS Path Reference

| Path | Contents |
|------|----------|
| `raw/zips/monthly/` | Raw Binance zip archives, monthly granularity |
| `raw/zips/daily/` | Daily patches for incomplete months |
| `v2/trades_parquet_flat/` | Converted trade data, one parquet per asset per month |
| `v2/features/` | Raw microstructure features |
| `v2/features_fracdiff/` | Fractionally differenced features (model input) |
| `v2/labels/` | HMM regime labels (0=calm, 1=elevated, 2=stress) |
| `v2/results/` | Baseline and extended training outputs |
| `v2/results_ablation/` | Ablation study outputs |
| `v2/results_production/` | Production run — 5 seeds × 4 folds × all models |
| `v2/paper_figures/` | Publication figures |
| `v2/pipeline_markers/` | Script completion markers |
| `v2/vm_backup_YYYYMMDD_HHMM/` | Dated backups |

---

## Dataset Summary

| Asset | Total Trades | Start | Columns |
|-------|-------------|-------|---------|
| BTCUSDT | 2,570,682,192 | Feb 2020 | 28 |
| ETHUSDT | 1,191,086,501 | Feb 2020 | 31 (+3 BTC cross-asset) |
| SOLUSDT | 516,787,062 | Nov 2020 | 34 (+6 cross-asset) |
| **Total** | **4,278,555,755** | | |

Feature files: 90 parquet files, 23,595,832 total rows across 5 windows × 3 assets.

---

## Notes for Replicators

- **Python 3.11 is required.** cuML, PyTorch nightly, and polars are sensitive to Python version. Do not use 3.12 without testing cuML compatibility first.
- All scripts are idempotent — safe to rerun if interrupted. Already completed files are skipped at both the item level (inside each script) and the script level (GCS pipeline markers).
- `gcloud auth application-default login` is not required when using a service account key via `GOOGLE_APPLICATION_CREDENTIALS`. `setup.sh` sets this environment variable automatically.
- Fold 4 training (18.8M rows) requires at least 20GB GPU VRAM. The production run was validated on a 34GB Blackwell GPU. Lower VRAM machines may need reduced batch sizes.
- The `v2/vm_backup_20260420_1709/` prefix in GCS contains the original HMM model pickles from the initial pipeline run, used to validate that the re-run HMM converged to an identical solution.

---

*Last updated: May 2026*