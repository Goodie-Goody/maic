# MAIC Pipeline — Walkthrough

## Overview

This walkthrough documents the full infrastructure and execution path for the paper *Early Warning System for Liquidity Stress in Cryptocurrency Markets Using Trade Flow Analysis and Machine Learning*. It covers everything from GCP account setup to trained models, paper figures, external validation, and the live inference system.

**Python version:** 3.11+ required. Python 3.11 is recommended — all scripts were developed and validated on 3.11. Python 3.12 may have cuML compatibility issues.

**Compute model:** Data preparation and analysis run on CPU (any machine). Model training requires a CUDA-capable GPU. The pipeline was built and validated on RunPod using an NVIDIA RTX PRO 4500 Blackwell GPU (34GB VRAM).

> **GCS is the single source of truth.** All data, features, labels, models, and results live in your GCS bucket. Compute instances are ephemeral — spin them up, run scripts, spin them down. Everything that matters persists in the cloud.

> **On what this system detects:** The inference system detects deteriorating **liquidity conditions**, not price direction. Microstructure stress reflects elevated sell pressure, thin market depth, and abnormal trade flow. Price impact is not guaranteed — liquid markets may absorb stress without significant price movement. For acute structural events, historical lead times are 1–3 hours. This distinction is both theoretically correct and practically important.

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
cp .env.example .env
cp /path/to/your-key.json gcp-key.json
```

Your `.env` file:

```bash
GCP_PROJECT_ID=your-project-id
GCP_BUCKET=your-bucket-name
GCP_REGION=us-central1
BQ_DATASET=your-bigquery-dataset
BQ_TABLE=your-external-table
GOOGLE_APPLICATION_CREDENTIALS=gcp-key.json
GITHUB_TOKEN=your-token
GITHUB_USER=your-username
GITHUB_EMAIL=your-email
```

---

## Step 2 — Environment Setup

Run `setup.sh` once per machine:

```bash
bash setup.sh
```

What `setup.sh` does:
- Validates `.env` and `gcp-key.json`
- Installs system packages including `nano`, `vim`, `tmux`, `htop`
- Installs all packages from `requirements.txt`
- Detects GPU compute capability via `nvidia-smi`
- Installs **PyTorch nightly cu128** for Blackwell (sm_120) GPUs
- Installs **PyTorch stable cu121** for all other CUDA GPUs
- Skips PyTorch reinstall if the correct version is already present
- Installs cuML for GPU-accelerated Random Forest
- Downloads HMM model pickles and production XGBoost model from GCS backup
- Makes all pipeline scripts executable

> **Important:** `setup.sh` exports PATH changes inside its own subshell. After running it, execute `source ~/.bashrc` in your terminal to ensure `gcloud` and `gsutil` are available in the current session.

After setup, validate before running any computation:

```bash
bash cpu_pipeline.sh --check
bash gpu_pipeline.sh --check
```

---

## Step 3 — Sampling Window Design

Five deliberate six-month windows capturing distinct market regimes:

| Window | Period | Market Context | Key Event |
|--------|--------|----------------|-----------|
| 0 | Feb 2020 – Jul 2020 | COVID crash and recovery | COVID crash (Mar 12 2020) |
| 1 | Nov 2020 – May 2021 | Bull run | May 2021 crash (May 19 2021) |
| 2 | Nov 2021 – May 2022 | Peak bull market and contagion | Terra-Luna collapse (May 9 2022) |
| 3 | Nov 2022 – Apr 2023 | Post-FTX recovery | FTX bankruptcy (Nov 8 2022) |
| 4 | Jul 2024 – Dec 2024 | Post-halving cycle | — |

> **Note on Window 0 and COVID:** Window 0 is always training data across all folds. No out-of-sample predictions exist for March 2020, so the COVID-19 crash is excluded from the lead-time analysis but included in crisis validation (Tier 1 of Script 11b).

> **Note on SOLUSDT:** Data collection begins November 2020. SOLUSDT has no data in Window 0 — local HMM Fold 1 cannot be fitted for SOL. This is documented in Script 11a and noted explicitly in the paper.

> **Note on window boundaries:** Windows 1 and 2 were extended to include May after two major stress events fell outside the original April boundaries by a single month.

---

## Step 4 — BigQuery External Table

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

```bash
bash cpu_pipeline.sh      # stages 01–05b (data preparation)
bash gpu_pipeline.sh      # stages 06a–06d (model training)
bash cpu_post_gpu.sh      # stages 07a–12 (analysis, validation, inference)
```

For long-running stages:

```bash
nohup bash gpu_pipeline.sh >> logs/gpu_pipeline.log 2>&1 &
tail -f logs/gpu_pipeline.log
bash status.sh
```

### Pipeline Flags

```bash
--check      # validate environment only
--dry-run    # print execution plan, run nothing
--from=N     # resume from stage N
--only=X     # run a single stage
```

---

## Step 6 — What Each Script Does

### 01 — Download

Downloads monthly trade archives for each asset and window from Binance Vision. SHA256 checksum verification inline. Falls back to daily files for incomplete months. Full resume capability.

### 02 — CSV to Parquet

Converts raw CSV zip files to Parquet using PyArrow's C++ streaming reader. Schema applied explicitly — `time` always written as UTC microsecond timestamp. Handles two known Binance anomalies: BTCUSDT March 2023 partial monthly archive, and duplicate CSV files in zip archives.

### 03 — Quality Audit

Three-layer audit: file inventory, schema consistency, BigQuery statistical audit (null IDs, bad prices, extreme outliers, duplicate trade IDs). Expected result: `status: PASSED, quality_score: 100.0`.

### 04a — Feature Engineering

Computes seven microstructure features at three temporal scales (10s, 60s, 300s) using Polars lazy evaluation. Cross-asset contagion features computed using BTC as market leader.

### 04b — Stationarity and Fractional Differencing

Runs ADF tests on all features. All seven microstructure features are natively stationary — they measure rates and ratios, not price levels. Only raw price requires transformation. Applies fractional differencing per asset (BTC: d=0.3, ETH: d=0.4, SOL: d=0.2). Output: `v2/features_fracdiff/` — this is the model input, not `v2/features/`.

> **Critical note for replicators:** The fractionally differenced features overwrite the `price` column in place. The column named `price` in `v2/features_fracdiff/` is the fractionally differenced value, not the raw price level. Script 12 (inference) handles this correctly.

### 05a — Label Generation

Fits a 3-state Gaussian HMM per asset on {RV_300s, OFI_300s, Kyle_lambda_300s, intensity_300s}. Five seeds, best log-likelihood retained. States mapped by RV mean rank: 0=calm, 1=elevated, 2=stress. Model pickles saved to `logs/` and backed up to GCS.

### 05b — Feature Verification

Reads only parquet metadata footers. Verifies row counts, column counts, schema consistency.

### 06a–06d — Model Training

| Script | Scope | Output |
|--------|-------|--------|
| 06a | Baseline asset-specific | `v2/results/` |
| 06b | Extended asset-specific + pooled | `v2/results/` |
| 06c | Ablation: fracdiff vs raw price | `v2/results_ablation/` |
| 06d | Production: 5 seeds × 4 folds × pooled | `v2/results_production/` |

06d is the primary production run. Purged expanding walk-forward, 30-minute embargo, GPU-accelerated cuML RF and XGBoost, BF16 mixed precision LSTM. Estimated runtime on RTX PRO 4500 Blackwell: 6–10 hours.

### 07a–07c — Aggregation

Assembles `production_results.csv` — the single source of truth for all reported metrics.

### 08 — Paper Figure Generation

Generates all five publication figures:
- Fig 1: SHAP XGBoost multiclass feature attribution
- Fig 2: Fold progression learning curves
- Fig 3: SHAP Random Forest binary (surrogate-based)
- Fig 4: LSTM Integrated Gradients
- Fig 5: ROC curves across all five architectures

### 09 — Lead-Time Analysis

Evaluates operational utility across three crisis events using genuine out-of-sample XGBoost predictions. Warning declared at first run of ≥ 2 consecutive 300-second bars with P(stress) > 0.85.

Results: Terra-Luna 108.3 min, FTX 176.0 min, May 2021 ≥ 240 min.

### 10 — HMM Robustness Check (Fold 3)

Compares global vs local HMM labels for Fold 3 specifically — the FTX bankruptcy fold, chosen as the hardest case for look-ahead bias. Cross-evaluation yields stress-class F1 of 0.4802, confirming global fitting is superior for Fold 3.

### 11a — HMM Stability and All-Fold Local Label Generation

**The most important methodological addition after the main paper.** Two-part analysis:

**Part A — Global HMM Stability Proof**

Compares the April 20 backup HMM pickles against current production pickles across all parameters. Results:
- ETHUSDT: max_abs_diff across all parameters = **0.00e+00**
- SOLUSDT: max_abs_diff = **0.00e+00**
- BTCUSDT: max_abs_diff_means = **2.66e-07** (floating point noise)

The global HMM is deterministic — it converges to the same unique solution regardless of random seed initialisation, at this data scale.

**Part B — Local HMM Generation Across All Folds**

Fits local HMMs for all four folds using only data available at each fold boundary. Key finding: local HMM stress percentages range from **6.0% to 94.3%** for the same assets and time periods. SOL Fold 2 calls 94.3% of all bars stressed — economically indefensible. BTC Fold 3 calls 78.4% stressed during the post-FTX stabilisation period.

This is a controlled experiment: same features, same architecture, same fitting procedure. The only variable is the data horizon. When global HMM wins against external judges, the conclusion is isolated: microstructure stress detection requires cross-cycle, long-horizon training data.

```bash
python3 scripts/11a_hmm_stability_and_local_labels.py
```

Outputs: `global_hmm_stability.csv`, `local_hmm_label_audit.csv`, `global_vs_local_summary.csv`

GCS output: `v2/labels_local/fold_{1,2,3,4}/`

### 11b — Three-Tier External Crisis Validation

Addresses the core circularity critique: "your model predicts labels your HMM defined — how do we know they're economically meaningful?"

**Tier 1 — Known Crisis Validation**

Z-tests comparing stress rates during vs before four documented crisis events. All z-scores 100–333, all p ≈ 0. Cross-asset simultaneous stress: May 2021 shows 90.6% all-asset simultaneous stress.

**Tier 2 — Global vs Local HMM: External Validator Comparison**

Both HMM variants measured against three genuinely independent validators:

1. **Price drawdown** — HMM never saw raw price levels (only fracdiff)
2. **Documented crisis timestamps** — externally defined from academic literature
3. **Cross-asset simultaneous stress** — HMM fitted per-asset, never saw joint dynamics

Note: RV, OFI, Kyle's lambda, and intensity are ALL HMM training features and are therefore excluded as validators — using them would be circular.

Global HMM achieves substantial to almost perfect cross-asset kappa across all folds. Local HMM collapses to slight or near-zero in most cases.

**Tier 3 — Silent Event Discovery**

Microstructure stress episodes outside known crisis windows — events the framework detected without being told to look. Includes the April 2021 flash crash, August 2024 yen carry trade unwind, and multiple 2024 bull market episodes.

```bash
python3 scripts/11b_crisis_validation_full.py
```

Outputs: `crisis_validation_summary.csv`, `crisis_validation_stats.csv`, `crisis_validation_crossasset.csv`, `global_vs_local_kappa.csv`, `crisis_validation_silent_events.csv`, and three publication figures.

### 12 — Live Inference

Runs the production XGBoost model against live Binance public API data. No API key required.

```bash
# Single reading
python3 scripts/12_inference.py --asset BTCUSDT

# All three assets
python3 scripts/12_inference.py --asset all

# Continuous monitoring (5-minute bars, indefinitely)
python3 scripts/12_inference.py --asset all --loop

# JSON output for downstream processing
python3 scripts/12_inference.py --asset all --json

# Asset-specific model instead of pooled
python3 scripts/12_inference.py --asset BTCUSDT --asset-specific
```

**Feature pipeline in inference:**
1. Fetches all aggregated trades in the last 300 seconds via `GET /api/v3/aggTrades` with `startTime/endTime` — exactly matching the 300-second bar construction in training
2. Paginates automatically if Binance returns 1,000 records (BTC regularly exceeds 1,000 aggTrades per 5 minutes)
3. Computes all 27 microstructure features identically to `04a_feature_engineering.py`
4. Applies fractional differencing to price in place (overwrites `price` column) matching `04b_stationarity_fracdiff.py` exactly
5. Adds `asset_id` for pooled model
6. Runs XGBoost forward pass (~1ms inference latency)
7. Applies continuity condition: WARNING when P(stress) > 0.85 for ≥ 2 consecutive bars

**Logging:**
- `logs/inference/inference_log.csv` — every inference, every bar, every asset (50MB rotation)
- `logs/inference/outcome_log.csv` — every WARNING, resolved 30 minutes later

**Outcome classification uses minimum price in the 30-minute window** (not snapshot at T+30) to correctly capture stress events that dipped and recovered — a common pattern where liquid markets absorb microstructure stress without permanent price dislocation (`STRESS_ABSORBED`).

**Feature count:** 27 for asset-specific models, 28 for pooled (+ `asset_id`). The script auto-detects which via `scaler.n_features_in_`.

The production model is downloaded automatically from GCS on first run and cached to `logs/xgb_binary_pooled_fold4_seed42.pkl`.

---

## Step 7 — Skip/Resume Logic

Every script checks for a GCS completion marker:

```
gs://<bucket>/v2/pipeline_markers/<script_name>.done
```

To force rerun:

```bash
# Using gcloud (recommended)
gcloud storage rm gs://your-bucket/v2/pipeline_markers/05a_label_generation.done

# After sourcing .bashrc if gcloud not in PATH
source ~/.bashrc
gcloud storage rm gs://your-bucket/v2/pipeline_markers/11b_crisis_validation.done
```

---

## Step 8 — Before Shutting Down a Compute Instance

```bash
BACKUP="v2/vm_backup_$(date +%Y%m%d_%H%M)/"
gsutil -m cp logs/*.pkl gs://your-bucket/${BACKUP}logs/
gsutil -m cp -r logs/ gs://your-bucket/${BACKUP}logs/
```

Do not back up `gcp-key.json` or `.env`.

---

## GCS Path Reference

| Path | Contents |
|------|----------|
| `v2/features/` | Raw microstructure features |
| `v2/features_fracdiff/` | Fractionally differenced features (model input) |
| `v2/labels/` | Global HMM labels (0=calm, 1=elevated, 2=stress) |
| `v2/labels_local/fold_{n}/` | Local HMM labels per fold (Script 11a) |
| `v2/results_production/` | Production outputs (5 seeds × 4 folds) |
| `v2/paper_figures/` | Publication figures |
| `v2/pipeline_markers/` | Script completion markers |
| `v2/vm_backup_YYYYMMDD_HHMM/` | Dated backups |

---

## Dataset Summary

| Asset | Total Trades | Start | Feature columns |
|-------|-------------|-------|-----------------|
| BTCUSDT | 2,570,682,192 | Feb 2020 | 28 |
| ETHUSDT | 1,191,086,501 | Feb 2020 | 31 |
| SOLUSDT | 516,787,062 | Nov 2020 | 34 |
| **Total** | **4,278,555,755** | | |

Feature files: 90 parquet files, 23,595,832 total rows across 5 windows × 3 assets.

---

## Notes for Replicators

- **Python 3.11 is required.** cuML and PyTorch nightly are sensitive to Python version.
- The `price` column in `v2/features_fracdiff/` is the fractionally differenced value, not raw price. Script 12 handles this correctly — do not use `v2/features/` as model input.
- `source ~/.bashrc` after each new terminal session on RunPod to ensure `gcloud` is in PATH.
- Fold 4 training requires at least 20GB GPU VRAM. Validated on 34GB Blackwell.
- SOL has no Window 0 data — Script 11a correctly skips SOL Fold 1 local HMM.
- The `v2/vm_backup_20260420_1709/` prefix in GCS contains the original HMM model pickles used to validate global HMM determinism in Script 11a Part A.
- All scripts are idempotent — safe to rerun if interrupted.
- CSV output files should not be committed to git — they are pipeline outputs, not source code. Use GCS for backup: `gcloud storage cp *.csv gs://your-bucket/v2/results/`

---

## Infrastructure Cost

Built and validated for under £50:

| Stage | Hardware | Estimated Cost |
|-------|----------|---------------|
| Data prep (01–05b) | CPU | < £2 |
| GPU training (06a–06d) | RTX PRO 4500 Blackwell | ~ £30 |
| Analysis + validation (07a–11b) | CPU | < £2 |
| Live inference (12) | Any machine with internet | £0 |
| GCS storage | — | ~ £10/month |