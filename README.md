# MAIC — Near Real-Time Liquidity Stress Detection in Cryptocurrency Markets

> **Paper:** *Near Real-Time Liquidity Stress Detection in Cryptocurrency Markets Using Trade Flow Analysis and Machine Learning*
> Goodness Kalu, Uchenna Ejike, Joseph Edet, Emmanuel Fagbuyi, Godfrey Kunde, and Hannah Igboke
> WorldQuant University · Preprint submitted May 2026

This repository contains the complete, fully reproducible research pipeline — from raw Binance trade data to trained models, publication figures, lead-time analysis, and HMM robustness validation. Every number in the paper traces back to a script in this repository.

---

## What This Repository Does

Cryptocurrency markets can collapse within minutes. Traditional risk systems are too slow. This project builds a near real-time early warning system for liquidity stress using only publicly available high-frequency trade data from Binance.

**The core findings:**
- XGBoost on seven trade-based microstructure features achieves binary weighted F1 of **0.9706** on the final out-of-sample window, with near-zero variance across random seeds (range: 0.0006)
- The framework provides **108–176 minutes** of actionable early warning before acute structural collapses (Terra-Luna, FTX), and detects sustained deterioration events (May 2021) at least 4 hours before HMM-defined onset
- A formal HMM robustness check confirms that global label fitting does not introduce material look-ahead bias — local refitting introduces its own calibration errors and is demonstrably inferior
- All seven microstructure features are natively stationary by construction; only the raw price series requires fractional differencing

---

## Repository Structure

```
maic/
├── config.py                        # All pipeline constants — assets, windows, paths
├── requirements.txt                 # Python dependencies
├── setup.sh                         # One-command environment setup (GPU-adaptive)
├── cpu_pipeline.sh                  # Run pre-GPU stages (01–05b)
├── gpu_pipeline.sh                  # Run GPU training stages (06a–06d)
├── cpu_post_gpu.sh                  # Run post-GPU analysis (07a–10)
├── status.sh                        # Pipeline status dashboard
├── .env.example                     # Environment variable template
├── scripts/
│   ├── 01_download.py               # Download raw trade data from Binance
│   ├── 02_csv_to_parquet.py         # Convert CSV archives to Parquet
│   ├── 03_quality_audit.py          # Three-layer data quality audit
│   ├── 04a_feature_engineering.py   # Compute microstructure features
│   ├── 04b_stationarity_fracdiff.py # ADF testing + fractional differencing
│   ├── 05a_label_generation.py      # HMM regime labelling
│   ├── 05b_verify_features.py       # Feature dataset verification
│   ├── 06a_train_models.py          # Baseline training
│   ├── 06b_train_models.py          # Extended training
│   ├── 06c_train_ablation.py        # Ablation study (fractional differencing)
│   ├── 06d_train_production.py      # Production run (5 seeds × 4 folds)
│   ├── 07a_aggregate_results.py     # Aggregate baseline results
│   ├── 07b_aggregate_ablation.py    # Aggregate ablation results
│   ├── 07c_aggregate_production.py  # Aggregate production results
│   ├── 08_generate_paper_figures.py # Generate all publication figures
│   ├── 09_lead_time_analysis.py     # Predictive lead-time analysis
│   └── 10_hmm_robustness_check.py   # HMM global vs local label comparison
└── docs/
    └── walkthrough.md               # Detailed step-by-step infrastructure guide
```

---

## Data

Raw trade data collected from Binance's public REST API. Three assets, five deliberate six-month windows designed to capture distinct market regimes:

| Window | Period | Market Context | Key Event |
|--------|--------|---------------|-----------|
| 0 | Feb 2020 – Jul 2020 | COVID crash and recovery | COVID crash (Mar 12 2020) |
| 1 | Nov 2020 – May 2021 | Bull run | May 2021 crash |
| 2 | Nov 2021 – May 2022 | Peak bull + contagion | Terra-Luna collapse |
| 3 | Nov 2022 – Apr 2023 | Post-FTX recovery | FTX bankruptcy (Nov 8 2022) |
| 4 | Jul 2024 – Dec 2024 | Post-halving cycle | — |

**Dataset summary:**

| Asset | Total Trades | Start | Note |
|-------|-------------|-------|------|
| BTCUSDT | 2,570,682,192 | Feb 2020 | — |
| ETHUSDT | 1,191,086,501 | Feb 2020 | — |
| SOLUSDT | 516,787,062 | Nov 2020 | Listed on Binance Nov 2020 |
| **Total** | **4,278,555,755** | | |

All data stored in Google Cloud Storage as partitioned Parquet files. No data is included in this repository — everything is fetched from Binance and stored in your own GCS bucket.

---

## Features

Seven microstructure features computed at 300-second resolution, each grounded in market microstructure theory:

| Feature | Description | Theoretical Basis |
|---------|-------------|-------------------|
| OFI | Order Flow Imbalance — directional pressure of informed trading | Lee & Ready (1991) |
| RV | Realised Volatility — model-free intra-period price variation | Andersen & Bollerslev |
| Kyle's λ | Price impact coefficient — market depth and adverse selection | Kyle (1985) |
| ILLIQ | Amihud illiquidity — price movement per unit of volume | Amihud (2002) |
| VWAP deviation | Execution quality and intrabar price pressure | — |
| Trade Intensity | Normalised trade count — participation signal | — |
| TCI | Trade Concentration Index — Herfindahl concentration of trade sizes | — |

**A key finding:** all seven features are natively stationary by construction. Each measures a rate, ratio, or local dynamic rather than a price level — OFI is bounded in [-1, 1], RV aggregates within-bar returns, Kyle's λ uses price changes not levels, ILLIQ is a ratio of return to volume. Only the raw price series requires transformation, which is applied via fractional differencing (d* = 0.3, 0.4, 0.2 for BTC, ETH, SOL respectively) to preserve maximum memory content while ensuring stationarity.

---

## Experimental Design

**Walk-forward validation** with purged expanding folds and a 30-minute embargo between training and test windows. Each fold's test window maps to a specific crisis event for the lead-time analysis:

| Fold | Train | Test Window | Crisis Event |
|------|-------|-------------|--------------|
| 1 | Window 0 | Window 1 (Nov 2020–May 2021) | May 2021 crash |
| 2 | Windows 0–1 | Window 2 (Nov 2021–May 2022) | Terra-Luna |
| 3 | Windows 0–2 | Window 3 (Nov 2022–Apr 2023) | FTX bankruptcy |
| 4 | Windows 0–3 | Window 4 (Jul 2024–Dec 2024) | — |

**Five model architectures** evaluated under a unified protocol across two classification formulations (3-class and binary), two dataset configurations (asset-specific and pooled), and five random seeds:

| Model | Notes |
|-------|-------|
| Logistic Regression | Interpretable linear baseline |
| Random Forest | GPU-accelerated via cuML (34GB VRAM for fold 4) |
| XGBoost | Primary architecture — GPU hist method |
| LSTM | 2-layer, 128 hidden units, BF16 mixed precision |
| CNN-GAF | Gramian Angular Field encoding, 4-layer CNN |

---

## Key Results

**Production stability (pooled, all folds, 5 seeds):**

| Model | Binary F1-W | Multiclass F1-W |
|-------|------------|-----------------|
| XGBoost | 0.947 ± 0.052 | 0.912 ± 0.064 |
| Random Forest | 0.945 ± 0.040 | 0.888 ± 0.080 |
| LSTM | 0.822 ± 0.072 | 0.661 ± 0.156 |
| CNN-GAF | 0.541 ± 0.140 | 0.415 ± 0.095 |

**At Fold 4 (18.8M training rows, most representative window):**
XGBoost achieves binary weighted F1 of **0.9706** with a seed range of only 0.0006 — effectively deterministic at this training data scale.

**Lead-time analysis** (P > 0.85, ≥ 2 consecutive bars, fold-specific out-of-sample predictions):

| Event | Fold | Lead Time |
|-------|------|-----------|
| May 2021 Crash | 1 | ≥ 240 min (lower bound) |
| Terra-Luna Collapse | 2 | 108.3 min |
| FTX Bankruptcy | 3 | 176.0 min |

---

## Replicating the Pipeline

### Requirements

- A GCP project with a Cloud Storage bucket
- A GCP service account key with Storage read/write permissions, saved as `gcp-key.json` in the repo root
- A `.env` file (copy from `.env.example`) with your project and bucket details
- A CUDA-capable GPU for training (minimum 20GB VRAM recommended for Fold 4)
- `setup.sh` run once to configure the environment

### Setup

```bash
git clone https://github.com/Goodie-Goody/maic.git
cd maic
cp .env.example .env          # fill in GCP_PROJECT_ID, GCP_BUCKET, GCP_REGION
cp /path/to/key.json gcp-key.json
bash setup.sh                  # installs packages, detects GPU, selects PyTorch variant
```

`setup.sh` automatically selects the correct PyTorch build for your hardware:
- **Blackwell (sm_120)** → PyTorch nightly cu128
- **All other GPUs** → PyTorch stable cu121

### Validate Before Running

```bash
bash cpu_pipeline.sh --check   # verify env, imports, GCS access
bash gpu_pipeline.sh --check   # verify GPU, CUDA, cuML, XGBoost
```

### Run the Pipeline

```bash
bash cpu_pipeline.sh           # stages 01–05b: data prep (CPU only)
bash gpu_pipeline.sh           # stages 06a–06d: model training (GPU required)
bash cpu_post_gpu.sh           # stages 07a–10: aggregation + analysis (CPU)
```

**For long-running stages, use nohup to survive terminal disconnects:**
```bash
nohup bash gpu_pipeline.sh >> logs/gpu_pipeline.log 2>&1 &
tail -f logs/gpu_pipeline.log
```

**Check progress any time:**
```bash
bash status.sh
```

### Skip/Resume Logic

All scripts check for a GCS completion marker before doing any work:
```
gs://<bucket>/v2/pipeline_markers/<script_name>.done
```

If the marker exists the script exits immediately — no computation, no risk of overwriting completed results. The marker is written only on successful completion.

To force a script to rerun after it has completed:
```bash
gsutil rm gs://<bucket>/v2/pipeline_markers/05a_label_generation.done
```

The bash pipeline runners additionally use local `.done` marker files in `logs/` to track stage completion across sessions. Scripts with internal GCS skip logic (04a, 06a–06d) have their own per-item skip checks independent of these markers.

### Pipeline Flags

All three bash scripts support the same flags:

```bash
bash cpu_pipeline.sh --dry-run    # print execution plan without running anything
bash cpu_pipeline.sh --from=4     # resume from stage 4
bash gpu_pipeline.sh --only=06d   # run a single stage
bash gpu_pipeline.sh --check      # validate environment only
```

---

## GCS Path Reference

| Path | Contents |
|------|----------|
| `raw/zips/monthly/` | Raw Binance zip archives |
| `v2/trades_parquet_flat/` | Converted trade data, one parquet per asset per month |
| `v2/features/` | Raw microstructure features |
| `v2/features_fracdiff/` | Fractionally differenced features (model input) |
| `v2/labels/` | HMM regime labels (0=calm, 1=elevated, 2=stress) |
| `v2/results/` | Baseline and ablation training outputs |
| `v2/results_production/` | Production run outputs (5 seeds × 4 folds) |
| `v2/paper_figures/` | Publication figures uploaded after generation |
| `v2/pipeline_markers/` | Script completion markers for skip/resume |
| `v2/vm_backup_YYYYMMDD_HHMM/` | Dated backups of logs and model pkl files |

---

## HMM Robustness

A formal robustness check (script 10) compares globally-fitted HMM labels against locally-fitted HMM labels trained only on each fold's training data. For ETHUSDT and SOLUSDT, label agreement exceeds 95% and 89% respectively with stress-class agreement above 99%. BTCUSDT shows lower overall agreement (28.9%) driven entirely by the local model overcalling stress in the post-FTX stabilisation period — a miscalibration caused by the local model lacking the distributional context of the 2024 recovery. A cross-evaluation in which a model trained on local labels is assessed against global labels yields a stress-class F1 of 0.4802, confirming that local refitting introduces its own calibration bias. The global HMM is the methodologically superior choice.

---

## Notes for Replicators

- SOLUSDT data starts November 2020. The COVID-19 crash (March 2020) is excluded from the lead-time analysis because Window 0 is always training data — no out-of-sample predictions exist for that period
- The pipeline was developed and validated on an NVIDIA RTX PRO 4500 Blackwell GPU (34GB VRAM) via RunPod cloud compute
- GPU training (06a–06d) requires at least 20GB VRAM for Fold 4's 18.8M training rows. Lower VRAM machines may need to reduce batch size or skip Fold 4
- `gcp-key.json` must never be committed to the repository — it is covered by `.gitignore`
- All scripts are idempotent: safe to rerun if interrupted. Files already present in GCS are skipped at the item level; completed scripts are skipped at the pipeline level via GCS markers
- The `logs/` directory is git-ignored. Back up HMM pkl files to GCS before shutting down a compute instance: `gsutil -m cp logs/*.pkl gs://<bucket>/v2/vm_backup_$(date +%Y%m%d)/logs/`

---

## Infrastructure Notes

The full pipeline was built and validated for under £50 of cloud compute — approximately £35 in RunPod GPU time and £10 in GCS storage. The compute breakdown:

| Stage | Hardware | Estimated Cost |
|-------|----------|---------------|
| Data prep (01–05b) | CPU instance | < £2 |
| GPU training (06a–06d) | RTX PRO 4500 Blackwell | ~ £30 |
| Analysis (07a–10) | CPU | < £2 |
| GCS storage | — | ~ £10/month |

For detailed infrastructure setup including BigQuery external tables, VM provisioning, and Colab Pro fallback, see [`docs/walkthrough.md`](docs/walkthrough.md).

---

## Citation

```bibtex
@article{kalu2026maic,
  title   = {Near Real-Time Liquidity Stress Detection in Cryptocurrency Markets
             Using Trade Flow Analysis and Machine Learning},
  author  = {Kalu, Goodness and Ejike, Uchenna and Edet, Joseph and
             Fagbuyi, Emmanuel and Kunde, Godfrey and Igboke, Hannah},
  journal = {SSRN Preprint},
  year    = {2026},
  url     = {https://github.com/Goodie-Goody/maic}
}
```

---

## Acknowledgements

WorldQuant University for educational support. The open-source community behind the scientific Python ecosystem. GPU computing provided via RunPod cloud services on an NVIDIA RTX PRO 4500 Blackwell GPU.
