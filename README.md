# MAIC — Early Warning System for Liquidity Stress in Cryptocurrency Markets

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20RunPod-lightgrey.svg)
![GPU](https://img.shields.io/badge/GPU-CUDA%2012.8-green.svg)

> **Paper:** *Early Warning System for Liquidity Stress in Cryptocurrency Markets Using Trade Flow Analysis and Machine Learning*
> Goodness Kalu, Uchenna Ejike, Joseph Edet, Emmanuel Fagbuyi, Godfrey Kunde, and Hannah Igboke
> WorldQuant University · Preprint submitted May 2026

This repository contains the complete, fully reproducible research pipeline — from raw Binance trade data to trained models, publication figures, lead-time analysis, HMM robustness validation, external crisis validation, and a live inference system. Every number in the paper traces back to a script in this repository.

---

## What This Repository Does

Cryptocurrency markets can collapse within minutes. Traditional risk systems are too slow. This project builds an early warning system for liquidity stress using only publicly available high-frequency trade data from Binance.

> **Important:** This system detects deteriorating **liquidity conditions** — not price direction. Microstructure stress reflects elevated sell pressure, thin market depth, and abnormal trade flow. Price impact is not guaranteed: liquid markets may absorb stress without significant price movement. For acute structural events, historical lead times are 1–3 hours (Terra-Luna: 108 min, FTX: 176 min).

**The core findings:**
- XGBoost on trade-based microstructure features achieves binary weighted F1 of **0.9706** on the final out-of-sample window, with near-zero variance across random seeds (range: 0.0006)
- The framework provides **108–176 minutes** of actionable early warning before acute structural collapses (Terra-Luna, FTX), and detects sustained deterioration events (May 2021) at least 4 hours before HMM-defined onset
- A controlled experiment comparing global vs local HMM labelling across all four folds proves global fitting is methodologically superior — local HMMs trained on narrow windows produce stress percentages ranging from 6% to 94.3% for the same assets and periods, demonstrating systematic miscalibration
- Three-tier external validation using genuinely independent validators (price drawdown, documented crisis timestamps, cross-asset simultaneous stress) confirms the HMM labels are economically meaningful — not self-referential
- Global HMM is near-deterministic: ETH and SOL models show **0.00e+00** parameter difference between independent training runs; BTC shows differences of 10⁻⁷ — floating point noise
- A live inference system (`scripts/12_inference.py`) runs against the Binance public API with no authentication required, logging stress probabilities and outcome tracking in real time

---

## Repository Structure

```
maic/
├── config.py                              # All pipeline constants
├── requirements.txt                       # Python dependencies
├── setup.sh                               # One-command environment setup (GPU-adaptive)
├── cpu_pipeline.sh                        # Stages 01–05b (CPU)
├── gpu_pipeline.sh                        # Stages 06a–06d (GPU required)
├── cpu_post_gpu.sh                        # Stages 07a–12 (CPU)
├── status.sh                              # Pipeline status dashboard
├── .env.example                           # Environment variable template
├── scripts/
│   ├── 01_download.py                     # Download raw trade data from Binance
│   ├── 02_csv_to_parquet.py               # Convert CSV archives to Parquet
│   ├── 03_quality_audit.py                # Three-layer data quality audit
│   ├── 04a_feature_engineering.py         # Compute microstructure features
│   ├── 04b_stationarity_fracdiff.py       # ADF testing + fractional differencing
│   ├── 05a_label_generation.py            # HMM regime labelling
│   ├── 05b_verify_features.py             # Feature dataset verification
│   ├── 06a_train_models.py                # Baseline training
│   ├── 06b_train_models.py                # Extended training
│   ├── 06c_train_ablation.py              # Ablation study
│   ├── 06d_train_production.py            # Production run (5 seeds × 4 folds)
│   ├── 07a_aggregate_results.py           # Aggregate baseline results
│   ├── 07b_aggregate_ablation.py          # Aggregate ablation results
│   ├── 07c_aggregate_production.py        # Aggregate production results
│   ├── 08_generate_paper_figures.py       # Generate all publication figures
│   ├── 09_lead_time_analysis.py           # Predictive lead-time analysis
│   ├── 10_hmm_robustness_check.py         # HMM Fold 3 robustness check
│   ├── 11a_hmm_stability_and_local_labels.py  # Global HMM stability + all-fold local HMM
│   ├── 11b_crisis_validation_full.py      # Three-tier external crisis validation
│   └── 12_inference.py                    # Live stress detection from Binance API
└── docs/
    └── walkthrough.md                     # Detailed step-by-step infrastructure guide
```

---

## Data

Raw trade data collected from Binance's public REST API. Three assets across five deliberate six-month windows designed to capture distinct market regimes:

| Window | Period | Market Context | Key Event |
|--------|--------|----------------|-----------|
| 0 | Feb 2020 – Jul 2020 | COVID crash and recovery | COVID crash (Mar 12 2020) |
| 1 | Nov 2020 – May 2021 | Bull run | May 2021 crash |
| 2 | Nov 2021 – May 2022 | Peak bull + contagion | Terra-Luna collapse |
| 3 | Nov 2022 – Apr 2023 | Post-FTX recovery | FTX bankruptcy (Nov 8 2022) |
| 4 | Jul 2024 – Dec 2024 | Post-halving cycle | — |

| Asset | Total Trades | Start | Note |
|-------|-------------|-------|------|
| BTCUSDT | 2,570,682,192 | Feb 2020 | — |
| ETHUSDT | 1,191,086,501 | Feb 2020 | — |
| SOLUSDT | 516,787,062 | Nov 2020 | Listed on Binance Nov 2020 |
| **Total** | **4,278,555,755** | | |

---

## Features

Seven microstructure features at 300-second resolution, each grounded in market microstructure theory. All seven are natively stationary by construction — only raw price requires fractional differencing (d* = 0.3/0.4/0.2 for BTC/ETH/SOL):

| Feature | Description | Theoretical Basis |
|---------|-------------|-------------------|
| OFI | Order Flow Imbalance | Lee & Ready (1991) |
| RV | Realised Volatility | Andersen & Bollerslev |
| Kyle's λ | Price impact coefficient | Kyle (1985) |
| ILLIQ | Amihud illiquidity ratio | Amihud (2002) |
| VWAP deviation | Intrabar price pressure | — |
| Trade Intensity | Normalised trade count | — |
| TCI | Trade Concentration Index | — |

---

## Experimental Design

Purged expanding walk-forward cross-validation with 30-minute embargo. Five model architectures across two formulations (binary and multiclass), two configurations (asset-specific and pooled), five random seeds:

| Fold | Train | Test Window | Crisis Event |
|------|-------|-------------|--------------|
| 1 | Window 0 | Window 1 | May 2021 crash |
| 2 | Windows 0–1 | Window 2 | Terra-Luna |
| 3 | Windows 0–2 | Window 3 | FTX bankruptcy |
| 4 | Windows 0–3 | Window 4 | — |

---

## Key Results

**Production stability (pooled, all folds, 5 seeds):**

| Model | Binary F1-W | Multiclass F1-W |
|-------|------------|-----------------|
| XGBoost | 0.947 ± 0.052 | 0.912 ± 0.064 |
| Random Forest | 0.945 ± 0.040 | 0.888 ± 0.080 |
| LSTM | 0.822 ± 0.072 | 0.661 ± 0.156 |
| CNN-GAF | 0.541 ± 0.140 | 0.415 ± 0.095 |

**Fold 4 (18.8M training rows):** XGBoost binary F1 = **0.9706**, seed range = 0.0006.

**Lead-time analysis:**

| Event | Lead Time |
|-------|-----------|
| May 2021 Crash | ≥ 240 min |
| Terra-Luna | 108.3 min |
| FTX Bankruptcy | 176.0 min |

**External validation (Script 11b, cross-asset kappa):**

| Fold | Asset | Global HMM κ | Local HMM κ |
|------|-------|-------------|-------------|
| 1 | ETHUSDT | 0.829 (almost perfect) | 0.334 (fair) |
| 2 | ETHUSDT | 0.762 (substantial) | 0.052 (slight) |
| 2 | SOLUSDT | 0.778 (substantial) | 0.021 (slight) |
| 3 | BTCUSDT | 0.775 (substantial) | 0.088 (slight) |
| 4 | ETHUSDT | 0.840 (almost perfect) | 0.621 (substantial) |

Global HMM consistently achieves substantially higher agreement with independent external validators than local HMM across all folds and assets.

---

## Live Inference

Run the early warning system against live Binance data — no API key required:

```bash
# Single snapshot
python3 scripts/12_inference.py --asset BTCUSDT

# All three assets
python3 scripts/12_inference.py --asset all

# Continuous monitoring (every 5 minutes)
python3 scripts/12_inference.py --asset all --loop

# JSON output
python3 scripts/12_inference.py --asset BTCUSDT --json
```

The inference script:
- Fetches all aggregated trades in the last 300 seconds via Binance public API
- Computes all microstructure features identically to the training pipeline
- Applies asset-specific fractional differencing to price
- Runs the production XGBoost model (Fold 4, Seed 42, pooled)
- Applies the continuity condition: WARNING when P(stress) > 0.85 for ≥ 2 consecutive bars
- Logs every inference to `logs/inference/inference_log.csv`
- Tracks outcomes 30 minutes after each WARNING to `logs/inference/outcome_log.csv`

**Outcome classifications:**
- `STRESS_CONFIRMED` — price dropped ≥ 0.3% and stayed down
- `STRESS_ABSORBED` — price dropped ≥ 0.3% but recovered (stress resolved by liquidity)
- `FALSE_POSITIVE` — price never moved meaningfully
- `PRICE_PUMPED` — price rose instead (short squeeze dynamic)

The first production model (`xgb_binary_pooled_fold4_seed42.pkl`) is downloaded automatically from GCS on first run and cached locally.

---

## Validation Scripts

### Script 11a — HMM Stability and Local Label Generation

Proves global HMM superiority through a controlled experiment:

```bash
python3 scripts/11a_hmm_stability_and_local_labels.py
```

**Part A:** Compares April 20 backup HMM models against current production models across all parameters. ETH and SOL show 0.00e+00 difference. BTC shows 10⁻⁷ — floating point noise. The global HMM is deterministic.

**Part B:** Fits local HMMs for all four folds using only training-available data. Produces `global_vs_local_summary.csv` showing local HMM stress% ranging from 6.0% to 94.3% for the same assets and periods — systematic miscalibration vs the global HMM's stable ~19%.

### Script 11b — Three-Tier External Crisis Validation

Validates HMM labels against genuinely independent external signals:

```bash
python3 scripts/11b_crisis_validation_full.py
```

**Tier 1:** Z-tests against four documented crisis events (z-scores 100–333, all p ≈ 0).

**Tier 2:** Global vs local HMM measured against three external validators — none of which were HMM training features:
- Price drawdown (HMM never saw raw price levels)
- Documented crisis timestamps (externally defined)
- Cross-asset simultaneous stress (HMM fitted per-asset independently)

**Tier 3:** Silent microstructure stress event discovery outside known crisis windows.

---

## Replicating the Pipeline

```bash
git clone https://github.com/Goodie-Goody/maic.git
cd maic
cp .env.example .env          # fill in GCP_PROJECT_ID, GCP_BUCKET, etc.
cp /path/to/key.json gcp-key.json
bash setup.sh                  # installs packages, detects GPU
bash cpu_pipeline.sh           # stages 01–05b: data prep
bash gpu_pipeline.sh           # stages 06a–06d: model training (GPU required)
bash cpu_post_gpu.sh           # stages 07a–12: analysis + validation + inference
```

For detailed setup including GCP provisioning, BigQuery external tables, and RunPod configuration, see [`docs/walkthrough.md`](docs/walkthrough.md).

---

## GCS Path Reference

| Path | Contents |
|------|----------|
| `v2/features_fracdiff/` | Fractionally differenced features (model input) |
| `v2/labels/` | Global HMM labels (0=calm, 1=elevated, 2=stress) |
| `v2/labels_local/fold_{n}/` | Local HMM labels per fold (generated by 11a) |
| `v2/results_production/` | Production run outputs (5 seeds × 4 folds) |
| `v2/paper_figures/` | Publication figures |
| `v2/pipeline_markers/` | Script completion markers |

---

## Infrastructure Notes

Built and validated for under £50 of cloud compute:

| Stage | Hardware | Estimated Cost |
|-------|----------|---------------|
| Data prep (01–05b) | CPU | < £2 |
| GPU training (06a–06d) | RTX PRO 4500 Blackwell 34GB | ~ £30 |
| Analysis (07a–12) | CPU | < £2 |
| GCS storage | — | ~ £10/month |

---

## Citation

```bibtex
@article{kalu2026maic,
  title   = {Early Warning System for Liquidity Stress in Cryptocurrency Markets
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

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.