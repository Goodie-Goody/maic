# MAIC — Early Warning System for Liquidity Stress in Cryptocurrency Markets

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20RunPod-lightgrey.svg)
![GPU](https://img.shields.io/badge/GPU-CUDA%2012.8-green.svg)
![SSRN](https://img.shields.io/badge/SSRN-10.2139%2Fssrn.6761438-blue.svg)

> **Paper:** *An Early Warning System for Liquidity Stress in Cryptocurrency Markets Using Trade Flow Analysis and Machine Learning*
> Goodness Kalu, Uchenna Ejike, Joseph Edet, Temitope E. Fagbuyi, Godfrey Kunde, and Hannah Igboke
> WorldQuant University · Preprint: [SSRN 10.2139/ssrn.6761438](https://doi.org/10.2139/ssrn.6761438) ·

This repository contains the complete, fully reproducible research pipeline — from raw Binance trade data to trained models, publication figures, lead-time analysis, HMM robustness validation, external crisis validation, persistence baselines, and a live inference system. Every number in the paper traces back to a script in this repository.

---

## What This Repository Does

Cryptocurrency markets can collapse within minutes. Traditional risk systems are too slow. This project builds an early warning system for liquidity stress in Binance spot markets using only publicly available high-frequency trade data.

> **Important:** This system detects deteriorating **liquidity conditions** — not price direction. Microstructure stress reflects elevated sell pressure, thin market depth, and abnormal trade flow. Price impact is not guaranteed: liquid markets may absorb stress without significant price movement. For acute structural events, historical lead times against external reference definitions are 56 min (FTX) and 108 min (Terra-Luna).

**The core findings:**
- XGBoost achieves binary weighted F1 of **0.9706** on the final out-of-sample window (18.8M training rows), with near-zero seed variance (range: 0.0006)
- Against externally documented crisis triggers: **56 minutes** before the FTX event and **108 minutes** before the Terra-Luna structural break — measured against publicly observable events, not HMM-defined state transitions
- A persistence baseline (predict today's label = yesterday's) achieves stress-class F1 of 0.517–0.746 across folds; XGBoost lifts this by +0.218 to +0.390, confirming genuine predictive skill beyond regime momentum
- A cross-evaluation falsification test confirms XGBoost is not recovering HMM label geometry: stress F1 collapses from 0.988 to 0.480 when a locally-trained model is evaluated against global labels
- Global vs local HMM controlled experiment: local HMMs trained on narrow windows produce stress percentages of 6.0% to 94.3% for identical assets and periods — systematic miscalibration
- Three-tier external validation using genuinely independent validators (price drawdown, documented crisis timestamps, cross-asset simultaneous stress) confirms the HMM labels are consistent with genuine economic stress
- Global HMM is near-deterministic: ETH and SOL show **0.00e+00** parameter difference between independent training runs; BTC shows **2.66e-07** — floating point noise
- A live inference system (`scripts/12_inference.py`) runs against the Binance public API with no authentication required

---

## Repository Structure

```
maic/
├── config.py                              # All pipeline constants
├── requirements.txt                       # Python dependencies
├── setup.sh                               # One-command environment setup (GPU-adaptive)
├── cpu_pipeline.sh                        # Stages 01–05b (CPU)
├── gpu_pipeline.sh                        # Stages 06a–06d (GPU required)
├── cpu_post_gpu.sh                        # Stages 07a–13c + inference prompt (CPU)
├── status.sh                              # Pipeline status dashboard
├── .env.example                           # Environment variable template
├── scripts/
│   ├── 01_download.py                     # Download raw trade data from Binance
│   ├── 02_csv_to_parquet.py               # Convert CSV archives to Parquet
│   ├── 03_quality_audit.py                # Three-layer data quality audit
│   ├── 04a_feature_engineering.py         # Compute microstructure features
│   ├── 04b_stationarity_fracdiff.py       # ADF testing + fractional differencing
│   ├── 05a_label_generation.py            # Global HMM regime labelling (Viterbi)
│   ├── 05b_verify_features.py             # Feature dataset verification
│   ├── 06a_train_models.py                # Baseline training
│   ├── 06b_train_models.py                # Extended training
│   ├── 06c_train_ablation.py              # Ablation study
│   ├── 06d_train_production.py            # Production run (5 seeds × 4 folds)
│   ├── 07a_aggregate_results.py           # Aggregate baseline results
│   ├── 07b_aggregate_ablation.py          # Aggregate ablation results
│   ├── 07c_aggregate_production.py        # Aggregate production results
│   ├── 08_generate_paper_figures.py       # Generate all publication figures
│   ├── 09_lead_time_analysis.py           # Lead-time analysis (HMM onset)
│   ├── 10_hmm_robustness_check.py         # HMM robustness check
│   ├── 11a_local_global_hmm.py            # Global HMM stability + local HMM experiment
│   ├── 11b_crisis_validation_full.py      # Three-tier external crisis validation
│   ├── 13a_persistence_baseline.py        # Persistence + majority-class baselines
│   ├── 13b_lead_time_external.py          # Lead-time against external reference defs
│   ├── 13c_block_bootstrap_ztest.py       # Block bootstrap robustness check (Tier 1 z-tests)
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

**Persistence baseline vs XGBoost (binary, stress-class F1):**

| Fold | Stress% | Persistence | XGBoost | Lift |
|------|---------|------------|---------|------|
| 1 | 31.2% | 0.566 | 0.793 | +0.227 |
| 2 | 9.9% | 0.539 | 0.896 | +0.357 |
| 3 | 11.0% | 0.517 | 0.907 | +0.390 |
| 4 | 41.5% | 0.746 | 0.964 | +0.218 |

**Lead-time analysis (XGBoost binary, seed 42, pooled):**

| Event | To HMM Onset | To External Reference |
|-------|--------------|-----------------------|
| May 2021 Crash | persistent signal† | persistent signal† |
| Terra-Luna Collapse | 108.3 min | 108.3 min |
| FTX Bankruptcy | 176.0 min | 56.0 min‡ |

†Signal present throughout 12-hour search window; true onset predates lookback. Consistent with the sustained deterioration character of this event.
‡External reference: CZ tweet / FTT sell order at 12:00 UTC. HMM onset at 14:00 UTC.

**External validation (Script 11b, cross-asset kappa):**

| Fold | Asset | Global HMM κ | Local HMM κ |
|------|-------|-------------|-------------|
| 1 | ETHUSDT | 0.829 (almost perfect) | 0.334 (fair) |
| 2 | ETHUSDT | 0.762 (substantial) | 0.052 (slight) |
| 2 | SOLUSDT | 0.778 (substantial) | 0.021 (slight) |
| 3 | BTCUSDT | 0.775 (substantial) | 0.088 (slight) |
| 4 | ETHUSDT | 0.840 (almost perfect) | 0.621 (substantial) |

**Cross-evaluation falsification test (Script 10):**

| Evaluation | F1-W | F1-Stress |
|------------|------|-----------|
| Local trained, local tested | 0.9917 | 0.9881 |
| Local trained, global tested | 0.8066 | 0.4802 |
| Global production | 0.9801 | 0.9065 |

The collapse from 0.988 to 0.480 on stress F1 when cross-evaluating confirms the two label sets encode genuinely different information. XGBoost is not recovering HMM label geometry.

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
- Fetches all aggregated trades in the last 300 seconds via Binance public API with full pagination
- Computes all microstructure features identically to the training pipeline
- Applies fractional differencing to price in place, using a relaxed
  threshold (1e-3) suited to the ~80 bars of live history available —
  training uses 1e-5 against years of history, so the threshold differs
  by design (see code comments in `apply_fracdiff_to_price`)
- Runs the production XGBoost model (Fold 4, Seed 42, pooled)
- Applies the continuity condition: WARNING when P(stress) > 0.85 for ≥ 2 consecutive bars
- Logs every inference to `logs/inference/inference_log.csv`
- Tracks outcomes 30 minutes after each WARNING to `logs/inference/outcome_log.csv`

**Only the pooled model is production-grade.** `06d_train_production.py`
runs pooled models across all 5 seeds by design; asset-specific training
was scoped to `06b`'s single-seed exploratory run
(`v2/results_run1/{asset}/`), not the production sweep. `--asset-specific`
exists as a flag but is currently disabled and will error with an
explanation if used — use the pooled model (default) instead.

**Fixed:** live inference previously diverged from training on four features
due to a training/inference feature-construction mismatch: RV was missing a
square root, intensity was 10x too large, ILLIQ used net displacement
instead of path variation, and Kyle's lambda was computed at the trade level
instead of training's bar-level rolling correlation (which can even flip
sign). All four now match training's exact formulas -- see
`build_10s_bars`, `window_features`, and inline comments in
`apply_fracdiff_to_price` / `compute_features_from_trades` for the specifics.

**Outcome classifications:**
- `STRESS_CONFIRMED` — price dropped ≥ 0.3% and stayed down
- `STRESS_ABSORBED` — price dropped ≥ 0.3% but recovered
- `FALSE_POSITIVE` — price never moved meaningfully
- `PRICE_PUMPED` — price rose instead

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
bash cpu_post_gpu.sh           # stages 07a–13c: analysis + validation + inference
```

For detailed setup including GCP provisioning, BigQuery external tables, and RunPod configuration, see [`docs/walkthrough.md`](docs/walkthrough.md).

---

---

## Artifacts

The trained model and all result tables/labels/logs are public and don't
require GCP credentials:

- **Model**: [huggingface.co/Goooddy/maic-models](https://huggingface.co/Goooddy/maic-models)
  — production pooled XGBoost (Fold 4, Seed 42, F1-W 0.9706) · DOI: [10.57967/hf/9684](https://doi.org/10.57967/hf/9684)
- **Results, labels, logs**: [huggingface.co/datasets/Goooddy/maic-results](https://huggingface.co/datasets/Goooddy/maic-results)
  — every result CSV from stages 07a-13c, global HMM labels, inference/outcome logs · DOI: [10.57967/hf/9683](https://doi.org/10.57967/hf/9683)

To sync fresh artifacts after a training run: `scripts/sync_models_to_hf.py`
and `scripts/sync_results_to_hf.py`. Both are idempotent and safe to re-run.

---

## GCS Path Reference

| Path | Contents |
|------|----------|
| `v2/features/` | Raw microstructure features |
| `v2/features_fracdiff/` | Fractionally differenced features (model input) |
| `v2/labels/` | Global HMM labels (0=calm, 1=elevated, 2=stress) |
| `v2/labels_local/fold_{n}/` | Local HMM labels per fold |
| `v2/results_production/` | Production run outputs, pooled only (5 seeds × 4 folds) |
| `v2/results_run1/` | Single-seed exploratory run, incl. asset-specific models (not production-grade) |
| `v2/results/` | Aggregated result CSVs |
| `v2/paper_figures/` | Publication figures |
| `v2/pipeline_markers/` | Script completion markers |

---

## Infrastructure Notes

Built and validated for under £50 of cloud compute:

| Stage | Hardware | Estimated Cost |
|-------|----------|---------------|
| Data prep (01–05b) | CPU | < £2 |
| GPU training (06a–06d) | RTX PRO 4500 Blackwell 34GB | ~ £30 |
| Analysis (07a–13c) | CPU | < £2 |
| GCS storage | — | ~ £10/month |

---

## Citation

```bibtex
@misc{kalu2026maic,
  title   = {An Early Warning System for Liquidity Stress in Cryptocurrency
             Markets Using Trade Flow Analysis and Machine Learning},
  author  = {Kalu, Goodness and Ejike, Uchenna and Edet, Joseph and
             Fagbuyi, {Temitope E.} and Kunde, Godfrey and Igboke, Hannah},
  year    = {2026},
  note    = {SSRN Preprint},
  doi     = {10.2139/ssrn.6761438},
  url     = {https://github.com/Goodie-Goody/maic}
}

@misc{kalu2026maicmodels,
  title   = {MAIC: Liquidity Stress Detection Model (Pooled XGBoost)},
  author  = {Kalu, Goodness},
  year    = {2026},
  doi     = {10.57967/hf/9684},
  url     = {https://huggingface.co/Goooddy/maic-models}
}

@misc{kalu2026maicresults,
  title   = {MAIC: Result Tables, HMM Labels, and Inference Logs},
  author  = {Kalu, Goodness},
  year    = {2026},
  doi     = {10.57967/hf/9683},
  url     = {https://huggingface.co/datasets/Goooddy/maic-results}
}
```

---

## Acknowledgements

WorldQuant University for educational support. The open-source community behind the scientific Python ecosystem. GPU computing provided via RunPod cloud services on an NVIDIA RTX PRO 4500 Blackwell GPU. Portions of this manuscript were prepared with the assistance of Claude (Anthropic) for editorial refinement and structural review.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.