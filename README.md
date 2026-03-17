# Cryptocurrency Confidence‑Based Execution System

A production‑ready ML system for cryptocurrency price‑movement prediction with **confidence‑based execution**. It combines high‑frequency order‑book microstructure with macro OHLCV features, optimizes **profit‑oriented thresholds (τ)**, and now includes **federated learning** and **privacy‑preserving secure aggregation (Shamir + DP)**.

---

## 🎯 Overview

This system implements a **two‑class trading approach** that:

- **Predicts direction** (Up vs Down) with neural models.
- **Executes only when confidence ≥ τ**, optimizing τ for **profit**, **expected value (profit × coverage)**, or **profit with minimum coverage**.
- **Accounts for trading costs** in all threshold searches.
- **Provides centralized, federated, and privacy‑preserving pipelines**, plus robust analysis & visualization utilities.

### What’s new (2025‑10)

- **Federated training runner** (`run_federated.py`) with **FedAvg / Delta‑FedAvg / FedAvgM** and **robust aggregators** (coordinate‑median, trimmed‑mean).
- **Privacy‑preserving FL** (`run_privacy_federated.py`) integrating **Shamir Secret Sharing** + **Differential Privacy** (RDP‑inspired composition) via `enhanced_shamir_privacy.py`.
- **Universal model comparator** (`compare_multiple_models.py`) — align predictions from N models and sweep τ **without retraining**.
- **τ‑curve plotting** (`plot_tau_curves.py`) and **sweep aggregator** (`sweep_results_aggregator.py`).
- **Federated preprocessing** (`federated/preprocessing.py`) that strictly aligns to centralized artifacts to prevent feature‑mismatch.
- **Cleaner logs & configs**, symbol‑aware temporal splits, two‑class selective execution metrics, and CLI “cheat‑sheet”.

---

## 🏗️ Repository Structure

```
crypto-confidence-execution/
├── data/
│   ├── macro/                         # Daily OHLCV
│   ├── micro/                         # Order-book snapshots
│   └── processed/                     # Unified datasets
├── data_preprocessing.py              # Build unified_dataset.parquet
├── centralized_training_two_classes.py# Centralized training (two-class, τ-optimized)
├── sweep_two_class.py                 # Grid sweeps (horizon, deadband, τ criteria)
├── sweep_results_aggregator.py        # Aggregate + visualize sweeps
├── plot_tau_curves.py                 # Profit/coverage curves vs τ (no retraining)
├── compare_multiple_models.py         # Align & compare N models across τ
├── experiments/
│   ├── centralized/                   # Centralized runs & sweeps
│   ├── federated/                     # Federated runs
│   └── privacy/                       # Privacy-preserving federated runs
├── federated/
│   ├── __init__.py
│   ├── core.py                        # Trainer + loop
│   ├── client.py                      # Client logic (balanced batches, best-val model)
│   ├── aggregation.py                 # FedAvg, Delta-FedAvg, FedAvgM, robust aggregators
│   ├── preprocessing.py               # Load centralized artifacts, strict feature alignment
│   └── utils.py                       # Data manager, shapes/cleaning, config helpers
├── run_federated.py                   # Simple federated runner
├── run_privacy_federated.py           # Federated + Shamir + DP
├── utils/
│   └── enhanced_shamir_privacy.py     # Shamir, secure aggregation, DP config
└── README.md
```

---

## 🚀 Quick Start

### 1) Data Preparation

- **Macro data**: daily OHLCV (CSV)
- **Micro data**: order‑book snapshots (CSV)

Place as:
```
data/macro/top_100_cryptos_with_correct_network.csv
data/micro/*.csv
```

Build the unified dataset:
```bash
python data_preprocessing.py
# → data/processed/unified_dataset.parquet
```

### 2) Centralized Training (Two‑Class)

Single run:
```bash
python centralized_training_two_classes.py \
  --unified data/processed/unified_dataset.parquet \
  --out experiments/centralized \
  --task classification \
  --model mlp \
  --horizon_min 600 --deadband_bps 10 \
  --two_class_mode \
  --use_robust_scaler \
  --calibrate_probabilities \
  --confidence_tau 0.80 \
  --profit_cost_bps 1.0 \
  --top_k_features 128 \
  --batch_size 1024 --epochs 20 --seed 42
```

Parameter sweep (τ optimization **without** retraining the model each time):
```bash
python sweep_two_class.py \
  --unified data/processed/unified_dataset.parquet \
  --out experiments/centralized \
  --horizons 300 600 900 \
  --deadbands 5 10 15 \
  --top_k_features 64 \
  --profit_cost_bps 1.0 \
  --grid 0.50:0.98:0.01 \
  --optimize_tau_by profit ev profit_with_min_coverage \
  --min_coverage 0.005 \
  --use_robust_scaler --calibrate
```

Analyze sweeps:
```bash
python sweep_results_aggregator.py --base_dir experiments/centralized
```

Plot τ‑curves for a single predictions CSV (no retraining):
```bash
python plot_tau_curves.py \
  --predictions experiments/centralized/.../test_predictions_two_class.csv \
  --cost_bps 1.0 --out plots/tau_curves
```

### 3) Federated Learning

Use centralized artifacts (imputer, scaler, feature schema, optional `decision_threshold.json`) for strict feature alignment.

Basic run:
```bash
python run_federated.py \
  --federated_data data/processed_federated_iid \
  --centralized_artifacts experiments/centralized/<BEST_RUN>/ \
  --output experiments/federated \
  --aggregation delta_fedavg \
  --rounds 30 --local_epochs 5 \
  --server_lr 0.5 --client_lr 0.01 \
  --batch_size 1024 \
  --horizon 600 --deadband 10.0 \
  --two_class --confidence_tau 0.80
```

Available aggregations:
- `fedavg`, `delta_fedavg`, `fedavgm` (+ `--server_momentum`), `coordinate_median`, `trimmed_mean`.

### 4) Privacy‑Preserving Federated Learning

Enable **Shamir secret sharing** and **differential privacy** atop the federated runner:

```bash
python run_privacy_federated.py \
  --federated_data data/processed_federated_iid \
  --centralized_artifacts experiments/centralized/<BEST_RUN>/ \
  --output experiments/privacy \
  --aggregation delta_fedavg \
  --rounds 15 --local_epochs 3 \
  --server_lr 0.5 --client_lr 0.01 \
  --batch_size 1024 \
  --horizon 600 --deadband 10.0 \
  --two_class --confidence_tau 0.80 \
  --privacy --shamir --dp \
  --threshold 3 --prime_bits 61 \
  --epsilon 1.0 --delta 1e-6 --clip_norm 1.0 --per_layer_dp
```

- `utils/enhanced_shamir_privacy.py` implements: vectorized encoding/decoding, centered modular arithmetic, seed derivation for secure aggregation, dropout recovery via seed shares, and an RDP‑inspired per‑round ε schedule.
- `test_privacy_integration.py` provides quick checks for **Shamir correctness**, **DP noise**, **mask cancellation**, and **dropout recovery**.

---

## 📊 Metrics & Selective Execution

On executed trades (confidence ≥ τ), we report:
- **Coverage** (% of samples executed)
- **Direction Accuracy**
- **Average/Median Profit (bps)** after costs
- **Win Rate**
- **Sharpe** and **EV per sample** (avg_profit × coverage)

Centralized training also logs class balances, split ranges, and feature‑selection summaries. Federated runs log per‑round validation/test accuracy and aggregation stats; privacy runs add DP/Shamir diagnostics.

---

## 🔬 Comparing Models Without Retraining

Align predictions and sweep τ on **multiple** models at once:
```bash
python compare_multiple_models.py \
  --model centralized=experiments/centralized/.../test_predictions_two_class.csv \
  --model federated=experiments/federated/.../predictions_on_centralized_test.csv \
  --model privacy=experiments/privacy/.../predictions_on_centralized_test.csv \
  --cost_bps 1.0 \
  --tau_min 0.50 --tau_max 0.90 --tau_step 0.005 \
  --out_dir model_comparison
```
Generates profit‑vs‑coverage trade‑off curves, optimal τ markers, and summary tables per model.

---

## ⚙️ Key Configuration (centralized)

| Parameter              | Description                                   | Default |
|------------------------|-----------------------------------------------|---------|
| `horizon_min`          | Prediction horizon (minutes)                  | 600     |
| `deadband_bps`         | Minimum movement threshold (bps)              | 10      |
| `two_class_mode`       | Binary Up/Down training                       | true    |
| `confidence_tau`       | Execution confidence threshold                | 0.70    |
| `profit_cost_bps`      | Trading cost per trade (bps)                  | 1.0     |
| `top_k_features`       | Feature selection (MI/F‑score/variance)       | 128     |
| `use_robust_scaler`    | Robust vs standard scaling                    | true    |
| `calibrate_probabilities` | Platt/Isotonic via `CalibratedClassifierCV`| true    |
| `optimize_tau_by`      | `profit` \| `ev` \| `profit_with_min_coverage`| profit  |
| `min_coverage`         | Min coverage for constrained optimization     | 0.0     |

### Federated notes
- **Delta‑FedAvg** applies server LR and gradient clipping to client **deltas**.
- **FedAvgM** uses server momentum; robust options: **coordinate‑median**, **trimmed‑mean**.
- **Feature alignment** uses centralized artifacts to ensure identical dimension/order on every client.

---

## 📈 Example Results (illustrative)

```
Coverage: 15.2%  | Direction Acc: 0.682
Avg Profit: 3.47 bps (after 1.0 bps cost)
Win Rate: 58.3%  | Sharpe: 1.24
```
Use the **sweep aggregator** + **τ‑curves** to find the desired profit/coverage regime (e.g., τ=0.60 vs τ=0.80).

---

## 🔁 Reproducibility & Seeds

- Deterministic seeds for data splits and batching.
- Symbol‑aware temporal splits to prevent leakage.
- Training size capping uses **proportional random sampling** per symbol.

---

## 🧩 Troubleshooting (quick)

- **Feature mismatch** in federated runs → ensure the federated preprocessor loads the **same** centralized `imputer.joblib`, `scaler.joblib`, and `feature_schema.json`.
- **Too few trade samples** in two‑class mode → relax `--deadband_bps` or increase dataset window.
- **Low coverage after τ optimization** → use `--optimize_tau_by profit_with_min_coverage --min_coverage 0.005`.
- **FedAvgM oscillations** → reduce `--server_momentum` or use `delta_fedavg` with smaller `--server_lr`.
- **Unicode logs on Windows** → runners set UTF‑8‑safe console/file handlers.

---

## 🧪 Dev Utilities

- **Grid search (federated)**: see `hyperparameter_search.py` for scripted exploration of rounds/epochs/batch/LRs/aggregations.
- **Privacy tests**: `test_privacy_integration.py` for Shamir/DP/secure‑aggregation sanity checks.

---

## 🗂️ Results Layout (per run)

- `config.json` / `config.yaml` — full parameters
- `training.log` / `federated_training.log` — structured logs
- `metrics_two_class.json` / `metrics_global.json` — core KPIs
- `test_predictions_two_class.csv` — per‑sample outputs
- `decision_threshold.json` — selected τ & validation stats
- `plots/` — τ‑curves and summary figures (if enabled)

---

## 📚 Datasets & Acknowledgements

This project relies on large-scale public cryptocurrency datasets combining macro-level market data and high-frequency order book microstructure.

### Datasets

- **Top 100 Cryptocurrencies Dataset (2020–2025)**  
  Daily OHLCV data across major cryptocurrencies:  
  https://www.kaggle.com/datasets/imtkaggleteam/top-100-cryptocurrency-2020-2025  

- **Cryptocurrency Order Book Data (Asks & Bids)**  
  High-frequency order book snapshots capturing market microstructure:  
  https://www.kaggle.com/datasets/ilyazawilsiv/cryptocurrency-order-book-data-asks-and-bids  

Due to their size and licensing constraints, these datasets are **not included in this repository** and must be downloaded directly from the original sources.

### Acknowledgements

We gratefully acknowledge the dataset creators and the Kaggle community for providing access to high-quality financial and microstructure data, which made this research possible.

---

## 📄 Related Publications

This repository is directly connected to the following research works:

Kuznetsov, O., Adilzhanova, S., Florov, S., Bushkov, V., Peremetchyk, D.  
*Privacy-Preserving Federated Learning for Distributed Financial IoT: A Blockchain-Based Framework for Secure Cryptocurrency Market Analytics.*  
IoT, 2025.  
https://doi.org/10.3390/iot6040078  

Kuznetsov, O., Kostenko, O., Klymenko, K., Hbur, Z., Kovalskyi, R.  
*Machine Learning Analytics for Blockchain-Based Financial Markets: A Confidence-Threshold Framework for Cryptocurrency Price Direction Prediction.*  
Applied Sciences, 2025.  
https://doi.org/10.3390/app152011145  

Kuznetsov, O., Prokopovych-Tkachenko, D., Bilan, M., Khruskov, B., Cherkaskyi, O.  
*Blockchain-Native Asset Direction Prediction: A Confidence-Threshold Approach to Decentralized Financial Analytics Using Multi-Scale Feature Integration.*  
Algorithms, 2025.  
https://doi.org/10.3390/a18120758  

These works introduce the core ideas implemented in this repository, including:
- confidence-based selective execution
- multi-scale feature integration (macro + microstructure)
- blockchain-oriented financial analytics
- federated and privacy-preserving learning

If you use this repository, please consider citing the above papers.

---

## 🧭 CLI Cheat‑Sheet

```bash
# Centralized (single)
python centralized_training_two_classes.py ...

# Centralized sweeps (+ aggregation/plots later)
python sweep_two_class.py ... && python sweep_results_aggregator.py --base_dir experiments/centralized

# τ curves on an existing predictions CSV
python plot_tau_curves.py --predictions <csv> --cost_bps 1.0 --out plots/tau_curves

# Federated (no privacy)
python run_federated.py --aggregation delta_fedavg --rounds 30 --local_epochs 5 ...

# Federated with privacy (Shamir + DP)
python run_privacy_federated.py --privacy --shamir --dp --threshold 3 --epsilon 1.0 ...
```

---

**License & Citation**: If you use this codebase or results in academic/industrial work, please include a reference to the repository and acknowledge the confidence‑based selective‑execution methodology and federated/privacy modules.
