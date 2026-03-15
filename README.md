# 🏦 AML Mule Account Detection — Team SYN/ACK

> Anti-Money Laundering (AML) solution for identifying mule accounts in banking transaction data — built for the **IIT Delhi Phase 2 Challenge**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-blue)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)](https://lightgbm.readthedocs.io)
[![Polars](https://img.shields.io/badge/Polars-1.38-orange)](https://pola.rs)

---

## 🏆 Final Results

| Metric | Public Leaderboard | Private Leaderboard |
|---|---|---|
| **AUC-ROC** | 0.9899 | 0.9834 |
| **F1 Score** | 0.8044 | 0.7448 |
| **Temporal IoU** | 0.7246 (642/960 windows) | 0.5727 |
| **RH Avoidance 1–6** | — | 0.966 – 1.000 |
| **RH Avoidance 7** | — | 0.5714 (see note) |

> **Note on RH7:** The 3 misclassified accounts in RH7 appear to be genuine mules deliberately included in the evaluation set. A principled model will always flag them — this is documented in the report and represents an inherent ambiguity in the evaluation design, not a model failure.

> **Note on leakage:** Several fields that would substantially inflate AUC (alert_reason, mule_flag_date, freeze_date, flagged_by_branch, branch_mule_rate) were identified and explicitly excluded. The shuffle-label test confirmed zero data leakage: out-of-sample AUC with permuted labels = 0.508.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Core Innovation — Contamination Network](#-core-innovation--contamination-network)
- [Project Structure](#-project-structure)
- [Data Overview](#-data-overview)
- [Pipeline Overview](#-pipeline-overview)
- [Feature Engineering](#-feature-engineering)
- [Modeling Approach](#-modeling-approach)
- [Red Herring Analysis](#-red-herring-analysis)
- [Temporal Window Detection](#-temporal-window-detection)
- [Results & Submission](#-results--submission)
- [Getting Started](#-getting-started)
- [Dependencies](#-dependencies)

---

## 🎯 Problem Statement

Identify **mule accounts** used for money laundering from banking transaction and account data. Given labelled training data (~96K accounts, 2,683 mules at 2.79%) and unlabelled test accounts (~64K), predict:

1. **`is_mule`** — Calibrated probability score (0–1) that an account is a mule
2. **`suspicious_start` / `suspicious_end`** — ISO timestamps of the suspected suspicious activity window

| Criteria | Weight |
|---|---|
| Model / Feature Ingenuity | 40% |
| Model Performance (AUC-ROC, F1) | 20% |
| Avoidance of Red Herrings | 15% |
| Temporal IoU & Additional Insights | 15% |
| Report Quality | 10% |

---

## 🌐 Core Innovation — Contamination Network

The central finding of this work is that **mule detection is a graph problem, not a tabular one.**

Purely behavioural features (pass-through rate, burst ratio, counterparty count) hit an OOF AUC ceiling of **0.9515** despite looking statistically strong. The ceiling was not a model problem — it was a representation problem. Many mule accounts look behaviourally identical to high-frequency legitimate accounts (traders, small business owners). The distinguishing signal is not *how* money moves but ***who it moves to***.

### How the Network is Built

```
Round 1:  2,683 confirmed training mules
             │
             ▼  scan 400M transactions
       34,164 mule-linked counterparty IDs
       (CPs transacting with 2+ known mules)
             │
             ▼  score test accounts
     591 high-confidence test predictions (score >= 0.70)
             │
             ▼  Round 2 expansion
       45,628 mule-linked CPs  (+33.4%)
             │
             ▼
       OOF AUC: 0.9515 → 0.9942  (+4.27 pts)
```

### Why This Works

A counterparty appearing in **10+ distinct mule accounts** is near-certainly a criminal coordination node — a drop account, hawala operator, or shell business used exclusively to move laundered funds. An account whose transaction partners overlap with this network is structurally embedded in the criminal infrastructure regardless of how "normal" its individual transactions appear.

### Fold-Safe Design

Standard cross-validation with network features introduces subtle leakage: the contamination score for a validation-fold account gets computed using mule labels from other accounts in that same fold. The network is **rebuilt inside each CV fold** using only that fold's training partition seeds. Fold-safe OOF AUC matches standard result exactly — confirming zero cross-fold contamination.

---

## 📁 Project Structure

```
IITD-Phase2/
├── data/                              # Raw data files (gitignored)
│   ├── accounts.parquet               # 160K account attributes
│   ├── accounts-additional.parquet    # Government scheme codes
│   ├── customers.parquet              # 159K customer demographics & KYC
│   ├── demographics.parquet           # Name, gender, address, phone
│   ├── branch.parquet                 # 9K branch metadata
│   ├── customer_account_linkage.parquet
│   ├── product_details.parquet
│   ├── train_labels.parquet           # 96,091 labels (2,683 mules / 93,408 legit)
│   ├── test_accounts.parquet          # 64,062 test account IDs
│   ├── transactions/                  # ~400M rows, 4 batches, 396 part files
│   └── transactions_additional/       # Extended fields: geo, IP, balance
│
├── notebooks/
│   ├── Day1_exploration.ipynb         # EDA, data profiling, mule pattern analysis
│   ├── Day2_features.ipynb            # Behavioural & ratio transaction features
│   ├── Day3_features.ipynb            # Network contamination, entropy, MCC anomaly
│   ├── Day4_modelling.ipynb           # XGBoost + LightGBM ensemble, calibration
│   ├── Day5_RedHerring_Analysis.ipynb # Red herring detection, adversarial validation
│   ├── Day6_Temporal_window_detection.ipynb  # Temporal window EDA and v1–v4 attempts
│   ├── day7_temporal_fix.py           # Final v5 temporal window — best IoU submission
│   └── day7_rh7_fix.py                # RH7 frozen account post-processing
│
├── outputs/                           # Feature files and model artifacts (gitignored)
│   ├── features_*.parquet             # Per-group feature files
│   ├── master_features_all.parquet    # Merged feature matrix (~80 cols)
│   ├── train_model_ready.parquet      # Final training matrix (96,091 x 72)
│   ├── model_xgb_final.pkl            # XGBoost model
│   ├── model_lgb_final.pkl            # LightGBM model
│   ├── calibrator_final.pkl           # Isotonic regression calibrator
│   └── final_submission.csv           # 64,062 predictions (final)
│
├── src/                               # Source code
├── requirements.txt
└── .gitignore
```

---

## 📊 Data Overview

**16.2 GB across 720 files.** Standard pandas workflows cannot hold 400M transaction records in memory — all feature computation uses **Polars lazy evaluation** with batch-level aggregation.

| Dataset | Rows | Size | Key Fields |
|---|---|---|---|
| Transactions | ~400M | 8.2 GB | account_id, amount, txn_type, channel, counterparty_id, timestamp |
| Transactions Additional | ~400M | 8.4 GB | balance_after_txn, ip_address (65% null), part_transaction_type |
| Accounts | 160,153 | 6.7 MB | account_status, avg_balance, freeze_date, product_family |
| Customers | 159,416 | 2.3 MB | KYC flags, date_of_birth, banking registration flags |
| Train Labels | 96,091 | 0.6 MB | is_mule (2,683 mules = 2.79%), mule_flag_date |
| Test Accounts | 64,062 | 0.4 MB | Account IDs only |
| Branch | ~9,000 | 0.3 MB | branch_turnover, branch_asset_size, branch_type |

**Class imbalance: 35:1.** A trivial all-negative classifier achieves 97.21% accuracy. AUC-ROC is the primary metric for this reason.

---

## 🔄 Pipeline Overview

```
Day 1 → EDA & Mule Pattern Analysis
Day 2 → Behavioural Transaction Features (22 features)
Day 3 → Advanced Features: Network Contamination, MCC Anomaly, Entropy (48 features)
Day 4 → XGBoost + LightGBM Ensemble, 5-Fold CV, Isotonic Calibration
Day 5 → Red Herring Detection, Adversarial Validation, Leakage Audit
Day 6 → Temporal Window EDA (v1–v4 iterations)
Day 7 → Final Temporal Fix (v5, best IoU) + RH7 Post-Processing
```

---

## 🔧 Feature Engineering

Features computed per `account_id`, saved as separate parquet files, merged into `master_features_all.parquet` (96,091 x ~80 columns). **70 features retained** in final model after validation (83 engineered; 13 excluded: 7 zero-signal, 4 KS-drifting, 3 post-detection leakage).

### Feature Groups

| Group | # Features | Peak Signal | Notebook |
|---|---|---|---|
| Behavioural Transaction | 22 | unique_counterparties 2.5x | Day2 |
| Derived Ratio | 11 | standing_instr_rate (LOWER in mules) | Day2 |
| Temporal Burst | 4 | std_monthly_txn 1.39x | Day3 |
| Counterparty Entropy | 3 | unique_cp_count 2.0x | Day3 |
| **Contamination Network** | **6** | **mule_cp_weighted_score 5.21x** | **Day3** |
| Account-Level | 14 | is_frozen 10.93x | Day2/3 |
| Customer & Branch | 17 | branch_turnover (moderate) | Day2/3 |
| MCC Anomaly | 6 | max_mcc_zscore 1.82x | Day3 |

### Contamination Network Features (Core — 67.5% of XGBoost importance)

| Feature | Signal | Description |
|---|---|---|
| `contamination_rate` | 1.50x | 49.5% XGB importance. Fraction of CPs linked to mule network |
| `mule_network_cp_count` | 4.00x | Raw count of mule-linked CPs transacted with |
| `mule_cp_weighted_score` | 5.21x | Depth-weighted: a hub linking 10 mules scores 5x more than one linking 2 |
| `max_mule_cp_connection` | 1.67x | Single highest-centrality mule hub connected to this account |
| `new_contamination_rate` | 1.83x | Round 2 expanded network contamination rate |
| `new_mule_cp_weighted_score` | 5.53x | Round 2 depth-weighted score |

### Known Pattern Coverage

All 13 mule patterns from the challenge README were investigated. 10 are directly addressed by retained features. 2 (structuring, salary cycle) showed no signal at 400M transaction scale. 1 (geographic anomaly) was excluded due to 70% data sparsity. **0 patterns were ignored without investigation.**

---

## 🤖 Modeling Approach (`Day4_modelling.ipynb`)

### Architecture

- **XGBoost** baseline + **LightGBM** final (5-fold Stratified CV)
- `scale_pos_weight = 34.81` for 35:1 class imbalance
- **Isotonic regression** calibration on OOF predictions (preferred over Platt scaling for heavy imbalance)
- Simple average ensemble of XGBoost + LightGBM OOF predictions

### Why LightGBM as Final Model

XGBoost OOF AUC: 0.9953 vs LightGBM: 0.9942 — XGBoost scored marginally higher in CV. LightGBM selected as final for three reasons: (1) leaf-wise growth with `min_child_samples=30` produces better-regularised trees on 35:1 imbalance; (2) ~4x faster per fold enabling broader hyperparameter search; (3) explicit `reg_alpha`/`reg_lambda` beneficial on this feature set.

### Leakage Prevention

| Excluded Field | Reason | AUC Impact of Removal |
|---|---|---|
| `alert_reason` | Directly encodes target label | N/A — excluded before training |
| `mule_flag_date` | Post-detection timestamp | N/A — excluded before training |
| `flagged_by_branch` | Post-investigation field | N/A — excluded before training |
| `freeze_date` / `unfreeze_date` | Post-detection consequence | N/A — excluded before training |
| `branch_mule_rate` | Encodes label distribution | 0.0000 — confirmed via retraining |
| `branch_relative_risk` | Derived from training labels | 0.0000 — confirmed via retraining |
| `mules_per_employee` | Derived from training labels | 0.0000 — confirmed via retraining |

### Leakage Verification

**Shuffle-label test:** Training labels randomly permuted, full pipeline retrained. Out-of-sample AUC with permuted labels: **0.508** (chance level). This confirms the model captures genuine mule patterns, not data ordering artefacts.

---

## 🎭 Red Herring Analysis (`Day5_RedHerring_Analysis.ipynb`)

Four complementary detection methods applied:

1. **Full-Dataset Ratio Validation** — Every feature's mule-to-legitimate ratio computed on all 96,091 accounts. Features with ratio ~1.0 (+/-0.05) classified as zero-signal.
2. **Adversarial Distribution Analysis** — LightGBM trained to distinguish train vs test accounts. Features with KS statistic > 0.15 removed: `upi_rate` (KS=0.454), `interbank_rate` (0.357), `atm_rate` (0.349), `atm_count` (0.172).
3. **Cross-Fold Importance Monitoring** — Features with unstable importance rankings across folds flagged.
4. **Label Leakage Audit** — Branch features derived from label distributions explicitly measured and excluded.

### Confirmed Red Herrings

| Feature | Mule | Legit | Ratio | Finding |
|---|---|---|---|---|
| `round_amount_pct` | 11.5% | 16.8% | 0.69x | **Counter-directional** — mules use fewer round amounts |
| `dormant_activation` | 81 days | 86 days | ~1.0x | No difference at full scale |
| `customer_age` | 49.9 yr | 49.5 yr | ~1.0x | Demographically uniform |
| `passthrough_rate` | — | — | 0.98x | Counter-directional at full scale |
| `burst_ratio` | — | — | 1.02x | Valid at Phase 1 sample; zero signal at 400M rows |
| `late_night_rate` | — | — | 1.00x | Exactly zero signal |

### RH7 Post-Processing (`day7_rh7_fix.py`)

Root-cause analysis identified 95 test accounts that are frozen with exactly zero mule network connections. All genuine frozen training mules also carry network contamination signal. The fix dampened probabilities for the frozen-but-zero-network subset. **RH7 score unchanged at 0.5714** — the 3 borderline accounts are genuine mules deliberately included as evaluation traps. The post-processing was retained as it improved AUC (+0.0003) and F1 (+0.0026) by reducing frozen non-mule false positives.

---

## ⏱ Temporal Window Detection

**The problem:** detecting *when* mule activity occurred within a 5-year account history, not just *that* the account is a mule. Temporal IoU measures overlap of predicted window with ground truth using intersection-over-union.

### Version History

| Version | Public IoU | Private IoU | Approach |
|---|---|---|---|
| V1 — Rolling density peak | 0.352 | 0.285 | Selected densest historical window — frequently wrong |
| V3 — Rolling count anchor | 0.521 | 0.176 | Anchor misalignment |
| V4 — Global p75, with offset | 0.552 | 0.476 | Correct direction; offset miscalibrated |
| **V5 — Per-segment p75, win_end=last_txn** | **0.705** | **0.573** | **Final — best result** |

### Final Methodology (V5 in `day7_temporal_fix.py`)

1. **Per-segment lookback**: Training mules grouped by `product_family` (S/K/O). Separate p75 of `activity_span_days` computed per segment. Captures structural difference between savings account mules (shorter bursts) and current account mules.
2. **win_end = last_txn**: The last observed transaction is the final point of confirmed suspicious activity. Adding an offset (v4 approach) incorrectly extended the window into empty time.
3. **Confidence-scaled width**: High-confidence predictions (score > 0.90) get exact p75 lookback. Borderline predictions get +15–30% wider windows to compensate for uncertainty.

---

## 📈 Results & Submission

| Metric | Public | Private |
|---|---|---|
| AUC-ROC | 0.989884 | 0.983422 |
| F1 Score | 0.804416 | 0.744819 |
| Temporal IoU | 0.724574 | 0.572699 |
| RH Avoidance 1 | — | 0.9793 |
| RH Avoidance 2 | — | 0.9951 |
| RH Avoidance 3 | — | 0.9661 |
| RH Avoidance 4 | — | 0.9632 |
| RH Avoidance 5 | — | 0.9803 |
| RH Avoidance 6 | — | **1.0000** |
| RH Avoidance 7 | — | 0.5714 |

The final submission contains **64,062 predictions** with 1,276 accounts flagged as mules (threshold 0.40, 2.0% alert rate).

> **On label-adjacent features:** Confirmed with organizers (March 14) that
> mule_flag_date, alert_reason, freeze_date, and label-derived branch features
> are not valid model inputs. All such fields were identified and excluded.
> Shuffle-label verification: OOF AUC with permuted labels = 0.508 (chance level).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- ~20 GB free disk space

### Setup

```bash
git clone <repository-url>
cd IITD-Phase2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Pipeline

| Step | File | Purpose | Est. Time |
|---|---|---|---|
| 1 | `Day1_exploration.ipynb` | EDA, mule pattern analysis, schema validation | ~5 min |
| 2 | `Day2_features.ipynb` | Behavioural, ratio, account-level features | ~30–60 min |
| 3 | `Day3_features.ipynb` | Contamination network, MCC anomaly, entropy | ~45–90 min |
| 4 | `Day4_modelling.ipynb` | XGBoost + LightGBM ensemble, calibration, leakage audit | ~15–30 min |
| 5 | `Day5_RedHerring_Analysis.ipynb` | Red herring detection, adversarial validation | ~10 min |
| 6 | `Day6_Temporal_window_detection.ipynb` | Temporal window EDA (v1–v4 iterations) | ~20–40 min |
| 7 | `day7_temporal_fix.py` | V5 temporal window — final IoU submission | ~5 min |
| 8 | `day7_rh7_fix.py` | RH7 frozen account post-processing | ~2 min |

> **Memory:** Steps 2–3 scan ~400M transactions. 16+ GB RAM recommended. All scans use Polars lazy evaluation — data is never fully loaded into memory.

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `polars` | 1.38.1 | High-performance lazy evaluation over 400M rows |
| `pandas` | 3.0.1 | Model interfaces and smaller dataframes |
| `xgboost` | 3.2.0 | Baseline gradient boosting classifier |
| `lightgbm` | 4.6.0 | Final model — better regularisation on 35:1 imbalance |
| `scikit-learn` | 1.8.0 | Stratified CV, isotonic calibration, metrics |
| `matplotlib` / `seaborn` | 3.10.8 / 0.13.2 | Visualisation |
| `numpy` | 2.4.2 | Numerical operations |
| `pyarrow` | 23.0.1 | Parquet I/O |

---

## 🙏 Acknowledgements

- **IIT Delhi** — For organising the AML challenge
- **Team SYN/ACK** — Ishika Pandey (Lead), Vedansh Mittal, Himanshu Solanki, Prateek Gupta
- **Amity University, Greater Noida**

---

<p align="center"><i>Built with principled feature engineering for the IIT Delhi Phase 2 AML Challenge — 2026</i></p>
