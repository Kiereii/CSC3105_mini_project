# Project Data Flow

## Overview

Two-task UWB LOS/NLOS pipeline: **Task 1** classifies each path as LOS or NLOS,
**Task 2** classifies pairs and estimates range for both dominant paths.

---

## Pipeline

```
Dataset/UWB-LOS-NLOS-Data-Set/dataset/Cleaned/
  uwb_cleaned_dataset_part*.csv
  (raw UWB measurements — RANGE, FP_AMP1-3, RXPACC, CIR0-1015, NLOS label, ...)
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/preprocessing/preprocess_data.py                           │
│                                                                 │
│  Input : cleaned CSV parts                                      │
│  Steps :                                                        │
│    1. Extract 16 core features + CIR730–849 (120 bins)          │
│    2. Engineer 12 new features via feature_engineering.py       │
│       (FP_AMP_ratio, SNR_per_acc, CIR_kurtosis, etc.)           │
│    3. Environment-based 70/15/15 train/val/test split           │
│    4. Fit StandardScaler on train, transform all splits         │
│  Output:                                                        │
│    X_train/val/test_unscaled.npy  (tree models)                 │
│    X_train/val/test_standard.npy  (linear models)               │
│    y_train/val/test.npy           (NLOS label: 0=LOS, 1=NLOS)  │
│    train_idx / val_idx / test_idx .npy  (shared split indices)  │
│    scaler_standard.pkl                                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
           ┌─────────────┴──────────────┐
           ▼                            ▼
┌────────────────────────┐   ┌──────────────────────────────────────────────┐
│  second_path_          │   │  TASK 1 — Per-Path LOS/NLOS Classification   │
│  features.py           │   │                                              │
│                        │   │  All classifiers use the same split indices  │
│  Input : cleaned CSVs  │   │  from preprocess_data.py.                    │
│          + train/test  │   │                                              │
│            indices     │   │  1. dt_exploratory.py                        │
│                        │   │     • Shallow DT (depth 4) on core features  │
│  Steps :               │   │     • Exploratory — understand decision      │
│    1. Find 2nd dominant│   │       boundaries before ensemble models      │
│       CIR peak per     │   │     • Input : X_train_unscaled (core only)   │
│       sample           │   │     • Output: dt_rules.txt, dt_tree_plot.png │
│    2. Compute PEAK2_   │   │                                              │
│       IDX/AMP/GAP/FOUND│   │  2. random_forest_classifier.py              │
│    3. Estimate Path 2  │   │     • Input : X_train_unscaled               │
│       range from peak  │   │     • Tuning: RandomizedSearchCV / Optuna    │
│       offset           │   │                                              │
│    4. Build pair label │   │  3. logreg_svm_classifier.py                 │
│       (mirrors NLOS):  │   │     • Input : X_train_standard               │
│       0 = LOS+NLOS     │   │     • LogisticRegression + LinearSVC         │
│       1 = NLOS+NLOS    │   │                                              │
│                        │   │  4. xgboost_classifier.py  ★ WINNER          │
│  Output:               │   │     • Input : X_train_unscaled               │
│    X_train/test_       │   │     • Tuning: RandomizedSearchCV / Optuna    │
│      regression.npy    │   │     • Best accuracy + F1 across all models   │
│    y_range_p1/p2_      │   │                                              │
│      train/test.npy    │   │            ▼                                 │
│    X_train/test_       │   │   compare_models.py                          │
│      pair.npy          │   │   • Consolidates metrics from all 3 models   │
│    y_train/test_       │   │   • Outputs bar charts, ROC curves,          │
│      pair.npy          │   │     confusion matrices, feature importance   │
└────────┬───────────────┘   └──────────────────────────────────────────────┘
         │
         ├──── X_train/test_pair.npy ──────────────────────────────────────┐
         │     y_train/test_pair.npy                                       ▼
         │                                          ┌──────────────────────────────────────┐
         │                                          │  TASK 2A — Pair Classification       │
         │                                          │                                      │
         │                                          │  XG_pair_classifier.py  ★ WINNER     │
         │                                          │  • Input : X_train/test_pair         │
         │                                          │  • Classifies: LOS+NLOS vs NLOS+NLOS │
         │                                          │  • Tuning: Optuna (40 trials, 3-fold)│
         │                                          │  • Output: pair_xgb_model.pkl        │
         │                                          │            confusion matrix, ROC     │
         │                                          │            feature importance        │
         │                                          │            error analysis            │
         │                                          │                                      │
         │                                          │  Path mapping rule (per brief):      │
         │                                          │    pair=0 → Path1=LOS,  Path2=NLOS   │
         │                                          │    pair=1 → Path1=NLOS, Path2=NLOS   │
         │                                          └──────────────────────────────────────┘
         │
         └──── X_train/test_regression.npy ──────────────────────────────────┐
               y_range_p1/p2_train/test.npy                                  ▼
                                                    ┌──────────────────────────────────────┐
                                                    │  TASK 2B — Range Regression          │
                                                    │                                      │
                                                    │  range_regressor.py                  │
                                                    │  • Trains RF + KNN + XGBoost         │
                                                    │  • Comparison baseline               │
                                                    │                                      │
                                                    │  xgb_range_regressor.py  ★ WINNER    │
                                                    │  • XGBoost standalone pipeline       │
                                                    │  • Predicts RANGE for Path 1 & 2     │
                                                    │  • Metrics: MAE, RMSE, R²            │
                                                    └──────────────────────────────────────┘
```

---

## Feature Sets by Script

| Script | Feature input | # Features |
|---|---|---|
| `dt_exploratory.py` | Core + second-path (no CIR) | ~20 |
| `random_forest_classifier.py` | Core + CIR + engineered | 148 |
| `logreg_svm_classifier.py` | Core + CIR + engineered (scaled) | 148 |
| `xgboost_classifier.py` | Core + CIR + engineered | 148 |
| `XG_pair_classifier.py` | Core + second-path + CIR | 140 |
| `range_regressor.py` / `xgb_range_regressor.py` | Core + second-path + CIR | 139 |

---

## Output Directory Structure

```
runs/split_env_70_15_15_seed42/
├── preprocessed_data/
│   ├── X_train/val/test_unscaled.npy
│   ├── X_train/val/test_standard.npy
│   ├── X_train/test_pair.npy
│   ├── X_train/test_regression.npy
│   ├── y_train/val/test.npy
│   ├── y_train/test_pair.npy
│   ├── y_range_p1/p2_train/test.npy
│   ├── train/val/test_idx.npy
│   ├── scaler_standard.pkl
│   ├── feature_names.txt
│   ├── pair_feature_names.txt
│   ├── regression_feature_names.txt
│   └── split_config.json
│
└── models/
    ├── dt_exploratory/
    │   ├── dt_rules.txt
    │   ├── dt_tree_plot.png
    │   └── dt_feature_importance.png
    ├── random_forest/
    ├── logreg_svm/
    ├── xgboost/          ← Task 1 winner
    ├── pair_classifier/  ← Task 2A winner (XG_pair_classifier.py)
    ├── range_regressor/  ← Task 2B winner (xgb_range_regressor.py)
    └── comparison/       ← consolidated plots + metrics CSV
```

---

## Script Execution Order

```
1. preprocess_data.py          (always first)
2. second_path_features.py     (always second — needs split indices)
3. dt_exploratory.py           (Task 1 — exploratory, before ensemble models)
4. random_forest_classifier.py (Task 1)
5. logreg_svm_classifier.py    (Task 1)
6. xgboost_classifier.py       (Task 1 — winner)
7. compare_models.py           (Task 1 evaluation)
8. XG_pair_classifier.py       (Task 2A)
9. xgb_range_regressor.py      (Task 2B)
```

> **Note:** `run_experiment.py` orchestrates steps 1–2 and 4–7 via `--only`/`--skip` flags.
> Steps 3, 8, and 9 must be run manually.
