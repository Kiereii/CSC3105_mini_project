# CSC3105 Data Analytics and AI Mini Project

**GitHub:** https://github.com/Kiereii/CSC3105_mini_project
**Youtube:** https://youtu.be/koB7NhexEbc

## Project overview

This project analyses the provided **UWB LOS/NLOS dataset** to address two related tasks:

- **Task 1:** Pair-level LOS/NLOS classification (`LOS+NLOS` vs `NLOS+NLOS`) from UWB CIR measurements using XGBoost
- **Task 2:** Range estimation for the two dominant paths using XGBoost regression

The project is organised around the three required stages of the coursework:

- **Data preparation** (cleaning, feature engineering, second-path extraction)
- **Data mining / machine learning** (XGBoost classifier and regressor with cross-validation)
- **Data visualisation and result analysis** (confusion matrices, ROC curves, feature importance, residual plots)

---

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── Dataset/                          # Cleaned UWB CSV data (7 environments)
│   └── Cleaned/
├── src/                              # All Python source code
│   ├── shared/preprocessing/         # Data loading, cleaning, feature engineering
│   ├── task1/                        # Pair classifier (XGBoost)
│   ├── task2/                        # Range regressor (XGBoost)
│   └── experimental/                 # Model comparison and exploration scripts
├── outputs/
│   ├── preprocessed/                 # Train/val/test splits, scalers, feature names
│   ├── eda/                          # Exploratory data analysis plots
│   ├── pair_classifier/              # Classification results, plots, metrics
│   └── range_regressor/              # Regression results, plots, metrics
└── notebooks/                        # Jupyter analysis notebooks
    ├── pair_classifier_analysis.ipynb
    └── range_regressor_analysis.ipynb
```

### Source code layout

- `src/shared/preprocessing/`
  - `preprocess_data.py` - dataset loading, feature extraction, environment-based train/val/test split
  - `feature_engineering.py` - 12 engineered features (SNR ratios, CIR kurtosis/skewness, rise time, etc.)
  - `second_path_features.py` - second dominant CIR peak detection and pair labelling
  - `eda_focused.py` - exploratory data analysis with focused CIR region visualisation
- `src/task1/`
  - `xgboost_pair_classifier.py` - XGBoost pair-level LOS/NLOS classifier with cross-validation
- `src/task2/`
  - `xgboost_range_regressor.py` - XGBoost range estimation for both dominant paths
- `src/experimental/`
  - model comparison and exploration scripts (Decision Tree, Random Forest, Logistic Regression, SVM, XGBoost single-path classifier)
  - `run_experiment.py` - pipeline runner for all steps

---

## Main scripts

### Shared preprocessing
- `src/shared/preprocessing/preprocess_data.py`
- `src/shared/preprocessing/feature_engineering.py`
- `src/shared/preprocessing/second_path_features.py`

### Task 1 - Pair classification
- `src/task1/xgboost_pair_classifier.py`

### Task 2 - Range estimation
- `src/task2/xgboost_range_regressor.py`

### Experimental / model comparison
- `src/experimental/random_forest_classifier.py`
- `src/experimental/xgboost_classifier.py`
- `src/experimental/compare_models.py`
- `src/experimental/logreg_svm_classifier.py`
- `src/experimental/dt_exploratory.py`
- `src/experimental/range_regressor.py`
- `src/experimental/run_experiment.py`

---

## Environment setup

### Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the pipeline

Main runner:

```bash
python src/experimental/run_experiment.py [options]
```

### Valid options

- `--val-size`
- `--test-size`
- `--seed`
- `--only`
- `--skip`

### Available step names

- `preprocess`
- `second_path`
- `dt_exploratory`
- `random_forest`
- `logreg_svm`
- `xgboost`
- `compare`
- `pair_classifier`
- `range_regressor`
- `xgboost_range_regressor`

### Example

Run the full pipeline with the environment-based `70/15/15` split:

```bash
python src/experimental/run_experiment.py --val-size 0.15 --test-size 0.15 --seed 42
```

Run only selected steps:

```bash
python src/experimental/run_experiment.py --only preprocess,second_path,pair_classifier,xgboost_range_regressor
```

---

## Outputs

### Pair classifier results (`outputs/pair_classifier/`)
- Accuracy: 93.23%, F1: 0.9303, ROC-AUC: 0.9832
- Visualisations: confusion matrix, ROC curve, feature importance, threshold optimisation
- Artifacts: trained model, predictions, error analysis

### Range regressor results (`outputs/range_regressor/`)
- Path 1: RMSE = 1.17 m, R² = 0.75
- Path 2: RMSE = 1.19 m, R² = 0.79
- Visualisations: actual vs predicted scatter, residual distribution, feature importance

### Preprocessed data (`outputs/preprocessed/`)
- 41,568 samples split 70/15/15 (environment-based)
- 148 features: 16 core + 120 CIR + 12 engineered

---
