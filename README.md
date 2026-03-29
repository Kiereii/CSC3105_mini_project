# CSC3105 Data Analytics and AI Mini Project

## Project overview

This project analyses the provided **UWB LOS/NLOS dataset** to address two related tasks:

- **Task 1:** LOS/NLOS classification from UWB CIR measurements
- **Task 2:** two-path analysis consisting of
  - pair-level classification: `LOS+NLOS` vs `NLOS+NLOS`
  - range estimation for the two dominant paths

The project is organised around the three required stages of the coursework:

- **Data preparation**
- **Data mining / machine learning**
- **Data visualisation and result analysis**

---

## Repository structure

```text
.
├── Dataset/
├── runs/
│   └── split_env_70_15_15_seed42/
├── src/
│   ├── shared/
│   │   └── preprocessing/
│   ├── task1/
│   ├── task2/
│   ├── experimental/
│   └── run_experiment.py
├── requirements.txt
└── README.md
```

### Source code layout

- `src/shared/preprocessing/`
  - dataset loading, cleaning pipeline usage, feature engineering, second-path feature construction
- `src/task1/`
  - final Task 1 classification scripts and comparison
- `src/task2/`
  - final Task 2 pair classification and XGBoost range estimation
- `src/experimental/`
  - exploratory / comparison scripts kept for methodology and model-selection evidence

---

## Main scripts

### Shared preprocessing
- `src/shared/preprocessing/preprocess_data.py`
- `src/shared/preprocessing/second_path_features.py`

### Task 1
- `src/task1/random_forest_classifier.py`
- `src/task1/xgboost_classifier.py`
- `src/task1/compare_models.py`

### Task 2
- `src/task1/xgboost_pair_classifier.py`
- `src/task2/xgboost_range_regressor.py`

### Experimental scripts
- `src/experimental/dt_exploratory.py`
- `src/experimental/logreg_svm_classifier.py`
- `src/experimental/range_regressor.py`

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
python src/run_experiment.py [options]
```

### Valid options

- `--val-size`
- `--test-size`
- `--seed`
- `--run-name`
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
python src/run_experiment.py --val-size 0.15 --test-size 0.15 --seed 42 --run-name split_env_70_15_15_seed42
```

Run only selected steps:

```bash
python src/run_experiment.py --run-name split_env_70_15_15_seed42 --only preprocess,second_path,random_forest,xgboost,compare,pair_classifier,xgboost_range_regressor
```

Skip selected steps:

```bash
python src/run_experiment.py --run-name split_env_70_15_15_seed42 --skip dt_exploratory,logreg_svm,range_regressor
```

---

## Final run used for results

The main submission-facing run is:

```text
runs/split_env_70_15_15_seed42/
```

This contains:

- `preprocessed_data/`
- `models/random_forest/`
- `models/xgboost/`
- `models/comparison/`
- `models/pair_classifier/`
- `models/range_regressor/`

---

## Notes on outputs

- Task 1 outputs are mainly under:
  - `runs/<RUN_NAME>/models/random_forest/`
  - `runs/<RUN_NAME>/models/xgboost/`
  - `runs/<RUN_NAME>/models/comparison/`
- Task 2 outputs are mainly under:
  - `runs/<RUN_NAME>/models/pair_classifier/`
  - `runs/<RUN_NAME>/models/range_regressor/`
- Pair-classifier internal artifacts are stored under:
  - `runs/<RUN_NAME>/models/pair_classifier/artifacts/`

---

## Important limitation

The dataset provides:

- LOS/NLOS labels
- one measured range per sample
- CIR and related physical features

It does **not** provide verified ground-truth distance for a second reflected path.

Therefore:

- **Path 1 range** uses measured data
- **Path 2 range** is treated as a **derived / engineered estimate** based on second-path CIR information

This is a dataset limitation, not a missing implementation step.

---

## Submission note

For submission packaging, exclude development-only folders such as:

- `.git/`
- `.venv/`
- `.idea/`
- `.ruff_cache/`
- `__pycache__/`
