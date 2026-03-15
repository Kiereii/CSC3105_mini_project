# Run Instructions (Pipeline v2)

This guide explains how to run the updated experiment pipeline with run-scoped outputs.

## 1) Create and activate virtual environment

### Bash/Zsh
```bash
python -m venv .venv
source .venv/bin/activate
```

### Fish
```fish
python -m venv .venv
source .venv/bin/activate.fish
```

## 2) Install dependencies

### Full requirements
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### If skipping CNN (no TensorFlow needed)
```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib scikit-learn seaborn scipy joblib xgboost
```

## 3) Run pipeline

Main runner:
```bash
python src/run_experiment.py [options]
```

### Common options
- `--test-size` test split ratio (example: `0.2` for 80/20, `0.3` for 70/30)
- `--seed` random seed
- `--run-name` output folder name under `runs/`
- `--skip-xgboost-tuning` disables XGBoost hyperparameter search
- `--only` run only selected steps (comma-separated)
- `--skip` skip selected steps (comma-separated)

### Available step names
- `preprocess`
- `second_path`
- `random_forest`
- `logreg_svm`
- `xgboost`
- `pair_classifier`
- `range_regressor`
- `compare`

## 4) Example commands

### A) Full run (80/20)
```bash
python src/run_experiment.py --test-size 0.2 --seed 42 --run-name split_80_20_seed42 --skip-xgboost-tuning
```

### B) Full run (70/30)
```bash
python src/run_experiment.py --test-size 0.3 --seed 42 --run-name split_70_30_seed42 --skip-xgboost-tuning
```

### C) Run only RF path (plus required preprocessing)
```bash
python src/run_experiment.py --run-name split_80_20_seed42 --only preprocess,second_path,random_forest
```

### D) Run everything except XGBoost and comparison
```bash
python src/run_experiment.py --run-name split_80_20_seed42 --skip xgboost,compare
```

## 5) Output locations

Each run is isolated in:

```text
runs/<RUN_NAME>/
  preprocessed_data/
  models/
    random_forest/
    logreg_svm/
    xgboost/
    pair_classifier/
    range_regressor/
    comparison/
```

This prevents 70/30 and 80/20 artifacts from overwriting each other.

## 6) Notes

- If you use `--only`, make sure prerequisite files already exist in the same `--run-name` folder.
- `compare` expects model outputs inside that same run folder.
- CNN is optional in comparison and is ignored if CNN files are missing.
