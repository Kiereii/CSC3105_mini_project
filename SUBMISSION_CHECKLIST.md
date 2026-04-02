# Submission Checklist

## Include in submission

### Required core
- `README.md`
- `requirements.txt`
- `src/`
- `Dataset/`

### Include results / evidence
- `outputs/preprocessed/`
- `outputs/`
  - `outputs/pair_classifier/` (plots, metrics, model artifacts)
  - `outputs/range_regressor/` (plots, metrics, predictions)

### Include analysis
- `notebooks/`
  - `pair_classifier_analysis.ipynb`
  - `range_regressor_analysis.ipynb`

---

## Exclude from submission

### Definitely exclude
- `.git/`
- `.venv/`
- `.idea/`
- `.ruff_cache/`
- all `__pycache__/`
- `CSC3105_Mini_Project_2026.pdf`
- `SUBMISSION_CHECKLIST.md`
- `.ipynb_checkpoints/`
- `.DS_Store`

---

## Before zipping

### 1. Check root looks clean

```text
UoG-DA_GroupXX/
├── README.md
├── requirements.txt
├── Dataset/
├── src/
├── outputs/
│   ├── preprocessed/
│   ├── eda/
│   ├── pair_classifier/
│   └── range_regressor/
└── notebooks/
```

### 2. Check `src/` looks clean

```text
src/
├── shared/
│   └── preprocessing/
├── task1/
├── task2/
└── experimental/
```

### 3. Remove Python cache folders
Delete any:
- `__pycache__/`

### 4. Check outputs exist
Confirm:
- `outputs/pair_classifier/` has PNGs and metrics
- `outputs/range_regressor/` has PNGs and metrics
- `outputs/preprocessed/` has split data and config

### 5. Check README exists and is accurate
Confirm:
- `README.md` references correct file paths

---

## Recommended final folder name

```text
UoG-DA_GroupXX/
```

Zip that folder as `UoG-DA.zip` for submission.

---

## Final pre-zip checklist

- [ ] `README.md` is present
- [ ] `requirements.txt` is present
- [ ] `src/` is present (no `__pycache__/`)
- [ ] `Dataset/` is present
- [ ] `outputs/` is present with PNGs, metrics, and preprocessed data
- [ ] `notebooks/` is present (no `.ipynb_checkpoints/`)
- [ ] no `.venv/`
- [ ] no `.git/`
- [ ] no `.idea/`
- [ ] no `.ruff_cache/`
- [ ] no `__pycache__/`
- [ ] no `.DS_Store`
- [ ] no `CSC3105_Mini_Project_2026.pdf`
