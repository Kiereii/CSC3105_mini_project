# Submission Checklist

## Include in submission

### Required core
- `README.md`
- `requirements.txt`
- `src/`
- `Dataset/`

### Include results / evidence
- `runs/`
  - preferably only:
    - `runs/split_env_70_15_15_seed42/`

---

## Exclude from submission

### Definitely exclude
- `.git/`
- `.venv/`
- `.idea/`
- `.ruff_cache/`
- all `__pycache__/`

### Probably exclude
- `CSC3105_Mini_Project_2026.pdf`
- `PROJECT_BRIEF_REQUIREMENTS_CHECKLIST.md`
- `docs/`

---

## Before zipping

### 1. Check root looks clean
Target root should look roughly like:

```text
<submission folder>/
├── README.md
├── requirements.txt
├── Dataset/
├── runs/
│   └── split_env_70_15_15_seed42/
└── src/
```

### 2. Check `src/` looks clean

```text
src/
├── experimental/
├── shared/
├── task1/
├── task2/
└── run_experiment.py
```

### 3. Remove Python cache folders
Delete any:
- `__pycache__/`

### 4. Check final run exists
Confirm:
- `runs/split_env_70_15_15_seed42/`

### 5. Check README exists
Confirm:
- `README.md`

---

## Recommended final folder name

Use something like:

```text
UoG-DA_GroupXX/
```

Zip that folder for submission.

---

## Recommended final contents

- `README.md`
- `requirements.txt`
- `src/`
- `Dataset/`
- `runs/split_env_70_15_15_seed42/`

---

## Final pre-zip checklist

- [ ] `README.md` is present
- [ ] `requirements.txt` is present
- [ ] `src/` is present
- [ ] `Dataset/` is present
- [ ] `runs/split_env_70_15_15_seed42/` is present
- [ ] no `.venv/`
- [ ] no `.git/`
- [ ] no `.idea/`
- [ ] no `.ruff_cache/`
- [ ] no `__pycache__/`
- [ ] no extra exploratory clutter in the submission root
