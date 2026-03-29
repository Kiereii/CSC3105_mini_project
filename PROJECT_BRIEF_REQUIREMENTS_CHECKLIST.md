# CSC3105 Mini Project Brief Checklist

Scope of this checklist:
- Based on `CSC3105_Mini_Project_2026.pdf`
- Excludes the separate report PDF and IEEE/Overleaf summary as requested

## Overall verdict

Your project is **mostly complete on the technical side**, but **not fully complete for the full brief**.

- **Technical implementation**: mostly satisfied
- **Full brief excluding only report/IEEE**: not fully satisfied yet

Main reason: the repo covers preprocessing, modeling, visualization, and generated outputs well, but some remaining submission deliverables are still missing. Also, the **second-path distance target is engineered**, not true measured ground truth.

---

## Requirement-by-requirement checklist

| Brief requirement | Status | Notes |
|---|---|---|
| Use the given UWB dataset | Satisfied | Dataset is present locally under `Dataset/UWB-LOS-NLOS-Data-Set/`. |
| Data preparation / cleaning | Satisfied | Cleaning script exists in `Dataset/UWB-LOS-NLOS-Data-Set/code/clean_local.py`. |
| Data preprocessing | Satisfied | Implemented in `src/preprocessing/preprocess_data.py`. |
| Data feature extraction / transformation | Satisfied | Engineered features implemented in `src/preprocessing/feature_engineering.py`. |
| Decide on data reduction need | Partial | CIR window reduced to focused range `730-850`, but this is more implied by implementation than explicitly framed as a formal reduction decision. |
| Data cleaning need / handling | Satisfied | Duplicates, nulls, invalid ranges, and noise filtering handled in cleaning script. |
| Class labelling for two-path task | Satisfied | Pair-label logic implemented in `src/preprocessing/second_path_features.py`. |
| Feature importance ranking | Satisfied | RF, XGBoost, LR, SVM, DT importance/coefficient outputs are generated. |
| Decide whether synthetic data is needed | Partial | No synthetic data generation is implemented. This is acceptable if justified as unnecessary, but the repo itself does not strongly document that decision. |
| Choose supervised vs unsupervised approach | Satisfied | Supervised approach is used and justified in `SUPERVISED_VS_UNSUPERVISED_JUSTIFICATION.md`. |
| Choose suitable data mining algorithms | Satisfied | Multiple algorithms are implemented for both classification and regression. |
| Choose train/test split ratio | Satisfied | Environment-based split implemented; runs exist for `70/15/15`, `80/20`, and `70/30`. |
| Classification evaluation metrics | Satisfied | Accuracy, precision, recall, F1, ROC-AUC, confusion matrices are produced. |
| Regression evaluation metrics | Satisfied | RMSE, MAE, R² are produced. |
| Data visualization | Satisfied | EDA, ROC, confusion matrices, feature importance, regression plots all exist. |
| Result analysis / justification | Partial | There is strong supporting material in markdown/docs and result summaries, but this is less formal in-repo than the implementation itself. |
| LOS/NLOS classification task | Satisfied | Implemented with multiple models in `src/classifiers/`. |
| Two dominant shortest paths classification | Satisfied | Implemented via pair-level classification in `src/task1/xgboost_pair_classifier.py`. |
| Distance estimation for two dominant shortest paths | Partial | Path 1 is real measured range; Path 2 is an engineered proxy derived from second-peak offset, not true measured ground truth. |
| Plot various performance indicators | Satisfied | Present in `runs/split_env_70_15_15_seed42/report_plots/` and `runs/.../models/comparison/`. |
| All python source code needed to reproduce plots/results | Satisfied | Main pipeline and plotting scripts are present. |
| Code structure / organization | Satisfied | Repo is structured into preprocessing, classifiers, regression, evaluation, docs, runs. |
| Comments / code documentation | Satisfied | Most main scripts are documented and readable. |

---

## What is already clearly done

### 1. Data Preparation
- cleaning script for raw dataset
- preprocessing pipeline
- feature engineering
- scaling for linear models
- environment-based split to reduce leakage

### 2. Data Mining
- classification models:
  - Random Forest
  - Logistic Regression
  - SVM
  - XGBoost
- exploratory decision tree
- pair-level two-path classifier
- regression models for range estimation

### 3. Data Visualization
- class distribution
- correlation heatmap
- feature distributions
- average CIR waveform plots
- confusion matrices
- ROC curves
- feature importance charts
- regression predicted-vs-actual plots
- regression comparison plots

### 4. Generated Results Already Present
- main completed run in `runs/split_env_70_15_15_seed42/`
- additional runs for `split_80_20_seed42` and `split_70_30_seed42`
- saved metrics and model outputs

---

## Important caveat: Task 2 is only partially literal

The brief asks for predicting the **measured range for the two dominant shortest paths**.

What your repo currently does:
- **Path 1**: predicts actual measured range
- **Path 2**: predicts a derived target `RANGE_PATH2_EST`

That Path 2 target is computed from:
- detected second CIR peak
- `PEAK2_GAP`
- a meters-per-sample conversion

So this is:
- **reasonable and defendable** as a project implementation
- but **not literally true ground-truth measured range for Path 2**

This should be stated clearly in the report/presentation/demo.

---

## Remaining items still missing from the brief

These are not part of the separate report/IEEE summary, so they still count as not yet complete:

### Missing submission/delivery items
- `UoG-DA.zip` or `UoG-DA.tgz`
- final `UoG-DA_Groupxx/` hand-in folder structure
- short YouTube demo video
- group presentation video
- signed declaration forms

### Weak / should-be-made-explicit items
- explicit justification that **synthetic data was considered but not needed**
- clearer explicit statement on **data reduction decision**
- clearer explicit statement of **novelty/originality** of the approach

---

## Final answer

If the question is:

### “Have I already done the core technical work from the brief?”
**Yes, mostly.**

### “Have I already satisfied everything except the report and IEEE summary?”
**No, not fully.**

You are missing some submission deliverables, and the second-path range task is only **partially satisfied** because it uses an engineered estimate rather than true measured labels.
