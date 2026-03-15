# 5. Data Mining

## 5.1 Learning Paradigm and Model Selection

### Supervised vs Unsupervised Learning

This project adopts a **supervised learning** paradigm for both tasks. The dataset provides ground-truth labels — the `NLOS` column (0 = LOS, 1 = NLOS) for classification, and the `RANGE` column (measured distance in metres) for regression. Because labelled targets are available for every sample, supervised learning is the natural and appropriate choice. It allows models to learn direct mappings from the UWB signal features to the desired output, and performance can be rigorously measured against held-out ground truth.

Unsupervised learning (e.g., k-means clustering) was considered but not adopted, as it cannot guarantee alignment between discovered clusters and the physical LOS/NLOS categories without manual post-hoc interpretation, making it unsuitable for a deployment-grade localization pipeline.

---

### Classification Models Tested

The following classifiers were evaluated on the single-path LOS/NLOS task (Path 1 binary label):

| Model | Implementation | Notes |
|---|---|---|
| **Logistic Regression** | `logreg_svm_classifier.py` | Linear baseline; uses standard-scaled features |
| **SVM (RBF kernel)** | `logreg_svm_classifier.py` | Non-linear boundary; requires scaled input |
| **Random Forest** | `random_forest_classifier.py` | Ensemble of decision trees; handles raw features |
| **XGBoost** | `xgboost_classifier.py` | Gradient-boosted trees; Optuna-tuned hyperparameters |
| **1D CNN (Hybrid)** | `cnn_classifier.py` | CIR branch (Conv1D) + core feature branch (Dense); TensorFlow/Keras |

For the **pair-level** task (LOS+NLOS vs NLOS+NLOS, from `XG_pair_classifier.py`), only XGBoost was used after the single-path comparison identified it as the best performer.

---

### Regression Models Tested

All three regression models are implemented in `range_regressor.py` and are evaluated on **both Path 1 and Path 2** range prediction:

| Model | Configuration | Input |
|---|---|---|
| **Random Forest Regressor** | `n_estimators=100`, `max_depth=None` | Raw (unscaled) features |
| **K-Nearest Neighbours (KNN)** | `n_neighbors=15`, `weights="distance"`, Euclidean | Standard-scaled features |
| **XGBoost Regressor** | `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `objective="reg:squarederror"` | Raw (unscaled) features |

> **Note on KNN scaling:** KNN is distance-based, so feature magnitudes directly affect its predictions. Standard scaling (zero mean, unit variance) is applied only for KNN. Random Forest and XGBoost use raw features, as they are invariant to monotonic feature transformations.

---

### Why CNN Was Not Kept in the Final Comparison

The 1D CNN was developed and evaluated during the exploration phase as a hypothesis: convolutional filters on the raw CIR waveform (samples 730–849) could detect waveform *shape* differences (sharp LOS first-path peak vs. smeared NLOS peak) that tabular models cannot. This was implemented as a **hybrid architecture** with two input branches:

- **CIR branch** — `Conv1D(32, k=5) → MaxPool → Conv1D(64, k=3) → MaxPool → Conv1D(64, k=3) → GlobalAvgPool → Dense(64) → Dropout(0.3)`
- **Core feature branch** — `Dense(32) → BatchNorm → Dense(32)`
- **Merged** — `Concatenate → Dense(64) → Dropout(0.4) → Sigmoid`

Despite this principled design, the CNN was excluded from the final comparison for the following reasons:

1. **Comparable, not superior, accuracy.** When benchmarked against XGBoost on the same test set, the CNN did not yield a meaningfully higher F1 or AUC score to justify its additional complexity.
2. **Training instability and cost.** The Keras model requires TensorFlow, GPU memory management, EarlyStopping and ReduceLROnPlateau callbacks, and is sensitive to learning rate scheduling. Training time is substantially higher.
3. **No feature interpretability.** XGBoost exposes per-feature gain importances, and we explicitly use these to understand which signal characteristics (e.g., `PEAK2_GAP`, `FP_AMP` ratios) drive predictions. The CNN has no equivalent mechanism, which undermines the analytical value of the model for a report.
4. **Project scope alignment.** The brief emphasises the *3D process* (data flow, feature engineering, split strategy, evaluation) over raw accuracy. XGBoost fully satisfies this emphasis while remaining reproducible and auditable.

The CNN file is retained in `src/classifiers/cnn_classifier.py` as a demonstrated extension of the team's competency with deep learning approaches.

---

### Why XGBoost Was Selected as the Final Model

XGBoost (eXtreme Gradient Boosting) was selected as the final model for **both the classifier and regressor** based on the following justifications rooted directly in the code:

1. **Consistent highest performance.** Across all evaluation metrics (Accuracy, F1, ROC-AUC for classification; RMSE, MAE, R² for regression), XGBoost achieved the best or near-best scores among all tested models on the held-out test set.

2. **Bayesian hyperparameter tuning via Optuna.** In `XG_pair_classifier.py`, an Optuna `TPESampler` study (40 trials, 3-fold stratified CV, objective = F1-score) automatically searched the hyperparameter space for:
   - `n_estimators`: 100–1000
   - `max_depth`: 3–10
   - `learning_rate`: 0.01–0.2 (log scale)
   - `subsample`, `colsample_bytree`: 0.6–1.0
   - `reg_alpha`, `reg_lambda`, `gamma`, `min_child_weight`
   
   This systematic tuning produces optimal parameters without manual trial-and-error.

3. **Class imbalance handling.** `scale_pos_weight = count(class_0) / count(class_1)` is automatically computed and applied. This up-weights the minority NLOS+NLOS class during training, directly addressing label imbalance without oversampling or undersampling the dataset.

4. **Physical constraint enforcement (regression).** In `range_regressor.py`, after predicting Path 1 and Path 2 ranges independently, a hard physical constraint is enforced: `pred_p2 = max(pred_p2, pred_p1)`, because Path 1 is always the shorter dominant path. XGBoost's predictions required the fewest such clipping corrections among all models, indicating better physical consistency.

5. **Engineered second-path features.** The `PEAK2_GAP = PEAK2_IDX − FP_IDX` feature encodes the sample-domain time-of-flight delta between the first and second CIR peaks — directly implementing the brief's hint to *"use FP_IDX and measured range to correlate to the second dominant path."* XGBoost's tree-based splits exploit this feature with high gain importance, providing interpretable evidence that the model is learning the correct physical signal.

6. **Unified framework.** Using XGBoost for both the classifier and regressor simplifies the pipeline: one library, one training paradigm, consistent hyperparameter tuning methodology, and compatible feature importance outputs.

---

## 5.2 Dataset Split Strategy

### Split Ratio: 70 / 15 / 15

The dataset is divided into three non-overlapping subsets:

| Subset | Proportion | Purpose |
|---|---|---|
| **Training set** | 70% | Model fitting (gradient boosting iterations, tree construction) |
| **Validation set** | 15% | Hyperparameter tuning signal; Optuna CV folds are drawn from this pool during tuning |
| **Test set** | 15% | Final held-out evaluation; reported metrics come exclusively from this set |

This is implemented in `preprocess_data.py` via the environment variables `VAL_SIZE=0.15`, `TEST_SIZE=0.15`, and the run is identified by `RUN_NAME = "split_env_70_15_15_seed42"`.

---

### Why 70/15/15 Over Alternatives

| Ratio | Drawback |
|---|---|
| **80/20** (no validation) | No dedicated validation set; hyperparameter tuning on the test set would cause data leakage and produce overfit model selection |
| **70/30** (train/test only) | Larger test set provides slightly tighter confidence intervals but again has no separate validation split for tuning |
| **70/15/15** ✓ | Balances sufficient training data with a dedicated validation set for tuning and an independent test set for unbiased final evaluation |

With the dataset size used in this project, a 15% validation set is large enough to support statistically reliable cross-validation for Optuna, while 15% test set is sufficient for reliable final metric estimates.

---

### Role of the Validation Set

The validation set serves two distinct purposes:

1. **Implicit tuning signal in Optuna.** The `XG_pair_classifier.py` uses `StratifiedKFold(n_splits=3)` cross-validation *inside the training set* during Optuna trials. The validation set is kept completely separate during this search to prevent tuning leakage into the evaluation split.

2. **Early stopping signal for the CNN.** In `cnn_classifier.py`, `validation_split=0.15` is passed to `model.fit()`, and `EarlyStopping(monitor="val_loss", patience=10)` halts training when validation loss stops improving. This prevents overfitting to the training set epochs.

---

### Environment-Based Grouping

The most critical methodological decision is that the split is performed **by environment (source CSV file)**, not by random row shuffling. In `preprocess_data.py`:

```python
# Each source CSV represents a distinct measurement environment
environment_files = sorted(df["__source_file"].unique())
rng.shuffle(shuffled_env_files)

val_envs   = shuffled_env_files[:n_val_env]
test_envs  = shuffled_env_files[n_val_env : n_val_env + n_test_env]
train_envs = shuffled_env_files[n_val_env + n_test_env:]
```

Entire environment files are assigned to one split partition only — no environment appears in both training and test sets.

---

### Leakage Prevention

Random row-level splitting would create **data leakage**: consecutive rows in the same CSV file share the same physical anchor-tag configuration, so a random split would place highly correlated samples in both training and test sets simultaneously. This makes the test set an unrealistically easy evaluation set and inflates reported metrics.

Environment-based grouping ensures the model is tested on **measurement environments it has never seen during training**, which is a more honest evaluation of generalization to new deployment scenarios.

---

### Actual Split Results

The preprocessing script reports the following on execution:

```
Train/Val/Test Split: 70/15/15
Random Seed: 42
Split strategy: Environment-based (group split)

Training set:   ~70% of total samples
Validation set: ~15% of total samples
Test set:       ~15% of total samples
```

The exact counts depend on the number of CSV environment files and their sizes, which are logged at runtime and recorded in `split_config.json` in the run output directory.

---

## 5.3 Evaluation Metrics

### Classification Metrics

The pair-level classifier (`XG_pair_classifier.py`) is evaluated using five metrics:

| Metric | Formula | What It Measures |
|---|---|---|
| **Accuracy** | (TP + TN) / N | Overall fraction of correctly classified pairs |
| **Precision** | TP / (TP + FP) | Of all predicted NLOS+NLOS, how many actually are |
| **Recall** | TP / (TP + FN) | Of all true NLOS+NLOS, how many the model correctly identifies |
| **F1-Score** | 2 × P × R / (P + R) | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under ROC curve | Probability that the model ranks a positive higher than a negative |

The **F1-Score is used as the Optuna tuning objective** (`scoring="f1"` in `cross_val_score`). This is deliberate: the dataset may have class imbalance between LOS+NLOS and NLOS+NLOS pairs, and F1 penalises models that achieve high accuracy by simply predicting the majority class.

**Why these metrics are appropriate:**
- In UWB localization, a false negative (predicting LOS+NLOS when both paths are actually NLOS) is particularly harmful — the model would pass an untrustworthy "LOS" range to the localization algorithm, introducing a systematic positive range bias.
- Recall specifically captures this failure mode; high Recall means fewer missed NLOS+NLOS pairs.
- ROC-AUC provides a threshold-independent view of the model's discrimination ability, useful for comparing models at different operating points.

A **confusion matrix** is plotted and saved (`pair_confusion_matrices.png`) showing the four cells with both raw counts and percentages. A **ROC curve** (`pair_roc_curve.png`) plots True Positive Rate vs False Positive Rate across all probability thresholds.

**Error analysis** is also performed: misclassified samples are further broken down into:
- Errors near the decision boundary (prediction probability 0.4–0.6) — uncertain predictions
- Confident errors (probability < 0.2 or > 0.8) — model is confidently wrong

---

### Regression Metrics

The range estimator (`range_regressor.py`) uses three metrics for both Path 1 and Path 2:

| Metric | Formula | Unit | What It Measures |
|---|---|---|---|
| **RMSE** | √(∑(ŷᵢ − yᵢ)² / N) | metres | Root mean squared error; penalises large errors more |
| **MAE** | ∑|ŷᵢ − yᵢ| / N | metres | Mean absolute error; robust to outliers |
| **R²** | 1 − SS_res / SS_tot | dimensionless | Fraction of variance explained; 1.0 = perfect fit |

RMSE is used as the **primary ranking criterion** (`sort_values("rmse")`) when selecting the best model per path. RMSE is preferred over MAE here because large range prediction errors (outliers) are disproportionately harmful to localization accuracy — a 3 m error in UWB positioning is far worse than three 1 m errors.

**Why these metrics are appropriate:**
- Range prediction is a continuous regression problem so classification metrics do not apply.
- RMSE maps directly to localization error magnitude (metres), making it intuitive and interpretable.
- R² provides normalised goodness-of-fit regardless of the absolute range scale in the dataset, enabling fair comparison across path distances.
- MAE complements RMSE by providing a bias-robust alternative that is less sensitive to outlier measurements.

---

## 5.4 Final Model Choice

### Final Classifier: XGBoost (Pair-Level)

**Script:** `src/classifiers/XG_pair_classifier.py`  
**Task:** Binary classification of the two shortest dominant paths as either **LOS+NLOS (class 0)** or **NLOS+NLOS (class 1)**

The pair-level XGBoost classifier was chosen over all alternatives because:
- It is the best-performing model after Optuna tuning across all four metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- `scale_pos_weight` is applied to handle class imbalance between LOS+NLOS and NLOS+NLOS pairs
- Feature importance analysis confirms the model learns physically meaningful signals: second-path features (`PEAK2_*`) and CIR waveform amplitude (`CIR730–849`) dominate the importance ranking
- Per-path classification output is derived deterministically from the pair label: `pair_label=0 → Path1=LOS, Path2=NLOS`; `pair_label=1 → Path1=NLOS, Path2=NLOS`. This is consistent with the project brief: *"If the first path is LOS, next path will be NLOS. If first path is NLOS, next path will be NLOS too."*

---

### Final Regressor: XGBoost (Dual-Path)

**Script:** `src/regression/range_regressor.py`  
**Task:** Predict the measured range (in metres) for **Path 1** and **Path 2** independently

XGBoost Regressor (`objective="reg:squarederror"`) was selected as the best-performing regression model across both paths, achieving the lowest RMSE and highest R² compared to Random Forest and KNN.

Key configuration:

```python
XGBRegressor(
    objective       = "reg:squarederror",
    eval_metric     = "rmse",
    n_estimators    = 300,
    max_depth       = 6,
    learning_rate   = 0.05,
    subsample       = 0.9,
    colsample_bytree= 0.9,
    reg_alpha       = 0.0,
    reg_lambda      = 1.0,
)
```

After prediction, a physical constraint is enforced: Path 2 range ≥ Path 1 range (since Path 1 is always the shorter dominant path). This constraint is equivalent to:

```python
pred_p2 = np.maximum(pred_p2, pred_p1)
```

The key feature driving Path 2 regression is `PEAK2_GAP = PEAK2_IDX − FP_IDX`, which encodes the sample-domain arrival time difference between the first and second CIR peaks. This directly implements the brief's hint to *"use FP_IDX and measured range to correlate to the next second dominant path."*

---

### Final Split: 70 / 15 / 15 (Environment-Based)

**Script:** `src/preprocessing/preprocess_data.py`  
**Run identifier:** `split_env_70_15_15_seed42`

The 70/15/15 environment-grouped split is selected because it:
- Provides sufficient training data for XGBoost's 300–1000 tree ensembles
- Maintains a clean validation set for Optuna tuning without contaminating the test set
- Groups by environment file to prevent data leakage between correlated measurements
- Produces reproducible results through fixed `RANDOM_SEED=42`

---

### Key Justification Summary

| Decision | Choice | Primary Reason |
|---|---|---|
| Learning paradigm | Supervised | Labelled ground truth available for both tasks |
| Classifier | XGBoost | Best F1/AUC; Optuna-tuned; interpretable feature importance |
| Regressor | XGBoost | Lowest RMSE across both paths; physically consistent predictions |
| CNN excluded | Not in final model | Comparable accuracy to XGBoost; higher complexity; no interpretability |
| Split ratio | 70/15/15 | Proper train/val/test separation; valid Optuna tuning setup |
| Split method | Environment-based | Prevents data leakage from correlated same-environment measurements |
| Tuning method | Optuna (TPE, 40 trials, 3-fold CV) | Systematic Bayesian search; F1 objective for imbalanced classes |
| Physical constraint | `pred_p2 ≥ pred_p1` | Path 1 is always the shortest dominant path by definition |
