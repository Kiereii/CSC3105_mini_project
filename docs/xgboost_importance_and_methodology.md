# Linking Gini/Entropy Intuition to XGBoost Feature Importance

## 1) How to Link Gini/Entropy Importance to XGBoost Importance

### 1.1 Classical tree intuition (CART / Random Forest)

For classification trees, split quality is measured by impurity reduction.

- Gini impurity at node `t`:

  `G(t) = 1 - sum_k p(k|t)^2`

- Entropy at node `t`:

  `H(t) = - sum_k p(k|t) log p(k|t)`

If split `s` sends samples from node `t` into `tL` and `tR`, impurity decrease is:

`DeltaI(s,t) = I(t) - (nL/nt)I(tL) - (nR/nt)I(tR)`

Feature importance in CART/Random Forest is then aggregated as mean decrease impurity (MDI) over all splits using that feature.

### 1.2 What transfers to XGBoost

This intuition still helps:

- Features are important if they are repeatedly chosen in useful splits.
- Better split quality generally means higher importance.
- Importance is aggregated across many trees.

So your intuition is directionally correct.

### 1.3 What changes in XGBoost (critical difference)

XGBoost does **not** optimize plain Gini/Entropy impurity directly.

It optimizes a regularized objective:

`L = sum_i l(yi, yhat_i) + Omega(f)`

with tree regularization such as:

`Omega(f) = gamma * T + (lambda/2) * sum_j wj^2 + alpha * sum_j |wj|`

Split gain is computed from gradient and hessian statistics of the current boosting stage, plus regularization. In short:

- CART importance: "how much impurity was reduced"
- XGBoost importance: "how much the regularized training objective improved"

That is why your Gini/Entropy understanding is a good intuition bridge, but not the exact XGBoost mechanism.

### 1.4 XGBoost importance types

Common importance types in XGBoost:

- `weight`: how many times a feature is used for splits
- `gain`: average gain of splits using that feature
- `cover`: average coverage (hessian/sample mass) of those splits
- `total_gain`: sum of gains over all splits using that feature
- `total_cover`: sum of coverage over all splits using that feature

In this project, `xgb_model.feature_importances_` is used in `src/classifiers/xgboost_classifier.py` and treated as gain-style ranking.

### 1.5 Why this matters for model behavior

Because importance is objective-driven in XGBoost:

1. Hyperparameters directly influence importance patterns (`max_depth`, `gamma`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `colsample_*`, etc.).
2. Class imbalance handling (`scale_pos_weight`) changes gradient statistics, which changes split gains and importance.
3. Importance is model-internal, not causal proof.

### 1.6 Interpretation caveats

- Split-based importance can be biased by high-cardinality or highly variable features.
- Correlated features can split importance between each other.
- Importance ranking can vary across seeds/splits.

Recommended complements:

- permutation importance on held-out data,
- SHAP for local/global attributions,
- stability checks across multiple seeds.

---

## 2) Algorithms Used Across the Solution (Data Prep -> Data Mining -> Data Analysis)

### 2.1 Data preparation algorithms

Main scripts:

- `src/preprocessing/preprocess_data.py`
- `src/preprocessing/feature_engineering.py`
- `src/preprocessing/second_path_features.py`

Core process:

1. Load and merge cleaned UWB dataset partitions.
2. Build feature set from core signal metrics + focused CIR window (730-849) + engineered features.
3. Build second-path features from peak detection (`PEAK2_*`, `PEAK2_GAP`, etc.).
4. Split into train/val/test (environment-aware grouping where configured).
5. Save unscaled and standardized arrays for model-specific use.

### 2.2 Data mining (modeling) algorithms

Classification models in final comparison flow:

- Random Forest (`src/classifiers/random_forest_classifier.py`)
- Logistic Regression (`src/classifiers/logreg_svm_classifier.py`)
- Linear SVM (`src/classifiers/logreg_svm_classifier.py`)
- XGBoost (`src/classifiers/xgboost_classifier.py`)

Pair-level framing:

- LOS+NLOS vs NLOS+NLOS (pair classification)
- Comparison emphasis in evaluation: RF and XGBoost vs pair-level RF/XGBoost outputs (when available)

Note: `src/classifiers/cnn_classifier.py` exists but is not part of final comparison flow.

### 2.3 Data analysis/visualization algorithms

Main scripts:

- `src/evaluation/compare_models.py`
- `src/evaluation/generate_report_plots.py`

These aggregate predictions into:

- metric tables,
- confusion matrices,
- ROC overlays,
- feature-importance plots,
- per-path vs pair-level comparison plots,
- safety-focused NLOS error views.

---

## 3) Evaluation Criteria and Methodology

### 3.1 Classification metrics

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | `(TP+TN)/(TP+TN+FP+FN)` | Overall correctness |
| Precision | `TP/(TP+FP)` | Reliability of predicted positives |
| Recall | `TP/(TP+FN)` | Fraction of positives detected |
| F1-score | `2PR/(P+R)` | Balance of precision and recall |
| ROC-AUC | Area under ROC | Threshold-independent separability |

For LOS/NLOS safety, false negatives (NLOS predicted as LOS) are critical and explicitly analyzed in comparison outputs.

### 3.2 Confusion matrix methodology

Confusion matrices are used to inspect:

- true LOS vs true NLOS recognition,
- dangerous misses (NLOS -> LOS),
- false alarms (LOS -> NLOS).

### 3.3 ROC and threshold methodology

ROC compares `TPR` vs `FPR` across thresholds.

Threshold selection (where used) is validation-driven (for example, maximizing F1 with recall-aware tie-breaking), then applied to test predictions.

### 3.4 Regression metrics (project scope includes range estimation)

| Metric | Formula | Use |
|---|---|---|
| RMSE | `sqrt(mean((y-yhat)^2))` | Penalizes large errors strongly |
| MAE | `mean(|y-yhat|)` | Robust average absolute error |
| R^2 | `1 - SSres/SStot` | Variance explained |

### 3.5 Similarity/dissimilarity framing

Distance-based models (e.g., KNN in regression experiments) rely on feature-space dissimilarity. This is why standardized input is required for those algorithms.

---

## 4) Pseudocode for Key Algorithms

```text
Algorithm A: End-to-End Pipeline
Input: dataset, run configuration
Output: model artifacts, metrics, plots

1. Preprocess data
   - load/merge files
   - engineer features
   - split train/val/test
   - save arrays + metadata

2. Build second-path and pair-level features

3. Train models
   - RF, LR, SVM, XGBoost
   - pair-level model outputs (if enabled)

4. Evaluate and compare
   - metrics table
   - confusion matrices
   - ROC plots
   - feature importance plots
   - per-path vs pair-level comparisons
```

```text
Algorithm B: XGBoost Training and Evaluation
Input: X_train, y_train, X_val, y_val, X_test, y_test
Output: trained model, threshold, test metrics, feature importances

1. Configure XGBoost hyperparameters (+ class weighting if needed)
2. Fit model on training data
3. Get validation probabilities
4. Sweep threshold t in [0.10, 0.90]
5. Select t* by best validation F1 (recall-aware tie break)
6. Apply t* on test probabilities
7. Compute Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix
8. Export feature importance rankings and plots
```

```text
Algorithm C: Per-Path vs Pair-Level Comparison
Input: per-path outputs, pair-level outputs
Output: comparison CSV + plots

1. Load per-path predictions/probabilities (RF, XGBoost)
2. Load pair-level predictions/probabilities (Pair RF, Pair XGBoost)
3. Compute metrics (Accuracy, F1, ROC-AUC) for each model
4. Plot:
   - pair confusion matrices
   - grouped bar comparison
   - combined ROC (solid=per-path, dashed=pair-level)
5. Save plots and comparison table
```

---

## 5) Source Code Mapping (for report section "all source codes")

| Component | Script | Purpose |
|---|---|---|
| Pipeline runner | `src/run_experiment.py` | Orchestrates steps and run folders |
| Preprocessing | `src/preprocessing/preprocess_data.py` | Splits, scaling, saved arrays |
| Feature engineering | `src/preprocessing/feature_engineering.py` | Derived features from raw/CIR |
| Second-path extraction | `src/preprocessing/second_path_features.py` | `PEAK2_*`, pair labels |
| RF classifier | `src/classifiers/random_forest_classifier.py` | RF train/eval + importance |
| LR/SVM classifier | `src/classifiers/logreg_svm_classifier.py` | Linear baselines + coefficients |
| XGBoost classifier | `src/classifiers/xgboost_classifier.py` | XGB train/eval + importance |
| Pair classifier | `src/classifiers/XG_pair_classifier.py` | Pair-level model training |
| Unified comparison | `src/evaluation/compare_models.py` | Consolidated metrics + visuals |
| Report plots | `src/evaluation/generate_report_plots.py` | Batch report figure generation |
| Regression models | `src/regression/range_regressor.py` | RMSE/MAE/R2 comparison |

---

## 6) Mathematical and Logical Explanation of Plots and Results

### 6.1 Metrics bar comparison

- Shows scalar performance differences at chosen operating thresholds.
- Best bar depends on metric priority (e.g., safety may prioritize recall over raw accuracy).

### 6.2 Confusion matrices

- Show exact error structure.
- In this domain, high `FN` (NLOS predicted LOS) is a safety risk and should be highlighted.

### 6.3 ROC curves

- Compare discrimination across all thresholds.
- AUC ranks separability, but deployment still needs a threshold selected from validation policy.

### 6.4 Feature-importance plots

- RF importance: impurity decrease aggregation.
- XGBoost importance: objective gain-based aggregation.
- Agreement between RF and XGB top features is useful evidence; disagreement is expected due to different split criteria.

### 6.5 Per-path vs pair-level plots

- Show whether reframing the target improves separability and safety profile.
- Interpret carefully: semantics of labels differ between tasks.

---

## 7) Interesting Aspects, Assumptions, and Optimizations

### 7.1 Assumptions

- Pair labels follow LOS+NLOS vs NLOS+NLOS framing from project brief.
- Second-path features derived from CIR peaks are valid proxies.
- Environment-aware split better estimates real generalization than random row split.

### 7.2 Optimizations used

- Class imbalance handling (`scale_pos_weight` / class weights).
- Validation-based threshold tuning.
- Hyperparameter optimization for tree models.
- Safety-oriented metric focus (NLOS recall and dangerous error counts).

### 7.3 Limitations and improvements

- Feature importance is not causal explanation.
- Correlated features can distort ranking.
- Recommend adding permutation importance + SHAP + seed stability study in future iterations.

---

## Final Interpretation Statement

You can present your understanding like this:

"Gini/Entropy importance gave us the foundational intuition that useful features are those producing strong split improvements. XGBoost retains this split-importance spirit but replaces impurity reduction with objective-driven gain (gradient/hessian and regularization aware). Therefore, XGBoost feature importance should be interpreted as contribution to optimized loss reduction in boosted trees, not as plain impurity decrease from a single standalone tree."
