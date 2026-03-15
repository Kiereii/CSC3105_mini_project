# Model Comparison & Unified Visualization — Explained

**Script:** `src/evaluation/compare_models.py`  
**Output directory:** `models/comparison/`

---

## What This Script Does (Big Picture)

After training four separate classifiers (Random Forest, Logistic Regression, SVM, XGBoost) and a pair-level classifier (Pair RF, Pair XGBoost), this script pulls all of their saved predictions together and produces **one unified set of report-ready plots and tables**. The goal is to make it easy to compare every model side-by-side on the same scale, using the same ground-truth labels.

---

## Step-by-Step Breakdown

### Step 1 — Load All Model Outputs

```python
y_test       = np.load("preprocessed_data/y_test.npy")
y_pred[...]  = np.load("models/.../y_pred_*.npy")
y_proba[...] = np.load("models/.../y_proba_*.npy")
```

Each trained model saves its predictions as `.npy` files after training. This script **does not retrain anything** — it only reads those saved files.

- `y_test` — the actual ground-truth labels (0 = LOS, 1 = NLOS) from the held-out test set.
- `y_pred` — the hard class predictions (0 or 1) made by each model.
- `y_proba` — the predicted probability that a sample is NLOS (a value between 0 and 1), needed for ROC curves and AUC scores.

---

### Step 2 — Build a Unified Metrics Table

For every model the script computes:

| Metric | Formula / Meaning |
|---|---|
| **Accuracy** | (Correct predictions) / (Total predictions) |
| **Precision** | Of everything predicted NLOS, how many were really NLOS? TP / (TP + FP) |
| **Recall** | Of all actual NLOS samples, how many were caught? TP / (TP + FN) |
| **F1-Score** | Harmonic mean of Precision and Recall — good overall balance metric |
| **ROC-AUC** | Area under the ROC curve — measures ranking ability (0.5 = random, 1.0 = perfect) |
| **NLOS Recall** | Same as Recall but explicitly labelled — how well does the model detect NLOS? |
| **NLOS FalseNeg** | Count of NLOS samples wrongly predicted as LOS — the **dangerous** error |
| **LOS FalsePos** | Count of LOS samples wrongly predicted as NLOS — the **conservative** error |

> **Why NLOS Recall matters more than raw Accuracy:**  
> In a real UWB positioning system, failing to detect an NLOS signal causes the distance estimate to be too long, which directly misleads localization. Missing an NLOS (False Negative) is far more harmful than being over-cautious about LOS (False Positive). That is why NLOS Recall is highlighted as the primary safety metric.

The table is saved as `metrics_comparison.csv` for use in the written report.

---

### Step 3 — Metrics Bar Chart (`metrics_bar_comparison.png`)

All four models are plotted side-by-side for five metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

- Each cluster of 4 bars represents one metric.
- Each bar colour represents one model (consistent colours throughout all plots).
- The exact score is printed above every bar.

**Use this plot to:** quickly see which model wins on which metric and whether any single model dominates across all metrics.

---

### Step 4 — All Confusion Matrices (`confusion_matrices_all.png`)

A confusion matrix breaks down predictions into four cells:

```
                Predicted LOS    Predicted NLOS
Actual LOS    [ True Negative    False Positive ]
Actual NLOS   [ False Negative   True Positive  ]
```

- **True Negative (TN):** Correctly said LOS → good.
- **True Positive (TP):** Correctly said NLOS → good.
- **False Positive (FP):** Said NLOS but was LOS → over-cautious, wastes positioning resources but not dangerous.
- **False Negative (FN):** Said LOS but was NLOS → ⚠ **dangerous** — the model let a corrupted signal through.

Each cell also shows its percentage of the total test set. The FN cell is labelled "⚠ Dangerous" so the most critical error type is immediately visible.

All four model matrices are placed in a 2×2 grid in one figure so they can be compared at a glance.

---

### Step 5 — Combined ROC Curve (`roc_all_models_clean.png`)

A **Receiver Operating Characteristic (ROC) curve** plots:
- **X-axis:** False Positive Rate (FPR) — how often LOS is wrongly called NLOS.
- **Y-axis:** True Positive Rate (TPR) / Recall — how often NLOS is correctly caught.

The diagonal dashed line represents a random classifier (AUC = 0.5). A model that hugs the top-left corner has a high AUC and is much better than random.

All four models are drawn on the same axes so you can see which curve is furthest to the top-left — that is the best model at ranking NLOS vs LOS probability.

> **AUC (Area Under the Curve)** is a single number summary: 1.0 is perfect, 0.5 is no better than a coin flip.

---

### Step 6 — Feature Importance: RF vs XGBoost (`feature_importance_rf_vs_xgb.png`)

Both Random Forest and XGBoost are **tree-based** models. During training, every time a feature is used to split a tree node, it reduces impurity (uncertainty). The more total impurity reduction a feature causes across all trees, the higher its importance score.

This plot takes the top 15 features by Random Forest ranking and shows both models' importance scores side-by-side.

- XGBoost scores are **normalised to sum to 1** so they are on the same visual scale as RF.
- Features with high importance in both models are reliably important — this is called **agreement** between models.
- CIR index features (e.g., `CIR_PEAK_IDX`, `FP_IDX`) and amplitude features (e.g., `FP_AMP1`, `PEAK_AMP`) typically rank highly because they capture the physical timing and strength of the first-arriving path.

---

### Step 7 — Feature Coefficients: LR vs SVM (`feature_importance_lr_vs_svm.png`)

Logistic Regression and SVM (LinearSVC) are **linear models** — they assign a weight (coefficient) to each feature.

- A **positive coefficient** pushes the prediction towards NLOS.
- A **negative coefficient** pushes the prediction towards LOS.
- The **magnitude** (absolute value) shows how strongly that feature influences the decision.

This plot shows the raw coefficient values (not absolute), so you can see both the direction and strength. Features where both models agree on sign and magnitude are the most trustworthy linear separators between LOS and NLOS.

---

### Step 8 — NLOS Safety Comparison (`nlos_safety_comparison.png`)

This figure has two panels specifically focused on **real-world positioning safety**:

1. **NLOS Recall per model** (higher is safer) — What fraction of actual NLOS signals does each model catch?
2. **NLOS misclassified as LOS count** (lower is safer) — Absolute count of dangerous errors in the test set.

These two panels tell the same story from different angles: a model with high NLOS Recall will also have a low dangerous-error count.

---

### Step 9 — Final Summary Table (console output)

After all plots are saved the script prints the full metrics table to the terminal and identifies:
- **Best F1-Score model** — best overall balance of precision and recall.
- **Best ROC-AUC model** — best at ranking / probability calibration.
- **Highest NLOS Recall model** — safest for real-world deployment.
- **Fewest dangerous errors model** — fewest NLOS-as-LOS mistakes.

---

### Step 10 — Pair-Level Classifier Comparison (if available)

If the pair classifier has been trained (`models/pair_classifier/` files exist), three additional plots are generated:

#### What is pair-level classification?
Instead of classifying each CIR measurement independently as LOS or NLOS, the pair classifier looks at **both dominant paths together** and predicts the type of path pair:
- **LOS + NLOS** — first path is LOS, second is NLOS.
- **NLOS + NLOS** — both paths are NLOS (no LOS path exists in the measurement).

This is closer to the actual physical question: "Does a line-of-sight path exist at all in this measurement?"

#### 10a — Pair Confusion Matrices (`pair_confusion_matrices.png`)
Same layout as the per-path confusion matrices but with class labels `LOS+NLOS` and `NLOS+NLOS`.

#### 10b — Per-Path vs Pair Accuracy Bar Chart (`per_path_vs_pair_comparison.png`)
Compares RF and XGBoost in both formulations (per-path vs pair-level) on Accuracy, F1-Score, and ROC-AUC. This answers the question: *"Does framing it as a pair problem help or hurt performance?"*

#### 10c — Combined ROC with Pair Models (`roc_per_path_vs_pair.png`)
- Solid lines = per-path classifiers.
- Dashed lines = pair-level classifiers.
All plotted on the same axes so the gain (or loss) from pair framing is visible.

---

## Output File Summary

| File | What It Shows |
|---|---|
| `metrics_comparison.csv` | Full numeric metrics table — paste into report |
| `metrics_bar_comparison.png` | All metrics side-by-side for all models |
| `confusion_matrices_all.png` | 2×2 grid of confusion matrices |
| `roc_all_models_clean.png` | Combined ROC curve for all 4 models |
| `feature_importance_rf_vs_xgb.png` | Tree model feature importance comparison |
| `feature_importance_lr_vs_svm.png` | Linear model coefficient comparison |
| `nlos_safety_comparison.png` | NLOS recall and dangerous error count |
| `pair_confusion_matrices.png` | Pair-level classifier confusion matrices |
| `per_path_vs_pair_comparison.png` | Per-path vs pair performance comparison |
| `roc_per_path_vs_pair.png` | Combined ROC for all models including pair |
| `per_path_vs_pair_comparison.csv` | Numeric table for per-path vs pair comparison |

---

## Key Concepts Recap

### Why Use Multiple Models?
No single algorithm is universally best. By training four different classifiers and comparing them on the same test set, you can:
- Identify which model generalises best.
- Spot if one model overfits (high training accuracy, lower test accuracy).
- Choose the right trade-off between precision and recall for your use case.

### Why Save Predictions as `.npy` Files?
Training large models (especially Random Forest with many trees) is slow. Saving predictions lets you re-run analysis and generate new plots without re-training, which is much faster during report writing.

### Why Normalise XGBoost Importance?
XGBoost's raw importance values can be on a different scale than Random Forest's. Normalising to sum to 1 makes both models comparable on the same Y-axis without distorting which features are relatively more or less important within each model.

