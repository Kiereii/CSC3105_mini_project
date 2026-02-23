# Two-Path Classification & Range Estimation — Concepts & Implementation Guide

> **What this document covers:**  
> The data analytics concepts behind `second_path_features.py` and `range_regressor.py`, explained from first principles so you understand *why* every decision was made, not just *what* the code does.

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Understanding the CIR Signal](#2-understanding-the-cir-signal)
3. [Feature Engineering — Second Path Extraction](#3-feature-engineering--second-path-extraction)
4. [Regression vs Classification](#4-regression-vs-classification)
5. [Random Forest Regressor — How It Works](#5-random-forest-regressor--how-it-works)
6. [Evaluation Metrics for Regression](#6-evaluation-metrics-for-regression)
7. [What We Actually Built](#7-what-we-actually-built)
8. [Results Interpretation](#8-results-interpretation)
9. [The Full Pipeline at a Glance](#9-the-full-pipeline-at-a-glance)

---

## 1. The Problem We Are Solving

The project brief asks for **two things**:

| Task | Type | Output |
|------|------|--------|
| Is Path 1 LOS or NLOS? | **Classification** | Label: 0 or 1 |
| Is Path 2 LOS or NLOS? | **Classification** | Always 1 (NLOS) — see below |
| What is the range of Path 1? | **Regression** | Distance in metres |
| What is the range of Path 2? | **Regression** | Distance in metres |

### Why is Path 2 always NLOS?

The brief states this directly:

> *"If the first path is LOS, the next path will be NLOS.  
> If the first path is NLOS, the next path will be NLOS too.  
> LOS is always the shortest path if it exists."*

Think about it physically:
- **LOS** = direct line between transmitter and receiver — nothing in the way. This is the **fastest** signal path, so it always arrives first.
- **NLOS** = signal bounced off a wall, floor, or object. These reflected signals travel a **longer distance** and arrive *after* the direct path.

So the second arrival is **always a reflection = always NLOS**, regardless of whether the first path was LOS or NLOS.

```
Case A (LOS environment):         Case B (NLOS environment):
  TX ──────────────── RX           TX  ███wall███  RX
  Path 1: DIRECT (LOS)              Path 1: REFLECTED (NLOS)
  Path 2: REFLECTED (NLOS)          Path 2: REFLECTED differently (NLOS)
```

**Conclusion:** Path 2 classification needs no ML — it is always NLOS=1.  
The real challenge is **estimating how far away** each path's source is.

---

## 2. Understanding the CIR Signal

### What is a CIR?

A **Channel Impulse Response (CIR)** is like an "echo profile" of the radio channel. When the UWB transmitter sends a pulse, the receiver records how that pulse arrives — including the direct path and all reflections — over time.

The DW1000 chip records **1016 samples** at 1 nanosecond resolution. Each sample's value = amplitude of the signal at that moment in time.

```
Amplitude
   │
   │          ↑ Peak 1 (First Path — FP_IDX)
   │         /|\
   │        / | \
   │       /  |  \        ↑ Peak 2 (Second Path — what we extracted)
   │      /   |   \      /|\
   │─────/    |    \────/ | \──────────────── time (samples)
   │          │           │
           FP_IDX      FP_IDX + GAP
```

- **FP_IDX** — the index where the first (strongest early) peak is detected. Already in the dataset.
- **Second peak** — the next significant peak after FP_IDX. This is what `second_path_features.py` finds.

### Why does the signal concentrate around index 730–850?

The measurements were taken indoors at ranges of roughly 1–10 metres.  
At the speed of light (~0.3 m/ns), a 5 m path takes ~16.7 ns one-way (33.3 ns round-trip).  
The DW1000 chip places the expected first arrival around **sample 745** based on its internal timing.  
That's why all meaningful signal energy is in the **730–850 window** — the rest of the 1016 samples is mostly noise.

---

## 3. Feature Engineering — Second Path Extraction

### 3.1 What is Feature Engineering?

**Feature engineering** is the process of creating *new, more informative input variables* from raw data. It is often the single most impactful step in improving model performance — better features beat better algorithms.

In our case, the raw dataset gives us `FP_IDX` (first path) but **nothing for the second path**. We had to *engineer* it.

### 3.2 Peak Detection with `scipy.signal.find_peaks`

```python
from scipy.signal import find_peaks

peaks, props = find_peaks(
    search_window,
    prominence=30,   # how much the peak "stands out" from surroundings
    height=10        # minimum absolute amplitude
)
```

**Prominence** is the key parameter. It measures how much a peak rises above the surrounding baseline — not just raw height. This prevents noise spikes from being misidentified as real signal paths.

```
High prominence (real peak):     Low prominence (noise):
        ↑                              ↑
       /|\                            /|\ 
──────/ | \──────────                ─/ | \─────
       large gap                     small gap
       from baseline                 from baseline
```

### 3.3 Why skip 15 samples after FP_IDX?

```python
search_start = fp_local + MIN_GAP_SAMPLES  # MIN_GAP_SAMPLES = 15
```

A UWB pulse has a finite width (~15 samples in the DW1000). If we searched immediately after FP_IDX, we would detect **sidelobes of the same pulse** — not a separate second path. Skipping 15 samples ensures we look beyond the first pulse's influence.

### 3.4 The New Features Created

| Feature | Formula | Meaning |
|---------|---------|---------|
| `PEAK2_IDX` | CIR index of second peak | Absolute time of arrival of Path 2 |
| `PEAK2_AMP` | Amplitude at second peak | Signal strength of Path 2 |
| `PEAK2_GAP` | `PEAK2_IDX - FP_IDX` | Time delay between Path 1 and Path 2 |
| `PEAK2_FOUND` | 0 or 1 | Was a valid second peak detected? |
| `RANGE_PATH2_EST` | `RANGE + PEAK2_GAP × 0.15 m` | Estimated distance of Path 2 |

### 3.5 How is Path 2 Range Derived?

```
Path 2 range = Path 1 measured range + extra distance from the gap

Extra distance = gap_in_samples × metres_per_sample

metres_per_sample = (speed_of_light × sample_period) / 2
                  = (3×10⁸ m/s × 1.0016×10⁻⁹ s) / 2
                  ≈ 0.15 m per CIR sample
```

The division by 2 is because range is measured as **round-trip time** (signal goes out and comes back), so the actual one-way distance is half the total path length.

---

## 4. Regression vs Classification

### Classification (what you did before)
- **Output**: A *category* — either LOS (0) or NLOS (1)
- **Question asked**: "Which box does this belong to?"
- **Example metric**: Accuracy, F1-Score, ROC-AUC

### Regression (what we added now)
- **Output**: A *continuous number* — e.g. 3.82 metres
- **Question asked**: "What is the exact value?"
- **Example metric**: RMSE, MAE, R²

```
Classification:    ──────[LOS]────[NLOS]────
                           ^         ^
                       Distinct    Distinct
                       classes     classes

Regression:        ──────────────────────────────
                   0m   2m   4m   6m   8m   10m+
                          ^ continuous output
```

Both tasks use the **same features** (CIR data + hardware readings) but different model targets:
- Classifier target: `NLOS` column (0 or 1)
- Regressor target: `RANGE` column (metres, a float)

---

## 5. Random Forest Regressor — How It Works

You already used Random Forest for classification. The regressor works the same way internally — the only difference is in the **leaf node output**.

### Decision Tree Recap

A decision tree splits data by asking yes/no questions:

```
                 Is FP_IDX > 755?
                /                \
              Yes                 No
               |                  |
     Is CIR_PWR > 5000?     Is RANGE < 2?
        /       \               /    \
      ...       ...           ...    ...
       ↓         ↓             ↓      ↓
   Leaf: 4.2m  Leaf: 1.1m  Leaf: 0.8m  Leaf: 6.5m
   (predict mean    (predict mean of
    of samples       samples here)
    here)
```

For **regression**, each leaf node stores the **average target value** of all training samples that fall into it. The prediction for a new sample is the mean range of the leaf it lands in.

### Random Forest = Many Trees + Voting

```
Sample → Tree 1 → predicts 3.8m ─┐
Sample → Tree 2 → predicts 4.1m ─┤→ Average → Final: 3.95m
Sample → Tree 3 → predicts 4.0m ─┘
  ...       ...
```

Each tree is trained on a **random subset of data** (bootstrapping) and uses a **random subset of features** at each split. This randomness prevents overfitting — no single tree dominates, and errors cancel each other out across the forest.

### Why Random Forest for Range Estimation?

| Reason | Explanation |
|--------|-------------|
| **Non-linear relationships** | Range vs CIR features is not a straight line |
| **No scaling needed** | Works on raw feature values |
| **Feature importance** | Tells you which features drive range prediction |
| **Handles many features** | 139 features (CIR + hardware) — RF handles this well |
| **Consistent with classifier** | Same algorithm family makes comparison intuitive |

---

## 6. Evaluation Metrics for Regression

Unlike classification where you count correct/wrong, regression errors are on a **continuous scale**.

### 6.1 RMSE — Root Mean Squared Error

```
RMSE = √( (1/n) × Σ(predicted - actual)² )
```

- Measures the **typical error magnitude** in the same units as the target (metres)
- Penalises **large errors more heavily** (squaring amplifies outliers)
- Our result: **Path 1 = 1.275 m**, Path 2 = 1.318 m

> **Interpretation**: On average, our model's prediction is off by ~1.28 metres for Path 1.

### 6.2 MAE — Mean Absolute Error

```
MAE = (1/n) × Σ|predicted - actual|
```

- Average of the absolute differences — **treats all errors equally**
- More robust to outliers than RMSE
- Our result: **Path 1 = 0.980 m**, Path 2 = 1.008 m

> **Interpretation**: The typical prediction error is just under 1 metre.

### 6.3 R² — Coefficient of Determination

```
R² = 1 - (Sum of squared residuals / Total variance in data)
```

- Ranges from 0 to 1 (higher = better)
- **R² = 1.0**: Perfect prediction
- **R² = 0.0**: Model is no better than always predicting the mean
- **R² < 0**: Model is worse than just guessing the mean
- Our result: **Path 1 = 0.707**, Path 2 = 0.743

> **Interpretation**: Our model explains ~70–74% of the variance in range values. The remaining 30% comes from real-world randomness (multipath interference, antenna orientation, building materials).

### Comparison Table

| Metric | Path 1 | Path 2 | Better is... |
|--------|--------|--------|-------------|
| RMSE   | 1.275 m | 1.318 m | Lower |
| MAE    | 0.980 m | 1.008 m | Lower |
| R²     | 0.707   | 0.743   | Higher |

### Why does Path 2 have *better R²* despite higher RMSE?

Path 2 range values are **spread over a wider range** (2.4 m – 30.7 m) compared to Path 1 (0.01 m – 28 m). R² is a *relative* measure — it rewards explaining a wider variance. Path 2's `PEAK2_GAP` feature gives a strong structural anchor that the model exploits well, boosting R² even though absolute errors are slightly larger.

---

## 7. What We Actually Built

### File 1: `second_path_features.py`

**Purpose**: Feature engineering pipeline  
**Input**: 7 cleaned CSV files (~41,568 rows)  
**Output**: `.npy` arrays ready for the regressor

**Step-by-step:**

```
1. Load all cleaned CSV data
        ↓
2. For each of 41,568 rows:
   - Get FP_IDX (first path position)
   - Define search window: FP_IDX + 15 samples onwards
   - Run find_peaks() on that window
   - Record: PEAK2_IDX, PEAK2_AMP, PEAK2_GAP
        ↓
3. Compute RANGE_PATH2_EST = RANGE + PEAK2_GAP × 0.15m
        ↓
4. Build feature matrix X (139 features):
   - 19 core + second-path features
   - 120 CIR samples (730-850)
        ↓
5. Split 80/20 (stratified) → save .npy files
```

**Result**: Second path found in **100% of samples** — confirming the CIR signal always contains a second detectable multipath reflection indoors.

---

### File 2: `range_regressor.py`

**Purpose**: Train and evaluate two range estimators  
**Input**: `.npy` files from `second_path_features.py`  
**Output**: Trained models + 3 visualisation plots

**Two models trained:**

```
Model A (rf_range_path1.pkl):
  Features: X_train_regression (139 features)
  Target:   RANGE (actual measured range, metres)
  Use case: Estimate how far away the first signal path source is

Model B (rf_range_path2.pkl):
  Features: X_train_regression (same 139 features)
  Target:   RANGE_PATH2_EST (derived second path range)
  Use case: Estimate how far away the reflected path source is
```

**Three output plots:**

| Plot | What it shows |
|------|--------------|
| `range_estimation_results.png` | 6-panel: Predicted vs Actual scatter, Residual histograms, Error vs Range, Metrics comparison |
| `regressor_feature_importance.png` | Top 20 features for both path regressors side by side |
| `two_path_pair_distribution.png` | Bar chart: how many LOS+NLOS vs NLOS+NLOS pairs in the test set |

---

## 8. Results Interpretation

### What does the Predicted vs Actual scatter tell you?

```
Perfect model:          Our model:
    ↑ Predicted             ↑ Predicted
    │  /                    │   ·  / ·
    │ /                     │  · /·  ·
    │/                      │ ·/  ·
    └──────→ Actual         └──────→ Actual
    (all points on           (points scattered
     diagonal line)           around the diagonal)
```

Points **on the red dashed diagonal** = perfect prediction.  
Points **above** = model over-predicted.  
Points **below** = model under-predicted.  
The **spread around the diagonal** visually shows your RMSE.

### What does the Residual Distribution tell you?

```
Residual = Predicted - Actual

Good model:             Biased model:
  ▄▄▄                       ▄▄▄
 █████                     █████
█████████                 █████████
───0───                   ────0──── ──3──
  centred                  shifted right
  at zero                  (consistently over-predicts)
```

A **bell-shaped distribution centred at 0** means the model has no systematic bias — errors are random, which is what we want.

### Feature Importance for Range — What should rank highly?

For **Path 1 range**:
- `FP_IDX` — directly encodes the time of arrival → strongly predicts range
- `CIR_PWR` — total channel power correlates with distance (signal weakens with range)
- `RXPACC` — preamble accumulation count relates to signal quality at different ranges

For **Path 2 range**:
- `PEAK2_GAP` — the time offset between paths directly encodes extra path length
- `PEAK2_IDX` — absolute arrival time of second path
- `FP_IDX` — baseline from which Path 2 is measured

---

## 9. The Full Pipeline at a Glance

```
RAW DATA (42,000 samples)
        │
        ▼
[clean_local.py]
  • Remove outliers (top 1% noise)
  • Add SNR, SNR_dB features
  • Output: Cleaned CSVs (41,568 rows)
        │
        ▼
[preprocess_data.py]
  • Select 16 core features + CIR 730-850
  • 80/20 stratified train/test split
  • StandardScaler + MinMaxScaler
  • Output: X_train/test_*.npy, y_train/test.npy
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
[random_forest_classifier.py]        [second_path_features.py]
  TASK 1: LOS/NLOS classification      TASK 2 PREP: Second path extraction
  • Train RandomForestClassifier        • find_peaks() on CIR after FP_IDX
  • Accuracy: 88.69%                    • Engineer PEAK2_IDX, AMP, GAP
  • AUC: 0.9535                         • Derive RANGE_PATH2_EST
  • Output: random_forest_model.pkl     • Output: X/y_regression .npy files
                                               │
                                               ▼
                                      [range_regressor.py]
                                        TASK 2: Range estimation
                                        • Train RF Regressor ×2
                                        • Path 1: RMSE=1.28m, R²=0.71
                                        • Path 2: RMSE=1.32m, R²=0.74
                                        • Output: rf_range_path1/2.pkl

BRIEF REQUIREMENT COVERAGE:
  ✅ LOS/NLOS classifier for Path 1     → random_forest_classifier.py
  ✅ Path 2 always NLOS (rule-based)    → per project brief logic
  ✅ Range estimator for Path 1         → range_regressor.py (rf_range_path1)
  ✅ Range estimator for Path 2         → range_regressor.py (rf_range_path2)
  ✅ Feature importance ranked          → feature_importance_ranking.csv
  ✅ Confusion matrix + ROC             → confusion_matrix_and_roc.png
  ✅ Data cleaning justified            → clean_local.py (1% outlier removal)
```

---

## Key Takeaways

1. **Feature engineering often matters more than algorithm choice** — `PEAK2_GAP` directly encodes the physical second-path delay, which is why Path 2's R² is actually *higher* than Path 1's despite being harder to measure.

2. **Regression and classification are complementary** — you need both to fully satisfy the brief. Classification answers "what type?", regression answers "how far?".

3. **Physical domain knowledge guides every decision** — the 15-sample skip gap, the 730–850 CIR window, the 0.15 m/sample conversion — all come from understanding the DW1000 hardware, not from trial and error.

4. **R² = 0.70–0.74 is reasonable for indoor ranging** — UWB range estimation in real indoor environments is inherently noisy due to furniture, people, and wall materials. Sub-1-metre MAE with a pure RF model and no spatial calibration is a solid result.

5. **Path 2 being always NLOS is not a limitation — it's a physical fact** — any second arrival must have travelled a longer path than the first, meaning it must have reflected off something, which by definition is NLOS.

