# Data Preparation — Cleaning, Feature Engineering & Preprocessing

This document explains every transformation applied to the raw UWB dataset before feeding it into any machine learning model.

---

## Overview

```
RAW DATA  (42,000 samples × 7 files)
    ↓  [clean_local.py]
CLEANED   (41,568 samples)  — duplicates / nulls / outliers removed, SNR added
    ↓  [preprocess_data.py]
FEATURE MATRIX  (41,568 × 136)  — 16 core features + 120 CIR samples
    ↓
SPLIT   → Train (33,254)  /  Test (8,314)   [80/20 stratified]
    ↓
SCALED  → Unscaled · StandardScaler · MinMaxScaler
    ↓
.npy arrays  ready for classifiers
```

---

## Stage 1: Data Cleaning (`clean_local.py`)

Applied identically to each of the 7 raw CSV files.

### Step 1 — Remove Exact Duplicates

```python
df = df.drop_duplicates()
```

A row is only removed if **every** column is identical. This guards against accidentally duplicated measurement records.

### Step 2 — Drop Missing Values

```python
df = df.dropna()
```

Any row containing at least one null (NaN) is dropped entirely. Null values would cause errors or silent bias in downstream calculations.

### Step 3 — Outlier & Integrity Filtering

Two rules are applied:

| Rule | Reason |
|------|--------|
| `RANGE > 0` | A measured range cannot be zero or negative — these are sensor errors |
| `STDEV_NOISE < 99th percentile` | Top 1% noise spikes are extreme outliers that skew model training |

```python
df = df[df['RANGE'] > 0]
threshold = df['STDEV_NOISE'].quantile(0.99)
df = df[df['STDEV_NOISE'] < threshold]
```

### Step 4 — Type Coercion

```python
df['NLOS'] = df['NLOS'].astype(int)
```

Ensures the target label is stored as an integer (0 = LOS, 1 = NLOS), not a float, for compatibility with scikit-learn.

### Step 5 — SNR Feature Engineering

Two new features are calculated from existing sensor readings:

```
SNR    = FP_AMP1 / (STDEV_NOISE + ε)      ← linear ratio
SNR_dB = 10 × log₁₀(SNR)                  ← decibel scale
```

```python
df['SNR']    = df['FP_AMP1'] / (df['STDEV_NOISE'] + 1e-6)
df['SNR_dB'] = 10 * np.log10(df['SNR'].clip(lower=1e-6))
```

**Why SNR matters:** LOS signals have a dominant first-path amplitude relative to noise. NLOS signals often show weak first-path peaks buried in noise. SNR directly captures this distinction.

### Cleaning Results

```
Input  (7 files combined):   42,000 samples
Rows removed:                   432 samples (1.0%)
Output (7 cleaned files):    41,568 samples

Breakdown of removed rows:
  • Exact duplicates:   small number per file
  • Null values:        0 (dataset was complete)
  • Outliers:           ~432 (noise spikes + invalid ranges)

Verdict: Excellent data quality — only 1% removed ✓
```

---

## Stage 2: Feature Selection (`preprocess_data.py`)

### Core Features (16)

These are structured sensor metadata columns:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `RANGE` | Measured distance between anchors (metres) |
| 2 | `FP_IDX` | First-path index — sample index of the first detected peak |
| 3 | `FP_AMP1` | First-path amplitude (primary peak) |
| 4 | `FP_AMP2` | Second amplitude reading |
| 5 | `FP_AMP3` | Third amplitude reading |
| 6 | `STDEV_NOISE` | Noise standard deviation |
| 7 | `CIR_PWR` | Total channel impulse response power |
| 8 | `MAX_NOISE` | Maximum noise value |
| 9 | `RXPACC` | Received preamble accumulation count — signal quality indicator |
| 10 | `CH` | UWB channel number |
| 11 | `FRAME_LEN` | Packet frame length |
| 12 | `PREAM_LEN` | Preamble length |
| 13 | `BITRATE` | Transmission bit rate |
| 14 | `PRFR` | Pulse repetition frequency |
| 15 | `SNR` | Computed signal-to-noise ratio (linear) |
| 16 | `SNR_dB` | Computed SNR in decibels |

### CIR Features (120)

Samples `CIR730` through `CIR849` — the **first-path region** of the Channel Impulse Response waveform identified during EDA.

```
Full CIR waveform: CIR0 – CIR1015  (1016 samples per reading)
Selected region:   CIR730 – CIR849 (120 samples)

Why this region?
  • EDA showed the first-path peak consistently falls here
  • LOS → single sharp peak in this window
  • NLOS → weaker, spread, or delayed peak in this window
  • Including the full CIR (1016 samples) adds noise without benefit
```

### Final Feature Matrix

```
Total features:  16 core + 120 CIR = 136 features
Samples:         41,568
Target column:   NLOS  (0 = Line-of-Sight, 1 = Non-Line-of-Sight)
```

---

## Stage 3: Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # preserves class balance in both sets
)
```

| Set | Samples | LOS | NLOS |
|-----|---------|-----|------|
| Train | 33,254 (80%) | ~50.5% | ~49.5% |
| Test  | 8,314  (20%) | ~50.5% | ~49.5% |

**Why stratified?** The dataset is nearly perfectly balanced (50.5% / 49.5%). Stratification guarantees the test set mirrors this balance exactly, preventing an unlucky split from biasing evaluation.

**Why fixed seed (42)?** Reproducibility — every team member running any script gets identical splits.

---

## Stage 4: Feature Scaling

Different algorithms require different scaling strategies. Three versions of the feature matrix are saved:

### Unscaled (raw values)

Used by **tree-based models** (Random Forest, XGBoost, Pair Classifier).

```
Why no scaling needed?
  Trees split on thresholds, not distances.
  Scaling the values does not change split decisions.
  Using raw values retains original physical meaning.
```

### StandardScaler (z-score normalisation)

Used by **SVM** and **Logistic Regression**.

$$z = \frac{x - \mu}{\sigma}$$

Result: mean = 0, standard deviation = 1 for every feature.

```python
scaler = StandardScaler()
X_train_standard = scaler.fit_transform(X_train)  # fit ONLY on train
X_test_standard  = scaler.transform(X_test)        # apply same transform
```

```
Why SVM needs this:
  SVM computes distances between points.
  A feature with range [0, 10000] would dominate one with range [0, 1].
  Scaling puts all features on equal footing.
```

### MinMaxScaler (0–1 normalisation)

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Saved for compatibility with neural networks or algorithms that expect bounded inputs.

```python
scaler = MinMaxScaler()
X_train_minmax = scaler.fit_transform(X_train)
X_test_minmax  = scaler.transform(X_test)
```

> **Critical rule:** Scalers are always **fit on training data only**, then applied to test data. Fitting on the full dataset would leak test-set statistics into training — a form of data leakage.

---

## Output Files

All files saved to `preprocessed_data/`:

```
preprocessed_data/
├── X_train_unscaled.npy    (33254, 136)   ← Random Forest, XGBoost
├── X_test_unscaled.npy     ( 8314, 136)
├── X_train_standard.npy    (33254, 136)   ← SVM, Logistic Regression
├── X_test_standard.npy     ( 8314, 136)
├── X_train_minmax.npy      (33254, 136)   ← Neural networks
├── X_test_minmax.npy       ( 8314, 136)
├── y_train.npy             (33254,)
├── y_test.npy              ( 8314,)
├── scaler_standard.pkl                    ← Reuse for new predictions
├── scaler_minmax.pkl
├── feature_names.txt
└── preprocessing_info.txt
```

### Loading in a Classifier Script

```python
import numpy as np

# Tree-based models
X_train = np.load('preprocessed_data/X_train_unscaled.npy')
X_test  = np.load('preprocessed_data/X_test_unscaled.npy')

# Distance-based models (SVM, LR)
X_train = np.load('preprocessed_data/X_train_standard.npy')
X_test  = np.load('preprocessed_data/X_test_standard.npy')

y_train = np.load('preprocessed_data/y_train.npy')
y_test  = np.load('preprocessed_data/y_test.npy')
```

---

## Summary

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| Cleaning | `clean_local.py` | 42,000 raw rows | 41,568 cleaned rows |
| Feature selection | `preprocess_data.py` | 41,568 × all columns | 41,568 × 136 |
| Train/test split | `preprocess_data.py` | 41,568 samples | 33,254 train / 8,314 test |
| Scaling | `preprocess_data.py` | raw split arrays | unscaled + standard + minmax |

Every downstream model uses the same split and the same cleaned data. The only thing that varies between algorithms is **which scaled version** they load.
