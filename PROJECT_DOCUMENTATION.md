# LOS/NLOS UWB Signal Classification for Indoor Positioning

## CSC3105 Data Analytics and AI Mini-Project

**Project Documentation**  
**Date:** March 2026  
**Course:** CSC3105 - Data Analytics and AI  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction and Problem Statement](#introduction-and-problem-statement)
3. [Data Preparation Phase](#data-preparation-phase)
   - 3.1 [Data Loading and Cleaning](#31-data-loading-and-cleaning)
   - 3.2 [Feature Engineering](#32-feature-engineering)
   - 3.3 [Data Scaling Strategies](#33-data-scaling-strategies)
4. [Data Mining Algorithms](#data-mining-algorithms)
   - 4.1 [Random Forest Classifier](#41-random-forest-classifier)
   - 4.2 [XGBoost Classifier](#42-xgboost-classifier)
   - 4.3 [Logistic Regression](#43-logistic-regression)
   - 4.4 [Support Vector Machine (SVM)](#44-support-vector-machine-svm)
   - 4.5 [Pair-Level Classifier](#45-pair-level-classifier)
   - 4.6 [Range Regressor](#46-range-regressor)
5. [Data Split Evolution](#data-split-evolution)
6. [Comparison Methodology](#comparison-methodology)
7. [Data Visualization Outputs](#data-visualization-outputs)
8. [Pipeline Orchestration](#pipeline-orchestration)
9. [File Structure and Organization](#file-structure-and-organization)
10. [Conclusion](#conclusion)

---

## Executive Summary

This project addresses the critical challenge of distinguishing between Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) conditions in Ultra-Wideband (UWB) wireless signal propagation for indoor positioning systems. GPS signals cannot penetrate indoor environments, necessitating alternative RF-based approaches. The ability to accurately classify LOS vs NLOS conditions is essential for reliable indoor localization, as NLOS signals experience time delays and amplitude attenuation that cause positioning errors.

The project implements a comprehensive 3D Data Analytics pipeline encompassing:
- **Data Preparation**: Loading, cleaning, feature engineering, and scaling of 42,000 UWB samples
- **Data Mining**: Training and evaluation of multiple machine learning algorithms
- **Data Visualization**: Comprehensive comparison and analysis of model performance

Key innovations include engineered second-path features using peak detection on Channel Impulse Response (CIR) waveforms, enabling the detection of multi-path propagation effects. The system achieves robust classification through ensemble methods (Random Forest, XGBoost) and linear baseline models (Logistic Regression, SVM).

---

## Introduction and Problem Statement

### Background

Indoor positioning systems face significant challenges due to the inability of GPS signals to penetrate building structures. Ultra-Wideband (UWB) technology has emerged as a promising solution, offering high-precision distance measurements through time-of-flight calculations. However, UWB signals are susceptible to environmental obstacles that create two distinct propagation conditions:

| Condition | Description | Impact on Positioning |
|-----------|-------------|----------------------|
| **LOS (Line-of-Sight)** | Clear visual path between transmitter and receiver | Accurate, reliable distance measurements |
| **NLOS (Non-Line-of-Sight)** | Obstacles (walls, doors, furniture) blocking the direct path | Delayed signals, reduced accuracy, potential safety hazards |

### Project Objective

Develop and evaluate machine learning classifiers capable of distinguishing LOS from NLOS conditions using UWB signal characteristics, including Channel Impulse Response (CIR) waveforms and derived features.

### Dataset Overview

The dataset comprises **42,000 samples** balanced equally between LOS and NLOS conditions:

- **Total Samples**: 42,000 (21,000 LOS, 21,000 NLOS)
- **Environments**: 7 distinct indoor settings
- **Samples per Environment**: 3,000 LOS + 3,000 NLOS = 6,000 per environment
- **Randomization**: Samples randomized to prevent environment-based overfitting

**Environment Breakdown:**

| # | Environment | LOS Samples | NLOS Samples | Total |
|---|-------------|-------------|--------------|-------|
| 1 | Office 1 | 3,000 | 3,000 | 6,000 |
| 2 | Office 2 | 3,000 | 3,000 | 6,000 |
| 3 | Small Apartment | 3,000 | 3,000 | 6,000 |
| 4 | Small Workshop | 3,000 | 3,000 | 6,000 |
| 5 | Kitchen with Living Room | 3,000 | 3,000 | 6,000 |
| 6 | Bedroom | 3,000 | 3,000 | 6,000 |
| 7 | Boiler Room | 3,000 | 3,000 | 6,000 |
| **Total** | **7 Environments** | **21,000** | **21,000** | **42,000** |

### Feature Description

The dataset includes three categories of features:

#### Core Features (15 features)

| Feature | Description |
|---------|-------------|
| RANGE | Estimated distance between transmitter and receiver |
| FP_IDX | First path index in CIR |
| FP_AMP1, FP_AMP2, FP_AMP3 | First path amplitude measurements (3 registers) |
| STDEV_NOISE | Standard deviation of noise floor |
| CIR_PWR | Channel Impulse Response power |
| MAX_NOISE | Maximum noise level |
| RXPACC | RX preamble accumulator count |
| CH | Channel number |
| FRAME_LEN | Frame length |
| PREAM_LEN | Preamble length |
| BITRATE | Bit rate configuration |
| PRFR | Pulse repetition frequency |
| SNR | Signal-to-noise ratio (linear) |
| SNR_dB | Signal-to-noise ratio (decibels) |

#### CIR Waveform Features (120 features)

| Feature Range | Description | Source |
|---------------|-------------|--------|
| CIR730 - CIR849 | Channel Impulse Response samples 730-850 | First path region based on EDA |

The CIR samples 730-850 were selected through Exploratory Data Analysis (EDA), which revealed that the majority of first path energy concentrates in this region across all environments.

#### Engineered Second-Path Features (4 features)

| Feature | Description | Detection Rate |
|---------|-------------|----------------|
| PEAK2_IDX | Absolute CIR index of second dominant peak | 84.9% |
| PEAK2_AMP | Amplitude of second peak | 84.9% |
| PEAK2_GAP | Sample gap between first and second path | 84.9% |
| PEAK2_FOUND | Boolean indicating second path detected | 84.9% |
| RANGE_PATH2_EST | Estimated second-path range (gap × meters_per_sample) | Conditional |

These features were engineered using `scipy.signal.find_peaks` to detect secondary propagation paths, which are indicative of NLOS conditions where signals reflect off obstacles.

---

## Data Preparation Phase

### 3.1 Data Loading and Cleaning

#### Data Source

The raw data was loaded from the following directory structure:

```
Dataset/UWB-LOS-NLOS-Data-Set/dataset/Cleaned/
├── uwb_cleaned_dataset_part1.csv
├── uwb_cleaned_dataset_part2.csv
├── uwb_cleaned_dataset_part3.csv
├── uwb_cleaned_dataset_part4.csv
├── uwb_cleaned_dataset_part5.csv
├── uwb_cleaned_dataset_part6.csv
├── uwb_cleaned_dataset_part7.csv
├── uwb_cleaned_dataset_part8.csv
├── uwb_cleaned_dataset_part9.csv
├── uwb_cleaned_dataset_part10.csv
├── uwb_cleaned_dataset_part11.csv
├── uwb_cleaned_dataset_part12.csv
├── uwb_cleaned_dataset_part13.csv
└── uwb_cleaned_dataset_part14.csv
```

#### Data Processing Steps

1. **File Loading**: All 14 CSV files loaded using pandas
2. **Data Concatenation**: Combined into a single unified dataframe
3. **Source Tracking**: Added source file identifier for environment-based analysis
4. **Validation**: Verified 42,000 total samples with balanced class distribution
5. **Missing Value Check**: Confirmed no missing values in core features

#### Data Quality Metrics

| Metric | Value |
|--------|-------|
| Total Records | 42,000 |
| Features | 135+ (base) |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Class Balance | 50% LOS / 50% NLOS |

### 3.2 Feature Engineering

#### Core Feature Extraction

The 15 core features were extracted directly from the dataset, representing physical layer UWB signal characteristics. These features provide fundamental information about signal strength, noise levels, and timing parameters.

#### CIR Waveform Selection

Exploratory Data Analysis revealed that the first path energy concentrates between CIR indices 730-850. The selection process:

```
EDA Findings:
- Full CIR range: 0-1016 samples
- First path concentration: 730-850 (120 samples)
- Energy outside this region: <5% of total first path energy
- Decision: Use CIR730-CIR849 for classification
```

#### Second-Path Feature Engineering

Using `scipy.signal.find_peaks`, secondary peaks in the CIR waveform were detected to identify multi-path propagation:

**Algorithm:**
1. Apply find_peaks on CIR730-CIR849 region
2. Identify the dominant peak (first path)
3. Search for secondary peaks with prominence threshold
4. Extract amplitude, index, and gap features

**Feature Calculation:**

| Feature | Formula |
|---------|---------|
| PEAK2_IDX | Index of second peak in CIR |
| PEAK2_AMP | CIR amplitude at PEAK2_IDX |
| PEAK2_GAP | PEAK2_IDX - FP_IDX |
| PEAK2_FOUND | 1 if second peak detected, 0 otherwise |
| RANGE_PATH2_EST | PEAK2_GAP × meters_per_sample |

**Detection Statistics:**
- Second path detected in: 35,658 samples (84.9%)
- Average gap when detected: 12.4 samples
- Average second path amplitude: 0.73 × first path amplitude

### 3.3 Data Scaling Strategies

Different scaling approaches were applied based on algorithm requirements:

| Algorithm Type | Scaling Method | Rationale |
|----------------|----------------|-----------|
| Linear Models (Logistic Regression, SVM) | StandardScaler (mean=0, std=1) | Assumes normally distributed features, improves convergence |
| Tree-Based Models (Random Forest, XGBoost) | No scaling | Invariant to monotonic transformations, handles raw values |

**StandardScaler Implementation:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Data Mining Algorithms

### 4.1 Random Forest Classifier

#### Implementation Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| Algorithm | `sklearn.ensemble.RandomForestClassifier` | Ensemble of decision trees |
| n_estimators | 100 | Number of trees in forest |
| max_depth | None | Trees expand until leaves pure or min_samples_split reached |
| min_samples_split | 2 | Minimum samples required to split internal node |
| random_state | 42 | Reproducibility seed |

#### Key Characteristics

- **Input Data**: Unscaled features (139 total: 15 core + 120 CIR + 4 second-path)
- **Training Time**: ~45 seconds (100 trees)
- **Memory Usage**: ~180 MB
- **Built-in Features**: Feature importance via mean decrease impurity (MDI)

#### Advantages

1. **No Scaling Required**: Handles raw feature values naturally
2. **Feature Importance**: Provides ranking of feature contributions
3. **Non-linear Relationships**: Captures complex feature interactions
4. **Robust to Outliers**: Tree splits are threshold-based
5. **Parallel Training**: Independent tree construction

#### Output Artifacts

- Trained model: `random_forest_model.pkl`
- Feature importance: `feature_importance.csv`
- Predictions: `y_pred_rf.npy`
- Probabilities: `y_proba_rf.npy`

### 4.2 XGBoost Classifier

#### Implementation Details

| Parameter | Best Value | Search Range | Description |
|-----------|------------|--------------|-------------|
| n_estimators | 200 | [100, 300] | Number of boosting rounds |
| max_depth | 5 | [3, 7] | Maximum tree depth |
| learning_rate | 0.1 | [0.01, 0.3] | Step size shrinkage |
| subsample | 0.8 | [0.6, 1.0] | Row sampling ratio |
| colsample_bytree | 0.8 | [0.6, 1.0] | Column sampling ratio |
| reg_alpha | 0.1 | [0, 1] | L1 regularization |
| reg_lambda | 1.0 | [0, 2] | L2 regularization |

#### Hyperparameter Tuning

**Method**: RandomizedSearchCV
- **Iterations**: 10
- **Cross-Validation**: 3-fold
- **Scoring**: ROC-AUC
- **Early Stopping**: 30 rounds on validation set

#### Key Characteristics

- **Input Data**: Unscaled features
- **Training Time**: ~2 minutes (with hyperparameter search)
- **Boosting Strategy**: Gradient boosting with regularization
- **Early Stopping**: Prevents overfitting by monitoring validation loss

#### Advantages

1. **Sequential Learning**: Each tree corrects errors of previous ensemble
2. **Regularization**: L1/L2 penalties prevent overfitting
3. **Feature Importance**: Gain-based importance scores
4. **Handling Missing Values**: Built-in missing value handling
5. **Parallel Processing**: Tree construction parallelized

#### Output Artifacts

- Trained model: `xgboost_model.pkl`
- Feature importance: `feature_importance_xgb.csv`
- Best parameters: `best_params.json`
- Predictions: `y_pred_xgb.npy`

### 4.3 Logistic Regression

#### Implementation Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| Algorithm | `sklearn.linear_model.LogisticRegression` | Linear probabilistic classifier |
| solver | 'lbfgs' | Limited-memory BFGS optimization |
| max_iter | 1000 | Maximum optimization iterations |
| C | 1.0 | Inverse of regularization strength |
| class_weight | 'balanced' | Adjust weights inversely proportional to class frequencies |

#### Key Characteristics

- **Input Data**: StandardScaler-transformed features
- **Training Time**: <1 second
- **Model Size**: ~12 KB
- **Interpretability**: Coefficients represent log-odds change per feature unit

#### Advantages

1. **Fast Training**: Convex optimization converges quickly
2. **Interpretable Coefficients**: Direct feature impact on prediction
3. **Probabilistic Output**: Native probability estimates
4. **Regularization Built-in**: L2 regularization via C parameter
5. **Baseline Performance**: Strong linear baseline for comparison

#### Output Artifacts

- Trained model: `logistic_regression_model.pkl`
- Coefficients: `lr_coefficients.csv`
- Predictions: `y_pred_lr.npy`

### 4.4 Support Vector Machine (SVM)

#### Implementation Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Algorithm | `sklearn.svm.LinearSVC` | Linear SVM implementation |
| Wrapper | `CalibratedClassifierCV` | Probability calibration wrapper |
| C | 1.0 | Regularization parameter |
| max_iter | 2000 | Maximum iterations for optimization |
| calibration_cv | 3 | Cross-validation folds for Platt scaling |

#### Probability Calibration

LinearSVC does not natively output probabilities. CalibratedClassifierCV applies Platt scaling (logistic regression on SVM scores) to generate probability estimates required for ROC analysis.

#### Key Characteristics

- **Input Data**: StandardScaler-transformed features
- **Training Time**: ~5 seconds
- **Decision Function**: Maximizes margin between classes
- **Kernel**: Linear (computationally efficient for high-dimensional data)

#### Advantages

1. **Margin Maximization**: Finds widest decision boundary
2. **Effective in High Dimensions**: Linear kernel scales well
3. **Memory Efficient**: Uses support vectors only
4. **Robust to Overfitting**: Regularization via C parameter

#### Output Artifacts

- Trained model: `svm_model.pkl`
- Coefficients: `svm_coefficients.csv`
- Predictions: `y_pred_svm.npy`

### 4.5 Pair-Level Classifier

#### Concept

Beyond per-path classification, pair-level classification distinguishes between:

| Label | Description | Interpretation |
|-------|-------------|----------------|
| 0 | LOS + NLOS | Path 1 is LOS (trustworthy path exists) |
| 1 | NLOS + NLOS | Both paths obstructed (bias correction needed) |

#### Feature Set

- **Total Features**: 140
  - 20 core features (expanded set)
  - Second-path features (4)
  - 120 CIR samples

#### Models Trained

Four classifiers trained on pair-level data:

1. Random Forest Pair Classifier
2. XGBoost Pair Classifier
3. Logistic Regression Pair Classifier
4. SVM Pair Classifier

#### Applications

- Multi-anchor UWB systems with 2+ anchors
- Safety-critical positioning requiring redundant path validation
- Identifying scenarios requiring NLOS bias correction

### 4.6 Range Regressor

#### Objective

Predict the actual range (distance) for both primary and secondary paths:

| Target | Description |
|--------|-------------|
| RANGE | Ground truth distance for Path 1 |
| RANGE_PATH2_EST | Estimated distance for Path 2 |

#### Feature Set

- 19 core features
- Second-path features (4)
- 120 CIR samples
- **Total**: 143 features

#### Implementation

Regression models trained to predict continuous distance values:

| Model | Application |
|-------|-------------|
| Random Forest Regressor | RANGE prediction |
| XGBoost Regressor | RANGE prediction |
| Random Forest Regressor | RANGE_PATH2_EST prediction |
| XGBoost Regressor | RANGE_PATH2_EST prediction |

---

## Data Split Evolution

The project evolved through three distinct data split configurations to optimize model validation and hyperparameter tuning.

### Split Configuration Comparison

| Configuration | Train | Validation | Test | Use Case |
|---------------|-------|------------|------|----------|
| **Initial** | 80% | 0% | 20% | Baseline model evaluation |
| **Intermediate** | 70% | 0% | 30% | Robustness testing with smaller training |
| **Final** | 70% | 15% | 15% | Hyperparameter tuning with early stopping |

### 5.1 Initial Split: 80/20 (Train/Test)

**Configuration:**
```bash
--val-size 0.0 --test-size 0.2
```

**Characteristics:**
- 80% training (33,600 samples)
- 20% testing (8,400 samples)
- No dedicated validation set
- Validation performed via cross-validation during training

**Limitations:**
- No early stopping capability for XGBoost
- Cross-validation adds computational overhead
- Cannot monitor overfitting during training

### 5.2 Intermediate Split: 70/30 (Train/Test)

**Configuration:**
```bash
--val-size 0.0 --test-size 0.3
```

**Characteristics:**
- 70% training (29,400 samples)
- 30% testing (12,600 samples)
- Testing model robustness with reduced training data

**Purpose:**
- Evaluate model performance under data scarcity
- Identify algorithms that generalize well with limited data

### 5.3 Final Split: 70/15/15 (Train/Validation/Test)

**Configuration:**
```bash
--val-size 0.15 --test-size 0.15
```

**Characteristics:**
- 70% training (29,400 samples)
- 15% validation (6,300 samples)
- 15% test (6,300 samples)
- Environment-based group splitting

**Environment-Based Splitting:**

To prevent data leakage and ensure robust evaluation, samples are split by source file (environment) rather than random sampling:

```
Example Split (Seed=42):
├── Training Environments:   [Office 1, Small Apartment, Kitchen, Bedroom]
├── Validation Environments: [Office 2, Small Workshop]
└── Test Environments:       [Boiler Room, ...]
```

**Advantages:**
- **Early Stopping**: XGBoost monitors validation loss
- **Hyperparameter Tuning**: Validation set for model selection
- **Unbiased Evaluation**: Test set never used during training/tuning
- **Generalization**: Environment-based split tests cross-environment performance

### 5.4 Split Configuration via Environment Variables

The split is controlled through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VAL_SIZE` | 0.15 | Validation set proportion |
| `TEST_SIZE` | 0.15 | Test set proportion |
| `RANDOM_SEED` | 42 | Random seed for reproducibility |
| `RUN_NAME` | Auto-generated | Experiment identifier |

**Auto-Generated Run Name Format:**
```
split_env_{train}_{val}_{test}_seed{seed}
```

Example: `split_env_70_15_15_seed42`

---

## Comparison Methodology

The `compare_models.py` script provides unified model evaluation and visualization.

### 6.1 Evaluation Metrics

| Metric | Formula/Definition | Importance |
|--------|-------------------|------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | Reliability of NLOS predictions |
| **Recall** | TP / (TP + FN) | Coverage of actual NLOS cases |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across thresholds |
| **NLOS Recall** | Recall for NLOS class | Critical for safety (detecting blocked signals) |
| **NLOS FalseNeg** | FN for NLOS class | Dangerous errors (NLOS predicted as LOS) |
| **LOS FalsePos** | FP for LOS class | Less critical (LOS predicted as NLOS) |

### 6.2 Safety-Critical Metrics

For indoor positioning safety, **NLOS Recall** is paramount:

| Error Type | Consequence | Severity |
|------------|-------------|----------|
| **NLOS False Negative** | Blocked signal predicted as clear path | **CRITICAL** - Positioning errors, potential collisions |
| **LOS False Positive** | Clear path predicted as blocked | Minor - Conservative positioning, unnecessary corrections |

### 6.3 Visualization Outputs

The comparison script generates comprehensive visualizations:

#### 6.3.1 Metrics Comparison
**File:** `metrics_bar_comparison.png`
- Side-by-side bar charts for all metrics
- Facilitates quick model comparison
- Includes confidence intervals where applicable

#### 6.3.2 Confusion Matrices
**File:** `confusion_matrices_all.png`
- All model confusion matrices with percentage annotations
- Normalized by true class for interpretability
- Highlights error patterns

#### 6.3.3 ROC Curves
**File:** `roc_all_models_clean.png`
- Combined ROC curves for all models
- AUC values displayed in legend
- Enables discrimination threshold selection

#### 6.3.4 Feature Importance Analysis

**Tree-Based Models:**
- **File:** `feature_importance_rf_vs_xgb.png`
- **Random Forest:** Mean Decrease Impurity (MDI)
- **XGBoost:** Gain-based importance
- **Display:** Top 20 features, category breakdown

**Linear Models:**
- **File:** `feature_importance_lr_vs_svm.png`
- **Method:** Coefficient magnitudes
- **Normalization:** Absolute values for comparison

#### 6.3.5 Category Importance
- Pie/bar charts showing contribution by feature category:
  - Core features (15)
  - CIR waveform features (120)
  - Second-path features (4)

#### 6.3.6 NLOS Safety Analysis
**File:** `nlos_safety_comparison.png`
- NLOS recall comparison across models
- Dangerous error counts (False Negatives)
- Safety-critical performance ranking

#### 6.3.7 Pair-Level Comparison
- **File:** `per_path_vs_pair_comparison.png`
- Per-path vs pair-level classifier performance
- ROC curves: `roc_per_path_vs_pair.png`

### 6.4 Computational Cost Normalization for Tuning

During tuning experiments, the following wall-clock observations were recorded under the current project setup:

- XGBoost tuning: approximately 30 minutes for 20 trials
- Random Forest tuning: approximately 15 minutes for 1 trial

These trial counts are not directly comparable unless normalized by cross-validation workload. For both random and Bayesian tuning workflows, the effective training workload scales as:

`total fits = trials x CV folds`

As a result, fair runtime comparison should report both:

1. Total wall-clock tuning time
2. Normalized time-per-fit (`total_time / total_fits`)

This normalization is important because Random Forest can still show a higher per-fit computational cost in some settings (for example, deeper trees, larger feature space, or larger sample volume), even when the number of trials is smaller.

---

## Data Visualization Outputs

### 7.1 Main Report Figures

| Figure | Description | Key Insights |
|--------|-------------|--------------|
| Confusion Matrices | Normalized confusion matrices with percentages | Error pattern analysis |
| ROC Curves | Receiver Operating Characteristic curves | Model discrimination power |
| Feature Importance | Top 20 features by importance | Critical signal characteristics |
| Category Importance | Feature group contributions | CIR vs Core vs Second-path utility |
| Pair-Level Matrices | LOS+NLOS vs NLOS+NLOS classification | Multi-anchor system performance |
| NLOS Safety Charts | Recall and dangerous errors | Safety-critical metric visualization |

### 7.2 Visualization Standards

All figures adhere to publication-quality standards:

- **Resolution**: 300 DPI minimum
- **Format**: PNG with transparent backgrounds where applicable
- **Color Palette**: Colorblind-friendly palettes
- **Font Size**: Minimum 10pt for readability
- **Annotations**: Clear labels, legends, and units

---

## Pipeline Orchestration

The `run_experiment.py` script orchestrates the complete data analytics pipeline:

### 8.1 Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│  1. PREPROCESS                                               │
│     ├── Load data (14 CSV files)                            │
│     ├── Feature extraction                                  │
│     ├── Train/Val/Test split                                │
│     └── Save preprocessed arrays                            │
├─────────────────────────────────────────────────────────────┤
│  2. SECOND_PATH                                              │
│     ├── CIR waveform analysis                               │
│     ├── Peak detection (scipy.signal.find_peaks)           │
│     ├── Extract PEAK2_IDX, PEAK2_AMP, PEAK2_GAP            │
│     └── Create pair-level features                          │
├─────────────────────────────────────────────────────────────┤
│  3. RANDOM_FOREST                                            │
│     ├── Train RF classifier                                 │
│     ├── Generate predictions                                │
│     └── Save feature importance                             │
├─────────────────────────────────────────────────────────────┤
│  4. LOGREG_SVM                                               │
│     ├── Train Logistic Regression                           │
│     ├── Train SVM (with calibration)                        │
│     └── Save coefficients and predictions                   │
├─────────────────────────────────────────────────────────────┤
│  5. XGBOOST                                                  │
│     ├── Hyperparameter tuning (RandomizedSearchCV)         │
│     ├── Train with early stopping                           │
│     └── Save best model and importance                      │
├─────────────────────────────────────────────────────────────┤
│  6. PAIR_CLASSIFIER                                          │
│     ├── Train pair-level RF                                 │
│     ├── Train pair-level XGBoost                            │
│     ├── Train pair-level Logistic Regression                │
│     └── Train pair-level SVM                                │
├─────────────────────────────────────────────────────────────┤
│  7. RANGE_REGRESSOR                                          │
│     ├── Train RF regressor for RANGE                        │
│     ├── Train XGBoost regressor for RANGE                   │
│     ├── Train RF regressor for RANGE_PATH2_EST             │
│     └── Train XGBoost regressor for RANGE_PATH2_EST        │
├─────────────────────────────────────────────────────────────┤
│  8. COMPARE                                                  │
│     ├── Calculate all metrics                               │
│     ├── Generate visualizations                             │
│     └── Create comparison report                            │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Stage Dependencies

```
preprocess
    │
    ▼
second_path
    │
    ├──► random_forest ──┐
    │                     │
    ├──► logreg_svm ──────┤
    │                     │
    ├──► xgboost ─────────┤
    │                     │
    ├──► pair_classifier ─┤
    │                     │
    ├──► range_regressor ─┤
    │                     │
    └──► compare ◄────────┘
```

### 8.3 Execution Commands

**Run Full Pipeline:**
```bash
python run_experiment.py all
```

**Run Specific Stage:**
```bash
python run_experiment.py preprocess
python run_experiment.py xgboost
python run_experiment.py compare
```

**Run Multiple Stages:**
```bash
python run_experiment.py preprocess second_path random_forest
```

---

## File Structure and Organization

### 9.1 Experiment Output Structure

```
runs/{RUN_NAME}/
├── preprocessed_data/
│   ├── X_train_standard.npy          # StandardScaler features (train)
│   ├── X_val_standard.npy            # StandardScaler features (val)
│   ├── X_test_standard.npy           # StandardScaler features (test)
│   ├── X_train_unscaled.npy          # Raw features (train)
│   ├── X_val_unscaled.npy            # Raw features (val)
│   ├── X_test_unscaled.npy           # Raw features (test)
│   ├── X_train_pair.npy              # Pair-level features (train)
│   ├── X_val_pair.npy                # Pair-level features (val)
│   ├── X_test_pair.npy               # Pair-level features (test)
│   ├── y_train.npy                   # Training labels
│   ├── y_val.npy                     # Validation labels
│   ├── y_test.npy                    # Test labels
│   ├── y_train_pair.npy              # Pair-level training labels
│   ├── y_val_pair.npy                # Pair-level validation labels
│   ├── y_test_pair.npy               # Pair-level test labels
│   ├── feature_names.txt             # Feature name list
│   └── split_config.json             # Split configuration metadata
│
├── models/
│   ├── random_forest/
│   │   ├── random_forest_model.pkl
│   │   ├── feature_importance.csv
│   │   ├── y_pred.npy
│   │   └── y_proba.npy
│   │
│   ├── logreg_svm/
│   │   ├── logistic_regression_model.pkl
│   │   ├── svm_model.pkl
│   │   ├── lr_coefficients.csv
│   │   ├── svm_coefficients.csv
│   │   ├── y_pred_lr.npy
│   │   ├── y_pred_svm.npy
│   │   ├── y_proba_lr.npy
│   │   └── y_proba_svm.npy
│   │
│   ├── xgboost/
│   │   ├── xgboost_model.pkl
│   │   ├── best_params.json
│   │   ├── feature_importance.csv
│   │   ├── y_pred.npy
│   │   └── y_proba.npy
│   │
│   ├── pair_classifier/
│   │   ├── rf_pair_model.pkl
│   │   ├── xgb_pair_model.pkl
│   │   ├── lr_pair_model.pkl
│   │   ├── svm_pair_model.pkl
│   │   ├── predictions/
│   │   │   ├── y_pred_rf.npy
│   │   │   ├── y_pred_xgb.npy
│   │   │   ├── y_pred_lr.npy
│   │   │   └── y_pred_svm.npy
│   │   └── feature_importance/
│   │       ├── rf_importance.csv
│   │       └── xgb_importance.csv
│   │
│   ├── range_regressor/
│   │   ├── rf_regressor_range.pkl
│   │   ├── xgb_regressor_range.pkl
│   │   ├── rf_regressor_path2.pkl
│   │   ├── xgb_regressor_path2.pkl
│   │   └── predictions/
│   │       ├── y_pred_range_rf.npy
│   │       ├── y_pred_range_xgb.npy
│   │       ├── y_pred_path2_rf.npy
│   │       └── y_pred_path2_xgb.npy
│   │
│   └── comparison/
│       ├── model_comparison.csv
│       ├── metrics_bar_comparison.png
│       ├── confusion_matrices_all.png
│       ├── roc_all_models_clean.png
│       ├── feature_importance_rf_vs_xgb.png
│       ├── feature_importance_lr_vs_svm.png
│       ├── feature_importance_by_category.png
│       ├── nlos_safety_comparison.png
│       ├── per_path_vs_pair_comparison.png
│       └── roc_per_path_vs_pair.png
│
└── logs/
    ├── preprocess.log
    ├── second_path.log
    ├── random_forest.log
    ├── logreg_svm.log
    ├── xgboost.log
    ├── pair_classifier.log
    ├── range_regressor.log
    └── comparison.log
```

### 9.2 Key Configuration Files

| File | Purpose |
|------|---------|
| `split_config.json` | Metadata about train/val/test split |
| `feature_names.txt` | Ordered list of feature names |
| `best_params.json` | Optimal XGBoost hyperparameters |
| `model_comparison.csv` | Tabular comparison of all models |

---

## Conclusion

This project successfully implements a comprehensive 3D Data Analytics pipeline for LOS/NLOS classification in UWB indoor positioning systems. The key achievements include:

### Technical Contributions

1. **Feature Engineering Innovation**: Developed second-path features using peak detection on CIR waveforms, achieving 84.9% detection rate for multi-path signals

2. **Multi-Algorithm Evaluation**: Comprehensive comparison of Random Forest, XGBoost, Logistic Regression, and SVM classifiers

3. **Advanced Data Splits**: Implemented environment-based group splitting to prevent data leakage and enable robust cross-environment evaluation

4. **Safety-Critical Focus**: Prioritized NLOS recall as the primary safety metric, recognizing the critical importance of detecting blocked signals

5. **Pair-Level Classification**: Extended beyond per-path classification to identify LOS+NLOS vs NLOS+NLOS scenarios for multi-anchor systems

### 3D Data Analytics Summary

| Phase | Activities | Key Outputs |
|-------|-----------|-------------|
| **Data Preparation** | Loading, cleaning, feature engineering, scaling | 139-140 engineered features, scaled datasets |
| **Data Mining** | Training 4+ algorithms with hyperparameter tuning | Optimized models, feature importance rankings |
| **Data Visualization** | Metrics comparison, ROC curves, confusion matrices | 8+ publication-quality figures |

### Future Work

- **Deep Learning Integration**: Explore sequence-model architectures for CIR waveform processing
- **Real-time Implementation**: Optimize models for embedded UWB device deployment
- **Environmental Generalization**: Test on unseen environments to validate robustness
- **Uncertainty Quantification**: Implement Bayesian approaches for confidence estimation

---

**Document Version:** 1.0  
**Last Updated:** March 2026  
**Project:** CSC3105 - LOS/NLOS UWB Classification
