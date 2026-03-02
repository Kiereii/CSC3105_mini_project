# Data Analytics & Machine Learning Guide: UWB LOS/NLOS Classification

A comprehensive guide to understanding the data analytics process, machine learning algorithms, and model evaluation using your UWB (Ultra-Wideband) indoor positioning project as a practical example.

---

## 1. Introduction to Data Analytics

### 1.1 The CRISP-DM Methodology

CRISP-DM (Cross-Industry Standard Process for Data Mining) is the standard framework for data analytics projects. It consists of 6 phases:

```
┌─────────────────────────────────────────┐
│  1. Business Understanding              │
│     ↓                                   │
│  2. Data Understanding                  │
│     ↓                                   │
│  3. Data Preparation                    │
│     ↓                                   │
│  4. Modeling                            │
│     ↓                                   │
│  5. Evaluation                          │
│     ↓                                   │
│  6. Deployment                          │
└─────────────────────────────────────────┘
```

**Your Project Application:**
- **Business Understanding**: Precise indoor localization using UWB signals
- **Data Understanding**: 41,568 samples with 136 features (CIR measurements)
- **Data Preparation**: Train/test split (80/20), feature selection
- **Modeling**: Random Forest, Logistic Regression, SVM classifiers
- **Evaluation**: 88.69% accuracy, feature importance analysis
- **Deployment**: Real-time LOS/NLOS detection for positioning systems

### 1.2 The 3D Process Framework

Your project brief mentioned the "3D Process":

1. **Data Preparation**: Cleaning, preprocessing, feature engineering
2. **Data Mining**: Applying ML algorithms to find patterns
3. **Data Visualization**: Interpreting results through plots and metrics

---

## 1.3 The Pipeline: Why This Order Matters

In this project, there are **4 scripts** that must be run in a strict order. Each one depends on the **output** of the previous step. Running them out of order will cause crashes or incorrect results.

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: clean_local.py                                         │
│  Input:  Raw CSV files (7 parts, ~6000 rows each)               │
│  Output: Cleaned CSV files (saved to dataset/Cleaned/)          │
│     ↓                                                           │
│  STEP 2: preprocess_data.py                                     │
│  Input:  Cleaned CSV files                                      │
│  Output: .npy array files + scaler .pkl files                   │
│          (saved to preprocessed_data/)                          │
│     ↓                                                           │
│  STEP 3a: random_forest_classifier.py                           │
│  Input:  .npy array files from preprocessed_data/              │
│  Output: Trained model, confusion matrix, feature importance    │
│          (saved to models/random_forest/)                       │
│     ↓                                                           │
│  STEP 3b: range_regressor.py                                    │
│  Input:  .npy array files from preprocessed_data/              │
│  Output: Trained regressor, range estimation plots              │
│          (saved to models/range_regressor/)                     │
└─────────────────────────────────────────────────────────────────┘

```
```
### What Each Step Is Responsible For

#### STEP 1 — `clean_local.py` : Data Cleaning
**Role:** Take the raw, unfiltered sensor data and make it trustworthy.

This script is run once on each of the 7 raw CSV files. It is responsible for:

| Task | What It Does | Why It's Needed |
|------|--------------|-----------------|
| **Remove duplicates** | Drops rows where every column is identical | Duplicates bias the model by over-representing certain measurements |
| **Remove nulls** | Drops rows with any missing value | ML algorithms cannot process `NaN` values — they will either crash or produce garbage predictions |
| **Remove outliers** | Drops rows where `RANGE <= 0` or `STDEV_NOISE` is in the top 1% | Negative/zero distances are physically impossible; extreme noise spikes are sensor malfunctions |
| **Type enforcement** | Casts the `NLOS` column to integer | Ensures the label is `0` or `1`, not `0.0` or `True/False`, which could confuse the classifier |
| **Feature engineering** | Calculates `SNR` and `SNR_dB` from existing columns | Adds a physically meaningful derived feature — LOS signals have higher SNR than NLOS |

**What it does NOT do:** It does not split the data, scale it, or train anything. It only ensures the raw data is clean and consistent.

**Output:** 7 cleaned CSV files, e.g. `uwb_cleaned_dataset_part1.csv`, each with ~60–64 fewer rows than the original (~1% removed per file).

---

#### STEP 2 — `preprocess_data.py` : Data Preprocessing
**Role:** Transform the cleaned data into a format that ML algorithms can directly consume.

This script loads all 7 cleaned CSVs, merges them into one dataset, and prepares it for training. It is responsible for:

| Task | What It Does | Why It's Needed |
|------|-------------|-----------------|
| **Feature selection** | Picks 16 core features + 120 CIR samples (730–850) = 136 total | Removes irrelevant columns (e.g. row IDs); focuses on physically meaningful signals |
| **Target extraction** | Separates the `NLOS` column as label `y` | The model needs to know what it is learning to predict — this is the "answer" |
| **Train/test split** | 80% training (33,254 samples), 20% testing (8,314 samples), stratified | Ensures the model is evaluated on data it has never seen, giving a fair measure of real-world performance |
| **Scaling** | Applies `StandardScaler` (mean=0, std=1) and `MinMaxScaler` (0–1) | Some algorithms (SVM, Logistic Regression) require features on the same scale to work correctly |
| **Saving arrays** | Saves `X_train`, `X_test`, `y_train`, `y_test` as `.npy` files | Allows any downstream script to load the exact same split without reprocessing the CSVs every time |
| **Saving scalers** | Saves fitted scalers as `.pkl` files | The same scaler used on training data MUST be applied to test/new data — saving it ensures consistency |

**What it does NOT do:** It does not train any model. It only prepares and saves the data.

**Why 80/20 split?**
```
Training set (80%) → The model learns patterns from this data
Test set (20%)     → The model is evaluated on this (never seen before)

If you tested on the same data used for training:
  → Model just memorizes the answers → fake 95%+ accuracy
  → Fails completely on new real-world measurements
  → This is called OVERFITTING
```

**Why stratified split?**
- The dataset is roughly 50/50 LOS vs NLOS (20,997 LOS vs 20,571 NLOS)
- Stratified split ensures both training AND test sets maintain this 50/50 balance
- Without it, you could accidentally get a test set that is 80% LOS and 20% NLOS
- This would make the model look more accurate than it really is (it would just always guess LOS)

**Output:** 10 files in `preprocessed_data/`:
```
X_train_unscaled.npy   ← For Random Forest (tree-based, doesn't need scaling)
X_test_unscaled.npy
X_train_standard.npy   ← For SVM / Logistic Regression (need zero-centred data)
X_test_standard.npy
X_train_minmax.npy     ← For Neural Networks (need data in [0,1] range)
X_test_minmax.npy
y_train.npy            ← Labels for training
y_test.npy             ← Labels for evaluation
scaler_standard.pkl    ← Saved StandardScaler object
scaler_minmax.pkl      ← Saved MinMaxScaler object
```

---

#### STEP 3a — `random_forest_classifier.py` : Supervised Learning (Classification)
**Role:** Train a machine learning model to classify each UWB measurement as LOS or NLOS.

This is the **supervised learning** step. The model is given labeled examples (136 features + known LOS/NLOS answer) and learns the patterns that distinguish them.

| Task | What It Does | Why It's Needed |
|------|-------------|-----------------|
| **Load preprocessed data** | Reads the `.npy` files produced in Step 2 | Gets the consistent, reproducible data split |
| **Train Random Forest** | Calls `rf_model.fit(X_train, y_train)` | Builds 100 decision trees, each learning different patterns from random subsets of training data |
| **Make predictions** | Calls `rf_model.predict(X_test)` | Applies the learned model to the 8,314 unseen test samples |
| **Evaluate performance** | Computes accuracy, precision, recall, F1, ROC-AUC | Quantifies how well the model generalises to new data |
| **Confusion matrix** | Plots a heatmap of TP, TN, FP, FN | Reveals exactly which types of errors the model makes, and which are more common |
| **Feature importance** | Ranks all 136 features by MDI score | Tells us which measurements most distinguish LOS from NLOS — RXPACC is #1 at 15.6% |

**Why use Random Forest instead of a single Decision Tree?**
```
Single Decision Tree:
  → Sees ALL 33,254 training samples
  → Builds very specific, detailed rules
  → Training accuracy: ~95% (looks great!)
  → Test accuracy: ~75% (fails on new data — overfitting!)

Random Forest (100 trees):
  → Each tree sees a different random 63% of training samples
  → Each split considers only √136 ≈ 12 random features
  → Trees make different mistakes that cancel each other out
  → Test accuracy: 88.69% (genuinely reliable)
```

**Output:** Saved to `models/random_forest/`:
```
random_forest_model.pkl          ← The trained model (can be loaded and reused)
confusion_matrix_and_roc.png     ← Visual evaluation (confusion matrix + ROC curve)
feature_importance.png           ← Top 20 features bar chart
feature_importance_ranking.csv   ← All 136 features ranked with scores
model_results.txt                ← All metrics in text form
```

---

#### STEP 3b — `range_regressor.py` : Supervised Learning (Regression)
**Role:** Predict the actual measured distance (range in metres) for both signal paths.

While Step 3a answers *"Is this LOS or NLOS?"*, Step 3b answers *"How far away is the device?"*. This is a **regression** problem — the output is a continuous number, not a category.

| Task | What It Does | Why It's Needed |
|------|-------------|-----------------|
| **Train Path 1 regressor** | Predicts the range of the first dominant signal path | The first path distance is the primary input for localization calculations |
| **Train Path 2 regressor** | Predicts the range of the second dominant signal path | The second path provides additional geometric information for more accurate positioning |
| **Evaluate with RMSE/MAE** | Measures average prediction error in metres | Tells us how accurate the distance estimates are in real physical units |
| **Plot predicted vs actual** | Scatter plot of predictions against ground truth | Visually confirms that the model follows real distances, not just guessing an average |

**Classification vs Regression — what's the difference?**

| | Classification (Step 3a) | Regression (Step 3b) |
|---|---|---|
| **Question** | "Which category is this?" | "What number is this?" |
| **Output** | A label: `0` (LOS) or `1` (NLOS) | A value: e.g. `2.47` metres |
| **Algorithm** | `RandomForestClassifier` | `RandomForestRegressor` |
| **Error metric** | Accuracy, F1, ROC-AUC | RMSE, MAE |
| **Loss function** | Gini Impurity | Mean Squared Error |

**Output:** Saved to `models/range_regressor/`:
```
rf_range_path1.pkl           ← Trained regressor for path 1 distance
rf_range_path2.pkl           ← Trained regressor for path 2 distance
range_estimation_results.png ← Predicted vs actual distance scatter plots
regression_results.txt       ← RMSE, MAE metrics for both paths
```

---

### Full Pipeline Summary

| Step | Script | Role | Input | Output |
|------|--------|------|-------|--------|
| 1 | `clean_local.py` | Data Cleaning | 7 raw CSVs | 7 cleaned CSVs |
| 2 | `preprocess_data.py` | Data Preprocessing | 7 cleaned CSVs | `.npy` arrays + `.pkl` scalers |
| 3a | `random_forest_classifier.py` | LOS/NLOS Classification | `.npy` arrays | Trained classifier + evaluation metrics |
| 3b | `range_regressor.py` | Distance Estimation | `.npy` arrays | Trained regressor + distance plots |

**The golden rule:** Each step **produces files** that the next step **reads**. You cannot skip a step because the files it produces simply won't exist yet.

---

## 2. Understanding Your UWB Data

### 2.1 What is UWB?

Ultra-Wideband (UWB) is a radio technology that:
- Operates at 3.1-10.6 GHz frequency range
- Uses very short pulses (nanoseconds)
- Provides high-precision distance measurements (centimeter accuracy)
- Used in Apple AirTag, Samsung SmartTag, indoor positioning

### 2.2 LOS vs NLOS: The Physics

**Line-of-Sight (LOS)**:
```
Transmitter ----------> Receiver
         (Direct path)
```
- Signal travels directly from transmitter to receiver
- Strongest signal amplitude
- Most accurate distance measurement
- First path index (FP_IDX) ~ 745-747

**Non-Line-of-Sight (NLOS)**:
```
Transmitter ----| Wall |----> Receiver
                (Reflection)
```
- Signal bounces off walls, furniture, people
- Weaker signal amplitude
- Longer measured distance (path is longer due to reflection)
- First path might be blocked or delayed

### 2.3 Your Dataset Features

**Core 16 Features:**

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| RXPACC | Received preamble symbols | **Most important!** NLOS collects more preambles due to reflections |
| RANGE | Measured distance (meters) | NLOS typically shows longer ranges |
| FP_IDX | First path index | Where the signal first arrives in CIR |
| FP_AMP1/2/3 | First path amplitudes | LOS has stronger first path |
| SNR/SNR_dB | Signal-to-Noise Ratio | LOS has better signal quality |
| CIR_PWR | Total CIR power | Overall signal strength |

**CIR Features (120 samples from index 730-850):**
- Channel Impulse Response samples
- Shows the "fingerprint" of the signal
- Peak location and shape differ between LOS/NLOS

**Dataset Statistics:**
- Total samples: 41,568
- LOS: 20,997 (50.5%)
- NLOS: 20,571 (49.5%)
- Features: 136 (16 core + 120 CIR)

### 2.4 Python: Loading Your Data

```python
import numpy as np
import pandas as pd

# Load preprocessed data
X_train = np.load('../preprocessed_data/X_train_unscaled.npy')
X_test = np.load('../preprocessed_data/X_test_unscaled.npy')
y_train = np.load('../preprocessed_data/y_train.npy')
y_test = np.load('../preprocessed_data/y_test.npy')

print(f"Training samples: {X_train.shape[0]:,}")
print(f"Features: {X_train.shape[1]}")
print(f"LOS in training: {np.sum(y_train == 0):,}")
print(f"NLOS in training: {np.sum(y_train == 1):,}")

# Output:
# Training samples: 33,254
# Features: 136
# LOS in training: 16,797
# NLOS in training: 16,457
```

---

## 3. Machine Learning Fundamentals

### 3.1 Supervised Learning

**Definition**: Learning from labeled examples to predict labels for new data.

```
Training Phase:
┌──────────────────────────────────────────┐
│  Input (X)    →    Model    →   Label (y)│
│  Features           Learns        (0=LOS,│
│  (136 dims)         Patterns      1=NLOS)│
└──────────────────────────────────────────┘

Prediction Phase:
┌──────────────────────────────────────────┐
│  New Input (X)  →  Trained  →  Predicted │
│  ( unseen data)     Model         Label   │
└──────────────────────────────────────────┘
```

**Your Problem**: Binary Classification
- Input: 136-dimensional feature vector
- Output: 0 (LOS) or 1 (NLOS)
- Type: Supervised (we have labels for training data)

### 3.2 Train/Test Split

**Why split data?**
- Train on 80% (33,254 samples)
- Test on 20% (8,314 samples) - **unseen data!**
- If we test on training data, we get fake high accuracy (memorization)

**Stratified Split:**
- Maintains class balance in both sets
- Training: 50.5% LOS, 49.5% NLOS
- Test: 50.5% LOS, 49.5% NLOS

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # 20% for testing
    random_state=42,      # Reproducibility
    stratify=y            # Keep class balance
)
```

### 3.3 Model Evaluation Metrics

**Confusion Matrix:**

```
                    Predicted
                 LOS      NLOS
Actual    LOS    TN=3899   FP=301
          NLOS   FN=639    TP=3475
```

Where:
- **TN (True Negative)**: Actual LOS, predicted LOS ✓
- **FP (False Positive)**: Actual LOS, predicted NLOS ✗
- **FN (False Negative)**: Actual NLOS, predicted LOS ✗
- **TP (True Positive)**: Actual NLOS, predicted NLOS ✓

**Formulas:**

```
                    TP + TN
Accuracy = -----------------------
           TP + TN + FP + FN

           TP
Precision = --------
           TP + FP

           TP
Recall = --------
        TP + FN

                2 * Precision * Recall
F1-Score = ----------------------------
              Precision + Recall
```

**Your Results:**
```
TP = 3475 (correctly identified NLOS)
TN = 3899 (correctly identified LOS)
FP = 301 (LOS predicted as NLOS)
FN = 639 (NLOS predicted as LOS)

Accuracy = (3475 + 3899) / 8314 = 88.69%
Precision = 3475 / (3475 + 301) = 92.03%
Recall = 3475 / (3475 + 639) = 84.47%
F1-Score = 2 * 0.9203 * 0.8447 / (0.9203 + 0.8447) = 88.09%
```

**Interpretation:**
- **88.69% Accuracy**: Overall, the model is correct 88.69% of the time
- **92.03% Precision**: When model predicts NLOS, it's right 92% of the time
- **84.47% Recall**: Model finds 84.47% of all actual NLOS cases
- **88.09% F1**: Balanced measure (good balance of precision and recall)

---

## 4. Random Forest Algorithm Deep Dive

### 4.1 Decision Trees: The Building Block

**What is a Decision Tree?**
A flowchart-like structure where each internal node represents a "test" on a feature.

```
Example Decision Tree:

                    RXPACC > 600?
                   /            \
                 Yes            No
                /                \
         RANGE > 3m?          FP_AMP2 > 8000?
          /      \               /        \
        Yes      No           Yes         No
        /          \          /            \
    NLOS          LOS      LOS           NLOS
   (90%)        (85%)    (80%)         (75%)
```

**How it decides splits:**

Using **Gini Impurity** (measures how mixed the classes are):

```
Gini = 1 - (p_LOS)^2 - (p_NLOS)^2

Where:
p_LOS = proportion of LOS samples
p_NLOS = proportion of NLOS samples

Perfect separation: Gini = 0
Maximum impurity: Gini = 0.5 (50/50 mix)
```

**Example Calculation:**
```
Node with 100 samples:
- 70 LOS, 30 NLOS

p_LOS = 70/100 = 0.7
p_NLOS = 30/100 = 0.3

Gini = 1 - (0.7)^2 - (0.3)^2
     = 1 - 0.49 - 0.09
     = 0.42

Lower Gini = better separation
```

### 4.2 The Problem with Single Decision Trees

**Overfitting**: A single tree might memorize the training data.

```
Single Tree Problem:
- Sees all 33,254 training samples
- Creates very specific rules
- Works great on training data (95% accuracy)
- Fails on new data (70% accuracy)

This is called OVERFITTING
```

### 4.3 Random Forest: The Solution

**Concept**: Combine 100 (or more) decision trees that each see different data.

**Two Sources of Randomness:**

**1. Bootstrap Aggregating (Bagging):**
```
Tree 1: Random sample of 33,254 (with replacement)
Tree 2: Different random sample
Tree 3: Another random sample
...
Tree 100: 100th random sample

Each tree sees ~63.2% of unique samples
Some samples repeated, some omitted
```

**2. Feature Randomness:**
```
At each split, consider only random subset of features:
- Total features: 136
- Features considered per split: sqrt(136) ≈ 12

This ensures trees are different from each other
```

### 4.4 Random Forest Algorithm

```
Algorithm: Random Forest Training

Input: Training data X (33,254 × 136), labels y
Output: Forest of 100 trees

For i = 1 to 100:
    1. Create bootstrap sample X_i from X
    2. Train Decision Tree on X_i:
       - At each node, consider m = sqrt(136) random features
       - Split using feature with lowest Gini impurity
       - Continue until pure leaves or max depth
    3. Store tree i

Prediction:
For new sample x:
    1. Get prediction from each of 100 trees
    2. Take majority vote
    3. Return final class (0 or 1)
```

### 4.5 Why Random Forest Works

**Bias-Variance Tradeoff:**

```
Error = Bias^2 + Variance + Irreducible Error

Single Tree:
- Low Bias (fits training data well)
- High Variance (sensitive to training data changes)

Random Forest:
- Low Bias (average of low-bias trees)
- Low Variance (errors cancel out across trees)
- Result: Better generalization!
```

**Mathematical Intuition:**

```
If one tree has error rate = 20%
And trees make independent errors

With 100 trees voting:
Probability ALL 100 are wrong = (0.2)^100 ≈ 0

Much better than single tree!
```

### 4.6 Python: Training Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Create Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=None,        # Let trees grow fully
    min_samples_split=2,   # Minimum samples to split
    min_samples_leaf=1,    # Minimum samples in leaf
    random_state=42,       # Reproducibility
    n_jobs=-1,             # Use all CPU cores
    verbose=1              # Show progress
)

# Train
start_time = time.time()
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")

# Predict
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Output:
# Training completed in 2.78 seconds
# Accuracy: 88.69%
```

### 4.7 Feature Importance: How It's Calculated

**Method**: Mean Decrease in Impurity (MDI)

```
For each feature:
    1. Look at all splits that use this feature across all trees
    2. Calculate how much Gini impurity decreased at each split
    3. Average this decrease across all trees
    4. Higher decrease = more important feature
```

**Your Top 5 Features:**
```
1. RXPACC:      0.1560 (15.6% of total importance)
2. RANGE:       0.0554 (5.5%)
3. MAX_NOISE:   0.0504 (5.0%)
4. FP_AMP2:     0.0480 (4.8%)
5. FP_AMP3:     0.0446 (4.5%)

Total importance sums to 1.0 (100%)
```

**Interpretation:**
- RXPACC alone accounts for 15.6% of the model's decision-making
- Core features (16): 46.4% total importance
- CIR features (120): 53.6% total importance

```python
# Get feature importances
importances = rf_model.feature_importances_

# Create DataFrame for ranking
feature_names = ['RANGE', 'FP_IDX', 'FP_AMP1', ..., 'CIR730', ..., 'CIR849']
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(importance_df.head(10))
```

---

## 5. Model Evaluation Deep Dive

### 5.1 ROC Curve and AUC

**ROC (Receiver Operating Characteristic) Curve:**
Plots True Positive Rate vs False Positive Rate at different thresholds.

```
Thresholds: 0.0, 0.1, 0.2, ..., 0.9, 1.0

For each threshold t:
    Predict NLOS if probability >= t
    
    TP Rate = TP / (TP + FN)  [Sensitivity]
    FP Rate = FP / (FP + TN)  [1 - Specificity]
```

**ASCII Diagram:**
```
TPR (Recall)
   1.0 |                    ****
       |                ****
       |            ****
       |        ****
       |    ****
       |****
   0.0 +------------------------
       0.0                   1.0  FPR
       
Perfect classifier: Curve hugs top-left corner
Random classifier: Diagonal line
Your model: AUC = 0.9535 (excellent!)
```

**AUC (Area Under Curve):**
```
AUC = 0.5   → Random guessing
AUC = 0.7   → Acceptable
AUC = 0.8   → Good
AUC = 0.9   → Excellent
AUC = 1.0   → Perfect

Your model: AUC = 0.9535 (Excellent!)
```

**Interpretation:**
- 95.35% probability that model ranks a random NLOS sample higher than a random LOS sample
- Very strong discriminative ability

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png')
```

### 5.2 Confusion Matrix Interpretation

**Your Confusion Matrix:**
```
                Predicted
            LOS        NLOS
Actual LOS   3899 (TN)   301 (FP)
      NLOS   639 (FN)   3475 (TP)
      
Total: 8314 test samples
Errors: 301 + 639 = 940 (11.31% error rate)
```

**Business Impact:**
- **False Positives (301)**: LOS predicted as NLOS
  - Cost: Might discard a good signal unnecessarily
  - Impact: Slightly reduced positioning accuracy
  
- **False Negatives (639)**: NLOS predicted as LOS
  - Cost: Use inaccurate NLOS signal for positioning
  - Impact: Larger positioning errors (more serious!)

**Precision vs Recall Tradeoff:**
```
If you want fewer False Positives → Increase threshold
    More conservative, higher precision
    
If you want fewer False Negatives → Decrease threshold
    More sensitive, higher recall
```

---

## 6. Algorithm Comparison

### 6.1 Logistic Regression

**Concept**: Finds linear decision boundary using sigmoid function.

```
Probability of NLOS = 1 / (1 + e^-(w1*x1 + w2*x2 + ... + w136*x136 + b))

Where:
- w1...w136: Learned weights for each feature
- b: Bias term
- e: Euler's number (~2.718)
```

**Decision Boundary:**
```
If Probability >= 0.5 → Predict NLOS (1)
If Probability < 0.5  → Predict LOS (0)
```

**Strengths:**
- Very fast training
- Highly interpretable (weights show feature direction)
- Probabilistic output
- Good baseline

**Weaknesses:**
- Assumes linear separability
- Can't capture complex feature interactions

**When to use:** Baseline model, when you need interpretability

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_standard, y_train)  # Needs scaled data!
```

### 6.2 Support Vector Machine (SVM)

**Concept**: Finds optimal hyperplane that maximizes margin between classes.

```
ASCII Diagram:

        + + +        ← NLOS samples (class +1)
       +     +
      +   |←─→|   +   ← Maximum Margin
     +    |Hyper|    +
    +     |plane|     +
         ←──┼──→
    -     |     -     ← LOS samples (class -1)
     -    |    -
      -   |   -
       -  |  -
        - - -
```

**Mathematical Formulation:**
```
Maximize: Margin = 2 / ||w||

Subject to:
    y_i * (w·x_i + b) >= 1 for all i
    
Where:
- w: Weight vector (perpendicular to hyperplane)
- b: Bias term
- y_i: Class label (+1 or -1)
- x_i: Feature vector
```

**Kernel Trick:**
For non-linear boundaries, map to higher dimension:
```
Linear kernel: K(x,y) = x·y
RBF kernel:    K(x,y) = exp(-gamma * ||x-y||^2)
```

**Strengths:**
- Excellent for high-dimensional data (your 136 features!)
- Maximum margin principle → good generalization
- Effective with clear margin of separation

**Weaknesses:**
- Requires feature scaling
- Slow on large datasets (33K samples)
- Hard to interpret
- Sensitive to hyperparameters (C, gamma)

**When to use:** When you have many features and clear class separation

```python
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_standard, y_train)  # Needs scaled data!
```

### 6.3 Comparison Summary

| Aspect | Logistic Regression | Random Forest | SVM |
|--------|-------------------|---------------|-----|
| **Speed** | ⚡ Very Fast | 🚀 Fast | 🐢 Slow |
| **Accuracy** | Good (80-85%) | Very Good (88%) | Excellent (87-90%) |
| **Needs Scaling** | ✅ Yes | ❌ No | ✅ Yes |
| **Interpretability** | ⭐⭐⭐ Excellent | ⭐⭐ Good | ⭐ Poor |
| **Feature Importance** | Weights | Built-in | ❌ No |
| **Handles Non-linear** | ❌ No | ✅ Yes | ✅ Yes (with kernel) |
| **Overfitting Risk** | Low | Low | Medium |

**Recommendation for your project:**
- **Random Forest**: Best all-rounder, feature importance, easy to use
- **Logistic Regression**: Quick baseline, interpretable
- **SVM**: If you want maximum accuracy and don't mind tuning

---

## 7. Best Practices & Next Steps

### 7.1 Avoiding Overfitting

**What is Overfitting?**
```
Training Accuracy: 95%
Test Accuracy: 70%
→ Model memorized training data, can't generalize
```

**Prevention Methods:**

**1. Cross-Validation:**
```
Split data into 5 folds:

Fold 1: Test | Train Train Train Train
Fold 2: Train | Test Train Train Train  
Fold 3: Train Train | Test Train Train
Fold 4: Train Train Train | Test Train
Fold 5: Train Train Train Train | Test

Average accuracy across 5 folds → more reliable estimate
```

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**2. Regularization:**
- Limit tree depth
- Minimum samples per leaf
- Number of trees

### 7.2 Hyperparameter Tuning

**Key Parameters for Random Forest:**

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees
    'max_depth': [10, 20, None],          # Tree depth
    'min_samples_split': [2, 5, 10],      # Min samples to split
    'min_samples_leaf': [1, 2, 4]         # Min samples in leaf
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

### 7.3 Feature Selection

**Why reduce features?**
- Faster training
- Less overfitting
- Better interpretability

**Methods:**

**1. Using Feature Importance:**
```python
# Keep only top 30 features
top_features_idx = np.argsort(importances)[-30:]
X_train_selected = X_train[:, top_features_idx]
X_test_selected = X_test[:, top_features_idx]
```

**2. Using Your Results:**
From Random Forest, top 10 features contribute ~46% of importance.
You could use just these 10 and still get good accuracy!

### 7.4 Your Next Steps

**Immediate Actions:**
1. ✅ Train Logistic Regression for comparison
2. ✅ Train SVM for comparison
3. ✅ Create comparison table of all three models

**Advanced (Optional):**
4. Hyperparameter tuning with GridSearchCV
5. Feature selection (reduce from 136 to top 30)
6. Build regression model for distance prediction
7. Ensemble methods (combine all three models)

**For Your Report:**
- Document the 3D process
- Show EDA visualizations
- Compare model performances
- Analyze feature importance
- Interpret results physically (why RXPACC matters)

---

## 8. Summary

### Key Concepts Learned:

1. **Data Preparation**: Train/test split, feature selection, preprocessing
2. **Random Forest**: Ensemble of 100 decision trees using bagging
3. **Model Evaluation**: Accuracy, precision, recall, F1, ROC-AUC
4. **Feature Importance**: RXPACC is most important (15.6%)
5. **Your Results**: 88.69% accuracy, AUC=0.9535 (excellent!)

### Your Achievements:

- ✅ Successfully classified LOS vs NLOS with 88.69% accuracy
- ✅ Identified key discriminative features
- ✅ Created reproducible preprocessing pipeline
- ✅ Built interpretable machine learning model

### Mathematical Formulas Reference:

```
Gini Impurity:           Gini = 1 - p_LOS² - p_NLOS²

Accuracy:                (TP + TN) / Total

Precision:               TP / (TP + FP)

Recall:                  TP / (TP + FN)

F1-Score:                2 * (Precision * Recall) / (Precision + Recall)

Logistic Sigmoid:        P = 1 / (1 + e^-(wx + b))

SVM Margin:              Maximize 2 / ||w||
```

---

## Resources for Further Learning

**Books:**
- "Hands-On Machine Learning" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Bishop

**Online:**
- Scikit-learn documentation: https://scikit-learn.org
- Towards Data Science (Medium)
- Kaggle Learn courses

**Practice:**
- Try different algorithms on your data
- Experiment with hyperparameters
- Visualize decision boundaries

---

**Created for**: UWB LOS/NLOS Classification Project  
**Dataset**: 41,568 samples, 136 features  
**Best Model**: Random Forest (88.69% accuracy)  
**Key Finding**: RXPACC is the most important feature (15.6% importance)

**Good luck with your project! 🚀**
