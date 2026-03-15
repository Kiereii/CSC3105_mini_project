# ML Algorithms: Random Forest, Logistic Regression, SVM, XGBoost

## Quick Comparison

| Algorithm | Speed | Accuracy | Interpretability | Best For |
|-----------|-------|----------|------------------|----------|
| **Random Forest** | ⚡⚡ | 88.69% | Good | General use ✓ |
| **Logistic Regression** | ⚡⚡⚡ | ~81% | Excellent | Baseline |
| **SVM** | ⚡ | ~87% | Poor | High-dim data |
| **XGBoost** | ⚡⚡ | ~88% | Moderate | Max accuracy |

**Winner: Random Forest** — Best balance of accuracy, speed, and interpretability.

---

## 1. Random Forest

### How It Works

**Problem with single decision tree:**
- Memorizes training data perfectly
- Fails on new data (overfitting)

**Solution — Random Forest:**
- Train 100 trees on random data subsets
- Each tree uses random features for splits
- Combine predictions via majority voting
- Errors cancel out → better generalization

### Algorithm

```
For each of 100 trees:
  1. Bootstrap sample: ~63% of training data (random with replacement)
  2. At each split, consider only √136 ≈ 12 random features
  3. Train decision tree (no pruning)

Prediction:
  Each tree votes (0 = LOS or 1 = NLOS)
  Final prediction = majority vote
```

### Feature Importance

Calculated by Gini impurity reduction at each split:

```
Top 5 Features:
1. RXPACC:    15.6% ← Most important!
2. RANGE:      5.5%
3. MAX_NOISE:  5.0%
4. FP_AMP2:    4.8%
5. FP_AMP3:    4.5%
```

### Code Example

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_unscaled, y_train)

# Predictions
y_pred = rf.predict(X_test_unscaled)  # Hard predictions (0 or 1)
y_proba = rf.predict_proba(X_test_unscaled)[:, 1]  # Soft (probability)

# Feature importance
importances = rf.feature_importances_
```

### When to Use

✓ Good balance of speed & accuracy  
✓ Built-in feature importance  
✓ Handles non-linear relationships  
✓ No scaling required  
✓ Robust to outliers  

---

## 2. Logistic Regression

### How It Works

Converts features to probability using sigmoid function:

```
P(NLOS) = 1 / (1 + e^-(w₁x₁ + w₂x₂ + ... + w₁₃₆x₁₃₆ + b))

Where:
  - w_i: learned weight for feature i
  - b: bias term
  - Output: probability between 0 and 1
  - If P > 0.5: predict NLOS (1)
  - If P < 0.5: predict LOS (0)
```

### Feature Weights

- **Positive weight:** Feature increases probability of NLOS
- **Negative weight:** Feature increases probability of LOS
- **Larger magnitude:** Stronger influence on prediction

### Code Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# MUST scale for Logistic Regression!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_standard)
X_test_scaled = scaler.transform(X_test_standard)

# Train
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Predictions
y_pred = lr.predict(X_test_scaled)
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

# Feature weights
weights = lr.coef_[0]  # One weight per feature
```

### When to Use

✓ Fast training  
✓ Highly interpretable (see feature weights)  
✓ Good baseline model  
✓ Probabilistic output  
✗ Assumes linear separability  
✗ May miss complex patterns  

---

## 3. Support Vector Machine (SVM)

### How It Works

Finds the **optimal hyperplane** that maximizes margin between classes:

```
     LOS ● ●
      ●  ●
         |←──→|  ← Maximum margin
         | ○○○|  ← Hyperplane (decision boundary)
         |●●●|
    NLOS ●  ●
```

### Kernels

- **Linear kernel:** For linearly separable data
- **RBF kernel (default):** For non-linear problems (maps to higher dimensions)

### Code Example

```python
from sklearn.svm import SVC

# MUST scale for SVM!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_standard)
X_test_scaled = scaler.transform(X_test_standard)

# Train with RBF kernel (non-linear)
svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm.predict(X_test_scaled)
y_proba = svm.predict_proba(X_test_scaled)[:, 1]
```

### When to Use

✓ Excellent for high-dimensional data (136 features)  
✓ Powerful non-linear capability (RBF kernel)  
✓ Good generalization  
✗ Slow on large datasets (33K samples)  
✗ Requires feature scaling  
✗ Hard to interpret (black box)  
✗ Sensitive to hyperparameters  

---

## 4. XGBoost

### How It Works

**Gradient Boosting:** Build trees sequentially, each corrects previous errors:

```
Tree 1 predicts y₁
Error: y - y₁
  ↓
Tree 2 predicts the error
Prediction: y₁ + y₂
Error: y - (y₁ + y₂)
  ↓
Tree 3 predicts new error
...
Final prediction: y₁ + y₂ + y₃ + ... + y₁₀₀
```

### Code Example

```python
from xgboost import XGBClassifier

# XGBoost can handle unscaled data
xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb.fit(X_train_unscaled, y_train)

# Predictions
y_pred = xgb.predict(X_test_unscaled)
y_proba = xgb.predict_proba(X_test_unscaled)[:, 1]

# Feature importance
importances = xgb.feature_importances_
```

### When to Use

✓ Often achieves best accuracy (especially with tuning)  
✓ Fast training  
✓ Built-in regularization (prevents overfitting)  
✓ No scaling required  
✗ Many hyperparameters to tune  
✗ Harder to interpret  
✗ More complex than RF  

---

## Algorithm Comparison Matrix

| Situation | Best Choice | Why |
|-----------|-------------|-----|
| Quick baseline, interpretability important | LR | Fast, transparent |
| Best balance of speed & accuracy | RF ✓ | 88.69% accuracy, feature importance |
| High-dimensional, need precision | SVM | Kernel trick powerful, but slow |
| Maximum accuracy, willing to tune | XGB | Often beats RF with tuning |
| Need to explain to non-technical | RF or LR | RF shows feature importance, LR shows weights |
| Production deployment | RF | Proven, fast, reliable |

---

## Key Mathematical Concepts

### Gini Impurity (Decision Trees, Random Forest, XGBoost)

```
Gini = 1 - p₀² - p₁²

Where:
  p₀ = proportion of LOS samples in node
  p₁ = proportion of NLOS samples in node

Gini = 0: Perfect separation (all one class)
Gini = 0.5: Maximum impurity (50/50 mix)

A split is good if it reduces Gini significantly.
```

### Sigmoid Function (Logistic Regression)

```
σ(z) = 1 / (1 + e^(-z))

Where z = w·x + b

Properties:
  - Input z: unbounded (-∞ to +∞)
  - Output σ(z): probability (0 to 1)
  - Smooth S-curve shape
  - Differentiable (good for gradient descent)
```

### Support Vector Margin (SVM)

```
Maximize: margin = 2 / ||w||

Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i

Intuition: Find widest "street" between classes
```

---

## Your Project Results

| Model | Accuracy | AUC | Top Feature |
|-------|----------|-----|-------------|
| Random Forest | 88.69% | 0.9535 | RXPACC (15.6%) |
| Logistic Regression | ~81% | ~0.87 | RXPACC (high weight) |
| SVM | ~87% | ~0.91 | RXPACC (high margin) |
| XGBoost | ~88% | ~0.93 | RXPACC (15.2%) |

**All models agree:** RXPACC is the most important feature for distinguishing LOS from NLOS.

---

Next: [06_Evaluation_Explained.md](06_Evaluation_Explained.md)

