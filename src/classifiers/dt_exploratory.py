"""
UWB LOS/NLOS - Exploratory Decision Tree
Fits a shallow decision tree on the top features to understand
decision boundaries before committing to an ensemble model.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report

plt.style.use("seaborn-v0_8-darkgrid")

RUN_NAME = os.getenv("RUN_NAME", "split_env_70_15_15_seed42")
DATA_DIR = Path("./runs") / RUN_NAME / "preprocessed_data"
OUTPUT_DIR = Path("./runs") / RUN_NAME / "models" / "dt_exploratory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DEPTH = int(os.getenv("DT_MAX_DEPTH", "4"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
CLASS_NAMES = ["LOS+NLOS", "NLOS+NLOS"]

print("=" * 80)
print("EXPLORATORY DECISION TREE - PAIR LEVEL")
print("=" * 80)
print(f"RUN_NAME  : {RUN_NAME}")
print(f"Max depth : {MAX_DEPTH}")
print(f"Features  : core + second-path (no CIR bins)")
print()


# ==============================================================================
# STEP 1: LOAD DATA + SELECT CORE FEATURES
# ==============================================================================
print("Step 1: Loading data and selecting core + second-path features...")

X_train = np.load(DATA_DIR / "X_train_pair.npy")
X_test = np.load(DATA_DIR / "X_test_pair.npy")
y_train = np.load(DATA_DIR / "y_train_pair.npy")
y_test = np.load(DATA_DIR / "y_test_pair.npy")

# Load full feature name list
pair_feature_names_path = DATA_DIR / "pair_feature_names.txt"
all_feature_names: list[str] = []

with open(pair_feature_names_path, "r") as f:
    lines = f.readlines()

reading_core = False
cir_start = None

for line in lines:
    stripped = line.strip()
    if stripped.startswith("Core + Second-Path Features"):
        reading_core = True
        continue
    if reading_core and stripped.startswith(tuple("0123456789")):
        all_feature_names.append(line.split(".")[-1].strip())
    if stripped.startswith("CIR Features"):
        reading_core = False
        continue
    if stripped.startswith("Range: CIR"):
        try:
            parts = stripped.replace("Range:", "").strip().split()
            cir_start = int(parts[0].replace("CIR", ""))
            cir_end = int(parts[2].replace("CIR", "")) + 1
        except (IndexError, ValueError):
            cir_start, cir_end = 730, 850
        for i in range(cir_start, cir_end):
            all_feature_names.append(f"CIR{i}")
        break

if len(all_feature_names) != X_train.shape[1]:
    all_feature_names = [f"f{i}" for i in range(X_train.shape[1])]

# Select only core + second-path features (exclude CIR bins)
core_idx = [i for i, name in enumerate(all_feature_names) if not name.startswith("CIR")]
top_features = [all_feature_names[i] for i in core_idx]

X_train_top = X_train[:, core_idx]
X_test_top = X_test[:, core_idx]

print(f"  Training samples  : {len(X_train):,}")
print(f"  Test samples      : {len(X_test):,}")
print(f"  Core features used: {len(top_features)} — {top_features}")
print()


# ==============================================================================
# STEP 2: FIT SHALLOW DECISION TREE
# ==============================================================================
print(f"Step 2: Fitting shallow decision tree (max_depth={MAX_DEPTH})...")

dt = DecisionTreeClassifier(
    max_depth=MAX_DEPTH,
    class_weight="balanced",
    random_state=RANDOM_SEED,
)
dt.fit(X_train_top, y_train)

y_pred = dt.predict(X_test_top)
accuracy = accuracy_score(y_test, y_pred)

print(f"  Decision tree accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print()
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


# ==============================================================================
# STEP 3: VISUALISE THE TREE
# ==============================================================================
print("Step 3: Saving tree visualisation...")

fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(
    dt,
    feature_names=top_features,
    class_names=CLASS_NAMES,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax,
    impurity=True,
    proportion=False,
)
ax.set_title(
    f"Exploratory Decision Tree (depth={MAX_DEPTH}) — Core + Second-Path Features",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dt_tree_plot.png", dpi=200, bbox_inches="tight")
print("Saved: dt_tree_plot.png")
plt.close()


# ==============================================================================
# STEP 4: PRINT + SAVE HUMAN-READABLE RULES
# ==============================================================================
print("Step 4: Saving decision rules...")

rules_text = export_text(dt, feature_names=top_features)

with open(OUTPUT_DIR / "dt_rules.txt", "w") as f:
    f.write("EXPLORATORY DECISION TREE — HUMAN-READABLE RULES\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Max depth        : {MAX_DEPTH}\n")
    f.write(f"Features used    : {len(top_features)} core + second-path (no CIR bins)\n")
    f.write(f"Test accuracy    : {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
    f.write("RULES:\n")
    f.write(rules_text)
    f.write("\nFEATURE IMPORTANCE (Gini):\n")
    for feat, imp in sorted(
        zip(top_features, dt.feature_importances_), key=lambda x: -x[1]
    ):
        f.write(f"  {feat:20s}: {imp:.4f}\n")

print("Saved: dt_rules.txt")
print()
print(rules_text)


# ==============================================================================
# STEP 5: FEATURE IMPORTANCE BAR CHART
# ==============================================================================
print("Step 5: Saving feature importance chart...")

imp_df = (
    pd.DataFrame({"Feature": top_features, "Importance": dt.feature_importances_})
    .sort_values("Importance", ascending=True)
    .reset_index(drop=True)
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp_df["Feature"], imp_df["Importance"], color="#2ecc71", alpha=0.85)
ax.set_xlabel("Gini Importance", fontsize=12)
ax.set_title(
    f"Decision Tree Feature Importance — Core Features (depth={MAX_DEPTH})",
    fontsize=12,
    fontweight="bold",
)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dt_feature_importance.png", dpi=200, bbox_inches="tight")
print("Saved: dt_feature_importance.png")
plt.close()

imp_df.sort_values("Importance", ascending=False).to_csv(
    OUTPUT_DIR / "dt_feature_importance.csv", index=False
)
print("Saved: dt_feature_importance.csv")


# ==============================================================================
# SUMMARY
# ==============================================================================
print()
print("=" * 80)
print("EXPLORATORY DT COMPLETE")
print("=" * 80)
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Max depth : {MAX_DEPTH}")
print(f"  Features  : {len(top_features)} core + second-path")
print()
print("FILES SAVED TO: models/dt_exploratory/")
for fn in ["dt_tree_plot.png", "dt_rules.txt", "dt_feature_importance.png"]:
    print(f"  - {fn}")
print("=" * 80)
