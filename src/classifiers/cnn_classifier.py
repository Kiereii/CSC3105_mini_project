"""
UWB LOS/NLOS Classification - 1D CNN (Hybrid Architecture)

Why a CNN for this problem:
1. The 120 CIR samples (730–849) form a 1D signal waveform. LOS signals
   have a sharp, clean first-path peak; NLOS signals have a smeared,
   attenuated peak. Convolutional filters can detect these local shape
   differences directly from the raw waveform — something tree-based
   models cannot do.

Architecture:
  - CIR Branch  : 1D Conv → MaxPool → 1D Conv → MaxPool → Flatten → Dense
  - Core Branch : Dense → Dense  (16 signal metrics)
  - Merge       : Concatenate → Dense → Dropout → Output (sigmoid)

Key difference from other classifiers:
  ✓ Learns waveform SHAPE features automatically (no manual feature engineering)
  ✓ Requires standard-scaled input (unlike tree-based models)
  ✗ Less interpretable — no direct feature importance ranking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR   = Path("./preprocessed_data")
OUTPUT_DIR = Path("./models/cnn")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_CIR       = 120   # CIR730 to CIR849
N_CORE      = 16    # Core signal metrics

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 80)
print("1D CNN CLASSIFIER - UWB LOS/NLOS PREDICTION")
print("=" * 80)
print()
print("Architecture: Hybrid 1D CNN")
print("  CIR branch  : Conv1D filters learn peak-shape features from waveform")
print("  Core branch : Dense layers process signal metrics (SNR, FP_AMP, etc.)")
print("  Merged      : Combined representation → sigmoid output")
print()

# ==============================================================================
# STEP 1: LOAD PREPROCESSED DATA (Standard-scaled — required for neural nets)
# ==============================================================================
print("Step 1: Loading preprocessed data...")
print("(Using STANDARD-SCALED data — neural networks require normalised inputs)")
print()

X_train_full = np.load(DATA_DIR / "X_train_standard.npy")
X_test_full  = np.load(DATA_DIR / "X_test_standard.npy")
y_train      = np.load(DATA_DIR / "y_train.npy")
y_test       = np.load(DATA_DIR / "y_test.npy")

# Split into core features (0:16) and CIR features (16:136)
X_train_core = X_train_full[:, :N_CORE]                        # (N, 16)
X_train_cir  = X_train_full[:, N_CORE:].reshape(-1, N_CIR, 1) # (N, 120, 1)
X_test_core  = X_test_full[:, :N_CORE]
X_test_cir   = X_test_full[:, N_CORE:].reshape(-1, N_CIR, 1)

print(f"Training set       : {X_train_full.shape[0]:,} samples")
print(f"Test set           : {X_test_full.shape[0]:,} samples")
print(f"CIR input shape    : {X_train_cir.shape}  (waveform per sample)")
print(f"Core input shape   : {X_train_core.shape}  (signal metrics per sample)")
print(f"  LOS  in train    : {np.sum(y_train == 0):,}")
print(f"  NLOS in train    : {np.sum(y_train == 1):,}")
print()

# ==============================================================================
# STEP 2: BUILD THE HYBRID CNN MODEL
# ==============================================================================
print("Step 2: Building hybrid 1D CNN model...")
print("-" * 80)

# ── CIR Branch (1D Convolutional) ─────────────────────────────────────────────
cir_input = keras.Input(shape=(N_CIR, 1), name="cir_input")

x = layers.Conv1D(32, kernel_size=5, activation="relu", padding="same",
                  name="conv1")(cir_input)
x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same",
                  name="conv2")(x)
x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)
x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same",
                  name="conv3")(x)
x = layers.GlobalAveragePooling1D(name="gap")(x)
x = layers.Dense(64, activation="relu", name="cir_dense")(x)
cir_out = layers.Dropout(0.3, name="cir_dropout")(x)

# ── Core Features Branch (Dense) ──────────────────────────────────────────────
core_input = keras.Input(shape=(N_CORE,), name="core_input")
y = layers.Dense(32, activation="relu", name="core_dense1")(core_input)
y = layers.BatchNormalization(name="core_bn")(y)
core_out = layers.Dense(32, activation="relu", name="core_dense2")(y)

# ── Merge and Classify ─────────────────────────────────────────────────────────
merged   = layers.Concatenate(name="merge")([cir_out, core_out])
merged   = layers.Dense(64, activation="relu", name="merged_dense")(merged)
merged   = layers.Dropout(0.4, name="merged_dropout")(merged)
output   = layers.Dense(1, activation="sigmoid", name="output")(merged)

model = Model(inputs=[cir_input, core_input], outputs=output,
              name="HybridCNN_LOS_NLOS")

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss      = "binary_crossentropy",
    metrics   = ["accuracy"],
)

model.summary()
print()

total_params = model.count_params()
print(f"Total trainable parameters: {total_params:,}")
print()

# ==============================================================================
# STEP 3: TRAIN THE MODEL
# ==============================================================================
print("Step 3: Training the model...")
print("-" * 80)
print("Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau (patience=5)")
print()

callbacks = [
    EarlyStopping(
        monitor   = "val_loss",
        patience  = 10,
        restore_best_weights = True,
        verbose   = 1,
    ),
    ReduceLROnPlateau(
        monitor   = "val_loss",
        factor    = 0.5,
        patience  = 5,
        min_lr    = 1e-6,
        verbose   = 1,
    ),
]

t0 = time.time()
history = model.fit(
    x               = [X_train_cir, X_train_core],
    y               = y_train,
    epochs          = 100,
    batch_size      = 256,
    validation_split= 0.15,
    callbacks       = callbacks,
    verbose         = 1,
)
train_time = time.time() - t0

best_epoch = np.argmin(history.history["val_loss"]) + 1
print()
print(f"✓ Training completed in {train_time:.2f} seconds")
print(f"✓ Best epoch: {best_epoch} / {len(history.history['val_loss'])}")
print()

# ==============================================================================
# STEP 4: EVALUATE ON TEST SET
# ==============================================================================
print("Step 4: Evaluating on test set...")
print("-" * 80)

y_pred_proba = model.predict(
    [X_test_cir, X_test_core], batch_size=256, verbose=0
).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_pred_proba)
cm        = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print()
print("  Confusion Matrix:")
print(f"    True LOS  → Pred LOS  (TN) : {tn:,}")
print(f"    True LOS  → Pred NLOS (FP) : {fp:,}")
print(f"    True NLOS → Pred LOS  (FN) : {fn:,}  ← Dangerous misclassification")
print(f"    True NLOS → Pred NLOS (TP) : {tp:,}")
print()
print(classification_report(y_test, y_pred, target_names=["LOS", "NLOS"]))

# ==============================================================================
# STEP 5: TRAINING HISTORY PLOT
# ==============================================================================
print("Step 5: Plotting training history...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

epochs_range = range(1, len(history.history["loss"]) + 1)

ax1.plot(epochs_range, history.history["loss"],      label="Train Loss",
         color="#3498db", linewidth=2)
ax1.plot(epochs_range, history.history["val_loss"],  label="Val Loss",
         color="#e74c3c", linewidth=2, linestyle="--")
ax1.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5,
            label=f"Best epoch ({best_epoch})")
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
ax1.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

ax2.plot(epochs_range, history.history["accuracy"],      label="Train Accuracy",
         color="#2ecc71", linewidth=2)
ax2.plot(epochs_range, history.history["val_accuracy"],  label="Val Accuracy",
         color="#e67e22", linewidth=2, linestyle="--")
ax2.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.5,
            label=f"Best epoch ({best_epoch})")
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

fig.suptitle("1D CNN Training History – UWB LOS/NLOS",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_history_cnn.png", dpi=300, bbox_inches="tight")
print("✓ Saved: training_history_cnn.png")
plt.close()

# ==============================================================================
# STEP 6: CONFUSION MATRIX & ROC CURVE
# ==============================================================================
print("Step 6: Plotting confusion matrix and ROC curve...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
            xticklabels=["LOS", "NLOS"], yticklabels=["LOS", "NLOS"],
            linewidths=0.5)
ax1.set_title("Confusion Matrix – 1D CNN", fontsize=13, fontweight="bold",
              color="#1a5276")
ax1.set_xlabel("Predicted", fontsize=11)
ax1.set_ylabel("Actual", fontsize=11)
total = cm.sum()
for i in range(2):
    for j in range(2):
        ax1.text(j + 0.5, i + 0.75, f"({cm[i,j]/total*100:.1f}%)",
                 ha="center", va="center", fontsize=9, color="red")
ax1.text(0.5, 1.75, f"⚠ Dangerous: {fn:,}", ha="center", va="center",
         fontsize=8, color="darkred", style="italic")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5,
         label="Random Classifier (AUC = 0.500)")
ax2.plot(fpr, tpr, color="#1a5276", linewidth=2.5,
         label=f"1D CNN (AUC = {auc:.4f})")
ax2.set_xlabel("False Positive Rate", fontsize=12)
ax2.set_ylabel("True Positive Rate", fontsize=12)
ax2.set_title("ROC Curve – 1D CNN", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, loc="lower right")
ax2.grid(alpha=0.3)

fig.suptitle("1D CNN – LOS/NLOS Classification Results",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_and_roc_cnn.png", dpi=300,
            bbox_inches="tight")
print("✓ Saved: confusion_matrix_and_roc_cnn.png")
plt.close()

# ==============================================================================
# STEP 7: SAVE MODEL, PREDICTIONS & RESULTS
# ==============================================================================
print("\nStep 7: Saving model and results...")
print("=" * 80)

model.save(OUTPUT_DIR / "cnn_model.keras")
np.save(OUTPUT_DIR / "y_pred_cnn.npy",       y_pred)
np.save(OUTPUT_DIR / "y_pred_proba_cnn.npy", y_pred_proba)

print("✓ Saved: cnn_model.keras")
print("✓ Saved: y_pred_cnn.npy / y_pred_proba_cnn.npy")

with open(OUTPUT_DIR / "model_results_cnn.txt", "w") as f:
    f.write("1D CNN CLASSIFICATION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset   : UWB LOS/NLOS\n")
    f.write(f"Features  : {X_train_full.shape[1]} (16 core + 120 CIR 730-849)\n")
    f.write(f"Scaling   : Standard (zero-mean, unit-variance)\n")
    f.write(f"Training samples : {len(X_train_full):,}\n")
    f.write(f"Test samples     : {len(X_test_full):,}\n\n")
    f.write("MODEL ARCHITECTURE:\n")
    f.write("  CIR Branch  : Conv1D(32,5) → MaxPool → Conv1D(64,3) → MaxPool\n")
    f.write("                → Conv1D(64,3) → GlobalAvgPool → Dense(64) → Dropout(0.3)\n")
    f.write("  Core Branch : Dense(32) → BN → Dense(32)\n")
    f.write("  Merged      : Concatenate → Dense(64) → Dropout(0.4) → Sigmoid\n")
    f.write(f"  Total params: {total_params:,}\n\n")
    f.write("TRAINING:\n")
    f.write(f"  Epochs (total ran) : {len(history.history['loss'])}\n")
    f.write(f"  Best epoch         : {best_epoch}\n")
    f.write(f"  Training time      : {train_time:.2f} seconds\n")
    f.write(f"  EarlyStopping      : patience=10\n")
    f.write(f"  ReduceLROnPlateau  : patience=5, factor=0.5\n\n")
    f.write("PERFORMANCE METRICS:\n")
    f.write(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)\n")
    f.write(f"  Precision : {precision:.4f}\n")
    f.write(f"  Recall    : {recall:.4f}\n")
    f.write(f"  F1-Score  : {f1:.4f}\n")
    f.write(f"  ROC-AUC   : {auc:.4f}\n\n")
    f.write("CONFUSION MATRIX:\n")
    f.write(f"  True LOS  predicted as LOS  : {tn:,}\n")
    f.write(f"  True LOS  predicted as NLOS : {fp:,}\n")
    f.write(f"  True NLOS predicted as LOS  : {fn:,}  ← Dangerous\n")
    f.write(f"  True NLOS predicted as NLOS : {tp:,}\n")

print("✓ Saved: model_results_cnn.txt")

# ==============================================================================
# STEP 8: SUMMARY
# ==============================================================================
print()
print("=" * 80)
print("1D CNN CLASSIFIER - COMPLETE!")
print("=" * 80)
print()
print(" KEY RESULTS:")
print(f"   Accuracy  : {accuracy * 100:.2f}%")
print(f"   F1-Score  : {f1:.4f}")
print(f"   ROC-AUC   : {auc:.4f}")
print()
print(" HOW IT DIFFERS FROM TREE-BASED MODELS:")
print("   • Learns CIR waveform shape automatically via convolutional filters")
print("   • No manual feature engineering — the network extracts its own features")
print("   • Requires scaled data (standard normalisation applied)")
print("   • Less interpretable — no direct feature importance ranking available")
print()
print(" Generated files (in ./models/cnn/):")
for fname in [
    "cnn_model.keras",
    "y_pred_cnn.npy",
    "y_pred_proba_cnn.npy",
    "training_history_cnn.png",
    "confusion_matrix_and_roc_cnn.png",
    "model_results_cnn.txt",
]:
    print(f"   • {fname}")
print()
print(" NEXT STEP:")
print("   Run src/evaluation/compare_models.py to include CNN in the comparison.")
print("=" * 80)
