"""
UWB LOS/NLOS - Second Path Feature Extraction
==============================================
The brief requires classifying and estimating range for TWO dominant paths.

Path logic (from brief):
  - If Path 1 = LOS  → Path 2 is always NLOS
  - If Path 1 = NLOS → Path 2 is always NLOS
  So Path 2 label is ALWAYS 1 (NLOS). The real value is in RANGE ESTIMATION.

This script:
  1. Finds the second dominant CIR peak after the first path region
  2. Engineers second-path features (index, amplitude, gap from first path)
  3. Derives Path 2 estimated range from the peak index offset
  4. Saves all features + targets ready for the range regressor
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
import pickle
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = Path("./Dataset/UWB-LOS-NLOS-Data-Set/dataset/Cleaned")
OUTPUT_DIR = Path("./preprocessed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE   = 0.2

# CIR hardware constants (DW1000 datasheet)
# Each CIR sample = 1 ns resolution → ~0.3 m per sample at speed of light
SPEED_OF_LIGHT   = 3e8          # m/s
SAMPLE_PERIOD_NS = 1.0016       # ns per CIR sample (DW1000 spec)
METERS_PER_SAMPLE = (SPEED_OF_LIGHT * SAMPLE_PERIOD_NS * 1e-9) / 2  # two-way ToF

# Focus region (from EDA — where actual signal energy lives)
CIR_START = 730
CIR_END   = 850

# Minimum gap (samples) between first path and second path search start
# Avoids picking sidelobes of the first path (~15 samples = one pulse width)
MIN_GAP_SAMPLES = 15

# Peak detection parameters (tune if needed)
PEAK_PROMINENCE = 30   # minimum prominence to count as a real peak
PEAK_HEIGHT     = 10   # minimum absolute amplitude

print("=" * 80)
print("SECOND PATH FEATURE EXTRACTION")
print("=" * 80)
print()

# ── Load cleaned data ──────────────────────────────────────────────────────────
print("Loading cleaned dataset...")
csv_files = sorted(DATA_DIR.glob("uwb_cleaned_dataset_part*.csv"))
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print(f"✓ Loaded {len(df):,} samples")
print()

CIR_COLUMNS   = [f"CIR{i}" for i in range(1016)]
FOCUS_COLUMNS = [f"CIR{i}" for i in range(CIR_START, CIR_END)]

# ── Second path peak detection ─────────────────────────────────────────────────
print("Extracting second-path peaks from CIR data...")
print("  (This may take ~30-60 seconds for 41k samples...)")

cir_array  = df[FOCUS_COLUMNS].values   # shape: (N, 120)
fp_idx_arr = df["FP_IDX"].values        # absolute CIR index of first path

peak2_abs_idx  = np.full(len(df), np.nan)   # absolute CIR index of 2nd peak
peak2_amp      = np.full(len(df), np.nan)   # amplitude of 2nd peak
peak2_found    = np.zeros(len(df), dtype=bool)

for i in range(len(df)):
    fp_abs   = int(fp_idx_arr[i])
    # Convert FP absolute index into an offset within the FOCUS window
    fp_local = fp_abs - CIR_START          # position inside our 120-sample window

    # Search ONLY after first path + MIN_GAP_SAMPLES
    search_start = max(fp_local + MIN_GAP_SAMPLES, 0)
    search_start = min(search_start, len(FOCUS_COLUMNS) - 1)

    if search_start >= len(FOCUS_COLUMNS) - 1:
        continue  # no room to search — leave as NaN

    search_window = cir_array[i, search_start:]

    peaks, props = find_peaks(
        search_window,
        prominence=PEAK_PROMINENCE,
        height=PEAK_HEIGHT
    )

    if len(peaks) > 0:
        # Take the most prominent peak as the second path
        best = np.argmax(props["prominences"])
        local_peak_idx = search_start + peaks[best]          # within focus window
        peak2_abs_idx[i]  = CIR_START + local_peak_idx       # absolute CIR index
        peak2_amp[i]      = search_window[peaks[best]]
        peak2_found[i]    = True

found_rate = peak2_found.mean() * 100
print(f"✓ Second path found in {peak2_found.sum():,} / {len(df):,} samples ({found_rate:.1f}%)")
print()

# ── Attach to dataframe ────────────────────────────────────────────────────────
df["PEAK2_IDX"]       = peak2_abs_idx
df["PEAK2_AMP"]       = peak2_amp
df["PEAK2_FOUND"]     = peak2_found.astype(int)

# Gap in samples between first and second path
df["PEAK2_GAP"]       = df["PEAK2_IDX"] - df["FP_IDX"]

# Estimated second-path range = Path1 range + extra distance from peak gap
# Extra distance = gap_samples × meters_per_sample
df["RANGE_PATH2_EST"] = df["RANGE"] + df["PEAK2_GAP"] * METERS_PER_SAMPLE

# Path 2 label is ALWAYS NLOS=1 per brief
df["PATH2_NLOS"] = 1

# ── Fill NaN for samples where no second peak was found ───────────────────────
# Use median imputation so ML models don't break on NaN
gap_median = df["PEAK2_GAP"].median()
amp_median = df["PEAK2_AMP"].median()
idx_median = df["PEAK2_IDX"].median()

df["PEAK2_GAP"]       = df["PEAK2_GAP"].fillna(gap_median)
df["PEAK2_AMP"]       = df["PEAK2_AMP"].fillna(amp_median)
df["PEAK2_IDX"]       = df["PEAK2_IDX"].fillna(idx_median)
df["RANGE_PATH2_EST"] = df["RANGE"] + df["PEAK2_GAP"] * METERS_PER_SAMPLE

print("Second path feature statistics:")
print(f"  PEAK2_IDX  — mean: {df['PEAK2_IDX'].mean():.1f},  std: {df['PEAK2_IDX'].std():.1f}")
print(f"  PEAK2_AMP  — mean: {df['PEAK2_AMP'].mean():.1f},  std: {df['PEAK2_AMP'].std():.1f}")
print(f"  PEAK2_GAP  — mean: {df['PEAK2_GAP'].mean():.1f},  std: {df['PEAK2_GAP'].std():.1f}")
print(f"  RANGE_P1   — mean: {df['RANGE'].mean():.2f} m,  std: {df['RANGE'].std():.2f} m")
print(f"  RANGE_P2   — mean: {df['RANGE_PATH2_EST'].mean():.2f} m,  std: {df['RANGE_PATH2_EST'].std():.2f} m")
print()

# ── Build feature matrix for regression ───────────────────────────────────────
# Core features (same as classifier) + new second-path features
core_features = [
    "FP_IDX", "FP_AMP1", "FP_AMP2", "FP_AMP3",
    "STDEV_NOISE", "CIR_PWR", "MAX_NOISE", "RXPACC",
    "CH", "FRAME_LEN", "PREAM_LEN", "BITRATE", "PRFR",
    "SNR", "SNR_dB",
    "PEAK2_IDX", "PEAK2_AMP", "PEAK2_GAP", "PEAK2_FOUND",
]

cir_features     = [f"CIR{i}" for i in range(CIR_START, CIR_END)]
feature_columns  = core_features + cir_features

X = df[feature_columns].values

# Two regression targets
y_range_p1 = df["RANGE"].values             # Path 1 actual measured range
y_range_p2 = df["RANGE_PATH2_EST"].values   # Path 2 estimated range
y_class_p1 = df["NLOS"].values              # Path 1 LOS/NLOS label (existing)
# Path 2 label is always 1, so no need to save a y_class_p2 array

# ── Train/test split (same seed + ratio as classifier for consistency) ─────────
print("Splitting data (80/20, stratified on Path1 class)...")

(X_train, X_test,
 y_p1_train, y_p1_test,
 y_p2_train, y_p2_test) = train_test_split(
    X, y_range_p1, y_range_p2,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_class_p1
)

print(f"  Training: {len(X_train):,} samples")
print(f"  Test:     {len(X_test):,} samples")
print()

# ── Save ───────────────────────────────────────────────────────────────────────
print("Saving second-path datasets...")

np.save(OUTPUT_DIR / "X_train_regression.npy",    X_train)
np.save(OUTPUT_DIR / "X_test_regression.npy",     X_test)
np.save(OUTPUT_DIR / "y_range_p1_train.npy",      y_p1_train)
np.save(OUTPUT_DIR / "y_range_p1_test.npy",       y_p1_test)
np.save(OUTPUT_DIR / "y_range_p2_train.npy",      y_p2_train)
np.save(OUTPUT_DIR / "y_range_p2_test.npy",       y_p2_test)

# Save feature names
with open(OUTPUT_DIR / "regression_feature_names.txt", "w") as f:
    f.write("REGRESSION FEATURE NAMES\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Core + Second-Path Features ({len(core_features)}):\n")
    for i, feat in enumerate(core_features, 1):
        f.write(f"  {i:2d}. {feat}\n")
    f.write(f"\nCIR Features ({len(cir_features)}):\n")
    f.write(f"  Range: CIR{CIR_START} to CIR{CIR_END - 1}\n")
    f.write(f"\nTotal features: {len(feature_columns)}\n")
    f.write(f"\nTarget 1: RANGE (Path 1 measured range in metres)\n")
    f.write(f"Target 2: RANGE_PATH2_EST (Path 2 estimated range in metres)\n")
    f.write(f"\nPath 2 class label: always NLOS=1 (per project brief)\n")
    f.write(f"Samples with peak2 detected: {peak2_found.sum():,} / {len(df):,} ({found_rate:.1f}%)\n")

print(f"✓ X_train_regression.npy      {X_train.shape}")
print(f"✓ X_test_regression.npy       {X_test.shape}")
print(f"✓ y_range_p1_train/test.npy")
print(f"✓ y_range_p2_train/test.npy")
print(f"✓ regression_feature_names.txt")
print()
print("=" * 80)
print("SECOND PATH EXTRACTION COMPLETE — ready for range_regressor.py")
print("=" * 80)

