"""
UWB LOS/NLOS - Second Path Feature Extraction
==============================================
The brief requires classifying and estimating range for TWO dominant paths.

Path logic (from brief):
  - If Path 1 = LOS  → Path 2 is always NLOS  → Pair label = 0 (LOS+NLOS)
  - If Path 1 = NLOS → Path 2 is always NLOS  → Pair label = 1 (NLOS+NLOS)
  So Path 2 label is ALWAYS 1 (NLOS). The real value is in RANGE ESTIMATION
  and PAIR-LEVEL CLASSIFICATION.

This script:
  1. Finds the second dominant CIR peak after the first path region
  2. Engineers second-path features (index, amplitude, gap from first path)
  3. Derives Path 2 estimated range from the peak index offset
  4. Saves all features + targets ready for the range regressor
  5. Saves pair-labelled dataset for the pair_classifier
     PAIR_LABEL = 0 → LOS+NLOS  (Path 1 was LOS, a trustworthy path exists)
     PAIR_LABEL = 1 → NLOS+NLOS (Both paths obstructed, no trustworthy path)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
import os
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "Dataset" / "UWB-LOS-NLOS-Data-Set" / "dataset" / "Cleaned"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CIR hardware constants (DW1000 datasheet)
# Each CIR sample = 1 ns resolution → ~0.3 m per sample at speed of light
SPEED_OF_LIGHT = 3e8  # m/s
SAMPLE_PERIOD_NS = 1.0016  # ns per CIR sample (DW1000 spec)
METERS_PER_SAMPLE = (SPEED_OF_LIGHT * SAMPLE_PERIOD_NS * 1e-9) / 2  # two-way ToF

# Focus region (from EDA — where actual signal energy lives)
CIR_START = 730
CIR_END = 850

# Minimum gap (samples) between first path and second path search start
# Avoids picking sidelobes of the first path (~15 samples = one pulse width)
MIN_GAP_SAMPLES = 15

# Peak detection parameters (tune if needed)
PEAK_PROMINENCE = 30  # minimum prominence to count as a real peak
PEAK_HEIGHT = 10  # minimum absolute amplitude

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

CIR_COLUMNS = [f"CIR{i}" for i in range(1016)]
FOCUS_COLUMNS = [f"CIR{i}" for i in range(CIR_START, CIR_END)]

# ── Second path peak detection ─────────────────────────────────────────────────
print("Extracting second-path peaks from CIR data...")
print("  (This may take ~30-60 seconds for 41k samples...)")

cir_array = df[FOCUS_COLUMNS].values  # shape: (N, 120) — fixed window for peak detection
fp_idx_arr = df["FP_IDX"].values  # absolute CIR index of first path

# ── Align CIR window to FP_IDX ──────────────────────────────────────────────
# Instead of a fixed CIR[730:850] slice, center on each sample's first path.
# This ensures every column has the same physical meaning across samples.
PRE_FP = 30   # samples before first path (noise floor baseline)
POST_FP = 90  # samples after first path (reflection region)
ALIGNED_LEN = PRE_FP + POST_FP  # 120 total

print("Aligning CIR windows to FP_IDX...")
full_cir = df[CIR_COLUMNS].values  # (N, 1016)
cir_aligned = np.zeros((len(df), ALIGNED_LEN))

for i in range(len(df)):
    fp = int(fp_idx_arr[i])
    src_start = max(fp - PRE_FP, 0)
    src_end = min(fp + POST_FP, 1016)
    dst_start = src_start - (fp - PRE_FP)
    dst_end = dst_start + (src_end - src_start)
    cir_aligned[i, dst_start:dst_end] = full_cir[i, src_start:src_end]

aligned_names = [f"CIR_aligned_{i}" for i in range(ALIGNED_LEN)]
print(f"✓ Aligned {len(df):,} CIR windows: [{PRE_FP} before FP, {POST_FP} after FP] = {ALIGNED_LEN} samples")
print()

del full_cir  # free memory

peak2_abs_idx = np.full(len(df), np.nan)  # absolute CIR index of 2nd peak
peak2_amp = np.full(len(df), np.nan)  # amplitude of 2nd peak
peak2_found = np.zeros(len(df), dtype=bool)

for i in range(len(df)):
    fp_abs = int(fp_idx_arr[i])
    # Convert FP absolute index into an offset within the FOCUS window
    fp_local = fp_abs - CIR_START  # position inside our 120-sample window

    # Search ONLY after first path + MIN_GAP_SAMPLES
    search_start = max(fp_local + MIN_GAP_SAMPLES, 0)
    search_start = min(search_start, len(FOCUS_COLUMNS) - 1)

    if search_start >= len(FOCUS_COLUMNS) - 1:
        continue  # no room to search — leave as NaN

    search_window = cir_array[i, search_start:]

    peaks, props = find_peaks(
        search_window, prominence=PEAK_PROMINENCE, height=PEAK_HEIGHT
    )

    if len(peaks) > 0:
        # Take the most prominent peak as the second path
        best = np.argmax(props["prominences"])
        local_peak_idx = search_start + peaks[best]  # within focus window
        peak2_abs_idx[i] = CIR_START + local_peak_idx  # absolute CIR index
        peak2_amp[i] = search_window[peaks[best]]
        peak2_found[i] = True

found_rate = peak2_found.mean() * 100
print(
    f"✓ Second path found in {peak2_found.sum():,} / {len(df):,} samples ({found_rate:.1f}%)"
)
print()

# ── Attach to dataframe ────────────────────────────────────────────────────────
df["PEAK2_IDX"] = peak2_abs_idx
df["PEAK2_AMP"] = peak2_amp
df["PEAK2_FOUND"] = peak2_found.astype(int)

# Gap in samples between first and second path
df["PEAK2_GAP"] = df["PEAK2_IDX"] - df["FP_IDX"]

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

df["PEAK2_GAP"] = df["PEAK2_GAP"].fillna(gap_median)
df["PEAK2_AMP"] = df["PEAK2_AMP"].fillna(amp_median)
df["PEAK2_IDX"] = df["PEAK2_IDX"].fillna(idx_median)
df["RANGE_PATH2_EST"] = df["RANGE"] + df["PEAK2_GAP"] * METERS_PER_SAMPLE

print("Second path feature statistics:")
print(
    f"  PEAK2_IDX  — mean: {df['PEAK2_IDX'].mean():.1f},  std: {df['PEAK2_IDX'].std():.1f}"
)
print(
    f"  PEAK2_AMP  — mean: {df['PEAK2_AMP'].mean():.1f},  std: {df['PEAK2_AMP'].std():.1f}"
)
print(
    f"  PEAK2_GAP  — mean: {df['PEAK2_GAP'].mean():.1f},  std: {df['PEAK2_GAP'].std():.1f}"
)
print(
    f"  RANGE_P1   — mean: {df['RANGE'].mean():.2f} m,  std: {df['RANGE'].std():.2f} m"
)
print(
    f"  RANGE_P2   — mean: {df['RANGE_PATH2_EST'].mean():.2f} m,  std: {df['RANGE_PATH2_EST'].std():.2f} m"
)
print()

# ── Engineered features ──────────────────────────────────────────────────────
print("Engineering ratio and CIR shape features...")

# Ratio features
df["FP_AMP_ratio"] = df["FP_AMP1"] / np.maximum(df["FP_AMP2"], 1e-10)
df["SNR_per_acc"] = df["SNR"] / np.maximum(df["RXPACC"], 1e-10)
df["signal_to_noise"] = df["CIR_PWR"] / np.maximum(df["STDEV_NOISE"], 1e-10)
df["noise_ratio"] = df["MAX_NOISE"] / np.maximum(df["STDEV_NOISE"], 1e-10)
df["FP_power"] = df["FP_AMP1"] ** 2 + df["FP_AMP2"] ** 2 + df["FP_AMP3"] ** 2

# CIR shape features (computed over the aligned CIR window)
df["CIR_energy"] = np.sum(cir_aligned ** 2, axis=1)
df["CIR_kurtosis"] = kurtosis(cir_aligned, axis=1, fisher=True)
df["CIR_skewness"] = skew(cir_aligned, axis=1)


def _rise_time(row):
    peak = np.max(row)
    if peak < 1e-10:
        return 0.0
    above_10 = np.where(row >= 0.1 * peak)[0]
    above_90 = np.where(row >= 0.9 * peak)[0]
    if len(above_10) == 0 or len(above_90) == 0:
        return 0.0
    return float(above_90[0] - above_10[0])


def _count_peaks(row):
    peak = np.max(row)
    if peak < 1e-10:
        return 0.0
    peaks, _ = find_peaks(row, height=0.3 * peak)
    return float(len(peaks))


df["CIR_rise_time"] = np.array([_rise_time(r) for r in cir_aligned])
df["CIR_num_peaks"] = np.array([_count_peaks(r) for r in cir_aligned])

engineered_features = [
    "FP_AMP_ratio",
    "SNR_per_acc",
    "signal_to_noise",
    "noise_ratio",
    "FP_power",
    "CIR_energy",
    "CIR_kurtosis",
    "CIR_skewness",
    "CIR_rise_time",
    "CIR_num_peaks",
]

print(f"✓ Engineered {len(engineered_features)} features: {engineered_features}")
print()

# ── Build feature matrix for regression ───────────────────────────────────────
# Core features (same as classifier) + new second-path features
core_features = [
    "FP_IDX",
    "FP_AMP1",
    "FP_AMP2",
    "FP_AMP3",
    "STDEV_NOISE",
    "CIR_PWR",
    "MAX_NOISE",
    "RXPACC",
    "CH",
    "FRAME_LEN",
    "PREAM_LEN",
    "BITRATE",
    "PRFR",
    "SNR",
    "SNR_dB",
    "PEAK2_IDX",
    "PEAK2_AMP",
    "PEAK2_GAP",
    "PEAK2_FOUND",
] + engineered_features

cir_features = aligned_names
feature_columns = core_features + cir_features

X = np.hstack([df[core_features].values, cir_aligned])

# Two regression targets
y_range_p1 = df["RANGE"].values  # Path 1 actual measured range
y_range_p2 = df["RANGE_PATH2_EST"].values  # Path 2 estimated range
y_class_p1 = df["NLOS"].values  # Path 1 LOS/NLOS label (existing)
# Path 2 label is always 1, so no need to save a y_class_p2 array

# ── Train/test split from preprocessing indices ────────────────────────────────
print("Loading shared split indices from preprocess_data.py...")

train_idx_path = OUTPUT_DIR / "train_idx.npy"
test_idx_path = OUTPUT_DIR / "test_idx.npy"

if not train_idx_path.exists() or not test_idx_path.exists():
    raise FileNotFoundError(
        "Missing train/test indices. Run preprocess_data.py first."
    )

train_idx = np.load(train_idx_path)
test_idx = np.load(test_idx_path)

X_train, X_test = X[train_idx], X[test_idx]
y_p1_train, y_p1_test = y_range_p1[train_idx], y_range_p1[test_idx]
y_p2_train, y_p2_test = y_range_p2[train_idx], y_range_p2[test_idx]

print(f"  Training: {len(X_train):,} samples")
print(f"  Test:     {len(X_test):,} samples")
print()

# ── Save ───────────────────────────────────────────────────────────────────────
print("Saving second-path datasets...")

np.save(OUTPUT_DIR / "X_train_regression.npy", X_train)
np.save(OUTPUT_DIR / "X_test_regression.npy", X_test)
np.save(OUTPUT_DIR / "y_range_p1_train.npy", y_p1_train)
np.save(OUTPUT_DIR / "y_range_p1_test.npy", y_p1_test)
np.save(OUTPUT_DIR / "y_range_p2_train.npy", y_p2_train)
np.save(OUTPUT_DIR / "y_range_p2_test.npy", y_p2_test)

# Save feature names
with open(OUTPUT_DIR / "regression_feature_names.txt", "w") as f:
    f.write("REGRESSION FEATURE NAMES\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Core + Second-Path Features ({len(core_features)}):\n")
    for i, feat in enumerate(core_features, 1):
        f.write(f"  {i:2d}. {feat}\n")
    f.write(f"\nAligned CIR Features ({len(cir_features)}):\n")
    f.write(f"  Window: {PRE_FP} before FP_IDX to {POST_FP} after FP_IDX ({ALIGNED_LEN} samples)\n")
    f.write(f"\nTotal features: {len(feature_columns)}\n")
    f.write(f"\nTarget 1: RANGE (Path 1 measured range in metres)\n")
    f.write(f"Target 2: RANGE_PATH2_EST (Path 2 estimated range in metres)\n")
    f.write(f"\nPath 2 class label: always NLOS=1 (per project brief)\n")
    f.write(
        f"Samples with peak2 detected: {peak2_found.sum():,} / {len(df):,} ({found_rate:.1f}%)\n"
    )

print(f"✓ X_train_regression.npy      {X_train.shape}")
print(f"✓ X_test_regression.npy       {X_test.shape}")
print(f"✓ y_range_p1_train/test.npy")
print(f"✓ y_range_p2_train/test.npy")
print(f"✓ regression_feature_names.txt")
print()

# ── Build pair-level feature matrix & labels ───────────────────────────────────
# PAIR_LABEL mirrors NLOS:
#   0 = LOS+NLOS  (Path 1 is LOS → a trustworthy path exists)
#   1 = NLOS+NLOS (Path 1 is NLOS → both paths obstructed)
print("Building pair-level classification dataset...")

pair_core_features = [
    "RANGE",
    "FP_IDX",
    "FP_AMP1",
    "FP_AMP2",
    "FP_AMP3",
    "STDEV_NOISE",
    "CIR_PWR",
    "MAX_NOISE",
    "RXPACC",
    "CH",
    "FRAME_LEN",
    "PREAM_LEN",
    "BITRATE",
    "PRFR",
    "SNR",
    "SNR_dB",
    # Second-path features — these make this a TRUE pair classifier
    "PEAK2_IDX",
    "PEAK2_AMP",
    "PEAK2_GAP",
    "PEAK2_FOUND",
] + engineered_features

pair_feature_columns = pair_core_features + cir_features

X_pair = np.hstack([df[pair_core_features].values, cir_aligned])
y_pair = df["NLOS"].values  # 0=LOS+NLOS, 1=NLOS+NLOS

# Reuse the same split indices as all other tasks for consistency
X_pair_train, X_pair_test = X_pair[train_idx], X_pair[test_idx]
y_pair_train, y_pair_test = y_pair[train_idx], y_pair[test_idx]

np.save(OUTPUT_DIR / "X_train_pair.npy", X_pair_train)
np.save(OUTPUT_DIR / "X_test_pair.npy", X_pair_test)
np.save(OUTPUT_DIR / "y_train_pair.npy", y_pair_train)
np.save(OUTPUT_DIR / "y_test_pair.npy", y_pair_test)

# Save pair feature names
with open(OUTPUT_DIR / "pair_feature_names.txt", "w") as f:
    f.write("PAIR CLASSIFIER FEATURE NAMES\n")
    f.write("=" * 50 + "\n\n")
    f.write("Target: PAIR_LABEL\n")
    f.write("  0 = LOS+NLOS  (Path 1 is LOS — trustworthy path exists)\n")
    f.write("  1 = NLOS+NLOS (Both paths obstructed — no trustworthy path)\n\n")
    f.write(f"Core + Second-Path Features ({len(pair_core_features)}):\n")
    for i, feat in enumerate(pair_core_features, 1):
        f.write(f"  {i:2d}. {feat}\n")
    f.write(f"\nAligned CIR Features ({len(cir_features)}):\n")
    f.write(f"  Window: {PRE_FP} before FP_IDX to {POST_FP} after FP_IDX ({ALIGNED_LEN} samples)\n")
    f.write(f"\nTotal features: {len(pair_feature_columns)}\n")
    f.write(f"\nClass balance:\n")
    f.write(
        f"  LOS+NLOS  (0): {(y_pair == 0).sum():,} ({(y_pair == 0).mean() * 100:.1f}%)\n"
    )
    f.write(
        f"  NLOS+NLOS (1): {(y_pair == 1).sum():,} ({(y_pair == 1).mean() * 100:.1f}%)\n"
    )

los_nlos_count = (y_pair == 0).sum()
nlos_nlos_count = (y_pair == 1).sum()
print(
    f"✓ Pair labels — LOS+NLOS: {los_nlos_count:,}  |  NLOS+NLOS: {nlos_nlos_count:,}"
)
print(f"✓ X_train_pair.npy        {X_pair_train.shape}")
print(f"✓ X_test_pair.npy         {X_pair_test.shape}")
print(f"✓ y_train_pair.npy / y_test_pair.npy")
print(f"✓ pair_feature_names.txt")
print()
print("=" * 80)
print("SECOND PATH EXTRACTION COMPLETE")
print("  → task2/experimental regression scripts use X_train_regression.npy")
print("  → pair_classifier.py  uses X_train_pair.npy")
print("=" * 80)
