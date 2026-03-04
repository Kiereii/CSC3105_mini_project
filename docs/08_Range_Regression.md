# Range Regression: Predicting Distance

## The Challenge

You need **two things** for accurate positioning:
1. **Classifier:** Is this path LOS or NLOS?
2. **Regressor:** What is the measured distance?

This document covers the regressor.

---

## Two Separate Regressors

### Path 1 Regressor
- **Input:** 136 features
- **Target:** `RANGE` (measured distance in dataset)
- **Dataset:** 41,568 samples
- **Algorithm:** RandomForestRegressor (100 trees)

### Path 2 Regressor
- **Input:** 136 features
- **Target:** `RANGE_PATH2_EST` (estimated from peak detection)
- **Dataset:** 41,568 samples
- **Algorithm:** RandomForestRegressor (100 trees)

---

## Feature Engineering for Path 2

Path 2 range is **not directly in the dataset**. We must extract it:

```
1. Locate first peak (FP_IDX)
2. Skip 15 samples (avoid pulse sidelobes)
3. Find next peak using scipy.signal.find_peaks()
4. Record: PEAK2_IDX, PEAK2_AMP, PEAK2_GAP
5. Derive: RANGE_PATH2_EST = RANGE + PEAK2_GAP × 0.15m

Why 0.15m per sample?
  c = 3×10⁸ m/s (speed of light)
  T = 1.0016 ns (DW1000 sample period)
  metres/sample = (c × T) / 2 ≈ 0.15m
  (divide by 2 for round-trip)
```

---

## Results

### Path 1 Range Prediction
```
RMSE: 1.275 meters    ← typical error magnitude
MAE:  0.980 meters    ← average absolute error
R²:   0.707           ← explains 70.7% of variance
```

### Path 2 Range Prediction
```
RMSE: 1.318 meters
MAE:  1.008 meters
R²:   0.743           ← higher than Path 1!
```

### Why Path 2 R² > Path 1?
- R² is relative to variance in target
- RANGE_PATH2_EST has wider dynamic range
- RMSE is absolute; similar magnitude to Path 1

---

## Feature Importance for Range

### Top Features for Path 1
1. **FP_IDX:** Direct time-of-arrival indicator
2. **RXPACC:** Signal quality correlates with range
3. **CIR_PWR:** Signal strength weakens with distance

### Top Features for Path 2
1. **PEAK2_GAP:** Time offset between paths (directly encodes distance)
2. **PEAK2_IDX:** Absolute arrival time
3. **FP_IDX:** Reference point for first path

---

## Real-World Interpretation

**1 meter error is good for indoor positioning:**
- WiFi: ±10m error
- Bluetooth: ±5m error
- UWB (without NLOS detection): ±5-20m error
- **UWB (with NLOS detection): ±1m error** ✓

This accuracy enables room-level (1-2m) positioning, suitable for:
- Indoor navigation
- Asset tracking
- AR/VR applications
- Contact tracing

---

Next: [07_Pair_Classification.md](07_Pair_Classification.md) or [09_Model_Comparison.md](09_Model_Comparison.md)

