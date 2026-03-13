# XGBoost Tuning Cheat Sheet

## Quick Diagnosis

- Overfitting (train high, val/test worse):
  - Decrease `max_depth`
  - Increase `min_child_weight`
  - Increase `gamma`
  - Decrease `subsample` / `colsample_bytree` a bit (for example `0.6-0.9`)
  - Increase `reg_alpha` / `reg_lambda`
  - Lower `learning_rate` and/or use early stopping

- Underfitting (both train and val low):
  - Increase `max_depth`
  - Decrease `min_child_weight`
  - Decrease `gamma`
  - Increase `subsample` / `colsample_bytree` toward `1.0`
  - Reduce regularization (`reg_alpha`, `reg_lambda`)
  - Increase `n_estimators` (or slightly increase `learning_rate`)

## Parameter Roles (Mental Model)

- `max_depth`: tree complexity knob
- `min_child_weight`: split conservativeness knob
- `gamma`: split only if worth it threshold
- `subsample`: row randomness and variance control
- `colsample_bytree`: feature randomness and variance control
- `reg_alpha` (L1): sparse, stronger pruning-like regularization
- `reg_lambda` (L2): smooths leaf weights
- `learning_rate` + `n_estimators`: paired knobs (small learning rate usually needs more trees)

## Good Starting Ranges (Binary Classification)

- `max_depth`: `3-8`
- `learning_rate`: `0.02-0.15`
- `n_estimators`: `200-1000` (with early stopping ideally)
- `min_child_weight`: `1-10`
- `gamma`: `0-1`
- `subsample`: `0.6-1.0`
- `colsample_bytree`: `0.6-1.0`
- `reg_alpha`: `0-1` (can go higher if noisy)
- `reg_lambda`: `0.5-10`

## Tuning Order (Simple and Effective)

1. Capacity first: `max_depth`, `min_child_weight`, `gamma`
2. Randomization: `subsample`, `colsample_bytree`
3. Regularization: `reg_alpha`, `reg_lambda`
4. Learning dynamics: `learning_rate` + `n_estimators`
5. Class balance and threshold: `scale_pos_weight`, decision threshold

## Fast If-Then Rules

- Too many false positives: raise threshold; increase `gamma` / `min_child_weight`
- Too many false negatives: lower threshold; allow slightly more capacity
- Unstable results across seeds: increase regularization, lower depth, maybe more folds
- Training too slow: reduce `n_estimators` max, lower trials, or reduce folds (`5` to `3`)
