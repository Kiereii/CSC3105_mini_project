# Pair-Level XGBoost Training Script (Pseudocode)

```text
BEGIN SCRIPT "XG_pair_classifier"

STEP 0: CONFIGURATION / ENVIRONMENT SETUP
  Set plotting style and suppress warnings
  Read env vars with defaults:
    RUN_NAME, RANDOM_SEED, USE_GPU
    RUN_TUNING, PAIR_BAYES_TRIALS, PAIR_CV_SPLITS
    Base XGBoost params:
      N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE,
      COLSAMPLE_BYTREE, MIN_CHILD_WEIGHT, GAMMA, REG_ALPHA, REG_LAMBDA
    Error-analysis thresholds:
      ERROR_BOUNDARY_LOW, ERROR_BOUNDARY_HIGH,
      CONFIDENT_ERROR_LOW, CONFIDENT_ERROR_HIGH
  Derive:
    XGB_DEVICE = "cuda" if USE_GPU else "cpu"
    XGB_TREE_METHOD = "hist"
  Define CLASS_NAMES = ["LOS+NLOS", "NLOS+NLOS"]
  Define paths:
    DATA_DIR = runs/RUN_NAME/preprocessed_data
    OUTPUT_DIR = runs/RUN_NAME/models/pair_classifier
  Ensure OUTPUT_DIR exists

STEP 1: LOAD PAIR DATASET
  Load:
    X_train_pair.npy -> X_train
    X_test_pair.npy -> X_test
    y_train_pair.npy -> y_train
    y_test_pair.npy -> y_test

  Define load_pair_feature_names(path, n_features):
    Parse pair_feature_names.txt
    Read "Core + Second-Path Features" names
    Parse explicit CIR range from "Range: CIRxxx to CIRyyy"
    Append CIR names for that range
    Fallback CIR730..CIR849 if parsing/range missing
    If parsed count != n_features:
      return generic names f0..f(n_features-1)
    Else:
      return parsed names

  Set feature_names = load_pair_feature_names(..., X_train column count)
  Print train/test counts and class distribution

STEP 2: OPTIONAL OPTUNA TUNING
  Compute class counts from y_train:
    n_class0 = count(y_train == 0)
    n_class1 = count(y_train == 1)
  Compute scale_pos_weight = n_class0 / max(n_class1, 1)

  Build default_params from env values
  Initialize:
    best_params = default_params
    best_cv_f1 = null
    tuning_time = 0
    tuning_used = false

  If RUN_TUNING is true:
    Try importing Optuna
    If unavailable:
      keep default_params
    Else:
      tuning_used = true
      Create StratifiedKFold(PAIR_CV_SPLITS, shuffle=true, seed=RANDOM_SEED)
      Create Optuna study(direction=maximize, sampler=TPE(seed=RANDOM_SEED))

      Define objective(trial):
        Sample XGBoost params:
          n_estimators, max_depth, learning_rate,
          min_child_weight, gamma, subsample,
          colsample_bytree, reg_alpha, reg_lambda
        Build temporary XGBClassifier with:
          sampled params + binary:logistic + logloss +
          tree_method/device/scale_pos_weight/random_state
        Compute mean CV F1 via cross_val_score
        Return mean CV F1

      Run study.optimize for PAIR_BAYES_TRIALS
      Save best_params, best_cv_f1, tuning_time
  Else:
    Use default_params directly

STEP 2B: FINAL MODEL TRAINING
  Build final XGBClassifier using best_params and fixed settings
  Fit model on X_train, y_train
  Record train_time

STEP 3: PREDICT + EVALUATE
  y_proba_xgb = predict_proba(X_test) for positive class
  y_pred_xgb = threshold(y_proba_xgb, 0.5)

  Compute metrics:
    accuracy, precision, recall, f1, roc_auc
  Print classification report

  Per-path mapping rule (project brief):
    if pair_label=0 (LOS+NLOS)  -> Path1=LOS,  Path2=NLOS
    if pair_label=1 (NLOS+NLOS) -> Path1=NLOS, Path2=NLOS
  Summarize Path1 LOS/NLOS counts and Path2 always-NLOS count

STEP 4: CONFUSION MATRIX PLOT
  Compute cm = confusion_matrix(y_test, y_pred_xgb)
  Plot heatmap with counts and per-cell percentages
  Save OUTPUT_DIR/pair_confusion_matrices.png

STEP 5: ROC CURVE PLOT
  Compute fpr/tpr from roc_curve(y_test, y_proba_xgb)
  Plot random baseline and model ROC curve with AUC label
  Save OUTPUT_DIR/pair_roc_curve.png

STEP 6: FEATURE IMPORTANCE + CATEGORY BREAKDOWN
  Build importance table (feature_names + model.feature_importances_)
  Sort descending and save CSV:
    OUTPUT_DIR/pair_feature_importance_xgb.csv

  Plot top-20 feature importance bars with colors:
    PEAK2* -> red, CIR* -> blue, others -> gray
  Save OUTPUT_DIR/pair_feature_importance_xgb.png

  Aggregate category contributions:
    peak2_imp = sum importance where feature starts with "PEAK2"
    cir_imp   = sum importance where feature starts with "CIR"
    core_imp  = max(0, 1 - peak2_imp - cir_imp)
  Plot category bar chart and save:
    OUTPUT_DIR/pair_importance_by_category.png

STEP 7: ERROR ANALYSIS
  error_mask = (y_pred_xgb != y_test)
  correct_mask = inverse(error_mask)

  Compute:
    n_errors, n_correct, n_fp, n_fn
    error_probs for misclassified samples
    near_boundary errors within [min(boundary), max(boundary)]
    confident_errors outside confidence thresholds

  If both correct and error samples exist:
    Compute per-feature mean difference between error and correct groups
    Rank features by absolute difference

  Write OUTPUT_DIR/pair_error_analysis_xgb.txt including:
    totals, FP/FN, boundary/confident error stats,
    error probability summary, top-10 distinguishing features

STEP 8: SAVE OUTPUTS
  Save arrays:
    y_pred_xgb.npy, y_proba_xgb.npy, y_test_pair.npy
  Save model:
    pair_xgb_model.pkl
  Save one-row metrics CSV:
    pair_metrics.csv
  Write comprehensive text summary:
    pair_results.txt
    (config, tuning info, metrics, confusion matrix,
     per-path classification, feature importance breakdown, top features)

FINAL SUMMARY
  Print key metrics (Accuracy, F1, ROC-AUC)
  Print list of generated output files

END SCRIPT
```
