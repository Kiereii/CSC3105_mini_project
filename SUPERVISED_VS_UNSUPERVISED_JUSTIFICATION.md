# Justification for Supervised Learning as the Primary Approach for UWB LOS/NLOS Classification

## Executive Summary

This document provides a comprehensive justification for selecting **supervised learning** as the primary and superior approach over unsupervised learning for Ultra-Wideband (UWB) Line-of-Sight (LOS) / Non-Line-of-Sight (NLOS) classification tasks.

**Project Context**: This CSC3105 Data Analytics mini project focuses on UWB indoor positioning with two explicit predictive objectives:
1. **Classify LOS vs NLOS** (binary labeled classes)
2. **Predict range values** for path(s) (continuous labeled targets)

The dataset contains **42,000 samples** with a **balanced LOS/NLOS class distribution**.

---

## 1. Problem Framing: Predictive vs. Exploratory

| Aspect | Supervised Learning | Unsupervised Learning |
|--------|---------------------|----------------------|
| Problem Type | Predictive (classification/regression) | Exploratory (clustering) |
| Target Variable | Known labels (LOS/NLOS) | No labels |
| Training Signal | Direct error feedback | Indirect metrics |
| Goal Alignment | Matches prediction objective | Discovers patterns |
| Decision Boundary | Explicit from examples | Implicit, may not match LOS/NLOS |

**Project-Specific Example**: Each CIR measurement has a binary LOS/NLOS label from data collection.

---

## 2. Label Availability and Feature Relationships

| Feature Type | Correlation | Example |
|--------------|-------------|---------|
| CIR Features | High | First path power, rise time |
| Signal Quality | High | SNR, RSSI |
| Second-Path | High | Delay of second arrival |

42,000 labeled samples make supervised learning statistically appropriate.

---

## 3. Classification and Regression Tasks

- **Classification**: LOS vs NLOS binary prediction
- **Regression**: Range estimation with CIR features

Pipeline: CIR Data -> Features -> Classification -> Range Correction

---

## 4. Evaluation Metrics

| Metric Category | Metrics | Target |
|-----------------|---------|--------|
| Classification | Recall, F1, ROC-AUC | Recall > 0.95 |
| Regression | MAE, RMSE | Error < 10cm |

| Dimension | Supervised | Unsupervised |
|-----------|------------|--------------|
| Objective | Optimizes LOS/NLOS targets | Optimizes cluster geometry |
| Safety | Quantifies NLOS false negatives | No native FN concept |

---

## 5. Safety-Critical Error Costs

| Error Type | Cost |
|------------|------|
| NLOS misclassified as LOS | CRITICAL - collision risk |
| LOS misclassified as NLOS | Moderate - conservative fallback |

Threshold tuning (e.g., 0.38 vs 0.5) optimizes for NLOS recall.

---

## 6. Bias-Variance and Generalization

| Technique | Purpose |
|-----------|---------|
| Regularization | Prevent overfitting |
| Cross-validation | Generalization estimates |
| Ensemble methods | Reduce variance |

| Split Strategy | Leakage Risk | Recommendation |
|----------------|--------------|----------------|
| Random k-fold | High | Avoid |
| Environment-based | Low | Recommended |

---

## 7. Leakage Prevention

| Leakage Type | Prevention |
|--------------|------------|
| Sample-level | Deduplicate |
| Environment-level | Hold out environments |
| Device-level | Device-stratified splitting |

---

## 8. Interpretability

| Tool | Application |
|------|-------------|
| SHAP | Feature contributions |
| Feature Importance | Global ranking |
| Partial Dependence | Feature effects |

---

## 9. Thresholding and Calibration

| Method | Best For |
|--------|----------|
| Platt Scaling | Small datasets |
| Isotonic | Large datasets |
| Temperature Scaling | Neural networks |

---

## 10. Reproducibility

- Fixed random seeds
- Version control
- Repeated experiments with CI

---

## 11. Deployment

| Requirement | Solution |
|-------------|----------|
| Latency | < 10ms |
| Memory | < 10MB |
| Monitoring | Drift detection |

---

## 12. Unsupervised as Support

Unsupervised methods support:
- EDA and visualization
- Anomaly detection
- Data quality checks
- Feature engineering

---

## 13. When Unsupervised is Preferable

| Scenario | Justification |
|----------|---------------|
| No labels | Cannot use supervised |
| Exploration | Discover unknown patterns |
| Anomaly detection | Find rare failures |

---

## 14. Conclusion

Supervised learning is superior because:
1. Labels available (42,000 samples)
2. Safety requires asymmetric error handling
3. Clear metrics and interpretability
4. Proven deployment practices

**Recommendation**: Use supervised learning as primary approach.

---

## Appendix: Viva Checklist

- [ ] Explain bias-variance tradeoff
- [ ] Justify metrics (recall > accuracy)
- [ ] Describe environment-based splitting
- [ ] Explain NLOS FN costs
- [ ] Show confusion matrix
- [ ] Present feature importance
- [ ] Describe threshold tuning
- [ ] Explain when unsupervised preferred

