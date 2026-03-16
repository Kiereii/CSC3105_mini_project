"""
Generate all report plots into one folder.

Output folder:
  runs/<RUN_NAME>/report_plots/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import validation_curve

try:
    import joblib

    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False


def load_dataset(dataset_dir: Path, data_type: str) -> pd.DataFrame:
    pattern = (
        "uwb_cleaned_dataset_part*.csv"
        if data_type.lower() == "cleaned"
        else "uwb_dataset_part*.csv"
    )
    files = sorted(dataset_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No dataset files found with pattern: {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def get_available_classification_models(models_dir: Path):
    candidates = {
        "Random Forest": {
            "pred": models_dir / "random_forest" / "y_pred.npy",
            "proba": models_dir / "random_forest" / "y_pred_proba.npy",
            "color": "#2ecc71",
        },
        "Logistic Regression": {
            "pred": models_dir / "logreg_svm" / "y_pred_lr.npy",
            "proba": models_dir / "logreg_svm" / "y_proba_lr.npy",
            "color": "#e74c3c",
        },
        "SVM (LinearSVC)": {
            "pred": models_dir / "logreg_svm" / "y_pred_svm.npy",
            "proba": models_dir / "logreg_svm" / "y_proba_svm.npy",
            "color": "#9b59b6",
        },
        "XGBoost": {
            "pred": models_dir / "xgboost" / "y_pred_xgb.npy",
            "proba": models_dir / "xgboost" / "y_pred_proba_xgb.npy",
            "color": "#f39c12",
        },
    }

    available = {}
    for name, spec in candidates.items():
        if spec["pred"].exists() and spec["proba"].exists():
            available[name] = spec
    return available


def plot_1_class_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["NLOS"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["LOS (0)", "NLOS (1)"], counts.values, color=["#2ecc71", "#e74c3c"])
    total = counts.sum()
    for bar, cnt in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{cnt:,} ({cnt / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("Class Distribution (LOS vs NLOS)", fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, out_dir / "01_class_distribution.png")


def plot_2_feature_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    selected = [
        "NLOS",
        "RANGE",
        "FP_IDX",
        "FP_AMP1",
        "FP_AMP2",
        "FP_AMP3",
        "STDEV_NOISE",
        "CIR_PWR",
        "MAX_NOISE",
        "RXPACC",
        "SNR",
        "SNR_dB",
    ]
    selected = [c for c in selected if c in df.columns]
    corr = df[selected].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    save_fig(fig, out_dir / "02_feature_correlation_heatmap.png")


def plot_3_feature_distributions_by_class(df: pd.DataFrame, out_dir: Path) -> None:
    feature_list = [
        f for f in ["RANGE", "FP_IDX", "FP_AMP1", "SNR_dB"] if f in df.columns
    ]
    n = len(feature_list)
    if n == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for i, feat in enumerate(feature_list):
        sns.histplot(
            data=df,
            x=feat,
            hue="NLOS",
            bins=40,
            kde=True,
            stat="density",
            common_norm=False,
            ax=axes[i],
            palette={0: "#2ecc71", 1: "#e74c3c"},
        )
        axes[i].set_title(f"{feat} by Class", fontweight="bold")
        axes[i].grid(alpha=0.3)

    for j in range(i + 1, 4):
        axes[j].axis("off")

    fig.suptitle("Feature Distributions by Class", fontsize=14, fontweight="bold")
    save_fig(fig, out_dir / "03_feature_distributions_by_class.png")


def plot_4_average_cir_waveforms(df: pd.DataFrame, out_dir: Path) -> None:
    cir_columns = [f"CIR{i}" for i in range(1016) if f"CIR{i}" in df.columns]
    focus_cols = [f"CIR{i}" for i in range(730, 850) if f"CIR{i}" in df.columns]
    if not cir_columns:
        return

    los = df[df["NLOS"] == 0]
    nlos = df[df["NLOS"] == 1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(
        los[cir_columns].mean().values, color="#2ecc71", label="LOS", linewidth=2
    )
    axes[0].plot(
        nlos[cir_columns].mean().values, color="#e74c3c", label="NLOS", linewidth=2
    )
    axes[0].set_title("Average CIR Waveform (Full 0-1015)", fontweight="bold")
    axes[0].set_xlabel("CIR Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if focus_cols:
        x = np.arange(730, 730 + len(focus_cols))
        axes[1].plot(
            x,
            los[focus_cols].mean().values,
            color="#2ecc71",
            label="LOS",
            linewidth=2.5,
        )
        axes[1].plot(
            x,
            nlos[focus_cols].mean().values,
            color="#e74c3c",
            label="NLOS",
            linewidth=2.5,
        )
        axes[1].set_title("Average CIR Waveform (Focused 730-850)", fontweight="bold")
        axes[1].set_xlabel("CIR Index")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.suptitle("Average CIR Waveform: LOS vs NLOS", fontsize=14, fontweight="bold")
    save_fig(fig, out_dir / "04_average_cir_waveform_los_vs_nlos.png")


def build_classification_metrics(y_test: np.ndarray, available_models: dict):
    y_pred = {name: np.load(spec["pred"]) for name, spec in available_models.items()}
    y_proba = {name: np.load(spec["proba"]) for name, spec in available_models.items()}

    rows = []
    for name in y_pred:
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred[name]),
                "Precision": precision_score(y_test, y_pred[name]),
                "Recall": recall_score(y_test, y_pred[name]),
                "F1-Score": f1_score(y_test, y_pred[name]),
                "ROC-AUC": roc_auc_score(y_test, y_proba[name]),
            }
        )
    metrics_df = pd.DataFrame(rows)
    return y_pred, y_proba, metrics_df


def plot_5_model_comparison_bar(
    metrics_df: pd.DataFrame, available_models: dict, out_dir: Path
) -> None:
    model_names = metrics_df["Model"].tolist()
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metric_names))
    width = 0.14
    offsets = np.linspace(
        -(width * (len(model_names) - 1) / 2),
        (width * (len(model_names) - 1) / 2),
        len(model_names),
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model_name in enumerate(model_names):
        scores = [
            metrics_df.loc[metrics_df["Model"] == model_name, m].values[0]
            for m in metric_names
        ]
        bars = ax.bar(
            x + offsets[i],
            scores,
            width,
            label=model_name,
            color=available_models[model_name]["color"],
            alpha=0.85,
        )
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Bar Chart", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, out_dir / "05_model_comparison_bar_chart.png")


def plot_6_combined_roc(
    y_test: np.ndarray, y_proba: dict, available_models: dict, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")

    for name, probs in y_proba.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=available_models[name]["color"],
            label=f"{name} (AUC={auc:.4f})",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Combined ROC Curve (All Models)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir / "06_combined_roc_curve_all_models.png")


def plot_7_confusion_matrices(y_test: np.ndarray, y_pred: dict, out_dir: Path) -> None:
    n_models = len(y_pred)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.3 * n_cols, 5.1 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    for idx, (name, preds) in enumerate(y_pred.items()):
        cm = confusion_matrix(y_test, preds)
        ax = axes_flat[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=0.5,
            xticklabels=["LOS", "NLOS"],
            yticklabels=["LOS", "NLOS"],
            ax=ax,
        )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Confusion Matrix Heatmaps", fontsize=14, fontweight="bold")
    save_fig(fig, out_dir / "07_confusion_matrix_heatmaps.png")


def plot_8_feature_importance(models_dir: Path, out_dir: Path) -> None:
    xgb_csv = models_dir / "xgboost" / "feature_importance_xgb.csv"
    rf_csv = models_dir / "random_forest" / "feature_importance_ranking.csv"

    if xgb_csv.exists():
        imp_df = pd.read_csv(xgb_csv).head(20)
        value_col = "Importance"
        title = "Feature Importance Top 20 (XGBoost)"
    elif rf_csv.exists():
        imp_df = pd.read_csv(rf_csv).head(20)
        value_col = "Importance"
        title = "Feature Importance Top 20 (Random Forest)"
    else:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    ordered = imp_df.iloc[::-1]
    ax.barh(ordered["Feature"], ordered[value_col], color="#f39c12", alpha=0.9)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    save_fig(fig, out_dir / "08_feature_importance_top20.png")


def plot_9_validation_curve(prep_dir: Path, out_dir: Path) -> None:
    try:
        from xgboost import XGBClassifier
    except Exception:
        return

    x_train = np.load(prep_dir / "X_train_unscaled.npy")
    y_train = np.load(prep_dir / "y_train.npy")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    param_range = np.array([3, 4, 5, 6, 8, 10])

    train_scores, val_scores = validation_curve(
        model,
        x_train,
        y_train,
        param_name="max_depth",
        param_range=param_range,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_range, train_mean, marker="o", label="Train F1", color="#2d7dd2")
    ax.plot(param_range, val_mean, marker="s", label="Validation F1", color="#d1495b")
    ax.fill_between(
        param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color="#d1495b"
    )
    ax.set_xlabel("max_depth")
    ax.set_ylabel("F1 Score")
    ax.set_title("Validation Curve (XGBoost)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir / "09_validation_curve.png")


def plot_10_threshold_curve(run_root: Path, prep_dir: Path, out_dir: Path) -> None:
    if not HAS_JOBLIB:
        return

    x_val = np.load(prep_dir / "X_val_unscaled.npy")
    y_val = np.load(prep_dir / "y_val.npy")

    model_path = run_root / "models" / "xgboost" / "xgboost_model.pkl"
    if not model_path.exists():
        return

    model = joblib.load(model_path)
    val_proba = model.predict_proba(x_val)[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        pred = (val_proba >= t).astype(int)
        precisions.append(precision_score(y_val, pred, zero_division=0))
        recalls.append(recall_score(y_val, pred, zero_division=0))
        f1s.append(f1_score(y_val, pred, zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#2d7dd2")
    ax.plot(thresholds, recalls, label="Recall", color="#f4a259")
    ax.plot(thresholds, f1s, label="F1", color="#3bb273", linewidth=2)
    ax.axvline(
        best_t, linestyle="--", color="black", label=f"Best threshold={best_t:.2f}"
    )
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Optimization Curve", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, out_dir / "10_threshold_optimization_curve.png")


def plot_11_predicted_vs_actual(run_root: Path, prep_dir: Path, out_dir: Path) -> None:
    reg_dir = run_root / "models" / "range_regressor"
    if not reg_dir.exists():
        return

    y_p1_test = np.load(prep_dir / "y_range_p1_test.npy")
    y_p2_test = np.load(prep_dir / "y_range_p2_test.npy")

    model_keys = []
    for key in ["rf", "knn", "xgb"]:
        if (reg_dir / f"y_p1_pred_{key}.npy").exists() and (
            reg_dir / f"y_p2_pred_{key}.npy"
        ).exists():
            model_keys.append(key)

    if not model_keys:
        return

    name_map = {"rf": "RandomForest", "knn": "KNN", "xgb": "XGBoost"}
    fig, axes = plt.subplots(
        2, len(model_keys), figsize=(6 * len(model_keys), 10), squeeze=False
    )

    for col, key in enumerate(model_keys):
        y_p1_pred = np.load(reg_dir / f"y_p1_pred_{key}.npy")
        y_p2_pred = np.load(reg_dir / f"y_p2_pred_{key}.npy")

        ax1 = axes[0, col]
        ax1.scatter(y_p1_test, y_p1_pred, alpha=0.25, s=8, color="#2d7dd2")
        lims1 = [
            min(y_p1_test.min(), y_p1_pred.min()),
            max(y_p1_test.max(), y_p1_pred.max()),
        ]
        ax1.plot(lims1, lims1, "k--", linewidth=1.5)
        ax1.set_xlabel("Actual Range (m)")
        ax1.set_ylabel("Predicted Range (m)")
        ax1.set_title(f"Path 1 - {name_map[key]}", fontweight="bold")
        ax1.grid(alpha=0.3)

        ax2 = axes[1, col]
        ax2.scatter(y_p2_test, y_p2_pred, alpha=0.25, s=8, color="#d1495b")
        lims2 = [
            min(y_p2_test.min(), y_p2_pred.min()),
            max(y_p2_test.max(), y_p2_pred.max()),
        ]
        ax2.plot(lims2, lims2, "k--", linewidth=1.5)
        ax2.set_xlabel("Actual Range (m)")
        ax2.set_ylabel("Predicted Range (m)")
        ax2.set_title(f"Path 2 - {name_map[key]}", fontweight="bold")
        ax2.grid(alpha=0.3)

    fig.suptitle("Predicted vs Actual Distance", fontsize=14, fontweight="bold")
    save_fig(fig, out_dir / "11_predicted_vs_actual_distance.png")


def plot_12_regression_comparison(run_root: Path, out_dir: Path) -> None:
    csv_path = (
        run_root / "models" / "range_regressor" / "regression_model_comparison.csv"
    )
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    metrics = ["rmse", "mae", "r2"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        pivot = df.pivot(index="model", columns="path", values=metric)
        pivot.plot(kind="bar", ax=ax, alpha=0.9, color=["#2d7dd2", "#d1495b"])
        ax.set_title(f"{metric.upper()} by Model and Path", fontweight="bold")
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Path")

    fig.suptitle("Regression Model Comparison", fontsize=14, fontweight="bold")
    save_fig(fig, out_dir / "12_regression_model_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all report plots in one folder."
    )
    parser.add_argument(
        "--run-name", default=os.getenv("RUN_NAME", "split_env_70_15_15_seed42")
    )
    parser.add_argument(
        "--data-type",
        default="Cleaned",
        choices=["Cleaned", "Raw", "cleaned", "raw"],
        help="Dataset split to use for EDA plots.",
    )
    args = parser.parse_args()

    run_root = Path("./runs") / args.run_name
    prep_dir = run_root / "preprocessed_data"
    models_dir = run_root / "models"
    out_dir = run_root / "report_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("deep")

    dataset_dir = (
        Path("./Dataset/UWB-LOS-NLOS-Data-Set/dataset") / args.data_type.capitalize()
    )
    df = load_dataset(dataset_dir, args.data_type)

    print("Generating 1-4 (EDA plots)...")
    plot_1_class_distribution(df, out_dir)
    plot_2_feature_correlation_heatmap(df, out_dir)
    plot_3_feature_distributions_by_class(df, out_dir)
    plot_4_average_cir_waveforms(df, out_dir)

    y_test_path = prep_dir / "y_test.npy"
    if y_test_path.exists():
        y_test = np.load(y_test_path)
        available_models = get_available_classification_models(models_dir)
        if available_models:
            print("Generating 5-8 (classification evaluation plots)...")
            y_pred, y_proba, metrics_df = build_classification_metrics(
                y_test, available_models
            )
            metrics_df.to_csv(
                out_dir / "classification_metrics_summary.csv", index=False
            )
            plot_5_model_comparison_bar(metrics_df, available_models, out_dir)
            plot_6_combined_roc(y_test, y_proba, available_models, out_dir)
            plot_7_confusion_matrices(y_test, y_pred, out_dir)
            plot_8_feature_importance(models_dir, out_dir)

    print("Generating 9-10 (advanced classification plots)...")
    if prep_dir.exists():
        plot_9_validation_curve(prep_dir, out_dir)
        plot_10_threshold_curve(run_root, prep_dir, out_dir)

    print("Generating 11-12 (regression plots)...")
    if prep_dir.exists():
        plot_11_predicted_vs_actual(run_root, prep_dir, out_dir)
    plot_12_regression_comparison(run_root, out_dir)

    generated = sorted(out_dir.glob("*.png"))
    print("\nDone.")
    print(f"Output folder: {out_dir.resolve()}")
    print("Generated files:")
    for file in generated:
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
