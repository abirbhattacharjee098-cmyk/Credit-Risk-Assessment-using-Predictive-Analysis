"""
evaluate.py
===========
Loads all trained models and evaluates them on the held-out test set.

Metrics reported:
  - Accuracy, Precision, Recall, F1-score (macro & weighted)
  - ROC-AUC
  - Confusion Matrix
  - Classification Report
  - ROC curve plot  (saved to reports/)
  - Confusion matrix plot (saved to reports/)
  - Feature importance plot for tree-based models (saved to reports/)

Usage:
    python src/evaluate.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, ConfusionMatrixDisplay
)

# -- local imports -------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from data_loader   import load_data
from preprocessing import preprocess

# -- Paths ---------------------------------------------------------------------
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree":       "decision_tree.pkl",
    "Random Forest":       "random_forest.pkl",
    "XGBoost":             "xgboost.pkl",
}

# -- Palette -------------------------------------------------------------------
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# -----------------------------------------------------------------------------
def load_models() -> dict:
    """Load all persisted model files from disk."""
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            print(f"[evaluate] WARNING: '{path}' not found - skipping {name}.")
    return models


# -----------------------------------------------------------------------------
def compute_metrics(model, X_test, y_test, name: str) -> dict:
    """Return a dict of evaluation metrics for one model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Model":         name,
        "Accuracy":      round(accuracy_score(y_test, y_pred), 4),
        "Precision":     round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":        round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":      round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":       round(roc_auc_score(y_test, y_proba), 4),
    }


# -----------------------------------------------------------------------------
def plot_roc_curves(models: dict, X_test, y_test):
    """Save a combined ROC-AUC curve for all models."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")

    for (name, model), color in zip(models.items(), PALETTE):
        y_proba        = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _    = roc_curve(y_test, y_proba)
        auc            = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves - Credit Risk Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "roc_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] ROC curve saved -> {out_path}")


# -----------------------------------------------------------------------------
def plot_confusion_matrices(models: dict, X_test, y_test):
    """Save a 2×2 grid of confusion matrices."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, model), color in zip(axes, models.items(), PALETTE):
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        disp   = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=["No Default", "Default"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=12, fontweight="bold")

    plt.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "confusion_matrices.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrices saved -> {out_path}")


# -----------------------------------------------------------------------------
def plot_feature_importance(models: dict, feature_names: list):
    """Save feature importance bar charts for tree-based models."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    tree_models = {k: v for k, v in models.items()
                   if hasattr(v, "feature_importances_")}

    for name, model in tree_models.items():
        importances = model.feature_importances_
        indices     = np.argsort(importances)[::-1][:20]   # top 20

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            [feature_names[i] for i in indices[::-1]],
            importances[indices[::-1]],
            color="#4C72B0",
        )
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(f"Top-20 Feature Importance - {name}",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").lower()
        out_path  = os.path.join(REPORTS_DIR, f"feature_importance_{safe_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[evaluate] Feature importance saved -> {out_path}")


# -----------------------------------------------------------------------------
def plot_metrics_comparison(results_df: pd.DataFrame):
    """Save a grouped bar chart comparing all models across metrics."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = results_df["Model"].tolist()
    n_models    = len(model_names)
    n_metrics   = len(metric_cols)
    x = np.arange(n_metrics)
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (name, color) in enumerate(zip(model_names, PALETTE)):
        vals = [results_df.loc[results_df["Model"]==name, m].values[0] for m in metric_cols]
        ax.bar(x + i * width, vals, width, label=name, color=color, edgecolor="#333")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_cols)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model Comparison - All Metrics", fontsize=14, fontweight="bold")
    ax.legend(title="Model", loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "metrics_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Metrics comparison saved -> {out_path}")


# -----------------------------------------------------------------------------
def evaluate_all():
    """Full evaluation pipeline - call this as main entry point."""
    # -- Load preprocessed data --------------------------------------------
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess(
        df, fit=False, apply_smote=False
    )

    # -- Load trained models -----------------------------------------------
    models = load_models()
    if not models:
        print("[evaluate] No models found. Run  python src/train.py  first.")
        return

    # -- Compute metrics ---------------------------------------------------
    results = []
    for name, model in models.items():
        metrics = compute_metrics(model, X_test, y_test, name)
        results.append(metrics)
        print(f"\n{'='*52}")
        print(f"  {name}")
        print(f"{'-'*52}")
        for k, v in metrics.items():
            if k != "Model":
                print(f"  {k:<18}: {v}")
        print(f"\n{classification_report(y_test, model.predict(X_test), target_names=['No Default','Default'])}")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 52)
    print("  SUMMARY TABLE")
    print("=" * 52)
    print(results_df.to_string(index=False))

    # Save summary CSV
    summary_path = os.path.join(REPORTS_DIR, "metrics_summary.csv")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results_df.to_csv(summary_path, index=False)
    print(f"\n[evaluate] Metrics summary saved -> {summary_path}")

    # -- Generate plots ----------------------------------------------------
    plot_roc_curves(models, X_test, y_test)
    plot_confusion_matrices(models, X_test, y_test)
    plot_feature_importance(models, feature_names)
    plot_metrics_comparison(results_df)

    # -- Best model --------------------------------------------------------
    best = results_df.loc[results_df["ROC-AUC"].idxmax(), "Model"]
    print("\n[BEST] Best model (by ROC-AUC): {}".format(best))
    best_model = models[best]
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    print("       Saved as 'best_model.pkl'")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 52)
    print("  CREDIT RISK ASSESSMENT - EVALUATION")
    print("=" * 52)
    evaluate_all()
