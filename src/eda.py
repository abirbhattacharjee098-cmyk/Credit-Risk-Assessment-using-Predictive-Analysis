"""
eda.py
======
Exploratory Data Analysis - uses only matplotlib (no seaborn dependency).

Generates and saves to reports/:
  1.  Target class distribution
  2.  Numeric feature distributions by default label
  3.  Correlation heatmap
  4.  Credit score vs income scatter
  5.  Default rate by loan purpose
  6.  Default rate by education level
  7.  Default rate by employment status
  8.  Box plots - key numeric features
  9.  Missing value summary (if any)

Usage:
    python src/eda.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# -- local imports -------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_data

# -- Paths ---------------------------------------------------------------------
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# -- Colour palette -------------------------------------------------------------
C_BLUE   = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN  = "#55A868"
C_RED    = "#C44E52"
PALETTE  = [C_BLUE, C_ORANGE, C_GREEN, C_RED,
            "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#333355",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})


def _save(fig, fname: str):
    path = os.path.join(REPORTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [eda] Saved -> {path}")


# -----------------------------------------------------------------------------
def run_eda(df: pd.DataFrame):
    print("\n" + "=" * 52)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 52)

    df = df.copy()
    df["Label"] = df["default"].map({0: "No Default", 1: "Default"})

    # -- 1. Class distribution ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["default"].value_counts().sort_index()
    labels = ["No Default", "Default"]
    colors = [C_BLUE, C_ORANGE]
    axes[0].bar(labels, counts.values, color=colors, edgecolor="#333", width=0.5)
    axes[0].set_title("Target Class Distribution")
    axes[0].set_ylabel("Count")
    for i, val in enumerate(counts.values):
        axes[0].text(i, val + 30, f"{val:,}", ha="center", fontsize=11)

    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        wedgeprops={"edgecolor": "#1a1a2e", "linewidth": 2}
    )
    for t in autotexts:
        t.set_color("white")
    axes[1].set_title("Default Rate (Pie Chart)")
    plt.suptitle("Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "01_class_distribution.png")

    # -- 2. Numeric feature distributions ---------------------------------
    numeric_cols = ["age", "income", "loan_amount", "credit_score",
                    "debt_to_income", "employment_years",
                    "num_delinquencies", "previous_defaults", "loan_to_income"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, col in zip(axes.flatten(), numeric_cols):
        for label, color in [("No Default", C_BLUE), ("Default", C_ORANGE)]:
            subset = df[df["Label"] == label][col].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(col.replace("_", " ").title())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Numeric Feature Distributions by Default Label",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "02_feature_distributions.png")

    # -- 3. Correlation heatmap --------------------------------------------
    num_df = df[numeric_cols + ["default"]].dropna()
    corr   = num_df.corr()
    mask   = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = np.where(mask, np.nan, corr.values)

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(corr_masked, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if not mask[i, j]:
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > 0.5 else "#ccc")
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_correlation_heatmap.png")

    # -- 4. Credit score vs income scatter ---------------------------------
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in [("No Default", C_BLUE), ("Default", C_ORANGE)]:
        subset = sample[sample["Label"] == label]
        ax.scatter(subset["credit_score"], subset["income"] / 1_000,
                   alpha=0.45, label=label, color=color, s=22)
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Annual Income (K $)")
    ax.set_title("Credit Score vs Income - colored by Default",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "04_credit_vs_income.png")

    # -- 5. Default rate by loan purpose ----------------------------------
    purpose_dr = (
        df.groupby("loan_purpose")["default"]
          .mean()
          .sort_values(ascending=False)
          .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(purpose_dr["loan_purpose"], purpose_dr["default"],
                  color=PALETTE[:len(purpose_dr)], edgecolor="#333")
    ax.set_title("Default Rate by Loan Purpose", fontsize=13, fontweight="bold")
    ax.set_ylabel("Default Rate")
    ax.set_ylim(0, 0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.1%}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "05_default_by_purpose.png")

    # -- 6. Default rate by education --------------------------------------
    edu_dr = (
        df.groupby("education")["default"]
          .mean()
          .sort_values(ascending=False)
          .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(edu_dr["education"], edu_dr["default"],
                  color=PALETTE[:len(edu_dr)], edgecolor="#333")
    ax.set_title("Default Rate by Education Level", fontsize=13, fontweight="bold")
    ax.set_ylabel("Default Rate")
    ax.set_ylim(0, 0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.1%}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "06_default_by_education.png")

    # -- 7. Default rate by employment status ------------------------------
    emp_dr = (
        df.groupby("employment_status")["default"]
          .mean()
          .sort_values(ascending=False)
          .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(emp_dr["employment_status"], emp_dr["default"],
                  color=PALETTE[:len(emp_dr)], edgecolor="#333")
    ax.set_title("Default Rate by Employment Status", fontsize=13, fontweight="bold")
    ax.set_ylabel("Default Rate")
    ax.set_ylim(0, 0.7)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.1%}", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "07_default_by_employment.png")

    # -- 8. Box plots ------------------------------------------------------
    box_cols = ["credit_score", "income", "debt_to_income",
                "loan_to_income", "previous_defaults"]
    fig, axes = plt.subplots(1, len(box_cols), figsize=(18, 6))
    for ax, col in zip(axes, box_cols):
        data_no  = df[df["Label"] == "No Default"][col].dropna()
        data_yes = df[df["Label"] == "Default"][col].dropna()
        bp = ax.boxplot([data_no, data_yes],
                        patch_artist=True,
                        labels=["No Default", "Default"],
                        medianprops={"color": "white", "linewidth": 2})
        bp["boxes"][0].set_facecolor(C_BLUE)
        bp["boxes"][1].set_facecolor(C_ORANGE)
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Key Features vs Default (Box Plots)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "08_boxplots.png")

    # -- 9. Missing value summary ------------------------------------------
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(missing.index, missing.values, color=C_BLUE, edgecolor="#333")
        ax.set_xlabel("Missing Count")
        ax.set_title("Missing Values per Feature", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        _save(fig, "09_missing_values.png")
    else:
        print("  [eda] No missing values to plot.")

    print("\n  EDA complete - all plots saved to reports/")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    run_eda(df)
