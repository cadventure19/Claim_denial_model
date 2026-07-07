"""
eda.py
------
Exploratory data analysis helpers: correlation matrix, heatmap,
boxplot/KDE pairs for numeric features, and denial-rate distribution
plots for categorical and boolean features.

All plotting functions accept an optional `save_dir`. When given, the
figure is written to disk (PNG) instead of / in addition to being
displayed - useful when running as a headless script rather than a
notebook.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # safe default for headless script execution
import matplotlib.pyplot as plt

from . import config


def _savefig(fig, save_dir, filename):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved figure: {path}")


def build_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categoricals, drop ID columns, and return the full
    numeric correlation matrix (including the target).
    """
    numeric_df = df.drop(columns=[c for c in config.ID_COLS if c in df.columns])
    numeric_df = pd.get_dummies(numeric_df, columns=config.CATEGORICAL_COLS, drop_first=True)
    numeric_df = numeric_df.select_dtypes(include=[np.number])
    return numeric_df.corr()


def print_target_correlation(corr_matrix: pd.DataFrame, target: str = config.TARGET_COL):
    """Print correlation of every feature with the target column, sorted descending."""
    print(f"--- Correlation with '{target}' (Sorted) ---")
    target_corr = corr_matrix[target].sort_values(ascending=False)
    print(target_corr)
    return target_corr


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_dir: str = None):
    """Render a masked (upper-triangle hidden) heatmap of the correlation matrix."""
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        square=True, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Inter-Variable Correlation Matrix Heatmap", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _savefig(fig, save_dir, "correlation_heatmap.png")
    plt.close(fig)


def plot_numeric_feature_distributions(df: pd.DataFrame, num_cols=config.NUMERICAL_COLS,
                                        target: str = config.TARGET_COL, save_dir: str = None):
    """For each numeric column, show a boxplot and KDE plot split by denial status."""
    for col in num_cols:
        if col not in df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        sns.boxplot(
            data=df, x=target, y=col, hue=target,
            palette=["#2b5c8f", "#d95f02"], legend=False, ax=axes[0],
        )
        axes[0].set_title(f"Boxplot of {col} by Denial Status", fontsize=12)
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(["Approved (0)", "Denied (1)"])
        axes[0].set_xlabel("Claim Status")

        sns.kdeplot(
            data=df, x=col, hue=target, fill=True,
            palette=["#2b5c8f", "#d95f02"], alpha=0.4, ax=axes[1],
        )
        axes[1].set_title(f"Distribution Density of {col}", fontsize=12)
        axes[1].set_xlabel(col)

        legend = axes[1].get_legend()
        if legend:
            legend.set_title("Claim Status")
            for text, label in zip(legend.get_texts(), ["Denied (1)", "Approved (0)"]):
                text.set_text(label)

        plt.suptitle(f"Analysis of Feature: {col}", fontsize=14, weight="bold", y=1.02)
        plt.tight_layout()
        _savefig(fig, save_dir, f"feature_distribution_{col}.png")
        plt.close(fig)


def print_and_plot_categorical_distributions(df: pd.DataFrame, cat_cols=config.CATEGORICAL_COLS,
                                              target: str = config.TARGET_COL, save_dir: str = None):
    """Print denial-rate crosstabs and plot a stacked bar chart for each categorical column."""
    for col in cat_cols:
        if col not in df.columns:
            continue

        print(f"\n--- Denial Distribution for {col} ---")
        dist_df = pd.crosstab(df[col], df[target], normalize="index") * 100
        print(dist_df.round(2).rename(columns={0: "Approved (%)", 1: "Denied (%)"}))

        fig, ax = plt.subplots(figsize=(10, 5))
        df_pct = df.groupby(col)[target].value_counts(normalize=True).unstack() * 100
        df_pct.plot(kind="bar", stacked=True, color=["#2b5c8f", "#d95f02"], ax=ax)
        ax.set_title(f"Denial Rate Distribution by {col}", fontsize=14, pad=15)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Percentage (%)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        ax.legend(["Approved (0)", "Denied (1)"], loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        _savefig(fig, save_dir, f"denial_rate_by_{col}.png")
        plt.close(fig)


def print_boolean_distributions(df: pd.DataFrame, bool_cols=config.PASSTHROUGH_COLS,
                                 target: str = config.TARGET_COL):
    """Print denial-rate crosstabs for boolean/flag columns."""
    for col in bool_cols:
        if col not in df.columns:
            continue
        print(f"\n--- Distribution for {col} against Rejections ---")
        ct = pd.crosstab(df[col], df[target], normalize="index") * 100
        print(ct.round(2).rename(columns={0: "Approved (%)", 1: "Denied (%)"}))


def run_full_eda(df: pd.DataFrame, save_dir: str = None):
    """Convenience wrapper running the entire EDA suite in one call."""
    corr_matrix = build_correlation_matrix(df)
    print_target_correlation(corr_matrix)
    plot_correlation_heatmap(corr_matrix, save_dir=save_dir)
    plot_numeric_feature_distributions(df, save_dir=save_dir)
    print_and_plot_categorical_distributions(df, save_dir=save_dir)
    print_boolean_distributions(df)
    return corr_matrix
