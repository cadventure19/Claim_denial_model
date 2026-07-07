"""
evaluate.py
-----------
Library functions for model evaluation: classification reports,
ROC-AUC, confusion matrix, and a precision/recall-by-score-bucket
breakdown. Also runnable directly as a CLI script:

    python -m src.evaluate --model_path outputs/model.pkl \\
        --data_path data/claims_history.csv --split test --output_dir outputs
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
import joblib

from . import config
from . import data_loader
from . import preprocessing


def evaluate_split(model, X_proc, y_true, split_name: str):
    """Print classification report + ROC-AUC for a given split, and return predictions/probas."""
    y_pred = model.predict(X_proc)
    y_proba = model.predict_proba(X_proc)[:, 1]

    print(f"--- {split_name.upper()} SET PERFORMANCE ---")
    print(classification_report(y_true, y_pred))
    print(f"{split_name.title()} ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}\n")

    return y_pred, y_proba


def plot_confusion_matrix(y_true, y_pred, class_labels, title="Confusion Matrix",
                           save_dir: str = None, filename: str = "confusion_matrix.png"):
    """Compute and display/save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(title)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved figure: {path}")
    plt.close(fig)
    return cm


def build_score_bucket_summary(y_true, y_proba, num_buckets: int = config.NUM_SCORE_BUCKETS) -> pd.DataFrame:
    """
    Bucket predicted probabilities into `num_buckets` equal-width bands and
    compute per-bucket volume, precision, and recall contribution.
    """
    bucket_df = pd.DataFrame({"actual": y_true, "prob": y_proba})

    bins = np.linspace(0, 1, num_buckets + 1)
    labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(num_buckets)]
    bucket_df["bucket"] = pd.cut(bucket_df["prob"], bins=bins, labels=labels, include_lowest=True)

    total_actual_positives = (bucket_df["actual"] == 1).sum()

    rows = []
    for label in labels:
        subset = bucket_df[bucket_df["bucket"] == label]
        total_in_bucket = len(subset)
        true_positives = (subset["actual"] == 1).sum()

        precision = true_positives / total_in_bucket if total_in_bucket > 0 else 0.0
        recall = true_positives / total_actual_positives if total_actual_positives > 0 else 0.0

        rows.append({
            "Prediction Score Bucket": label,
            "Total Claims": total_in_bucket,
            "Actual Denials": true_positives,
            "Precision": round(precision, 4),
            "Recall Contribution": round(recall, 4),
        })

    return pd.DataFrame(rows)


def plot_score_bucket_summary(summary_table: pd.DataFrame, save_dir: str = None,
                               filename: str = "score_bucket_summary.png"):
    """Plot claim volume (bars) with precision/recall trend lines across score buckets."""
    fig, ax1 = plt.subplots(figsize=(16, 6))

    ax1.bar(summary_table["Prediction Score Bucket"], summary_table["Total Claims"],
            color="#e0e0e0", alpha=0.6, label="Volume of Claims")
    ax1.set_xlabel("Prediction Score (Probability Bucket)", fontsize=12)
    ax1.set_ylabel("Number of Claims", fontsize=12)
    ax1.set_xticks(range(len(summary_table)))
    ax1.set_xticklabels(summary_table["Prediction Score Bucket"], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(summary_table["Prediction Score Bucket"], summary_table["Precision"],
             color="#d95f02", marker="o", linewidth=2, label="Precision")
    ax2.plot(summary_table["Prediction Score Bucket"], summary_table["Recall Contribution"],
             color="#2b5c8f", marker="s", linewidth=2, label="Recall Contribution")
    ax2.set_ylabel("Metric Rate (0.0 - 1.0)", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)

    plt.title("Precision and Recall Trends Across Prediction Score Buckets",
              fontsize=14, weight="bold", pad=15)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved figure: {path}")
    plt.close(fig)


def assign_risk_tier(prob: float) -> str:
    """Map a probability to a High/Medium/Low risk tier."""
    if prob >= config.HIGH_RISK_THRESHOLD:
        return "High"
    elif prob >= config.MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained claim-denial model.")
    parser.add_argument("--model_path", type=str, default=config.DEFAULT_MODEL_PATH,
                         help="Path to the pickled {preprocessor, model} bundle from train.py")
    parser.add_argument("--data_path", type=str, default=config.DEFAULT_CLAIMS_HISTORY_PATH,
                         help="Path to the labeled claims CSV (must contain a 'split' column)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"],
                         help="Which split to evaluate on")
    parser.add_argument("--output_dir", type=str, default=config.DEFAULT_OUTPUT_DIR,
                         help="Directory to write evaluation plots/metrics to")
    return parser.parse_args()


def main():
    args = parse_args()

    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]

    df = data_loader.load_csv(args.data_path)
    df = data_loader.cast_boolean_columns(df)

    train_df, val_df, test_df = preprocessing.split_by_column(df)
    split_map = {"train": train_df, "validation": val_df, "test": test_df}
    split_df = split_map[args.split]

    X, y = preprocessing.get_features_and_target(split_df)
    X_proc = preprocessor.transform(X)

    y_pred, y_proba = evaluate_split(model, X_proc, y, args.split)

    plot_confusion_matrix(
        y, y_pred, model.classes_,
        title=f"Confusion Matrix - {args.split.title()} Set",
        save_dir=args.output_dir,
        filename=f"confusion_matrix_{args.split}.png",
    )

    bucket_summary = build_score_bucket_summary(y, y_proba)
    print("--- PRECISION AND RECALL DISTRIBUTION BY SCORE BUCKET ---")
    print(bucket_summary.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    bucket_csv_path = os.path.join(args.output_dir, f"score_bucket_summary_{args.split}.csv")
    bucket_summary.to_csv(bucket_csv_path, index=False)
    print(f"Saved bucket summary: {bucket_csv_path}")

    plot_score_bucket_summary(
        bucket_summary, save_dir=args.output_dir,
        filename=f"score_bucket_summary_{args.split}.png",
    )


if __name__ == "__main__":
    main()
