"""
predict.py
----------
Scores a new batch of claims (e.g. current_claims.csv) using an already
-fitted preprocessor + model, and attaches risk tier and explanations.

CLI usage:
    python -m src.predict --model_path outputs/model.pkl \\
        --data_path data/current_claims.csv --output_path outputs/predictions_current_claims.csv
"""

import argparse

import pandas as pd
import joblib

from . import config
from . import data_loader
from . import preprocessing
from . import explain
from .evaluate import assign_risk_tier


def score_new_claims(current_claims: pd.DataFrame, preprocessor, model) -> pd.DataFrame:
    """
    Transform new claims with the fitted preprocessor, predict denial
    probability, assign a risk tier, and attach top-driver explanations.
    Returns the original claims data with prediction columns appended.
    """
    X_new_proc = preprocessor.transform(current_claims)
    X_new_proc_dense = X_new_proc.toarray() if hasattr(X_new_proc, "toarray") else X_new_proc

    new_pred_prob = model.predict_proba(X_new_proc_dense)[:, 1]

    feature_names = preprocessing.get_processed_feature_names(preprocessor)
    X_new_proc_df = pd.DataFrame(X_new_proc_dense, columns=feature_names)

    raw_claims_reset = current_claims.reset_index(drop=True)

    if hasattr(model, "coef_"):
        # Linear model: exact per-claim contribution = value * coefficient
        contribution_df = explain.compute_contribution_matrix(X_new_proc_df, model.coef_[0])
        risk_drivers, prompt_context_list = explain.build_prompt_context(
            contribution_df, raw_claims_reset, X_new_proc_df
        )
    elif hasattr(model, "feature_importances_"):
        # Tree-based model: no per-claim linear contribution, so fall back to
        # the model's global top-3 most important features for every row.
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_names = importances.sort_values(ascending=False).index[:3].tolist()
        drivers_string = ", ".join(top_names)
        risk_drivers = [drivers_string] * len(raw_claims_reset)
        prompt_context_list = [
            f"Top global drivers (model-wide, not claim-specific): {drivers_string}"
        ] * len(raw_claims_reset)
    else:
        risk_drivers = ["N/A"] * len(raw_claims_reset)
        prompt_context_list = ["N/A"] * len(raw_claims_reset)

    predictions_df = pd.DataFrame({
        "High_score": new_pred_prob,
        "Risk_Tier": [assign_risk_tier(p) for p in new_pred_prob],
        "Top_drivers": risk_drivers,
        "Prompt_Context": prompt_context_list,
    })

    final_output_df = pd.concat([raw_claims_reset, predictions_df], axis=1)
    return final_output_df


def top_k_riskiest_claims(final_output_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Return the top-k claims sorted by descending denial risk score."""
    cols = ["claim_id", "High_score", "Risk_Tier", "Top_drivers", "Prompt_Context"]
    cols = [c for c in cols if c in final_output_df.columns]
    return final_output_df[cols].sort_values(by="High_score", ascending=False).head(k)


def save_predictions(final_output_df: pd.DataFrame, path: str):
    """Persist the scored claims to CSV."""
    final_output_df.to_csv(path, index=False)
    print(f"Predictions saved to {path}")


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Score new claims with a trained model.")
    parser.add_argument("--model_path", type=str, default=config.DEFAULT_MODEL_PATH,
                         help="Path to the pickled {preprocessor, model} bundle from train.py")
    parser.add_argument("--data_path", type=str, default=config.DEFAULT_CURRENT_CLAIMS_PATH,
                         help="Path to the new/unlabeled claims CSV to score")
    parser.add_argument("--output_path", type=str,
                         default=f"{config.DEFAULT_OUTPUT_DIR}/predictions_current_claims.csv",
                         help="Where to write the scored claims CSV")
    parser.add_argument("--top_k", type=int, default=10,
                         help="How many highest-risk claims to print to console")
    return parser.parse_args()


def main():
    args = parse_args()

    bundle = joblib.load(args.model_path)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]

    current_claims = data_loader.load_csv(args.data_path)

    final_output_df = score_new_claims(current_claims, preprocessor, model)

    print("--- NEW DATA PREDICTIONS & EXPLANATIONS COMPLETE ---")
    print(final_output_df[["High_score", "Risk_Tier", "Top_drivers", "Prompt_Context"]].head(10))

    print(f"\n--- TOP {args.top_k} RISKIEST CLAIMS ---")
    print(top_k_riskiest_claims(final_output_df, k=args.top_k))

    save_predictions(final_output_df, args.output_path)


if __name__ == "__main__":
    main()
