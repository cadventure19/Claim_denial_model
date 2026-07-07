"""
explain.py
----------
Builds per-claim explanations: which features (scaled contribution =
processed_value * coefficient) drove the risk score, plus a human/LLM
-readable "Prompt_Context" string that includes each top feature's raw
value from the original (unprocessed) data.
"""

import pandas as pd

from . import config


def compute_contribution_matrix(X_proc_df: pd.DataFrame, coefficients) -> pd.DataFrame:
    """Elementwise multiply processed feature values by model coefficients."""
    contribution_values = X_proc_df.values * coefficients
    return pd.DataFrame(contribution_values, columns=X_proc_df.columns)


def top_n_drivers_per_row(contribution_df: pd.DataFrame, n: int = 3) -> list:
    """Return a list of comma-separated top-N feature names (by positive contribution) per row."""
    drivers = []
    for _, row in contribution_df.iterrows():
        sorted_features = row.sort_values(ascending=False)
        top_names = sorted_features.index[:n]
        drivers.append(", ".join(top_names))
    return drivers


def _resolve_raw_value(feat_name: str, idx, raw_df: pd.DataFrame, X_proc_df: pd.DataFrame,
                        categorical_cols=config.CATEGORICAL_COLS):
    """
    Find the human-readable raw value for a (possibly one-hot-encoded) feature name.
    Falls back to 'N/A' if it can't be resolved.
    """
    if feat_name in raw_df.columns:
        return raw_df.loc[idx, feat_name]

    for orig_col in categorical_cols:
        prefix = orig_col + "_"
        if feat_name.startswith(prefix) and X_proc_df.loc[idx, feat_name] == 1:
            return feat_name.split(prefix)[-1]

    return "N/A"


def build_prompt_context(contribution_df: pd.DataFrame, raw_df: pd.DataFrame,
                          X_proc_df: pd.DataFrame, n: int = 3):
    """
    For each row, build:
      - a short comma-separated string of top-N driver feature names
      - a detailed "Prompt_Context" string with rank, feature, impact, and raw value,
        suitable for feeding to an LLM to generate a plain-English explanation.
    """
    risk_drivers = []
    prompt_context_list = []

    for idx, row in contribution_df.iterrows():
        sorted_features = row.sort_values(ascending=False)
        top_names = sorted_features.index[:n]
        risk_drivers.append(", ".join(top_names))

        row_details = []
        for rank, (feat_name, impact_val) in enumerate(list(sorted_features.items())[:n], 1):
            raw_val = _resolve_raw_value(feat_name, idx, raw_df, X_proc_df)
            row_details.append(f"{rank}. {feat_name} (Impact: +{impact_val:.2f} | Raw: {raw_val})")

        prompt_context_list.append(" | ".join(row_details))

    return risk_drivers, prompt_context_list


LLM_EXPLANATION_PROMPT = """You are an expert healthcare risk auditor. Your task is to \
summarize claim risk using the provided data into a layman-friendly explanation of \
exactly 2 lines.

Line 1: State the risk score, tier, and a brief, plain-English summary of why the claim \
is a risk based on its top features (using their raw/scaled values).
Line 2: Provide a direct, actionable next step on what needs to be done to resolve or \
audit this specific case.

Do not include any introductory or concluding text. Strictly output exactly 2 lines.
"""
