"""
llm_audit.py
------------
Reads the CSV produced by `predict.py` (e.g. outputs/final_data.csv) and,
for each row, sends its data to an LLM (via OpenRouter) using the same
strict "2-line explanation" system prompt from explain.py. Responses are
PRINTED ONLY — nothing is written back to the CSV or disk.

Requires:
    pip install openai
    export OPENROUTER_API_KEY=sk-...      (Linux/macOS)
    $env:OPENROUTER_API_KEY = "sk-..."    (Windows PowerShell)

CLI usage:
    python -m src.llm_audit --data_path outputs/final_data.csv
    python -m src.llm_audit --data_path outputs/final_data.csv --limit 5 --model openrouter/free
"""

import argparse
import os

import pandas as pd
from openai import OpenAI

from . import config
from .explain import LLM_EXPLANATION_PROMPT


def build_client(api_key: str = None, base_url: str = "https://openrouter.ai/api/v1") -> OpenAI:
    """Create the OpenRouter-backed OpenAI client. Falls back to the OPENROUTER_API_KEY env var."""
    key = "key" or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "No API key found. Pass --api_key or set the OPENROUTER_API_KEY environment variable."
        )
    return OpenAI(base_url=base_url, api_key=key)


def row_to_prompt_string(row: pd.Series) -> str:
    """Convert a single claim's row into a readable string for the LLM prompt."""
    return row.to_string()


def get_llm_response(client: OpenAI, row_data_string: str, model: str = "openrouter/free",
                      temperature: float = 0.2) -> str:
    """Send one claim's data to the LLM and return its (expected 2-line) response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_EXPLANATION_PROMPT},
            {"role": "user", "content": f"Analyze this claim data:\n{row_data_string}"},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def audit_claims(df: pd.DataFrame, client: OpenAI, model: str = "openrouter/free",
                  limit: int = None):
    """
    Iterate over claims and print each one's LLM-generated 2-line risk
    explanation. Nothing is saved back to `df` or to disk.
    """
    rows = df.head(limit) if limit else df

    for position, (idx, row) in enumerate(rows.iterrows(), start=1):
        claim_label = row.get("claim_id", idx)
        print(f"--- Auditing Claim {claim_label} ({position}/{len(rows)}) ---")

        try:
            row_data_string = row_to_prompt_string(row)
            reply = get_llm_response(client, row_data_string, model=model)
            print(reply)
            print()
        except Exception as e:
            print(f"⚠️ Error processing claim {claim_label}: {e}\n")


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Print LLM-generated 2-line risk explanations for scored claims (no CSV write-back)."
    )
    parser.add_argument("--data_path", type=str,
                         default=f"{config.DEFAULT_OUTPUT_DIR}/final_data.csv",
                         help="Path to the CSV produced by predict.py (e.g. outputs/final_data.csv)")
    parser.add_argument("--model", type=str, default="openrouter/free",
                         help="OpenRouter model name")
    parser.add_argument("--api_key", type=str, default=None,
                         help="OpenRouter API key (defaults to OPENROUTER_API_KEY env var)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N rows (useful for testing/cost control)")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} claims from {args.data_path}. Generating audit explanations...\n")

    client = build_client(api_key=args.api_key)
    audit_claims(df, client, model=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
