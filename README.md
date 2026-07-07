# Claim Denial Prediction

Predicts the probability that a healthcare insurance claim will be **denied**, so claims teams can prioritize audits/documentation before submission. The model also surfaces the top features driving each claim's risk score for explainability and uses an LLM to turn those drivers into a clear, plain-English explanation

## High-Level Approach

1. **Data**: `claims_history.csv` (labeled, with a `split` column of `train` / `validation` / `test`) and `current_claims.csv` (new, unlabeled claims to be scored).

2. **EDA**: correlation of each feature with `is_denied`, boxplot/KDE distributions of numeric features by denial status, and denial-rate breakdowns by categorical/boolean flags (`src/eda.py`).

3. **Preprocessing** (`src/preprocessing.py`): numeric features are standardized, categorical features are one-hot encoded, boolean flags are passed through. The transformer is **fit only on the training split** to avoid leakage, then applied to validation/test/new data.

4. **Model** (`src/train.py`): a `class_weight="balanced"` classifier (logistic regression by default; random forest also available) trained to predict `is_denied`.

5. **Evaluation** (`src/evaluate.py`): classification report, ROC-AUC, confusion matrix, and a precision/recall-by-score-bucket breakdown (20 probability bands) to see how reliable the model is at different risk thresholds.

6. **Explainability & Scoring** (`src/explain.py`, `src/predict.py`): for linear models, each claim's risk score is decomposed into `processed_feature_value x coefficient` to find its top 3 drivers, with their raw (unscaled) values attached in a `Prompt_Context` string ready to hand to an LLM for a plain-English, 2-line explanation + next step. New claims are assigned a **High / Medium / Low** risk tier.

## Repository Structure

```
claim-denial-prediction/
├── README.md
├── requirements.txt
├── data/                     # place claims_history.csv / current_claims.csv here
├── outputs/                  # trained model + plots/metrics land here
└── src/
    ├── config.py              # paths, column groups, thresholds
    ├── data_loader.py         # CSV loading, boolean casting, null/duplicate checks
    ├── eda.py                 # correlation matrix, heatmap, distribution plots
    ├── preprocessing.py       # train/val/test split + ColumnTransformer
    ├── train.py               # CLI: trains + saves the model bundle
    ├── evaluate.py            # CLI: evaluates a saved model on any split
    ├── explain.py             # feature-contribution / prompt-context builder
    ├── predict.py             # CLI: scores new claims + risk tiers
    └── llm_audit.py           # CLI: prints (does not save) LLM 2-line audit explanations
```

## Setup

```bash
git clone <this-repo-url>
cd claim-denial-prediction
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Place your data under `data/`:

- `data/claims_history.csv` — must include a `split` column (`train`/`validation`/`test`) and the target column `is_denied`.
- `data/current_claims.csv` — new claims to score (same feature columns, no target needed).

## Reproduce

**1. Train** (fits preprocessor + model, evaluates on validation, saves `outputs/model.pkl`):

```bash
python -m src.train --data_path data/claims_history.csv --model logreg --seed 42 --run_eda
```

Optional flags:

```bash
# Train a random forest instead, and also dump EDA plots to outputs/eda/
python -m src.train --data_path data/claims_history.csv --model random_forest --seed 42 --run_eda
```

**2. Evaluate** (loads the saved model bundle, reports metrics + plots on a given split):

```bash
python -m src.evaluate --model_path outputs/model.pkl --data_path data/claims_history.csv --split test
```

This writes `outputs/confusion_matrix_test.png`, `outputs/score_bucket_summary_test.png`, and `outputs/score_bucket_summary_test.csv`.

**3. Predict / Score new claims**:

```bash
python -m src.predict --model_path outputs/model.pkl --data_path data/current_claims.csv --output_path outputs/predictions_current_claims.csv
```

This prints the top 10 highest-risk claims and writes the full scored dataset (with `High_score`, `Risk_Tier`, `Top_drivers`, `Prompt_Context` columns) to `outputs/predictions_current_claims.csv`.

> Run all commands from the repository root so the `src` package resolves correctly (`python -m src.train ...`, not `python src/train.py ...`).

**4. (Optional) LLM audit explanations** — print a plain-English, 2-line risk explanation per claim using an LLM (via OpenRouter), based on the `Prompt_Context` already computed in `outputs/predictions_current_claims.csv`. This only **prints** the responses; it does not modify or re-save the CSV.

```bash
export OPENROUTER_API_KEY=sk-...          # Windows PowerShell: $env:OPENROUTER_API_KEY = "sk-..."
python -m src.llm_audit --data_path outputs/predictions_current_claims.csv
```

Optional flags:

```bash
# Only audit the first 5 claims, and use a specific OpenRouter model
python -m src.llm_audit --data_path outputs/predictions_current_claims.csv --limit 5 --model openrouter/free
```

## Notes on Explainability

- For **logistic regression**, `Top_drivers` / `Prompt_Context` are exact per-claim contributions (`feature_value x coefficient`).
- For **random forest**, per-claim linear contributions aren't well-defined, so `Top_drivers` falls back to the model's global top-3 most important features (noted explicitly in the output) rather than a claim-specific breakdown.
