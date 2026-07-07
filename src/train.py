"""
train.py
--------
Trains a claim-denial-prediction model end to end: load data, split by
the `split` column, fit the preprocessor + model, evaluate on
train/validation, and persist a {preprocessor, model} bundle to disk.

CLI usage:
    python -m src.train --data_path data/claims_history.csv --model logreg --seed 42
    python -m src.train --data_path data/claims_history.csv --model random_forest
"""

import argparse
import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from . import config
from . import data_loader
from . import preprocessing
from . import evaluate
from . import eda as eda_module


MODEL_REGISTRY = {
    "logreg": lambda seed: LogisticRegression(
        max_iter=config.MAX_ITER, class_weight=config.CLASS_WEIGHT, random_state=seed
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=300, class_weight=config.CLASS_WEIGHT, random_state=seed
    ),
}


def build_model(model_type: str, seed: int):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type](seed)


def train_model(X_train_proc, y_train, model_type: str = config.DEFAULT_MODEL_TYPE,
                 seed: int = config.RANDOM_STATE):
    """Instantiate and fit the requested model type on processed training data."""
    model = build_model(model_type, seed)
    model.fit(X_train_proc, y_train)
    return model


def save_model_bundle(preprocessor, model, model_type: str, path: str = config.DEFAULT_MODEL_PATH):
    """Persist preprocessor + model + metadata together so evaluate.py/predict.py stay in sync."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "model_type": model_type,
        "feature_names": preprocessing.get_processed_feature_names(preprocessor),
    }
    joblib.dump(bundle, path)
    print(f"Saved model bundle: {path}")


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a claim-denial-prediction model.")
    parser.add_argument("--data_path", type=str, default=config.DEFAULT_CLAIMS_HISTORY_PATH,
                         help="Path to the labeled claims history CSV (must contain a 'split' column)")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL_TYPE,
                         choices=list(MODEL_REGISTRY.keys()), help="Which model type to train")
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE,
                         help="Random seed for reproducibility")
    parser.add_argument("--model_path", type=str, default=config.DEFAULT_MODEL_PATH,
                         help="Where to save the trained {preprocessor, model} bundle")
    parser.add_argument("--output_dir", type=str, default=config.DEFAULT_OUTPUT_DIR,
                         help="Directory to write validation plots/metrics to")
    parser.add_argument("--run_eda", action="store_true",
                         help="If set, also generate and save EDA plots to output_dir/eda")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load and lightly clean historical claims data
    claims_history = data_loader.load_csv(args.data_path)
    claims_history = data_loader.cast_boolean_columns(claims_history)
    data_loader.check_duplicate_payer_ids(claims_history)
    data_loader.check_nulls(claims_history)

    # 2. Optional EDA (saved as PNGs, since this runs headless)
    if args.run_eda:
        eda_module.run_full_eda(claims_history, save_dir=os.path.join(args.output_dir, "eda"))

    # 3. Split + preprocess (fit on train only, to avoid leakage)
    prepped = preprocessing.prepare_train_val_test(claims_history)
    preprocessor = prepped["preprocessor"]

    # 4. Train model
    model = train_model(prepped["X_train_proc"], prepped["y_train"],
                         model_type=args.model, seed=args.seed)

    # 5. Quick validation check
    evaluate.evaluate_split(model, prepped["X_val_proc"], prepped["y_val"], "validation")

    # 6. Persist model bundle for evaluate.py / predict.py
    save_model_bundle(preprocessor, model, args.model, path=args.model_path)


if __name__ == "__main__":
    main()
