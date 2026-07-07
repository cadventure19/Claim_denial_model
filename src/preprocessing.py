"""
preprocessing.py
----------------
Train/validation/test splitting (based on the existing `split` column)
and building/fitting the scikit-learn ColumnTransformer used to scale
numeric features and one-hot encode categoricals.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from . import config


def split_by_column(df: pd.DataFrame):
    """Split a DataFrame into train/validation/test using its `split` column."""
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]
    return train_df, val_df, test_df


def get_features_and_target(df: pd.DataFrame, ignore_cols=config.IGNORE_COLS_FOR_MODEL,
                             target: str = config.TARGET_COL):
    """Split a DataFrame into X (features) and y (target), dropping ID/target columns."""
    X = df.drop(columns=ignore_cols, errors="ignore")
    y = df[target]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build (but do not fit) the ColumnTransformer for numeric/categorical/passthrough columns."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERICAL_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
             config.CATEGORICAL_COLS),
            ("pass", "passthrough", config.PASSTHROUGH_COLS),
        ]
    )


def fit_transform_splits(preprocessor: ColumnTransformer, X_train, X_val, X_test):
    """
    Fit the preprocessor on X_train only (prevents data leakage) and
    transform train/val/test. Returns the processed arrays.
    """
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    return X_train_proc, X_val_proc, X_test_proc


def get_processed_feature_names(preprocessor: ColumnTransformer,
                                 categorical_cols=config.CATEGORICAL_COLS,
                                 numerical_cols=config.NUMERICAL_COLS,
                                 passthrough_cols=config.PASSTHROUGH_COLS):
    """Reconstruct readable column names for the processed (numpy array) feature matrix."""
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_cols)
    return list(numerical_cols) + list(encoded_cat_cols) + list(passthrough_cols)


def prepare_train_val_test(claims_history: pd.DataFrame):
    """
    End-to-end prep: split by `split` column, separate X/y, build and fit
    the preprocessor, and return everything needed for training/evaluation.
    """
    train_df, val_df, test_df = split_by_column(claims_history)

    X_train, y_train = get_features_and_target(train_df)
    X_val, y_val = get_features_and_target(val_df)
    X_test, y_test = get_features_and_target(test_df)

    preprocessor = build_preprocessor()
    X_train_proc, X_val_proc, X_test_proc = fit_transform_splits(
        preprocessor, X_train, X_val, X_test
    )

    return {
        "preprocessor": preprocessor,
        "X_train_proc": X_train_proc, "y_train": y_train,
        "X_val_proc": X_val_proc, "y_val": y_val,
        "X_test_proc": X_test_proc, "y_test": y_test,
    }
