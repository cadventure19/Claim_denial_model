"""
data_loader.py
--------------
Functions for reading raw CSVs into DataFrames and doing light
type-cleanup (boolean flag columns to 0/1 ints).
"""

import pandas as pd

from . import config


def load_csv(path: str) -> pd.DataFrame:
    """Load any CSV path into a DataFrame."""
    return pd.read_csv(path)


def cast_boolean_columns(df: pd.DataFrame, bool_cols=config.BOOLEAN_COLS) -> pd.DataFrame:
    """Convert boolean/flag columns to numeric 0/1 in place, returning the same df."""
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def check_duplicate_payer_ids(df: pd.DataFrame, col: str = "payer_id") -> int:
    """Return count of duplicate values in a given column (excluding first occurrence)."""
    if col not in df.columns:
        return 0
    duplicate_count = df[col].duplicated().sum()
    print(f"Total duplicate rows in {col}: {duplicate_count}")
    return duplicate_count


def check_nulls(df: pd.DataFrame) -> pd.Series:
    """Return and print null-value counts per column."""
    null_counts = df.isnull().sum()
    print(null_counts)
    return null_counts
