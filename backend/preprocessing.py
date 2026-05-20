"""
preprocessing.py — Feature engineering and preprocessing pipeline.

Handles merging, null imputation, categorical encoding, and numeric scaling
for the IEEE-CIS Fraud Detection dataset.  The processed output matches the
column schema of ``dataset/processed_data/X_train.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Columns that are purely identifiers and should be dropped
_DROP_COLS = ["TransactionID"]

# Categorical columns that will be label-encoded
_CAT_COLS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]


def _drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that carry no predictive signal (IDs, etc.)."""
    cols_to_drop = [c for c in _DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def _impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - Numeric columns → median of the column.
    - Categorical / object columns → literal string "missing".
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("missing")
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode known categorical columns that are present in *df*."""
    df = df.copy()
    for col in _CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    # Also encode any remaining object columns not in the explicit list
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def preprocess(df: pd.DataFrame, target_col: str | None = "isFraud") -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw merged DataFrame (transaction + identity).
    target_col : str or None
        Name of the target column.  If present it is separated and returned
        as *y*; pass ``None`` for test data where no label exists.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix ready for model training / inference.
    y : pd.Series or None
        Target series, or ``None`` if *target_col* was not found / not requested.
    """
    y: pd.Series | None = None

    if target_col and target_col in df.columns:
        y = df[target_col].astype(int)
        df = df.drop(columns=[target_col])

    df = _drop_identifier_columns(df)
    df = _impute_nulls(df)
    df = _encode_categoricals(df)

    # Ensure all remaining columns are numeric (cast booleans, etc.)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            except Exception:
                df[col] = 0

    return df, y


def preprocess_single(record: dict) -> pd.DataFrame:
    """
    Preprocess a single transaction record (dict) for inference.

    Returns a single-row DataFrame with the same column schema produced by
    :func:`preprocess`.
    """
    df = pd.DataFrame([record])
    X, _ = preprocess(df, target_col=None)
    return X
