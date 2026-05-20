"""
data_loader.py — Raw and processed data loading utilities.

Provides helpers to load the raw IEEE-CIS CSVs (transaction + identity),
merge them, and load the pre-processed train/val splits produced by the
data preparation pipeline.
"""
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, PROJECT_ROOT
from .dataset_service import load_training_split

# Raw dataset directory (dataset/ at project root)
RAW_DIR = PROJECT_ROOT / "dataset"

RAW_TRAIN_TRANSACTION = RAW_DIR / "train_transaction.csv"
RAW_TRAIN_IDENTITY = RAW_DIR / "train_identity.csv"
RAW_TEST_TRANSACTION = RAW_DIR / "test_transaction.csv"
RAW_TEST_IDENTITY = RAW_DIR / "test_identity.csv"


def load_raw(split: str = "train") -> pd.DataFrame:
    """
    Load and merge the raw transaction + identity CSVs.

    Parameters
    ----------
    split : {"train", "test"}
        Which split to load.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with transaction and identity columns.
        The ``isFraud`` column is present only for the train split.
    """
    if split == "train":
        transaction_path = RAW_TRAIN_TRANSACTION
        identity_path = RAW_TRAIN_IDENTITY
    elif split == "test":
        transaction_path = RAW_TEST_TRANSACTION
        identity_path = RAW_TEST_IDENTITY
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'train' or 'test'.")

    if not transaction_path.exists():
        raise FileNotFoundError(f"Transaction file not found: {transaction_path}")

    df_transaction = pd.read_csv(transaction_path)

    if identity_path.exists():
        df_identity = pd.read_csv(identity_path)
        df = df_transaction.merge(df_identity, on="TransactionID", how="left")
    else:
        df = df_transaction

    return df


def load_processed() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load the pre-processed train/val splits from ``dataset/processed_data/``.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    return load_training_split()


def get_feature_names() -> list[str]:
    """
    Return the list of feature column names from the processed training set.
    """
    X_train, _, _, _ = load_processed()
    return list(X_train.columns)
