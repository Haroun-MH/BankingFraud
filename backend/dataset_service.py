import json
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, REPORTS_DIR

WORKING_DIR = DATA_DIR / "working"
CACHE_PATH = REPORTS_DIR / "dataset_cache.json"
METADATA_PATH = WORKING_DIR / "dataset_metadata.json"

BASE_X_TRAIN = DATA_DIR / "X_train.csv"
BASE_X_VAL = DATA_DIR / "X_val.csv"
BASE_Y_TRAIN = DATA_DIR / "y_train.csv"
BASE_Y_VAL = DATA_DIR / "y_val.csv"

WORKING_X_TRAIN = WORKING_DIR / "X_train.csv"
WORKING_X_VAL = WORKING_DIR / "X_val.csv"
WORKING_Y_TRAIN = WORKING_DIR / "y_train.csv"
WORKING_Y_VAL = WORKING_DIR / "y_val.csv"

DEFAULT_METADATA = {
    "name": "IEEE-CIS Fraud Detection",
    "version": "v1.0",
    "source": "processed_data",
}


class DatasetError(Exception):
    pass


def _ensure_working_dir():
    WORKING_DIR.mkdir(parents=True, exist_ok=True)


def _active_paths():
    if WORKING_X_TRAIN.exists() and WORKING_Y_TRAIN.exists():
        return {
            "x_train": WORKING_X_TRAIN,
            "y_train": WORKING_Y_TRAIN,
            "x_val": WORKING_X_VAL if WORKING_X_VAL.exists() else BASE_X_VAL,
            "y_val": WORKING_Y_VAL if WORKING_Y_VAL.exists() else BASE_Y_VAL,
        }
    return {
        "x_train": BASE_X_TRAIN,
        "y_train": BASE_Y_TRAIN,
        "x_val": BASE_X_VAL,
        "y_val": BASE_Y_VAL,
    }


def get_dataset_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return dict(DEFAULT_METADATA)


def _write_metadata(name, version, source):
    _ensure_working_dir()
    payload = {"name": name, "version": version, "source": source}
    METADATA_PATH.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _dataset_signature(paths):
    signature = {
        "x_train": str(paths["x_train"]),
        "y_train": str(paths["y_train"]),
        "x_val": str(paths["x_val"]),
        "y_val": str(paths["y_val"]),
    }
    for key, path in paths.items():
        if path.exists():
            signature[f"{key}_mtime"] = path.stat().st_mtime
    return signature


def _load_cached(signature):
    if not CACHE_PATH.exists():
        return None
    try:
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached.get("signature") != signature:
        return None
    return cached


def _save_cache(signature, summary, features):
    payload = {"signature": signature, "summary": summary, "features": features}
    REPORTS_DIR.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")


def _load_train_data(paths):
    X_train = pd.read_csv(paths["x_train"])
    y_train = pd.read_csv(paths["y_train"]).iloc[:, 0]
    return X_train, y_train


def _load_val_data(paths):
    if not paths["x_val"].exists() or not paths["y_val"].exists():
        return None, None
    X_val = pd.read_csv(paths["x_val"])
    y_val = pd.read_csv(paths["y_val"]).iloc[:, 0]
    return X_val, y_val


def _compute_stats(paths):
    X_train, y_train = _load_train_data(paths)
    nulls = X_train.isna().sum().to_dict()
    uniques = X_train.nunique(dropna=True).to_dict()
    dtypes = X_train.dtypes.to_dict()

    features = []
    for col in X_train.columns:
        features.append(
            {
                "name": col,
                "type": str(dtypes[col]),
                "nulls": int(nulls[col]),
                "unique": int(uniques[col]),
            }
        )

    metadata = get_dataset_metadata()
    fraud_rate = float(y_train.mean() * 100) if len(y_train) else 0.0
    summary = {
        "name": metadata.get("name", DEFAULT_METADATA["name"]),
        "version": metadata.get("version", DEFAULT_METADATA["version"]),
        "rows": int(len(X_train)),
        "columns": int(len(X_train.columns)),
        "fraud_rate": round(fraud_rate, 3),
    }
    return summary, features


def get_dataset_summary():
    paths = _active_paths()
    signature = _dataset_signature(paths)
    cached = _load_cached(signature)
    if cached:
        return cached.get("summary", {})
    summary, features = _compute_stats(paths)
    _save_cache(signature, summary, features)
    return summary


def get_dataset_features():
    paths = _active_paths()
    signature = _dataset_signature(paths)
    cached = _load_cached(signature)
    if cached:
        return cached.get("features", [])
    summary, features = _compute_stats(paths)
    _save_cache(signature, summary, features)
    return features


def get_dataset_preview(limit=10):
    paths = _active_paths()
    X_preview = pd.read_csv(paths["x_train"], nrows=limit)
    y_preview = pd.read_csv(paths["y_train"], nrows=limit).iloc[:, 0]
    X_preview = X_preview.copy()
    X_preview["isFraud"] = y_preview.values
    records = X_preview.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and pd.isna(value):
                record[key] = None
    return records


def reset_dataset():
    if WORKING_DIR.exists():
        for path in WORKING_DIR.glob("*"):
            if path.is_file():
                path.unlink()
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()


def _write_working_split(X_train, y_train, X_val=None, y_val=None):
    _ensure_working_dir()
    X_train.to_csv(WORKING_X_TRAIN, index=False)
    y_train.to_frame("isFraud").to_csv(WORKING_Y_TRAIN, index=False)
    if X_val is not None and y_val is not None:
        X_val.to_csv(WORKING_X_VAL, index=False)
        y_val.to_frame("isFraud").to_csv(WORKING_Y_VAL, index=False)


def apply_cleaning(action, column):
    paths = _active_paths()
    X_train, y_train = _load_train_data(paths)
    X_val, y_val = _load_val_data(paths)

    if column not in X_train.columns:
        raise DatasetError(f"Column not found: {column}")

    if action == "drop_nulls":
        mask = X_train[column].notna()
        X_train = X_train.loc[mask]
        y_train = y_train.loc[mask]
        if X_val is not None:
            mask_val = X_val[column].notna()
            X_val = X_val.loc[mask_val]
            y_val = y_val.loc[mask_val]
    elif action == "fill_nulls":
        if pd.api.types.is_numeric_dtype(X_train[column]):
            fill_value = X_train[column].median()
        else:
            mode = X_train[column].mode()
            fill_value = mode.iloc[0] if not mode.empty else "missing"
        X_train[column] = X_train[column].fillna(fill_value)
        if X_val is not None:
            X_val[column] = X_val[column].fillna(fill_value)
    else:
        raise DatasetError(f"Unknown action: {action}")

    _write_working_split(X_train, y_train, X_val, y_val)
    _write_metadata(
        get_dataset_metadata().get("name", DEFAULT_METADATA["name"]),
        f"cleaned-{int(time.time())}",
        "cleaning",
    )
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()


def upload_dataset(file_path, filename):
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise DatasetError("Unsupported file type")

    if "isFraud" not in df.columns:
        raise DatasetError("Uploaded dataset must include an isFraud column")

    y = df["isFraud"]
    X = df.drop(columns=["isFraud"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    _write_working_split(X_train, y_train, X_val, y_val)
    _write_metadata(filename, f"upload-{int(time.time())}", "upload")
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()


def export_dataset(columns=None, limit=None):
    paths = _active_paths()
    X_train, y_train = _load_train_data(paths)
    if columns:
        missing = [col for col in columns if col not in X_train.columns]
        if missing:
            raise DatasetError(f"Unknown columns: {', '.join(missing)}")
        X_train = X_train[columns]
    X_train = X_train.copy()
    X_train["isFraud"] = y_train.values
    if limit:
        X_train = X_train.head(limit)

    export_dir = REPORTS_DIR / "exports"
    export_dir.mkdir(exist_ok=True)
    path = export_dir / f"dataset_export_{int(time.time())}.csv"
    X_train.to_csv(path, index=False)
    return path


def load_training_split():
    paths = _active_paths()
    X_train, y_train = _load_train_data(paths)
    X_val, y_val = _load_val_data(paths)
    if X_val is None or y_val is None:
        raise DatasetError("Validation split is missing")
    return X_train, y_train, X_val, y_val
