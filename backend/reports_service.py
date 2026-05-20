import json
from pathlib import Path

import pandas as pd

from .config import REPORTS_DIR

ARTIFACT_ALLOWLIST = {
    "rf_feature_importance.png",
    "rf_best_confusion_matrix.png",
    "decision_tree_confusion.png",
}


def read_markdown(filename):
    path = REPORTS_DIR / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_csv_records(filename):
    path = REPORTS_DIR / filename
    if not path.exists():
        return []
    df = pd.read_csv(path)
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and pd.isna(value):
                record[key] = None
    return records


def read_json(filename):
    path = REPORTS_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_curves(run_id):
    if not run_id:
        return None
    path = REPORTS_DIR / "curves" / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_url_map():
    return {name: f"/api/artifacts/{name}" for name in ARTIFACT_ALLOWLIST}


def resolve_artifact_path(name):
    if name not in ARTIFACT_ALLOWLIST:
        return None
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    return path
