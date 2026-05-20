from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import zipfile

from mlflow.tracking import MlflowClient

from .config import MLRUNS_DIR

MODEL_TYPE_LABELS = {
    "knn": ("KNN", "KNN"),
    "svm": ("SVM", "SVM"),
    "logreg": ("Logistic Regression", "LR"),
    "random_forest": ("Random Forest", "RF"),
    "decision_tree": ("Decision Tree", "DT"),
    "adaboost": ("AdaBoost", "ADA"),
    "xgboost": ("XGBoost", "XGB"),
}

MODEL_ID_MAP = {
    "knn": "knn",
    "svm": "svm",
    "logreg": "lr",
    "random_forest": "rf",
    "decision_tree": "dt",
    "adaboost": "ada",
    "xgboost": "xgb",
}


def get_client():
    return MlflowClient(tracking_uri=MLRUNS_DIR.as_uri())


def _format_time(timestamp_ms):
    if not timestamp_ms:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M")


def _duration_seconds(start_ms, end_ms):
    if not start_ms or not end_ms:
        return None
    return max(0.0, (end_ms - start_ms) / 1000)


def _normalize_status(status):
    if not status:
        return "unknown"
    status = status.lower()
    if status == "finished":
        return "completed"
    if status == "running":
        return "running"
    if status == "failed":
        return "failed"
    return status


def _extract_model_type(run):
    tags = run.data.tags or {}
    model_type = tags.get("model_type")
    if model_type:
        return model_type
    run_name = run.data.tags.get("mlflow.runName") or run.info.run_name or ""
    run_name = run_name.lower()
    if run_name.startswith("knn"):
        return "knn"
    if run_name.startswith("linearsvc") or "svm" in run_name:
        return "svm"
    if run_name.startswith("logreg"):
        return "logreg"
    if run_name.startswith("rf"):
        return "random_forest"
    if run_name.startswith("decisiontree"):
        return "decision_tree"
    if run_name.startswith("ada"):
        return "adaboost"
    if run_name.startswith("xgb") or "xgboost" in run_name:
        return "xgboost"
    return None


def _safe(val):
    """Replace NaN/Inf with None for JSON safety."""
    if val is None:
        return None
    if isinstance(val, float):
        import math
        if math.isnan(val) or math.isinf(val):
            return None
    return val


def _format_run(run):
    model_type = _extract_model_type(run)
    algo_name, short = MODEL_TYPE_LABELS.get(model_type, (run.info.run_name or "Run", None))
    metrics = run.data.metrics or {}
    params = run.data.params or {}
    start_time = run.info.start_time
    end_time = run.info.end_time

    return {
        "id": run.info.run_id,
        "name": run.info.run_name or run.info.run_id[:8],
        "algorithm": algo_name,
        "shortName": short,
        "date": _format_time(start_time),
        "status": _normalize_status(run.info.status),
        "metrics": {
            "accuracy": _safe(metrics.get("val_accuracy")),
            "precision": _safe(metrics.get("val_precision")),
            "recall": _safe(metrics.get("val_recall")),
            "f1": _safe(metrics.get("val_f1")),
            "auc": _safe(metrics.get("val_roc_auc")),
        },
        "params": params,
        "modelVersion": f"run-{run.info.run_id[:8]}",
        "datasetVersion": params.get("dataset_version", "v1.0"),
        "startTime": _format_time(start_time),
        "endTime": _format_time(end_time),
        "durationSec": _duration_seconds(start_time, end_time),
        "modelType": model_type,
    }


def list_runs(limit=100):
    client = get_client()
    experiments = client.search_experiments()
    runs = []
    for exp in experiments:
        runs.extend(
            client.search_runs(
                [exp.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=limit,
            )
        )
    runs_sorted = sorted(runs, key=lambda r: r.info.start_time or 0, reverse=True)
    return [_format_run(run) for run in runs_sorted[:limit]]


def summarize_runs(runs):
    summary = {}
    for run in runs:
        model_type = run.get("modelType")
        if not model_type:
            continue
        model_id = MODEL_ID_MAP.get(model_type)
        if not model_id:
            continue
        current = summary.get(model_id)
        current_f1 = current["metrics"].get("f1") if current else None
        new_f1 = run["metrics"].get("f1")
        if current is None or (new_f1 is not None and (current_f1 is None or new_f1 > current_f1)):
            summary[model_id] = {
                "run_id": run["id"],
                "name": run["name"],
                "algorithm": run["algorithm"],
                "metrics": run["metrics"],
                "params": run["params"],
                "durationSec": run["durationSec"],
                "modelType": model_type,
            }
    return summary


def export_model_artifact(run_id, export_dir):
    client = get_client()
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    path = Path(urlparse(artifact_uri).path)
    model_dir = path / "model"
    if not model_dir.exists():
        raise FileNotFoundError("Model artifact not found")
    export_dir.mkdir(exist_ok=True)
    zip_path = export_dir / f"{run_id}_model.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in model_dir.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(model_dir.parent))
    return zip_path
