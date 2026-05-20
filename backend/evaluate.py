"""
evaluate.py — Model evaluation utilities.

Provides standalone evaluation helpers used by training.py and the /api/evaluate endpoint.
"""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import REPORTS_DIR


def evaluate_model(model, X, y, run_id: str | None = None) -> dict:
    """
    Evaluate *model* on (X, y) and return a metrics dict.

    If *run_id* is provided the classification report text is saved to
    ``REPORTS_DIR/classification_report_{run_id}.txt`` and logged as an
    MLflow artifact.
    """
    y_pred = model.predict(X)

    # Probability / decision score for ROC-AUC
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)

    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, y_score))
        except ValueError:
            pass

    # Classification report text
    report_text = classification_report(y, y_pred, zero_division=0)
    metrics["classification_report"] = report_text

    # Confusion matrix as dict
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    # Persist and log the classification report when a run_id is given
    if run_id:
        REPORTS_DIR.mkdir(exist_ok=True)
        report_path = REPORTS_DIR / f"classification_report_{run_id}.txt"
        report_path.write_text(report_text, encoding="utf-8")
        try:
            if mlflow.active_run() is not None:
                mlflow.log_artifact(str(report_path))
            else:
                from mlflow.tracking import MlflowClient
                from .config import MLRUNS_DIR

                client = MlflowClient(tracking_uri=MLRUNS_DIR.as_uri())
                client.log_artifact(run_id, str(report_path))
        except Exception:
            pass  # Logging is best-effort; don't break evaluation

    return metrics


def get_feature_importances(model, feature_names: list) -> list[dict]:
    """
    Return a list of ``{"feature": name, "importance": value}`` dicts sorted
    descending by importance.  Returns an empty list for models that don't
    expose ``feature_importances_``.
    """
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    pairs = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )
    return [{"feature": f, "importance": float(imp)} for f, imp in pairs]
