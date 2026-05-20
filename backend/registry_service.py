"""
registry_service.py — MLflow Model Registry integration.

Handles registering runs, tagging versions, and managing the
Staging → Production lifecycle via the MLflow tracking client.
"""
from __future__ import annotations

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from .config import MLRUNS_DIR

PRODUCTION_ACCURACY_THRESHOLD = 0.85
DEFAULT_MODEL_NAME = "FraudDetectionModel"


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=MLRUNS_DIR.as_uri())


def register_model(run_id: str, model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """
    Register the model artifact from *run_id* in the MLflow Model Registry.

    Returns a dict with ``name``, ``version``, and ``status``.
    """
    import mlflow

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = _client()
    # Add a description to the registered model
    try:
        client.update_registered_model(
            name=model_name,
            description=(
                "Fraud detection model trained on the IEEE-CIS dataset. "
                "Tracks KNN, SVM, Logistic Regression, Random Forest, AdaBoost, and XGBoost experiments."
            ),
        )
    except MlflowException:
        pass

    # Tag the version with the source run id
    try:
        client.set_model_version_tag(model_name, mv.version, "source_run_id", run_id)
        client.set_model_version_tag(model_name, mv.version, "registered_by", "fraud_detection_api")
    except MlflowException:
        pass

    return {
        "name": mv.name,
        "version": mv.version,
        "status": mv.status,
        "run_id": run_id,
    }


def promote_model(
    model_name: str = DEFAULT_MODEL_NAME,
    version: str | None = None,
    accuracy_threshold: float = PRODUCTION_ACCURACY_THRESHOLD,
) -> dict:
    """
    Promote the latest (or specified) model version through Staging → Production.

    1. Transitions the version to **Staging**.
    2. Reads the ``val_accuracy`` metric from the source run.
    3. If accuracy >= *accuracy_threshold*, transitions to **Production**.

    Returns a dict describing the outcome.
    """
    client = _client()

    # Resolve version
    if version is None:
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        mv = sorted(versions, key=lambda v: int(v.version))[-1]
        version = mv.version
    else:
        mv = client.get_model_version(model_name, version)

    run_id = mv.run_id

    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )

    # Check accuracy from the MLflow run
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("val_accuracy")

    promoted_to_production = False
    if accuracy is not None and accuracy >= accuracy_threshold:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        promoted_to_production = True

    return {
        "name": model_name,
        "version": version,
        "run_id": run_id,
        "val_accuracy": accuracy,
        "threshold": accuracy_threshold,
        "stage": "Production" if promoted_to_production else "Staging",
        "promoted_to_production": promoted_to_production,
    }


def list_registered_models(model_name: str = DEFAULT_MODEL_NAME) -> list[dict]:
    """Return all registered versions for *model_name*."""
    client = _client()
    try:
        versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production", "Archived"])
    except MlflowException:
        return []
    return [
        {
            "name": v.name,
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "status": v.status,
            "creation_timestamp": v.creation_timestamp,
        }
        for v in versions
    ]


def get_production_model_uri(model_name: str = DEFAULT_MODEL_NAME) -> str | None:
    """Return the model URI for the current Production version, or None."""
    client = _client()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            v = versions[0]
            return f"models:/{model_name}/Production"
    except MlflowException:
        pass
    return None
