from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
from uuid import uuid4

try:
    import psutil
except ImportError:
    psutil = None

from .config import ALLOWED_ORIGINS, REPORTS_DIR
from .jobs import JobStore
from .mlflow_service import list_runs, summarize_runs, export_model_artifact
from .model_registry import get_active_model, set_active_model
from .dataset_service import (
    get_dataset_summary,
    get_dataset_features,
    get_dataset_preview,
    apply_cleaning,
    reset_dataset,
    export_dataset,
    upload_dataset,
    DatasetError,
)
from .reports_service import (
    artifact_url_map,
    read_csv_records,
    read_json,
    read_markdown,
    read_curves,
    resolve_artifact_path,
)
from .training import run_task3_task4, run_tuning, CancelledError
from .registry_service import (
    register_model as svc_register_model,
    promote_model as svc_promote_model,
    list_registered_models,
    get_production_model_uri,
    DEFAULT_MODEL_NAME,
)
from .preprocessing import preprocess_single

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = JobStore()


class TrainRequest(BaseModel):
    run_task4: bool = True
    selected_models: Optional[List[str]] = None
    use_sample: bool = True
    costly_max_samples: Optional[int] = None
    run_pca: bool = False
    run_tsne: bool = False
    custom_params: Optional[dict] = None
    search_method: Optional[str] = None
    max_trials: Optional[int] = None
    time_limit: Optional[float] = None


class TuneRequest(BaseModel):
    model_type: str
    search_method: Optional[str] = "random"
    max_trials: Optional[int] = 12
    search_space: Optional[List[Dict[str, Any]]] = None


class CleanRequest(BaseModel):
    action: str
    column: str


class ActiveModelRequest(BaseModel):
    run_id: str


class RegisterModelRequest(BaseModel):
    run_id: str
    model_name: Optional[str] = None


class PromoteModelRequest(BaseModel):
    model_name: Optional[str] = None
    version: Optional[str] = None
    accuracy_threshold: Optional[float] = None


def _run_training_job(job_id, options):
    jobs.start_job(job_id, total_steps=None)
    try:
        result = run_task3_task4(options=options, job_store=jobs, job_id=job_id)
        jobs.finish_job(job_id, "completed", result=result, message="Training complete")
    except CancelledError as exc:
        jobs.finish_job(job_id, "cancelled", message=str(exc))
    except Exception as exc:
        jobs.finish_job(job_id, "failed", message=str(exc))


def _run_tuning_job(job_id, options):
    jobs.start_job(job_id, total_steps=None)
    try:
        result = run_tuning(options=options, job_store=jobs, job_id=job_id)
        jobs.finish_job(job_id, "completed", result=result, message="Tuning complete")
    except CancelledError as exc:
        jobs.finish_job(job_id, "cancelled", message=str(exc))
    except Exception as exc:
        jobs.finish_job(job_id, "failed", message=str(exc))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/runs")
def get_runs(limit: int = 50):
    runs = list_runs(limit=limit)
    return runs


@app.get("/api/summary")
def get_summary():
    runs = list_runs(limit=200)
    models = summarize_runs(runs)

    curves = {}
    for model_id, info in models.items():
        curves[model_id] = read_curves(info.get("run_id"))

    tables = {
        "task3_comparison": read_csv_records("task3_comparison.csv"),
        "rf_bias_variance": read_csv_records("rf_bias_variance.csv"),
        "rf_stability": read_csv_records("rf_stability.csv"),
        "rf_misclassified_samples": read_csv_records("rf_misclassified_samples.csv"),
    }

    reports = {
        "task3_md": read_markdown("task3_experiments.md"),
        "task4_md": read_markdown("task4_random_forest.md"),
    }

    confusion = {
        "rf": read_json("rf_confusion.json"),
        "dt": read_json("decision_tree_confusion.json"),
    }

    active_model = get_active_model()

    return {
        "models": models,
        "runs": runs,
        "tables": tables,
        "reports": reports,
        "artifacts": artifact_url_map(),
        "confusion": confusion,
        "curves": curves,
        "active_model": active_model,
    }


@app.post("/api/train")
def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    params = request.dict()
    if params.get("costly_max_samples") is None:
        params["costly_max_samples"] = 20000

    job = jobs.create_job("training", params=params)
    background_tasks.add_task(_run_training_job, job["id"], params)
    return {"job_id": job["id"]}


@app.post("/api/tune")
def start_tuning(request: TuneRequest, background_tasks: BackgroundTasks):
    params = request.dict()
    job = jobs.create_job("tuning", params=params)
    background_tasks.add_task(_run_tuning_job, job["id"], params)
    return {"job_id": job["id"]}


@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str):
    job = jobs.pause_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    job = jobs.resume_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = jobs.cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/artifacts/{name}")
def get_artifact(name: str):
    path = resolve_artifact_path(name)
    if not path:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)


@app.get("/api/system")
def get_system_metrics():
    if not psutil:
        return {"cpu": None, "gpu": None, "ram": None, "gpu_mem": None}
    return {
        "cpu": psutil.cpu_percent(interval=None),
        "gpu": None,
        "ram": psutil.virtual_memory().percent,
        "gpu_mem": None,
    }


@app.get("/api/dataset/summary")
def dataset_summary():
    return get_dataset_summary()


@app.get("/api/dataset/features")
def dataset_features():
    return get_dataset_features()


@app.get("/api/dataset/preview")
def dataset_preview(limit: int = 10):
    return get_dataset_preview(limit=limit)


@app.post("/api/dataset/clean")
def dataset_clean(request: CleanRequest):
    try:
        apply_cleaning(request.action, request.column)
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "summary": get_dataset_summary(),
        "features": get_dataset_features(),
    }


@app.post("/api/dataset/reset")
def dataset_reset():
    reset_dataset()
    return {
        "summary": get_dataset_summary(),
        "features": get_dataset_features(),
    }


@app.post("/api/dataset/upload")
async def dataset_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    upload_dir = REPORTS_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    path = upload_dir / file.filename
    content = await file.read()
    path.write_bytes(content)
    try:
        upload_dataset(path, file.filename)
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "summary": get_dataset_summary(),
        "features": get_dataset_features(),
    }


@app.get("/api/dataset/export")
def dataset_export(columns: Optional[str] = None, limit: Optional[int] = None):
    try:
        column_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        path = export_dataset(column_list, limit)
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return FileResponse(path, filename=path.name)


@app.get("/api/model/active")
def get_active_model_endpoint():
    return get_active_model() or {}


@app.post("/api/model/active")
def set_active_model_endpoint(request: ActiveModelRequest):
    return set_active_model(request.run_id)


@app.get("/api/runs/{run_id}/export")
def export_run_model(run_id: str):
    export_dir = REPORTS_DIR / "exports"
    try:
        path = export_model_artifact(run_id, export_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model artifact not found")
    return FileResponse(path, filename=path.name)


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    model_dir = REPORTS_DIR / "model_uploads"
    model_dir.mkdir(exist_ok=True)
    model_id = uuid4().hex
    suffix = Path(file.filename).suffix or ".bin"
    path = model_dir / f"{model_id}{suffix}"
    content = await file.read()
    path.write_bytes(content)
    return {"model_id": model_id, "filename": file.filename}


# ---------------------------------------------------------------------------
# MLflow Model Registry endpoints
# ---------------------------------------------------------------------------

@app.post("/api/registry/register")
def registry_register(request: RegisterModelRequest):
    """Register a run's model artifact in the MLflow Model Registry."""
    try:
        result = svc_register_model(
            run_id=request.run_id,
            model_name=request.model_name or DEFAULT_MODEL_NAME,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/api/registry/promote")
def registry_promote(request: PromoteModelRequest):
    """Promote the latest (or specified) model version to Staging/Production."""
    kwargs: Dict[str, Any] = {}
    if request.model_name:
        kwargs["model_name"] = request.model_name
    if request.version:
        kwargs["version"] = request.version
    if request.accuracy_threshold is not None:
        kwargs["accuracy_threshold"] = request.accuracy_threshold
    try:
        result = svc_promote_model(**kwargs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/api/registry/versions")
def registry_versions(model_name: Optional[str] = None):
    """List all registered model versions."""
    return list_registered_models(model_name or DEFAULT_MODEL_NAME)


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

# Cache the loaded model to avoid reloading on every request
_predict_cache: Dict[str, Any] = {"model": None, "run_id": None}


def _load_predict_model():
    """Load the active model for inference, with a simple in-memory cache."""
    active = get_active_model()
    run_id = active.get("run_id") if active else None

    if run_id and _predict_cache["run_id"] == run_id and _predict_cache["model"] is not None:
        return _predict_cache["model"]

    if run_id:
        try:
            import mlflow.sklearn
            from .config import MLRUNS_DIR
            mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
            model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
            _predict_cache["model"] = model
            _predict_cache["run_id"] = run_id
            return model
        except Exception:
            pass

    # Fallback: try Production model from registry
    try:
        import mlflow.sklearn
        from .config import MLRUNS_DIR
        mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
        prod_uri = get_production_model_uri()
        if prod_uri:
            model = mlflow.sklearn.load_model(prod_uri)
            _predict_cache["model"] = model
            _predict_cache["run_id"] = "production"
            return model
    except Exception:
        pass

    return None


@app.post("/api/predict")
def predict(transaction: Dict[str, Any]):
    """
    Run fraud prediction on a single transaction record.

    Accepts a JSON object with transaction fields and returns
    ``{"fraud_probability": float, "prediction": 0|1}``.
    """
    model = _load_predict_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No active model available. Train a model and set it as active first.",
        )
    try:
        X = preprocess_single(transaction)
        # Align columns to what the model was trained on
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            X = X[model.feature_names_in_]

        prediction = int(model.predict(X)[0])
        probability: float | None = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0][1])
        elif hasattr(model, "decision_function"):
            score = float(model.decision_function(X)[0])
            # Sigmoid approximation for models without predict_proba
            import math
            probability = 1.0 / (1.0 + math.exp(-score))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    return {
        "prediction": prediction,
        "fraud_probability": probability,
        "is_fraud": bool(prediction == 1),
    }


# ---------------------------------------------------------------------------
# Drift detection endpoint
# ---------------------------------------------------------------------------

@app.post("/api/drift/detect")
def detect_drift(background_tasks: BackgroundTasks, retrain: bool = False):
    """
    Run the drift detection pipeline in the background.
    Returns a job_id to poll via /api/jobs/{job_id}.
    """
    def _run(job_id: str, force_retrain: bool):
        jobs.start_job(job_id)
        try:
            from .simulate_drift import run_drift_detection
            result = run_drift_detection(retrain=force_retrain)
            jobs.finish_job(job_id, "completed", result=result, message=f"Drift status: {result['status']}")
        except Exception as exc:
            jobs.finish_job(job_id, "failed", message=str(exc))

    job = jobs.create_job("drift_detection", params={"retrain": retrain})
    background_tasks.add_task(_run, job["id"], retrain)
    return {"job_id": job["id"]}
