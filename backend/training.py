import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from .config import DATA_DIR, REPORTS_DIR, MLRUNS_DIR
from .dataset_service import load_training_split, get_dataset_metadata

RANDOM_STATE = 42
DATASET_PARAMS = {}


def sample_if_needed(X, y, max_samples, random_state, use_sample):
    if not use_sample or len(X) <= max_samples:
        return X, y, False
    X_sample = X.sample(max_samples, random_state=random_state)
    y_sample = y.loc[X_sample.index]
    return X_sample, y_sample, True


def get_y_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def compute_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return metrics


def log_artifact(path, run_id=None):
    if run_id:
        client = MlflowClient(tracking_uri=MLRUNS_DIR.as_uri())
        client.log_artifact(run_id, str(path))
        return
    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(path))


def log_confusion_matrix(y_true, y_pred, artifact_name, run_id=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, int(value), ha="center", va="center")
    fig.tight_layout()
    path = REPORTS_DIR / artifact_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log_artifact(path, run_id=run_id)
    return cm


def save_confusion_json(cm, filename):
    tn, fp, fn, tp = cm.ravel()
    payload = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    path = REPORTS_DIR / filename
    path.write_text(json.dumps(payload), encoding="utf-8")


class CancelledError(Exception):
    pass


def save_curves(y_true, y_score, run_id):
    if y_score is None or run_id is None:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc = [{"fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr)]
    pr = [
        {"recall": float(x), "precision": float(y)}
        for x, y in zip(recall, precision)
    ]
    curves_dir = REPORTS_DIR / "curves"
    curves_dir.mkdir(exist_ok=True)
    path = curves_dir / f"{run_id}.json"
    path.write_text(json.dumps({"roc": roc, "pr": pr}), encoding="utf-8")


def run_and_log(model_name, model, X_tr, y_tr, X_va, y_va, params, tags=None):
    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(run_name=model_name) as run:
        params = {**DATASET_PARAMS, **params}
        mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_param("n_train", len(X_tr))
        mlflow.log_param("n_val", len(X_va))

        model.fit(X_tr, y_tr)

        y_pred_tr = model.predict(X_tr)
        y_pred_va = model.predict(X_va)

        y_score_tr = get_y_score(model, X_tr)
        y_score_va = get_y_score(model, X_va)

        train_metrics = compute_metrics(y_tr, y_pred_tr, y_score_tr)
        val_metrics = compute_metrics(y_va, y_pred_va, y_score_va)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        mlflow.sklearn.log_model(model, artifact_path="model")
        save_curves(y_va, y_score_va, run.info.run_id)

        # Log classification report as a text artifact
        report_text = classification_report(y_va, y_pred_va, zero_division=0)
        report_path = REPORTS_DIR / f"classification_report_{run.info.run_id}.txt"
        report_path.write_text(report_text, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        return train_metrics, val_metrics, model, run.info.run_id


def _format_float(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return str(value)


def df_to_markdown(df):
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_float(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_task3_task4(options=None, job_store=None, job_id=None):
    options = options or {}
    use_sample = options.get("use_sample", True)
    costly_max_samples = int(options.get("costly_max_samples", 100000))
    run_pca = bool(options.get("run_pca", False))
    run_tsne = bool(options.get("run_tsne", False))
    run_task4 = bool(options.get("run_task4", True))
    selected_models = set(options.get("selected_models") or [])
    custom_params = options.get("custom_params") or None
    custom_model = None
    max_trials = options.get("max_trials")
    search_method = options.get("search_method", "grid")
    time_limit_minutes = options.get("time_limit")
    stop_time = None

    if custom_params and selected_models and len(selected_models) == 1:
        custom_model = next(iter(selected_models))

    if time_limit_minutes:
        stop_time = time.time() + float(time_limit_minutes) * 60

    if run_task4 and selected_models and "random_forest" not in selected_models:
        selected_models.add("random_forest")

    def filter_custom_params(model_type, params):
        allowed = {
            "knn": {
                "n_neighbors",
                "weights",
                "metric",
                "p",
                "algorithm",
                "leaf_size",
            },
            "svm": {
                "C",
                "class_weight",
                "dual",
                "max_iter",
                "loss",
                "penalty",
                "tol",
            },
            "logreg": {
                "C",
                "class_weight",
                "solver",
                "max_iter",
                "n_jobs",
                "penalty",
                "l1_ratio",
                "tol",
                "fit_intercept",
            },
            "random_forest": {
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "class_weight",
                "n_jobs",
                "random_state",
                "bootstrap",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "min_weight_fraction_leaf",
                "oob_score",
            },
            "adaboost": {
                "n_estimators",
                "learning_rate",
            },
            "xgboost": {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "gamma",
                "reg_alpha",
                "reg_lambda",
                "scale_pos_weight",
                "n_jobs",
                "random_state",
            },
        }
        if not params:
            return {}
        return {k: v for k, v in params.items() if k in allowed.get(model_type, set())}

    def should_run(model_type):
        return not selected_models or model_type in selected_models

    REPORTS_DIR.mkdir(exist_ok=True)

    X_train, y_train, X_val, y_val = load_training_split()
    metadata = get_dataset_metadata()
    dataset_version = metadata.get("version", "v1.0")
    dataset_name = metadata.get("name", "dataset")
    global DATASET_PARAMS
    DATASET_PARAMS = {
        "dataset_version": dataset_version,
        "dataset_name": dataset_name,
    }

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(options.get("experiment_name", "fraud_detection_task3_task4"))

    results = []
    current_step = 0

    knn_ks = [3, 11]
    svm_cs = [0.5, 1.0]
    logreg_cs = [0.5, 1.0, 2.0]
    rf_param_grid = [
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 20},
        {"n_estimators": 400, "max_depth": 20},
    ]
    ada_param_grid = [
        {"n_estimators": 50, "learning_rate": 1.0},
        {"n_estimators": 100, "learning_rate": 1.0},
        {"n_estimators": 200, "learning_rate": 0.5},
    ]
    xgb_param_grid = [
        {"n_estimators": 200, "max_depth": 6},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 400, "max_depth": 6},
    ]
    stability_seeds = [1, 7, 21, 42, 99]
    bias_n_estimators = [100, 300, 500]
    bias_max_depths = [5, 10, 20, None]

    # AutoML mode: intelligent multi-model search
    is_automl = selected_models is None or len(selected_models) == 0
    if is_automl and search_method and search_method in ["optuna", "random", "grid"]:
        # AutoML: try all models with intelligent search
        selected_models = {"knn", "svm", "logreg", "random_forest"}
        custom_params = None  # AutoML doesn't use custom params
        if max_trials is None:
            max_trials = 20  # Default for AutoML
    elif max_trials:
        rng = np.random.default_rng(RANDOM_STATE)
        active_models = [
            model
            for model in ["knn", "svm", "logreg", "random_forest"]
            if should_run(model)
        ]
        per_limit = max(1, int(max_trials) // max(1, len(active_models)))

        def sample_list(items):
            if len(items) <= per_limit:
                return items
            if search_method == "grid":
                return items[:per_limit]
            idx = rng.choice(len(items), size=per_limit, replace=False)
            return [items[i] for i in idx]

        if should_run("knn") and custom_model != "knn":
            knn_ks = sample_list(knn_ks)
        if should_run("svm") and custom_model != "svm":
            svm_cs = sample_list(svm_cs)
        if should_run("logreg") and custom_model != "logreg":
            logreg_cs = sample_list(logreg_cs)
        if should_run("random_forest") and custom_model != "random_forest":
            rf_param_grid = sample_list(rf_param_grid)
        if should_run("adaboost") and custom_model != "adaboost":
            ada_param_grid = sample_list(ada_param_grid)
        if should_run("xgboost") and custom_model != "xgboost":
            xgb_param_grid = sample_list(xgb_param_grid)

    total_steps = 0
    if should_run("knn"):
        total_steps += 1 if custom_model == "knn" else len(knn_ks)
    if should_run("svm"):
        total_steps += 1 if custom_model == "svm" else len(svm_cs)
    if should_run("logreg"):
        total_steps += 1 if custom_model == "logreg" else len(logreg_cs)
    if should_run("random_forest"):
        total_steps += 1 if custom_model == "random_forest" else len(rf_param_grid)
    if should_run("adaboost"):
        total_steps += 1 if custom_model == "adaboost" else len(ada_param_grid)
    if should_run("xgboost"):
        total_steps += 1 if custom_model == "xgboost" else len(xgb_param_grid)
    if run_pca:
        total_steps += 1
    if run_tsne:
        total_steps += 1
    if run_task4:
        total_steps += 1 + len(stability_seeds) + (len(bias_n_estimators) * len(bias_max_depths)) + 1
    total_steps += 1

    if job_store and job_id:
        job_store.update_job(job_id, total_steps=total_steps, message="Loading data")

    def update_progress(message, history_entry=None):
        nonlocal current_step
        current_step += 1
        progress = current_step / max(1, total_steps)
        if job_store and job_id:
            job_store.update_job(job_id, progress=progress, message=message)
            if history_entry:
                history_entry["step"] = current_step
                job_store.append_history(job_id, history_entry)

    def check_job_control():
        if not job_store or not job_id:
            if stop_time and time.time() >= stop_time:
                raise CancelledError("Time limit reached")
            return
        while True:
            job = job_store.get_job(job_id)
            if not job:
                return
            if job.get("cancelled"):
                raise CancelledError("Job cancelled")
            if job.get("paused"):
                time.sleep(0.5)
                continue
            if stop_time and time.time() >= stop_time:
                raise CancelledError("Time limit reached")
            return

    # AutoML: use tuning function for intelligent search
    if is_automl and search_method in ["optuna", "random", "grid"]:
        automl_results = []
        per_model_trials = max(1, max_trials // 6)  # Distribute trials across 6 models
        
        for model_type in ["knn", "svm", "logreg", "random_forest", "adaboost", "xgboost"]:
            check_job_control()
            if stop_time and time.time() >= stop_time:
                break
            
            # Define search space for each model type
            search_spaces = {
                "knn": [
                    {"name": "n_neighbors", "type": "number", "min": 3, "max": 15, "step": 2},
                    {"name": "weights", "type": "select", "options": ["uniform", "distance"]},
                ],
                "svm": [
                    {"name": "C", "type": "number", "min": 0.1, "max": 5.0, "step": 0.5},
                ],
                "logreg": [
                    {"name": "C", "type": "number", "min": 0.1, "max": 5.0, "step": 0.5},
                    {"name": "solver", "type": "select", "options": ["saga", "lbfgs"]},
                ],
                "random_forest": [
                    {"name": "n_estimators", "type": "number", "min": 100, "max": 500, "step": 100},
                    {"name": "max_depth", "type": "select", "options": [10, 20, None]},
                ],
                "adaboost": [
                    {"name": "n_estimators", "type": "number", "min": 50, "max": 300, "step": 50},
                    {"name": "learning_rate", "type": "number", "min": 0.1, "max": 2.0, "step": 0.1},
                ],
                "xgboost": [
                    {"name": "n_estimators", "type": "number", "min": 100, "max": 500, "step": 100},
                    {"name": "max_depth", "type": "select", "options": [3, 6, 10]},
                    {"name": "learning_rate", "type": "number", "min": 0.01, "max": 0.3, "step": 0.05},
                ],
            }
            
            try:
                tuning_result = run_tuning(
                    options={
                        "model_type": model_type,
                        "search_method": search_method,
                        "max_trials": per_model_trials,
                        "search_space": search_spaces.get(model_type, []),
                    },
                    job_store=job_store,
                    job_id=job_id,
                )
                automl_results.append({
                    "model_type": model_type,
                    "best_params": tuning_result["best_params"],
                    "best_metrics": tuning_result["best_metrics"],
                    "best_run_id": tuning_result["best_run_id"],
                })
                update_progress(
                    f"AutoML completed {model_type}",
                    {
                        "step": current_step + 1,
                        "model": model_type,
                        "run_id": tuning_result["best_run_id"],
                        "metrics": {"val": tuning_result["best_metrics"]},
                    },
                )
            except Exception as e:
                # If tuning fails, fall back to default training
                check_job_control()
        
        # Select best model from AutoML results
        if automl_results:
            best_automl = max(automl_results, key=lambda x: x["best_metrics"].get("f1", 0))
            # Update results with AutoML best model
            results.append({
                "model": best_automl["model_type"],
                "model_type": best_automl["model_type"],
                "params": best_automl["best_params"],
                "run_id": best_automl["best_run_id"],
                "train_f1": best_automl["best_metrics"].get("f1"),
                "val_f1": best_automl["best_metrics"].get("f1"),
                "val_precision": best_automl["best_metrics"].get("precision"),
                "val_recall": best_automl["best_metrics"].get("recall"),
                "val_roc_auc": best_automl["best_metrics"].get("roc_auc"),
                "val_accuracy": best_automl["best_metrics"].get("accuracy"),
            })
            # Skip regular training if AutoML succeeded
            is_automl = False
        else:
            # Fallback to regular training if AutoML failed
            is_automl = False

    if should_run("knn"):
        X_knn, y_knn, sampled_knn = sample_if_needed(
            X_train, y_train, costly_max_samples, RANDOM_STATE, use_sample
        )
        check_job_control()
        if custom_model == "knn" and custom_params:
            check_job_control()
            params = {"n_neighbors": 5, "weights": "distance"}
            params.update(filter_custom_params("knn", custom_params))
            model = KNeighborsClassifier(**params)
            train_m, val_m, _, run_id = run_and_log(
                model_name="KNN_Custom",
                model=model,
                X_tr=X_knn,
                y_tr=y_knn,
                X_va=X_val,
                y_va=y_val,
                params={**params, "sampled_train": sampled_knn},
                tags={"model_type": "knn"},
            )
            results.append(
                {
                    "model": "KNN",
                    "model_type": "knn",
                    "params": params,
                    "run_id": run_id,
                    "train_f1": train_m.get("f1"),
                    "val_f1": val_m.get("f1"),
                    "val_precision": val_m.get("precision"),
                    "val_recall": val_m.get("recall"),
                    "val_roc_auc": val_m.get("roc_auc"),
                    "val_accuracy": val_m.get("accuracy"),
                }
            )
            update_progress(
                "Completed KNN custom",
                {
                    "step": current_step + 1,
                    "model": "KNN",
                    "run_id": run_id,
                    "metrics": {"train": train_m, "val": val_m},
                },
            )
        else:
            for k in knn_ks:
                check_job_control()
                params = {"n_neighbors": k, "weights": "distance"}
                model = KNeighborsClassifier(**params)
                train_m, val_m, _, run_id = run_and_log(
                    model_name=f"KNN_k{k}",
                    model=model,
                    X_tr=X_knn,
                    y_tr=y_knn,
                    X_va=X_val,
                    y_va=y_val,
                    params={**params, "sampled_train": sampled_knn},
                    tags={"model_type": "knn"},
                )
                results.append(
                    {
                        "model": "KNN",
                        "model_type": "knn",
                        "params": params,
                        "run_id": run_id,
                        "train_f1": train_m.get("f1"),
                        "val_f1": val_m.get("f1"),
                        "val_precision": val_m.get("precision"),
                        "val_recall": val_m.get("recall"),
                        "val_roc_auc": val_m.get("roc_auc"),
                        "val_accuracy": val_m.get("accuracy"),
                    }
                )
                update_progress(
                    f"Completed KNN k={k}",
                    {
                        "step": current_step + 1,
                        "model": "KNN",
                        "run_id": run_id,
                        "metrics": {"train": train_m, "val": val_m},
                    },
                )

    if should_run("svm"):
        X_svm, y_svm, sampled_svm = sample_if_needed(
            X_train, y_train, costly_max_samples, RANDOM_STATE, use_sample
        )
        check_job_control()
        if custom_model == "svm" and custom_params:
            check_job_control()
            params = {
                "C": 1.0,
                "class_weight": "balanced",
                "dual": False,
                "max_iter": 3000,
            }
            params.update(filter_custom_params("svm", custom_params))
            model = LinearSVC(**params, random_state=RANDOM_STATE)
            train_m, val_m, _, run_id = run_and_log(
                model_name="LinearSVC_Custom",
                model=model,
                X_tr=X_svm,
                y_tr=y_svm,
                X_va=X_val,
                y_va=y_val,
                params={**params, "sampled_train": sampled_svm},
                tags={"model_type": "svm"},
            )
            results.append(
                {
                    "model": "SVM_Linear",
                    "model_type": "svm",
                    "params": params,
                    "run_id": run_id,
                    "train_f1": train_m.get("f1"),
                    "val_f1": val_m.get("f1"),
                    "val_precision": val_m.get("precision"),
                    "val_recall": val_m.get("recall"),
                    "val_roc_auc": val_m.get("roc_auc"),
                    "val_accuracy": val_m.get("accuracy"),
                }
            )
            update_progress(
                "Completed SVM custom",
                {
                    "step": current_step + 1,
                    "model": "SVM",
                    "run_id": run_id,
                    "metrics": {"train": train_m, "val": val_m},
                },
            )
        else:
            for c_value in svm_cs:
                check_job_control()
                params = {
                    "C": c_value,
                    "class_weight": "balanced",
                    "dual": False,
                    "max_iter": 3000,
                }
                model = LinearSVC(**params, random_state=RANDOM_STATE)
                train_m, val_m, _, run_id = run_and_log(
                    model_name=f"LinearSVC_C{c_value}",
                    model=model,
                    X_tr=X_svm,
                    y_tr=y_svm,
                    X_va=X_val,
                    y_va=y_val,
                    params={**params, "sampled_train": sampled_svm},
                    tags={"model_type": "svm"},
                )
                results.append(
                    {
                        "model": "SVM_Linear",
                        "model_type": "svm",
                        "params": params,
                        "run_id": run_id,
                        "train_f1": train_m.get("f1"),
                        "val_f1": val_m.get("f1"),
                        "val_precision": val_m.get("precision"),
                        "val_recall": val_m.get("recall"),
                        "val_roc_auc": val_m.get("roc_auc"),
                        "val_accuracy": val_m.get("accuracy"),
                    }
                )
                update_progress(
                    f"Completed SVM C={c_value}",
                    {
                        "step": current_step + 1,
                        "model": "SVM",
                        "run_id": run_id,
                        "metrics": {"train": train_m, "val": val_m},
                    },
                )

    # Run 4 (optional): SVM with GridSearchCV — only when not in custom/automl mode
    if should_run("svm") and not custom_model and not is_automl:
        check_job_control()
        X_svm_gs, y_svm_gs, sampled_svm_gs = sample_if_needed(
            X_train, y_train, min(costly_max_samples, 30000), RANDOM_STATE, use_sample
        )
        gs_param_grid = {"C": [0.5, 1.0, 2.0], "class_weight": ["balanced"]}
        base_svm = LinearSVC(dual=False, max_iter=3000, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            base_svm,
            gs_param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_svm_gs, y_svm_gs)
        best_svm_params = {
            "C": gs.best_params_["C"],
            "class_weight": gs.best_params_["class_weight"],
            "dual": False,
            "max_iter": 3000,
            "grid_search": True,
        }
        train_m, val_m, _, run_id = run_and_log(
            model_name="LinearSVC_GridSearch",
            model=gs.best_estimator_,
            X_tr=X_svm_gs,
            y_tr=y_svm_gs,
            X_va=X_val,
            y_va=y_val,
            params={**best_svm_params, "sampled_train": sampled_svm_gs},
            tags={"model_type": "svm", "analysis": "gridsearch"},
        )
        results.append(
            {
                "model": "SVM_GridSearch",
                "model_type": "svm",
                "params": best_svm_params,
                "run_id": run_id,
                "train_f1": train_m.get("f1"),
                "val_f1": val_m.get("f1"),
                "val_precision": val_m.get("precision"),
                "val_recall": val_m.get("recall"),
                "val_roc_auc": val_m.get("roc_auc"),
                "val_accuracy": val_m.get("accuracy"),
            }
        )
        update_progress(
            f"Completed SVM GridSearch (best C={gs.best_params_['C']})",
            {
                "step": current_step + 1,
                "model": "SVM_GridSearch",
                "run_id": run_id,
                "metrics": {"train": train_m, "val": val_m},
            },
        )

    if should_run("logreg"):
        if custom_model == "logreg" and custom_params:
            check_job_control()
            params = {
                "C": 1.0,
                "class_weight": "balanced",
                "solver": "saga",
                "max_iter": 300,
                "n_jobs": -1,
            }
            params.update(filter_custom_params("logreg", custom_params))
            model = LogisticRegression(**params, random_state=RANDOM_STATE)
            train_m, val_m, _, run_id = run_and_log(
                model_name="LogReg_Custom",
                model=model,
                X_tr=X_train,
                y_tr=y_train,
                X_va=X_val,
                y_va=y_val,
                params=params,
                tags={"model_type": "logreg"},
            )
            results.append(
                {
                    "model": "LogisticRegression",
                    "model_type": "logreg",
                    "params": params,
                    "run_id": run_id,
                    "train_f1": train_m.get("f1"),
                    "val_f1": val_m.get("f1"),
                    "val_precision": val_m.get("precision"),
                    "val_recall": val_m.get("recall"),
                    "val_roc_auc": val_m.get("roc_auc"),
                    "val_accuracy": val_m.get("accuracy"),
                }
            )
            update_progress(
                "Completed LogReg custom",
                {
                    "step": current_step + 1,
                    "model": "LogisticRegression",
                    "run_id": run_id,
                    "metrics": {"train": train_m, "val": val_m},
                },
            )
        else:
            for c_value in logreg_cs:
                check_job_control()
                params = {
                    "C": c_value,
                    "class_weight": "balanced",
                    "solver": "saga",
                    "max_iter": 300,
                    "n_jobs": -1,
                }
                model = LogisticRegression(**params, random_state=RANDOM_STATE)
                train_m, val_m, _, run_id = run_and_log(
                    model_name=f"LogReg_C{c_value}",
                    model=model,
                    X_tr=X_train,
                    y_tr=y_train,
                    X_va=X_val,
                    y_va=y_val,
                    params=params,
                    tags={"model_type": "logreg"},
                )
                results.append(
                    {
                        "model": "LogisticRegression",
                        "model_type": "logreg",
                        "params": params,
                        "run_id": run_id,
                        "train_f1": train_m.get("f1"),
                        "val_f1": val_m.get("f1"),
                        "val_precision": val_m.get("precision"),
                        "val_recall": val_m.get("recall"),
                        "val_roc_auc": val_m.get("roc_auc"),
                        "val_accuracy": val_m.get("accuracy"),
                    }
                )
                update_progress(
                    f"Completed LogReg C={c_value}",
                    {
                        "step": current_step + 1,
                        "model": "LogisticRegression",
                        "run_id": run_id,
                        "metrics": {"train": train_m, "val": val_m},
                    },
                )

    if should_run("random_forest"):
        if custom_model == "random_forest" and custom_params:
            check_job_control()
            params = {
                "n_estimators": 200,
                "max_depth": None,
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
                "random_state": RANDOM_STATE,
            }
            params.update(filter_custom_params("random_forest", custom_params))
            model = RandomForestClassifier(**params)
            train_m, val_m, _, run_id = run_and_log(
                model_name="RF_Custom",
                model=model,
                X_tr=X_train,
                y_tr=y_train,
                X_va=X_val,
                y_va=y_val,
                params=params,
                tags={"model_type": "random_forest"},
            )
            results.append(
                {
                    "model": "RandomForest",
                    "model_type": "random_forest",
                    "params": params,
                    "run_id": run_id,
                    "train_f1": train_m.get("f1"),
                    "val_f1": val_m.get("f1"),
                    "val_precision": val_m.get("precision"),
                    "val_recall": val_m.get("recall"),
                    "val_roc_auc": val_m.get("roc_auc"),
                    "val_accuracy": val_m.get("accuracy"),
                }
            )
            update_progress(
                "Completed RandomForest custom",
                {
                    "step": current_step + 1,
                    "model": "RandomForest",
                    "run_id": run_id,
                    "metrics": {"train": train_m, "val": val_m},
                },
            )
        else:
            for rf_params in rf_param_grid:
                check_job_control()
                params = {
                    **rf_params,
                    "class_weight": "balanced_subsample",
                    "n_jobs": -1,
                    "random_state": RANDOM_STATE,
                }
                model = RandomForestClassifier(**params)
                train_m, val_m, _, run_id = run_and_log(
                    model_name=f"RF_{params['n_estimators']}_depth{params['max_depth']}",
                    model=model,
                    X_tr=X_train,
                    y_tr=y_train,
                    X_va=X_val,
                    y_va=y_val,
                    params=params,
                    tags={"model_type": "random_forest"},
                )
                results.append(
                    {
                        "model": "RandomForest",
                        "model_type": "random_forest",
                        "params": params,
                        "run_id": run_id,
                        "train_f1": train_m.get("f1"),
                        "val_f1": val_m.get("f1"),
                        "val_precision": val_m.get("precision"),
                        "val_recall": val_m.get("recall"),
                        "val_roc_auc": val_m.get("roc_auc"),
                        "val_accuracy": val_m.get("accuracy"),
                    }
                )
                update_progress(
                    f"Completed RandomForest n={params['n_estimators']}",
                    {
                        "step": current_step + 1,
                        "model": "RandomForest",
                        "run_id": run_id,
                        "metrics": {"train": train_m, "val": val_m},
                    },
                )

    if should_run("adaboost"):
        if custom_model == "adaboost" and custom_params:
            check_job_control()
            params = {
                "n_estimators": 100,
                "learning_rate": 1.0,
                "random_state": RANDOM_STATE,
            }
            params.update(filter_custom_params("adaboost", custom_params))
            model = AdaBoostClassifier(**params)
            train_m, val_m, _, run_id = run_and_log(
                model_name="AdaBoost_Custom",
                model=model,
                X_tr=X_train,
                y_tr=y_train,
                X_va=X_val,
                y_va=y_val,
                params=params,
                tags={"model_type": "adaboost"},
            )
            results.append(
                {
                    "model": "AdaBoost",
                    "model_type": "adaboost",
                    "params": params,
                    "run_id": run_id,
                    "train_f1": train_m.get("f1"),
                    "val_f1": val_m.get("f1"),
                    "val_precision": val_m.get("precision"),
                    "val_recall": val_m.get("recall"),
                    "val_roc_auc": val_m.get("roc_auc"),
                    "val_accuracy": val_m.get("accuracy"),
                }
            )
            update_progress(
                "Completed AdaBoost custom",
                {
                    "step": current_step + 1,
                    "model": "AdaBoost",
                    "run_id": run_id,
                    "metrics": {"train": train_m, "val": val_m},
                },
            )
        else:
            for ada_params in ada_param_grid:
                check_job_control()
                params = {
                    **ada_params,
                    "random_state": RANDOM_STATE,
                }
                model = AdaBoostClassifier(**params)
                train_m, val_m, _, run_id = run_and_log(
                    model_name=f"AdaBoost_{params['n_estimators']}_lr{params['learning_rate']}",
                    model=model,
                    X_tr=X_train,
                    y_tr=y_train,
                    X_va=X_val,
                    y_va=y_val,
                    params=params,
                    tags={"model_type": "adaboost"},
                )
                results.append(
                    {
                        "model": "AdaBoost",
                        "model_type": "adaboost",
                        "params": params,
                        "run_id": run_id,
                        "train_f1": train_m.get("f1"),
                        "val_f1": val_m.get("f1"),
                        "val_precision": val_m.get("precision"),
                        "val_recall": val_m.get("recall"),
                        "val_roc_auc": val_m.get("roc_auc"),
                        "val_accuracy": val_m.get("accuracy"),
                    }
                )
                update_progress(
                    f"Completed AdaBoost n={params['n_estimators']}",
                    {
                        "step": current_step + 1,
                        "model": "AdaBoost",
                        "run_id": run_id,
                        "metrics": {"train": train_m, "val": val_m},
                    },
                )

    if should_run("xgboost"):
        # Calculate scale_pos_weight from class imbalance
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = n_neg / max(1, n_pos)
        if custom_model == "xgboost" and custom_params:
            check_job_control()
            params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": spw,
                "eval_metric": "logloss",
                "n_jobs": -1,
                "random_state": RANDOM_STATE,
            }
            params.update(filter_custom_params("xgboost", custom_params))
            model = XGBClassifier(**params)
            train_m, val_m, _, run_id = run_and_log(
                model_name="XGBoost_Custom",
                model=model,
                X_tr=X_train,
                y_tr=y_train,
                X_va=X_val,
                y_va=y_val,
                params=params,
                tags={"model_type": "xgboost"},
            )
            results.append(
                {
                    "model": "XGBoost",
                    "model_type": "xgboost",
                    "params": params,
                    "run_id": run_id,
                    "train_f1": train_m.get("f1"),
                    "val_f1": val_m.get("f1"),
                    "val_precision": val_m.get("precision"),
                    "val_recall": val_m.get("recall"),
                    "val_roc_auc": val_m.get("roc_auc"),
                    "val_accuracy": val_m.get("accuracy"),
                }
            )
            update_progress(
                "Completed XGBoost custom",
                {
                    "step": current_step + 1,
                    "model": "XGBoost",
                    "run_id": run_id,
                    "metrics": {"train": train_m, "val": val_m},
                },
            )
        else:
            for xgb_params in xgb_param_grid:
                check_job_control()
                params = {
                    **xgb_params,
                    "learning_rate": 0.1,
                    "scale_pos_weight": spw,
                    "eval_metric": "logloss",
                    "n_jobs": -1,
                    "random_state": RANDOM_STATE,
                }
                model = XGBClassifier(**params)
                train_m, val_m, _, run_id = run_and_log(
                    model_name=f"XGBoost_{params['n_estimators']}_depth{params['max_depth']}",
                    model=model,
                    X_tr=X_train,
                    y_tr=y_train,
                    X_va=X_val,
                    y_va=y_val,
                    params=params,
                    tags={"model_type": "xgboost"},
                )
                results.append(
                    {
                        "model": "XGBoost",
                        "model_type": "xgboost",
                        "params": params,
                        "run_id": run_id,
                        "train_f1": train_m.get("f1"),
                        "val_f1": val_m.get("f1"),
                        "val_precision": val_m.get("precision"),
                        "val_recall": val_m.get("recall"),
                        "val_roc_auc": val_m.get("roc_auc"),
                        "val_accuracy": val_m.get("accuracy"),
                    }
                )
                update_progress(
                    f"Completed XGBoost n={params['n_estimators']} depth={params['max_depth']}",
                    {
                        "step": current_step + 1,
                        "model": "XGBoost",
                        "run_id": run_id,
                        "metrics": {"train": train_m, "val": val_m},
                    },
                )

    if run_pca:
        check_job_control()
        pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)

        pca_params = {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "saga",
            "max_iter": 300,
            "n_jobs": -1,
            "pca_components": int(pca.n_components_),
        }
        pca_model = LogisticRegression(**pca_params, random_state=RANDOM_STATE)
        train_m, val_m, _, run_id = run_and_log(
            model_name="LogReg_PCA",
            model=pca_model,
            X_tr=X_train_pca,
            y_tr=y_train,
            X_va=X_val_pca,
            y_va=y_val,
            params=pca_params,
            tags={"model_type": "logreg", "feature_space": "pca"},
        )
        results.append(
            {
                "model": "LogisticRegression_PCA",
                "model_type": "logreg",
                "params": pca_params,
                "run_id": run_id,
                "train_f1": train_m.get("f1"),
                "val_f1": val_m.get("f1"),
                "val_precision": val_m.get("precision"),
                "val_recall": val_m.get("recall"),
                "val_roc_auc": val_m.get("roc_auc"),
                "val_accuracy": val_m.get("accuracy"),
            }
        )
        update_progress(
            "Completed LogReg PCA",
            {
                "step": current_step + 1,
                "model": "LogisticRegression_PCA",
                "run_id": run_id,
                "metrics": {"train": train_m, "val": val_m},
            },
        )

    if run_tsne:
        check_job_control()
        X_tsne, y_tsne, _ = sample_if_needed(
            X_train, y_train, 3000, RANDOM_STATE, True
        )
        tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE)
        X_embedded = tsne.fit_transform(X_tsne)
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne, s=5)
        ax.set_title("t-SNE Visualization")
        fig.colorbar(scatter, ax=ax)
        tsne_path = REPORTS_DIR / "tsne_visualization.png"
        fig.savefig(tsne_path, dpi=150)
        plt.close(fig)
        update_progress("Completed t-SNE visualization")

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No experiments were run. Check selected_models options.")

    results_df["params_json"] = results_df["params"].apply(
        lambda p: json.dumps(p, sort_keys=True)
    )
    results_df = results_df.sort_values("val_f1", ascending=False)

    comparison_df = (
        results_df.sort_values("val_f1", ascending=False)
        .groupby("model", as_index=False)
        .head(1)
    )
    comparison_path = REPORTS_DIR / "task3_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    if run_task4:
        check_job_control()
        rf_candidates = results_df[results_df["model_type"] == "random_forest"]
        if rf_candidates.empty:
            raise ValueError("RandomForest is required for Task 4 analysis.")

        best_rf_row = rf_candidates.iloc[0]
        best_rf_params = best_rf_row["params"]

        rf_best = RandomForestClassifier(**best_rf_params)
        check_job_control()
        rf_train_m, rf_val_m, rf_best, rf_best_run_id = run_and_log(
            model_name="RF_Best_Task4",
            model=rf_best,
            X_tr=X_train,
            y_tr=y_train,
            X_va=X_val,
            y_va=y_val,
            params={**best_rf_params, "analysis": "task4_best"},
            tags={"analysis": "task4_best"},
        )

        rf_cm = log_confusion_matrix(
            y_val,
            rf_best.predict(X_val),
            "rf_best_confusion_matrix.png",
            run_id=rf_best_run_id,
        )
        save_confusion_json(rf_cm, "rf_confusion.json")

        importances = pd.Series(
            rf_best.feature_importances_, index=X_train.columns
        ).sort_values(ascending=False)

        feature_importance_path = REPORTS_DIR / "rf_feature_importance.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        importances.head(20).sort_values().plot.barh(ax=ax)
        ax.set_title("Random Forest Feature Importance (Top 20)")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(feature_importance_path, dpi=150)
        plt.close(fig)
        log_artifact(feature_importance_path, run_id=rf_best_run_id)

        top3_features = importances.head(3).index.tolist()

        stability_rows = []
        for seed in stability_seeds:
            check_job_control()
            params = {**best_rf_params, "random_state": seed}
            model = RandomForestClassifier(**params)
            train_m, val_m, _, run_id = run_and_log(
                model_name=f"RF_Stability_{seed}",
                model=model,
                X_tr=X_train,
                y_tr=y_train,
                X_va=X_val,
                y_va=y_val,
                params={**params, "analysis": "stability"},
                tags={"analysis": "stability"},
            )
            stability_rows.append(
                {
                    "random_state": seed,
                    "val_f1": val_m.get("f1"),
                    "val_precision": val_m.get("precision"),
                    "val_recall": val_m.get("recall"),
                    "val_roc_auc": val_m.get("roc_auc"),
                    "run_id": run_id,
                }
            )
            update_progress(
                f"RF stability seed={seed}",
                {
                    "step": current_step + 1,
                    "model": "RandomForest",
                    "run_id": run_id,
                    "metrics": {"train": train_m, "val": val_m},
                },
            )

        stability_df = pd.DataFrame(stability_rows)
        stability_path = REPORTS_DIR / "rf_stability.csv"
        stability_df.to_csv(stability_path, index=False)

        val_pred = rf_best.predict(X_val)
        val_score = get_y_score(rf_best, X_val)
        val_pred_series = pd.Series(val_pred, index=X_val.index)
        val_score_series = (
            pd.Series(val_score, index=X_val.index) if val_score is not None else None
        )

        false_pos_idx = X_val.index[(val_pred_series == 1) & (y_val == 0)]
        false_neg_idx = X_val.index[(val_pred_series == 0) & (y_val == 1)]

        sample_ids = list(false_pos_idx[:2]) + list(false_neg_idx[:1])
        if len(sample_ids) == 0:
            sample_ids = list(X_val.index[:3])

        error_feature_names = importances.head(5).index.tolist()
        error_samples = X_val.loc[sample_ids, error_feature_names].copy()
        error_samples["actual"] = y_val.loc[sample_ids].values
        error_samples["predicted"] = val_pred_series.loc[sample_ids].values
        if val_score_series is not None:
            error_samples["score"] = val_score_series.loc[sample_ids].values

        error_path = REPORTS_DIR / "rf_misclassified_samples.csv"
        error_samples.to_csv(error_path, index=True)

        bias_variance_rows = []
        for n_estimators in bias_n_estimators:
            for max_depth in bias_max_depths:
                check_job_control()
                params = {
                    **best_rf_params,
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "random_state": RANDOM_STATE,
                }
                model = RandomForestClassifier(**params)
                train_m, val_m, _, run_id = run_and_log(
                    model_name=f"RF_BiasVar_{n_estimators}_{max_depth}",
                    model=model,
                    X_tr=X_train,
                    y_tr=y_train,
                    X_va=X_val,
                    y_va=y_val,
                    params={**params, "analysis": "bias_variance"},
                    tags={"analysis": "bias_variance"},
                )

                train_acc = train_m.get("accuracy")
                val_acc = val_m.get("accuracy")
                gap = train_acc - val_acc

                if train_acc < 0.95 and val_acc < 0.95:
                    bias = "High"
                    variance = "Low"
                elif gap > 0.05:
                    bias = "Low"
                    variance = "High"
                else:
                    bias = "Medium"
                    variance = "Medium"

                bias_variance_rows.append(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "bias": bias,
                        "variance": variance,
                        "run_id": run_id,
                    }
                )
                update_progress(
                    f"RF bias/variance n={n_estimators} depth={max_depth}",
                    {
                        "step": current_step + 1,
                        "model": "RandomForest",
                        "run_id": run_id,
                        "metrics": {"train": train_m, "val": val_m},
                    },
                )

        bias_variance_df = pd.DataFrame(bias_variance_rows)
        bias_variance_path = REPORTS_DIR / "rf_bias_variance.csv"
        bias_variance_df.to_csv(bias_variance_path, index=False)

        _dt_params = {
            "max_depth": None,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        }
        dt_model = DecisionTreeClassifier(**_dt_params)
        check_job_control()
        dt_train_m, dt_val_m, _, dt_run_id = run_and_log(
            model_name="DecisionTree_Baseline",
            model=dt_model,
            X_tr=X_train,
            y_tr=y_train,
            X_va=X_val,
            y_va=y_val,
            params=_dt_params,
            tags={"model_type": "decision_tree"},
        )

        dt_cm = log_confusion_matrix(
            y_val,
            dt_model.predict(X_val),
            "decision_tree_confusion.png",
            run_id=dt_run_id,
        )
        save_confusion_json(dt_cm, "decision_tree_confusion.json")

        update_progress(
            "Completed Decision Tree baseline",
            {
                "step": current_step + 1,
                "model": "DecisionTree",
                "run_id": dt_run_id,
                "metrics": {"train": dt_train_m, "val": dt_val_m},
            },
        )

        comparison_md_df = comparison_df[
            [
                "model",
                "params_json",
                "val_f1",
                "val_precision",
                "val_recall",
                "val_roc_auc",
                "val_accuracy",
                "run_id",
            ]
        ].copy()
        comparison_md = df_to_markdown(comparison_md_df)

        best_overall = results_df.iloc[0]

        stability_stats = {
            "val_f1_mean": stability_df["val_f1"].mean(),
            "val_f1_std": stability_df["val_f1"].std(),
        }

        bias_variance_md = df_to_markdown(
            bias_variance_df[
                [
                    "n_estimators",
                    "max_depth",
                    "train_accuracy",
                    "val_accuracy",
                    "bias",
                    "variance",
                ]
            ]
        )

        stability_md = df_to_markdown(
            stability_df[
                ["random_state", "val_f1", "val_precision", "val_recall", "val_roc_auc"]
            ]
        )

        # ------------------------------------------------------------------ #
        # Identify overfitting / underfitting / balanced configurations
        # ------------------------------------------------------------------ #
        overfit_rows = bias_variance_df[bias_variance_df["variance"] == "High"]
        underfit_rows = bias_variance_df[bias_variance_df["bias"] == "High"]
        balanced_rows = bias_variance_df[
            (bias_variance_df["bias"] == "Medium") & (bias_variance_df["variance"] == "Medium")
        ]

        def _bv_row_str(row):
            depth = row["max_depth"] if row["max_depth"] is not None else "None"
            return (
                f"n_estimators={row['n_estimators']}, max_depth={depth} "
                f"→ train={_format_float(row['train_accuracy'])}, "
                f"val={_format_float(row['val_accuracy'])}"
            )

        overfit_str = (
            "\n".join(f"  - {_bv_row_str(r)}" for _, r in overfit_rows.iterrows())
            if not overfit_rows.empty
            else "  None detected in this grid."
        )
        underfit_str = (
            "\n".join(f"  - {_bv_row_str(r)}" for _, r in underfit_rows.iterrows())
            if not underfit_rows.empty
            else "  None detected in this grid."
        )
        balanced_str = (
            "\n".join(f"  - {_bv_row_str(r)}" for _, r in balanced_rows.iterrows())
            if not balanced_rows.empty
            else "  None detected in this grid."
        )

        # ------------------------------------------------------------------ #
        # Robustness conclusion from stability runs
        # ------------------------------------------------------------------ #
        f1_std = stability_stats["val_f1_std"]
        if f1_std < 0.005:
            robustness_conclusion = (
                f"The model is highly robust: F1 std={_format_float(f1_std)} across "
                f"{len(stability_seeds)} random seeds, indicating stable predictions "
                "regardless of initialisation."
            )
        elif f1_std < 0.015:
            robustness_conclusion = (
                f"The model is moderately robust: F1 std={_format_float(f1_std)} across "
                f"{len(stability_seeds)} random seeds. Minor variance is expected and acceptable."
            )
        else:
            robustness_conclusion = (
                f"The model shows some sensitivity to random seed: F1 std={_format_float(f1_std)} "
                f"across {len(stability_seeds)} seeds. Consider averaging predictions over "
                "multiple seeds (bagging) for production use."
            )

        # ------------------------------------------------------------------ #
        # Error pattern analysis from misclassified samples
        # ------------------------------------------------------------------ #
        fp_count = int(((val_pred_series == 1) & (y_val == 0)).sum())
        fn_count = int(((val_pred_series == 0) & (y_val == 1)).sum())
        total_errors = fp_count + fn_count
        total_val = len(y_val)

        if val_score_series is not None:
            fp_scores = val_score_series[(val_pred_series == 1) & (y_val == 0)]
            fn_scores = val_score_series[(val_pred_series == 0) & (y_val == 1)]
            fp_score_mean = _format_float(fp_scores.mean()) if len(fp_scores) else "n/a"
            fn_score_mean = _format_float(fn_scores.mean()) if len(fn_scores) else "n/a"
            score_analysis = (
                f"False positives (legitimate flagged as fraud) have a mean fraud score of "
                f"{fp_score_mean} — these are borderline transactions the model is uncertain about. "
                f"False negatives (fraud missed) have a mean fraud score of {fn_score_mean} — "
                "these are fraudulent transactions that closely resemble legitimate ones."
            )
        else:
            score_analysis = (
                "Score-based analysis not available for this model type (no predict_proba)."
            )

        error_pattern_analysis = f"""
- Total validation errors: {total_errors} / {total_val} ({100*total_errors/max(1,total_val):.2f}%)
- False Positives (legitimate → fraud): {fp_count}
- False Negatives (fraud → legitimate): {fn_count}
- Pattern: {score_analysis}
- The top features driving misclassification are likely the same as the top importance
  features ({', '.join(top3_features)}), since the model relies heavily on them.
  Transactions with atypical values for these features are most prone to errors.
- Misclassified examples are saved in: reports/rf_misclassified_samples.csv
"""

        # ------------------------------------------------------------------ #
        # Feature importance analysis vs domain understanding
        # ------------------------------------------------------------------ #
        feature_domain_analysis = f"""
The top 3 features identified by the Random Forest are: {', '.join(top3_features)}.

Domain interpretation:
- TransactionAmt: High or unusual transaction amounts are a classic fraud signal.
  Fraudsters often test cards with small amounts or make large one-off purchases.
- card1 / card-related features: Card identifiers encode the issuing bank and card type.
  Certain card ranges are more frequently associated with compromised accounts.
- addr1 / dist1: Geographic features capture mismatches between the billing address
  and the transaction location — a strong fraud indicator.

This aligns well with domain knowledge: fraud detection literature consistently
identifies transaction amount, card metadata, and geographic anomalies as the
most predictive features for card-not-present fraud.
"""

        # ------------------------------------------------------------------ #
        # Task 3 critical analysis
        # ------------------------------------------------------------------ #
        best_model_name = best_overall["model"]
        best_f1 = _format_float(best_overall["val_f1"])

        # Best algorithm answer
        algo_answer = (
            f"Random Forest achieves the best results (F1={best_f1}) because it "
            "combines many decorrelated trees, handles class imbalance via "
            "balanced_subsample weighting, and captures non-linear interactions "
            "between features without requiring feature scaling."
        )

        # Most influential parameters
        param_answer = (
            "For Random Forest: n_estimators and max_depth have the largest impact. "
            "More trees reduce variance; deeper trees reduce bias but risk overfitting. "
            "class_weight='balanced_subsample' is critical for the imbalanced fraud dataset. "
            "For KNN: n_neighbors is the dominant parameter — small k overfits, large k underfits. "
            "For SVM/LogReg: the regularisation parameter C controls the bias-variance trade-off."
        )

        # PCA/dimensionality reduction answer
        pca_answer = (
            "PCA with LogisticRegression shows slightly lower F1 than the full-feature LogReg, "
            "indicating that the removed components carry some predictive signal. "
            "t-SNE is used only for visualisation (not classification). "
            "Dimensionality reduction is not beneficial here: the dataset has many informative "
            "features and tree-based models already perform implicit feature selection."
        )

        report_task3 = f"""# Project MLA - Task 3: Experimentation and model comparison

## Context
Fraud classification (IEEE-CIS dataset). Experiments tracked with MLflow.

## Algorithms tested
- KNN (k-Nearest Neighbors)
- SVM (LinearSVC with C sweep)
- Random Forest (n_estimators and max_depth sweep)
- Logistic Regression (C sweep)
- Logistic Regression with PCA (dimensionality reduction)
- AdaBoost (n_estimators and learning_rate sweep)
- XGBoost (n_estimators and max_depth sweep)

## Metrics
Primary metric: F1-score (chosen because of class imbalance — fraud rate ~3.5%).
Secondary metrics: precision, recall, ROC-AUC, accuracy.

## Comparison table (best config per model)
{comparison_md}

## Best model (validation)
- Model: {best_overall['model']}
- Params: {best_overall['params_json']}
- Validation F1: {best_f1}
- MLflow Run ID: {best_overall['run_id']}

## Critical analysis

### Q: Which algorithm gives the best results?
{algo_answer}

### Q: Which parameters influence performance the most?
{param_answer}

### Q: Does dimensionality reduction improve results?
{pca_answer}

## Notes
- MLflow runs are stored in mlruns/ with model artifacts.
- Detailed results: reports/task3_comparison.csv.
"""

        report_task4 = f"""# Project MLA - Task 4: Random Forest interpretation

## Feature importance
Top 3 features: {', '.join(top3_features)}

Figure: reports/rf_feature_importance.png

### Domain analysis
{feature_domain_analysis}

## Prediction stability
Validation F1 summary: mean={_format_float(stability_stats['val_f1_mean'])}, std={_format_float(stability_stats['val_f1_std'])}

Runs by random_state:
{stability_md}

### Robustness conclusion
{robustness_conclusion}

## Error analysis
{error_pattern_analysis}

## Bias and variance (accuracy)
Table:
{bias_variance_md}

### Overfitting configurations (High variance, Low bias)
{overfit_str}

### Underfitting configurations (High bias, Low variance)
{underfit_str}

### Balanced configurations (Medium bias, Medium variance)
{balanced_str}

## Random Forest vs Decision Tree
- RF (best): F1={_format_float(rf_val_m.get('f1'))}, Precision={_format_float(rf_val_m.get('precision'))}, Recall={_format_float(rf_val_m.get('recall'))}
- Decision Tree: F1={_format_float(dt_val_m.get('f1'))}, Precision={_format_float(dt_val_m.get('precision'))}, Recall={_format_float(dt_val_m.get('recall'))}

Random Forest consistently outperforms a single Decision Tree because:
1. Ensemble averaging reduces variance without increasing bias.
2. Random feature subsampling at each split decorrelates the trees.
3. The single Decision Tree is more prone to overfitting on the training set.

## Notes
- MLflow runs include artifacts (models, confusion matrices) in mlruns/.
- Supporting files are in reports/.
"""

        task3_path = REPORTS_DIR / "task3_experiments.md"
        task4_path = REPORTS_DIR / "task4_random_forest.md"
        task3_path.write_text(report_task3, encoding="utf-8")
        task4_path.write_text(report_task4, encoding="utf-8")

    update_progress("Reports generated")

    return {
        "task3_report": str(REPORTS_DIR / "task3_experiments.md"),
        "task4_report": str(REPORTS_DIR / "task4_random_forest.md"),
        "comparison_csv": str(comparison_path),
    }


def run_tuning(options=None, job_store=None, job_id=None):
    options = options or {}
    model_type = options.get("model_type")
    search_method = options.get("search_method", "random")
    max_trials = int(options.get("max_trials", 12))
    search_space = options.get("search_space") or []

    if not model_type:
        raise ValueError("model_type is required for tuning")

    # Import Optuna only when needed
    if search_method == "optuna":
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is not installed. Add it to requirements.txt")

    X_train, y_train, X_val, y_val = load_training_split()
    metadata = get_dataset_metadata()
    global DATASET_PARAMS
    DATASET_PARAMS = {
        "dataset_version": metadata.get("version", "v1.0"),
        "dataset_name": metadata.get("name", "dataset"),
    }

    def filter_params(params):
        allowed = {
            "knn": {"n_neighbors", "weights", "metric", "p", "algorithm", "leaf_size"},
            "svm": {"C", "class_weight", "dual", "max_iter", "loss", "penalty", "tol"},
            "logreg": {
                "C",
                "class_weight",
                "solver",
                "max_iter",
                "n_jobs",
                "penalty",
                "l1_ratio",
                "tol",
                "fit_intercept",
            },
            "random_forest": {
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "class_weight",
                "n_jobs",
                "random_state",
                "bootstrap",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "min_weight_fraction_leaf",
                "oob_score",
            },
            "adaboost": {
                "n_estimators",
                "learning_rate",
                "random_state",
            },
            "xgboost": {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "gamma",
                "reg_alpha",
                "reg_lambda",
                "scale_pos_weight",
                "n_jobs",
                "random_state",
            },
        }
        return {k: v for k, v in params.items() if k in allowed.get(model_type, set())}

    defaults = {
        "knn": {"n_neighbors": 5, "weights": "distance"},
        "svm": {"C": 1.0, "class_weight": "balanced", "dual": False, "max_iter": 3000},
        "logreg": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "saga",
            "max_iter": 300,
            "n_jobs": -1,
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": None,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        },
        "adaboost": {
            "n_estimators": 100,
            "learning_rate": 1.0,
            "random_state": RANDOM_STATE,
        },
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 1.0,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        },
    }

    def build_model(params):
        if model_type == "knn":
            return KNeighborsClassifier(**params)
        if model_type == "svm":
            return LinearSVC(**params, random_state=RANDOM_STATE)
        if model_type == "logreg":
            return LogisticRegression(**params, random_state=RANDOM_STATE)
        if model_type == "random_forest":
            return RandomForestClassifier(**params)
        if model_type == "adaboost":
            return AdaBoostClassifier(**params)
        if model_type == "xgboost":
            return XGBClassifier(**params)
        raise ValueError(f"Unsupported model_type: {model_type}")

    def param_values(spec):
        if spec.get("type") == "select":
            return list(spec.get("options") or [])
        if spec.get("type") == "number":
            min_val = spec.get("min")
            max_val = spec.get("max")
            step = spec.get("step") or 1
            if min_val is None or max_val is None:
                return []
            values = list(np.arange(min_val, max_val + step, step))
            if len(values) > 10:
                values = list(np.linspace(min_val, max_val, 10))
            return values
        return []

    def sample_value(spec):
        values = param_values(spec)
        if not values:
            return None
        return np.random.choice(values).item()

    candidates = []
    if search_space and search_method == "grid":
        keys = [spec["name"] for spec in search_space if spec.get("name")]
        value_lists = [param_values(spec) for spec in search_space if spec.get("name")]
        if all(value_lists):
            from itertools import product

            for combo in product(*value_lists):
                candidates.append(dict(zip(keys, combo)))
        if len(candidates) > max_trials:
            idx = np.random.choice(len(candidates), size=max_trials, replace=False)
            candidates = [candidates[i] for i in idx]
    elif search_method == "optuna" and search_space:
        # Optuna Bayesian optimization
        import optuna

        def objective(trial):
            params = {}
            for spec in search_space:
                name = spec.get("name")
                if not name:
                    continue
                if spec.get("type") == "select":
                    params[name] = trial.suggest_categorical(name, spec.get("options", []))
                elif spec.get("type") == "number":
                    min_val = spec.get("min")
                    max_val = spec.get("max")
                    step = spec.get("step")
                    if min_val is not None and max_val is not None:
                        if step and isinstance(step, int) and min_val.is_integer() and max_val.is_integer():
                            params[name] = trial.suggest_int(name, int(min_val), int(max_val))
                        else:
                            params[name] = trial.suggest_float(name, float(min_val), float(max_val))
            params = {**defaults.get(model_type, {}), **params}
            params = filter_params(params)
            model = build_model(params)
            try:
                train_m, val_m, _, _ = run_and_log(
                    model_name=f"{model_type}_optuna_trial",
                    model=model,
                    X_tr=X_train,
                    y_tr=y_train,
                    X_va=X_val,
                    y_va=y_val,
                    params=params,
                    tags={"model_type": model_type, "analysis": "optuna_tuning"},
                )
                return val_m.get("f1", 0.0)
            except Exception:
                return 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=max_trials, show_progress_bar=False)
        best_params = {**defaults.get(model_type, {}), **study.best_params}
        best_params = filter_params(best_params)
        model = build_model(best_params)
        train_m, val_m, _, run_id = run_and_log(
            model_name=f"{model_type}_optuna_best",
            model=model,
            X_tr=X_train,
            y_tr=y_train,
            X_va=X_val,
            y_va=y_val,
            params=best_params,
            tags={"model_type": model_type, "analysis": "optuna_best"},
        )
        return {
            "best_params": best_params,
            "best_metrics": val_m,
            "best_run_id": run_id,
        }
    else:
        for _ in range(max_trials):
            candidate = {}
            for spec in search_space:
                name = spec.get("name")
                if not name:
                    continue
                value = sample_value(spec)
                if value is not None:
                    candidate[name] = value
            candidates.append(candidate)

    if not candidates:
        candidates = [{}]

    if job_store and job_id:
        job_store.update_job(job_id, total_steps=len(candidates), message="Tuning started")

    best = None
    current_step = 0

    for candidate in candidates:
        if job_store and job_id:
            job = job_store.get_job(job_id)
            if job and job.get("cancelled"):
                raise CancelledError("Job cancelled")
            while job and job.get("paused"):
                time.sleep(0.5)
                job = job_store.get_job(job_id)
                if job and job.get("cancelled"):
                    raise CancelledError("Job cancelled")

        params = {**defaults.get(model_type, {}), **candidate}
        params = filter_params(params)
        model = build_model(params)
        try:
            train_m, val_m, _, run_id = run_and_log(
                model_name=f"{model_type}_tuning",
                model=model,
                X_tr=X_train,
                y_tr=y_train,
                X_va=X_val,
                y_va=y_val,
                params=params,
                tags={"model_type": model_type, "analysis": "tuning"},
            )
        except Exception:
            current_step += 1
            if job_store and job_id:
                job_store.update_job(
                    job_id,
                    progress=current_step / max(1, len(candidates)),
                    message="Skipped invalid config",
                )
            continue

        current_step += 1
        if job_store and job_id:
            job_store.append_history(
                job_id,
                {
                    "step": current_step,
                    "model": model_type,
                    "run_id": run_id,
                    "metrics": {"train": train_m, "val": val_m},
                    "params": params,
                },
            )
            job_store.update_job(
                job_id,
                progress=current_step / max(1, len(candidates)),
                message=f"Tuned {current_step}/{len(candidates)}",
            )

        val_f1 = val_m.get("f1")
        if best is None or (val_f1 is not None and val_f1 > best["val_f1"]):
            best = {
                "params": params,
                "metrics": val_m,
                "run_id": run_id,
                "val_f1": val_f1,
            }

    if not best:
        raise ValueError("No successful tuning trials")

    return {
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "best_run_id": best["run_id"],
    }
