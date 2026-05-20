"""
simulate_drift.py — Data drift simulation, detection, and automated retraining trigger.

Usage (from project root):
    python -m backend.simulate_drift [--retrain]

What it does
------------
1. Loads the processed training data as the *reference* distribution.
2. Simulates a *production* dataset by shifting the mean and adding noise on
   two numeric features (TransactionAmt and card1).
3. Generates an Evidently HTML report (DataDriftPreset + DataQualityPreset)
   and logs it as an MLflow artifact under the ``monitoring_drift`` experiment.
4. Extracts numeric drift metrics and logs them to MLflow.
5. Applies a KS-test per numeric feature, logs p-values, and saves
   ``ks_drift_results.csv`` as an artifact.
6. Evaluates drift against SEUIL_DRIFT / SEUIL_WARN thresholds and
   conditionally triggers retraining.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------------
# Drift thresholds
# ---------------------------------------------------------------------------
SEUIL_DRIFT = 0.30   # Fraction of drifted features → trigger retraining
SEUIL_WARN = 0.15    # Fraction of drifted features → emit warning

# ---------------------------------------------------------------------------
# Project paths (resolved relative to this file)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Processed data paths
DATA_DIR = PROJECT_ROOT / "dataset" / "processed_data"
X_TRAIN_PATH = DATA_DIR / "X_train.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_reference() -> pd.DataFrame:
    """Load the processed training features as the reference distribution."""
    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Processed training data not found: {X_TRAIN_PATH}")
    return pd.read_csv(X_TRAIN_PATH)


def _simulate_production(reference: pd.DataFrame, n_samples: int = 5000) -> pd.DataFrame:
    """
    Create a synthetic production dataset by:
    - Sampling *n_samples* rows from *reference*.
    - Shifting the mean of ``TransactionAmt`` by +50 (if present).
    - Shifting the mean of ``card1`` by +2000 (if present).
    - Adding Gaussian noise to both shifted features.
    """
    rng = np.random.default_rng(seed=2024)
    n = min(n_samples, len(reference))
    production = reference.sample(n, random_state=42).copy().reset_index(drop=True)

    if "TransactionAmt" in production.columns:
        production["TransactionAmt"] = (
            production["TransactionAmt"]
            + 50.0
            + rng.normal(0, 20, size=n)
        )

    if "card1" in production.columns:
        production["card1"] = (
            production["card1"]
            + 2000.0
            + rng.normal(0, 500, size=n)
        )

    return production


def _verify_mean_shift(reference: pd.DataFrame, production: pd.DataFrame) -> None:
    """Print mean differences for the two shifted features."""
    for col in ["TransactionAmt", "card1"]:
        if col in reference.columns and col in production.columns:
            ref_mean = reference[col].mean()
            prod_mean = production[col].mean()
            print(f"  {col}: reference mean={ref_mean:.2f}, production mean={prod_mean:.2f}, "
                  f"delta={prod_mean - ref_mean:.2f}")


# ---------------------------------------------------------------------------
# Evidently report
# ---------------------------------------------------------------------------

def _generate_evidently_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    output_path: Path,
) -> dict:
    """
    Generate an Evidently HTML report and return extracted drift metrics.

    Returns a dict with keys: drift_share, drifted_columns, total_columns,
    dataset_drifted.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    except ImportError:
        print("  [WARN] Evidently not installed. Skipping HTML report generation.")
        print("         Install with: pip install evidently")
        return {}

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=production)
    report.save_html(str(output_path))
    print(f"  Evidently report saved: {output_path}")

    # Extract numeric metrics from the report JSON
    report_dict = report.as_dict()
    drift_metrics: dict = {}
    try:
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})
            if "dataset_drift" in result:
                drift_metrics["dataset_drifted"] = result["dataset_drift"]
            if "drift_share" in result:
                drift_metrics["drift_share"] = result["drift_share"]
            if "number_of_drifted_columns" in result:
                drift_metrics["drifted_columns"] = result["number_of_drifted_columns"]
            if "number_of_columns" in result:
                drift_metrics["total_columns"] = result["number_of_columns"]
    except Exception as exc:
        print(f"  [WARN] Could not extract Evidently metrics: {exc}")

    return drift_metrics


# ---------------------------------------------------------------------------
# KS-test
# ---------------------------------------------------------------------------

def _ks_test_per_feature(
    reference: pd.DataFrame,
    production: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply scipy.stats.ks_2samp to each numeric feature.

    Returns a DataFrame with columns: feature, ks_statistic, p_value, drifted.
    """
    from scipy.stats import ks_2samp

    rows = []
    numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in production.columns:
            continue
        ref_vals = reference[col].dropna().values
        prod_vals = production[col].dropna().values
        if len(ref_vals) == 0 or len(prod_vals) == 0:
            continue
        stat, pvalue = ks_2samp(ref_vals, prod_vals)
        rows.append({
            "feature": col,
            "ks_statistic": float(stat),
            "p_value": float(pvalue),
            "drifted": bool(pvalue < 0.05),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_drift_detection(retrain: bool = False) -> dict:
    """
    Full drift detection pipeline.

    Parameters
    ----------
    retrain : bool
        If True, force retraining regardless of drift level.

    Returns
    -------
    dict with keys: drift_share, status, retrain_triggered.
    """
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Loading reference data...")
    reference = _load_reference()
    print(f"  Reference shape: {reference.shape}")

    print("Simulating production data...")
    production = _simulate_production(reference)
    print(f"  Production shape: {production.shape}")

    print("Mean shift verification:")
    _verify_mean_shift(reference, production)

    # MLflow setup
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("monitoring_drift")

    with mlflow.start_run(run_name="drift_detection") as run:
        run_id = run.info.run_id
        print(f"\nMLflow run: {run_id}")

        # --- Pipeline diagram artifact (6.6) ---
        diagram_path = REPORTS_DIR / "mlops_pipeline_diagram.txt"
        diagram_path.write_text(PIPELINE_DIAGRAM, encoding="utf-8")
        mlflow.log_artifact(str(diagram_path))
        print(f"  Pipeline diagram logged: {diagram_path}")

        # --- Evidently report ---
        print("\nGenerating Evidently report...")
        report_path = REPORTS_DIR / f"drift_report_{run_id}.html"
        evidently_metrics = _generate_evidently_report(reference, production, report_path)

        if report_path.exists():
            mlflow.log_artifact(str(report_path))

        # Log Evidently metrics
        for key, value in evidently_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
            else:
                mlflow.log_param(key, str(value))

        drift_share: float = evidently_metrics.get("drift_share", 0.0)

        # --- KS-test ---
        print("\nRunning KS-test per feature...")
        ks_df = _ks_test_per_feature(reference, production)
        print(f"  Features tested: {len(ks_df)}, drifted: {ks_df['drifted'].sum()}")

        # Log p-values per feature
        for _, row in ks_df.iterrows():
            safe_col = str(row["feature"]).replace(" ", "_")[:50]
            mlflow.log_metric(f"ks_pvalue_{safe_col}", row["p_value"])

        # Save KS results CSV
        ks_path = REPORTS_DIR / "ks_drift_results.csv"
        ks_df.to_csv(ks_path, index=False)
        mlflow.log_artifact(str(ks_path))
        print(f"  KS results saved: {ks_path}")

        # Fallback drift_share from KS-test if Evidently not available
        if drift_share == 0.0 and len(ks_df) > 0:
            drift_share = float(ks_df["drifted"].mean())
            mlflow.log_metric("drift_share", drift_share)

        # --- Threshold logic ---
        if drift_share >= SEUIL_DRIFT or retrain:
            status = "CRITIQUE"
            retrain_triggered = True
        elif drift_share >= SEUIL_WARN:
            status = "AVERTISSEMENT"
            retrain_triggered = False
        else:
            status = "OK"
            retrain_triggered = False

        mlflow.log_param("drift_status", status)
        mlflow.log_param("seuil_drift", SEUIL_DRIFT)
        mlflow.log_param("seuil_warn", SEUIL_WARN)
        mlflow.log_metric("retrain_triggered", int(retrain_triggered))

        print(f"\nDrift status: {status}")
        print(f"  drift_share={drift_share:.3f}  SEUIL_WARN={SEUIL_WARN}  SEUIL_DRIFT={SEUIL_DRIFT}")
        print(f"  retrain_triggered={retrain_triggered}")

        # --- Trigger retraining ---
        if retrain_triggered:
            print("\nTriggering retraining via backend training module...")
            try:
                # Import and call directly when running as a module
                from backend.training import run_task3_task4
                run_task3_task4(options={"run_task4": False, "use_sample": True})
                print("  Retraining completed.")
            except Exception as exc:
                print(f"  [WARN] Retraining failed: {exc}")
                print("  You can also run: python -m backend.simulate_drift --retrain")

    return {
        "run_id": run_id,
        "drift_share": drift_share,
        "status": status,
        "retrain_triggered": retrain_triggered,
    }


# ---------------------------------------------------------------------------
# MLOps pipeline diagram (6.6)
# ---------------------------------------------------------------------------

PIPELINE_DIAGRAM = """
╔══════════════════════════════════════════════════════════════════════╗
║          MLOps Pipeline — Fraud Detection (Closed Loop)             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  ║
║   │  Données │───▶│Prétraitement │───▶│  Entraînement (MLflow)   │  ║
║   │  brutes  │    │ preprocessing│    │  KNN / SVM / LR / RF     │  ║
║   └──────────┘    └──────────────┘    └────────────┬─────────────┘  ║
║                                                    │                 ║
║                                                    ▼                 ║
║   ┌──────────────────────────────────────────────────────────────┐  ║
║   │              MLflow Model Registry                           │  ║
║   │   None ──▶ Staging (accuracy ≥ 0.85) ──▶ Production         │  ║
║   └────────────────────────────┬─────────────────────────────────┘  ║
║                                │                                     ║
║                                ▼                                     ║
║   ┌──────────────────────────────────────────────────────────────┐  ║
║   │              Serving (FastAPI /api/predict)                  │  ║
║   │         or  mlflow models serve --port 1234                  │  ║
║   └────────────────────────────┬─────────────────────────────────┘  ║
║                                │                                     ║
║                                ▼                                     ║
║   ┌──────────────────────────────────────────────────────────────┐  ║
║   │   Monitoring — simulate_drift.py                             │  ║
║   │   • Evidently DataDriftPreset + DataQualityPreset            │  ║
║   │   • KS-test par feature (scipy.stats.ks_2samp)               │  ║
║   │   • drift_share logged to MLflow experiment monitoring_drift  │  ║
║   └────────────────────────────┬─────────────────────────────────┘  ║
║                                │                                     ║
║          drift_share < 0.15    │    drift_share ≥ 0.30              ║
║          ──────────────────────┤────────────────────────────────     ║
║               OK               │         CRITIQUE                   ║
║                                ▼                                     ║
║   ┌──────────────────────────────────────────────────────────────┐  ║
║   │   Ré-entraînement automatique (run_task3_task4)              │  ║
║   │   retrain_triggered = 1  logged to MLflow                    │  ║
║   └────────────────────────────┬─────────────────────────────────┘  ║
║                                │                                     ║
║                                └──────────────────────────────────▶ ║
║                                        (retour au début)             ║
╚══════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud detection data drift monitor")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining regardless of drift level",
    )
    parser.add_argument(
        "--diagram",
        action="store_true",
        help="Print the MLOps pipeline diagram and exit",
    )
    args = parser.parse_args()

    if args.diagram:
        print(PIPELINE_DIAGRAM)
        sys.exit(0)

    print(PIPELINE_DIAGRAM)
    result = run_drift_detection(retrain=args.retrain)
    print("\nSummary:", json.dumps(result, indent=2))
