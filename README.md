# Fraud Detection — MLA Project

End-to-end machine learning pipeline for IEEE-CIS fraud detection, covering
experimentation (Task 3), Random Forest analysis (Task 4), and a full local
MLOps pipeline (Task 5).

---

## Project structure

```
fraud_detection/
├── backend/                  # Python FastAPI backend
│   ├── app.py                # REST API (FastAPI)
│   ├── training.py           # Model training + MLflow logging (Tasks 3 & 4)
│   ├── evaluate.py           # Standalone evaluation utilities
│   ├── data_loader.py        # Raw and processed data loading
│   ├── preprocessing.py      # Feature engineering pipeline
│   ├── dataset_service.py    # Dataset management (clean, upload, export)
│   ├── mlflow_service.py     # MLflow run listing and summarisation
│   ├── model_registry.py     # Active model persistence (JSON)
│   ├── registry_service.py   # MLflow Model Registry (register, promote)
│   ├── reports_service.py    # Artifact and report reading
│   ├── simulate_drift.py     # Data drift detection (Evidently + KS-test)
│   ├── serve_mlflow.py       # Native MLflow model serving helper
│   ├── check_versions.py     # Stack version verification
│   ├── jobs.py               # Background job store
│   ├── config.py             # Path configuration
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx           # Router and layout
│   │   ├── pages/            # Dashboard, Models, Results, MLOps, Predict, …
│   │   ├── components/       # Sidebar, PredictForm, Tooltip, Notifications
│   │   └── services/api.js   # All backend API calls
│   └── package.json
├── dataset/
│   ├── raw/                  # Original Kaggle CSVs (place here)
│   └── processed_data/       # Pre-processed train/val splits
├── mlruns/                   # MLflow tracking store
├── models/                   # Saved model artefacts
├── notebooks/                # Exploratory data analysis
├── reports/                  # Generated reports, charts, CSVs
├── hooks/
│   └── pre-commit            # Git pre-commit accuracy validation hook
├── requirements.txt          # Root-level Python dependencies
└── Makefile                  # Pipeline orchestration
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the backend

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

---

## Running the full pipeline

```bash
make pipeline        # setup → train (Task 3 + 4) → drift detection
make serve           # start FastAPI server
make register        # register + promote best model (server must be running)
make drift           # run drift detection standalone
make test            # run test suite
```

---

## Tasks implemented

### Task 3 — ML experimentation with MLflow
- **Algorithms**: KNN, SVM (LinearSVC + GridSearchCV), Logistic Regression, Random Forest
- **Dimensionality reduction**: PCA (LogisticRegression on reduced space), t-SNE visualisation
- **MLflow**: all runs logged with params, metrics, model artefacts, classification reports
- **Reports**: `reports/task3_experiments.md`, `reports/task3_comparison.csv`

### Task 4 — Random Forest interpretation
- **Feature importance**: top-20 chart + domain analysis (`reports/rf_feature_importance.png`)
- **Stability**: 5 random seeds, F1 std logged
- **Error analysis**: misclassified samples with fraud scores (`reports/rf_misclassified_samples.csv`)
- **Bias/variance**: 3×4 grid of n_estimators × max_depth with overfitting/underfitting labels
- **Comparison**: RF vs Decision Tree (confusion matrices, all metrics)
- **Reports**: `reports/task4_random_forest.md`

### Task 5 — Local MLOps pipeline
- **Tracking**: MLflow experiment `fraud_detection_task3_task4`, params + metrics + artefacts
- **Registry**: `registry_service.py` — register, tag, Staging → Production promotion
- **Serving**: FastAPI `/api/predict` + native `mlflow models serve` via `serve_mlflow.py`
- **Drift detection**: `simulate_drift.py` — Evidently HTML report, KS-test per feature,
  `SEUIL_DRIFT=0.30` / `SEUIL_WARN=0.15`, automatic retraining trigger
- **CI/CD**: `Makefile` with `setup`, `train`, `register`, `serve`, `test`, `pipeline` targets
- **Pre-commit hook**: `hooks/pre-commit` — rejects commits if `val_accuracy < 0.80`

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/runs` | List MLflow runs |
| GET | `/api/summary` | Full summary (models, metrics, artefacts) |
| POST | `/api/train` | Start training job (background) |
| POST | `/api/tune` | Start hyperparameter tuning job |
| GET | `/api/jobs/{id}` | Poll job status |
| POST | `/api/jobs/{id}/pause` | Pause a running job |
| POST | `/api/jobs/{id}/cancel` | Cancel a job |
| POST | `/api/predict` | Fraud prediction on a transaction |
| POST | `/api/registry/register` | Register model in MLflow Registry |
| POST | `/api/registry/promote` | Promote model to Staging/Production |
| GET | `/api/registry/versions` | List registered model versions |
| POST | `/api/drift/detect` | Run drift detection (background) |
| GET | `/api/dataset/summary` | Dataset statistics |
| POST | `/api/dataset/clean` | Apply cleaning action |
| GET | `/api/dataset/export` | Export dataset as CSV |
| GET | `/api/artifacts/{name}` | Serve a report artefact |

---

## Dataset

IEEE-CIS Fraud Detection (Kaggle).
Download from: https://www.kaggle.com/c/ieee-fraud-detection/data

Place the raw CSVs in `dataset/raw/`. The processed splits in `dataset/processed_data/`
are generated by the data preparation pipeline.

---

## MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open http://localhost:5000 to browse experiments, compare runs, and download artefacts.
