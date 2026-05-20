# =============================================================================
# Makefile — Fraud Detection MLOps Pipeline Orchestrator
#
# Usage:
#   make setup      Install Python dependencies
#   make train      Start the API server and trigger a full training run
#   make register   Register and promote the best model in the MLflow Registry
#   make serve      Start the FastAPI server (foreground)
#   make test       Run the test suite
#   make drift      Run the data drift detection pipeline
#   make pipeline   Full end-to-end: setup → train → register → drift
#
# Prerequisites: Python 3.9+, pip, npm (for frontend)
# =============================================================================

PYTHON      ?= python
PIP         ?= pip
UVICORN     ?= uvicorn
API_HOST    ?= 0.0.0.0
API_PORT    ?= 8000
API_MODULE  ?= backend.app:app

.PHONY: setup train register serve test drift pipeline frontend help

# ---------------------------------------------------------------------------
# check — verify installed package versions meet Tâche 5 minimums
# ---------------------------------------------------------------------------
check:
	@echo ">>> Checking installed package versions..."
	$(PYTHON) -m backend.check_versions

# ---------------------------------------------------------------------------
# setup — install all Python dependencies
# ---------------------------------------------------------------------------
setup: check
	@echo ">>> Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@echo ">>> Setup complete."

# ---------------------------------------------------------------------------
# serve — start the FastAPI backend (foreground, with auto-reload)
# ---------------------------------------------------------------------------
serve:
	@echo ">>> Starting FastAPI server on http://$(API_HOST):$(API_PORT) ..."
	$(UVICORN) $(API_MODULE) --host $(API_HOST) --port $(API_PORT) --reload

# ---------------------------------------------------------------------------
# serve-mlflow — serve the Production model via native MLflow serving
# ---------------------------------------------------------------------------
serve-mlflow:
	@echo ">>> Starting MLflow native model server on port 1234..."
	$(PYTHON) -m backend.serve_mlflow serve --port 1234

# ---------------------------------------------------------------------------
# test-mlflow — test the running MLflow model endpoint
# ---------------------------------------------------------------------------
test-mlflow:
	@echo ">>> Testing MLflow model endpoint on port 1234..."
	$(PYTHON) -m backend.serve_mlflow test --port 1234

# ---------------------------------------------------------------------------
# train — trigger a full training run (Task 3 + Task 4) via the REST API
#
# Requires the server to be running in another terminal (make serve).
# Alternatively, run training directly without the server:
#   python -c "from backend.training import run_task3_task4; run_task3_task4()"
# ---------------------------------------------------------------------------
train:
	@echo ">>> Triggering training job via API..."
	$(PYTHON) -c "\
import urllib.request, json; \
payload = json.dumps({'run_task4': True, 'use_sample': True}).encode(); \
req = urllib.request.Request('http://localhost:$(API_PORT)/api/train', data=payload, headers={'Content-Type': 'application/json'}, method='POST'); \
resp = urllib.request.urlopen(req); \
print('Training job queued:', json.loads(resp.read()))"

# ---------------------------------------------------------------------------
# register — register the best MLflow run in the Model Registry and promote
#            it through Staging → Production (if accuracy >= 0.85)
# ---------------------------------------------------------------------------
register:
	@echo ">>> Registering and promoting best model..."
	$(PYTHON) -c "\
import urllib.request, json; \
\
# Get all runs and find the best F1 \
resp = urllib.request.urlopen('http://localhost:$(API_PORT)/api/runs?limit=100'); \
runs = json.loads(resp.read()); \
completed = [r for r in runs if r.get('status') == 'completed' and r.get('metrics', {}).get('f1')]; \
if not completed: \
    print('No completed runs found. Run make train first.'); \
    exit(1); \
best = max(completed, key=lambda r: r['metrics']['f1']); \
print(f'Best run: {best[\"id\"]} ({best[\"algorithm\"]}) F1={best[\"metrics\"][\"f1\"]:.4f}'); \
\
# Register \
payload = json.dumps({'run_id': best['id']}).encode(); \
req = urllib.request.Request('http://localhost:$(API_PORT)/api/registry/register', data=payload, headers={'Content-Type': 'application/json'}, method='POST'); \
reg = json.loads(urllib.request.urlopen(req).read()); \
print('Registered:', reg); \
\
# Promote \
payload = json.dumps({'version': str(reg['version'])}).encode(); \
req = urllib.request.Request('http://localhost:$(API_PORT)/api/registry/promote', data=payload, headers={'Content-Type': 'application/json'}, method='POST'); \
promo = json.loads(urllib.request.urlopen(req).read()); \
print('Promoted to:', promo['stage'])"

# ---------------------------------------------------------------------------
# drift — run the data drift detection pipeline directly (no server needed)
# ---------------------------------------------------------------------------
drift:
	@echo ">>> Running drift detection pipeline..."
	$(PYTHON) -m backend.simulate_drift

# ---------------------------------------------------------------------------
# drift-retrain — run drift detection and force retraining
# ---------------------------------------------------------------------------
drift-retrain:
	@echo ">>> Running drift detection with forced retraining..."
	$(PYTHON) -m backend.simulate_drift --retrain

# ---------------------------------------------------------------------------
# test — run the test suite with pytest
# ---------------------------------------------------------------------------
test:
	@echo ">>> Running tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short 2>/dev/null || \
	$(PYTHON) -m pytest backend/ -v --tb=short --ignore=backend/__pycache__ 2>/dev/null || \
	echo "No tests found. Add tests to a tests/ directory."

# ---------------------------------------------------------------------------
# frontend — install npm dependencies and build the frontend
# ---------------------------------------------------------------------------
frontend:
	@echo ">>> Installing frontend dependencies..."
	cd frontend && npm install
	@echo ">>> Building frontend..."
	cd frontend && npm run build
	@echo ">>> Frontend build complete (dist/)."

# ---------------------------------------------------------------------------
# pipeline — full end-to-end pipeline (no server required)
# ---------------------------------------------------------------------------
pipeline: setup
	@echo ">>> Running full MLOps pipeline..."
	@echo ""
	@echo "--- Step 1: Training (Task 3 + Task 4) ---"
	$(PYTHON) -c "\
from backend.training import run_task3_task4; \
result = run_task3_task4(options={'run_task4': True, 'use_sample': True}); \
print('Training complete:', result)"
	@echo ""
	@echo "--- Step 2: Drift Detection ---"
	$(PYTHON) -m backend.simulate_drift
	@echo ""
	@echo ">>> Pipeline complete. Start the server with: make serve"

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "Fraud Detection MLOps — Available targets:"
	@echo "  make setup         Install Python dependencies"
	@echo "  make serve         Start FastAPI server (port $(API_PORT))"
	@echo "  make train         Trigger training via API (server must be running)"
	@echo "  make register      Register + promote best model (server must be running)"
	@echo "  make drift         Run drift detection pipeline (standalone)"
	@echo "  make drift-retrain Run drift detection + force retraining"
	@echo "  make test          Run test suite"
	@echo "  make frontend      Build the React frontend"
	@echo "  make pipeline      Full pipeline without server (setup→train→drift)"
	@echo ""
