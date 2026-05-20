# Backend API (Local)

This service exposes MLflow runs, reports, and a training trigger for the frontend.

## Run

From the repo root:

```
python -m uvicorn backend.app:app --reload --port 8000
```

## Endpoints

- GET /api/health
- GET /api/runs?limit=50
- GET /api/summary
- POST /api/train
- GET /api/jobs
- GET /api/jobs/{job_id}
- GET /api/artifacts/{name}

## Notes

- MLflow tracking is read from the local mlruns/ directory.
- Reports and images are served from reports/.
- Set ALLOWED_ORIGINS to change CORS (default is http://localhost:5173).
