from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = Path(os.getenv("MLFLOW_TRACKING_DIR", PROJECT_ROOT / "mlruns")).resolve()
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports")).resolve()
DATA_DIR = Path(
    os.getenv("PROCESSED_DATA_DIR", PROJECT_ROOT / "dataset" / "processed_data")
).resolve()

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:5174"
    ).split(",")
    if origin.strip()
]
