import json
from pathlib import Path

from .config import REPORTS_DIR

REGISTRY_PATH = REPORTS_DIR / "current_model.json"


def get_active_model():
    if not REGISTRY_PATH.exists():
        return None
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def set_active_model(run_id):
    REPORTS_DIR.mkdir(exist_ok=True)
    payload = {"run_id": run_id}
    REGISTRY_PATH.write_text(json.dumps(payload), encoding="utf-8")
    return payload
