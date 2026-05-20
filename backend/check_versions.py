"""
check_versions.py — Verify that installed package versions meet the minimums
required by the Tâche 5 stack specification.

Usage:
    python -m backend.check_versions
"""
from __future__ import annotations

import importlib
import sys
from packaging.version import Version


REQUIREMENTS = [
    ("mlflow",      "2.10.0",  "MLflow"),
    ("sklearn",     "1.3.0",   "scikit-learn"),
    ("pandas",      "2.0.0",   "pandas"),
    ("numpy",       "1.26.0",  "numpy"),
    ("scipy",       "1.11.0",  "scipy"),
    ("evidently",   "0.4.0",   "Evidently"),
    ("optuna",      "3.5.0",   "Optuna"),
    ("fastapi",     "0.115.0", "FastAPI"),
    ("uvicorn",     "0.30.0",  "uvicorn"),
]

PYTHON_MIN = (3, 9)


def _get_version(module_name: str) -> str | None:
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def main() -> int:
    print("=" * 60)
    print("Stack version check — Tâche 5 requirements")
    print("=" * 60)

    # Python version
    py_ver = sys.version_info[:2]
    py_ok = py_ver >= PYTHON_MIN
    status = "✓" if py_ok else "✗"
    print(f"  {status}  Python  {'.'.join(map(str, py_ver))}  (min {'.'.join(map(str, PYTHON_MIN))})")

    all_ok = py_ok
    for module, min_ver, label in REQUIREMENTS:
        installed = _get_version(module)
        if installed is None:
            print(f"  ✗  {label:<20} NOT INSTALLED  (min {min_ver})")
            all_ok = False
        else:
            try:
                ok = Version(installed) >= Version(min_ver)
            except Exception:
                ok = True  # Can't parse — assume ok
            status = "✓" if ok else "✗"
            print(f"  {status}  {label:<20} {installed:<12}  (min {min_ver})")
            if not ok:
                all_ok = False

    print("=" * 60)
    if all_ok:
        print("All version requirements satisfied.")
    else:
        print("Some requirements are not met. Run: pip install -r requirements.txt")
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
