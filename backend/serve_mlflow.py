"""
serve_mlflow.py — Native MLflow model serving helper.

Provides utilities to:
1. Serve the Production model via ``mlflow models serve`` (subprocess).
2. Test the running endpoint with a sample transaction (curl-equivalent via requests).

Usage
-----
# Serve the Production model on port 1234:
    python -m backend.serve_mlflow serve --port 1234

# Test the running endpoint:
    python -m backend.serve_mlflow test --port 1234

# Serve then test (blocks until Ctrl-C):
    python -m backend.serve_mlflow serve-and-test --port 1234
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

DEFAULT_MODEL_NAME = "FraudDetectionModel"
DEFAULT_PORT = 1234


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_production_uri(model_name: str = DEFAULT_MODEL_NAME) -> str:
    """Return the MLflow model URI for the current Production version."""
    return f"models:/{model_name}/Production"


def _serve(port: int = DEFAULT_PORT, model_name: str = DEFAULT_MODEL_NAME) -> subprocess.Popen:
    """
    Launch ``mlflow models serve`` as a subprocess.

    Returns the Popen handle so the caller can wait or terminate it.
    """
    model_uri = _get_production_uri(model_name)
    cmd = [
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_uri,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--no-conda",
        "--env-manager", "local",
    ]
    env = {"MLFLOW_TRACKING_URI": MLRUNS_DIR.as_uri()}
    import os
    full_env = {**os.environ, **env}

    print(f"Starting MLflow model server: {model_uri} on port {port}")
    print(f"Command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=full_env)
    return proc


def _wait_for_server(port: int, timeout: int = 30) -> bool:
    """Poll until the server responds or timeout expires."""
    import urllib.request
    url = f"http://localhost:{port}/ping"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def _test_endpoint(port: int = DEFAULT_PORT) -> dict:
    """
    Send a sample transaction to the MLflow /invocations endpoint.

    Uses urllib (stdlib) so no extra dependencies are needed.
    Returns the parsed JSON response.
    """
    import urllib.request

    # Sample transaction — matches the processed feature schema
    sample = {
        "dataframe_records": [
            {
                "TransactionAmt": 68.5,
                "ProductCD": 2,       # label-encoded 'W'
                "card1": 13926,
                "card2": 321.0,
                "card3": 150.0,
                "card4": 3,           # label-encoded 'visa'
                "card5": 226.0,
                "card6": 0,           # label-encoded 'debit'
                "addr1": 299.0,
                "addr2": 87.0,
                "dist1": 0.0,
                "P_emaildomain": 0,   # label-encoded 'gmail.com'
                "R_emaildomain": 0,
            }
        ]
    }

    url = f"http://localhost:{port}/invocations"
    payload = json.dumps(sample).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"\nPOST {url}")
    print(f"Payload: {json.dumps(sample, indent=2)}")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            print(f"\nResponse: {json.dumps(result, indent=2)}")
            return result
    except Exception as exc:
        print(f"\nRequest failed: {exc}")
        print("Make sure the server is running: python -m backend.serve_mlflow serve")
        return {}


def _test_with_requests(port: int = DEFAULT_PORT) -> dict:
    """
    Same test using the ``requests`` library (if available).
    Falls back to urllib if requests is not installed.
    """
    try:
        import requests  # type: ignore
    except ImportError:
        print("requests not installed, falling back to urllib")
        return _test_endpoint(port)

    sample = {
        "dataframe_records": [
            {
                "TransactionAmt": 68.5,
                "ProductCD": 2,
                "card1": 13926,
                "card2": 321.0,
                "card3": 150.0,
                "card4": 3,
                "card5": 226.0,
                "card6": 0,
                "addr1": 299.0,
                "addr2": 87.0,
                "dist1": 0.0,
                "P_emaildomain": 0,
                "R_emaildomain": 0,
            }
        ]
    }

    url = f"http://localhost:{port}/invocations"
    print(f"\nPOST {url} (via requests)")
    resp = requests.post(url, json=sample, timeout=10)
    resp.raise_for_status()
    result = resp.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MLflow native model serving helper"
    )
    parser.add_argument(
        "command",
        choices=["serve", "test", "serve-and-test"],
        help=(
            "serve: start the MLflow model server | "
            "test: send a test request to the running server | "
            "serve-and-test: start server, wait, test, then keep serving"
        ),
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Registered model name")
    args = parser.parse_args()

    if args.command == "serve":
        proc = _serve(port=args.port, model_name=args.model_name)
        print(f"\nServer PID: {proc.pid}")
        print("Press Ctrl-C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\nServer stopped.")

    elif args.command == "test":
        _test_endpoint(port=args.port)
        print("\n--- Also testing via requests library ---")
        _test_with_requests(port=args.port)

    elif args.command == "serve-and-test":
        proc = _serve(port=args.port, model_name=args.model_name)
        print(f"Waiting for server to be ready (up to 30s)...")
        ready = _wait_for_server(args.port, timeout=30)
        if ready:
            print("Server is ready.")
            _test_endpoint(port=args.port)
            _test_with_requests(port=args.port)
        else:
            print("Server did not start in time.")
            proc.terminate()
            sys.exit(1)
        print("\nServer still running. Press Ctrl-C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
