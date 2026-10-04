"""
startup.py
----------
Convenience launcher that:
  1. Trains the model (if model.pkl is missing)
  2. Starts the FastAPI backend in a background thread
  3. Launches the Streamlit frontend

Usage:
    python startup.py
"""

import os
import subprocess
import sys
import threading
import time


def run_backend():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        check=True,
    )


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "model.pkl")

    # Train if needed
    if not os.path.exists(model_path):
        print("🔧  Training model (first-time setup)…")
        result = subprocess.run(
            [sys.executable, os.path.join(base, "train_model.py")], check=True
        )
        if result.returncode != 0:
            print("❌ Training failed. Aborting.")
            sys.exit(1)
        print("✅  Model trained and saved.")

    # Start backend in background thread
    print("🚀  Starting FastAPI backend on http://localhost:8000 …")
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    time.sleep(2)  # Give uvicorn a moment to start

    # Start frontend (blocking)
    print("🌐  Launching Streamlit frontend on http://localhost:8501 …")
    os.chdir(base)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            "8501",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
