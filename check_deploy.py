"""Pre-flight check before pushing to GitHub / deploying on Streamlit Cloud.

Usage: python check_deploy.py
Verifies required files exist, sizes are below GitHub's limit, .gitignore is sane,
the saved model loads and predicts, and prints the library versions to pin.
"""
import sys
from pathlib import Path

import config as C

ROOT = C.ROOT_DIR
REQUIRED = [
    "app.py", "predict.py", "config.py", "requirements.txt", "README.md", ".gitignore",
    "src/__init__.py", "src/preprocessing.py", "src/data_loader.py", "src/utils.py",
    "src/model.py", "src/evaluation.py", "src/training.py",
    "models/digit_cnn_model.keras", "models/model_config.json", "results/metrics.json",
]
problems = []


def ok(msg): print(f"[OK]      {msg}")
def bad(msg): print(f"[PROBLEM] {msg}"); problems.append(msg)


for rel in REQUIRED:
    (ok if (ROOT / rel).exists() else bad)(f"{rel}" if (ROOT / rel).exists() else f"missing file: {rel} (run python train.py?)")

big = [p for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts
       and p.stat().st_size > 50 * 1024 * 1024]
for p in big:
    bad(f"file over 50 MB (GitHub blocks 100 MB): {p.relative_to(ROOT)}")
if not big:
    ok("no oversized files")

gi = (ROOT / ".gitignore").read_text() if (ROOT / ".gitignore").exists() else ""
for pattern in (".venv", "__pycache__"):
    (ok if pattern in gi else bad)(f".gitignore covers {pattern}")
if "digit_cnn_model" in gi or "*.keras" in gi:
    bad(".gitignore excludes the model - Streamlit Cloud could not load it")

try:
    import numpy as np
    from predict import load_trained_model
    from src.data_loader import get_splits

    model = load_trained_model()
    s = get_splits(C.SEED)
    acc = float((np.argmax(model.predict(s["X_test"], verbose=0), axis=1) == s["y_test"]).mean())
    ok(f"saved model loads; test accuracy re-computed now: {acc:.4f}")
except Exception as err:  # noqa: BLE001
    bad(f"model could not be loaded/evaluated: {err}")

try:
    import keras
    import tensorflow as tf

    print(f"\nPin these in requirements.txt for an identical environment on the cloud:\n"
          f"    tensorflow=={tf.__version__}\n    (Keras {keras.__version__} comes with it)")
except Exception:
    pass

print("\nRESULT:", "READY to push." if not problems else f"{len(problems)} problem(s) to fix first.")
sys.exit(1 if problems else 0)
