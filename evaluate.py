"""Re-evaluate the SAVED model on the same held-out test split.

Usage: python evaluate.py
Recreates the identical stratified split (fixed seed), loads
models/digit_cnn_model.keras and regenerates metrics.json,
classification_report.txt and confusion_matrix.png.
"""
import sys

import config as C
from predict import load_trained_model
from src.data_loader import get_splits
from src.evaluation import evaluate_and_save, interpret_metrics


def main() -> int:
    model = load_trained_model(C.MODEL_PATH)
    splits = get_splits(C.SEED)
    name = "final"
    if C.MODEL_CONFIG_PATH.exists():
        from src.utils import load_json
        name = load_json(C.MODEL_CONFIG_PATH).get("model_name", "final")
    metrics = evaluate_and_save(model, splits["X_test"], splits["y_test"], name)
    print(f"Test images: {metrics['test_set_size']}")
    print(f"Accuracy {metrics['accuracy']:.4f} | macro precision {metrics['precision_macro']:.4f} | "
          f"macro recall {metrics['recall_macro']:.4f} | macro F1 {metrics['f1_macro']:.4f}")
    print(interpret_metrics(metrics))
    print(f"Results written to {C.RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
