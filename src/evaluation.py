"""Evaluation: metrics, confusion matrix, classification report, curves."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import CLASS_LABELS, METRICS_PATH, REPORT_PATH, RESULTS_DIR
from src.utils import save_json


def predict_classes(model, X):
    """Return (predicted classes, probability matrix)."""
    probs = model.predict(X, verbose=0)
    return np.argmax(probs, axis=1), probs


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus macro / weighted precision, recall, F1 and per-class values."""
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_LABELS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_m), "recall_macro": float(r_m), "f1_macro": float(f_m),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "per_class": {
            str(c): {"precision": float(p_c[i]), "recall": float(r_c[i]),
                     "f1": float(f_c[i]), "support": int(s_c[i])}
            for i, c in enumerate(CLASS_LABELS)
        },
    }


def save_confusion_matrix(y_true, y_pred, path: Path, title="Confusion matrix (test set)"):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cm


def most_confused_pairs(cm: np.ndarray, top: int = 3):
    """Largest off-diagonal confusion-matrix cells -> [(true, predicted, count)]."""
    pairs = [(i, j, int(cm[i, j])) for i in range(cm.shape[0])
             for j in range(cm.shape[1]) if i != j and cm[i, j] > 0]
    return sorted(pairs, key=lambda t: -t[2])[:top]


def plot_training_history(history: dict, out_dir: Path, prefix: str = "training", title_suffix: str = ""):
    """Save <prefix>_accuracy.png and <prefix>_loss.png (train vs validation)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)
    for key, label in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history[key], marker="o", markersize=3, label=f"Training {key}")
        ax.plot(epochs, history[f"val_{key}"], marker="o", markersize=3, label=f"Validation {key}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Training vs validation {key}{title_suffix}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{key}.png", dpi=130)
        plt.close(fig)


def evaluate_and_save(model, X_test, y_test, model_name: str = "final") -> dict:
    """Evaluate on the held-out test set and write every result file.

    Writes results/metrics.json, classification_report.txt, confusion_matrix.png.
    Every number comes from the model's actual predictions.
    """
    y_pred, _ = predict_classes(model, X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["model_name"] = model_name
    metrics["test_set_size"] = int(len(y_test))

    report = classification_report(y_test, y_pred, labels=CLASS_LABELS, digits=4, zero_division=0)
    cm = save_confusion_matrix(y_test, y_pred, RESULTS_DIR / "confusion_matrix.png")
    metrics["most_confused_pairs"] = [
        {"true": t, "predicted": p, "count": c} for t, p, c in most_confused_pairs(cm)
    ]
    metrics["confusion_matrix"] = cm.tolist()

    save_json(metrics, METRICS_PATH)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        f"Classification report - model '{model_name}' - held-out test set "
        f"({len(y_test)} images)\n\n{report}\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n", encoding="utf-8")
    return metrics


def interpret_metrics(metrics: dict) -> str:
    """Plain-language interpretation generated from the actual numbers."""
    pc = metrics["per_class"]
    worst = min(pc, key=lambda c: pc[c]["f1"])
    best = max(pc, key=lambda c: pc[c]["f1"])
    lines = [
        f"Test accuracy is {metrics['accuracy']:.2%}; macro F1 is {metrics['f1_macro']:.4f} "
        f"(macro precision {metrics['precision_macro']:.4f}, macro recall {metrics['recall_macro']:.4f}).",
        f"Best class: digit {best} (F1 {pc[best]['f1']:.3f}); weakest class: digit {worst} (F1 {pc[worst]['f1']:.3f}).",
    ]
    pairs = metrics.get("most_confused_pairs", [])
    if pairs:
        txt = ", ".join(f"true {p['true']} predicted as {p['predicted']} ({p['count']}x)" for p in pairs)
        lines.append(f"Most frequent confusions: {txt}.")
    else:
        lines.append("No misclassifications on the test set.")
    return "\n".join(lines)
