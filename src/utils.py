"""Small helpers: seeding, JSON I/O and dataset visualisations."""
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; works on servers without a display
import matplotlib.pyplot as plt
import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow/Keras."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)  # seeds python, numpy and backend
    except Exception:  # TensorFlow not installed (e.g. preprocessing-only use)
        pass


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_sample_images(images, labels, path: Path, n_per_class: int = 5) -> None:
    """Grid with n_per_class example images for each digit 0-9."""
    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.3, 13))
    for digit in range(10):
        idx = np.where(labels == digit)[0][:n_per_class]
        for j in range(n_per_class):
            ax = axes[digit, j]
            ax.imshow(images[idx[j]], cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{digit}", rotation=0, labelpad=12, fontsize=12)
    fig.suptitle("Sample images from the scikit-learn Digits dataset (8x8)", y=0.995)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_distribution(labels, path: Path) -> None:
    counts = np.bincount(labels, minlength=10)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(10), counts, color="#4C72B0")
    ax.bar_label(bars)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit class")
    ax.set_ylabel("Number of images")
    ax.set_title("Class distribution (full dataset)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
