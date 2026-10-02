"""Data acquisition: load the scikit-learn Digits dataset and split it.

The dataset ships inside scikit-learn, so no manual download is needed.
Splitting is done BEFORE any preprocessing and by index, so test images can
never leak into training or validation.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from config import SEED, TEST_SIZE, VAL_SIZE
from src.preprocessing import prepare_inputs


def load_digits_dataset():
    """Return (images, labels): images (1797, 8, 8) float32 in 0..16, labels int64."""
    data = load_digits()
    return data.images.astype("float32"), data.target.astype("int64")


def dataset_statistics(images, labels) -> dict:
    """Every number is computed from the data, none is hard-coded."""
    counts = np.bincount(labels, minlength=10)
    return {
        "dataset": "scikit-learn Digits (sklearn.datasets.load_digits)",
        "num_samples": int(images.shape[0]),
        "num_classes": int(len(np.unique(labels))),
        "image_shape": [int(images.shape[1]), int(images.shape[2])],
        "pixel_min": float(images.min()),
        "pixel_max": float(images.max()),
        "pixel_mean": round(float(images.mean()), 4),
        "class_distribution": {str(i): int(c) for i, c in enumerate(counts)},
    }


def make_raw_splits(images, labels, seed: int = SEED) -> dict:
    """Stratified train / validation / test split of RAW images (0..16).

    1. 20 % of all data -> test set (never used for training or model selection)
    2. 20 % of the remainder -> validation set (early stopping, model selection)
    3. the rest -> training set
    """
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=TEST_SIZE, random_state=seed, stratify=labels
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=VAL_SIZE,
        random_state=seed,
        stratify=labels[idx_trainval],
    )
    return {
        "idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test,
        "X_train_raw": images[idx_train], "y_train": labels[idx_train],
        "X_val_raw": images[idx_val], "y_val": labels[idx_val],
        "X_test_raw": images[idx_test], "y_test": labels[idx_test],
    }


def get_splits(seed: int = SEED) -> dict:
    """Raw splits plus CNN-ready arrays X_train / X_val / X_test (N, 8, 8, 1) in 0..1."""
    images, labels = load_digits_dataset()
    splits = make_raw_splits(images, labels, seed)
    for part in ("train", "val", "test"):
        splits[f"X_{part}"] = prepare_inputs(splits[f"X_{part}_raw"])
    return splits
