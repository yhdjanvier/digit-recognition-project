"""Training pipeline pieces: one training run, diagnosis, comparison."""
import keras

from config import CHECKPOINT_DIR, SEED
from src.evaluation import compute_metrics, predict_classes
from src.model import build_cnn
from src.preprocessing import augment_with_shifts
from src.utils import ensure_dirs, set_global_seed


def train_one(cfg: dict, splits: dict, seed: int = SEED, verbose: int = 2):
    """Train one configuration. Returns (model, history_dict).

    - Training data is optionally augmented with shifted copies (TRAIN ONLY).
    - Validation data is only used to monitor training / pick the best weights.
    """
    set_global_seed(seed)
    X_train, y_train = splits["X_train"], splits["y_train"]
    if cfg["augment"]:
        X_train, y_train = augment_with_shifts(X_train, y_train, seed=seed)

    model = build_cnn(cfg)
    callbacks = []
    if cfg["early_stopping"]:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["patience"], restore_best_weights=True))
        ensure_dirs(CHECKPOINT_DIR)
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / f"best_{cfg['name']}.keras"),
            monitor="val_loss", save_best_only=True))
    if cfg["reduce_lr"]:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5))

    history = model.fit(
        X_train, y_train,
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        callbacks=callbacks, shuffle=True, verbose=verbose,
    )
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()
            if k in ("loss", "accuracy", "val_loss", "val_accuracy")}
    return model, hist


def diagnose_history(hist: dict) -> dict:
    """Read the learning curves and report problems, as numbers + sentences."""
    n = len(hist["loss"])
    val_loss = hist["val_loss"]
    best_epoch = int(min(range(n), key=lambda i: val_loss[i])) + 1
    gap = hist["accuracy"][-1] - hist["val_accuracy"][-1]
    loss_rise = val_loss[-1] - min(val_loss)
    overfitting = gap > 0.02 or loss_rise > 0.05 * max(min(val_loss), 1e-6)
    still_improving = best_epoch >= n - 1  # best validation loss at (or next to) the last epoch
    findings = []
    if overfitting:
        findings.append(
            f"Possible overfitting: final train-validation accuracy gap {gap:.3f}; "
            f"validation loss is {loss_rise:.4f} above its minimum (epoch {best_epoch}).")
    if still_improving:
        findings.append(
            f"Training stopped while validation loss was still improving (best epoch "
            f"{best_epoch} of {n}): the model may be under-trained.")
    if not findings:
        findings.append("No strong overfitting or under-training signal in the curves.")
    return {
        "epochs_run": n, "best_val_loss_epoch": best_epoch,
        "final_train_accuracy": hist["accuracy"][-1], "final_val_accuracy": hist["val_accuracy"][-1],
        "train_val_gap": gap, "overfitting": bool(overfitting),
        "under_trained": bool(still_improving), "findings": findings,
    }


def score_on_split(model, X, y) -> dict:
    """Metrics + loss of a model on one split (used for validation selection)."""
    y_pred, _ = predict_classes(model, X)
    metrics = compute_metrics(y, y_pred)
    loss = float(model.evaluate(X, y, verbose=0)[0])
    metrics["loss"] = loss
    return metrics
