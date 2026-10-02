"""Train the CNN: baseline -> evaluate -> diagnose -> adjust -> re-evaluate -> select.

Usage:
    python train.py            # full run (about a minute on a normal CPU)
    python train.py --quick    # 2-epoch smoke test (overwrites the saved model!)
"""
import argparse
import copy
import sys

import keras
import tensorflow as tf

import config as C
from src.data_loader import dataset_statistics, get_splits, load_digits_dataset
from src.evaluation import evaluate_and_save, interpret_metrics, plot_training_history
from src.training import diagnose_history, score_on_split, train_one
from src.utils import (ensure_dirs, plot_class_distribution, plot_sample_images,
                       save_json, set_global_seed)


def main(quick: bool = False) -> int:
    set_global_seed(C.SEED)
    ensure_dirs(C.MODELS_DIR, C.RESULTS_DIR, C.SCREENSHOTS_DIR)
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | seed {C.SEED}")

    # ---- 1. data acquisition + inspection -----------------------------------
    images, labels = load_digits_dataset()
    stats = dataset_statistics(images, labels)
    save_json(stats, C.DATASET_STATS_PATH)
    plot_sample_images(images, labels, C.RESULTS_DIR / "dataset_samples.png")
    plot_class_distribution(labels, C.RESULTS_DIR / "class_distribution.png")
    print("Dataset:", stats)

    # ---- 2. preprocessing + split (train / validation / test) ----------------
    splits = get_splits(C.SEED)
    print(f"Split sizes -> train {len(splits['y_train'])}, validation {len(splits['y_val'])}, "
          f"test {len(splits['y_test'])}; input shape {splits['X_train'].shape[1:]}")

    base_cfg, adj_cfg = copy.deepcopy(C.BASELINE_CONFIG), copy.deepcopy(C.ADJUSTED_CONFIG)
    if quick:
        base_cfg["epochs"], adj_cfg["epochs"] = 2, 2

    # ---- 3. BASELINE model --------------------------------------------------
    print("\n=== BASELINE MODEL ===", base_cfg)
    base_model, base_hist = train_one(base_cfg, splits)
    base_model.summary()
    base_val = score_on_split(base_model, splits["X_val"], splits["y_val"])
    base_diag = diagnose_history(base_hist)
    plot_training_history(base_hist, C.RESULTS_DIR, "baseline", " (baseline)")
    print("Baseline validation accuracy:", round(base_val["accuracy"], 4))
    print("Baseline diagnosis:\n  - " + "\n  - ".join(base_diag["findings"]))

    # ---- 4. ADJUSTED model (response to the diagnosis) ----------------------
    print("\n=== ADJUSTED MODEL ===", adj_cfg)
    adj_model, adj_hist = train_one(adj_cfg, splits)
    adj_val = score_on_split(adj_model, splits["X_val"], splits["y_val"])
    adj_diag = diagnose_history(adj_hist)
    plot_training_history(adj_hist, C.RESULTS_DIR, "adjusted", " (adjusted)")
    print("Adjusted validation accuracy:", round(adj_val["accuracy"], 4))
    print("Adjusted diagnosis:\n  - " + "\n  - ".join(adj_diag["findings"]))

    # ---- 5. model selection on VALIDATION data (test set stays untouched) ----
    adj_wins = (adj_val["accuracy"], -adj_val["loss"]) >= (base_val["accuracy"], -base_val["loss"])
    final_name, final_model, final_cfg, final_hist = (
        ("adjusted", adj_model, adj_cfg, adj_hist) if adj_wins
        else ("baseline", base_model, base_cfg, base_hist))
    print(f"\nSelected model (by validation accuracy, then validation loss): {final_name}")

    # ---- 6. final test-set evaluation of BOTH models (reporting only) --------
    base_test = score_on_split(base_model, splits["X_test"], splits["y_test"])
    adj_test = score_on_split(adj_model, splits["X_test"], splits["y_test"])
    comparison = {
        "baseline": {"config": base_cfg, "diagnosis": base_diag,
                     "validation": base_val, "test": base_test},
        "adjusted": {"config": adj_cfg, "diagnosis": adj_diag,
                     "validation": adj_val, "test": adj_test},
        "selected_model": final_name,
        "selection_rule": "higher validation accuracy, ties broken by lower validation loss",
        "quick_mode": quick,
    }
    save_json(comparison, C.COMPARISON_PATH)
    write_comparison_markdown(comparison)

    # ---- 7. save final model, history, configuration, results ---------------
    final_model.save(C.MODEL_PATH)
    base_model.save(C.BASELINE_MODEL_PATH)
    save_json({"baseline": base_hist, "adjusted": adj_hist, "selected": final_name}, C.HISTORY_PATH)
    plot_training_history(final_hist, C.RESULTS_DIR, "training", f" ({final_name} model)")
    save_json({
        "model_name": final_name, "model_file": C.MODEL_PATH.name,
        "hyperparameters": final_cfg, "input_shape": list(C.INPUT_SHAPE),
        "num_classes": C.NUM_CLASSES, "class_labels": C.CLASS_LABELS,
        "normalization": f"pixel / {C.PIXEL_MAX} (0..16 -> 0..1)",
        "seed": C.SEED, "test_size": C.TEST_SIZE, "validation_size": C.VAL_SIZE,
        "low_confidence_threshold": C.LOW_CONFIDENCE_THRESHOLD,
        "dataset": stats["dataset"],
        "tensorflow_version": tf.__version__, "keras_version": keras.__version__,
        "quick_mode": quick,
    }, C.MODEL_CONFIG_PATH)

    metrics = evaluate_and_save(final_model, splits["X_test"], splits["y_test"], final_name)
    print("\n=== FINAL TEST-SET RESULTS ===")
    print(interpret_metrics(metrics))
    print("\nSaved:", C.MODEL_PATH, "and results in", C.RESULTS_DIR)
    return 0


def write_comparison_markdown(cmp: dict) -> None:
    keys = [("accuracy", "Accuracy"), ("precision_macro", "Precision (macro)"),
            ("recall_macro", "Recall (macro)"), ("f1_macro", "F1-score (macro)")]
    lines = ["# Baseline vs adjusted model (generated by train.py)\n",
             "| Metric | Baseline val | Adjusted val | Baseline test | Adjusted test |",
             "|---|---|---|---|---|"]
    for k, label in keys:
        lines.append(f"| {label} | {cmp['baseline']['validation'][k]:.4f} | "
                     f"{cmp['adjusted']['validation'][k]:.4f} | "
                     f"{cmp['baseline']['test'][k]:.4f} | {cmp['adjusted']['test'][k]:.4f} |")
    lines += ["", f"Selected model: **{cmp['selected_model']}** ({cmp['selection_rule']}).", "",
              "## Baseline diagnosis", *[f"- {f}" for f in cmp["baseline"]["diagnosis"]["findings"]],
              "", "## Adjusted diagnosis", *[f"- {f}" for f in cmp["adjusted"]["diagnosis"]["findings"]]]
    C.COMPARISON_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="2-epoch smoke test")
    sys.exit(main(parser.parse_args().quick))
