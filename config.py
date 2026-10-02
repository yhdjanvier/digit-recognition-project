"""Central configuration for the Handwritten Digit Recognition project.

Every other module imports its constants from here, so the training pipeline,
the evaluation script, the notebook and the Streamlit app always agree.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths ----
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

MODEL_PATH = MODELS_DIR / "digit_cnn_model.keras"          # final (selected) model
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_cnn_model.keras"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

METRICS_PATH = RESULTS_DIR / "metrics.json"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.json"
COMPARISON_MD_PATH = RESULTS_DIR / "model_comparison.md"
REPORT_PATH = RESULTS_DIR / "classification_report.txt"
DATASET_STATS_PATH = RESULTS_DIR / "dataset_statistics.json"
HISTORY_PATH = RESULTS_DIR / "training_history.json"

# ------------------------------------------------------------ reproducibility
SEED = 42

# --------------------------------------------------------------- data facts --
IMG_HEIGHT = 8
IMG_WIDTH = 8
CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
NUM_CLASSES = 10
CLASS_LABELS = list(range(NUM_CLASSES))
PIXEL_MAX = 16.0            # sklearn digits pixel range is 0..16

TEST_SIZE = 0.20            # 20 % of all data held out for the final test
VAL_SIZE = 0.20             # 20 % of the remaining training data used for validation

# Predictions below this confidence trigger a warning in the app / CLI.
LOW_CONFIDENCE_THRESHOLD = 0.80

# ------------------------------------------------------- model configurations
# BASELINE: the configuration from the original project proposal
# (10 epochs, no dropout, no augmentation, no callbacks).
BASELINE_CONFIG = {
    "name": "baseline",
    "conv1_filters": 32,
    "conv2_filters": 64,
    "kernel_size": 3,
    "dense_units": 64,
    "dropout": 0.0,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 10,
    "augment": False,
    "early_stopping": False,
    "reduce_lr": False,
    "patience": 0,
}

# ADJUSTED: the response to the baseline evaluation (see README, section
# "Evaluation-based adjustment"): regularisation (dropout + shift
# augmentation), a longer training budget controlled by EarlyStopping, and
# learning-rate reduction on plateau.
ADJUSTED_CONFIG = {
    "name": "adjusted",
    "conv1_filters": 32,
    "conv2_filters": 64,
    "kernel_size": 3,
    "dense_units": 64,
    "dropout": 0.30,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 60,
    "augment": True,
    "early_stopping": True,
    "reduce_lr": True,
    "patience": 8,
}
