"""Inference: load the saved CNN and predict a digit from an image.

CLI:   python predict.py path/to/image.png [--polarity auto|dark_on_light|light_on_dark]
Import: used by app.py (Streamlit) and the tests.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import config as C
from src.preprocessing import preprocess_uploaded_image


def load_trained_model(path: Path = C.MODEL_PATH):
    """Load the saved Keras model (no retraining)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python train.py` first "
            "(or commit models/digit_cnn_model.keras to your repository).")
    import keras  # imported lazily so preprocessing tests run without TensorFlow

    return keras.models.load_model(path)


def predict_array(model, x: np.ndarray):
    """x: (1, 8, 8, 1) in 0..1  ->  (digit, confidence 0..1, probabilities[10])."""
    probs = np.asarray(model.predict(x, verbose=0))[0].astype(float)
    digit = int(np.argmax(probs))
    return digit, float(probs[digit]), probs


def predict_image(model, image: Image.Image, polarity: str = "auto",
                  threshold: float = C.LOW_CONFIDENCE_THRESHOLD) -> dict:
    """Full inference: uploaded image -> preprocessing -> CNN -> result dict.

    Raises ValueError (from preprocessing) when no clear digit is found.
    """
    x, preview, used = preprocess_uploaded_image(image, polarity)
    digit, confidence, probs = predict_array(model, x)
    order = np.argsort(probs)[::-1]
    return {
        "digit": digit,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "second_choice": int(order[1]),
        "second_confidence": float(probs[order[1]]),
        "low_confidence": confidence < threshold,
        "preview": preview,
        "polarity_used": used,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="path to a PNG/JPG containing one handwritten digit")
    parser.add_argument("--polarity", default="auto", choices=["auto", "dark_on_light", "light_on_dark"])
    args = parser.parse_args()
    model = load_trained_model()
    try:
        res = predict_image(model, Image.open(args.image), args.polarity)
    except ValueError as err:
        print("Could not read a digit:", err)
        return 1
    print(f"Predicted digit: {res['digit']}")
    print(f"Confidence: {res['confidence']:.2%}  (polarity used: {res['polarity_used']})")
    if res["low_confidence"]:
        print("WARNING: low confidence - please verify manually or upload a clearer image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
