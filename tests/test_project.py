"""Automated tests. Run with:  pytest -v

Tests that need TensorFlow are skipped automatically if it is not installed.
"""
import numpy as np
import pytest
from PIL import Image, ImageOps

import config as C
from src.data_loader import dataset_statistics, load_digits_dataset, make_raw_splits
from src.preprocessing import (augment_with_shifts, digit_array_to_pil, normalize_pixels,
                               preprocess_uploaded_image, prepare_inputs)


@pytest.fixture(scope="module")
def data():
    return load_digits_dataset()


# --------------------------------------------------------------- data tests --
def test_dataset_loading(data):
    images, labels = data
    assert images.shape == (1797, 8, 8)
    assert labels.shape == (1797,)
    assert set(np.unique(labels)) == set(range(10))
    assert images.min() >= 0 and images.max() <= 16


def test_dataset_statistics_are_computed(data):
    stats = dataset_statistics(*data)
    assert stats["num_samples"] == 1797 and stats["num_classes"] == 10
    assert sum(stats["class_distribution"].values()) == 1797


def test_split_is_stratified_disjoint_and_deterministic(data):
    images, labels = data
    a, b = make_raw_splits(images, labels, 42), make_raw_splits(images, labels, 42)
    tr, va, te = set(a["idx_train"]), set(a["idx_val"]), set(a["idx_test"])
    assert not (tr & va) and not (tr & te) and not (va & te)       # no leakage
    assert len(tr) + len(va) + len(te) == 1797
    assert np.array_equal(a["idx_test"], b["idx_test"])             # reproducible
    for y in (a["y_train"], a["y_val"], a["y_test"]):               # every class present
        assert set(np.unique(y)) == set(range(10))


def test_preprocessing_shape_and_range(data):
    images, _ = data
    x = prepare_inputs(images[:20])
    assert x.shape == (20, 8, 8, 1) and x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert np.allclose(normalize_pixels(np.array([0.0, 16.0])), [0.0, 1.0])


def test_augmentation_only_adds_shifted_copies(data):
    images, labels = data
    X, y = prepare_inputs(images[:50]), labels[:50]
    Xa, ya = augment_with_shifts(X, y, seed=1)
    assert Xa.shape == (100, 8, 8, 1) and np.array_equal(Xa[:50], X) and np.array_equal(ya[50:], y)
    assert Xa[50:].sum() <= X.sum() + 1e-3        # zero-padded shifts can only lose ink, never add
    assert Xa.min() >= 0.0 and Xa.max() <= 1.0


# ------------------------------------------- inference preprocessing (app) --
def _similarity(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_upload_preprocessing_output_format(data):
    images, _ = data
    x, preview, used = preprocess_uploaded_image(digit_array_to_pil(images[0]))
    assert x.shape == (1, 8, 8, 1) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0 and np.isclose(x.max(), 1.0)
    assert preview.shape == (8, 8) and used == "dark_on_light"


def test_polarity_auto_detection_does_not_blindly_invert(data):
    images, _ = data
    dark_on_light = digit_array_to_pil(images[5])
    light_on_dark = ImageOps.invert(dark_on_light)
    x1, _, p1 = preprocess_uploaded_image(dark_on_light)
    x2, _, p2 = preprocess_uploaded_image(light_on_dark)
    assert (p1, p2) == ("dark_on_light", "light_on_dark")
    assert np.allclose(x1, x2, atol=0.05)           # both end up as the same bright-ink image


def test_upload_preprocessing_recovers_dataset_digit(data):
    """A dataset digit upscaled to 256 px should come back close to itself."""
    images, _ = data
    sims = [_similarity(preprocess_uploaded_image(digit_array_to_pil(images[i]))[1], images[i] / 16)
            for i in range(0, 200, 10)]
    assert np.mean(sims) > 0.7


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P"])
def test_other_image_modes_work(data, mode):
    images, _ = data
    img = digit_array_to_pil(images[3]).convert("RGB")
    x, _, _ = preprocess_uploaded_image(img.convert(mode))
    assert x.shape == (1, 8, 8, 1)


def test_blank_image_is_rejected():
    with pytest.raises(ValueError):
        preprocess_uploaded_image(Image.new("L", (200, 200), 255))


def test_invalid_polarity_rejected(data):
    with pytest.raises(ValueError):
        preprocess_uploaded_image(digit_array_to_pil(data[0][0]), polarity="sideways")


class _StubModel:
    """Stands in for the CNN so the app's inference path is tested without TensorFlow."""

    def predict(self, x, verbose=0):
        out = np.zeros((len(x), 10), dtype="float32")
        out[:, 4] = 0.7
        out[:, 9] = 0.3
        return out


def test_inference_path_with_stub_model(data):
    from predict import predict_image

    res = predict_image(_StubModel(), digit_array_to_pil(data[0][4]), threshold=0.8)
    assert res["digit"] == 4 and np.isclose(res["confidence"], 0.7)
    assert res["low_confidence"] is True and res["second_choice"] == 9
    assert res["preview"].shape == (8, 8)


# ----------------------------------------------------- model tests (need TF) --
@pytest.fixture(scope="module")
def tiny_model():
    pytest.importorskip("tensorflow")
    from src.model import build_cnn

    return build_cnn(C.ADJUSTED_CONFIG)


def test_model_shapes(tiny_model):
    assert tiny_model.input_shape == (None, 8, 8, 1)
    assert tiny_model.output_shape == (None, 10)


def test_model_predicts_valid_probabilities(tiny_model, data):
    probs = tiny_model.predict(prepare_inputs(data[0][:16]), verbose=0)
    assert probs.shape == (16, 10)
    assert np.all(probs >= 0) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert set(np.argmax(probs, axis=1)) <= set(range(10))


def test_short_training_run_learns_something(data):
    """Smoke test of the real training function (3 epochs on the real split)."""
    pytest.importorskip("tensorflow")
    from src.data_loader import get_splits
    from src.training import train_one

    cfg = dict(C.BASELINE_CONFIG, epochs=3)
    model, hist = train_one(cfg, get_splits(C.SEED), verbose=0)
    assert len(hist["loss"]) == 3
    assert hist["loss"][-1] < hist["loss"][0]
    assert hist["val_accuracy"][-1] > 0.3        # far above the 10 % chance level


def test_save_and_load_roundtrip(tiny_model, data, tmp_path):
    import keras

    path = tmp_path / "m.keras"
    tiny_model.save(path)
    loaded = keras.models.load_model(path)
    x = prepare_inputs(data[0][:8])
    assert np.allclose(tiny_model.predict(x, verbose=0), loaded.predict(x, verbose=0), atol=1e-5)


def test_inference_pipeline_end_to_end(tiny_model, data):
    from predict import predict_image

    res = predict_image(tiny_model, digit_array_to_pil(data[0][7]))
    assert 0 <= res["digit"] <= 9
    assert 0.0 <= res["confidence"] <= 1.0
    assert np.isclose(sum(res["probabilities"]), 1.0, atol=1e-4)


def test_shipped_model_is_accurate_on_test_split():
    """Only runs after `python train.py` has produced models/digit_cnn_model.keras."""
    pytest.importorskip("tensorflow")
    if not C.MODEL_PATH.exists():
        pytest.skip("Run `python train.py` first to create the saved model.")
    from predict import load_trained_model
    from src.data_loader import get_splits

    s = get_splits(C.SEED)
    pred = np.argmax(load_trained_model().predict(s["X_test"], verbose=0), axis=1)
    assert (pred == s["y_test"]).mean() > 0.90
