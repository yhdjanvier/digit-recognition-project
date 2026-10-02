"""Data preprocessing for training AND inference.

Two pipelines live here, and they must stay consistent:

1. Training pipeline  - prepare_inputs(): normalise 0..16 -> 0..1 and add the
   channel axis. Optional shift augmentation for the training split only.
2. Inference pipeline - preprocess_uploaded_image(): turns an arbitrary
   PNG/JPG (phone photo, scanned page, drawing) into an 8x8 image that looks
   like the training data: bright ink (value 1) on a dark background (0), the
   digit cropped, aspect-ratio preserved, stroke width normalised and centred.
"""
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from config import IMG_HEIGHT, IMG_WIDTH, PIXEL_MAX

# ============================================================================
# 1. TRAINING PIPELINE
# ============================================================================


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """Scale pixel values from 0..16 to 0..1. A fixed constant is used (not
    statistics of the data), so there is no train/test leakage."""
    return (images / PIXEL_MAX).astype("float32")


def add_channel_dim(images: np.ndarray) -> np.ndarray:
    """(N, 8, 8) -> (N, 8, 8, 1): Conv2D expects a channel axis."""
    return images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)


def prepare_inputs(images_raw: np.ndarray) -> np.ndarray:
    """Raw (N, 8, 8) 0..16  ->  CNN input (N, 8, 8, 1) 0..1."""
    return add_channel_dim(normalize_pixels(images_raw))


def augment_with_shifts(X: np.ndarray, y: np.ndarray, seed: int, max_shift: int = 1):
    """Return the original images PLUS one randomly shifted copy of each.

    Shifts of up to `max_shift` pixels (zero padded) make the model less
    sensitive to digit position, which matters for uploaded images that are
    never perfectly centred. Apply to the TRAINING split only.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    padded = np.pad(X, ((0, 0), (max_shift, max_shift), (max_shift, max_shift), (0, 0)))
    shifted = np.empty_like(X)
    for i in range(n):
        dy, dx = rng.integers(-max_shift, max_shift + 1, size=2)
        top, left = max_shift + dy, max_shift + dx
        shifted[i] = padded[i, top : top + IMG_HEIGHT, left : left + IMG_WIDTH, :]
    return np.concatenate([X, shifted]), np.concatenate([y, y])


# ============================================================================
# 2. INFERENCE PIPELINE FOR UPLOADED IMAGES
# ============================================================================

FIT_SIZE = 8            # longest side of the cropped digit is scaled to 8 px (as in the dataset)
MIN_CONTRAST = 0.15     # below this, the image is considered blank / unusable
NOISE_FLOOR = 0.15      # normalised ink below this is treated as background
INK_THRESHOLD = 0.30    # normalised ink above this counts as "digit" for cropping
TARGET_STROKE_RATIO = 0.17   # typical stroke width / digit height in the dataset
VALID_POLARITIES = ("auto", "dark_on_light", "light_on_dark")


def _to_gray(image: Image.Image) -> np.ndarray:
    """Any PIL image -> float grayscale array in 0..1 (transparency -> white)."""
    image = ImageOps.exif_transpose(image)  # honour phone camera rotation
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    image = image.convert("L")
    if max(image.size) > 1024:  # keep processing fast for large photos
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def _border_pixels(arr: np.ndarray) -> np.ndarray:
    t = max(1, int(round(0.05 * min(arr.shape))))
    return np.concatenate(
        [arr[:t].ravel(), arr[-t:].ravel(), arr[:, :t].ravel(), arr[:, -t:].ravel()]
    )


def detect_polarity(gray: np.ndarray) -> str:
    """Decide whether the digit is dark on a light page or light on a dark one.

    The background dominates the image border, so the border median tells us
    which case we have. This is why we do NOT blindly invert every image.
    """
    return "dark_on_light" if float(np.median(_border_pixels(gray))) > 0.5 else "light_on_dark"


def _normalise_stroke_width(ink: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Make stroke thickness (relative to digit size) similar to the dataset.

    A thin pen line on a large photo would vanish when averaged down to 6x6,
    so thin strokes are thickened; extremely bold ones are thinned slightly.
    """
    side = max(ink.shape)
    dist = ndimage.distance_transform_edt(mask)
    width = 4.0 * float(dist[mask].mean())  # for a strip of width w, mean(EDT) ~ w/4
    target = TARGET_STROKE_RATIO * side
    if width < 0.75 * target:
        size = int(round(target - width)) + 1
        size = max(2, min(size, max(2, side // 6)))
        return ndimage.maximum_filter(ink, size=size)
    if width > 2.0 * target:
        size = int(round(width - 1.4 * target)) + 1
        size = max(2, min(size, max(2, side // 6)))
        return ndimage.minimum_filter(ink, size=size)
    return ink


def _center_by_mass(img: np.ndarray) -> np.ndarray:
    """Shift the digit (at most 1 px) so its centre of mass is the image centre."""
    if img.sum() <= 0:
        return img
    cy, cx = ndimage.center_of_mass(img)
    sy = int(np.clip(round(3.5 - cy), -1, 1))
    sx = int(np.clip(round(3.5 - cx), -1, 1))
    return ndimage.shift(img, (sy, sx), order=0, mode="constant", cval=0.0)


def preprocess_uploaded_image(
    image: Image.Image, polarity: str = "auto"
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Convert an uploaded image into model input.

    Steps: grayscale -> polarity detection/inversion -> background removal ->
    contrast stretch -> noise suppression -> crop to the digit -> stroke-width
    normalisation -> scale the longest side to 8 px keeping the aspect ratio
    (a thin "1" stays thin) -> place on an 8x8 canvas -> centre by mass.

    Returns (model_input (1,8,8,1) float32 in 0..1, preview (8,8), polarity_used).
    Raises ValueError if no clear digit can be found.
    """
    if polarity not in VALID_POLARITIES:
        raise ValueError(f"polarity must be one of {VALID_POLARITIES}")

    gray = _to_gray(image)
    used = detect_polarity(gray) if polarity == "auto" else polarity

    ink = 1.0 - gray if used == "dark_on_light" else gray.copy()  # ink = bright
    ink = np.clip(ink - float(np.median(_border_pixels(ink))), 0.0, None)
    hi = float(np.percentile(ink, 99.5))
    if hi < MIN_CONTRAST:
        raise ValueError(
            "No clear digit found: the image has too little contrast. "
            "Use a dark pen on plain light paper (or the reverse) and try again."
        )
    ink = np.clip(ink / hi, 0.0, 1.0)
    ink[ink < NOISE_FLOOR] = 0.0

    mask = ink >= INK_THRESHOLD
    if mask.sum() < max(10, 0.001 * mask.size):
        raise ValueError("No clear digit found: could not locate any written strokes.")

    ys, xs = np.nonzero(mask)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    crop, crop_mask = ink[sl], mask[sl]

    crop = _normalise_stroke_width(crop, crop_mask)

    h, w = crop.shape
    scale = FIT_SIZE / max(h, w)
    new_h = int(np.clip(round(h * scale), 2, FIT_SIZE))
    new_w = int(np.clip(round(w * scale), 2, FIT_SIZE))
    small = np.asarray(
        Image.fromarray((np.clip(crop, 0, 1) * 255).round().astype(np.uint8)).resize(
            (new_w, new_h), Image.Resampling.BOX
        ),
        dtype=np.float32,
    ) / 255.0
    peak = float(small.max())
    if peak <= 0:
        raise ValueError("No clear digit found after resizing.")
    small /= peak  # dataset digits reach (almost) full intensity

    canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    top, left = (IMG_HEIGHT - new_h) // 2, (IMG_WIDTH - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = small
    canvas = _center_by_mass(canvas).astype("float32")

    return canvas.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1), canvas, used


def digit_array_to_pil(digit_8x8_raw: np.ndarray, size: int = 256) -> Image.Image:
    """Render a raw dataset digit (0..16) as a black-ink-on-white PNG-like image.
    Used for the demo 'dataset sample' option and in tests."""
    arr = 255.0 - (np.asarray(digit_8x8_raw, dtype=np.float32) / PIXEL_MAX) * 255.0
    img = Image.fromarray(arr.round().astype(np.uint8), mode="L")
    return img.resize((size, size), Image.Resampling.BICUBIC)
