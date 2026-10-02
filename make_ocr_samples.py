"""Create synthetic test pages WITH known ground-truth text for OCR evaluation.

Usage: python make_ocr_samples.py
Writes tests/ocr_samples/<name>.png and <name>.txt (the exact text on the page).
These are PRINTED-style pages (clean, then progressively degraded). For handwriting,
add your own photos + a matching .txt file to the same folder (see README).
"""
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent / "tests" / "ocr_samples"
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"   # ships with matplotlib

PAGES = {
    "report_numbers": [
        "Monthly Sales Report 2026",
        "Region North: 1,250 units",
        "Region South: 980 units",
        "Region East: 2,415 units",
        "Total revenue: $48,720.50",
        "Growth rate: 12.5%",
    ],
    "paragraph": [
        "The committee met on 14 March 2026 to review",
        "the annual budget. Members agreed that spending",
        "on training should increase by 8 percent, and that",
        "a final report will be submitted next quarter.",
    ],
    "invoice_mixed": [
        "INVOICE No. 004871",
        "Date: 02/10/2026",
        "Item: Laptop charger x3 = 75.00",
        "Item: USB cable x10 = 32.50",
        "TOTAL DUE: 107.50 (paid by cash)",
    ],
}


def render(lines, size=34, noise=0.0, blur=0.0, angle=0.0, seed=0):
    font = ImageFont.truetype(str(FONT), size)
    w, h = 1100, 120 + int(size * 1.9) * len(lines)
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((60, 60 + i * int(size * 1.9)), line, fill=20, font=font)
    if angle:
        img = img.rotate(angle, expand=False, fillcolor=255, resample=Image.Resampling.BICUBIC)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        arr = np.asarray(img, dtype=np.float32)
        arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    variants = {"": {}, "_degraded": {"noise": 18, "blur": 0.8, "angle": 1.5}}
    for name, lines in PAGES.items():
        for suffix, kw in variants.items():
            render(lines, **kw).save(OUT / f"{name}{suffix}.png")
            (OUT / f"{name}{suffix}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Created {2 * len(PAGES)} sample pages in {OUT}")


if __name__ == "__main__":
    main()
