"""Document mode: read a whole page (printed + handwritten) with a PRETRAINED OCR model.

Uses EasyOCR (text detection + recognition). No model is trained here; the
CNN digit classifier elsewhere in this project is unchanged.

CLI test:  python -m src.document_ocr path/to/page.png   (or a .pdf)
"""
import io
import sys

import numpy as np
from PIL import Image, ImageOps

_reader = None
MAX_SIDE = 2000          # larger pages are shrunk for speed
LOW_CONFIDENCE = 0.50    # lines below this are flagged for human checking


def get_reader(languages=("en",), gpu=False):
    """Load the OCR model once (first call downloads the weights)."""
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
    return _reader


def load_pages(file_bytes: bytes, filename: str):
    """PNG/JPG -> [image]; PDF -> one image per page."""
    if filename.lower().endswith(".pdf"):
        import fitz  # PyMuPDF

        pages = []
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        return pages
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes)))
    return [img.convert("RGB")]


def group_into_lines(results):
    """EasyOCR gives separate boxes; merge boxes on the same row into text lines."""
    items = []
    for bbox, text, conf in results:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({"text": text, "conf": float(conf), "x": float(min(xs)),
                      "yc": (min(ys) + max(ys)) / 2.0, "h": float(max(ys) - min(ys))})
    if not items:
        return []
    items.sort(key=lambda d: d["yc"])
    med_h = max(float(np.median([d["h"] for d in items])), 1.0)
    rows = [[items[0]]]
    for it in items[1:]:
        ref = float(np.mean([d["yc"] for d in rows[-1]]))
        if abs(it["yc"] - ref) <= 0.6 * med_h:
            rows[-1].append(it)
        else:
            rows.append([it])
    lines = []
    for row in rows:
        row.sort(key=lambda d: d["x"])
        conf = float(np.mean([d["conf"] for d in row]))
        lines.append({"text": " ".join(d["text"] for d in row),
                      "confidence": conf, "low_confidence": conf < LOW_CONFIDENCE})
    return lines


def read_page(image: Image.Image):
    """One page image -> list of {text, confidence, low_confidence} (top to bottom)."""
    image = image.convert("RGB")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    results = get_reader().readtext(np.array(image))
    return group_into_lines(results)


def read_document(file_bytes: bytes, filename: str):
    """File -> list of pages, each a list of lines."""
    return [read_page(p) for p in load_pages(file_bytes, filename)]


def pages_to_text(pages) -> str:
    out = []
    for i, lines in enumerate(pages, 1):
        if len(pages) > 1:
            out.append(f"--- Page {i} ---")
        out.extend(l["text"] for l in lines)
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.document_ocr <image-or-pdf>")
        sys.exit(1)
    with open(sys.argv[1], "rb") as f:
        pages = read_document(f.read(), sys.argv[1])
    for i, lines in enumerate(pages, 1):
        print(f"\n===== Page {i} =====")
        for l in lines:
            flag = "  <-- CHECK" if l["low_confidence"] else ""
            print(f"[{l['confidence']:.0%}] {l['text']}{flag}")
