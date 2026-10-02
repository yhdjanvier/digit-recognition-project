"""Tests for Document mode that run WITHOUT downloading any OCR model."""
import io

import numpy as np
import pytest
from PIL import Image

import src.document_ocr as D
from src.ocr_metrics import cer, edit_distance, wer


def test_edit_distance_and_rates():
    assert edit_distance("kitten", "sitting") == 3
    assert cer("hello world", "hello world") == 0.0
    assert np.isclose(cer("abcd", "abxd"), 0.25)
    assert np.isclose(wer("the cat sat", "the dog sat"), 1 / 3)
    assert cer("a   b\n c", "a b c") == 0.0          # whitespace differences are ignored
    assert cer("", "") == 0.0 and cer("", "x") == 1.0


class _StubReader:
    def readtext(self, arr):
        box = lambda x, y: [[x, y], [x + 60, y], [x + 60, y + 20], [x, y + 20]]
        return [(box(150, 10), "world", 0.9), (box(10, 10), "Hello", 0.95), (box(10, 80), "42", 0.3)]


def test_read_page_with_stub_reader(monkeypatch):
    monkeypatch.setattr(D, "_reader", _StubReader())
    lines = D.read_page(Image.new("RGB", (300, 200), "white"))
    assert [l["text"] for l in lines] == ["Hello world", "42"]
    assert lines[1]["low_confidence"] is True and lines[0]["low_confidence"] is False
    assert D.pages_to_text([lines]) == "Hello world\n42"


def test_load_pages_image_and_pdf():
    buf = io.BytesIO()
    Image.new("RGB", (50, 50), "white").save(buf, "PNG")
    assert len(D.load_pages(buf.getvalue(), "a.png")) == 1
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page(); doc.new_page()
    assert len(D.load_pages(doc.tobytes(), "two_pages.pdf")) == 2
