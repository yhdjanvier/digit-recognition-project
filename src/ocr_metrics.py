"""OCR evaluation metrics: Character Error Rate (CER) and Word Error Rate (WER).

CER = (substitutions + deletions + insertions) / number of reference characters
WER = the same, counted on words. 0.0 = perfect; lower is better.
"""
import re


def normalize(text: str) -> str:
    """Collapse all whitespace so line-break differences are not counted as errors."""
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or word lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)
