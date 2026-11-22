"""Pure logic for the Clickbait Headline Checker.

No Streamlit, TensorFlow or PyTorch imports here, so everything in this file
can be unit-tested quickly (see tests/test_core.py).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_THRESHOLD = 0.50     # decision threshold used during evaluation
BORDERLINE_MARGIN = 0.15     # |p - threshold| below this => "borderline"
MIN_WORDS = 2                # hard minimum for a valid headline
SHORT_WORDS = 4              # below this we warn the result may be unreliable
MAX_CHARS = 500              # input cap (text area)


# =============================================================================
# Text validation & preprocessing
# =============================================================================

def preprocess_text(text: str) -> str:
    """Identical to the preprocessing used in the training pipeline."""
    text = str(text).lower()
    text = text.replace("!", " tanda seru ")
    text = text.replace("?", " tanda tanya ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_input(text: str) -> str:
    """Normalise unicode, drop control characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Validation:
    ok: bool
    text: str = ""
    error: str = ""
    warnings: List[str] = field(default_factory=list)


def validate_headline(raw: str) -> Validation:
    """Validate user input and return actionable messages (never raises)."""
    text = clean_input(raw)
    if not text:
        return Validation(False, error="Please enter a headline to check.")
    if not re.search(r"[^\W_]", text):
        return Validation(False, error="The text has no letters or numbers. Please enter a real headline.")

    words = preprocess_text(text).split()
    if len(words) < MIN_WORDS:
        return Validation(False, error=f"A headline needs at least {MIN_WORDS} words to be classified.")

    warnings: List[str] = []
    if len(words) < SHORT_WORDS:
        warnings.append("Very short headline. Predictions on short text are less reliable.")
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        warnings.append(f"Input was trimmed to {MAX_CHARS} characters.")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < 0.6:
        warnings.append("Mostly non-Latin script detected. The models were trained on Latin-script Indonesian news.")
    if len(words) > 40:
        warnings.append("Long input. Only the first 128 tokens are analysed.")
    return Validation(True, text=text, warnings=warnings)


# =============================================================================
# Prediction helpers
# =============================================================================

def label_of(prob: float, threshold: float) -> int:
    return int(prob >= threshold)


def confidence_of(prob: float, threshold: float) -> float:
    """Confidence in the predicted class (0.5 = coin flip)."""
    return prob if prob >= threshold else 1.0 - prob


def is_borderline(prob: float, threshold: float) -> bool:
    return abs(prob - threshold) < BORDERLINE_MARGIN


def parse_label(value) -> Optional[int]:
    s = str(value).strip().lower()
    if s in {"1", "1.0", "clickbait", "true", "yes", "click"}:
        return 1
    if s in {"0", "0.0", "non-clickbait", "non clickbait", "no-clickbait", "false", "no", "news", "non"}:
        return 0
    return None


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    import numpy as np  # local import keeps module import light

    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    acc = (tp + tn) / max(len(yt), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


# =============================================================================
# Plain-language explanation (for non-technical users)
# =============================================================================

# (name, test(raw, words) -> bool, plain-language reason)
CUE_RULES = [
    ("Exclamation mark", lambda raw, w: "!" in raw,
     "It uses an exclamation mark to sound shocking or urgent."),
    ("Question headline", lambda raw, w: "?" in raw,
     "It is phrased as a question, which makes you curious enough to click."),
    ("Starts with a number", lambda raw, w: bool(re.match(r"^\s*\d+", raw)),
     "It starts with a number, like a list article (e.g. “7 Cara…”)."),
    ("Addresses the reader", lambda raw, w: bool({"kamu", "anda", "kalian", "you", "your"} & set(w)),
     "It talks directly to you (“kamu”, “Anda”)."),
    ("Curiosity-gap wording", lambda raw, w: bool(
        {"ternyata", "inilah", "rahasia", "viral", "heboh", "wow", "mengejutkan", "shocking"} & set(w)
        or "ini dia" in " ".join(w) or "ini alasan" in " ".join(w) or "bikin" in w),
     "It hints at a secret or surprise without telling you what it is."),
    ("Superlative", lambda raw, w: bool({"paling", "terbaik", "terburuk", "termahal", "tercepat"} & set(w)),
     "It makes an extreme claim (“paling”, “terbaik”)."),
    ("ALL-CAPS words", lambda raw, w: any(len(t) > 3 and t.isupper() for t in raw.split()),
     "It uses ALL-CAPS words to grab attention."),
]


def style_cues(raw: str) -> List[Tuple[str, str]]:
    """Heuristic wording cues for explanation only; NOT used by the classifiers."""
    words = preprocess_text(raw).split()
    return [(name, why) for name, test, why in CUE_RULES if test(raw, words)]


def explain(cues: Sequence[Tuple[str, str]], label: int, conf: float,
            models_agree: Optional[bool] = None) -> Tuple[str, List[str]]:
    """Return (summary sentence, bullet reasons) in plain language."""
    reasons = [why for _, why in cues]
    if label:
        summary = ("This headline was flagged because it looks like headlines that try to win clicks "
                   "instead of simply informing the reader.")
        if not cues:
            reasons.append("No single trick word was found, but its overall style matches the clickbait "
                           "examples the model learned from.")
    else:
        summary = ("This headline reads like a normal news report: it tells you what happened "
                   "instead of teasing you.")
        if cues:
            reasons.insert(0, "It has some clickbait-style wording (listed below), "
                              "but the model still judged the whole headline as normal news.")
        else:
            reasons.append("It states the facts directly. No shouting, teasing, questions, "
                           "or talking to the reader.")
    if conf < 0.65:
        reasons.append("The model is not very sure about this one. Please double-check the headline yourself.")
    if models_agree is True:
        reasons.append("Both models gave the same answer.")
    elif models_agree is False:
        reasons.append("The two models gave different answers, which is another reason to double-check.")
    return summary, reasons