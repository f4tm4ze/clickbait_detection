import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import (  # noqa: E402
    MAX_CHARS,
    binary_metrics,
    confidence_of,
    explain,
    label_of,
    parse_label,
    preprocess_text,
    style_cues,
    validate_headline,
)


# ---------------------------------------------------------------- preprocess_text
def test_preprocess_maps_exclamation_and_question():
    assert preprocess_text("Wow! Benarkah?") == "wow tanda seru benarkah tanda tanya"


def test_preprocess_removes_symbols_and_extra_spaces():
    assert preprocess_text("Harga BBM   naik, 5%!") == "harga bbm naik 5 tanda seru"


def test_preprocess_empty():
    assert preprocess_text("") == ""


# ------------------------------------------------------------- validate_headline
@pytest.mark.parametrize("raw", ["", "   ", None])
def test_validate_rejects_empty(raw):
    assert not validate_headline(raw).ok


def test_validate_rejects_symbols_only():
    v = validate_headline("!!! ???")
    assert not v.ok and "letters or numbers" in v.error


def test_validate_rejects_single_word():
    assert not validate_headline("Halo").ok


def test_validate_accepts_normal_headline():
    v = validate_headline("Pemerintah umumkan kebijakan baru hari ini")
    assert v.ok and v.warnings == []


def test_validate_warns_on_short_headline():
    v = validate_headline("Harga naik lagi")
    assert v.ok and len(v.warnings) == 1


def test_validate_trims_long_input():
    v = validate_headline("kata " * 400)
    assert v.ok
    assert len(v.text) == MAX_CHARS
    assert any("trimmed" in w for w in v.warnings)


def test_validate_warns_on_non_latin():
    v = validate_headline("Привет мир сегодня")
    assert v.ok and any("non-Latin" in w for w in v.warnings)


# ---------------------------------------------------------------- binary_metrics
def test_metrics_mixed():
    m = binary_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m == {"Accuracy": 0.5, "Precision": 0.5, "Recall": 0.5, "F1": 0.5}


def test_metrics_perfect():
    m = binary_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert all(v == 1.0 for v in m.values())


def test_metrics_no_positive_predictions_does_not_divide_by_zero():
    m = binary_metrics([1, 1], [0, 0])
    assert m["Precision"] == 0.0 and m["Recall"] == 0.0 and m["F1"] == 0.0


# ------------------------------------------------------------------- small helpers
def test_label_and_confidence():
    assert label_of(0.5, 0.5) == 1
    assert label_of(0.49, 0.5) == 0
    assert confidence_of(0.3, 0.5) == pytest.approx(0.7)
    assert confidence_of(0.9, 0.5) == pytest.approx(0.9)


def test_parse_label():
    assert parse_label("1") == 1
    assert parse_label("Clickbait") == 1
    assert parse_label("non-clickbait") == 0
    assert parse_label("maybe") is None


# ------------------------------------------------------------ cues and explanation
def test_cues_detected():
    names = [n for n, _ in style_cues("7 Cara Hemat Uang!")]
    assert "Starts with a number" in names and "Exclamation mark" in names


def test_no_cues_on_plain_headline():
    assert style_cues("Pakistan Didorong Lakukan Reformasi Pendidikan sebagai Penentu Arah Jangka Panjang") == []


def test_short_acronym_is_not_shouting():
    assert style_cues("Harga BBM naik") == []


def test_explain_clickbait_without_cues_mentions_overall_style():
    summary, bullets = explain([], 1, 0.9)
    assert "flagged" in summary
    assert any("overall style" in b for b in bullets)


def test_explain_not_clickbait_without_cues_says_direct():
    summary, bullets = explain([], 0, 0.95)
    assert "normal news" in summary
    assert any("directly" in b for b in bullets)


def test_explain_low_confidence_and_disagreement():
    _, bullets = explain([], 0, 0.55, models_agree=False)
    assert any("not very sure" in b for b in bullets)
    assert any("different answers" in b for b in bullets)