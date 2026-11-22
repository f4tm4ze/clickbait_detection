from __future__ import annotations

import html as _html
import io
import logging
import os
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Heavy imports are wrapped so the UI can still start and explain what is
# missing instead of crashing with a traceback.
try:
    import tensorflow as tf
    import torch
    from transformers import AutoModel, AutoTokenizer

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    tf = torch = AutoModel = AutoTokenizer = None
    IMPORT_ERROR = str(exc)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clickbait-checker")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Clickbait Headline Checker",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(BASE_DIR, "deployment_baseline")
PROPOSED_DIR = os.path.join(BASE_DIR, "deployment_proposed")

BASELINE_MODEL_FILE = "baseline_bilstm.keras"
BASELINE_TOKENIZER_FILE = "baseline_tokenizer.pkl"
BASELINE_WORD_INDEX_FILE = "baseline_word_index.pkl"
BASELINE_CONFIG_FILE = "baseline_config.pkl"
PROPOSED_MODEL_FILE = "xlmr_bilstm_attention.keras"
PROPOSED_CONFIG_FILE = "xlmr_config.txt"

DEFAULT_THRESHOLD = 0.50     # decision threshold used during evaluation
BORDERLINE_MARGIN = 0.15     # |p - threshold| below this => "borderline"
MIN_WORDS = 2                # hard minimum for a valid headline
SHORT_WORDS = 4              # below this we warn the result may be unreliable
MAX_CHARS = 500              # input cap (text area)
MAX_BATCH_ROWS = 500         # batch-mode cap
ENCODER_BATCH_SIZE = 16      # XLM-R mini-batch (memory / speed trade-off)
HISTORY_LIMIT = 50

BENCHMARKS = [
    # name, accuracy, precision, recall, f1
    ("Baseline BiLSTM", 0.7461, 0.7030, 0.6830, 0.6929),
    ("BiLSTM + XLM-R", 0.7651, 0.7183, 0.7235, 0.7209),
    ("BiLSTM + XLM-R + Attention (proposed)", 0.7855, 0.7705, 0.6956, 0.7311),
]

# =============================================================================
# 2. TEXT VALIDATION & PREPROCESSING
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
# 3. MODEL LOADING (cached; each component fails independently)
# =============================================================================

def find_file(*candidates: str) -> Optional[str]:
    """Return the first existing path (also checks the app folder itself)."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class WordIndexTokenizer:
    """Minimal stand-in for keras Tokenizer when only word_index.pkl exists.

    Words outside the vocabulary are dropped (same as a Keras Tokenizer
    without an OOV token); ids >= vocab_limit are dropped to match the
    embedding layer size.
    """

    def __init__(self, word_index: Dict[str, int], vocab_limit: Optional[int] = None):
        self.word_index = word_index
        self.vocab_limit = vocab_limit

    def texts_to_sequences(self, texts: Sequence[str]) -> List[List[int]]:
        out = []
        for t in texts:
            seq = []
            for w in t.split():
                idx = self.word_index.get(w)
                if idx is not None and (self.vocab_limit is None or idx < self.vocab_limit):
                    seq.append(idx)
            out.append(seq)
        return out


def _read_kv_config(path: Optional[str]) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    cfg[k.strip()] = v.strip()
    return cfg


@st.cache_resource(show_spinner=False)
def load_system() -> dict:
    """Load all models once per server process. Failures are recorded, not raised."""
    t0 = time.perf_counter()
    system = {"baseline": None, "proposed": None, "errors": {}, "load_s": 0.0}

    if IMPORT_ERROR:
        system["errors"]["libraries"] = IMPORT_ERROR
        return system

    # ---- Baseline BiLSTM -------------------------------------------------
    try:
        model_path = find_file(
            os.path.join(BASELINE_DIR, BASELINE_MODEL_FILE), os.path.join(BASE_DIR, BASELINE_MODEL_FILE)
        )
        if not model_path:
            raise FileNotFoundError(f"{BASELINE_MODEL_FILE} not found in deployment_baseline/")
        model = tf.keras.models.load_model(model_path, compile=False)

        cfg_path = find_file(
            os.path.join(BASELINE_DIR, BASELINE_CONFIG_FILE), os.path.join(BASE_DIR, BASELINE_CONFIG_FILE)
        )
        config = {}
        if cfg_path:
            with open(cfg_path, "rb") as fh:
                config = pickle.load(fh)

        vocab_limit = None
        for layer in model.layers:
            if layer.__class__.__name__ == "Embedding":
                vocab_limit = int(layer.input_dim)
                break

        tok_path = find_file(
            os.path.join(BASELINE_DIR, BASELINE_TOKENIZER_FILE), os.path.join(BASE_DIR, BASELINE_TOKENIZER_FILE)
        )
        if tok_path:
            with open(tok_path, "rb") as fh:
                tokenizer = pickle.load(fh)
        else:
            wi_path = find_file(
                os.path.join(BASELINE_DIR, BASELINE_WORD_INDEX_FILE), os.path.join(BASE_DIR, BASELINE_WORD_INDEX_FILE)
            )
            if not wi_path:
                raise FileNotFoundError("Neither baseline_tokenizer.pkl nor baseline_word_index.pkl was found.")
            with open(wi_path, "rb") as fh:
                tokenizer = WordIndexTokenizer(pickle.load(fh), vocab_limit)

        max_len = int(config.get("max_length", model.input_shape[1] or 128))
        system["baseline"] = {"model": model, "tokenizer": tokenizer, "config": config, "max_len": max_len}
    except Exception as exc:
        log.exception("Baseline load failed")
        system["errors"]["baseline"] = str(exc)

    # ---- Proposed XLM-R + BiLSTM + Attention -----------------------------
    try:
        model_path = find_file(
            os.path.join(PROPOSED_DIR, PROPOSED_MODEL_FILE), os.path.join(BASE_DIR, PROPOSED_MODEL_FILE)
        )
        if not model_path:
            raise FileNotFoundError(f"{PROPOSED_MODEL_FILE} not found in deployment_proposed/")
        model = tf.keras.models.load_model(model_path, compile=False)
        config = _read_kv_config(
            find_file(os.path.join(PROPOSED_DIR, PROPOSED_CONFIG_FILE), os.path.join(BASE_DIR, PROPOSED_CONFIG_FILE))
        )
        name = config.get("MODEL_NAME", "xlm-roberta-base")
        local = os.path.join(BASE_DIR, name)          # optional offline copy
        source = local if os.path.isdir(local) else name
        tokenizer = AutoTokenizer.from_pretrained(source)
        encoder = AutoModel.from_pretrained(source)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        system["proposed"] = {
            "model": model,
            "tokenizer": tokenizer,
            "encoder": encoder,
            "config": config,
            "max_len": int(config.get("MAX_LENGTH", 128)),
        }
    except Exception as exc:
        log.exception("Proposed model load failed")
        system["errors"]["proposed"] = str(exc)

    # ---- Warm-up: first TF/torch call is slow; do it now so timings are fair
    try:
        if system["baseline"]:
            predict_baseline(["warm up"], system["baseline"])
        if system["proposed"]:
            predict_proposed(["warm up"], system["proposed"])
    except Exception:
        log.exception("Warm-up failed")

    system["load_s"] = time.perf_counter() - t0
    return system


# =============================================================================
# 4. INFERENCE
# =============================================================================

@dataclass
class Prediction:
    prob: float                       # P(clickbait)
    ms: float                         # per-headline latency (ms)
    coverage: Optional[float] = None  # baseline: share of words found in vocabulary
    encoder_ms: Optional[float] = None
    head_ms: Optional[float] = None


def predict_baseline(texts: Sequence[str], bl: dict) -> List[Prediction]:
    """Baseline BiLSTM on a batch of raw headlines."""
    t0 = time.perf_counter()
    processed = [preprocess_text(t) for t in texts]
    seqs = bl["tokenizer"].texts_to_sequences(processed)
    padded = tf.keras.utils.pad_sequences(seqs, maxlen=bl["max_len"], padding="post", truncating="post")
    probs = np.asarray(bl["model"](padded, training=False)).reshape(-1)
    per = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
    out = []
    for p, seq, proc in zip(probs, seqs, processed):
        total = len(proc.split())
        out.append(Prediction(float(p), per, coverage=(len(seq) / total if total else 0.0)))
    return out


def encode_xlmr(texts: Sequence[str], px: dict) -> np.ndarray:
    """Frozen XLM-R last_hidden_state, shape (n, max_len, 768)."""
    enc = px["tokenizer"](
        list(texts), padding="max_length", truncation=True, max_length=px["max_len"], return_tensors="pt"
    )
    with torch.inference_mode():
        out = px["encoder"](input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return out.last_hidden_state.cpu().numpy().astype(np.float32)


def predict_proposed(texts: Sequence[str], px: dict, progress=None) -> List[Prediction]:
    """XLM-R + BiLSTM + Attention on a batch, processed in mini-batches."""
    processed = [preprocess_text(t) for t in texts]
    results: List[Prediction] = []
    for start in range(0, len(processed), ENCODER_BATCH_SIZE):
        chunk = processed[start:start + ENCODER_BATCH_SIZE]
        t0 = time.perf_counter()
        emb = encode_xlmr(chunk, px)
        t1 = time.perf_counter()
        probs = np.asarray(px["model"](emb, training=False)).reshape(-1)
        t2 = time.perf_counter()
        n = len(chunk)
        for p in probs:
            results.append(Prediction(float(p), (t2 - t0) * 1000 / n,
                                      encoder_ms=(t1 - t0) * 1000 / n, head_ms=(t2 - t1) * 1000 / n))
        if progress:
            progress(min(start + n, len(processed)) / len(processed))
    return results


# =============================================================================
# 5. ANALYSIS HELPERS
# =============================================================================

def label_of(prob: float, threshold: float) -> int:
    return int(prob >= threshold)


def confidence_of(prob: float, threshold: float) -> float:
    """Confidence in the predicted class (0.5 = coin flip)."""
    return prob if prob >= threshold else 1.0 - prob


def is_borderline(prob: float, threshold: float) -> bool:
    return abs(prob - threshold) < BORDERLINE_MARGIN


CUE_RULES = [
    ("Exclamation mark", lambda raw, w: "!" in raw, "Emphasis punctuation is common in sensational headlines."),
    ("Question headline", lambda raw, w: "?" in raw, "Questions can create a curiosity gap."),
    ("Starts with a number", lambda raw, w: bool(re.match(r"^\s*\d+", raw)), "Listicle pattern (e.g. “7 Cara…”)."),
    ("Addresses the reader", lambda raw, w: bool({"kamu", "anda", "kalian", "you", "your"} & set(w)),
     "Direct address (“kamu”, “Anda”)."),
    ("Curiosity-gap wording", lambda raw, w: bool(
        {"ternyata", "inilah", "rahasia", "viral", "heboh", "wow", "mengejutkan", "shocking"} & set(w)
        or "ini dia" in " ".join(w) or "ini alasan" in " ".join(w) or "bikin" in w),
     "Teases information instead of stating it."),
    ("Superlative", lambda raw, w: bool({"paling", "terbaik", "terburuk", "termahal", "tercepat"} & set(w)),
     "Extreme claims attract clicks."),
    ("ALL-CAPS words", lambda raw, w: len([t for t in raw.split() if len(t) > 2 and t.isupper()]) >= 1,
     "Capitalised words signal urgency."),
]


def style_cues(raw: str) -> List[Tuple[str, str]]:
    """Heuristic wording cues for explanation only; NOT used by the classifiers."""
    words = preprocess_text(raw).split()
    return [(name, why) for name, test, why in CUE_RULES if test(raw, words)]


def parse_label(value) -> Optional[int]:
    s = str(value).strip().lower()
    if s in {"1", "1.0", "clickbait", "true", "yes", "click"}:
        return 1
    if s in {"0", "0.0", "non-clickbait", "non clickbait", "no-clickbait", "false", "no", "news", "non"}:
        return 0
    return None


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
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
# 6. UI COMPONENTS
# =============================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
:root{
  --bg:#FAFAFA; --card:#FFFFFF; --soft:#F4F4F5; --ink:#111111; --muted:#6B7280; --line:#E4E4E7; --line-strong:#D4D4D8;
  --red:#DC2626; --red-soft:#FEF2F2; --red-line:#FECACA;
  --green:#16A34A; --green-soft:#F0FDF4; --green-line:#BBF7D0;
  --amber:#B45309; --amber-soft:#FFFBEB;
  --shadow:0 1px 2px rgba(0,0,0,.04);
}
:root:has(.theme-marker.dark){
  --bg:#0A0A0A; --card:#111111; --soft:#18181B; --ink:#F4F4F5; --muted:#A1A1AA; --line:#27272A; --line-strong:#3F3F46;
  --red:#F87171; --red-soft:#1F1111; --red-line:#5B2020;
  --green:#4ADE80; --green-soft:#0D1A12; --green-line:#1F4D31;
  --amber:#FBBF24; --amber-soft:#1F1A0D; --shadow:none;
}
html, body, .stApp{background:var(--bg)!important;font-family:'Inter',sans-serif;color:var(--ink);}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stHeader"] *,[data-testid="stToolbar"] *{color:var(--ink)!important;}
.block-container{max-width:960px;padding-top:1.5rem!important;padding-bottom:2.5rem!important;}
[data-testid="stSidebar"]{background:var(--card)!important;border-right:1px solid var(--line);}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp li,.stApp label,
[data-testid="stWidgetLabel"] p,[data-testid="stMetricLabel"] *,[data-testid="stMetricValue"] *,
[data-testid="stSlider"] *,[data-testid="stExpander"] summary *,[data-testid="stFileUploader"] *{color:var(--ink)!important;}
[data-testid="stCaptionContainer"],.stApp .muted{color:var(--muted)!important;}

/* header */
.page-header{padding:.2rem 0 1.1rem;margin-bottom:1rem;border-bottom:1px solid var(--line);}
.page-title{font-size:1.7rem;font-weight:700;letter-spacing:-.02em;color:var(--ink);line-height:1.2;}
.page-sub{font-size:.95rem;color:var(--muted);margin-top:.25rem;}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.9rem 1.2rem;background:var(--card);
  border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);margin-bottom:.7rem;flex-wrap:wrap;}
.brand{display:flex;align-items:center;gap:.8rem;}
.logo{width:34px;height:34px;border-radius:8px;background:var(--ink);color:var(--card);display:flex;align-items:center;justify-content:center;
  font-weight:700;font-size:.85rem;letter-spacing:.02em;}
.brand-title{font-weight:600;font-size:1.02rem;line-height:1.25;color:var(--ink);letter-spacing:-.01em;}
.brand-sub{font-size:.8rem;color:var(--muted);}
.status{display:inline-flex;align-items:center;gap:.45rem;font-size:.78rem;color:var(--muted);border:1px solid var(--line);
  border-radius:99px;padding:.25rem .7rem;}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);} .dot.off{background:var(--red);}

/* cards */
.card,.st-key-inputcard{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:1.2rem 1.3rem;margin-bottom:1rem;
  box-shadow:var(--shadow);color:var(--ink);}
.card-title{font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)!important;margin-bottom:.7rem;}
.muted{color:var(--muted);font-size:.88rem;}

/* inputs */
[data-testid="stTextArea"] textarea{background:var(--card)!important;color:var(--ink)!important;border:1px solid var(--line-strong)!important;
  border-radius:10px!important;font-size:1.02rem!important;line-height:1.55!important;padding:.85rem 1rem!important;}
[data-testid="stTextArea"] textarea::placeholder{color:var(--muted)!important;opacity:.85;}
[data-testid="stTextArea"] textarea:focus{border-color:var(--ink)!important;box-shadow:0 0 0 3px color-mix(in srgb,var(--ink) 10%,transparent)!important;}
[data-testid="stTextArea"] [data-baseweb="textarea"],[data-testid="stTextArea"] [data-baseweb="base-input"]{background:transparent!important;border:none!important;}
[data-testid="stFileUploaderDropzone"]{background:var(--soft)!important;border:1px dashed var(--line-strong)!important;border-radius:10px!important;}
[data-testid="stFileUploaderDropzone"] button{background:var(--card)!important;border:1px solid var(--line-strong)!important;border-radius:10px!important;}
[data-testid="stFileUploaderDropzone"] button,[data-testid="stFileUploaderDropzone"] button *{color:var(--ink)!important;}
[data-testid="stFileUploaderDropzoneInstructions"] *{color:var(--muted)!important;}
[data-testid="stFileUploaderFile"],[data-testid^="stFileChip"]:not([data-testid*="Name"]){background:var(--soft)!important;border:1px solid var(--line-strong)!important;border-radius:8px!important;}
[data-testid="stFileUploaderFile"] *,[data-testid^="stFileChip"] *,[data-testid^="stFileUploaderFile"] *{color:var(--ink)!important;}
[data-testid^="stFileChip"] button,[data-testid="stFileUploaderDeleteBtn"] button{background:transparent!important;border:none!important;}
[data-testid="stExpander"]{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:12px!important;}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem;}

/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:10px!important;font-weight:500!important;padding:.55rem 1rem!important;transition:background .12s ease,border-color .12s ease!important;}
.stButton>button[kind="primary"],.stDownloadButton>button{background:var(--ink)!important;color:var(--card)!important;border:1px solid var(--ink)!important;}
.stButton>button[kind="primary"] *,.stDownloadButton>button *{color:var(--card)!important;}
.stButton>button[kind="primary"]:hover:not(:disabled),.stDownloadButton>button:hover{opacity:.88;}
.stButton>button[kind="primary"]:disabled{opacity:.55!important;cursor:not-allowed!important;}
.stButton>button[kind="secondary"]{background:var(--card)!important;color:var(--ink)!important;border:1px solid var(--line-strong)!important;}
.stButton>button[kind="secondary"] *{color:var(--ink)!important;}
.stButton>button[kind="secondary"]:hover:not(:disabled){background:var(--soft)!important;}
.stButton>button:disabled{cursor:not-allowed!important;}

/* tabs */
.stTabs [data-baseweb="tab-list"]{gap:1.4rem;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{padding:.5rem 0;height:auto;background:transparent!important;}
.stTabs [data-baseweb="tab"] *{color:var(--muted)!important;font-weight:500;}
.stTabs [aria-selected="true"] *{color:var(--ink)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--ink)!important;height:2px!important;}
.stTabs [data-baseweb="tab-border"]{display:none;}

/* RESULT (visual centrepiece) */
.result{border-radius:14px;padding:1.6rem 1.6rem 1.4rem;margin:.4rem 0 1rem;border:1px solid var(--line);}
.result.bad{background:var(--red-soft);border-color:var(--red-line);}
.result.good{background:var(--green-soft);border-color:var(--green-line);}
.result-head{display:flex;align-items:center;gap:1rem;}
.result-icon{flex-shrink:0;width:56px;height:56px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:var(--card);}
.result.bad .result-icon{color:var(--red);border:1px solid var(--red-line);}
.result.good .result-icon{color:var(--green);border:1px solid var(--green-line);}
.result-label{font-size:2.15rem;font-weight:700;letter-spacing:-.02em;line-height:1.1;}
.result.bad .result-label{color:var(--red);} .result.good .result-label{color:var(--green);}
.result-desc{font-size:.95rem;color:var(--muted);margin-top:.3rem;}
.quote{margin:1.1rem 0 1.2rem;padding:.7rem .95rem;background:var(--card);border:1px solid var(--line);border-radius:10px;font-size:.98rem;color:var(--ink);}
.conf-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem;}
.conf-row span{font-size:.85rem;font-weight:500;color:var(--muted);}
.conf-row b{font-size:1.5rem;font-weight:700;color:var(--ink);}
.track{height:10px;background:var(--line);border-radius:99px;overflow:hidden;}
.fill{height:100%;border-radius:99px;}
.result.bad .fill{background:var(--red);} .result.good .fill{background:var(--green);}
.conf-note{font-size:.78rem;color:var(--muted);margin-top:.55rem;}
.notice{border-radius:10px;padding:.65rem .9rem;font-size:.88rem;margin:.5rem 0;border:1px solid var(--line);background:var(--soft);color:var(--ink);}
.notice.warn{background:var(--amber-soft);color:var(--amber);border-color:transparent;}
.notice.bad{background:var(--red-soft);color:var(--red);border-color:var(--red-line);}

/* tables, chips, flow */
.tbl,.tbl tr,.tbl th,.tbl td{border-left:none!important;border-right:none!important;border-top:none!important;}
.tbl{width:100%;border-collapse:collapse;font-size:.88rem;border:none!important;}
.tbl th{text-align:left;color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.07em;font-weight:600;padding:.5rem .6rem;border-bottom:1px solid var(--line);}
.tbl td{padding:.65rem .6rem;border-bottom:1px solid var(--line);color:var(--ink);}
.tbl tr:last-child td{border-bottom:none;}
.tbl tr.best td{background:var(--soft);font-weight:600;}
.tag{display:inline-block;border-radius:6px;padding:.12rem .5rem;font-size:.72rem;font-weight:600;border:1px solid var(--line);color:var(--muted);}
.tag.bad{color:var(--red);background:var(--red-soft);border-color:var(--red-line);}
.tag.good{color:var(--green);background:var(--green-soft);border-color:var(--green-line);}
.chip{display:inline-block;background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:.25rem .6rem;margin:.15rem .25rem .15rem 0;font-size:.8rem;}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:.4rem;margin:.4rem 0 .9rem;}
.flow span.s{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:.3rem .6rem;font-size:.78rem;font-weight:500;color:var(--ink);}
.flow span.s.hl{background:var(--ink);color:var(--card);border-color:var(--ink);}
.flow i{color:var(--muted);font-style:normal;}
.mono{font-family:'JetBrains Mono',monospace;font-size:.82rem;}
.card code{background:var(--soft);padding:.1rem .35rem;border-radius:6px;color:var(--ink);}
.card li,.card p,.card b{color:var(--ink);}
.footer{text-align:center;color:var(--muted);font-size:.78rem;padding:1.4rem 0 .2rem;}
@media (max-width:700px){
  .result-label{font-size:1.5rem}.result-icon{width:44px;height:44px}.result{padding:1.2rem}
  .topbar{align-items:flex-start}
}
</style>
"""

ICON_WARN = ('<svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
             '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>')
ICON_OK = ('<svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" stroke-width="2" '
           'stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
           '<polyline points="22 4 12 14.01 9 11.01"/></svg>')

PROPOSED_FLOW = (
    '<div class="flow"><span class="s">XLM-R tokenizer</span><i>→</i><span class="s">Frozen XLM-R base (768-D)</span><i>→</i>'
    '<span class="s">BiLSTM (64)</span><i>→</i><span class="s">Self-attention</span><i>→</i><span class="s">Residual add</span><i>→</i>'
    '<span class="s">LayerNorm</span><i>→</i><span class="s">Dropout 0.2</span><i>→</i><span class="s">BatchNorm</span><i>→</i>'
    '<span class="s">Global max-pool</span><i>→</i><span class="s hl">Sigmoid</span></div>'
)
BASELINE_FLOW = (
    '<div class="flow"><span class="s">Tokenizer</span><i>→</i><span class="s">300-D trainable embedding</span><i>→</i>'
    '<span class="s">BiLSTM (64)</span><i>→</i><span class="s">Dropout 0.2</span><i>→</i><span class="s">BatchNorm</span><i>→</i>'
    '<span class="s">Global max-pool</span><i>→</i><span class="s hl">Sigmoid</span></div>'
)


def esc(s: str) -> str:
    return _html.escape(str(s), quote=True)


def md(*lines: str) -> None:
    """Render raw HTML. Lines are stripped and joined so Markdown never sees
    indentation or blank lines (which would break the HTML block)."""
    st.markdown("\n".join(l.strip() for l in lines if l and l.strip()), unsafe_allow_html=True)


def card(title: str, *inner: str) -> None:
    md(f'<div class="card"><div class="card-title">{esc(title)}</div>', *inner, "</div>")


def render_header() -> None:
    md('<div class="page-header"><div class="page-title">Clickbait Headline Detection</div>',
       '<div class="page-sub">News Headline Classification</div></div>')


def interpretation(conf: float, lab: int) -> str:
    level = "highly confident" if conf >= 0.85 else "moderately confident" if conf >= 0.65 else "only weakly confident"
    what = "clickbait" if lab else "not clickbait"
    tail = " Consider reviewing this headline manually." if conf < 0.65 else ""
    return f"The model is {level} that this headline is {what}.{tail}"


def render_result(res: dict, threshold: float) -> None:
    """Primary verdict, confidence, prediction details and technical details."""
    base: Optional[Prediction] = res.get("baseline")
    prop: Optional[Prediction] = res.get("proposed")
    primary = prop or base
    if primary is None:
        st.error("No model produced a prediction. Open the sidebar to check model status.")
        return

    lab = label_of(primary.prob, threshold)
    conf = confidence_of(primary.prob, threshold)
    cls = "bad" if lab else "good"
    title = "CLICKBAIT DETECTED" if lab else "NOT CLICKBAIT"
    desc = ("This headline contains characteristics commonly associated with clickbait content." if lab
            else "This headline appears to be a straightforward news headline.")

    md(
        f'<div class="result {cls}"><div class="result-head"><div>',
        f'<div class="result-label">{title}</div><div class="result-desc">{desc}</div></div></div>',
        f'<div class="quote">“{esc(res["headline"])}”</div>',
        f'<div class="conf-row"><span>Model Confidence</span><b>{conf * 100:.1f}%</b></div>',
        f'<div class="track"><div class="fill" style="width:{conf * 100:.1f}%"></div></div>',
        '<div class="conf-note">Confidence is the model’s probability for the predicted class. It is not the model’s accuracy '
        f'(test accuracy: {BENCHMARKS[-1][1] * 100:.1f}%).</div></div>',
    )
    for w in res.get("warnings", []):
        md(f'<div class="notice warn">{esc(w)}</div>')
    if not prop:
        md('<div class="notice warn">The proposed model is unavailable; this result comes from the baseline model.</div>')

    # ---- Prediction details ------------------------------------------------
    rows = ""
    for name, pred, is_primary in (("Baseline BiLSTM", base, primary is base),
                                   ("XLM-R + BiLSTM + Attention", prop, primary is prop)):
        if not pred:
            continue
        l = label_of(pred.prob, threshold)
        tag = '<span class="tag bad">Clickbait</span>' if l else '<span class="tag good">Not clickbait</span>'
        rows += (f'<tr class="{"best" if is_primary else ""}"><td>{name}{" · primary" if is_primary and base and prop else ""}</td>'
                 f'<td>{tag}</td><td>{pred.prob * 100:.1f}%</td><td class="mono">{pred.ms:.0f} ms</td></tr>')
    agree = ""
    if base and prop:
        same = label_of(base.prob, threshold) == label_of(prop.prob, threshold)
        agree = (f'<div class="notice">{"Both models agree on this headline." if same else "The models disagree. The proposed model is the primary result (higher test accuracy and F1)."}</div>')
    card("Prediction details",
         f'<p style="margin:0 0 .8rem">{esc(interpretation(conf, lab))}</p>',
         '<table class="tbl"><tr><th>Model</th><th>Classification</th><th>Clickbait probability</th><th>Time</th></tr>',
         rows, '</table>', agree)

    with st.expander("Technical details"):
        cues = style_cues(res["raw"])
        st.markdown("**Preprocessed input**")
        st.code(preprocess_text(res["headline"]), language=None)
        if prop and prop.encoder_ms is not None:
            st.caption(f"Proposed model: XLM-R encoder {prop.encoder_ms:.0f} ms + BiLSTM/attention head {prop.head_ms:.0f} ms.")
        if base and base.coverage is not None:
            st.caption(f"Baseline vocabulary coverage: {base.coverage * 100:.0f}% of words found in the training vocabulary.")
        st.markdown("**Wording cues** (rule-based, for explanation only; not used by the models)")
        md("".join(f'<span class="chip" title="{esc(w)}">{esc(n)}</span>' for n, w in cues)
           if cues else '<span class="muted">No common clickbait wording patterns detected.</span>')


def render_benchmarks() -> str:
    best = max(r[4] for r in BENCHMARKS)
    rows = "".join(
        f'<tr class="{"best" if r[4] == best else ""}"><td>{esc(r[0])}</td>'
        f'<td>{r[1]:.4f}</td><td>{r[2]:.4f}</td><td>{r[3]:.4f}</td><td>{r[4]:.4f}</td></tr>'
        for r in BENCHMARKS
    )
    return ('<table class="tbl"><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'
            f'{rows}</table>')


def render_sidebar(system: dict) -> float:
    with st.sidebar:
        st.markdown("### Inference Parameter")
        threshold = st.slider(
            "Classification threshold(τ)", 0.30, 0.70, DEFAULT_THRESHOLD, 0.01,
            help="Models output a clickbait probability from 0 to 1. A headline is labelled clickbait "
                 "when its probability is at or above this cut-off. 0.50 is the value used in the thesis evaluation.",
        )
        st.caption("Lower = flags more headlines (more false alarms). Higher = stricter.")
        if abs(threshold - DEFAULT_THRESHOLD) > 1e-9:
            st.caption("Differs from the 0.50 used in the thesis results.")

        st.markdown("### Appearance")
        cur = st.session_state.theme
        a, b = st.columns(2)
        a.button("Light", key="theme_light", use_container_width=True, on_click=set_theme, args=("light",),
                 type="primary" if cur == "light" else "secondary")
        b.button("Dark", key="theme_dark", use_container_width=True, on_click=set_theme, args=("dark",),
                 type="primary" if cur == "dark" else "secondary")

        st.markdown("### System status")
        for key, label in (("baseline", "Baseline BiLSTM"), ("proposed", "XLM-R + BiLSTM + Attention")):
            ok = bool(system.get(key))
            md(f'<div style="margin:.3rem 0"><span class="status"><span class="dot {"" if ok else "off"}"></span>'
               f'{label}: {"ready" if ok else "unavailable"}</span></div>')
            if not ok:
                err = system["errors"].get(key) or system["errors"].get("libraries")
                if err:
                    st.caption(err[:220])
        if system.get("load_s"):
            st.caption(f"Models loaded in {system['load_s']:.1f}s (cached).")
        st.button("Clear history", use_container_width=True, on_click=clear_history)
    return threshold


# =============================================================================
# 7. PAGES
# =============================================================================

def run_single(raw: str, system: dict) -> Tuple[Optional[dict], Optional[Tuple[str, str]]]:
    """Validate, run both models (each isolated). Returns (result, (kind, message))."""
    v = validate_headline(raw)
    if not v.ok:
        return None, ("error", v.error)
    res = {"headline": v.text, "raw": v.text, "warnings": v.warnings, "baseline": None, "proposed": None}
    errors = []
    for key, fn in (("baseline", predict_baseline), ("proposed", predict_proposed)):
        if not system.get(key):
            continue
        try:
            res[key] = fn([v.text], system[key])[0]
        except Exception as exc:
            log.exception("%s inference failed", key)
            errors.append(f"{key.title()} model failed: {exc}")
    if not (res["baseline"] or res["proposed"]):
        return None, ("error", " ".join(errors) or "No model is available. Open the sidebar to check model status.")
    return res, (("warning", " ".join(errors)) if errors else None)


# ---- callbacks (run before widgets are drawn, so session_state edits are safe)
def start_analysis() -> None:
    st.session_state.processing = True
    st.session_state.pending = st.session_state.headline_input
    st.session_state.msg = None


def clear_input() -> None:
    st.session_state.headline_input = ""
    st.session_state.result = None
    st.session_state.msg = None


def clear_history() -> None:
    st.session_state.history = []


def clear_batch() -> None:
    """Reset batch inputs and results. Bumping the uploader key empties the file box."""
    st.session_state.batch_text = ""
    st.session_state.batch = None
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


def set_theme(name: str) -> None:
    st.session_state.theme = name


def tab_check(system: dict, threshold: float) -> None:
    busy = st.session_state.processing
    try:
        box = st.container(key="inputcard")
    except TypeError:  # older Streamlit without container keys
        box = st.container()
    with box:
        md('<div class="card-title">Enter News Headline</div>')
        st.text_area("Enter News Headline", key="headline_input", height=120, max_chars=MAX_CHARS,
                     placeholder="Paste or type a news headline here...", label_visibility="collapsed", disabled=busy)
        c1, c2 = st.columns([3, 1])
        c1.button("Analyzing…" if busy else "Analyze Headline", type="primary", use_container_width=True,
                  disabled=busy, on_click=start_analysis)
        c2.button("Clear", use_container_width=True, disabled=busy, on_click=clear_input)

    if busy:
        with st.spinner("Analyzing headline…"):
            try:
                res, msg = run_single(st.session_state.pending, system)
                st.session_state.msg = msg
                if res:
                    st.session_state.result = res
                    main_pred = res["proposed"] or res["baseline"]
                    hist = st.session_state.history
                    hist.insert(0, {
                        "Headline": res["headline"],
                        "Baseline %": round(res["baseline"].prob * 100, 1) if res["baseline"] else None,
                        "Proposed %": round(res["proposed"].prob * 100, 1) if res["proposed"] else None,
                        "Verdict": "Clickbait" if label_of(main_pred.prob, threshold) else "Not clickbait",
                    })
                    del hist[HISTORY_LIMIT:]
                else:
                    st.session_state.result = None
            except Exception as exc:  # last-resort guard: never leave the UI locked
                log.exception("Analysis failed")
                st.session_state.msg = ("error", f"Analysis failed: {exc}")
            finally:
                st.session_state.processing = False
        st.rerun()

    kind, text = st.session_state.msg or (None, None)
    if kind == "error":
        st.error(text)
    elif kind == "warning":
        st.warning(text)

    if st.session_state.result:
        render_result(st.session_state.result, threshold)

    if st.session_state.history:
        with st.expander(f"Session history ({len(st.session_state.history)})"):
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button("Download history (CSV)", df.to_csv(index=False).encode("utf-8"),
                               "headline_history.csv", "text/csv")

    card("About the Model",
         '<p style="margin:0 0 .6rem"><b>Enhanced BiLSTM with XLM-RoBERTa.</b> The system classifies news headlines as clickbait or not clickbait. '
         'A frozen XLM-RoBERTa encoder provides multilingual contextual embeddings, a bidirectional LSTM models word order, '
         'and a self-attention layer focuses on the most informative tokens. A baseline BiLSTM is shown alongside for comparison.</p>',
         PROPOSED_FLOW)


def read_upload(upload) -> Tuple[pd.DataFrame, Optional[str]]:
    """Read a CSV upload; returns (dataframe, error)."""
    raw = upload.getvalue()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc), None
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            return pd.DataFrame(), f"Could not read the CSV: {exc}"
    return pd.DataFrame(), "Could not decode the file. Save it as UTF-8 CSV."


def tab_batch(system: dict, threshold: float) -> None:
    card("Batch analysis",
         '<div class="muted">Check many headlines at once. Upload a CSV with a <b>headline</b> column '
         '(optional <b>label</b> column, 1 = clickbait, 0 = not, to compute accuracy / precision / recall / F1), '
         f'or paste one headline per line. Up to {MAX_BATCH_ROWS} rows.</div>')
    upload = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed",
                              key=f"batch_upload_{st.session_state.get('uploader_key', 0)}")
    pasted = st.text_area("Or paste headlines (one per line)", height=130, key="batch_text")

    b1, b2 = st.columns([3, 1])
    run = b1.button("Run batch analysis", type="primary", use_container_width=True)
    b2.button("Clear", key="batch_clear", use_container_width=True, on_click=clear_batch)
    if not run:
        return

    texts: List[str] = []
    truth: List[Optional[int]] = []
    if upload is not None:
        df, err = read_upload(upload)
        if err:
            st.error(err)
            return
        cols = {c.lower().strip(): c for c in df.columns}
        hcol = next((cols[k] for k in ("headline", "title", "judul", "text", "headlines") if k in cols), None)
        if hcol is None:
            obj = [c for c in df.columns if df[c].dtype == object]
            hcol = obj[0] if obj else None
        if hcol is None:
            st.error("No text column found. Add a column named “headline”.")
            return
        texts = df[hcol].astype(str).tolist()
        lcol = next((cols[k] for k in ("label", "class", "kelas") if k in cols), None)
        truth = [parse_label(x) for x in df[lcol]] if lcol else [None] * len(texts)
    elif pasted.strip():
        texts = [ln for ln in pasted.splitlines() if ln.strip()]
        truth = [None] * len(texts)
    else:
        st.warning("Upload a CSV or paste at least one headline.")
        return

    pairs = [(clean_input(t), y) for t, y in zip(texts, truth)]
    valid = [(t, y) for t, y in pairs if validate_headline(t).ok]
    skipped = len(pairs) - len(valid)
    if len(valid) > MAX_BATCH_ROWS:
        st.warning(f"Only the first {MAX_BATCH_ROWS} valid headlines were analysed.")
        valid = valid[:MAX_BATCH_ROWS]
    if not valid:
        st.error("No valid headlines found (each needs at least two words).")
        return
    if skipped:
        st.info(f"{skipped} empty or invalid row(s) were skipped.")

    heads = [t for t, _ in valid]
    ytrue = [y for _, y in valid]
    out = pd.DataFrame({"headline": heads})
    bar = st.progress(0.0, text="Analysing…")
    t0 = time.perf_counter()
    try:
        if system.get("baseline"):
            out["baseline_prob"] = [round(p.prob, 4) for p in predict_baseline(heads, system["baseline"])]
        if system.get("proposed"):
            preds = predict_proposed(heads, system["proposed"], progress=lambda f: bar.progress(f, text="Analysing…"))
            out["proposed_prob"] = [round(p.prob, 4) for p in preds]
    except Exception as exc:
        log.exception("Batch failed")
        st.error(f"Batch analysis failed: {exc}")
        return
    elapsed = time.perf_counter() - t0
    bar.empty()

    for key in ("baseline", "proposed"):
        if f"{key}_prob" in out:
            out[f"{key}_label"] = (out[f"{key}_prob"] >= threshold).astype(int)
    if any(y is not None for y in ytrue):
        out["true_label"] = ytrue

    st.session_state.batch = {"df": out, "elapsed": elapsed, "threshold": threshold}


def show_batch(threshold: float) -> None:
    data = st.session_state.get("batch")
    if not data:
        return
    out: pd.DataFrame = data["df"]
    if "proposed_prob" not in out and "baseline_prob" not in out:
        st.warning("No model produced predictions for this batch. Open the sidebar to check model status.")
        return
    n = len(out)
    c = st.columns(4)
    c[0].metric("Headlines", n)
    c[1].metric("Total time", f"{data['elapsed']:.1f} s")
    c[2].metric("Per headline", f"{data['elapsed'] / n * 1000:.0f} ms")
    primary = "proposed" if "proposed_prob" in out else "baseline"
    pl = (out[f"{primary}_prob"] >= threshold).astype(int)
    c[3].metric("Flagged clickbait", f"{int(pl.sum())} ({pl.mean() * 100:.0f}%)")

    if "true_label" in out and out["true_label"].notna().any():
        known = out.dropna(subset=["true_label"])
        rows = {}
        for key, name in (("baseline", "Baseline BiLSTM"), ("proposed", "XLM-R + BiLSTM + Attention")):
            if f"{key}_prob" in known:
                preds = (known[f"{key}_prob"] >= threshold).astype(int)
                rows[name] = binary_metrics(known["true_label"].astype(int), preds)
        if rows:
            st.markdown(f"**Evaluation on {len(known)} labelled rows (threshold {threshold:.2f})**")
            st.dataframe(pd.DataFrame(rows).T.style.format("{:.4f}"), use_container_width=True)

    st.dataframe(out, use_container_width=True, hide_index=True)
    st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "clickbait_batch_results.csv", "text/csv")


def tab_about(system: dict) -> None:
    card("Purpose",
         '<p style="margin:0">Clickbait headlines attract clicks with exaggerated or incomplete information. This prototype lets a user '
         'enter any headline and compare a conventional <b>baseline BiLSTM</b> with the proposed '
         '<b>XLM-RoBERTa + BiLSTM + Self-Attention</b> model, showing the predicted class, confidence and speed.</p>')

    card("Workflow",
         '<div class="flow"><span class="s">1 · Headline input</span><i>→</i><span class="s">2 · Validation</span><i>→</i>'
         '<span class="s">3 · Preprocessing</span><i>→</i><span class="s">4 · Both models</span><i>→</i>'
         '<span class="s hl">5 · Verdict + comparison</span></div>',
         '<p class="muted" style="margin:0">Preprocessing: lowercase, “!” → “tanda seru”, “?” → “tanda tanya”, punctuation removal, whitespace normalisation (same as training).</p>')

    card("Model architectures", '<b>Baseline</b>', BASELINE_FLOW, '<b>Proposed</b>', PROPOSED_FLOW)

    card("Experimental results (test set)", render_benchmarks(),
         '<p class="muted" style="margin:.7rem 0 0">The proposed model has the best accuracy, precision and F1. '
         'Recall is highest for BiLSTM + XLM-R without attention.</p>')

    px, bl = system.get("proposed"), system.get("baseline")
    pcfg = (px or {}).get("config", {})
    cfg_rows = [
        ("Max sequence length", (px or {}).get("max_len", (bl or {}).get("max_len", 128))),
        ("Encoder", pcfg.get("MODEL_NAME", "xlm-roberta-base")),
        ("LSTM units", pcfg.get("LSTM_UNITS", 64)),
        ("Dropout", pcfg.get("DROPOUT_RATE", 0.2)),
        ("Learning rate", pcfg.get("LEARNING_RATE", 0.001)),
        ("Baseline embedding dim", (bl or {}).get("config", {}).get("embedding_dim", 300)),
    ]
    card("Configuration",
         '<table class="tbl">' + "".join(f"<tr><td>{esc(k)}</td><td><b>{esc(v)}</b></td></tr>" for k, v in cfg_rows) + "</table>")

    card("Limitations & responsible use",
         '<ul style="margin:0"><li>Trained on Indonesian-language headlines; other languages or scripts may be misclassified.</li>'
         '<li>Test F1 is about 0.73, so errors are expected. Use the result as decision support, not as proof.</li>'
         '<li>Low-confidence results should be reviewed by a person.</li>'
         '<li>The XLM-R encoder makes the proposed model slower than the baseline; batch mode amortises this cost.</li></ul>')

    card("Deployment requirements",
         '<ul style="margin:0"><li>Python 3.10+ with <code>streamlit</code>, <code>tensorflow</code>, <code>torch</code>, <code>transformers</code>, <code>pandas</code>, <code>numpy</code>.</li>'
         '<li>Runs on CPU. About 2 GB RAM (XLM-R base ≈ 1.1 GB, downloaded once from Hugging Face, or placed in a local <code>xlm-roberta-base/</code> folder for offline use).</li>'
         '<li>Models and tokenizer load once per server process (cached) and are warmed up before the first request.</li></ul>')


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    st.session_state.setdefault("theme", "light")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("headline_input", "")
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("pending", "")
    st.session_state.setdefault("msg", None)
    st.session_state.setdefault("batch_text", "")
    st.session_state.setdefault("batch", None)

    st.markdown(CSS, unsafe_allow_html=True)
    md(f'<div class="theme-marker {st.session_state.theme}" aria-hidden="true"></div>')

    with st.spinner("Loading models (first start can take a minute)…"):
        system = load_system()

    render_header()

    if IMPORT_ERROR:
        st.error(f"Required libraries are missing: {IMPORT_ERROR}. Run: pip install -r requirements.txt")
    elif not system["baseline"] and not system["proposed"]:
        st.error("No model could be loaded. Open the sidebar (top left) to see the reason.")

    threshold = render_sidebar(system)

    t1, t2, t3 = st.tabs(["Classifier", "Batch analysis", "About"])
    with t1:
        tab_check(system, threshold)
    with t2:
        tab_batch(system, threshold)
        show_batch(threshold)
    with t3:
        tab_about(system)

    md('<div class="footer">Research prototype. Predictions may be wrong.</div>')


main()