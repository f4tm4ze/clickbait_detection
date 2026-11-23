from __future__ import annotations

import html as _html
import io
import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from core import (
    DEFAULT_THRESHOLD,
    MAX_CHARS,
    binary_metrics,
    clean_input,
    confidence_of,
    explain,
    label_of,
    parse_label,
    preprocess_text,
    style_cues,
    validate_headline,
)

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
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(BASE_DIR, "deployment_baseline")
PROPOSED_DIR = os.path.join(BASE_DIR, "deployment_proposed")
CSS_FILE = os.path.join(BASE_DIR, "styles.css")

BASELINE_MODEL_FILE = "baseline_bilstm.keras"
BASELINE_TOKENIZER_FILE = "baseline_tokenizer.pkl"
BASELINE_WORD_INDEX_FILE = "baseline_word_index.pkl"
BASELINE_CONFIG_FILE = "baseline_config.pkl"
PROPOSED_MODEL_FILE = "xlmr_bilstm_attention.keras"
PROPOSED_CONFIG_FILE = "xlmr_config.txt"

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
# 2. MODEL LOADING (cached; each component fails independently)
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
# 3. INFERENCE
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
# 4. UI COMPONENTS
# =============================================================================

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


def load_css() -> str:
    """Read styles.css (kept as a separate file so styling is easy to maintain)."""
    try:
        with open(CSS_FILE, "r", encoding="utf-8") as fh:
            css = "\n".join(line for line in fh.read().splitlines() if line.strip())
        return f"<style>\n{css}\n</style>"
    except OSError:
        log.warning("styles.css not found; falling back to default Streamlit styling")
        return ""


def esc(s: str) -> str:
    return _html.escape(str(s), quote=True)


def md(*lines: str) -> None:
    """Render raw HTML. Lines are stripped and joined so Markdown never sees
    indentation or blank lines (which would break the HTML block)."""
    st.markdown("\n".join(l.strip() for l in lines if l and l.strip()), unsafe_allow_html=True)


def alert(kind: str, message: str) -> None:
    """Bright, theme-independent alert. kind: 'error' | 'warning' | 'info'."""
    md(f'<div class="alert {kind}" role="alert">{esc(message)}</div>')


def card(title: str, *inner: str) -> None:
    md(f'<div class="card"><div class="card-title">{esc(title)}</div>', *inner, "</div>")


def show_table(df: pd.DataFrame, max_height: int = 420) -> None:
    """Themed HTML table (st.dataframe ignores our light/dark toggle)."""
    table = df.to_html(index=False, border=0, classes="tbl", na_rep="–", escape=True)
    md(f'<div class="tbl-wrap" style="max-height:{max_height}px">', table, "</div>")


def render_header() -> None:
    md('<div class="page-header"><div class="page-title">Clickbait Headline Detection</div>',
       '<div class="page-sub">News Headline Classification</div></div>')


def interpretation(conf: float, lab: int) -> str:
    level = "highly confident" if conf >= 0.85 else "moderately confident" if conf >= 0.65 else "only weakly confident"
    what = "clickbait" if lab else "not clickbait"
    tail = " Consider reviewing this headline manually." if conf < 0.65 else ""
    return f"The model is {level} that this headline is {what}.{tail}"


def render_result(res: dict, threshold: float) -> None:
    """Verdict, plain-language reasons, prediction details and technical details."""
    base: Optional[Prediction] = res.get("baseline")
    prop: Optional[Prediction] = res.get("proposed")
    primary = prop or base
    if primary is None:
        alert("error", "No model produced a prediction. Open the sidebar to check model status.")
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

    agree_flag = None
    if base and prop:
        agree_flag = label_of(base.prob, threshold) == label_of(prop.prob, threshold)

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
    if agree_flag is not None:
        agree = (f'<div class="notice">{"Both models agree on this headline." if agree_flag else "The models disagree. The proposed model is the primary result (higher test accuracy and F1)."}</div>')
    card("Prediction details",
         f'<p style="margin:0 0 .8rem">{esc(interpretation(conf, lab))}</p>',
         '<table class="tbl"><tr><th>Model</th><th>Classification</th><th>Clickbait probability</th><th>Time</th></tr>',
         rows, '</table>', agree)

    # ---- Plain-language reasons -------------------------------------------
    summary, bullets = explain(style_cues(res["raw"]), lab, conf, agree_flag)
    card("Why this result?",
         f'<p style="margin:0 0 .6rem">{esc(summary)}</p>',
         '<ul style="margin:0 0 .7rem">' + "".join(f"<li>{esc(b)}</li>" for b in bullets) + "</ul>",
         '<div class="muted">These reasons come from simple wording checks. '
         'The AI model itself decides from patterns it learned during training.</div>')

    with st.expander("Technical details"):
        st.markdown("**Preprocessed input**")
        st.code(preprocess_text(res["headline"]), language=None)
        if prop and prop.encoder_ms is not None:
            st.caption(f"Proposed model: XLM-R encoder {prop.encoder_ms:.0f} ms + BiLSTM/attention head {prop.head_ms:.0f} ms.")
        if base and base.coverage is not None:
            st.caption(f"Baseline vocabulary coverage: {base.coverage * 100:.0f}% of words found in the training vocabulary.")


def render_sidebar(system: dict) -> float:
    with st.sidebar:
        st.markdown("### Inference Parameter")
        threshold = st.slider(
            "Classification threshold (τ)", 0.30, 0.70, DEFAULT_THRESHOLD, 0.01,
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
# 5. PAGES
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
        c1.button("Analyzing…" if busy else "Analyze Headline", key="analyze_btn", type="primary",
                  use_container_width=True, disabled=busy, on_click=start_analysis)
        c2.button("Clear", key="clear_btn", use_container_width=True, disabled=busy, on_click=clear_input)

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
        alert("error", text)
    elif kind == "warning":
        alert("warning", text)

    if st.session_state.result:
        render_result(st.session_state.result, threshold)

    if st.session_state.history:
        with st.expander(f"Session history ({len(st.session_state.history)})"):
            df = pd.DataFrame(st.session_state.history)
            show_table(df, 300)
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
            alert("error", err)
            return
        cols = {c.lower().strip(): c for c in df.columns}
        hcol = next((cols[k] for k in ("headline", "title", "judul", "text", "headlines") if k in cols), None)
        if hcol is None:
            obj = [c for c in df.columns if df[c].dtype == object]
            hcol = obj[0] if obj else None
        if hcol is None:
            alert("error", "No text column found. Add a column named “headline”.")
            return
        texts = df[hcol].astype(str).tolist()
        lcol = next((cols[k] for k in ("label", "class", "kelas") if k in cols), None)
        truth = [parse_label(x) for x in df[lcol]] if lcol else [None] * len(texts)
    elif pasted.strip():
        texts = [ln for ln in pasted.splitlines() if ln.strip()]
        truth = [None] * len(texts)
    else:
        alert("warning", "Upload a CSV or paste at least one headline.")
        return

    pairs = [(clean_input(t), y) for t, y in zip(texts, truth)]
    valid = [(t, y) for t, y in pairs if validate_headline(t).ok]
    skipped = len(pairs) - len(valid)
    if len(valid) > MAX_BATCH_ROWS:
        alert("warning", f"Only the first {MAX_BATCH_ROWS} valid headlines were analysed.")
        valid = valid[:MAX_BATCH_ROWS]
    if not valid:
        alert("error", "No valid headlines found (each needs at least two words).")
        return
    if skipped:
        alert("info", f"{skipped} empty or invalid row(s) were skipped.")

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
        alert("error", f"Batch analysis failed: {exc}")
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
        alert("warning", "No model produced predictions for this batch. Open the sidebar to check model status.")
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
            metrics = pd.DataFrame(rows).T.round(4).reset_index().rename(columns={"index": "Model"})
            show_table(metrics, 200)

    show_table(out, 420)
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

    st.markdown(load_css(), unsafe_allow_html=True)
    md(f'<div class="theme-marker {st.session_state.theme}" aria-hidden="true"></div>')

    with st.spinner("Loading models (first start can take a minute)…"):
        system = load_system()

    render_header()

    if IMPORT_ERROR:
        alert("error", f"Required libraries are missing: {IMPORT_ERROR}. Run: pip install -r requirements.txt")
    elif not system["baseline"] and not system["proposed"]:
        alert("error", "No model could be loaded. Open the sidebar (top left) to see the reason.")

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
