"""Review sentiment analysis.

Backends
--------
``transformers``  Pretrained DistilBERT (default: SST-2). CPU-friendly, batched,
                  length-sorted for efficient padding, resumable via a Parquet cache.
``lexicon``       Small rule/lexicon fallback with negation handling. Much weaker
                  than a transformer; results are labelled ``lexicon`` everywhere.
``none``          Skip sentiment. Dependent components are reported as missing.

Neutral class
-------------
The default DistilBERT model is *binary* (positive / negative). A "neutral" label
is therefore **derived from model uncertainty**: reviews with
``|P(positive) - 0.5| < neutral_band`` are labelled neutral. This is not a trained
neutral class and is documented as such in the UI. If a 3-class model is supplied
via ``SENTIMENT_MODEL`` its native neutral probability is used.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from trust_assessor.config import Settings

log = logging.getLogger(__name__)

SENT_COLUMNS = ["sent_pos", "sent_neu", "sent_neg", "sent_label", "sent_score", "sentiment_method"]


# ------------------------------------------------------------------ label logic
def _labels_from_probs(p_pos: np.ndarray, p_neu: np.ndarray | None, p_neg: np.ndarray, band: float) -> np.ndarray:
    if p_neu is not None:
        stacked = np.vstack([p_neg, p_neu, p_pos]).T
        return np.array(["negative", "neutral", "positive"])[stacked.argmax(axis=1)]
    label = np.where(p_pos >= 0.5, "positive", "negative")
    return np.where(np.abs(p_pos - 0.5) < band, "neutral", label)


# ------------------------------------------------------------------ transformer
class TransformersBackend:
    def __init__(self, model_name: str, batch_size: int, max_length: int, n_threads: int):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Sentiment backend 'transformers' needs torch + transformers. Install them "
                "(see README) or set SENTIMENT_BACKEND=lexicon / none."
            ) from exc
        self.torch = torch
        torch.set_num_threads(max(1, n_threads))
        log.info("Loading sentiment model '%s' (first run downloads ~270 MB)", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.batch_size, self.max_length, self.name = batch_size, max_length, model_name

        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.idx_pos = self._find(id2label, ("pos",))
        self.idx_neg = self._find(id2label, ("neg",))
        self.idx_neu = self._find(id2label, ("neu",), required=False)
        if self.idx_pos is None or self.idx_neg is None:
            if len(id2label) == 2:  # generic LABEL_0 / LABEL_1: assume index 1 is positive
                self.idx_neg, self.idx_pos = 0, 1
                log.warning("Model labels %s are generic; assuming index 1 = positive.", id2label)
            else:
                raise RuntimeError(f"Cannot map model labels {id2label} to positive/negative.")

    @staticmethod
    def _find(id2label: dict[int, str], keys: tuple[str, ...], required: bool = True):
        for i, name in id2label.items():
            if any(k in name for k in keys):
                return i
        return None

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return an (n, n_classes) probability matrix in model class order."""
        torch = self.torch
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def score(self, texts: list[str], band: float, progress=None) -> pd.DataFrame:
        order = np.argsort([len(t) for t in texts])           # similar lengths -> less padding
        probs = np.zeros((len(texts), self.model.config.num_labels), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            idx = order[start:start + self.batch_size]
            probs[idx] = self.predict([texts[i] for i in idx])
            if progress:
                progress(len(idx))
        p_pos, p_neg = probs[:, self.idx_pos], probs[:, self.idx_neg]
        p_neu = probs[:, self.idx_neu] if self.idx_neu is not None else None
        return pd.DataFrame({
            "sent_pos": p_pos,
            "sent_neu": p_neu if p_neu is not None else np.nan,
            "sent_neg": p_neg,
            "sent_label": _labels_from_probs(p_pos, p_neu, p_neg, band),
        })


# ------------------------------------------------------------------ lexicon
_POS = set("""good great excellent amazing awesome love loved loves perfect perfectly best nice wonderful fantastic
happy pleased satisfied recommend recommended works worked quality beautiful soft smooth gentle lovely favorite
favourite fabulous superb brilliant enjoy enjoyed easy comfortable effective helpful value worth impressed
delighted fresh pleasant sturdy durable fast cute glad""".split())
_NEG = set("""bad terrible awful horrible worst hate hated poor waste useless disappointed disappointing disappointment
broken broke cheap junk fake defective refund return returned returning leak leaked leaking smell smells smelly
irritated irritation rash burn burned burning allergic difficult uncomfortable flimsy damaged garbage scam
never unhappy overpriced faulty regret stopped sticky greasy dried""".split())
_NEGATORS = {"not", "no", "never", "n't", "dont", "don't", "didnt", "didn't", "isnt", "isn't", "wasnt", "wasn't",
             "cant", "can't", "wont", "won't", "hardly", "without"}
_INTENSIFIERS = {"very", "really", "extremely", "so", "absolutely", "totally", "super", "incredibly"}
_TOKEN = re.compile(r"[a-z']+")


def lexicon_net_score(text: str) -> float:
    tokens = _TOKEN.findall(text.lower())
    net = 0.0
    for i, tok in enumerate(tokens):
        pol = 1.0 if tok in _POS else -1.0 if tok in _NEG else 0.0
        if pol == 0.0:
            continue
        window = tokens[max(0, i - 3):i]
        if any(w in _NEGATORS for w in window):
            pol = -pol
        if i > 0 and tokens[i - 1] in _INTENSIFIERS:
            pol *= 1.5
        net += pol
    return net


def lexicon_score(texts: list[str], band: float) -> pd.DataFrame:
    net = np.array([lexicon_net_score(t) for t in texts])
    p_pos = 1.0 / (1.0 + np.exp(-0.9 * net))
    return pd.DataFrame({
        "sent_pos": p_pos,
        "sent_neu": np.nan,
        "sent_neg": 1.0 - p_pos,
        "sent_label": _labels_from_probs(p_pos, None, 1.0 - p_pos, band),
    })


# ------------------------------------------------------------------ orchestration
def method_name(settings: Settings) -> str:
    if settings.sentiment_backend == "transformers":
        return f"transformers:{settings.sentiment_model}"
    return settings.sentiment_backend


def run_sentiment(reviews: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Return a frame indexed like ``reviews`` with the SENT_COLUMNS (NaN when not scored).

    Results are cached in ``settings.sentiment_path`` keyed by ``review_id`` and method, so
    interrupted runs resume and re-runs on the same sample are instant.
    """
    method = method_name(settings)
    out = pd.DataFrame(index=reviews.index, columns=SENT_COLUMNS, dtype=object)
    out["sentiment_method"] = method if settings.sentiment_backend != "none" else pd.NA

    if settings.sentiment_backend == "none":
        log.info("Sentiment backend = none; skipping sentiment analysis.")
        return _finalise(out)

    cache = pd.DataFrame(columns=["review_id", *SENT_COLUMNS])
    if settings.sentiment_path.exists():
        cached = pd.read_parquet(settings.sentiment_path)
        if len(cached) and cached["sentiment_method"].iloc[0] == method:
            cache = cached
            log.info("Sentiment cache hit for %s reviews", f"{len(cache):,}")
        else:
            log.info("Sentiment cache was produced by a different method; recomputing.")

    wanted = reviews[["review_id", "full_text"]].copy()
    if settings.sentiment_max_reviews and settings.sentiment_max_reviews < len(wanted):
        keep = wanted.sample(n=settings.sentiment_max_reviews, random_state=settings.seed)
        log.info("Scoring a random subset of %s / %s reviews (SENTIMENT_MAX_REVIEWS)",
                 f"{len(keep):,}", f"{len(wanted):,}")
        wanted = keep
    todo = wanted[~wanted["review_id"].isin(set(cache["review_id"]))]
    log.info("Sentiment: %s to score, %s already cached", f"{len(todo):,}", f"{len(wanted) - len(todo):,}")

    new_parts: list[pd.DataFrame] = []
    if len(todo):
        texts = todo["full_text"].fillna("").astype(str).str.slice(0, 2000).tolist()
        if settings.sentiment_backend == "transformers":
            from tqdm import tqdm

            backend = TransformersBackend(
                settings.sentiment_model, settings.sentiment_batch_size,
                settings.sentiment_max_length, settings.effective_jobs,
            )
            chunk = 4096
            bar = tqdm(total=len(texts), desc="sentiment", unit="rev")
            for s in range(0, len(texts), chunk):
                part = backend.score(texts[s:s + chunk], settings.sentiment_neutral_band, progress=bar.update)
                part.insert(0, "review_id", todo["review_id"].iloc[s:s + chunk].to_numpy())
                part["sent_score"] = part["sent_pos"] - part["sent_neg"]
                part["sentiment_method"] = method
                new_parts.append(part)
                # checkpoint so an interrupted run can resume
                _write_cache(pd.concat([cache, *new_parts], ignore_index=True), settings)
            bar.close()
        elif settings.sentiment_backend == "lexicon":
            part = lexicon_score(texts, settings.sentiment_neutral_band)
            part.insert(0, "review_id", todo["review_id"].to_numpy())
            part["sent_score"] = part["sent_pos"] - part["sent_neg"]
            part["sentiment_method"] = method
            new_parts.append(part)
        else:
            raise ValueError(f"Unknown SENTIMENT_BACKEND '{settings.sentiment_backend}'")

    full = pd.concat([cache, *new_parts], ignore_index=True) if new_parts else cache
    if len(full):
        _write_cache(full, settings)
    full = full[full["review_id"].isin(set(wanted["review_id"]))]
    merged = reviews[["review_id"]].merge(full, on="review_id", how="left")
    merged.index = reviews.index
    return _finalise(merged[SENT_COLUMNS])


def _write_cache(df: pd.DataFrame, settings: Settings) -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.sentiment_path, index=False)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("sent_pos", "sent_neu", "sent_neg", "sent_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sent_label"] = df["sent_label"].astype("object")
    df["sentiment_method"] = df["sentiment_method"].astype("object")
    return df[SENT_COLUMNS]


# ------------------------------------------------------------------ proxy sanity check
def proxy_agreement_report(reviews: pd.DataFrame, method: str) -> dict[str, Any]:
    """How well do sentiment predictions agree with star ratings?

    Star ratings are only a *noisy proxy* for sentiment (e.g. a 4-star review can contain
    complaints), so these numbers are a sanity check, **not** an accuracy evaluation of the
    sentiment model.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score, roc_auc_score

    scored = reviews[reviews["sent_label"].notna()]
    report: dict[str, Any] = {
        "method": method,
        "n_scored": int(len(scored)),
        "label_distribution": {k: float(v) for k, v in scored["sent_label"].value_counts(normalize=True).items()},
        "caveat": ("Star ratings are a noisy proxy for sentiment. These figures are an agreement "
                   "check, not a ground-truth accuracy evaluation."),
    }
    if len(scored) >= 30 and scored["sent_score"].std() > 0 and scored["rating"].std() > 0:
        rho = spearmanr(scored["sent_score"], scored["rating"]).statistic
        report["spearman_score_vs_rating"] = None if math.isnan(rho) else float(rho)
        proxy = scored[scored["rating"].isin([1, 2, 4, 5])]
        y = (proxy["rating"] >= 4).astype(int)
        if y.nunique() == 2:
            report["n_proxy_labelled"] = int(len(proxy))
            report["roc_auc_vs_proxy"] = float(roc_auc_score(y, proxy["sent_score"]))
            decisive = proxy[proxy["sent_label"] != "neutral"]
            if len(decisive):
                pred = (decisive["sent_label"] == "positive").astype(int)
                truth = (decisive["rating"] >= 4).astype(int)
                report["agreement_non_neutral"] = float((pred == truth).mean())
                report["macro_f1_non_neutral"] = float(f1_score(truth, pred, average="macro"))
                report["share_neutral_among_proxy"] = float(1 - len(decisive) / len(proxy))
    return report
