"""Review-level feature engineering (text statistics + behavioural context)."""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_REPEAT_RE = re.compile(r"(.)\1{3,}")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SENT_SPLIT = re.compile(r"[.!?]+")


def normalise_text(text: str) -> str:
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add content statistics computed from ``full_text`` (title + body)."""
    df = df.copy()
    texts = df["full_text"].fillna("").astype(str).tolist()

    word_count, unique_ratio, avg_word_len = [], [], []
    caps_ratio, digit_ratio, repeated, sentences = [], [], [], []
    for t in texts:
        words = _WORD_RE.findall(t)
        n = len(words)
        word_count.append(n)
        unique_ratio.append(len({w.lower() for w in words}) / n if n else 1.0)
        avg_word_len.append(sum(len(w) for w in words) / n if n else 0.0)
        letters = sum(c.isalpha() for c in t)
        caps_ratio.append(sum(c.isupper() for c in t) / letters if letters else 0.0)
        digit_ratio.append(sum(c.isdigit() for c in t) / len(t) if t else 0.0)
        repeated.append(bool(_REPEAT_RE.search(t)))
        sentences.append(sum(1 for s in _SENT_SPLIT.split(t) if s.strip()))

    char_len = df["full_text"].fillna("").str.len()
    df["char_len"] = char_len.astype(int)
    df["word_count"] = word_count
    df["avg_word_len"] = avg_word_len
    df["unique_word_ratio"] = unique_ratio
    df["caps_ratio"] = caps_ratio
    df["digit_ratio"] = digit_ratio
    df["repeated_char_flag"] = repeated
    df["sentence_count"] = sentences
    df["exclaim_count"] = df["full_text"].fillna("").str.count("!").astype(int)
    df["exclaim_ratio"] = df["exclaim_count"] / np.maximum(char_len, 1)
    df["question_ratio"] = df["full_text"].fillna("").str.count(r"\?") / np.maximum(char_len, 1)
    df["title_len"] = df["title"].fillna("").str.len().astype(int)
    df["has_title"] = df["title_len"] > 0
    return df


def _near_duplicate_similarity(norm: pd.Series, groups: dict, block: int = 500) -> np.ndarray:
    """Max TF-IDF cosine similarity of each review to any *other* review of the same product."""
    out = np.zeros(len(norm), dtype=float)
    if len(norm) == 0:
        return out
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    try:
        X = vec.fit_transform(norm.tolist()).tocsr()
    except ValueError:  # empty vocabulary
        return out
    for _, idx in groups.items():
        n = len(idx)
        if n < 2:
            continue
        sub = X[idx]
        for start in range(0, n, block):
            blk = sub[start:start + block]
            sim = (blk @ sub.T).toarray()
            rows = np.arange(blk.shape[0])
            sim[rows, start + rows] = 0.0
            out[idx[start:start + blk.shape[0]]] = sim.max(axis=1)
    return out


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-review features: duplication, user behaviour, rating deviation, day clustering."""
    df = df.reset_index(drop=True).copy()
    norm = df["full_text"].fillna("").map(normalise_text)
    has_text = norm.str.len() > 0

    # --- duplication -------------------------------------------------------
    key = norm.where(has_text, other=pd.NA)
    df["dup_text_global"] = (key.groupby(key).transform("size") - 1).fillna(0).astype(int)
    pkey = df["parent_asin"].astype(str) + "||" + norm
    df["dup_text_product"] = (pkey.groupby(pkey).transform("size") - 1).where(has_text, 0).astype(int)

    groups = df.groupby("parent_asin").indices
    log.info("Computing within-product near-duplicate similarity for %s products", f"{len(groups):,}")
    df["max_sim_in_product"] = _near_duplicate_similarity(norm, groups)
    df.loc[~has_text, "max_sim_in_product"] = 0.0

    # --- rating deviation (leave-one-out product mean) ---------------------
    g = df.groupby("parent_asin")["rating"]
    n = g.transform("size")
    loo = (g.transform("sum") - df["rating"]) / (n - 1).where(n > 1)
    df["rating_deviation"] = df["rating"] - loo   # NaN when only one review
    df["rating_extreme"] = df["rating"].isin([1.0, 5.0])

    # --- user behaviour (within the sample) --------------------------------
    ug = df.groupby("user_id")
    df["user_review_count"] = ug["rating"].transform("size").astype(int)
    df["user_mean_rating"] = ug["rating"].transform("mean")
    ext_share = df["rating_extreme"].astype(float).groupby(df["user_id"]).transform("mean")
    df["user_all_extreme"] = (df["user_review_count"] >= 3) & (ext_share >= 1.0)
    day = df["timestamp"].dt.normalize()
    df["user_same_day_count"] = day.groupby([df["user_id"], day]).transform("size").astype(int)

    # --- day clustering (burstiness proxy at review level) -----------------
    df["day_cluster_size"] = day.groupby([df["parent_asin"], day]).transform("size").astype(int)
    df["day_cluster_share"] = df["day_cluster_size"] / n
    return df
