"""Unsupervised anomaly detection for review-level suspicious signals.

There are no ground-truth fake-review labels in Amazon Reviews 2023, so this
model is **unsupervised** (IsolationForest) and **cannot be evaluated for
accuracy**. Its output ("how statistically unusual is this review relative to
the rest of the sample?") is combined with the transparent rule signals into a
``suspicion_score``. A high score means "worth a closer look", never "fake".
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from trust_assessor.config import Settings

log = logging.getLogger(__name__)

ANOMALY_FEATURES = [
    "char_len_log", "word_count_log", "avg_word_len", "unique_word_ratio", "exclaim_ratio", "caps_ratio",
    "digit_ratio", "rating", "rating_deviation", "dup_text_global_log", "max_sim_in_product",
    "user_review_count_log", "user_same_day_count", "day_cluster_share", "verified_purchase",
    "sent_score", "sent_rating_gap",
]

ANOMALY_PCT_FLOOR = 0.80   # only the top 20% most unusual reviews contribute
ANOMALY_WEIGHT = 0.60      # maximum contribution of the anomaly model to suspicion_score


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["char_len_log"] = np.log1p(df["char_len"])
    X["word_count_log"] = np.log1p(df["word_count"])
    X["dup_text_global_log"] = np.log1p(df["dup_text_global"])
    X["user_review_count_log"] = np.log1p(df["user_review_count"])
    for c in ("avg_word_len", "unique_word_ratio", "exclaim_ratio", "caps_ratio", "digit_ratio", "rating",
              "rating_deviation", "max_sim_in_product", "user_same_day_count", "day_cluster_share"):
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["verified_purchase"] = df["verified_purchase"].astype(float)
    sent = pd.to_numeric(df["sent_score"], errors="coerce") if "sent_score" in df else np.nan
    X["sent_score"] = sent
    X["sent_rating_gap"] = (sent - (df["rating"] - 3.0) / 2.0).abs()
    return X[ANOMALY_FEATURES]


def fit_isolation_forest(df: pd.DataFrame, settings: Settings, path) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit on all reviews; return (anomaly percentile in [0,1] per review, report)."""
    X = build_matrix(df)
    usable = [c for c in ANOMALY_FEATURES if X[c].notna().any()]   # e.g. drop sentiment features if absent
    X = X[usable]
    medians = X.median().fillna(0.0)
    Xf = X.fillna(medians)

    iso = IsolationForest(
        n_estimators=200, contamination=settings.anomaly_contamination,
        random_state=settings.seed, n_jobs=settings.effective_jobs,
    ).fit(Xf)
    raw = -iso.score_samples(Xf)                       # higher = more anomalous
    ref = np.sort(raw if len(raw) <= 50_000 else np.random.default_rng(settings.seed).choice(raw, 50_000, replace=False))
    pct = np.searchsorted(ref, raw, side="right") / len(ref)

    joblib.dump({"model": iso, "features": usable, "medians": medians.to_dict(), "reference_scores": ref}, path)
    report = {
        "algorithm": "IsolationForest",
        "n_estimators": 200,
        "contamination": settings.anomaly_contamination,
        "features": usable,
        "n_reviews_fitted": int(len(df)),
        "share_predicted_outlier": float((iso.predict(Xf) == -1).mean()),
        "evaluation": ("Not evaluated: Amazon Reviews 2023 has no ground-truth fake-review labels, "
                       "so no accuracy, precision or recall can be reported for this model."),
    }
    return pct, report


def combine_suspicion(rule_score: pd.Series, anomaly_pct: pd.Series | np.ndarray, rule_reasons: pd.Series) -> pd.DataFrame:
    """Noisy-OR combination of the rule score and the (thresholded) anomaly percentile."""
    anomaly = pd.Series(np.asarray(anomaly_pct, dtype=float), index=rule_score.index)
    comp = ((anomaly - ANOMALY_PCT_FLOOR) / (1.0 - ANOMALY_PCT_FLOOR)).clip(0.0, 1.0).fillna(0.0)
    score = 1.0 - (1.0 - rule_score) * (1.0 - ANOMALY_WEIGHT * comp)
    reasons = rule_reasons.copy()
    tag = comp.map(lambda c: "Statistically unusual combination of features (anomaly model)" if c > 0 else "")
    reasons = [f"{r}; {t}" if r and t else (r or t) for r, t in zip(reasons, tag)]
    return pd.DataFrame({"anomaly_score": anomaly, "suspicion_score": score, "suspicion_reasons": reasons})
