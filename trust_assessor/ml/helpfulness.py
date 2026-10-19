"""Supervised XGBoost model: will a review be marked helpful by other shoppers?

Why this target? ``helpful_vote`` is the one genuine, observed outcome label in
the dataset, so it supports honest train / test evaluation. It measures how
*useful* a review is to readers; it is **not** a measure of authenticity. In the
trust score it only feeds a low-weight "review informativeness" component.

Leakage controls
----------------
* ``helpful_vote`` and anything derived from it are never used as features.
* Review age is excluded (older reviews accumulate more votes).
* Train / validation / test splits are grouped by product.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from trust_assessor.config import Settings

log = logging.getLogger(__name__)

FEATURE_LABELS = {
    "char_len": "Review length (characters)", "word_count": "Word count", "avg_word_len": "Average word length",
    "unique_word_ratio": "Unique-word ratio", "sentence_count": "Sentence count", "exclaim_ratio": "Exclamation ratio",
    "question_ratio": "Question-mark ratio", "caps_ratio": "Capital-letter ratio", "digit_ratio": "Digit ratio",
    "title_len": "Title length", "has_title": "Has title", "n_review_images": "Images attached",
    "rating": "Star rating", "rating_extreme": "Rating is 1 or 5 stars", "verified_purchase": "Verified purchase",
    "sent_score": "Sentiment score (P(pos) - P(neg))", "sent_abs": "Sentiment strength",
    "sent_rating_gap": "Sentiment vs rating gap", "rating_deviation": "Rating minus product mean",
    "repeated_char_flag": "Has repeated characters",
}
ALL_FEATURES = list(FEATURE_LABELS)


class InsufficientDataError(RuntimeError):
    pass


def make_matrix(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in ("char_len", "word_count", "avg_word_len", "unique_word_ratio", "sentence_count", "exclaim_ratio",
              "question_ratio", "caps_ratio", "digit_ratio", "title_len", "n_review_images", "rating",
              "rating_deviation"):
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("has_title", "rating_extreme", "verified_purchase", "repeated_char_flag"):
        X[c] = df[c].astype(float)
    sent = pd.to_numeric(df["sent_score"], errors="coerce") if "sent_score" in df else pd.Series(np.nan, index=df.index)
    X["sent_score"] = sent
    X["sent_abs"] = sent.abs()
    X["sent_rating_gap"] = (sent - (df["rating"] - 3.0) / 2.0).abs()
    X = X[ALL_FEATURES]
    return X[features] if features else X


@dataclass
class HelpfulnessResult:
    model: XGBClassifier
    features: list[str]
    metrics: dict[str, Any]
    shap_global: dict[str, Any]
    oof_pred: pd.Series


def _xgb(settings: Settings, n_estimators: int, early_stopping: int | None) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=1.0, tree_method="hist", eval_metric="aucpr",
        early_stopping_rounds=early_stopping, n_jobs=settings.effective_jobs, random_state=settings.seed,
    )


def _threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    pred = (p >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def _best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def train_helpfulness(df: pd.DataFrame, settings: Settings) -> HelpfulnessResult:
    import shap
    from scipy.stats import spearmanr

    y = (df["helpful_vote"] >= 1).astype(int).to_numpy()
    X_all = make_matrix(df)
    features = [c for c in ALL_FEATURES if X_all[c].notna().any()]
    X = X_all[features]
    groups = df["parent_asin"].to_numpy()

    n_pos = int(y.sum())
    if len(df) < 500 or n_pos < 50 or (len(df) - n_pos) < 50:
        raise InsufficientDataError(
            f"Not enough labelled data to train and evaluate the helpfulness model "
            f"({len(df)} reviews, {n_pos} helpful). Use a larger SAMPLE_REVIEWS."
        )

    tr, te = next(GroupShuffleSplit(1, test_size=0.20, random_state=settings.seed).split(X, y, groups))
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=settings.seed).split(X.iloc[tr], y[tr], groups[tr]))
    tr_idx, va_idx = tr[tr_i], tr[va_i]
    if len(np.unique(y[te])) < 2 or len(np.unique(y[va_idx])) < 2:
        raise InsufficientDataError("A held-out split contains a single class; use a larger sample.")

    log.info("Training XGBoost helpfulness model on %s rows (%s features), test %s rows",
             f"{len(tr_idx):,}", len(features), f"{len(te):,}")
    clf = _xgb(settings, 600, 30).fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])], verbose=False)
    best_n = int(getattr(clf, "best_iteration", clf.n_estimators - 1)) + 1

    thr = _best_f1_threshold(y[va_idx], clf.predict_proba(X.iloc[va_idx])[:, 1])
    p_te = clf.predict_proba(X.iloc[te])[:, 1]

    # Baseline: standardised logistic regression on the same split, for context.
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(X.iloc[tr_idx], y[tr_idx])
    p_lr = lr.predict_proba(X.iloc[te])[:, 1]

    metrics: dict[str, Any] = {
        "task": "Binary classification: helpful_vote >= 1",
        "split": "GroupShuffleSplit by product (80% train+val / 20% test; 15% of train for early stopping)",
        "n_train": int(len(tr_idx)), "n_validation": int(len(va_idx)), "n_test": int(len(te)),
        "n_test_products": int(len(set(groups[te]))),
        "positive_rate_train": float(y[tr_idx].mean()), "positive_rate_test": float(y[te].mean()),
        "best_iteration": best_n, "features": features,
        "xgboost": {
            "roc_auc": float(roc_auc_score(y[te], p_te)),
            "pr_auc": float(average_precision_score(y[te], p_te)),
            "brier": float(brier_score_loss(y[te], p_te)),
            "log_loss": float(log_loss(y[te], p_te)),
            **_threshold_metrics(y[te], p_te, thr),
        },
        "baseline_logistic_regression": {
            "roc_auc": float(roc_auc_score(y[te], p_lr)),
            "pr_auc": float(average_precision_score(y[te], p_lr)),
            **_threshold_metrics(y[te], p_lr, _best_f1_threshold(y[va_idx], lr.predict_proba(X.iloc[va_idx])[:, 1])),
        },
        "baseline_majority_pr_auc": float(y[te].mean()),
        "threshold_note": "Threshold chosen to maximise F1 on the validation split, then applied to the test split.",
        "caveat": ("Helpfulness is partly driven by review age and visibility, which the dataset does not "
                   "fully expose. Metrics describe prediction of helpful votes only, not authenticity or trust."),
    }

    # --- SHAP (global) on a test sample --------------------------------
    rng = np.random.default_rng(settings.seed)
    sample_idx = rng.choice(te, size=min(2000, len(te)), replace=False)
    explainer = shap.TreeExplainer(clf)
    sv = np.asarray(explainer.shap_values(X.iloc[sample_idx]))
    mean_abs = np.abs(sv).mean(axis=0)
    direction = []
    for j, f in enumerate(features):
        col = X.iloc[sample_idx][f].to_numpy(dtype=float)
        ok = ~np.isnan(col)
        varies = ok.sum() > 10 and np.nanstd(col[ok]) > 0 and np.std(sv[ok, j]) > 0
        rho = spearmanr(col[ok], sv[ok, j]).statistic if varies else np.nan
        direction.append(None if np.isnan(rho) else float(rho))
    gain = clf.get_booster().get_score(importance_type="gain")
    shap_global = {
        "n_rows_explained": int(len(sample_idx)),
        "features": features,
        "labels": [FEATURE_LABELS[f] for f in features],
        "mean_abs_shap": [float(v) for v in mean_abs],
        "direction_corr": direction,
        "gain_importance": [float(gain.get(f, 0.0)) for f in features],
        "base_value": float(np.ravel(explainer.expected_value)[0]),
        "note": "SHAP values are in log-odds units of the XGBoost model; computed on a held-out test sample.",
    }

    # --- out-of-fold predictions for every review (grouped 5-fold) -----------
    oof = np.full(len(df), np.nan)
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        m = _xgb(settings, best_n, None).fit(X.iloc[a], y[a], verbose=False)
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        log.info("OOF fold %s/5 done", k + 1)
    metrics["oof_roc_auc"] = float(roc_auc_score(y, oof))

    return HelpfulnessResult(clf, features, metrics, shap_global, pd.Series(oof, index=df.index))
