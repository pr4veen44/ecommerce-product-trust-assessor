"""SHAP explanations for the supported ML models (XGBoost helpfulness, IsolationForest anomaly)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from trust_assessor.ml import registry
from trust_assessor.ml.helpfulness import FEATURE_LABELS, make_matrix
from trust_assessor.ml.suspicious import build_matrix as anomaly_matrix

log = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _tree_explainer(key: str):
    import shap

    if key == "helpfulness":
        model = registry.helpfulness_model()
    else:
        bundle = registry.anomaly_bundle()
        model = bundle["model"] if bundle else None
    return shap.TreeExplainer(model) if model is not None else None


def clear_cache() -> None:
    _tree_explainer.cache_clear()


def global_helpfulness_importance() -> dict[str, Any] | None:
    return registry.metrics(registry.HELPFULNESS_SHAP)


def explain_helpfulness(review: pd.Series, top_k: int = 12) -> dict[str, Any] | None:
    """Local SHAP explanation of the helpfulness prediction for one review (log-odds scale)."""
    model = registry.helpfulness_model()
    info = registry.metrics(registry.HELPFULNESS_METRICS)
    if model is None or not info:
        return None
    features = info["features"]
    X = make_matrix(pd.DataFrame([review]), features)
    explainer = _tree_explainer("helpfulness")
    sv = np.asarray(explainer.shap_values(X))[0]
    pred = float(model.predict_proba(X)[0, 1])
    items = [
        {"feature": f, "label": FEATURE_LABELS[f], "value": None if pd.isna(X.iloc[0][f]) else float(X.iloc[0][f]),
         "shap": float(v)} for f, v in zip(features, sv)
    ]
    items.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return {"prediction": pred, "base_value": float(np.ravel(explainer.expected_value)[0]), "items": items[:top_k],
            "unit": "log-odds of 'helpful'"}


def explain_anomaly(review: pd.Series, top_k: int = 10) -> dict[str, Any] | None:
    """Local SHAP attribution for the IsolationForest score of one review.

    Returns None if the explainer cannot be built for the installed library versions;
    callers then fall back to the rule-based explanation.
    """
    bundle = registry.anomaly_bundle()
    if bundle is None:
        return None
    try:
        explainer = _tree_explainer("anomaly")
        feats = bundle["features"]
        X = anomaly_matrix(pd.DataFrame([review]))[feats].fillna(pd.Series(bundle["medians"]))
        sv = np.asarray(explainer.shap_values(X))[0]
    except Exception as exc:  # noqa: BLE001 - optional explanation; never break the page
        log.warning("IsolationForest SHAP unavailable: %s", exc)
        return None
    items = [{"feature": f, "value": float(X.iloc[0][f]), "shap": float(v)} for f, v in zip(feats, sv)]
    # In shap's IsolationForest output, NEGATIVE values push toward "more anomalous".
    items.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return {"items": items[:top_k],
            "note": "Negative SHAP values push the review toward 'more unusual' in the IsolationForest."}
