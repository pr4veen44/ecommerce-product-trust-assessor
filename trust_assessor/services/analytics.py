"""Aggregations that feed the dashboard and model-insight pages."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from trust_assessor.config import get_settings
from trust_assessor.data.storage import DataStore, read_json
from trust_assessor.db import repository as repo
from trust_assessor.features.suspicion_rules import RULE_CATALOG
from trust_assessor.scoring.config import COMPONENT_KEYS, COMPONENT_LABELS

_CACHE: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


def clear_cache() -> None:
    _CACHE.clear()


def dashboard_summary() -> dict[str, Any]:
    return _cached("dashboard", 10.0, _dashboard_summary)


def _dashboard_summary() -> dict[str, Any]:
    rows = repo.latest_assessments_table()
    scores = [r["trust_score"] for r in rows if r["trust_score"] is not None]
    grades = Counter(r["grade"] for r in rows)
    comp_vals: dict[str, list[float]] = {k: [] for k in COMPONENT_KEYS}
    for r in rows:
        for c in r["components"]:
            if c["available"] and c["score"] is not None:
                comp_vals[c["key"]].append(c["score"])
    means = {COMPONENT_LABELS[k]: float(np.mean(v)) for k, v in comp_vals.items() if v}
    avail = {COMPONENT_LABELS[k]: len(v) / max(len(rows), 1) for k, v in comp_vals.items()}
    return {
        "n_products": len(rows),
        "n_scored": len(scores),
        "scores": scores,
        "mean_score": float(np.mean(scores)) if scores else None,
        "median_score": float(np.median(scores)) if scores else None,
        "grades": dict(grades),
        "component_means": means,
        "component_availability": avail,
        "mean_coverage": float(np.mean([r["evidence_coverage"] for r in rows])) if rows else None,
    }


def review_flag_stats() -> dict[str, Any]:
    def compute():
        s = get_settings()
        df = pd.read_parquet(s.reviews_path, columns=["suspicion_flag", "suspicion_score"])
        return {"n_reviews": int(len(df)), "n_flagged": int(df["suspicion_flag"].sum()),
                "share_flagged": float(df["suspicion_flag"].mean()),
                "suspicion_scores": df["suspicion_score"].to_numpy()}

    return _cached("flags", 30.0, compute)


def dataset_rule_rates() -> dict[str, float]:
    """Share of all reviews in the sample triggering each rule (keyed by human label)."""
    def compute():
        s = get_settings()
        cols = [f"sig_{k}" for k in RULE_CATALOG]
        df = pd.read_parquet(s.reviews_path, columns=cols)
        return {RULE_CATALOG[k]["label"]: float(df[f"sig_{k}"].mean()) for k in RULE_CATALOG}

    return _cached("rule_rates", 60.0, compute)


def product_rule_rates(reviews: pd.DataFrame) -> dict[str, float]:
    return {RULE_CATALOG[k]["label"]: float(reviews[f"sig_{k}"].mean()) if len(reviews) else 0.0 for k in RULE_CATALOG}


def pipeline_info() -> dict[str, Any]:
    s = get_settings()
    return {
        "dataset": read_json(s.dataset_manifest_path),
        "stats": read_json(s.dataset_stats_path),
        "models": read_json(s.model_manifest_path),
        "sentiment_report": read_json(s.sentiment_report_path),
        "has_data": DataStore(s).has_processed_data(),
    }
