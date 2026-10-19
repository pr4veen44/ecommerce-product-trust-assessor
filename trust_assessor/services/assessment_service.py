"""Glue between Parquet data, the scoring engine and the SQLite database."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from trust_assessor.config import Settings, get_settings
from trust_assessor.data.storage import DataStore, read_json
from trust_assessor.db import repository as repo
from trust_assessor.db.session import init_db
from trust_assessor.ml import registry
from trust_assessor.scoring.config import ScoringConfig
from trust_assessor.scoring.engine import TrustAssessment, assess_product

log = logging.getLogger(__name__)


def load_scoring_config(settings: Settings | None = None) -> ScoringConfig:
    s = settings or get_settings()
    return ScoringConfig.load(s.scoring_config_path)


def dataset_context(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    return read_json(s.dataset_stats_path) or {"global_mean_rating": 4.0, "suspicion_method": "rules_only"}


def _clean(v: Any) -> Any:
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def sync_products(settings: Settings, products: pd.DataFrame, dataset_version: str | None) -> int:
    rows = []
    for r in products.to_dict("records"):
        title = str(r.get("title") or f"Product {r['parent_asin']}")
        rows.append({
            "parent_asin": r["parent_asin"], "title": title, "title_lower": title.lower(),
            "category": _clean(r.get("category")), "store": _clean(r.get("store")), "price": _clean(r.get("price")),
            "meta_avg_rating": _clean(r.get("meta_avg_rating")),
            "meta_rating_number": _clean(r.get("meta_rating_number")),
            "image_url": _clean(r.get("image_url")), "has_metadata": bool(r.get("has_metadata", False)),
            "n_reviews": int(r["n_reviews"]), "mean_rating": _clean(r.get("mean_rating")),
            "dataset_version": dataset_version, "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
        })
    repo.upsert_products(rows)
    repo.delete_products_not_in(products["parent_asin"])
    return len(rows)


def _assessment_row(a: TrustAssessment, run_id: str, source: str, versions: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_asin": a.parent_asin, "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "run_id": run_id, "source": source, "trust_score": a.trust_score, "raw_score": a.raw_score,
        "grade": a.grade, "confidence": a.confidence, "evidence_coverage": a.evidence_coverage,
        "n_reviews": a.n_reviews, "components": [c.__dict__ for c in a.components],
        "strengths": a.strengths, "warnings": a.warnings, "missing": a.missing, "adjustments": a.adjustments,
        "scoring_config": a.config, "config_hash": a.config_hash, **versions,
    }


def _versions(settings: Settings, ctx: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(settings.dataset_manifest_path) or {}
    return {
        "dataset_version": manifest.get("dataset_version"),
        "model_version": registry.model_version(),
        "sentiment_method": ctx.get("sentiment_method"),
        "suspicion_method": ctx.get("suspicion_method"),
    }


def assess_all(settings: Settings | None = None, cfg: ScoringConfig | None = None, source: str = "batch") -> dict[str, Any]:
    """Load products into SQLite (if needed) and store a fresh assessment for every product."""
    s = settings or get_settings()
    cfg = cfg or load_scoring_config(s)
    init_db()
    products = DataStore(s).products()
    ctx = dataset_context(s)
    versions = _versions(s, ctx)
    manifest = read_json(s.dataset_manifest_path) or {}

    sync_products(s, products, versions["dataset_version"])
    if manifest:
        repo.register_dataset_version({
            "version": manifest["dataset_version"], "category": manifest["category"],
            "sample_reviews": manifest["requested_sample_reviews"], "seed": manifest["seed"],
            "n_reviews": manifest["stats"]["n_reviews"], "n_products": manifest["stats"]["n_products"],
            "manifest": manifest,
        })
    mm = registry.manifest()
    if mm:
        repo.register_model_version({"version": mm["model_version"], "trained_at": mm["trained_at"],
                                     "dataset_version": mm.get("dataset_version"), "manifest": mm})

    run_id = uuid.uuid4().hex[:12]
    assessments = [assess_product(r, ctx, cfg) for r in products.to_dict("records")]
    repo.save_assessments([_assessment_row(a, run_id, source, versions) for a in assessments])
    scored = [a.trust_score for a in assessments if a.trust_score is not None]
    log.info("Run %s: assessed %s products (%s scored, mean %.1f)", run_id, len(assessments), len(scored),
             float(np.mean(scored)) if scored else float("nan"))
    return {"run_id": run_id, "n_products": len(assessments), "n_scored": len(scored),
            "mean_score": float(np.mean(scored)) if scored else None, "config_hash": cfg.fingerprint()}


def assess_one(asin: str, cfg: ScoringConfig | None = None, source: str = "interactive") -> TrustAssessment | None:
    s = get_settings()
    row = DataStore(s).product(asin)
    if row is None:
        return None
    cfg = cfg or load_scoring_config(s)
    ctx = dataset_context(s)
    a = assess_product(row.to_dict(), ctx, cfg)
    repo.save_assessments([_assessment_row(a, uuid.uuid4().hex[:12], source, _versions(s, ctx))])
    return a


def ensure_database_ready() -> str:
    """Called at app start-up. Returns a status string: 'no_data' | 'ready' | 'initialised'."""
    s = get_settings()
    init_db()
    store = DataStore(s)
    if not store.has_processed_data():
        return "no_data"
    manifest = read_json(s.dataset_manifest_path) or {}
    mm = registry.manifest() or {}
    stale = (
        repo.count_products() == 0
        or repo.current_dataset_version() != manifest.get("dataset_version")
        or repo.count_assessments() == 0
        or _db_model_version() != mm.get("model_version")
    )
    if stale:
        log.info("Database is empty or out of date; running a batch assessment ...")
        assess_all(s, source="batch")
        return "initialised"
    return "ready"


def _db_model_version() -> str | None:
    rows = repo.run_summaries(limit=1)
    return rows[0].get("model_version") if rows else None
