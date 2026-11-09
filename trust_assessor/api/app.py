"""FastAPI application exposing the Product Trust Assessor data API."""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from trust_assessor.config import get_settings
from trust_assessor.data.storage import DataStore
from trust_assessor.db import repository as repo
from trust_assessor.explain.score_explainer import decomposition, explain_text
from trust_assessor.explain import shap_explainer
from trust_assessor.features.product_features import COMPLETENESS_FIELDS, parse_json_dict, parse_json_list
from trust_assessor.features.suspicion_rules import RULE_CATALOG, triggered_rules
from trust_assessor.logging_setup import setup_logging
from trust_assessor.ml import registry
from trust_assessor.scoring.config import COMPONENT_DESCRIPTIONS, COMPONENT_KEYS, COMPONENT_LABELS
from trust_assessor.services import analytics
from trust_assessor.services.assessment_service import assess_all, assess_one, ensure_database_ready, load_scoring_config

log = logging.getLogger(__name__)
FRONTEND_PATH = Path(__file__).with_name("static") / "index.html"


def _jsonable(value: Any) -> Any:
    """Convert pandas, NumPy and datetime values into JSON-compatible values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _product_payload(product: dict[str, Any], assessment: dict[str, Any] | None) -> dict[str, Any]:
    payload = {"product": product, "assessment": assessment}
    if assessment:
        payload["assessment"] = {
            **assessment,
            "decomposition": decomposition(assessment),
            "explanation": explain_text(assessment),
        }
    return _jsonable(payload)


def _full_product(asin: str) -> dict[str, Any] | None:
    row = DataStore().product(asin)
    return None if row is None else row.to_dict()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    status = ensure_database_ready()
    log.info("Database status: %s", status)
    if status == "no_data":
        log.warning("No processed data found. Run the dataset and model pipeline first.")
    yield


app = FastAPI(
    title="Product Trust Assessor API",
    description="A frontend-agnostic API for explainable e-commerce product trust assessments.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(FRONTEND_PATH, media_type="text/html")


@app.get("/api/health", tags=["system"])
def health() -> dict[str, Any]:
    settings = get_settings()
    return _jsonable({
        "status": "ok",
        "has_processed_data": DataStore(settings).has_processed_data(),
        "products": repo.count_products(),
        "assessments": repo.count_assessments(),
    })


@app.get("/api/dashboard", tags=["dashboard"])
def dashboard() -> dict[str, Any]:
    info = analytics.pipeline_info()
    if not info["has_data"] or repo.count_products() == 0:
        return _jsonable({"ready": False, "pipeline": info})
    return _jsonable({
        "ready": True,
        "pipeline": info,
        "summary": analytics.dashboard_summary(),
        "review_flags": analytics.review_flag_stats(),
        "scoring_config": load_scoring_config().to_dict(),
    })


@app.get("/api/products", tags=["products"])
def products(
    q: str = "",
    grade: list[str] | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0, le=100),
    max_score: float | None = Query(default=None, ge=0, le=100),
    min_reviews: int = Query(default=0, ge=0),
    sort_by: str = Query(default="trust_score"),
    descending: bool = True,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    allowed_sorts = {"trust_score", "n_reviews", "title", "mean_rating", "evidence_coverage"}
    if sort_by not in allowed_sorts:
        raise HTTPException(status_code=422, detail=f"sort_by must be one of {sorted(allowed_sorts)}")
    rows, total = repo.search_products(q, grade, min_score, max_score, min_reviews, sort_by, descending, limit, offset)
    return _jsonable({"items": rows, "total": total, "limit": limit, "offset": offset})


@app.get("/api/products/{asin}", tags=["products"])
def product(asin: str, include_reviews: bool = False, review_limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
    product_row = repo.get_product(asin)
    if product_row is None or _full_product(asin) is None:
        raise HTTPException(status_code=404, detail="Product not found")
    assessment = repo.latest_assessment(asin)
    if assessment is None:
        assess_one(asin)
        assessment = repo.latest_assessment(asin)
    payload = _product_payload(_full_product(asin) or product_row, assessment)
    if include_reviews:
        reviews = DataStore().reviews_for(asin).sort_values("timestamp", ascending=False).head(review_limit)
        payload["reviews"] = _jsonable(reviews.to_dict("records"))
        payload["review_analytics"] = _jsonable({
            "rule_rates": analytics.product_rule_rates(reviews),
            "dataset_rule_rates": analytics.dataset_rule_rates(),
            "review_count": len(reviews),
            "flagged_count": int(reviews["suspicion_flag"].sum()),
        })
    payload["listing_fields"] = _jsonable({key: {"label": label, "present": bool((_full_product(asin) or {}).get(key))}
                                           for key, label in COMPLETENESS_FIELDS.items()})
    payload["catalogue"] = _jsonable({
        "categories": parse_json_list((payload["product"] or {}).get("categories")),
        "features": parse_json_list((payload["product"] or {}).get("features")),
        "details": parse_json_dict((payload["product"] or {}).get("details")),
    })
    return payload


@app.get("/api/products/{asin}/reviews/{review_id}", tags=["products"])
def review_detail(asin: str, review_id: str) -> dict[str, Any]:
    row = DataStore().review_row(review_id)
    if row is None or row.get("parent_asin") != asin:
        raise HTTPException(status_code=404, detail="Review not found")
    return _jsonable({
        "review": row.to_dict(),
        "rules": triggered_rules(row),
        "anomaly": shap_explainer.explain_anomaly(row),
        "helpfulness": shap_explainer.explain_helpfulness(row),
    })


@app.post("/api/products/{asin}/reassess", tags=["products"])
def reassess(asin: str) -> dict[str, Any]:
    if repo.get_product(asin) is None and DataStore().product(asin) is None:
        raise HTTPException(status_code=404, detail="Product not found")
    assess_one(asin)
    analytics.clear_cache()
    return product(asin)


@app.get("/api/compare", tags=["products"])
def compare(ids: str = Query(..., description="Comma-separated product IDs; up to four")) -> dict[str, Any]:
    asins = list(dict.fromkeys(item.strip() for item in ids.split(",") if item.strip()))[:4]
    if len(asins) < 2:
        raise HTTPException(status_code=422, detail="Provide two to four product IDs")
    missing = [asin for asin in asins if repo.get_product(asin) is None]
    if missing:
        raise HTTPException(status_code=404, detail={"missing_products": missing})
    return _jsonable({"items": [_product_payload(_full_product(asin) or repo.get_product(asin), repo.latest_assessment(asin)) for asin in asins]})


@app.get("/api/history", tags=["history"])
def history(asin: str | None = None, source: str | None = None, limit: int = Query(default=500, ge=1, le=2000)) -> dict[str, Any]:
    return _jsonable({"runs": repo.run_summaries(30), "items": repo.assessment_history(asin, source, limit)})


@app.get("/api/models", tags=["models"])
def models() -> dict[str, Any]:
    info = analytics.pipeline_info()
    return _jsonable({
        **info,
        "versions": repo.version_rows(),
        "review_flags": analytics.review_flag_stats(),
        "helpfulness_metrics": registry.metrics(registry.HELPFULNESS_METRICS),
        "helpfulness_shap": registry.metrics(registry.HELPFULNESS_SHAP),
        "anomaly_report": registry.metrics(registry.ANOMALY_REPORT),
        "rule_catalog": RULE_CATALOG,
        "rule_rates": analytics.dataset_rule_rates(),
    })


@app.get("/api/methodology", tags=["methodology"])
def methodology() -> dict[str, Any]:
    from trust_assessor.scoring.engine import NOT_ASSESSED

    cfg = load_scoring_config()
    return _jsonable({
        "config": cfg.to_dict(),
        "components": [{"key": key, "label": COMPONENT_LABELS[key], "description": COMPONENT_DESCRIPTIONS[key],
                        "weight": cfg.normalised_weights()[key]} for key in COMPONENT_KEYS],
        "not_assessed": [{"item": item, "reason": reason} for item, reason in NOT_ASSESSED],
        "limitations": [
            "The Trust Score is a transparent heuristic and is not validated against ground-truth trust labels.",
            "Suspicious-review signals flag patterns for investigation; they do not prove a review is fake.",
            "Only a sample of one category is analysed and it may not represent the full catalogue.",
            "Duplicate and reviewer-behaviour signals only see reviews inside the sample.",
            "Sentiment can be wrong for sarcasm, short reviews, and mixed opinions.",
            "Review helpfulness is influenced by age and visibility, so informativeness has a low weight.",
            "Metadata is a snapshot; missing metadata is not treated as evidence of poor quality.",
        ],
    })


@app.post("/api/methodology/rescore", tags=["methodology"])
def rescore(config: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from trust_assessor.config import get_settings
    from trust_assessor.scoring.config import ScoringConfig

    try:
        new_config = ScoringConfig.from_dict(config)
        if sum(new_config.weights.values()) <= 0:
            raise ValueError("At least one component weight must be above zero")
        new_config.save(get_settings().scoring_config_path)
        result = assess_all(None, new_config, "reweight")
        analytics.clear_cache()
        return _jsonable(result)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
