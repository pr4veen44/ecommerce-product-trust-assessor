"""Load persisted model artifacts and their metadata (cached)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from trust_assessor.config import get_settings
from trust_assessor.data.storage import read_json

HELPFULNESS_MODEL = "helpfulness_xgb.json"
ANOMALY_MODEL = "anomaly_isolation_forest.joblib"
HELPFULNESS_METRICS = "helpfulness_metrics.json"
HELPFULNESS_SHAP = "helpfulness_shap_global.json"
ANOMALY_REPORT = "anomaly_report.json"


def models_dir() -> Path:
    return get_settings().models_dir


def manifest() -> dict[str, Any] | None:
    return read_json(get_settings().model_manifest_path)


def model_version() -> str | None:
    m = manifest()
    return m.get("model_version") if m else None


def metrics(name: str) -> dict[str, Any] | None:
    return read_json(models_dir() / name)


@lru_cache(maxsize=2)
def _helpfulness(path: str, mtime: float):
    from xgboost import XGBClassifier

    clf = XGBClassifier()
    clf.load_model(path)
    return clf


def helpfulness_model():
    p = models_dir() / HELPFULNESS_MODEL
    return _helpfulness(str(p), p.stat().st_mtime) if p.exists() else None


@lru_cache(maxsize=2)
def _anomaly(path: str, mtime: float):
    return joblib.load(path)


def anomaly_bundle():
    p = models_dir() / ANOMALY_MODEL
    return _anomaly(str(p), p.stat().st_mtime) if p.exists() else None
