"""Train the ML models, write predictions back into the processed data and refresh products."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from trust_assessor.config import Settings
from trust_assessor.data.storage import read_json, write_json, write_reviews
from trust_assessor.ml import registry
from trust_assessor.ml.helpfulness import InsufficientDataError, train_helpfulness
from trust_assessor.ml.suspicious import combine_suspicion, fit_isolation_forest
from trust_assessor.services.pipeline import rebuild_products

log = logging.getLogger(__name__)


def train_all(settings: Settings) -> dict:
    if not settings.reviews_path.exists():
        raise FileNotFoundError("No processed reviews found. Run: python scripts/prepare_dataset.py")
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(settings.reviews_path).reset_index(drop=True)
    log.info("Loaded %s reviews / %s products", f"{len(df):,}", f"{df['parent_asin'].nunique():,}")

    # 1) Unsupervised anomaly model + combined suspicion score ------------------
    pct, anomaly_report = fit_isolation_forest(df, settings, settings.models_dir / registry.ANOMALY_MODEL)
    combined = combine_suspicion(df["rule_score"], pct, df["rule_reasons"])
    df["anomaly_score"] = combined["anomaly_score"]
    df["suspicion_score"] = combined["suspicion_score"]
    df["suspicion_reasons"] = combined["suspicion_reasons"]
    df["suspicion_flag"] = df["suspicion_score"] >= settings.suspicion_flag_threshold
    anomaly_report.update({
        "flag_threshold": settings.suspicion_flag_threshold,
        "n_flagged_reviews": int(df["suspicion_flag"].sum()),
        "share_flagged_reviews": float(df["suspicion_flag"].mean()),
        "share_flagged_by_rules_only": float((df["rule_score"] >= settings.suspicion_flag_threshold).mean()),
    })
    write_json(settings.models_dir / registry.ANOMALY_REPORT, anomaly_report)

    # 2) Supervised helpfulness model ------------------------------------------
    helpfulness_info: dict = {"trained": False}
    try:
        res = train_helpfulness(df, settings)
        res.model.save_model(str(settings.models_dir / registry.HELPFULNESS_MODEL))
        write_json(settings.models_dir / registry.HELPFULNESS_METRICS, res.metrics)
        write_json(settings.models_dir / registry.HELPFULNESS_SHAP, res.shap_global)
        df["helpful_prob"] = res.oof_pred
        helpfulness_info = {"trained": True, "roc_auc_test": res.metrics["xgboost"]["roc_auc"],
                            "pr_auc_test": res.metrics["xgboost"]["pr_auc"]}
    except InsufficientDataError as exc:
        log.warning("Helpfulness model NOT trained: %s", exc)
        for name in (registry.HELPFULNESS_MODEL, registry.HELPFULNESS_METRICS, registry.HELPFULNESS_SHAP):
            (settings.models_dir / name).unlink(missing_ok=True)
        df["helpful_prob"] = float("nan")
        helpfulness_info = {"trained": False, "reason": str(exc)}

    # 3) Persist predictions, rebuild product table, write manifest -----------------
    write_reviews(df, settings)
    stats = rebuild_products(settings)

    dataset_manifest = read_json(settings.dataset_manifest_path) or {}
    version = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    manifest = {
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": dataset_manifest.get("dataset_version"),
        "anomaly_model": {"algorithm": "IsolationForest", "contamination": settings.anomaly_contamination},
        "helpfulness_model": helpfulness_info,
        "suspicion_method": stats["suspicion_method"],
        "seed": settings.seed,
    }
    write_json(settings.model_manifest_path, manifest)
    log.info("Training complete. Model version %s", version)
    return manifest
