"""Dataset preparation pipeline: ingest -> clean -> features -> sentiment -> product table."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trust_assessor.config import Settings
from trust_assessor.data import ingestion, preprocessing
from trust_assessor.data.storage import DataStore, read_json, write_json, write_products, write_reviews
from trust_assessor.features.product_features import build_product_table
from trust_assessor.features.review_features import add_context_features, add_text_features
from trust_assessor.features.suspicion_rules import compute_rule_signals
from trust_assessor.nlp import sentiment as sentiment_mod

log = logging.getLogger(__name__)

META_CLEAN = "meta_clean.parquet"


def init_ml_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Rules-only baseline for the suspicion columns; ML training later overwrites them."""
    df = df.copy()
    df["anomaly_score"] = np.nan
    df["suspicion_score"] = df["rule_score"]
    df["suspicion_flag"] = df["suspicion_score"] >= threshold
    df["suspicion_reasons"] = df["rule_reasons"]
    df["helpful_prob"] = np.nan
    return df


def rebuild_products(settings: Settings) -> dict:
    """Rebuild products.parquet + dataset_stats.json from the stored reviews and clean metadata."""
    reviews = pd.read_parquet(settings.reviews_path)
    meta = pd.read_parquet(settings.processed_dir / META_CLEAN)
    products, stats = build_product_table(reviews, meta, settings.suspicion_flag_threshold)
    write_products(products, settings)
    write_json(settings.dataset_stats_path, stats)
    return stats


def prepare_dataset(settings: Settings, refresh: bool = False, skip_sentiment: bool = False) -> dict:
    settings.ensure_dirs()
    raw = ingestion.fetch_reviews(settings, refresh=refresh)
    reviews, clean_report = preprocessing.clean_reviews(raw, settings.min_reviews_per_product)
    if reviews.empty:
        raise RuntimeError(
            "No products have enough reviews after cleaning. Increase SAMPLE_REVIEWS, "
            "set SCAN_MULTIPLIER, lower MIN_REVIEWS_PER_PRODUCT, or choose a smaller category."
        )

    meta_raw = ingestion.fetch_metadata(settings, set(reviews["parent_asin"].astype(str)), refresh=refresh)
    meta = preprocessing.clean_metadata(meta_raw, reviews["parent_asin"])
    meta.to_parquet(settings.processed_dir / META_CLEAN, index=False)

    log.info("Engineering review features ...")
    reviews = add_text_features(reviews)
    reviews = add_context_features(reviews)

    if skip_sentiment:
        settings = _with_backend(settings, "none")
    sent = sentiment_mod.run_sentiment(reviews, settings)
    reviews = pd.concat([reviews.drop(columns=[c for c in sent.columns if c in reviews.columns]), sent], axis=1)

    reviews = compute_rule_signals(reviews)
    reviews = init_ml_columns(reviews, settings.suspicion_flag_threshold)
    reviews["parent_asin"] = reviews["parent_asin"].astype(str)
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["asin"] = reviews["asin"].astype(str)
    write_reviews(reviews, settings)

    stats = rebuild_products(settings)

    if reviews["sent_label"].notna().any():
        write_json(settings.sentiment_report_path,
                   sentiment_mod.proxy_agreement_report(reviews, sentiment_mod.method_name(settings)))
    elif settings.sentiment_report_path.exists():
        settings.sentiment_report_path.unlink()

    version = (
        f"amazon-reviews-2023/{settings.category}/n{settings.sample_reviews}/seed{settings.seed}/"
        f"{datetime.now(timezone.utc):%Y%m%d}"
    )
    manifest = {
        "dataset_version": version,
        "source": f"https://huggingface.co/datasets/{settings.hf_repo}",
        "category": settings.category,
        "requested_sample_reviews": settings.sample_reviews,
        "scan_multiplier": settings.scan_multiplier,
        "seed": settings.seed,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cleaning_report": clean_report,
        "stats": stats,
        "sampling_note": (
            "Reviews are the first N lines of the category file"
            + (" (randomly subsampled from a larger prefix)" if settings.scan_multiplier > 1 else "")
            + ", so the sample is not guaranteed to be representative of the whole category."
        ),
    }
    write_json(settings.dataset_manifest_path, manifest)
    DataStore.clear_cache()
    log.info("Prepared %s reviews / %s products. Dataset version: %s",
             f"{stats['n_reviews']:,}", f"{stats['n_products']:,}", version)
    return manifest


def _with_backend(settings: Settings, backend: str) -> Settings:
    from dataclasses import replace

    return replace(settings, sentiment_backend=backend)


def load_dataset_manifest(settings: Settings) -> dict | None:
    return read_json(settings.dataset_manifest_path)
