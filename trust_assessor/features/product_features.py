"""Aggregate review-level data and metadata into one row per product."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COMPLETENESS_FIELDS = {
    "has_title": "Title",
    "has_description": "Description",
    "has_features": "Feature bullets",
    "has_images": "Images",
    "has_price": "Price",
    "has_store": "Store / brand name",
    "has_categories": "Category path",
    "has_details": "Product details",
}


def _max_in_window(days: np.ndarray, window: int = 7) -> int:
    d = np.sort(days)
    right = np.searchsorted(d, d + window, side="left")
    return int((right - np.arange(len(d))).max())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _per_product_extras(reviews: pd.DataFrame) -> pd.DataFrame:
    """Features that need an ordered look at each product's reviews."""
    day_idx = (reviews["timestamp"].dt.normalize() - pd.Timestamp("1995-01-01")).dt.days.to_numpy()
    reviews = reviews.assign(_day=day_idx)
    rows = []
    for asin, g in reviews.sort_values("timestamp").groupby("parent_asin", sort=False):
        n = len(g)
        r = g["rating"].to_numpy()
        burst = _max_in_window(g["_day"].to_numpy()) / n if n >= 10 else float("nan")
        if n >= 10:
            half = n // 2
            gap = float(abs(r[half:].mean() - r[:half].mean()))
        else:
            gap = float("nan")
        corr = float("nan")
        if "sent_score" in g:
            m = g["sent_score"].notna().to_numpy()
            if m.sum() >= 10:
                corr = _spearman(g["sent_score"].to_numpy()[m], r[m])
        rows.append((asin, burst, gap, corr))
    return pd.DataFrame(rows, columns=["parent_asin", "burst_ratio", "rating_trend_gap", "sent_rating_corr"])


def build_product_table(reviews: pd.DataFrame, meta: pd.DataFrame, flag_threshold: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (products, dataset_stats)."""
    r = reviews
    g = r.groupby("parent_asin")

    base = g.agg(
        n_reviews=("review_id", "size"),
        mean_rating=("rating", "mean"),
        rating_std=("rating", "std"),
        verified_share=("verified_purchase", "mean"),
        mean_helpful_votes=("helpful_vote", "mean"),
        first_review=("timestamp", "min"),
        last_review=("timestamp", "max"),
        mean_word_count=("word_count", "mean"),
    )
    base["rating_std"] = base["rating_std"].fillna(0.0)

    # rating distribution
    dist = (
        r.assign(star=r["rating"].round().astype(int))
        .groupby(["parent_asin", "star"]).size().unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    )
    dist.columns = [f"n_star_{c}" for c in dist.columns]
    base = base.join(dist)
    base["share_5star"] = base["n_star_5"] / base["n_reviews"]
    base["share_1star"] = base["n_star_1"] / base["n_reviews"]
    base["rating_agreement_share"] = (
        r.assign(_in=(r["rating"] - r.groupby("parent_asin")["rating"].transform("mean")).abs() <= 1.0)
        .groupby("parent_asin")["_in"].mean()
    )

    # sentiment aggregation (only over reviews that were actually scored)
    if "sent_label" in r and r["sent_label"].notna().any():
        scored = r[r["sent_label"].notna()]
        sg = scored.groupby("parent_asin")
        sent = pd.DataFrame({
            "n_sentiment": sg.size(),
            "sent_pos_share": sg["sent_label"].apply(lambda s: (s == "positive").mean()),
            "sent_neu_share": sg["sent_label"].apply(lambda s: (s == "neutral").mean()),
            "sent_neg_share": sg["sent_label"].apply(lambda s: (s == "negative").mean()),
            "sent_mean_score": sg["sent_score"].mean(),
        })
        base = base.join(sent)
    for col in ("n_sentiment", "sent_pos_share", "sent_neu_share", "sent_neg_share", "sent_mean_score"):
        if col not in base:
            base[col] = np.nan
    base["n_sentiment"] = base["n_sentiment"].fillna(0).astype(int)

    # suspicious-signal aggregation
    base["suspicious_share"] = g["suspicion_flag"].mean()
    base["mean_suspicion"] = g["suspicion_score"].mean()
    base["duplicate_share"] = g["sig_exact_duplicate"].mean()
    base["near_duplicate_share"] = g["sig_near_duplicate"].mean()
    base["anomaly_flag_share"] = (
        r.assign(_a=(r["anomaly_score"] >= 0.95).where(r["anomaly_score"].notna()))
        .groupby("parent_asin")["_a"].mean()
        if r["anomaly_score"].notna().any() else np.nan
    )
    base["n_suspicious"] = g["suspicion_flag"].sum().astype(int)

    # review-helpfulness model output
    if r["helpful_prob"].notna().any():
        base["helpful_prob_mean"] = g["helpful_prob"].mean()
        base["helpful_prob_pct"] = base["helpful_prob_mean"].rank(pct=True)
    else:
        base["helpful_prob_mean"] = np.nan
        base["helpful_prob_pct"] = np.nan

    base = base.join(_per_product_extras(r).set_index("parent_asin"))
    products = base.reset_index()
    products["parent_asin"] = products["parent_asin"].astype(str)

    # metadata
    m = meta.copy()
    m["parent_asin"] = m["parent_asin"].astype(str)
    products = products.merge(m, on="parent_asin", how="left")
    products["has_metadata"] = products["has_metadata"].fillna(False).astype(bool)
    products["title"] = products["meta_title"].where(
        products["meta_title"].notna() & (products["meta_title"].astype(str).str.len() > 0),
        "Product " + products["parent_asin"],
    ).astype(str)
    products["category"] = products["main_category"].astype("string").fillna("Unknown").astype(str)

    desc_len = products["description"].fillna("").astype(str).str.len()
    mt = products["has_metadata"]
    products["has_title"] = mt & products["meta_title"].notna()
    products["has_description"] = mt & (desc_len >= 20)
    products["has_features"] = mt & (products["n_features"].fillna(0) > 0)
    products["has_images"] = mt & (products["n_images"].fillna(0) > 0)
    products["has_price"] = mt & products["price"].notna()
    products["has_store"] = mt & products["store"].notna() & (products["store"].astype(str).str.len() > 0)
    products["has_categories"] = mt & (products["n_categories"].fillna(0) > 0)
    products["has_details"] = mt & (products["n_details"].fillna(0) > 0)

    for col in ("first_review", "last_review"):
        products[col] = pd.to_datetime(products[col])

    stats = {
        "n_reviews": int(len(r)),
        "n_products": int(len(products)),
        "n_products_with_metadata": int(products["has_metadata"].sum()),
        "global_mean_rating": float(r["rating"].mean()),
        "global_verified_share": float(r["verified_purchase"].mean()),
        "suspicion_flag_threshold": float(flag_threshold),
        "suspicion_method": "rules+isolation_forest" if r["anomaly_score"].notna().any() else "rules_only",
        "sentiment_scored_reviews": int(r["sent_label"].notna().sum()) if "sent_label" in r else 0,
        "sentiment_method": (
            str(r["sentiment_method"].dropna().iloc[0])
            if "sentiment_method" in r and r["sentiment_method"].notna().any() else "none"
        ),
        "helpfulness_model_applied": bool(r["helpful_prob"].notna().any()),
        "date_min": str(r["timestamp"].min().date()),
        "date_max": str(r["timestamp"].max().date()),
    }
    log.info("Product table: %s products (%s with metadata)", f"{len(products):,}", f"{stats['n_products_with_metadata']:,}")
    return products, stats


def parse_json_list(value: Any) -> list:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        out = json.loads(value)
        return out if isinstance(out, list) else []
    except (TypeError, ValueError):
        return []


def parse_json_dict(value: Any) -> dict:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        out = json.loads(value)
        return out if isinstance(out, dict) else {}
    except (TypeError, ValueError):
        return {}
