"""Cleaning of raw review / metadata samples."""

from __future__ import annotations

import hashlib
import html
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_text(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    s = s.map(html.unescape)
    s = s.str.replace(_TAG_RE, " ", regex=True).str.replace(_WS_RE, " ", regex=True).str.strip()
    return s


def _make_review_id(row: tuple) -> str:
    raw = "|".join("" if v is None else str(v) for v in row)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def clean_reviews(raw: pd.DataFrame, min_reviews_per_product: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return cleaned reviews plus a report of what was removed and why."""
    report: dict[str, Any] = {"raw_rows": int(len(raw))}
    df = raw.copy()

    df["parent_asin"] = df["parent_asin"].astype("string").str.strip()
    df["asin"] = df["asin"].astype("string").str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)
    df["n_review_images"] = pd.to_numeric(df["n_review_images"], errors="coerce").fillna(0).astype(int)
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", errors="coerce")
    df["title"] = _clean_text(df["title"])
    df["text"] = _clean_text(df["text"])

    before = len(df)
    df = df[df["parent_asin"].notna() & (df["parent_asin"].str.len() > 0)]
    report["dropped_missing_product_id"] = int(before - len(df))

    before = len(df)
    df = df[df["rating"].between(1, 5)]
    report["dropped_invalid_rating"] = int(before - len(df))

    before = len(df)
    df = df[(df["title"].str.len() + df["text"].str.len()) >= 3]
    report["dropped_empty_review"] = int(before - len(df))

    before = len(df)
    df = df[df["timestamp"].notna()]
    report["dropped_invalid_timestamp"] = int(before - len(df))

    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "parent_asin", "timestamp", "text"])
    report["dropped_exact_duplicate_rows"] = int(before - len(df))

    df["user_id"] = df["user_id"].astype("string")
    ids = [
        _make_review_id(r)
        for r in zip(df["user_id"], df["parent_asin"], df["timestamp"].astype("int64"), df["text"].str[:60])
    ]
    df["review_id"] = ids
    df = df.drop_duplicates(subset=["review_id"])
    # Users without an id must not be grouped together in user-level features.
    df["user_id"] = df["user_id"].fillna("unknown:" + df["review_id"])
    df["rating"] = df["rating"].astype(float)

    counts = df["parent_asin"].value_counts()
    keep = counts[counts >= min_reviews_per_product].index
    report["products_before_min_reviews_filter"] = int(len(counts))
    before = len(df)
    df = df[df["parent_asin"].isin(keep)]
    report["dropped_by_min_reviews_filter"] = int(before - len(df))
    report["min_reviews_per_product"] = int(min_reviews_per_product)
    report["clean_rows"] = int(len(df))
    report["clean_products"] = int(df["parent_asin"].nunique())

    df["full_text"] = (df["title"] + ". " + df["text"]).str.strip(" .")
    df = df.sort_values(["parent_asin", "timestamp"]).reset_index(drop=True)
    for k, v in report.items():
        log.info("cleaning | %-36s %s", k, f"{v:,}" if isinstance(v, int) else v)
    return df, report


def clean_metadata(meta: pd.DataFrame, product_ids: pd.Series) -> pd.DataFrame:
    """Left-align metadata to the products present in the cleaned reviews.

    Products without metadata are kept with ``has_metadata = False``; no values are invented.
    """
    base = pd.DataFrame({"parent_asin": pd.Series(product_ids).astype("string").unique()})
    if meta is None or meta.empty:
        out = base.copy()
        for col in ("meta_title", "main_category", "store", "description", "features",
                    "categories", "details", "image_url"):
            out[col] = pd.NA
        for col in ("price", "meta_avg_rating", "meta_rating_number"):
            out[col] = np.nan
        for col in ("n_features", "n_categories", "n_details", "n_images", "n_videos"):
            out[col] = 0
        out["has_metadata"] = False
    else:
        meta = meta.copy()
        meta["parent_asin"] = meta["parent_asin"].astype("string")
        meta = meta.drop_duplicates("parent_asin")
        out = base.merge(meta, on="parent_asin", how="left")
        out["has_metadata"] = out["meta_title"].notna() | out["description"].notna() | out["store"].notna()
        for col in ("n_features", "n_categories", "n_details", "n_images", "n_videos"):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    for col in ("meta_title", "main_category", "store", "description", "image_url"):
        out[col] = out[col].astype("string").str.strip()
    out["has_metadata"] = out["has_metadata"].astype(bool)
    return out
