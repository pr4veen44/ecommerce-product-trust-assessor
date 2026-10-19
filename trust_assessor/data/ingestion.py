"""Stream a *sample* of one Amazon Reviews 2023 category from Hugging Face.

The dataset's loading script is no longer supported by ``datasets``, so we read
the raw ``.jsonl`` files directly through ``huggingface_hub.HfFileSystem``.
The file is read as a stream and reading stops after the requested number of
lines, so the full category (often many GB) is never downloaded.

Only data returned by the dataset is used. Nothing is scraped or invented.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from trust_assessor.config import Settings

log = logging.getLogger(__name__)

REVIEW_FIELDS = (
    "rating", "title", "text", "images", "asin", "parent_asin",
    "user_id", "timestamp", "helpful_vote", "verified_purchase",
)


def _fs(settings: Settings):
    from huggingface_hub import HfFileSystem  # imported lazily: optional at import time

    return HfFileSystem(token=settings.hf_token or None)


def _repo_path(settings: Settings, sub: str) -> str:
    return f"datasets/{settings.hf_repo}/{sub}"


def _list_dir(settings: Settings, sub: str) -> list[str]:
    fs = _fs(settings)
    entries = fs.ls(_repo_path(settings, sub), detail=False)
    return [e.rsplit("/", 1)[-1] for e in entries]


def available_categories(settings: Settings) -> list[str]:
    names = _list_dir(settings, "raw/review_categories")
    return sorted(n.removesuffix(".jsonl") for n in names if n.endswith(".jsonl"))


def _resolve_file(settings: Settings, sub: str, candidates: list[str]) -> str:
    """Return the repo path of the first candidate file name that exists."""
    names = set(_list_dir(settings, sub))
    for cand in candidates:
        if cand in names:
            return _repo_path(settings, f"{sub}/{cand}")
    raise FileNotFoundError(
        f"None of {candidates} found in '{sub}'. Available (first 40): {sorted(names)[:40]}"
    )


def _stream_jsonl(settings: Settings, path: str, limit: int | None) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects, with a few retries for transient network errors."""
    fs = _fs(settings)
    attempts, produced = 0, 0
    while True:
        try:
            with fs.open(path, "rb") as fh:
                for i, line in enumerate(fh):
                    if i < produced:  # resume after a retry
                        continue
                    if limit is not None and produced >= limit:
                        return
                    line = line.strip()
                    if not line:
                        produced += 1
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass
                    produced += 1
            return
        except (OSError, ConnectionError, TimeoutError) as exc:
            attempts += 1
            if attempts > 3:
                raise
            wait = 2 ** attempts
            log.warning("Network error (%s); retrying in %ss (attempt %s/3)", exc, wait, attempts)
            time.sleep(wait)


# --------------------------------------------------------------------------- reviews
def _slim_review(rec: dict[str, Any]) -> dict[str, Any]:
    images = rec.get("images")
    return {
        "rating": rec.get("rating"),
        "title": rec.get("title"),
        "text": rec.get("text"),
        "n_review_images": len(images) if isinstance(images, list) else 0,
        "asin": rec.get("asin"),
        "parent_asin": rec.get("parent_asin"),
        "user_id": rec.get("user_id"),
        "timestamp": rec.get("timestamp"),
        "helpful_vote": rec.get("helpful_vote"),
        "verified_purchase": rec.get("verified_purchase"),
    }


def fetch_reviews(settings: Settings, refresh: bool = False) -> pd.DataFrame:
    """Download (or load the cached) review sample and cache it as Parquet."""
    path = settings.raw_reviews_path
    if path.exists() and not refresh:
        log.info("Using cached raw review sample: %s", path)
        return pd.read_parquet(path)

    settings.ensure_dirs()
    remote = _resolve_file(settings, "raw/review_categories", [f"{settings.category}.jsonl"])
    n_scan = settings.sample_reviews * settings.scan_multiplier
    log.info("Streaming up to %s review lines from %s", f"{n_scan:,}", remote)

    rows = [
        _slim_review(rec)
        for rec in tqdm(_stream_jsonl(settings, remote, n_scan), total=n_scan, desc="reviews", unit="rev")
    ]
    if not rows:
        raise RuntimeError("No reviews were read from the dataset.")
    df = pd.DataFrame(rows)

    if len(df) > settings.sample_reviews:
        df = df.sample(n=settings.sample_reviews, random_state=settings.seed).reset_index(drop=True)
    log.info("Review sample: %s rows", f"{len(df):,}")
    df.to_parquet(path, index=False)
    return df


# --------------------------------------------------------------------------- metadata
def _first_image_url(images: Any) -> str | None:
    """Metadata images are a dict of lists; be tolerant of other shapes."""
    if isinstance(images, dict):
        for key in ("large", "hi_res", "thumb"):
            vals = images.get(key)
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str) and v.startswith("http"):
                        return v
    elif isinstance(images, list):
        for v in images:
            if isinstance(v, dict):
                for key in ("large", "hi_res", "thumb"):
                    if isinstance(v.get(key), str):
                        return v[key]
            elif isinstance(v, str) and v.startswith("http"):
                return v
    return None


def _count_images(images: Any) -> int:
    if isinstance(images, dict):
        lists = [v for v in images.values() if isinstance(v, list)]
        return max((len(v) for v in lists), default=0)
    return len(images) if isinstance(images, list) else 0


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        f = float(str(value).replace("$", "").replace(",", "").strip())
        return f if np.isfinite(f) and f > 0 else None
    except (TypeError, ValueError):
        return None


def _join_text(value: Any, limit: int) -> str:
    if isinstance(value, list):
        value = " ".join(str(v) for v in value if v)
    return (str(value) if value else "")[:limit]


def _slim_meta(rec: dict[str, Any]) -> dict[str, Any]:
    features = rec.get("features") if isinstance(rec.get("features"), list) else []
    categories = rec.get("categories") if isinstance(rec.get("categories"), list) else []
    details = rec.get("details") if isinstance(rec.get("details"), dict) else {}
    return {
        "parent_asin": rec.get("parent_asin"),
        "meta_title": rec.get("title"),
        "main_category": rec.get("main_category"),
        "store": rec.get("store"),
        "price": _to_float(rec.get("price")),
        "meta_avg_rating": _to_float(rec.get("average_rating")),
        "meta_rating_number": _to_float(rec.get("rating_number")),
        "description": _join_text(rec.get("description"), 3000),
        "features": json.dumps([str(f)[:300] for f in features[:12]]),
        "n_features": len(features),
        "n_categories": len(categories),
        "categories": json.dumps([str(c) for c in categories[:8]]),
        "n_details": len(details),
        "details": json.dumps({str(k): str(v)[:120] for k, v in list(details.items())[:15]}),
        "n_images": _count_images(rec.get("images")),
        "n_videos": len(rec.get("videos")) if isinstance(rec.get("videos"), list) else 0,
        "image_url": _first_image_url(rec.get("images")),
    }


def fetch_metadata(settings: Settings, needed_asins: set[str], refresh: bool = False) -> pd.DataFrame:
    """Stream product metadata until all needed products are found (or the scan limit is hit)."""
    path = settings.raw_meta_path
    if path.exists() and not refresh:
        log.info("Using cached raw metadata sample: %s", path)
        return pd.read_parquet(path)

    settings.ensure_dirs()
    cat = settings.category
    remote = _resolve_file(
        settings, "raw/meta_categories", [f"meta_{cat}.jsonl", f"{cat}.jsonl"]
    )
    log.info("Scanning up to %s metadata lines in %s for %s products",
             f"{settings.meta_scan_limit:,}", remote, f"{len(needed_asins):,}")

    found: dict[str, dict[str, Any]] = {}
    remaining = set(needed_asins)
    bar = tqdm(total=settings.meta_scan_limit, desc="metadata", unit="item")
    for rec in _stream_jsonl(settings, remote, settings.meta_scan_limit):
        bar.update(1)
        asin = rec.get("parent_asin")
        if asin in remaining:
            found[asin] = _slim_meta(rec)
            remaining.discard(asin)
            if not remaining:
                break
    bar.close()

    log.info("Metadata found for %s / %s products (%s without metadata)",
             f"{len(found):,}", f"{len(needed_asins):,}", f"{len(remaining):,}")
    df = pd.DataFrame(list(found.values()))
    if df.empty:
        df = pd.DataFrame(columns=list(_slim_meta({}).keys()))
    df.to_parquet(path, index=False)
    return df
