"""Parquet I/O helpers and a small cached accessor used by the UI and services."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from trust_assessor.config import Settings, get_settings

log = logging.getLogger(__name__)

REVIEW_ROW_GROUP = 20_000


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("Could not read %s", path)
        return None


def write_reviews(df: pd.DataFrame, settings: Settings) -> None:
    """Persist reviews sorted by product so single-product reads can use row-group pruning."""
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["parent_asin", "timestamp"]).reset_index(drop=True)
    df.to_parquet(settings.reviews_path, index=False, row_group_size=REVIEW_ROW_GROUP)
    DataStore.clear_cache()


def write_products(df: pd.DataFrame, settings: Settings) -> None:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.products_path, index=False)
    DataStore.clear_cache()


class DataStore:
    """Read access to processed Parquet data with light in-memory caching."""

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    # ---- availability ------------------------------------------------------
    def has_processed_data(self) -> bool:
        return self.s.products_path.exists() and self.s.reviews_path.exists()

    @staticmethod
    def clear_cache() -> None:
        _products_cached.cache_clear()
        _reviews_for_product.cache_clear()

    # ---- reads -------------------------------------------------------------
    def products(self) -> pd.DataFrame:
        return _products_cached(str(self.s.products_path), self._mtime(self.s.products_path)).copy()

    def product(self, asin: str) -> pd.Series | None:
        df = _products_cached(str(self.s.products_path), self._mtime(self.s.products_path))
        hit = df[df["parent_asin"] == asin]
        return None if hit.empty else hit.iloc[0]

    def reviews_for(self, asin: str) -> pd.DataFrame:
        return _reviews_for_product(str(self.s.reviews_path), self._mtime(self.s.reviews_path), asin).copy()

    def reviews_all(self, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.s.reviews_path, columns=columns)

    def review_row(self, review_id: str) -> pd.Series | None:
        df = pd.read_parquet(self.s.reviews_path, filters=[("review_id", "==", review_id)])
        return None if df.empty else df.iloc[0]

    @staticmethod
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0


@lru_cache(maxsize=2)
def _products_cached(path: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache on rewrite
    return pd.read_parquet(path)


@lru_cache(maxsize=128)
def _reviews_for_product(path: str, mtime: float, asin: str) -> pd.DataFrame:
    return pd.read_parquet(path, filters=[("parent_asin", "==", asin)])
