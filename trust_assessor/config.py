"""Central, environment-driven configuration.

All values can be overridden through a ``.env`` file (see ``.env.example``) or
real environment variables. Scripts may additionally override selected values
via command-line flags using :func:`dataclasses.replace`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value or not value.strip():
        return default
    p = Path(value.strip()).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    # --- Dataset -----------------------------------------------------------
    hf_repo: str = "McAuley-Lab/Amazon-Reviews-2023"
    hf_token: str = ""
    category: str = "All_Beauty"
    sample_reviews: int = 50_000
    scan_multiplier: int = 1          # >1 reads more lines then randomly subsamples
    meta_scan_limit: int = 400_000    # max metadata lines scanned while matching products
    min_reviews_per_product: int = 5
    seed: int = 42

    # --- Paths -------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    db_path: Path = PROJECT_ROOT / "data" / "trust_assessor.sqlite"
    scoring_config_path: Path = PROJECT_ROOT / "data" / "scoring_config.json"

    # --- Sentiment ---------------------------------------------------------
    sentiment_backend: str = "transformers"   # transformers | lexicon | none
    sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_batch_size: int = 32
    sentiment_max_length: int = 128
    sentiment_max_reviews: int = 0            # 0 = score every review
    sentiment_neutral_band: float = 0.15      # |P(pos) - 0.5| below this -> neutral

    # --- ML ----------------------------------------------------------------
    anomaly_contamination: float = 0.05
    suspicion_flag_threshold: float = 0.50
    n_jobs: int = 0                            # 0 = all cores

    # --- UI ----------------------------------------------------------------
    ui_host: str = "127.0.0.1"
    ui_port: int = 8080
    show_product_images: bool = True
    storage_secret: str = "change-me-locally"
    log_level: str = "INFO"

    # ----- derived paths ----------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_prefix(self) -> str:
        return f"{self.category}_n{self.sample_reviews}_s{self.seed}"

    @property
    def raw_reviews_path(self) -> Path:
        return self.raw_dir / f"{self.raw_prefix}_reviews.parquet"

    @property
    def raw_meta_path(self) -> Path:
        return self.raw_dir / f"{self.raw_prefix}_meta.parquet"

    @property
    def reviews_path(self) -> Path:
        return self.processed_dir / "reviews.parquet"

    @property
    def products_path(self) -> Path:
        return self.processed_dir / "products.parquet"

    @property
    def sentiment_path(self) -> Path:
        return self.processed_dir / "sentiment.parquet"

    @property
    def dataset_manifest_path(self) -> Path:
        return self.processed_dir / "dataset_manifest.json"

    @property
    def dataset_stats_path(self) -> Path:
        return self.processed_dir / "dataset_stats.json"

    @property
    def sentiment_report_path(self) -> Path:
        return self.processed_dir / "sentiment_report.json"

    @property
    def model_manifest_path(self) -> Path:
        return self.models_dir / "manifest.json"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path.as_posix()}"

    @property
    def effective_jobs(self) -> int:
        return self.n_jobs if self.n_jobs > 0 else (os.cpu_count() or 1)

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.models_dir, self.db_path.parent):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = _path("DATA_DIR", PROJECT_ROOT / "data")
    return Settings(
        hf_repo=_str("HF_DATASET_REPO", Settings.hf_repo),
        hf_token=_str("HF_TOKEN", ""),
        category=_str("AMAZON_CATEGORY", Settings.category),
        sample_reviews=_int("SAMPLE_REVIEWS", Settings.sample_reviews),
        scan_multiplier=max(1, _int("SCAN_MULTIPLIER", Settings.scan_multiplier)),
        meta_scan_limit=_int("META_SCAN_LIMIT", Settings.meta_scan_limit),
        min_reviews_per_product=max(1, _int("MIN_REVIEWS_PER_PRODUCT", Settings.min_reviews_per_product)),
        seed=_int("RANDOM_SEED", Settings.seed),
        data_dir=data_dir,
        models_dir=_path("MODELS_DIR", PROJECT_ROOT / "models"),
        db_path=_path("DB_PATH", data_dir / "trust_assessor.sqlite"),
        scoring_config_path=_path("SCORING_CONFIG_PATH", data_dir / "scoring_config.json"),
        sentiment_backend=_str("SENTIMENT_BACKEND", Settings.sentiment_backend).lower(),
        sentiment_model=_str("SENTIMENT_MODEL", Settings.sentiment_model),
        sentiment_batch_size=_int("SENTIMENT_BATCH_SIZE", Settings.sentiment_batch_size),
        sentiment_max_length=_int("SENTIMENT_MAX_LENGTH", Settings.sentiment_max_length),
        sentiment_max_reviews=_int("SENTIMENT_MAX_REVIEWS", Settings.sentiment_max_reviews),
        sentiment_neutral_band=_float("SENTIMENT_NEUTRAL_BAND", Settings.sentiment_neutral_band),
        anomaly_contamination=_float("ANOMALY_CONTAMINATION", Settings.anomaly_contamination),
        suspicion_flag_threshold=_float("SUSPICION_FLAG_THRESHOLD", Settings.suspicion_flag_threshold),
        n_jobs=_int("N_JOBS", Settings.n_jobs),
        ui_host=_str("UI_HOST", Settings.ui_host),
        ui_port=_int("UI_PORT", Settings.ui_port),
        show_product_images=_bool("SHOW_PRODUCT_IMAGES", Settings.show_product_images),
        storage_secret=_str("STORAGE_SECRET", Settings.storage_secret),
        log_level=_str("LOG_LEVEL", Settings.log_level).upper(),
    )
