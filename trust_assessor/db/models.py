"""SQLAlchemy ORM models. Large review data stays in Parquet; SQLite holds products + assessments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    version: Mapped[str] = mapped_column(String(200), primary_key=True)
    category: Mapped[str] = mapped_column(String(100))
    sample_reviews: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(Integer)
    n_reviews: Mapped[int] = mapped_column(Integer)
    n_products: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    version: Mapped[str] = mapped_column(String(100), primary_key=True)
    trained_at: Mapped[str] = mapped_column(String(40))
    dataset_version: Mapped[str | None] = mapped_column(String(200), nullable=True)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Product(Base):
    __tablename__ = "products"
    parent_asin: Mapped[str] = mapped_column(String(20), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    title_lower: Mapped[str] = mapped_column(Text, index=True)
    category: Mapped[str | None] = mapped_column(String(200), nullable=True)
    store: Mapped[str | None] = mapped_column(String(200), nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta_avg_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta_rating_number: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    has_metadata: Mapped[bool] = mapped_column(Boolean, default=False)
    n_reviews: Mapped[int] = mapped_column(Integer, default=0)
    mean_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(200), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class Assessment(Base):
    __tablename__ = "assessments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_asin: Mapped[str] = mapped_column(ForeignKey("products.parent_asin", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    source: Mapped[str] = mapped_column(String(30))               # batch | interactive | reweight
    trust_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    raw_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    grade: Mapped[str] = mapped_column(String(40), index=True)
    confidence: Mapped[str] = mapped_column(String(20))
    evidence_coverage: Mapped[float] = mapped_column(Float)
    n_reviews: Mapped[int] = mapped_column(Integer)
    components: Mapped[list[dict[str, Any]]] = mapped_column(JSON)      # full per-component results
    strengths: Mapped[list[str]] = mapped_column(JSON)
    warnings: Mapped[list[str]] = mapped_column(JSON)
    missing: Mapped[list[str]] = mapped_column(JSON)
    adjustments: Mapped[list[str]] = mapped_column(JSON)
    scoring_config: Mapped[dict[str, Any]] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(20))
    dataset_version: Mapped[str | None] = mapped_column(String(200), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sentiment_method: Mapped[str | None] = mapped_column(String(200), nullable=True)
    suspicion_method: Mapped[str | None] = mapped_column(String(60), nullable=True)

    __table_args__ = (Index("ix_assessments_asin_created", "parent_asin", "created_at"),)
