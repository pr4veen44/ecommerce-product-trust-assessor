"""All SQL access lives here so the UI and services never build queries themselves."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from trust_assessor.db.models import Assessment, DatasetVersion, ModelVersion, Product
from trust_assessor.db.session import session_scope

_CHUNK = 500


def _latest_ids():
    return select(func.max(Assessment.id)).group_by(Assessment.parent_asin)


def _assessment_dict(a: Assessment) -> dict[str, Any]:
    return {c.name: getattr(a, c.name) for c in Assessment.__table__.columns}


def _product_dict(p: Product) -> dict[str, Any]:
    return {c.name: getattr(p, c.name) for c in Product.__table__.columns}


# --------------------------------------------------------------------------- writes
def upsert_products(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with session_scope() as s:
        for i in range(0, len(rows), _CHUNK):
            chunk = rows[i:i + _CHUNK]
            stmt = sqlite_insert(Product).values(chunk)
            update = {c: stmt.excluded[c] for c in chunk[0] if c != "parent_asin"}
            s.execute(stmt.on_conflict_do_update(index_elements=["parent_asin"], set_=update))


def register_dataset_version(row: dict[str, Any]) -> None:
    with session_scope() as s:
        stmt = sqlite_insert(DatasetVersion).values(**row)
        s.execute(stmt.on_conflict_do_update(index_elements=["version"],
                                             set_={k: stmt.excluded[k] for k in row if k != "version"}))


def register_model_version(row: dict[str, Any]) -> None:
    with session_scope() as s:
        stmt = sqlite_insert(ModelVersion).values(**row)
        s.execute(stmt.on_conflict_do_nothing(index_elements=["version"]))


def save_assessments(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with session_scope() as s:
        for i in range(0, len(rows), _CHUNK):
            s.execute(Assessment.__table__.insert(), rows[i:i + _CHUNK])


def delete_products_not_in(asins: Iterable[str]) -> int:
    keep = set(asins)
    with session_scope() as s:
        stale = [a for (a,) in s.execute(select(Product.parent_asin)) if a not in keep]
        for i in range(0, len(stale), _CHUNK):
            part = stale[i:i + _CHUNK]
            s.execute(Assessment.__table__.delete().where(Assessment.parent_asin.in_(part)))
            s.execute(Product.__table__.delete().where(Product.parent_asin.in_(part)))
        return len(stale)


# --------------------------------------------------------------------------- reads
def count_products() -> int:
    with session_scope() as s:
        return int(s.scalar(select(func.count()).select_from(Product)) or 0)


def count_assessments() -> int:
    with session_scope() as s:
        return int(s.scalar(select(func.count()).select_from(Assessment)) or 0)


def current_dataset_version() -> str | None:
    with session_scope() as s:
        return s.scalar(select(Product.dataset_version).limit(1))


def get_product(asin: str) -> dict[str, Any] | None:
    with session_scope() as s:
        p = s.get(Product, asin)
        return _product_dict(p) if p else None


def latest_assessment(asin: str) -> dict[str, Any] | None:
    with session_scope() as s:
        a = s.scalars(select(Assessment).where(Assessment.parent_asin == asin)
                      .order_by(Assessment.id.desc()).limit(1)).first()
        return _assessment_dict(a) if a else None


def search_products(query: str = "", grades: list[str] | None = None, min_score: float | None = None,
                    max_score: float | None = None, min_reviews: int = 0, sort_by: str = "trust_score",
                    descending: bool = True, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    join = and_(Assessment.parent_asin == Product.parent_asin, Assessment.id.in_(_latest_ids()))
    where = []
    for tok in query.lower().split():
        like = f"%{tok}%"
        where.append(or_(Product.title_lower.like(like), Product.parent_asin.ilike(like),
                         func.lower(func.coalesce(Product.store, "")).like(like)))
    if grades:
        where.append(Assessment.grade.in_(grades))
    if min_score is not None:
        where.append(Assessment.trust_score >= min_score)
    if max_score is not None:
        where.append(Assessment.trust_score <= max_score)
    if min_reviews:
        where.append(Product.n_reviews >= min_reviews)

    sort_cols = {
        "trust_score": Assessment.trust_score, "n_reviews": Product.n_reviews, "title": Product.title_lower,
        "mean_rating": Product.mean_rating, "evidence_coverage": Assessment.evidence_coverage,
    }
    col = sort_cols.get(sort_by, Assessment.trust_score)
    order = col.desc().nulls_last() if descending else col.asc().nulls_last()

    with session_scope() as s:
        total = s.scalar(select(func.count()).select_from(Product).outerjoin(Assessment, join).where(*where)) or 0
        stmt = (select(Product, Assessment).outerjoin(Assessment, join).where(*where)
                .order_by(order, Product.parent_asin).limit(limit).offset(offset))
        out = []
        for p, a in s.execute(stmt):
            d = _product_dict(p)
            d.update({"trust_score": a.trust_score if a else None, "grade": a.grade if a else None,
                      "confidence": a.confidence if a else None,
                      "evidence_coverage": a.evidence_coverage if a else None})
            out.append(d)
        return out, int(total)


def latest_assessments_table() -> list[dict[str, Any]]:
    """Latest assessment per product with the fields the dashboard aggregates."""
    with session_scope() as s:
        stmt = (select(Product.parent_asin, Product.title, Product.n_reviews, Assessment.trust_score, Assessment.grade,
                       Assessment.confidence, Assessment.evidence_coverage, Assessment.components)
                .join(Assessment, Assessment.parent_asin == Product.parent_asin)
                .where(Assessment.id.in_(_latest_ids())))
        return [dict(r._mapping) for r in s.execute(stmt)]


def assessment_history(asin: str | None = None, source: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    with session_scope() as s:
        stmt = select(Assessment, Product.title).join(Product, Product.parent_asin == Assessment.parent_asin)
        if asin:
            stmt = stmt.where(Assessment.parent_asin == asin)
        if source:
            stmt = stmt.where(Assessment.source == source)
        stmt = stmt.order_by(Assessment.id.desc()).limit(limit)
        out = []
        for a, title in s.execute(stmt):
            d = _assessment_dict(a)
            d["title"] = title
            out.append(d)
        return out


def run_summaries(limit: int = 50) -> list[dict[str, Any]]:
    with session_scope() as s:
        stmt = (select(Assessment.run_id, Assessment.source, func.min(Assessment.created_at).label("started"),
                       func.count().label("n"), func.avg(Assessment.trust_score).label("mean_score"),
                       func.min(Assessment.config_hash).label("config_hash"),
                       func.min(Assessment.dataset_version).label("dataset_version"),
                       func.min(Assessment.model_version).label("model_version"))
                .group_by(Assessment.run_id, Assessment.source).order_by(func.min(Assessment.id).desc()).limit(limit))
        return [dict(r._mapping) for r in s.execute(stmt)]


def version_rows() -> dict[str, list[dict[str, Any]]]:
    with session_scope() as s:
        ds = [{c.name: getattr(r, c.name) for c in DatasetVersion.__table__.columns if c.name != "manifest"}
              for r in s.scalars(select(DatasetVersion).order_by(DatasetVersion.created_at.desc()))]
        ms = [{c.name: getattr(r, c.name) for c in ModelVersion.__table__.columns if c.name != "manifest"}
              for r in s.scalars(select(ModelVersion).order_by(ModelVersion.recorded_at.desc()))]
        return {"datasets": ds, "models": ms}


def last_assessment_time() -> datetime | None:
    with session_scope() as s:
        return s.scalar(select(func.max(Assessment.created_at)))
