"""Engine / session management for the local SQLite database."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from trust_assessor.config import get_settings
from trust_assessor.db.models import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    s = get_settings()
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(s.database_url, connect_args={"check_same_thread": False}, future=True)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return engine


@lru_cache(maxsize=1)
def _factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
