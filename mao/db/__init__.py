from __future__ import annotations
import logging
import os
import threading
from contextlib import contextmanager
from collections.abc import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from mao.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None
_init_lock = threading.Lock()


def resolve_database_url() -> str:
    """The single source of truth for the database URL.

    `MAO_DATABASE_URL` is an explicit override; otherwise the same
    `DATABASE_URL` the rest of the system reads via `cfg.postgres_url` is used.
    Previously this module read only `MAO_DATABASE_URL` while `.env` set only
    `DATABASE_URL`, so init_db() silently fell back to localhost while other
    call sites pointed at the real database (P1-10).
    """
    override = os.environ.get("MAO_DATABASE_URL", "").strip()
    if override:
        return override
    from mao.core.config import cfg

    return cfg.postgres_url


def configure_engine(url: str) -> None:
    """Bind the session factory to an explicit URL. Test/CLI seam."""
    global _engine, _SessionLocal
    with _init_lock:
        engine = create_engine(url)
        Base.metadata.create_all(engine)
        _engine = engine
        _SessionLocal = sessionmaker(bind=engine)


def reset_engine() -> None:
    """Drop the bound engine. Test seam."""
    global _engine, _SessionLocal
    with _init_lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _SessionLocal = None


def is_initialised() -> bool:
    return _SessionLocal is not None


def init_db() -> None:
    global _engine, _SessionLocal
    with _init_lock:
        if _SessionLocal is not None:
            return
        url = resolve_database_url()
        try:
            engine = create_engine(
                url,
                pool_pre_ping=True,   # recycles stale connections automatically
                pool_size=10,         # base pool: 10 persistent connections
                max_overflow=20,      # burst: 20 extra connections under load
                pool_timeout=30,      # raise if no free connection after 30 s
                pool_recycle=1800,    # recycle after 30 min (avoids DB-side idle timeouts)
            )
            Base.metadata.create_all(engine)
            _engine = engine
            _SessionLocal = sessionmaker(bind=engine)
            # Never log the URL — it carries the password (P1-11).
            logger.info("Database initialised pool_size=10 max_overflow=20")
        except Exception as exc:
            logger.critical("Database init failed: %s", exc)
            raise


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
