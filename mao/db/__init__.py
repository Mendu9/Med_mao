from __future__ import annotations
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from mao.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None

POSTGRES_URL = "postgresql://mao:mao@localhost:5432/mao"


def init_db() -> None:
    global _engine, _SessionLocal
    try:
        _engine = create_engine(POSTGRES_URL, pool_pre_ping=True)
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(bind=_engine)
        logger.info("Database initialised: %s", POSTGRES_URL)
    except Exception as exc:
        logger.critical("Database init failed: %s", exc)


@contextmanager
def get_db_session():
    if _SessionLocal is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
