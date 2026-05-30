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


def init_db() -> None:
    global _engine, _SessionLocal
    with _init_lock:
        if _SessionLocal is not None:
            return
        try:
            url = os.environ.get("MAO_DATABASE_URL", "postgresql://mao:mao@localhost:5432/mao")
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
            logger.info("Database initialised pool_size=10 max_overflow=20: %s", url)
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
