"""Dual-backend database connection utilities for MAT-X.
SQLite remains the default for local development and tests.
PostgreSQL is selected explicitly with MATX_DATABASE_BACKEND=postgres.
"""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from app.core.paths import EVIDENCE_DB_PATH
SUPPORTED_BACKENDS = {"sqlite", "postgres"}
def database_backend(db_path=None):
    """Return the configured backend without silently switching databases."""
    if db_path is not None:
        return "sqlite"
    backend = os.getenv("MATX_DATABASE_BACKEND", "sqlite").strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise RuntimeError(
            f"Unsupported MATX_DATABASE_BACKEND: {backend!r}"
        )
    if backend == "postgres" and not os.getenv("DATABASE_URL"):
        raise RuntimeError(
            "MATX_DATABASE_BACKEND=postgres requires DATABASE_URL."
        )
    return backend
def is_postgres(db_path=None):
    return database_backend(db_path) == "postgres"
def connect_database(db_path=None, *, timeout=30):
    """Open a raw SQLite or PostgreSQL connection.
    A supplied db_path always selects SQLite, preserving test isolation.
    The caller is responsible for closing the returned connection.
    """
    backend = database_backend(db_path)
    if backend == "sqlite":
        path = Path(db_path or EVIDENCE_DB_PATH).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), timeout=timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL backend requires psycopg 3."
        ) from exc
    return psycopg.connect(
        os.environ["DATABASE_URL"],
        connect_timeout=timeout,
        row_factory=dict_row,
    )
@contextmanager
def database_connection(db_path=None, *, timeout=30):
    """Yield a connection, commit on success, rollback on failure, close."""
    connection = connect_database(db_path, timeout=timeout)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
def get_table_columns(connection, table_name):
    """Return column names using the active database's catalog."""
    if not table_name or not table_name.replace("_", "").isalnum():
        raise ValueError("Invalid table name.")
    if isinstance(connection, sqlite3.Connection):
        rows = connection.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()
        return [row["name"] for row in rows]
    rows = connection.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table_name,),
    ).fetchall()
    return [row["column_name"] for row in rows]
