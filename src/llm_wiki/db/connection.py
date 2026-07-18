"""SQLite connection helpers, including sqlite-vec loading and schema bootstrap."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Name of the vec0 virtual table holding page embeddings. It is created at
# runtime rather than in schema.sql because its dimension depends on the
# configured embedding model.
EMBEDDING_TABLE = "wiki_page_embeddings"


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded and sane pragmas."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection, embedding_dim: int) -> None:
    """Create tables if absent. Safe to call on every run."""
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {EMBEDDING_TABLE} USING vec0(
            page_id INTEGER PRIMARY KEY,
            embedding FLOAT[{embedding_dim}] distance_metric=cosine
        )
        """
    )


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block atomically.

    Connections are opened in autocommit mode (isolation_level=None), so
    transactions are managed explicitly here.
    """
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
