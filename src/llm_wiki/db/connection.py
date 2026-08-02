"""PostgreSQL connection helpers, including pgvector setup and schema bootstrap.

Connections wrap psycopg2 in a thin :class:`Connection` that keeps the
sqlite-style ``conn.execute(sql, params).fetchone()`` call shape the repo layer
was written against. Rows come back as dict-like ``RealDictRow`` objects, so
``row["col"]`` access is unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Drops every table this app owns (for `llm-wiki reset`). Only our own tables,
# never the whole schema: the database server may be shared with other apps.
# CASCADE clears the foreign-key web without needing a specific drop order.
DROP_ALL_SQL = """
DROP TABLE IF EXISTS
    wiki_links,
    wiki_source,
    page_aliases,
    wiki_pages,
    source
CASCADE
"""


class Connection:
    """Thin wrapper over a psycopg2 connection.

    Exposes ``execute(sql, params)`` returning a cursor, so repo functions read
    the same as they did on sqlite3. The underlying connection runs in
    autocommit mode; explicit transactions are opened via :func:`transaction`.
    """

    def __init__(self, raw: psycopg2.extensions.connection) -> None:
        self._raw = raw

    @property
    def raw(self) -> psycopg2.extensions.connection:
        return self._raw

    def execute(self, sql: str, params: Sequence[Any] = ()) -> RealDictCursor:
        cur = self._raw.cursor()
        cur.execute(sql, params)
        return cur

    def close(self) -> None:
        self._raw.close()


def connect(db_url: str) -> Connection:
    """Open a connection with pgvector registered and autocommit on.

    Autocommit mirrors the old sqlite ``isolation_level=None`` setup: every
    statement commits on its own unless wrapped in :func:`transaction`.
    """
    raw = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
    raw.autocommit = True

    # pgvector's adapters/typecasters need the `vector` type to exist first.
    with raw.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(raw)

    return Connection(raw)


def init_db(conn: Connection) -> None:
    """Create tables if absent. Safe to call on every run."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))


@contextmanager
def transaction(conn: Connection) -> Iterator[Connection]:
    """Run a block atomically.

    The connection is in autocommit mode, so a transaction is opened explicitly
    here with BEGIN/COMMIT and rolled back on error. Transactions are
    per-connection, so statements run through ``conn.execute`` inside the block
    all join this transaction.
    """
    cur = conn.raw.cursor()
    cur.execute("BEGIN")
    try:
        yield conn
    except Exception:
        cur.execute("ROLLBACK")
        raise
    else:
        cur.execute("COMMIT")
    finally:
        cur.close()
