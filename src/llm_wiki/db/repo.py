"""Query helpers over the ingest schema.

Every function takes an explicit connection so callers can compose them inside
a single transaction (see connection.transaction).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import sqlite_vec

from llm_wiki.db.connection import EMBEDDING_TABLE

_WHITESPACE = re.compile(r"\s+")

# How many neighbours to pull from the vector index before filtering by
# namespace. Over-fetching keeps namespace filtering correct without pushing a
# WHERE clause into the KNN scan.
_KNN_OVERFETCH = 20


def normalize_name(name: str) -> str:
    """Normalize an entity name into its alias lookup key."""
    return _WHITESPACE.sub(" ", name).strip().casefold()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# source
# --------------------------------------------------------------------------


def find_source_by_sha(conn: sqlite3.Connection, namespace: str, sha256: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM source WHERE namespace = ? AND sha256 = ?", (namespace, sha256)
    ).fetchone()


def insert_source(
    conn: sqlite3.Connection,
    *,
    filename: str,
    namespace: str,
    sha256: str,
    timestamp: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO source (filename, timestamp, namespace, sha256, ingest_time)
        VALUES (?, ?, ?, ?, ?)
        """,
        (filename, timestamp, namespace, sha256, _now()),
    )
    return int(cur.lastrowid)


def get_source(conn: sqlite3.Connection, source_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM source WHERE id = ?", (source_id,)).fetchone()


# --------------------------------------------------------------------------
# wiki_pages / page_aliases
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PageMatch:
    page_id: int
    page_name: str
    how: str  # 'alias' | 'embedding_sim'
    distance: float | None = None


def find_page_by_alias(conn: sqlite3.Connection, namespace: str, name: str) -> PageMatch | None:
    row = conn.execute(
        """
        SELECT p.page_id, p.page_name
        FROM page_aliases a
        JOIN wiki_pages p ON p.page_id = a.page_id
        WHERE a.namespace = ? AND a.query_name = ?
        """,
        (namespace, normalize_name(name)),
    ).fetchone()
    if row is None:
        return None
    return PageMatch(page_id=row["page_id"], page_name=row["page_name"], how="alias")


def find_page_by_embedding(
    conn: sqlite3.Connection,
    namespace: str,
    embedding: Sequence[float],
    threshold: float,
) -> PageMatch | None:
    """Nearest page within `threshold` cosine distance, or None."""
    rows = conn.execute(
        f"""
        SELECT e.page_id, e.distance, p.page_name
        FROM {EMBEDDING_TABLE} e
        JOIN wiki_pages p ON p.page_id = e.page_id
        WHERE e.embedding MATCH ? AND e.k = ? AND p.namespace = ?
        ORDER BY e.distance
        """,
        (sqlite_vec.serialize_float32(embedding), _KNN_OVERFETCH, namespace),
    ).fetchall()
    if not rows:
        return None
    best = rows[0]
    if best["distance"] > threshold:
        return None
    return PageMatch(
        page_id=best["page_id"],
        page_name=best["page_name"],
        how="embedding_sim",
        distance=float(best["distance"]),
    )


def insert_page(conn: sqlite3.Connection, *, page_name: str, namespace: str, type_: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO wiki_pages (page_name, namespace, type, create_time)
        VALUES (?, ?, ?, ?)
        """,
        (page_name, namespace, type_, _now()),
    )
    return int(cur.lastrowid)


def get_page(conn: sqlite3.Connection, page_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM wiki_pages WHERE page_id = ?", (page_id,)).fetchone()


def set_needs_review(conn: sqlite3.Connection, page_id: int, flag: bool) -> None:
    conn.execute(
        "UPDATE wiki_pages SET needs_review = ? WHERE page_id = ?", (1 if flag else 0, page_id)
    )


def list_pages(conn: sqlite3.Connection, namespace: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT page_id, page_name, type, needs_review, create_time
        FROM wiki_pages WHERE namespace = ? ORDER BY page_name COLLATE NOCASE
        """,
        (namespace,),
    ).fetchall()


def upsert_alias(
    conn: sqlite3.Connection, *, namespace: str, name: str, page_id: int, type_: str
) -> None:
    """Record an alias. An existing alias is left alone: a name already bound to
    a page (canonical, or confirmed manually) must not be silently re-pointed by
    a later fuzzy match."""
    conn.execute(
        """
        INSERT OR IGNORE INTO page_aliases (namespace, query_name, page_id, type)
        VALUES (?, ?, ?, ?)
        """,
        (namespace, normalize_name(name), page_id, type_),
    )


def upsert_embedding(conn: sqlite3.Connection, page_id: int, embedding: Sequence[float]) -> None:
    conn.execute(f"DELETE FROM {EMBEDDING_TABLE} WHERE page_id = ?", (page_id,))
    conn.execute(
        f"INSERT INTO {EMBEDDING_TABLE} (page_id, embedding) VALUES (?, ?)",
        (page_id, sqlite_vec.serialize_float32(embedding)),
    )


# --------------------------------------------------------------------------
# wiki_source
# --------------------------------------------------------------------------


def link_source(conn: sqlite3.Connection, *, wiki_id: int, source_id: int, namespace: str) -> int:
    """Link a source to a page and return its reference number.

    Idempotent: if the source already cites this page, its existing reference
    number is returned rather than allocating a new one.
    """
    existing = conn.execute(
        "SELECT reference_order FROM wiki_source WHERE wiki_id = ? AND source_id = ?",
        (wiki_id, source_id),
    ).fetchone()
    if existing is not None:
        return int(existing["reference_order"])

    row = conn.execute(
        "SELECT COALESCE(MAX(reference_order), 0) AS n FROM wiki_source WHERE wiki_id = ?",
        (wiki_id,),
    ).fetchone()
    reference_order = int(row["n"]) + 1

    conn.execute(
        """
        INSERT INTO wiki_source (wiki_id, source_id, reference_order, namespace)
        VALUES (?, ?, ?, ?)
        """,
        (wiki_id, source_id, reference_order, namespace),
    )
    return reference_order


def list_references(conn: sqlite3.Connection, wiki_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT ws.reference_order, s.filename, s.namespace, s.ingest_time
        FROM wiki_source ws
        JOIN source s ON s.id = ws.source_id
        WHERE ws.wiki_id = ?
        ORDER BY ws.reference_order
        """,
        (wiki_id,),
    ).fetchall()
