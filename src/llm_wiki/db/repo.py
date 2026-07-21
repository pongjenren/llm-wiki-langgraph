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
from rapidfuzz import fuzz

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
    how: str  # 'alias'


@dataclass(frozen=True)
class PageCandidate:
    """A possible resolution target, from either the vector index or string matching."""

    page_id: int
    page_name: str
    type: str
    how: str  # 'embedding_sim' | 'string_sim'
    # Signal strength in this candidate's own units: cosine *distance* for
    # embeddings (smaller is closer), token_sort_ratio 0-100 for strings.
    score: float


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


def find_page_candidates_by_embedding(
    conn: sqlite3.Connection,
    namespace: str,
    embedding: Sequence[float],
    threshold: float,
    limit: int,
) -> list[PageCandidate]:
    """Up to `limit` nearest pages within `threshold` cosine distance, closest first."""
    rows = conn.execute(
        f"""
        SELECT e.page_id, e.distance, p.page_name, p.type
        FROM {EMBEDDING_TABLE} e
        JOIN wiki_pages p ON p.page_id = e.page_id
        WHERE e.embedding MATCH ? AND e.k = ? AND p.namespace = ?
        ORDER BY e.distance
        """,
        (sqlite_vec.serialize_float32(embedding), _KNN_OVERFETCH, namespace),
    ).fetchall()
    candidates: list[PageCandidate] = []
    for row in rows:
        if row["distance"] > threshold:
            break  # rows are ordered by distance, so nothing further qualifies
        candidates.append(
            PageCandidate(
                page_id=row["page_id"],
                page_name=row["page_name"],
                type=row["type"],
                how="embedding_sim",
                score=float(row["distance"]),
            )
        )
        if len(candidates) >= limit:
            break
    return candidates


def find_page_candidates_by_string(
    conn: sqlite3.Connection,
    namespace: str,
    name: str,
    threshold: float,
    limit: int,
) -> list[PageCandidate]:
    """Up to `limit` pages whose alias best-matches `name` at/above `threshold`.

    Every alias in the namespace is scored with rapidfuzz token_sort_ratio; a page
    is kept once, under its best-scoring alias. Highest score first.
    """
    rows = conn.execute(
        """
        SELECT a.query_name, a.page_id, p.page_name, p.type
        FROM page_aliases a
        JOIN wiki_pages p ON p.page_id = a.page_id
        WHERE a.namespace = ?
        """,
        (namespace,),
    ).fetchall()
    query = normalize_name(name)
    best: dict[int, PageCandidate] = {}
    for row in rows:
        score = fuzz.token_sort_ratio(query, row["query_name"])
        if score < threshold:
            continue
        current = best.get(row["page_id"])
        if current is None or score > current.score:
            best[row["page_id"]] = PageCandidate(
                page_id=row["page_id"],
                page_name=row["page_name"],
                type=row["type"],
                how="string_sim",
                score=score,
            )
    ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
    return ranked[:limit]


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


# --------------------------------------------------------------------------
# wiki_links  (page-to-page cross references)
# --------------------------------------------------------------------------


def load_link_dictionary(conn: sqlite3.Connection, namespace: str) -> list[sqlite3.Row]:
    """Every alias in a namespace with the page it resolves to.

    This is the dictionary the linker scans page bodies against. query_name is
    already normalized (see normalize_name); page_name keeps the display form so
    the caller can build the link target path.
    """
    return conn.execute(
        """
        SELECT a.query_name, a.page_id, p.page_name
        FROM page_aliases a
        JOIN wiki_pages p ON p.page_id = a.page_id
        WHERE a.namespace = ?
        """,
        (namespace,),
    ).fetchall()


def replace_page_links(
    conn: sqlite3.Connection,
    *,
    src_page_id: int,
    namespace: str,
    links: Sequence[tuple[int, str]],
) -> None:
    """Replace all outgoing links for a page.

    Delete-then-insert keeps the table in sync with the freshly rewritten body:
    a mention that disappeared drops its row rather than lingering.
    """
    conn.execute("DELETE FROM wiki_links WHERE src_page_id = ?", (src_page_id,))
    for dst_page_id, anchor_text in links:
        conn.execute(
            """
            INSERT OR IGNORE INTO wiki_links (src_page_id, dst_page_id, namespace, anchor_text)
            VALUES (?, ?, ?, ?)
            """,
            (src_page_id, dst_page_id, namespace, anchor_text),
        )


def list_backlinks(conn: sqlite3.Connection, dst_page_id: int) -> list[sqlite3.Row]:
    """Pages that link to the given page."""
    return conn.execute(
        """
        SELECT l.src_page_id, p.page_name, l.anchor_text
        FROM wiki_links l
        JOIN wiki_pages p ON p.page_id = l.src_page_id
        WHERE l.dst_page_id = ?
        ORDER BY p.page_name COLLATE NOCASE
        """,
        (dst_page_id,),
    ).fetchall()


def list_outgoing_links(conn: sqlite3.Connection, src_page_id: int) -> list[sqlite3.Row]:
    """Pages the given page links to."""
    return conn.execute(
        """
        SELECT l.dst_page_id, p.page_name, l.anchor_text
        FROM wiki_links l
        JOIN wiki_pages p ON p.page_id = l.dst_page_id
        WHERE l.src_page_id = ?
        ORDER BY p.page_name COLLATE NOCASE
        """,
        (src_page_id,),
    ).fetchall()


# --------------------------------------------------------------------------
# ingest_run / ingest_doc  (dashboard telemetry)
# --------------------------------------------------------------------------


def start_run(conn: sqlite3.Connection, *, started_at: str | None = None) -> int:
    cur = conn.execute(
        "INSERT INTO ingest_run (started_at) VALUES (?)", (started_at or _now(),)
    )
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    total_seconds: float,
    doc_count: int,
    created: int,
    merged: int,
    skipped: int,
    flagged: int,
    failed: int,
    finished_at: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE ingest_run
        SET finished_at = ?, total_seconds = ?, doc_count = ?, created = ?,
            merged = ?, skipped = ?, flagged = ?, failed = ?
        WHERE run_id = ?
        """,
        (
            finished_at or _now(),
            total_seconds,
            doc_count,
            created,
            merged,
            skipped,
            flagged,
            failed,
            run_id,
        ),
    )


def insert_ingest_doc(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    namespace: str,
    filename: str,
    path: str,
    status: str,
    seconds: float | None,
    skip_reason: str | None = None,
    error: str | None = None,
    items_json: str = "[]",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO ingest_doc
            (run_id, namespace, filename, path, status, seconds, skip_reason, error, items_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, namespace, filename, path, status, seconds, skip_reason, error, items_json),
    )
    return int(cur.lastrowid)


def list_runs(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM ingest_run ORDER BY run_id DESC LIMIT ?", (limit,)
    ).fetchall()


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM ingest_run WHERE run_id = ?", (run_id,)).fetchone()


def latest_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM ingest_run ORDER BY run_id DESC LIMIT 1"
    ).fetchone()


def list_run_docs(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM ingest_doc WHERE run_id = ? ORDER BY id", (run_id,)
    ).fetchall()


def latest_ingest_docs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Most recent ingest_doc per (namespace, path), oldest id first.

    Used to annotate raw files with their last-known outcome. Later runs win: a
    file that failed once and later succeeded shows as ok. Returned in id order
    so callers building their own indexes get last-write-wins for free.
    """
    return conn.execute(
        """
        SELECT d.*, r.finished_at AS run_finished_at, r.started_at AS run_started_at
        FROM ingest_doc d
        JOIN (
            SELECT namespace, path, MAX(id) AS max_id
            FROM ingest_doc GROUP BY namespace, path
        ) latest ON latest.max_id = d.id
        JOIN ingest_run r ON r.run_id = d.run_id
        ORDER BY d.id
        """
    ).fetchall()


def source_filenames_by_namespace(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """(namespace, filename) pairs that already have a `source` row.

    Lets the dashboard mark files ingested before telemetry existed (which have
    no ingest_doc row) as already ingested rather than pending.
    """
    rows = conn.execute("SELECT namespace, filename FROM source").fetchall()
    return {(row["namespace"], row["filename"]) for row in rows}
