"""Query helpers over the ingest schema.

Every function takes an explicit connection so callers can compose them inside
a single transaction (see connection.transaction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from pgvector import Vector
from rapidfuzz import fuzz

from llm_wiki.db.connection import Connection

_WHITESPACE = re.compile(r"\s+")

# How many neighbours to pull from the vector index before applying the
# distance threshold and candidate limit.
_KNN_OVERFETCH = 20

Row = Mapping[str, Any]


def normalize_name(name: str) -> str:
    """Normalize an entity name into its alias lookup key."""
    return _WHITESPACE.sub(" ", name).strip().casefold()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# source
# --------------------------------------------------------------------------


def find_source_by_sha(conn: Connection, namespace: str, sha256: str) -> Row | None:
    return conn.execute(
        "SELECT * FROM source WHERE namespace = %s AND sha256 = %s", (namespace, sha256)
    ).fetchone()


def insert_source(
    conn: Connection,
    *,
    filename: str,
    namespace: str,
    sha256: str,
    timestamp: str | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO source (filename, timestamp, namespace, sha256, ingest_time)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """,
        (filename, timestamp, namespace, sha256, _now()),
    ).fetchone()
    return int(row["id"])


def get_source(conn: Connection, source_id: int) -> Row | None:
    return conn.execute("SELECT * FROM source WHERE id = %s", (source_id,)).fetchone()


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


def find_page_by_alias(conn: Connection, namespace: str, name: str) -> PageMatch | None:
    row = conn.execute(
        """
        SELECT p.page_id, p.page_name
        FROM page_aliases a
        JOIN wiki_pages p ON p.page_id = a.page_id
        WHERE a.namespace = %s AND a.query_name = %s
        """,
        (namespace, normalize_name(name)),
    ).fetchone()
    if row is None:
        return None
    return PageMatch(page_id=row["page_id"], page_name=row["page_name"], how="alias")


def find_page_candidates_by_embedding(
    conn: Connection,
    namespace: str,
    embedding: Sequence[float],
    threshold: float,
    limit: int,
) -> list[PageCandidate]:
    """Up to `limit` nearest pages within `threshold` cosine distance, closest first.

    The `<=>` operator is pgvector's cosine distance (smaller is closer), so it
    matches the sqlite-vec semantics the thresholds were tuned against. Pages
    without an embedding yet are skipped.
    """
    rows = conn.execute(
        """
        SELECT page_id, (embedding <=> %s) AS distance, page_name, type
        FROM wiki_pages
        WHERE namespace = %s AND embedding IS NOT NULL
        ORDER BY distance
        LIMIT %s
        """,
        (Vector(embedding), namespace, _KNN_OVERFETCH),
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
    conn: Connection,
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
        WHERE a.namespace = %s
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


def insert_page(
    conn: Connection,
    *,
    page_name: str,
    namespace: str,
    type_: str,
    embedding: Sequence[float] | None = None,
) -> int:
    """Insert a page, optionally with its name embedding, and return its id."""
    row = conn.execute(
        """
        INSERT INTO wiki_pages (page_name, namespace, type, create_time, embedding)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING page_id
        """,
        (page_name, namespace, type_, _now(), None if embedding is None else Vector(embedding)),
    ).fetchone()
    return int(row["page_id"])


def get_page(conn: Connection, page_id: int) -> Row | None:
    # Explicit columns, not SELECT *: the embedding column is a wide vector that
    # metadata callers never need, and Postgres would otherwise fetch it TOASTed.
    return conn.execute(
        """
        SELECT page_id, page_name, namespace, type, needs_review, create_time
        FROM wiki_pages WHERE page_id = %s
        """,
        (page_id,),
    ).fetchone()


def set_needs_review(conn: Connection, page_id: int, flag: bool) -> None:
    conn.execute(
        "UPDATE wiki_pages SET needs_review = %s WHERE page_id = %s", (flag, page_id)
    )


def list_pages(conn: Connection, namespace: str) -> list[Row]:
    return conn.execute(
        """
        SELECT page_id, page_name, type, needs_review, create_time
        FROM wiki_pages WHERE namespace = %s ORDER BY lower(page_name)
        """,
        (namespace,),
    ).fetchall()


def upsert_alias(
    conn: Connection, *, namespace: str, name: str, page_id: int, type_: str
) -> None:
    """Record an alias. An existing alias is left alone: a name already bound to
    a page (canonical, or confirmed manually) must not be silently re-pointed by
    a later fuzzy match."""
    conn.execute(
        """
        INSERT INTO page_aliases (namespace, query_name, page_id, type)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (namespace, query_name) DO NOTHING
        """,
        (namespace, normalize_name(name), page_id, type_),
    )


# --------------------------------------------------------------------------
# wiki_source
# --------------------------------------------------------------------------


def link_source(conn: Connection, *, wiki_id: int, source_id: int, namespace: str) -> int:
    """Link a source to a page and return its reference number.

    Idempotent: if the source already cites this page, its existing reference
    number is returned rather than allocating a new one.
    """
    existing = conn.execute(
        "SELECT reference_order FROM wiki_source WHERE wiki_id = %s AND source_id = %s",
        (wiki_id, source_id),
    ).fetchone()
    if existing is not None:
        return int(existing["reference_order"])

    row = conn.execute(
        "SELECT COALESCE(MAX(reference_order), 0) AS n FROM wiki_source WHERE wiki_id = %s",
        (wiki_id,),
    ).fetchone()
    reference_order = int(row["n"]) + 1

    conn.execute(
        """
        INSERT INTO wiki_source (wiki_id, source_id, reference_order, namespace)
        VALUES (%s, %s, %s, %s)
        """,
        (wiki_id, source_id, reference_order, namespace),
    )
    return reference_order


def list_references(conn: Connection, wiki_id: int) -> list[Row]:
    return conn.execute(
        """
        SELECT ws.reference_order, s.filename, s.namespace, s.ingest_time
        FROM wiki_source ws
        JOIN source s ON s.id = ws.source_id
        WHERE ws.wiki_id = %s
        ORDER BY ws.reference_order
        """,
        (wiki_id,),
    ).fetchall()


# --------------------------------------------------------------------------
# wiki_links  (page-to-page cross references)
# --------------------------------------------------------------------------


def load_link_dictionary(conn: Connection, namespace: str) -> list[Row]:
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
        WHERE a.namespace = %s
        """,
        (namespace,),
    ).fetchall()


def replace_page_links(
    conn: Connection,
    *,
    src_page_id: int,
    namespace: str,
    links: Sequence[tuple[int, str, str]],
) -> None:
    """Replace all outgoing links for a page.

    Delete-then-insert keeps the table in sync with the freshly rewritten body:
    a mention that disappeared drops its row rather than lingering. Each link is
    (dst_page_id, anchor_text, context_sentence).
    """
    conn.execute("DELETE FROM wiki_links WHERE src_page_id = %s", (src_page_id,))
    for dst_page_id, anchor_text, context_sentence in links:
        conn.execute(
            """
            INSERT INTO wiki_links
                (src_page_id, dst_page_id, namespace, anchor_text, context_sentence)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (src_page_id, dst_page_id) DO NOTHING
            """,
            (src_page_id, dst_page_id, namespace, anchor_text, context_sentence),
        )


def list_backlinks(conn: Connection, dst_page_id: int) -> list[Row]:
    """Pages that link to the given page."""
    return conn.execute(
        """
        SELECT l.src_page_id, p.page_name, l.anchor_text, l.context_sentence
        FROM wiki_links l
        JOIN wiki_pages p ON p.page_id = l.src_page_id
        WHERE l.dst_page_id = %s
        ORDER BY lower(p.page_name)
        """,
        (dst_page_id,),
    ).fetchall()


def list_outgoing_links(conn: Connection, src_page_id: int) -> list[Row]:
    """Pages the given page links to."""
    return conn.execute(
        """
        SELECT l.dst_page_id, p.page_name, l.anchor_text, l.context_sentence
        FROM wiki_links l
        JOIN wiki_pages p ON p.page_id = l.dst_page_id
        WHERE l.src_page_id = %s
        ORDER BY lower(p.page_name)
        """,
        (src_page_id,),
    ).fetchall()


# --------------------------------------------------------------------------
# ingest_run / ingest_doc  (dashboard telemetry)
# --------------------------------------------------------------------------


def start_run(conn: Connection, *, started_at: datetime | None = None) -> int:
    row = conn.execute(
        "INSERT INTO ingest_run (started_at) VALUES (%s) RETURNING run_id",
        (started_at or _now(),),
    ).fetchone()
    return int(row["run_id"])


def finish_run(
    conn: Connection,
    run_id: int,
    *,
    total_seconds: float,
    doc_count: int,
    created: int,
    merged: int,
    skipped: int,
    flagged: int,
    failed: int,
    finished_at: datetime | None = None,
) -> None:
    conn.execute(
        """
        UPDATE ingest_run
        SET finished_at = %s, total_seconds = %s, doc_count = %s, created = %s,
            merged = %s, skipped = %s, flagged = %s, failed = %s
        WHERE run_id = %s
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
    conn: Connection,
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
    row = conn.execute(
        """
        INSERT INTO ingest_doc
            (run_id, namespace, filename, path, status, seconds, skip_reason, error, items_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (run_id, namespace, filename, path, status, seconds, skip_reason, error, items_json),
    ).fetchone()
    return int(row["id"])


def list_runs(conn: Connection, limit: int = 50) -> list[Row]:
    return conn.execute(
        "SELECT * FROM ingest_run ORDER BY run_id DESC LIMIT %s", (limit,)
    ).fetchall()


def get_run(conn: Connection, run_id: int) -> Row | None:
    return conn.execute("SELECT * FROM ingest_run WHERE run_id = %s", (run_id,)).fetchone()


def latest_run(conn: Connection) -> Row | None:
    return conn.execute(
        "SELECT * FROM ingest_run ORDER BY run_id DESC LIMIT 1"
    ).fetchone()


def list_run_docs(conn: Connection, run_id: int) -> list[Row]:
    return conn.execute(
        "SELECT * FROM ingest_doc WHERE run_id = %s ORDER BY id", (run_id,)
    ).fetchall()


def latest_ingest_docs(conn: Connection) -> list[Row]:
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


def source_filenames_by_namespace(conn: Connection) -> set[tuple[str, str]]:
    """(namespace, filename) pairs that already have a `source` row.

    Lets the dashboard mark files ingested before telemetry existed (which have
    no ingest_doc row) as already ingested rather than pending.
    """
    rows = conn.execute("SELECT namespace, filename FROM source").fetchall()
    return {(row["namespace"], row["filename"]) for row in rows}
