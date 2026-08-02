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
    """The existing row for this document, whatever its outcome.

    A failed row counts: a document is attempted once. Retrying one means
    deleting its source row first.
    """
    return conn.execute(
        "SELECT * FROM source WHERE namespace = %s AND sha256 = %s", (namespace, sha256)
    ).fetchone()


def insert_source(
    conn: Connection,
    *,
    filename: str,
    namespace: str,
    sha256: str | None,
    timestamp: str | None = None,
    status: str = "ok",
    error_msg: str | None = None,
    seconds: float | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO source
            (filename, timestamp, namespace, sha256, ingest_time, status, error_msg, seconds)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (filename, timestamp, namespace, sha256, _now(), status, error_msg, seconds),
    ).fetchone()
    return int(row["id"])


def finish_source(
    conn: Connection,
    source_id: int,
    *,
    status: str,
    error_msg: str | None,
    seconds: float | None,
) -> None:
    """Stamp a source with the outcome of the ingest that created it.

    The row is written mid-pipeline (items need a source_id to cite), so its
    final status is only known once every item has been folded in.
    """
    conn.execute(
        "UPDATE source SET status = %s, error_msg = %s, seconds = %s WHERE id = %s",
        (status, error_msg, seconds, source_id),
    )


def get_source(conn: Connection, source_id: int) -> Row | None:
    return conn.execute("SELECT * FROM source WHERE id = %s", (source_id,)).fetchone()


def list_sources(conn: Connection, limit: int = 50) -> list[Row]:
    """Most recently ingested documents first. Powers the dashboard history."""
    return conn.execute(
        "SELECT * FROM source ORDER BY id DESC LIMIT %s", (limit,)
    ).fetchall()


def sources_by_namespace_filename(conn: Connection) -> dict[tuple[str, str], Row]:
    """Every source keyed by (namespace, filename), latest attempt winning.

    Lets the dashboard annotate the files under raw/ with their outcome. The key
    is looser than a path -- two same-named files in different sub-directories of
    one namespace collide -- but namespaces are flat in practice.
    """
    rows = conn.execute("SELECT * FROM source ORDER BY id").fetchall()
    return {(row["namespace"], row["filename"]): row for row in rows}


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


def list_source_pages(conn: Connection, source_id: int) -> list[Row]:
    """Pages this source contributed to. The dashboard's per-document detail."""
    return conn.execute(
        """
        SELECT p.page_id, p.page_name, p.type, p.needs_review, ws.reference_order
        FROM wiki_source ws
        JOIN wiki_pages p ON p.page_id = ws.wiki_id
        WHERE ws.source_id = %s
        ORDER BY lower(p.page_name)
        """,
        (source_id,),
    ).fetchall()


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
