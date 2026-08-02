"""Stamp each ingested document's outcome onto its `source` row.

This is a pure side-channel: it records what happened, and never changes the
ingest result. Recording is best-effort — a failure to write telemetry must not
turn a successful ingest into a failed command — so callers wrap it in a guard.

There is no run-level table: one ingest is one document, so the document's own
row carries everything the dashboard needs.
"""

from __future__ import annotations

from llm_wiki.db import repo
from llm_wiki.db.connection import Connection
from llm_wiki.pipeline import DocumentOutcome


def _error_message(outcome: DocumentOutcome) -> str | None:
    """The failure text for a document, or None if nothing failed.

    A document-level error is reported as-is. Item-level failures are listed by
    item name — items that succeeded are not recorded at all.
    """
    if outcome.error:
        return outcome.error
    failures = [f"{item.name}: {item.error}" for item in outcome.items if item.error]
    return "\n".join(failures) or None


def record_document(conn: Connection, outcome: DocumentOutcome) -> int | None:
    """Record one document's outcome. Returns its source id, if it has one.

    A document skipped as a duplicate already has a source row from its first
    ingest, so nothing is written for it.
    """
    if outcome.skipped:
        return outcome.source_id

    error_msg = _error_message(outcome)
    status = "failed" if error_msg else "ok"
    seconds = round(outcome.elapsed_seconds, 3)

    if outcome.source_id is not None:
        # The row was written mid-pipeline so items could cite it; only its
        # outcome is still unknown.
        repo.finish_source(
            conn, outcome.source_id, status=status, error_msg=error_msg, seconds=seconds
        )
        return outcome.source_id

    # Failed before the source row existed (unreadable file, extraction error).
    # Recording it here is what makes the failure stick: the next ingest finds
    # this row by hash and skips the document rather than retrying it.
    return repo.insert_source(
        conn,
        filename=outcome.path.name,
        namespace=outcome.namespace,
        sha256=outcome.sha256,
        status=status,
        error_msg=error_msg,
        seconds=seconds,
    )


def record_documents(conn: Connection, outcomes: list[DocumentOutcome]) -> None:
    for outcome in outcomes:
        record_document(conn, outcome)
