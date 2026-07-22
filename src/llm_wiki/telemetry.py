"""Persist ingest outcomes as run/document telemetry for the dashboard.

This is a pure side-channel: it records what happened, and never changes the
ingest result. Recording is best-effort — a failure to write telemetry must not
turn a successful ingest into a failed command — so callers wrap it in a guard.
"""

from __future__ import annotations

import json

from llm_wiki.db import repo
from llm_wiki.db.connection import Connection
from llm_wiki.pipeline import DocumentOutcome


def _doc_status(outcome: DocumentOutcome) -> str:
    if outcome.error is not None or any(item.error for item in outcome.items):
        return "failed"
    if outcome.skipped:
        return "skipped"
    return "ok"


def _items_payload(outcome: DocumentOutcome) -> list[dict]:
    """Per-item log detail, mirroring what the CLI prints, for the dashboard."""
    payload: list[dict] = []
    for item in outcome.items:
        if item.error:
            action = "failed"
        elif item.is_new_page:
            action = "created"
        else:
            action = "merged"
        payload.append(
            {
                "name": item.name,
                "page_name": item.page_name,
                "action": action,
                "reference_number": item.reference_number,
                "needs_review": item.needs_review,
                "error": item.error,
            }
        )
    return payload


def record_run(
    conn: Connection, outcomes: list[DocumentOutcome], total_seconds: float
) -> int:
    """Write one ingest_run plus one ingest_doc per document. Returns run_id."""
    run_id = repo.start_run(conn)

    created = merged = skipped = flagged = failed = 0
    for outcome in outcomes:
        if outcome.error:
            failed += 1
        if outcome.skipped:
            skipped += 1
        for item in outcome.items:
            if item.error:
                failed += 1
                continue
            if item.is_new_page:
                created += 1
            else:
                merged += 1
            if item.needs_review:
                flagged += 1

        repo.insert_ingest_doc(
            conn,
            run_id=run_id,
            namespace=outcome.namespace,
            filename=outcome.path.name,
            # Always canonical: `ingest raw/x.md` and `ingest /abs/raw/x.md`
            # must produce the same key, or the dashboard cannot match the file
            # back to the raw/ scan.
            path=str(outcome.path.resolve()),
            status=_doc_status(outcome),
            seconds=round(outcome.elapsed_seconds, 3),
            skip_reason=outcome.skip_reason,
            error=outcome.error,
            items_json=json.dumps(_items_payload(outcome), ensure_ascii=False),
        )

    repo.finish_run(
        conn,
        run_id,
        total_seconds=round(total_seconds, 3),
        doc_count=len(outcomes),
        created=created,
        merged=merged,
        skipped=skipped,
        flagged=flagged,
        failed=failed,
    )
    return run_id
