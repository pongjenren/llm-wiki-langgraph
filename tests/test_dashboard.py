"""Telemetry recording, raw/ status computation, and dashboard route smoke."""

from __future__ import annotations

import json
from pathlib import Path

from llm_wiki import telemetry
from llm_wiki.dashboard import status as status_mod
from llm_wiki.dashboard.app import create_app
from llm_wiki.db import repo
from llm_wiki.pipeline import DocumentOutcome, ItemOutcome


def _outcome(path: Path, **kw) -> DocumentOutcome:
    return DocumentOutcome(path=path, namespace="ml", **kw)


def test_record_run_persists_counts_and_statuses(conn, tmp_path):
    outcomes = [
        _outcome(
            tmp_path / "a.md",
            elapsed_seconds=1.0,
            items=[
                ItemOutcome("A", "A", is_new_page=True, reference_number=1, needs_review=False),
                ItemOutcome("B", "B", is_new_page=False, reference_number=2, needs_review=True),
            ],
        ),
        _outcome(tmp_path / "b.md", skipped=True, skip_reason="duplicate", elapsed_seconds=0.1),
        _outcome(tmp_path / "c.md", error="RuntimeError: boom", elapsed_seconds=0.2),
        _outcome(
            tmp_path / "d.md",
            elapsed_seconds=0.3,
            items=[
                ItemOutcome("E", "E", is_new_page=False, reference_number=0,
                            needs_review=True, error="LLMError: nope"),
            ],
        ),
    ]

    run_id = telemetry.record_run(conn, outcomes, total_seconds=12.5)

    run = repo.get_run(conn, run_id)
    assert run["total_seconds"] == 12.5
    assert run["doc_count"] == 4
    assert (run["created"], run["merged"], run["skipped"], run["flagged"], run["failed"]) == (
        1, 1, 1, 1, 2,
    )

    docs = repo.list_run_docs(conn, run_id)
    assert [d["status"] for d in docs] == ["ok", "skipped", "failed", "failed"]

    a_items = json.loads(docs[0]["items_json"])
    assert [i["action"] for i in a_items] == ["created", "merged"]
    assert a_items[1]["reference_number"] == 2


def test_compute_raw_status_classifies_every_file(conn, workspace, doc):
    a = doc("a.md", "x")
    b = doc("b.md", "y")
    doc("c.md", "z")  # pre-telemetry: has a source row but no run
    doc("d.md", "w")  # never touched -> pending

    telemetry.record_run(
        conn,
        [
            _outcome(a, elapsed_seconds=0.5,
                     items=[ItemOutcome("A", "A", is_new_page=True, reference_number=1,
                                        needs_review=False)]),
            _outcome(b, error="RuntimeError: boom", elapsed_seconds=0.4),
        ],
        total_seconds=1.0,
    )
    repo.insert_source(conn, filename="c.md", namespace="ml", sha256="deadbeef")

    summary = status_mod.compute_raw_status(conn, workspace / "raw")
    by_name = {f.filename: f for f in summary.files}

    assert by_name["a.md"].status == status_mod.STATUS_OK
    assert by_name["b.md"].status == status_mod.STATUS_FAILED
    assert by_name["b.md"].error == "RuntimeError: boom"
    assert by_name["c.md"].status == status_mod.STATUS_OK  # from source, no run
    assert by_name["d.md"].status == status_mod.STATUS_PENDING
    assert summary.counts == {"pending": 1, "ok": 2, "skipped": 0, "failed": 1}


def test_relative_ingest_path_still_matches_the_raw_scan(conn, workspace, doc, monkeypatch):
    """Regression: `llm-wiki ingest raw/x.md` stored a cwd-relative path, which
    never matched the dashboard's absolute raw/ scan, so timing showed as '—'."""
    a = doc("a.md", "x")
    monkeypatch.chdir(workspace)

    telemetry.record_run(
        conn, [_outcome(Path("raw/ml/a.md"), elapsed_seconds=7.5)], total_seconds=8.0
    )

    summary = status_mod.compute_raw_status(conn, workspace / "raw")
    entry = {f.filename: f for f in summary.files}["a.md"]
    assert entry.status == status_mod.STATUS_OK
    assert entry.seconds == 7.5
    assert entry.last_processed is not None
    assert a.exists()


def test_legacy_relative_path_rows_still_match(conn, workspace, doc):
    """Rows written before paths were canonicalized are matched by filename."""
    doc("a.md", "x")
    run_id = repo.start_run(conn)
    repo.insert_ingest_doc(
        conn,
        run_id=run_id,
        namespace="ml",
        filename="a.md",
        path="raw/ml/a.md",  # legacy: relative, unresolvable from here
        status="ok",
        seconds=3.25,
    )
    repo.finish_run(
        conn, run_id, total_seconds=3.25, doc_count=1,
        created=1, merged=0, skipped=0, flagged=0, failed=0,
    )

    summary = status_mod.compute_raw_status(conn, workspace / "raw")
    entry = {f.filename: f for f in summary.files}["a.md"]
    assert entry.status == status_mod.STATUS_OK
    assert entry.seconds == 3.25


def test_dashboard_routes_render(conn, settings, doc):
    from starlette.testclient import TestClient

    a = doc("a.md", "x")
    run_id = telemetry.record_run(
        conn,
        [_outcome(a, elapsed_seconds=0.5,
                  items=[ItemOutcome("A", "A", is_new_page=True, reference_number=1,
                                     needs_review=False)])],
        total_seconds=2.0,
    )
    conn.close()  # dashboard opens its own connection

    client = TestClient(create_app(settings))

    home = client.get("/")
    assert home.status_code == 200
    assert "ingest dashboard" in home.text
    assert "a.md" in home.text

    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    assert "每檔 log" in detail.text

    assert client.get("/runs/9999").status_code == 404
