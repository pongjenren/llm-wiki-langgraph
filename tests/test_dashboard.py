"""Source outcome recording, raw/ status computation, and dashboard route smoke."""

from __future__ import annotations

from pathlib import Path

from llm_wiki import telemetry
from llm_wiki.dashboard import status as status_mod
from llm_wiki.dashboard.app import create_app
from llm_wiki.db import repo
from llm_wiki.pipeline import DocumentOutcome, ItemOutcome


def _outcome(path: Path, **kw) -> DocumentOutcome:
    return DocumentOutcome(path=path, namespace="ml", **kw)


def test_successful_document_stamps_its_existing_source_row(conn, tmp_path):
    """The source row exists by the time items run; recording only fills it in."""
    source_id = repo.insert_source(conn, filename="a.md", namespace="ml", sha256="aaa")

    telemetry.record_document(
        conn,
        _outcome(
            tmp_path / "a.md",
            source_id=source_id,
            sha256="aaa",
            elapsed_seconds=1.25,
            items=[
                ItemOutcome("A", "A", is_new_page=True, reference_number=1, needs_review=False),
                ItemOutcome("B", "B", is_new_page=False, reference_number=2, needs_review=True),
            ],
        ),
    )

    source = repo.get_source(conn, source_id)
    assert source["status"] == "ok"
    assert source["error_msg"] is None  # successful items are not recorded
    assert source["seconds"] == 1.25


def test_failed_items_are_listed_by_name_in_error_msg(conn, tmp_path):
    source_id = repo.insert_source(conn, filename="a.md", namespace="ml", sha256="aaa")

    telemetry.record_document(
        conn,
        _outcome(
            tmp_path / "a.md",
            source_id=source_id,
            sha256="aaa",
            elapsed_seconds=0.3,
            items=[
                ItemOutcome("A", "A", is_new_page=True, reference_number=1, needs_review=False),
                ItemOutcome("E", "E", is_new_page=False, reference_number=0,
                            needs_review=True, error="LLMError: nope"),
            ],
        ),
    )

    source = repo.get_source(conn, source_id)
    assert source["status"] == "failed"
    assert source["error_msg"] == "E: LLMError: nope"
    assert "A" not in source["error_msg"]


def test_document_failing_before_its_source_row_still_gets_one(conn, tmp_path):
    """A document that died during extraction is recorded under its own hash, so
    the next ingest finds it and does not retry it."""
    source_id = telemetry.record_document(
        conn,
        _outcome(
            tmp_path / "c.md",
            sha256="ccc",
            error="RuntimeError: boom",
            elapsed_seconds=0.2,
        ),
    )

    source = repo.get_source(conn, source_id)
    assert source["status"] == "failed"
    assert source["error_msg"] == "RuntimeError: boom"
    assert source["sha256"] == "ccc"
    assert repo.find_source_by_sha(conn, "ml", "ccc") is not None


def test_unreadable_document_is_recorded_without_a_hash(conn, tmp_path):
    source_id = telemetry.record_document(
        conn, _outcome(tmp_path / "bad.xlsx", error="UnsupportedFileType: nope")
    )

    source = repo.get_source(conn, source_id)
    assert source["status"] == "failed"
    assert source["sha256"] is None


def test_duplicate_document_writes_nothing(conn, tmp_path):
    source_id = repo.insert_source(conn, filename="a.md", namespace="ml", sha256="aaa")

    telemetry.record_document(
        conn,
        _outcome(
            tmp_path / "a.md",
            skipped=True,
            skip_reason=f"already ingested as source #{source_id}",
            source_id=source_id,
            sha256="aaa",
            elapsed_seconds=0.1,
        ),
    )

    assert len(repo.list_sources(conn)) == 1
    assert repo.get_source(conn, source_id)["seconds"] is None  # untouched


def test_compute_raw_status_classifies_every_file(conn, workspace, doc):
    a = doc("a.md", "x")
    b = doc("b.md", "y")
    doc("d.md", "w")  # never ingested -> pending

    telemetry.record_documents(
        conn,
        [
            _outcome(a, sha256="aaa", elapsed_seconds=0.5,
                     items=[ItemOutcome("A", "A", is_new_page=True, reference_number=1,
                                        needs_review=False)]),
            _outcome(b, sha256="bbb", error="RuntimeError: boom", elapsed_seconds=0.4),
        ],
    )

    summary = status_mod.compute_raw_status(conn, workspace / "raw")
    by_name = {f.filename: f for f in summary.files}

    assert by_name["a.md"].status == status_mod.STATUS_OK
    assert by_name["a.md"].seconds == 0.5
    assert by_name["a.md"].last_processed is not None
    assert by_name["b.md"].status == status_mod.STATUS_FAILED
    assert by_name["b.md"].error_msg == "RuntimeError: boom"
    assert by_name["d.md"].status == status_mod.STATUS_PENDING
    assert summary.counts == {"pending": 1, "ok": 1, "failed": 1}


def test_raw_status_matches_by_filename_whatever_path_was_ingested(conn, workspace, doc,
                                                                   monkeypatch):
    """`llm-wiki ingest raw/ml/a.md` and an absolute path record the same file."""
    doc("a.md", "x")
    monkeypatch.chdir(workspace)

    telemetry.record_document(
        conn, _outcome(Path("raw/ml/a.md"), sha256="aaa", elapsed_seconds=7.5)
    )

    summary = status_mod.compute_raw_status(conn, workspace / "raw")
    entry = {f.filename: f for f in summary.files}["a.md"]
    assert entry.status == status_mod.STATUS_OK
    assert entry.seconds == 7.5


def test_dashboard_routes_render(conn, settings, doc):
    from starlette.testclient import TestClient

    a = doc("a.md", "x")
    source_id = telemetry.record_document(conn, _outcome(a, sha256="aaa", elapsed_seconds=0.5))
    conn.close()  # dashboard opens its own connection

    client = TestClient(create_app(settings))

    home = client.get("/")
    assert home.status_code == 200
    assert "ingest dashboard" in home.text
    assert "a.md" in home.text

    detail = client.get(f"/sources/{source_id}")
    assert detail.status_code == 200
    assert "建立/合併的頁面" in detail.text

    assert client.get("/sources/9999").status_code == 404
