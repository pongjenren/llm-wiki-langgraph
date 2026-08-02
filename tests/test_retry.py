"""Retrying just the items a document failed on."""

from __future__ import annotations

import dataclasses

import pytest

from llm_wiki import telemetry
from llm_wiki.db import repo
from llm_wiki.llm.schemas import ExtractedItem, ResolveDecision
from llm_wiki.pipeline import (
    find_source_file,
    ingest_document,
    parse_failed_names,
    retry_failed_items,
)
from tests.conftest import ScriptedClient

NAMES = ["Alpha", "Beta", "Gamma"]


class FlakyClient(ScriptedClient):
    """Three unrelated concepts; the ones named in `broken` blow up when written.

    Failure is keyed off the prompt because that is the only place the item under
    way is visible from the client stand-in.
    """

    def __init__(self, broken: set[str] | None = None, names: list[str] | None = None) -> None:
        super().__init__()
        self.broken = broken or set()
        self.names = names or NAMES

    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        return [
            ExtractedItem(name=name, type="concept", description=f"About {name}.")
            for name in self.names
        ]

    def run_text(self, prompt: str, *, label: str = "step") -> str:
        for name in self.broken:
            if name in prompt:
                raise RuntimeError(f"provider exploded on {name}")
        return super().run_text(prompt, label=label)


@pytest.fixture
def failed_source(make_deps, doc, conn):
    """Ingest a document where one of three items fails, and record the outcome."""
    path = doc("a.md", "Alpha, Beta and Gamma.\n")
    outcome = ingest_document(make_deps(FlakyClient(broken={"Beta"})), "ml", path)
    assert not outcome.ok
    telemetry.record_document(conn, outcome)

    source = repo.get_source(conn, outcome.source_id)
    assert source["status"] == "failed"
    assert source["error_msg"].startswith("Beta: RuntimeError")
    return source


# --------------------------------------------------------------------------
# Reading the failed names back out
# --------------------------------------------------------------------------


def test_parse_failed_names_reads_one_name_per_line():
    assert parse_failed_names("Alpha: RuntimeError: boom\nBeta Gamma: ValueError: nope") == [
        "Alpha",
        "Beta Gamma",
    ]


def test_parse_failed_names_tolerates_multiline_errors():
    """A wrapped traceback adds junk names; the caller drops what never extracted."""
    names = parse_failed_names("Alpha: RuntimeError: boom\n  File 'x.py': line 3\nplain line")
    assert names[0] == "Alpha"
    assert "plain line" not in names, "a line without a separator is not a name"


def test_parse_failed_names_deduplicates_and_handles_empty():
    assert parse_failed_names("Alpha: one\nalpha: two") == ["Alpha"]
    assert parse_failed_names(None) == []
    assert parse_failed_names("") == []


# --------------------------------------------------------------------------
# The retry itself
# --------------------------------------------------------------------------


def test_retry_recovers_only_the_failed_item(make_deps, failed_source, conn, settings):
    deps = make_deps(FlakyClient())
    outcome = retry_failed_items(deps, failed_source)

    assert outcome.ok
    assert outcome.requested == ["Beta"]
    assert [item.name for item in outcome.items] == ["Beta"], (
        "the items that already succeeded must not be run again"
    )
    assert (settings.wiki_dir / "ml" / "Beta.md").exists()
    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_pages").fetchone()["n"] == 3


def test_retry_reuses_the_original_source_row(make_deps, failed_source, conn):
    deps = make_deps(FlakyClient())
    outcome = retry_failed_items(deps, failed_source)

    assert outcome.source_id == failed_source["id"]
    assert conn.execute("SELECT COUNT(*) AS n FROM source").fetchone()["n"] == 1, (
        "a retry must not write a second source row for the same document"
    )
    # One reference per page, all citing the one source: the row the failed item
    # already claimed in its half-finished create is reused, not duplicated.
    rows = conn.execute("SELECT wiki_id, reference_order FROM wiki_source").fetchall()
    assert len(rows) == 3
    assert all(row["reference_order"] == 1 for row in rows)


def test_retry_clears_the_source_failure(make_deps, failed_source, conn):
    outcome = retry_failed_items(make_deps(FlakyClient()), failed_source)
    telemetry.record_retry(conn, outcome)

    source = repo.get_source(conn, failed_source["id"])
    assert source["status"] == "ok"
    assert source["error_msg"] is None


def test_retry_that_fails_again_keeps_the_source_failed(make_deps, failed_source, conn):
    outcome = retry_failed_items(make_deps(FlakyClient(broken={"Beta"})), failed_source)
    telemetry.record_retry(conn, outcome)

    assert not outcome.ok
    source = repo.get_source(conn, failed_source["id"])
    assert source["status"] == "failed"
    assert source["error_msg"].startswith("Beta: RuntimeError")


def test_retry_leaves_the_successful_pages_untouched(make_deps, failed_source, settings):
    before = {
        name: (settings.wiki_dir / "ml" / f"{name}.md").read_text(encoding="utf-8")
        for name in ("Alpha", "Gamma")
    }

    retry_failed_items(make_deps(FlakyClient()), failed_source)

    for name, body in before.items():
        after = (settings.wiki_dir / "ml" / f"{name}.md").read_text(encoding="utf-8")
        assert after == body, f"{name} did not fail and must not be rewritten"


def test_retry_refuses_a_document_that_changed_on_disk(make_deps, failed_source, doc, conn):
    doc("a.md", "Alpha, Beta and Gamma, rewritten.\n")

    outcome = retry_failed_items(make_deps(FlakyClient()), failed_source)

    assert outcome.error is not None and "has changed" in outcome.error
    assert outcome.items == []
    telemetry.record_retry(conn, outcome)
    source = repo.get_source(conn, failed_source["id"])
    assert source["status"] == "failed", "a refused retry must not touch the recorded failure"
    assert source["error_msg"].startswith("Beta: RuntimeError")


def test_retry_reports_a_missing_raw_file(make_deps, failed_source, workspace):
    (workspace / "raw" / "ml" / "a.md").unlink()

    outcome = retry_failed_items(make_deps(FlakyClient()), failed_source)

    assert outcome.error is not None and "no longer under" in outcome.error


def test_retry_with_nothing_left_to_run_keeps_the_original_error(
    make_deps, failed_source, conn
):
    """Beta is no longer extracted, so nothing ran and nothing may be overwritten."""
    deps = make_deps(FlakyClient(names=["Alpha", "Gamma"]))
    outcome = retry_failed_items(deps, failed_source)

    assert outcome.items == []
    assert outcome.error is not None and "nothing to retry" in outcome.error
    assert not outcome.ok

    telemetry.record_retry(conn, outcome)
    source = repo.get_source(conn, failed_source["id"])
    assert source["status"] == "failed"
    assert source["error_msg"].startswith("Beta: RuntimeError"), (
        "a retry that ran nothing must leave the recorded error intact"
    )


def test_retry_records_a_name_that_no_longer_extracts_alongside_one_that_ran(
    make_deps, doc, conn
):
    """Two failed items, one of which vanishes: the survivor is fixed, the other stays."""
    path = doc("a.md", "Alpha, Beta and Gamma.\n")
    outcome = ingest_document(make_deps(FlakyClient(broken={"Beta", "Gamma"})), "ml", path)
    telemetry.record_document(conn, outcome)
    source = repo.get_source(conn, outcome.source_id)

    retried = retry_failed_items(make_deps(FlakyClient(names=["Alpha", "Beta"])), source)

    assert [item.name for item in retried.items] == ["Beta"]
    assert retried.unmatched == ["Gamma"]
    assert not retried.ok

    telemetry.record_retry(conn, retried)
    refreshed = repo.get_source(conn, source["id"])
    assert refreshed["status"] == "failed"
    assert refreshed["error_msg"] == "Gamma: not extracted on retry", (
        "the item that was fixed drops out; the one nothing ran for stays"
    )


def test_retry_refuses_a_document_level_failure(make_deps, conn, workspace):
    """A row for a document that was never read has no items to pick from."""
    source_id = repo.insert_source(
        conn,
        filename="broken.md",
        namespace="ml",
        sha256=None,
        status="failed",
        error_msg="UnicodeDecodeError: invalid start byte",
    )
    outcome = retry_failed_items(make_deps(FlakyClient()), repo.get_source(conn, source_id))

    assert outcome.error is not None and "failed before it was read" in outcome.error
    telemetry.record_retry(conn, outcome)
    assert repo.get_source(conn, source_id)["error_msg"].startswith("UnicodeDecodeError")


class SharedPageClient(ScriptedClient):
    """Beta merges into the page Alpha owns, and that merge is what fails."""

    def __init__(self, broken: set[str] | None = None) -> None:
        super().__init__()
        self.broken = broken or set()

    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        if document_index == 1:
            return [ExtractedItem(name="Alpha", type="concept", description="About Alpha.")]
        if document_index == 2:
            return [ExtractedItem(name="Beta", type="concept", description="About Beta.")]
        # Anything later is the retry re-extracting the second document, which
        # this time also names Alpha.
        return [
            ExtractedItem(name="Alpha", type="concept", description="About Alpha."),
            ExtractedItem(name="Beta", type="concept", description="About Beta."),
        ]

    def resolve(self) -> ResolveDecision:
        return ResolveDecision(matched_page_id=1, reason="same page")

    def run_text(self, prompt: str, *, label: str = "step") -> str:
        for name in self.broken:
            if name in prompt:
                raise RuntimeError(f"provider exploded on {name}")
        return super().run_text(prompt, label=label)


def test_retry_flags_a_page_it_shares_with_an_item_that_succeeded(make_deps, doc, conn):
    """Beta's failed merge left an alias onto Alpha's page, so the retry folds them.

    One assignment now covers both names, and re-running it puts Alpha's material
    through the page a second time. That is allowed, but flagged.
    """
    first = doc("a.md", "About Alpha.\n")
    second = doc("b.md", "About Beta.\n")

    client = SharedPageClient(broken={"Beta"})
    deps = make_deps(client)
    # Every existing page is a candidate, so resolving Beta reaches the judge.
    deps.settings = dataclasses.replace(deps.settings, string_candidate_threshold=0.0)

    ingest_document(deps, "ml", first)
    outcome = ingest_document(deps, "ml", second)
    assert not outcome.ok
    telemetry.record_document(conn, outcome)
    source = repo.get_source(conn, outcome.source_id)

    client.broken = set()
    retried = retry_failed_items(deps, source)

    assert retried.requested == ["Beta"]
    assert retried.collateral == ["Alpha"]
    assert len(retried.items) == 1
    assert retried.items[0].merged_names == ["Alpha", "Beta"]
    assert retried.items[0].needs_review, "a page carrying re-merged good material is flagged"
    assert (
        conn.execute("SELECT needs_review FROM wiki_pages WHERE page_id = 1").fetchone()[
            "needs_review"
        ]
        is True
    )


# --------------------------------------------------------------------------
# Locating the raw file
# --------------------------------------------------------------------------


def test_find_source_file_searches_sub_directories(workspace):
    nested = workspace / "raw" / "ml" / "papers"
    nested.mkdir()
    (nested / "deep.md").write_text("x", encoding="utf-8")

    assert find_source_file(workspace / "raw", "ml", "deep.md") == nested / "deep.md"


def test_find_source_file_refuses_an_ambiguous_name(workspace, doc):
    doc("dup.md", "x")
    nested = workspace / "raw" / "ml" / "papers"
    nested.mkdir()
    (nested / "dup.md").write_text("x", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="ambiguous"):
        find_source_file(workspace / "raw", "ml", "dup.md")


def test_find_source_file_reports_an_unknown_namespace(workspace):
    with pytest.raises(FileNotFoundError, match="no raw directory"):
        find_source_file(workspace / "raw", "nope", "a.md")
