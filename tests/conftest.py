"""Shared fixtures. Tests never call a real LLM: every step is scripted."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_wiki import embedding
from llm_wiki.config import Settings
from llm_wiki.db.connection import DROP_ALL_SQL, connect, init_db
from llm_wiki.graph import Deps
from llm_wiki.llm.schemas import (
    ExtractedItem,
    ExtractionResult,
    ResolveDecision,
    Review,
)


class ScriptedClient:
    """Stands in for LLMClient.

    Replies are looked up by step label. Subclasses override `extract_items` to
    control what each document yields.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.documents_seen = 0

    # -- hooks for tests ---------------------------------------------------
    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        raise NotImplementedError

    def extraction_review(self) -> Review:
        return Review(verdict="pass", issues=[])

    def resolve(self) -> ResolveDecision:
        # Default: none of the candidates is the same entity, so a new page is made.
        return ResolveDecision(matched_page_id=None, reason="no match")

    def extraction_refine(self, document_index: int) -> ExtractionResult:
        raise AssertionError("unexpected extraction refinement")

    def page_review(self) -> Review:
        return Review(verdict="pass", issues=[])

    def create_page(self) -> str:
        return "# Page\n\nA claim [1].\n"

    def merge_page(self, attempt: int) -> str:
        return "# Page\n\nA claim [1]. Another claim [2].\n"

    def refine_page(self) -> str:
        raise AssertionError("unexpected page refinement")

    # -- LLMClient interface -----------------------------------------------
    async def run_json(self, prompt: str, schema, *, label: str = "step"):
        self.calls.append(label)
        if label == "extract":
            self.documents_seen += 1
            return ExtractionResult(items=self.extract_items(self.documents_seen))
        if label == "extraction-review":
            return self.extraction_review()
        if label == "extraction-refine":
            return self.extraction_refine(self.documents_seen)
        if label == "page-review":
            return self.page_review()
        if label == "resolve":
            return self.resolve()
        raise AssertionError(f"unscripted run_json label: {label}")

    async def run_text(self, prompt: str, *, label: str = "step") -> str:
        self.calls.append(label)
        if label == "create-page":
            return self.create_page()
        if label.startswith("merge-page"):
            return self.merge_page(int(label.rsplit("#", 1)[1]))
        if label == "page-refine":
            return self.refine_page()
        raise AssertionError(f"unscripted run_text label: {label}")

    def count(self, label_prefix: str) -> int:
        return sum(1 for call in self.calls if call.startswith(label_prefix))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "raw" / "ml").mkdir(parents=True)
    return tmp_path


# Tests run against a real PostgreSQL (with pgvector). Point LLM_WIKI_TEST_DB_URL
# at a throwaway database; the conn fixture drops and recreates every table for
# each test, so its contents are not preserved.
TEST_DB_URL = os.getenv(
    "LLM_WIKI_TEST_DB_URL", "postgresql://llm_wiki:llm_wiki@localhost:5432/llm_wiki"
)


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings(
        raw_dir=workspace / "raw",
        wiki_dir=workspace / "wiki",
        db_url=TEST_DB_URL,
    )


@pytest.fixture
def conn(settings: Settings):
    try:
        connection = connect(settings.db_url)
    except Exception as exc:  # no reachable Postgres -> nothing to test against
        pytest.skip(f"PostgreSQL not available at {settings.db_url}: {exc}")
    # Start each test from a clean schema so tests never see each other's rows.
    connection.execute(DROP_ALL_SQL)
    init_db(connection, embedding.embedding_dim(model_name=settings.embedding_model))
    yield connection
    connection.close()


@pytest.fixture
def make_deps(conn, settings):
    def _make(client) -> Deps:
        return Deps(conn=conn, client=client, settings=settings)

    return _make


@pytest.fixture
def doc(workspace: Path):
    def _write(name: str, text: str) -> Path:
        path = workspace / "raw" / "ml" / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write
