"""Ingest pipeline behaviour, end to end, with scripted LLM replies."""

from __future__ import annotations

import pytest

from llm_wiki.llm.schemas import ExtractedItem, Review
from llm_wiki.pipeline import ingest_document
from tests.conftest import ScriptedClient

pytestmark = pytest.mark.asyncio

PAGE_V1 = """# Transformer

The Transformer is built on self-attention [1].

## Architecture

It replaces recurrence with attention [1].
"""

PAGE_MERGED = """# Transformer

The Transformer is built on self-attention [1].

## Architecture

It replaces recurrence with attention [1].

## Scaling

Transformers scale predictably [2].
"""

PAGE_MERGED_LOSSY = """# Transformer

The Transformer is built on self-attention [1].

## Scaling

Transformers scale predictably [2].
"""


class TransformerClient(ScriptedClient):
    """Two documents, both about the Transformer."""

    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        if document_index == 1:
            return [
                ExtractedItem(
                    name="Transformer",
                    type="concept",
                    description="Built on self-attention [CURRENT].",
                    aliases=["Transformer architecture"],
                )
            ]
        return [
            ExtractedItem(
                name="Transformer",
                type="concept",
                description="Scales predictably with parameter count [CURRENT].",
            )
        ]

    def create_page(self) -> str:
        return PAGE_V1

    def merge_page(self, attempt: int) -> str:
        return PAGE_MERGED


@pytest.fixture
def docs(doc):
    return (
        doc("doc1.md", "The Transformer is a sequence model built on self-attention.\n"),
        doc("doc2.md", "Transformer models scale predictably with parameter count.\n"),
    )


async def test_new_document_creates_page(make_deps, docs, settings, conn):
    deps = make_deps(TransformerClient())
    outcome = await ingest_document(deps, "ml", docs[0])

    assert outcome.ok
    assert len(outcome.items) == 1
    item = outcome.items[0]
    assert item.is_new_page
    assert item.reference_number == 1
    assert not item.needs_review

    body = (settings.wiki_dir / "ml" / "Transformer.md").read_text(encoding="utf-8")
    assert "[1]" in body
    assert "## References" in body and "doc1.md" in body
    assert "[CURRENT]" not in body
    assert (settings.wiki_dir / "ml" / "index.md").exists()


async def test_reingesting_same_bytes_short_circuits(make_deps, docs):
    client = TransformerClient()
    deps = make_deps(client)
    await ingest_document(deps, "ml", docs[0])
    calls_before = list(client.calls)

    outcome = await ingest_document(deps, "ml", docs[0])

    assert outcome.skipped
    assert outcome.items == []
    assert client.calls == calls_before, "a duplicate document must not reach the LLM"


async def test_second_document_merges_into_existing_page(make_deps, docs, settings, conn):
    deps = make_deps(TransformerClient())
    await ingest_document(deps, "ml", docs[0])
    outcome = await ingest_document(deps, "ml", docs[1])

    item = outcome.items[0]
    assert not item.is_new_page
    assert item.reference_number == 2

    body = (settings.wiki_dir / "ml" / "Transformer.md").read_text(encoding="utf-8")
    assert "[1]" in body and "[2]" in body
    assert "## Architecture" in body, "existing section must survive the merge"
    assert "## Scaling" in body
    assert body.count("## References") == 1
    assert "doc1.md" in body and "doc2.md" in body

    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_pages").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_source").fetchone()["n"] == 2


async def test_aliases_from_extraction_are_recorded(make_deps, docs, conn):
    deps = make_deps(TransformerClient())
    await ingest_document(deps, "ml", docs[0])

    rows = conn.execute("SELECT query_name, type FROM page_aliases").fetchall()
    by_name = {row["query_name"]: row["type"] for row in rows}
    assert by_name["transformer"] == "canonical"
    assert "transformer architecture" in by_name


class LossyMergeClient(TransformerClient):
    """First merge attempt drops a section; the second is correct."""

    def merge_page(self, attempt: int) -> str:
        return PAGE_MERGED_LOSSY if attempt == 0 else PAGE_MERGED


async def test_merge_that_drops_a_section_is_retried(make_deps, docs, settings):
    client = LossyMergeClient()
    deps = make_deps(client)
    await ingest_document(deps, "ml", docs[0])
    outcome = await ingest_document(deps, "ml", docs[1])

    assert client.count("merge-page") == 2
    assert not outcome.items[0].needs_review
    body = (settings.wiki_dir / "ml" / "Transformer.md").read_text(encoding="utf-8")
    assert "## Architecture" in body


class AlwaysLossyMergeClient(TransformerClient):
    def merge_page(self, attempt: int) -> str:
        return PAGE_MERGED_LOSSY


async def test_merge_that_never_recovers_is_flagged(make_deps, docs, conn, settings):
    deps = make_deps(AlwaysLossyMergeClient())
    await ingest_document(deps, "ml", docs[0])
    outcome = await ingest_document(deps, "ml", docs[1])

    assert outcome.items[0].needs_review
    assert conn.execute("SELECT needs_review FROM wiki_pages").fetchone()["needs_review"] == 1
    body = (settings.wiki_dir / "ml" / "Transformer.md").read_text(encoding="utf-8")
    assert "needs a human check" in body


class NeverPassesReviewClient(TransformerClient):
    """Item review always fails and refinement never improves it."""

    def item_review(self) -> Review:
        return Review(verdict="fail", issues=["description says nothing substantive"])

    def item_refine(self) -> ExtractedItem:
        return ExtractedItem(
            name="Transformer", type="concept", description="A thing [CURRENT]."
        )


async def test_item_failing_review_is_written_but_flagged(make_deps, docs, conn, settings):
    client = NeverPassesReviewClient()
    deps = make_deps(client)
    outcome = await ingest_document(deps, "ml", docs[0])

    assert client.count("item-refine") == settings.review_retries
    assert outcome.items[0].written_path is not None, "the page is still written"
    assert outcome.items[0].needs_review
    assert conn.execute("SELECT needs_review FROM wiki_pages").fetchone()["needs_review"] == 1
    assert "needs review" in (settings.wiki_dir / "ml" / "index.md").read_text(encoding="utf-8")


class VariantNameClient(ScriptedClient):
    """The same entity appears under a singular and a plural name."""

    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        name = "Self-attention network" if document_index == 1 else "Self-attention networks"
        return [ExtractedItem(name=name, type="concept", description=f"Described [CURRENT].")]

    def create_page(self) -> str:
        return "# Self-attention network\n\nProcesses sequences in parallel [1].\n"

    def merge_page(self, attempt: int) -> str:
        return "# Self-attention network\n\nParallel [1]. No recurrence [2].\n"


async def test_name_variants_resolve_to_one_page(make_deps, doc, conn):
    a = doc("a.md", "Self-attention networks process sequences in parallel.\n")
    b = doc("b.md", "Attention-based models avoid recurrence entirely.\n")
    deps = make_deps(VariantNameClient())

    await ingest_document(deps, "ml", a)
    outcome = await ingest_document(deps, "ml", b)

    assert not outcome.items[0].is_new_page, "the plural variant must not create a second page"
    assert outcome.items[0].reference_number == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_pages").fetchone()["n"] == 1
    alias = conn.execute(
        "SELECT type FROM page_aliases WHERE query_name = 'self-attention networks'"
    ).fetchone()
    assert alias["type"] == "embedding_sim"


class DistinctEntitiesClient(ScriptedClient):
    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        name = "Transformer" if document_index == 1 else "Convolutional neural network"
        return [ExtractedItem(name=name, type="concept", description="Described [CURRENT].")]


async def test_distinct_entities_get_separate_pages(make_deps, doc, conn):
    a = doc("a.md", "About transformers.\n")
    b = doc("b.md", "About convolutional networks.\n")
    deps = make_deps(DistinctEntitiesClient())

    await ingest_document(deps, "ml", a)
    outcome = await ingest_document(deps, "ml", b)

    assert outcome.items[0].is_new_page
    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_pages").fetchone()["n"] == 2


class TwoItemsSamePageClient(ScriptedClient):
    """One document yields two items naming the same entity."""

    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        return [
            ExtractedItem(name="Transformer", type="concept", description="First [CURRENT]."),
            ExtractedItem(name="Transformer", type="concept", description="Second [CURRENT]."),
        ]

    def create_page(self) -> str:
        return "# Transformer\n\nFirst [1].\n"

    def merge_page(self, attempt: int) -> str:
        return "# Transformer\n\nFirst [1]. Second [1].\n"


async def test_two_items_from_one_document_share_a_reference(make_deps, doc, conn):
    path = doc("a.md", "About transformers, twice.\n")
    outcome = await ingest_document(make_deps(TwoItemsSamePageClient()), "ml", path)

    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_pages").fetchone()["n"] == 1
    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_source").fetchone()["n"] == 1, (
        "a source cites a page once, however many items point at it"
    )
    assert [item.reference_number for item in outcome.items] == [1, 1]


class FailingItemClient(TransformerClient):
    def item_review(self) -> Review:
        raise RuntimeError("provider exploded")


async def test_item_error_is_contained(make_deps, docs):
    outcome = await ingest_document(make_deps(FailingItemClient()), "ml", docs[0])

    assert not outcome.ok
    assert outcome.items[0].error is not None
    assert outcome.items[0].needs_review


class ExtractionFailsClient(TransformerClient):
    async def run_json(self, prompt, schema, *, label="step"):
        if label == "extract":
            raise RuntimeError("provider exploded during extraction")
        return await super().run_json(prompt, schema, label=label)


async def test_document_failing_extraction_can_be_retried(make_deps, docs, conn):
    """A document that never got extracted must not look already-ingested."""
    await ingest_document(make_deps(ExtractionFailsClient()), "ml", docs[0])
    assert conn.execute("SELECT COUNT(*) AS n FROM source").fetchone()["n"] == 0

    outcome = await ingest_document(make_deps(TransformerClient()), "ml", docs[0])
    assert not outcome.skipped, "the failed document must be retried, not skipped"
    assert outcome.ok
