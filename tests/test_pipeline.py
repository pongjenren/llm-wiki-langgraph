"""Ingest pipeline behaviour, end to end, with scripted LLM replies."""

from __future__ import annotations

import dataclasses

import pytest

from llm_wiki.llm.schemas import ExtractedItem, ExtractionResult, ResolveDecision, Review
from llm_wiki.pipeline import ingest_document
from tests.conftest import ScriptedClient


def _always_ask_llm(deps):
    """Neutralise both auto-accept tiers so resolution always reaches the LLM judge.

    Auto-accept depends on exact embedding distances, which vary by model; forcing
    the judge path keeps these tests about the judge, not the embedding backend.
    """
    deps.settings = dataclasses.replace(
        deps.settings,
        string_candidate_threshold=0.0,  # every existing page is a candidate
        string_autoaccept_threshold=101.0,  # unreachable: never string auto-accept
        embedding_autoaccept_threshold=-1.0,  # unreachable: never embedding auto-accept
    )
    return deps

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
                    description="Built on self-attention.",
                    aliases=["Transformer architecture"],
                )
            ]
        return [
            ExtractedItem(
                name="Transformer",
                type="concept",
                description="Scales predictably with parameter count.",
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


class VariantNameClient(ScriptedClient):
    """The same entity appears under a singular and a plural name."""

    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        name = "Self-attention network" if document_index == 1 else "Self-attention networks"
        return [ExtractedItem(name=name, type="concept", description=f"Described.")]

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
    assert alias["type"] == "string_sim"


class DistinctEntitiesClient(ScriptedClient):
    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        name = "Transformer" if document_index == 1 else "Convolutional neural network"
        return [ExtractedItem(name=name, type="concept", description="Described.")]


async def test_distinct_entities_get_separate_pages(make_deps, doc, conn):
    a = doc("a.md", "About transformers.\n")
    b = doc("b.md", "About convolutional networks.\n")
    deps = make_deps(DistinctEntitiesClient())

    await ingest_document(deps, "ml", a)
    outcome = await ingest_document(deps, "ml", b)

    assert outcome.items[0].is_new_page
    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_pages").fetchone()["n"] == 2


class TwoConceptsClient(ScriptedClient):
    """Two differently-named concepts, one per document."""

    def __init__(self, decision: ResolveDecision) -> None:
        super().__init__()
        self._decision = decision

    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        name = "Gradient descent" if document_index == 1 else "Backpropagation"
        return [ExtractedItem(name=name, type="concept", description="Described.")]

    def resolve(self) -> ResolveDecision:
        return self._decision

    def create_page(self) -> str:
        return "# Page\n\nA claim [1].\n"

    def merge_page(self, attempt: int) -> str:
        return "# Page\n\nA claim [1]. Another claim [2].\n"


async def test_llm_judge_keeps_distinct_items_apart(make_deps, doc, conn):
    a = doc("a.md", "Gradient descent minimises a loss.\n")
    b = doc("b.md", "Backpropagation computes gradients.\n")
    deps = _always_ask_llm(make_deps(TwoConceptsClient(ResolveDecision(matched_page_id=None, reason="different"))))

    await ingest_document(deps, "ml", a)
    outcome = await ingest_document(deps, "ml", b)

    assert deps.client.count("resolve") == 1, "the second item must consult the judge"
    assert outcome.items[0].is_new_page
    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_pages").fetchone()["n"] == 2


async def test_llm_judge_merges_when_it_returns_a_page_id(make_deps, doc, conn):
    a = doc("a.md", "Gradient descent minimises a loss.\n")
    b = doc("b.md", "Backpropagation computes gradients.\n")
    deps = _always_ask_llm(make_deps(TwoConceptsClient(ResolveDecision(matched_page_id=1, reason="same"))))

    await ingest_document(deps, "ml", a)
    outcome = await ingest_document(deps, "ml", b)

    assert not outcome.items[0].is_new_page
    assert outcome.items[0].reference_number == 2
    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_pages").fetchone()["n"] == 1
    alias = conn.execute(
        "SELECT type FROM page_aliases WHERE query_name = 'backpropagation'"
    ).fetchone()
    assert alias["type"] == "llm_sim"


async def test_llm_low_confidence_match_flags_review(make_deps, doc, conn):
    a = doc("a.md", "Gradient descent minimises a loss.\n")
    b = doc("b.md", "Backpropagation computes gradients.\n")
    decision = ResolveDecision(matched_page_id=1, confidence="low", reason="maybe")
    deps = _always_ask_llm(make_deps(TwoConceptsClient(decision)))

    await ingest_document(deps, "ml", a)
    outcome = await ingest_document(deps, "ml", b)

    assert not outcome.items[0].is_new_page
    assert outcome.items[0].needs_review, "an uncertain merge must ask for a human check"
    assert conn.execute("SELECT needs_review FROM wiki_pages").fetchone()["needs_review"] == 1


class TwoItemsSamePageClient(ScriptedClient):
    """One document yields two items naming the same entity."""

    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        return [
            ExtractedItem(name="Transformer", type="concept", description="First."),
            ExtractedItem(name="Transformer", type="concept", description="Second."),
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
    def page_review(self) -> Review:
        raise RuntimeError("provider exploded")


async def test_item_error_is_contained(make_deps, docs):
    outcome = await ingest_document(make_deps(FailingItemClient()), "ml", docs[0])

    assert not outcome.ok
    assert outcome.items[0].error is not None
    assert outcome.items[0].needs_review


class SpuriousExtractionClient(TransformerClient):
    """Extraction over-produces; the extraction review drops the bad item."""

    def extract_items(self, document_index: int) -> list[ExtractedItem]:
        return [
            ExtractedItem(name="Transformer", type="concept", description="Built on attention."),
            ExtractedItem(name="Passing mention", type="entity", description="Mentioned once."),
        ]

    def extraction_review(self) -> Review:
        # Fails the first look, passes once the spurious item has been dropped.
        if self.count("extraction-refine") == 0:
            return Review(verdict="fail", issues=["'Passing mention' is only mentioned in passing"])
        return Review(verdict="pass", issues=[])

    def extraction_refine(self, document_index: int) -> ExtractionResult:
        return ExtractionResult(
            items=[
                ExtractedItem(
                    name="Transformer", type="concept", description="Built on attention."
                )
            ]
        )


async def test_extraction_review_drops_spurious_item(make_deps, docs, conn):
    client = SpuriousExtractionClient()
    outcome = await ingest_document(make_deps(client), "ml", docs[0])

    assert client.count("extraction-refine") == 1
    assert [item.name for item in outcome.items] == ["Transformer"], "the spurious item is gone"
    assert conn.execute("SELECT COUNT(*) AS n FROM wiki_pages").fetchone()["n"] == 1


class UnrecoverableExtractionClient(TransformerClient):
    """Extraction review never passes; after the retries we proceed as-is."""

    def extraction_review(self) -> Review:
        return Review(verdict="fail", issues=["still not right"])

    def extraction_refine(self, document_index: int) -> ExtractionResult:
        return ExtractionResult(items=self.extract_items(document_index))


async def test_extraction_review_gives_up_after_retries(make_deps, docs, settings, conn):
    client = UnrecoverableExtractionClient()
    outcome = await ingest_document(make_deps(client), "ml", docs[0])

    assert client.count("extraction-refine") == settings.review_retries
    assert outcome.ok, "a document that exhausts extraction review is still ingested"
    assert conn.execute("SELECT COUNT(*) AS n FROM source").fetchone()["n"] == 1


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
