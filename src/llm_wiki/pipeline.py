"""Drives the two graphs: a document is extracted, then its items are folded in."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm_wiki.graph import Deps, build_doc_graph, build_item_graph

log = logging.getLogger(__name__)


@dataclass
class ItemOutcome:
    name: str
    page_name: str
    is_new_page: bool
    reference_number: int
    needs_review: bool
    page_id: int | None = None
    written_path: str | None = None
    error: str | None = None


@dataclass
class DocumentOutcome:
    path: Path
    namespace: str
    skipped: bool = False
    skip_reason: str | None = None
    summarized: bool = False
    source_id: int | None = None
    items: list[ItemOutcome] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and all(item.error is None for item in self.items)


async def ingest_document(deps: Deps, namespace: str, path: Path) -> DocumentOutcome:
    """Run a single document through both stages."""
    doc_graph = build_doc_graph(deps)
    item_graph = build_item_graph(deps)
    outcome = DocumentOutcome(path=path, namespace=namespace)
    start = time.monotonic()

    try:
        try:
            doc_state = await doc_graph.ainvoke({"namespace": namespace, "path": str(path)})
        except Exception as exc:  # a bad document must not abort the whole run
            log.error("failed to process %s: %s", path, exc)
            log.debug("traceback for %s", path, exc_info=True)
            outcome.error = f"{type(exc).__name__}: {exc}"
            return outcome

        outcome.source_id = doc_state.get("source_id")
        outcome.summarized = bool(doc_state.get("summarized"))

        if doc_state.get("skipped"):
            outcome.skipped = True
            outcome.skip_reason = doc_state.get("skip_reason")
            return outcome

        # Serial by design: concurrent items would race to create the same page.
        for item in doc_state.get("items", []):
            try:
                item_state = await item_graph.ainvoke(
                    {
                        "namespace": namespace,
                        "source_id": doc_state["source_id"],
                        "item": item,
                        "item_attempts": 0,
                        "page_attempts": 0,
                        "needs_review": False,
                    }
                )
            except Exception as exc:
                log.error("failed to process item %r from %s: %s", item.name, path, exc)
                log.debug("traceback for item %r", item.name, exc_info=True)
                outcome.items.append(
                    ItemOutcome(
                        name=item.name,
                        page_name=item.name,
                        is_new_page=False,
                        reference_number=0,
                        needs_review=True,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            outcome.items.append(
                ItemOutcome(
                    name=item.name,
                    page_name=item_state.get("page_name", item.name),
                    is_new_page=bool(item_state.get("is_new_page")),
                    reference_number=int(item_state.get("reference_number", 0)),
                    needs_review=bool(item_state.get("needs_review")),
                    page_id=item_state.get("page_id"),
                    written_path=item_state.get("written_path"),
                )
            )

        return outcome
    finally:
        outcome.elapsed_seconds = time.monotonic() - start


async def ingest_documents(deps: Deps, targets: list[tuple[str, Path]]) -> list[DocumentOutcome]:
    return [await ingest_document(deps, namespace, path) for namespace, path in targets]
