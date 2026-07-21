"""Stage 1: raw file -> SHA dedup -> optional summarize -> entity/concept list."""

from __future__ import annotations

import logging
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from llm_wiki import loaders
from llm_wiki.db import repo
from llm_wiki.db.connection import transaction
from llm_wiki.graph.state import Deps, DocState
from llm_wiki.llm import prompts
from llm_wiki.llm.schemas import ExtractionResult, Review, SummarizeDecision

log = logging.getLogger(__name__)


def build_doc_graph(deps: Deps):
    """Compile the stage-1 graph."""

    async def load_document(state: DocState) -> DocState:
        document = loaders.load(Path(state["path"]))
        return {
            "filename": document.path.name,
            "text": document.text,
            "sha256": document.sha256,
            "timestamp": document.timestamp,
            "working_text": document.text,
        }

    async def check_sha(state: DocState) -> DocState:
        """Skip documents already ingested into this namespace."""
        existing = repo.find_source_by_sha(deps.conn, state["namespace"], state["sha256"])
        if existing is not None:
            log.info("skip %s: already ingested as source #%s", state["filename"], existing["id"])
            return {
                "skipped": True,
                "skip_reason": f"already ingested as source #{existing['id']}",
                "source_id": int(existing["id"]),
            }
        return {"skipped": False}

    async def decide_summarize(state: DocState) -> DocState:
        decision = await deps.client.run_json(
            prompts.summarize_decision(state["filename"], state["text"]),
            SummarizeDecision,
            label="summarize-decision",
        )
        log.info(
            "summarize %s: %s (%s)",
            state["filename"],
            decision.should_summarize,
            decision.reason,
        )
        return {"summarized": decision.should_summarize}

    async def summarize(state: DocState) -> DocState:
        text = await deps.client.run_text(
            prompts.summarize(state["filename"], state["text"]), label="summarize"
        )
        return {"working_text": text}

    async def extract(state: DocState) -> DocState:
        result = await deps.client.run_json(
            prompts.extract(state["filename"], state["working_text"]),
            ExtractionResult,
            label="extract",
        )
        log.info("extracted %d item(s) from %s", len(result.items), state["filename"])
        return {"items": result.items}

    async def review_extraction(state: DocState) -> DocState:
        """Check the whole item set against the source: recall, precision, type."""
        review = await deps.client.run_json(
            prompts.review_extraction(state["filename"], state["working_text"], state["items"]),
            Review,
            label="extraction-review",
        )
        if review.verdict == "pass":
            return {"extraction_issues": []}
        log.info("extraction review failed for %s: %s", state["filename"], review.issues)
        return {"extraction_issues": review.issues}

    async def refine_extraction(state: DocState) -> DocState:
        """Rebuild the item set from the source, adding, dropping, and fixing."""
        result = await deps.client.run_json(
            prompts.refine_extraction(
                state["filename"],
                state["working_text"],
                state["items"],
                state["extraction_issues"],
            ),
            ExtractionResult,
            label="extraction-refine",
        )
        log.info("refined extraction for %s: %d item(s)", state["filename"], len(result.items))
        return {
            "items": result.items,
            "extraction_attempts": state.get("extraction_attempts", 0) + 1,
        }

    async def record_source(state: DocState) -> DocState:
        # The source row is recorded only once extraction has succeeded and been
        # reviewed. Writing it earlier would mean a document that failed
        # mid-pipeline is treated as already ingested on the next run, and could
        # never be retried.
        with transaction(deps.conn):
            source_id = repo.insert_source(
                deps.conn,
                filename=state["filename"],
                namespace=state["namespace"],
                sha256=state["sha256"],
                timestamp=state.get("timestamp"),
            )
        return {"source_id": source_id}

    def after_sha_check(state: DocState) -> str:
        return "skip" if state.get("skipped") else "continue"

    def after_summarize_decision(state: DocState) -> str:
        return "summarize" if state.get("summarized") else "extract"

    def after_extraction_review(state: DocState) -> str:
        if not state.get("extraction_issues"):
            return "accept"
        if state.get("extraction_attempts", 0) >= deps.settings.review_retries:
            log.warning(
                "extraction for %s still failing review after %d refine attempt(s); accepting as-is",
                state["filename"],
                state.get("extraction_attempts", 0),
            )
            return "accept"
        return "refine"

    graph = StateGraph(DocState)
    graph.add_node("load_document", load_document)
    graph.add_node("check_sha", check_sha)
    graph.add_node("decide_summarize", decide_summarize)
    graph.add_node("summarize", summarize)
    graph.add_node("extract", extract)
    graph.add_node("review_extraction", review_extraction)
    graph.add_node("refine_extraction", refine_extraction)
    graph.add_node("record_source", record_source)

    graph.add_edge(START, "load_document")
    graph.add_edge("load_document", "check_sha")
    graph.add_conditional_edges(
        "check_sha", after_sha_check, {"skip": END, "continue": "decide_summarize"}
    )
    graph.add_conditional_edges(
        "decide_summarize", after_summarize_decision, {"summarize": "summarize", "extract": "extract"}
    )
    graph.add_edge("summarize", "extract")
    graph.add_edge("extract", "review_extraction")
    graph.add_conditional_edges(
        "review_extraction",
        after_extraction_review,
        {"accept": "record_source", "refine": "refine_extraction"},
    )
    graph.add_edge("refine_extraction", "review_extraction")
    graph.add_edge("record_source", END)

    return graph.compile()
