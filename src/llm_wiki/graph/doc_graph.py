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
from llm_wiki.llm.schemas import ExtractionResult, SummarizeDecision

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

        # The source row is recorded only once extraction has succeeded. Writing
        # it earlier would mean a document that failed mid-pipeline is treated as
        # already ingested on the next run, and could never be retried.
        with transaction(deps.conn):
            source_id = repo.insert_source(
                deps.conn,
                filename=state["filename"],
                namespace=state["namespace"],
                sha256=state["sha256"],
                timestamp=state.get("timestamp"),
            )
        return {"items": result.items, "source_id": source_id}

    def after_sha_check(state: DocState) -> str:
        return "skip" if state.get("skipped") else "continue"

    def after_summarize_decision(state: DocState) -> str:
        return "summarize" if state.get("summarized") else "extract"

    graph = StateGraph(DocState)
    graph.add_node("load_document", load_document)
    graph.add_node("check_sha", check_sha)
    graph.add_node("decide_summarize", decide_summarize)
    graph.add_node("summarize", summarize)
    graph.add_node("extract", extract)

    graph.add_edge(START, "load_document")
    graph.add_edge("load_document", "check_sha")
    graph.add_conditional_edges(
        "check_sha", after_sha_check, {"skip": END, "continue": "decide_summarize"}
    )
    graph.add_conditional_edges(
        "decide_summarize", after_summarize_decision, {"summarize": "summarize", "extract": "extract"}
    )
    graph.add_edge("summarize", "extract")
    graph.add_edge("extract", END)

    return graph.compile()
