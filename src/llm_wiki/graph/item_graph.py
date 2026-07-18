"""Stage 2: one entity/concept item is folded into a new or existing wiki page.

Items are processed one at a time. Serial execution is what makes the reference
numbering and the "does this page already exist?" check safe: two items naming
the same entity would otherwise race to create two pages for it.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from llm_wiki import embedding, pages
from llm_wiki.db import repo
from llm_wiki.db.connection import transaction
from llm_wiki.graph.state import Deps, ItemState
from llm_wiki.llm import prompts
from llm_wiki.llm.schemas import ExtractedItem, Review

log = logging.getLogger(__name__)


def build_item_graph(deps: Deps):
    """Compile the stage-2 graph."""
    settings = deps.settings

    async def review_item(state: ItemState) -> ItemState:
        review = await deps.client.run_json(
            prompts.review_item(state["item"]), Review, label="item-review"
        )
        if review.verdict == "pass":
            return {"item_issues": []}
        log.info("item review failed for %r: %s", state["item"].name, review.issues)
        return {"item_issues": review.issues}

    async def refine_item(state: ItemState) -> ItemState:
        refined = await deps.client.run_json(
            prompts.refine_item(state["item"], state["item_issues"]),
            ExtractedItem,
            label="item-refine",
        )
        return {"item": refined, "item_attempts": state.get("item_attempts", 0) + 1}

    async def resolve_entity(state: ItemState) -> ItemState:
        """Find the page this item belongs to: exact alias first, then vectors."""
        item = state["item"]
        namespace = state["namespace"]

        match = repo.find_page_by_alias(deps.conn, namespace, item.name)
        if match is None:
            vector = embedding.embed(
                embedding.identity_text(item.name), model_name=settings.embedding_model
            )
            match = repo.find_page_by_embedding(
                deps.conn, namespace, vector, settings.similarity_threshold
            )

        if match is None:
            log.info("new page for %r", item.name)
            return {"is_new_page": True, "page_name": item.name}

        log.info("%r resolves to existing page #%s (%s)", item.name, match.page_id, match.how)
        return {
            "is_new_page": False,
            "page_id": match.page_id,
            "page_name": match.page_name,
            "matched_how": match.how,
        }

    def _record_aliases(item: ExtractedItem, namespace: str, page_id: int, canonical: bool) -> None:
        if canonical:
            repo.upsert_alias(
                deps.conn, namespace=namespace, name=item.name, page_id=page_id, type_="canonical"
            )
        else:
            # The name we searched under resolved to this page by similarity;
            # recording it makes the next lookup an exact hit.
            repo.upsert_alias(
                deps.conn, namespace=namespace, name=item.name, page_id=page_id, type_="embedding_sim"
            )
        for alias in item.aliases:
            repo.upsert_alias(
                deps.conn, namespace=namespace, name=alias, page_id=page_id, type_="embedding_sim"
            )

    async def create_page(state: ItemState) -> ItemState:
        """Insert the page and its first reference, then write its body."""
        item = state["item"]
        namespace = state["namespace"]

        vector = embedding.embed(
            embedding.identity_text(item.name), model_name=settings.embedding_model
        )
        with transaction(deps.conn):
            page_id = repo.insert_page(
                deps.conn, page_name=item.name, namespace=namespace, type_=item.type
            )
            _record_aliases(item, namespace, page_id, canonical=True)
            repo.upsert_embedding(deps.conn, page_id, vector)
            reference_number = repo.link_source(
                deps.conn, wiki_id=page_id, source_id=state["source_id"], namespace=namespace
            )

        numbered = item.model_copy(
            update={"description": pages.substitute_current(item.description, reference_number)}
        )
        body = await deps.client.run_text(
            prompts.create_page(numbered, reference_number), label="create-page"
        )
        return {
            "page_id": page_id,
            "page_name": item.name,
            "reference_number": reference_number,
            "existing_body": "",
            "body": pages.strip_references(body),
        }

    async def merge_page(state: ItemState) -> ItemState:
        """Link the new source to an existing page and integrate the material."""
        item = state["item"]
        namespace = state["namespace"]
        page_id = state["page_id"]

        with transaction(deps.conn):
            _record_aliases(item, namespace, page_id, canonical=False)
            reference_number = repo.link_source(
                deps.conn, wiki_id=page_id, source_id=state["source_id"], namespace=namespace
            )

        existing_body = pages.read_page(deps.wiki_dir, namespace, state["page_name"])
        numbered = item.model_copy(
            update={"description": pages.substitute_current(item.description, reference_number)}
        )

        prompt = prompts.merge_page(numbered, existing_body, reference_number)
        merged = ""
        problems: list[str] = []

        # A full rewrite can silently drop content, so the result is checked
        # programmatically and re-requested with the specific loss named.
        for attempt in range(settings.review_retries + 1):
            merged = pages.strip_references(
                await deps.client.run_text(prompt, label=f"merge-page#{attempt}")
            )
            problems = pages.validate_merge(existing_body, merged)
            if not problems:
                break
            log.info("merge validation failed for %r: %s", state["page_name"], problems)
            prompt = (
                f"{prompts.merge_page(numbered, existing_body, reference_number)}\n\n"
                f"Your previous merge was rejected: {pages.format_issues(problems)}\n"
                "Produce the merged page again, preserving everything named above."
            )

        return {
            "reference_number": reference_number,
            "existing_body": existing_body,
            "body": merged,
            # Never clear a flag an earlier step set: an item that failed its own
            # review still needs a human even if the merge came out clean.
            "needs_review": bool(problems) or bool(state.get("needs_review")),
            "page_issues": problems,
        }

    async def review_page(state: ItemState) -> ItemState:
        review = await deps.client.run_json(
            prompts.review_page(state["page_name"], state["body"]), Review, label="page-review"
        )
        if review.verdict == "pass":
            return {"page_issues": []}
        log.info("page review failed for %r: %s", state["page_name"], review.issues)
        return {"page_issues": review.issues}

    async def refine_page(state: ItemState) -> ItemState:
        body = await deps.client.run_text(
            prompts.refine_page(state["body"], state["page_issues"]), label="page-refine"
        )
        return {
            "body": pages.strip_references(body),
            "page_attempts": state.get("page_attempts", 0) + 1,
        }

    async def persist(state: ItemState) -> ItemState:
        """Write the page and refresh the namespace index."""
        namespace = state["namespace"]
        needs_review = bool(state.get("needs_review"))

        with transaction(deps.conn):
            repo.set_needs_review(deps.conn, state["page_id"], needs_review)

        path = pages.write_page(
            deps.conn,
            wiki_dir=deps.wiki_dir,
            namespace=namespace,
            page_name=state["page_name"],
            body=state["body"],
            needs_review=needs_review,
        )
        pages.write_index(deps.conn, wiki_dir=deps.wiki_dir, namespace=namespace)
        log.info("wrote %s%s", path, " (needs review)" if needs_review else "")
        return {"written_path": str(path)}

    def after_item_review(state: ItemState) -> str:
        if not state.get("item_issues"):
            return "resolve"
        if state.get("item_attempts", 0) >= settings.review_retries:
            log.warning(
                "item %r still failing review after %d refine attempt(s); flagging",
                state["item"].name,
                state.get("item_attempts", 0),
            )
            return "give_up"
        return "refine"

    async def flag_item(state: ItemState) -> ItemState:
        """Item review never passed: carry on, but mark the page for a human."""
        return {"needs_review": True}

    def after_resolve(state: ItemState) -> str:
        return "create" if state.get("is_new_page") else "merge"

    def after_page_review(state: ItemState) -> str:
        if not state.get("page_issues"):
            return "persist"
        if state.get("page_attempts", 0) >= settings.review_retries:
            log.warning(
                "page %r still failing review after %d refine attempt(s); flagging",
                state["page_name"],
                state.get("page_attempts", 0),
            )
            return "give_up"
        return "refine"

    async def flag_page(state: ItemState) -> ItemState:
        return {"needs_review": True}

    graph = StateGraph(ItemState)
    graph.add_node("review_item", review_item)
    graph.add_node("refine_item", refine_item)
    graph.add_node("flag_item", flag_item)
    graph.add_node("resolve_entity", resolve_entity)
    graph.add_node("create_page", create_page)
    graph.add_node("merge_page", merge_page)
    graph.add_node("review_page", review_page)
    graph.add_node("refine_page", refine_page)
    graph.add_node("flag_page", flag_page)
    graph.add_node("persist", persist)

    graph.add_edge(START, "review_item")
    graph.add_conditional_edges(
        "review_item",
        after_item_review,
        {"resolve": "resolve_entity", "refine": "refine_item", "give_up": "flag_item"},
    )
    graph.add_edge("refine_item", "review_item")
    graph.add_edge("flag_item", "resolve_entity")

    graph.add_conditional_edges(
        "resolve_entity", after_resolve, {"create": "create_page", "merge": "merge_page"}
    )
    graph.add_edge("create_page", "review_page")
    graph.add_edge("merge_page", "review_page")

    graph.add_conditional_edges(
        "review_page",
        after_page_review,
        {"persist": "persist", "refine": "refine_page", "give_up": "flag_page"},
    )
    graph.add_edge("refine_page", "review_page")
    graph.add_edge("flag_page", "persist")
    graph.add_edge("persist", END)

    return graph.compile()
