"""Stage 2: one entity/concept item is folded into a new or existing wiki page.

Entity resolution already happened in stage 1, which hands each run of this
graph a page nothing else will touch. That is what makes it safe to run several
of these concurrently: the reference numbering and the page file belong to one
page, and one page belongs to one run.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from llm_wiki import embedding, links, pages
from llm_wiki.db import repo
from llm_wiki.db.connection import transaction
from llm_wiki.graph.state import Deps, ItemState
from llm_wiki.llm import prompts
from llm_wiki.llm.schemas import ExtractedItem, Review

log = logging.getLogger(__name__)


def build_item_graph(deps: Deps):
    """Compile the stage-2 graph."""
    settings = deps.settings

    def _record_aliases(
        item: ExtractedItem, namespace: str, page_id: int, *, canonical: bool, how: str = "alias"
    ) -> None:
        # The name we searched under resolved to this page; recording it under the
        # method that matched makes the next lookup an exact alias hit.
        name_type = "canonical" if canonical else how
        repo.upsert_alias(
            deps.conn, namespace=namespace, name=item.name, page_id=page_id, type_=name_type
        )
        for alias in item.aliases:
            repo.upsert_alias(
                deps.conn, namespace=namespace, name=alias, page_id=page_id, type_="alias"
            )

    def create_page(state: ItemState) -> ItemState:
        """Insert the page and its first reference, then write its body."""
        item = state["item"]
        namespace = state["namespace"]
        page_name = state["page_name"]

        # Stage 1 embedded this name to resolve it; reuse that vector rather
        # than paying for the same call again.
        vector = state.get("name_embedding") or embedding.embed(
            embedding.identity_text(page_name), model_name=settings.embedding_model
        )
        with transaction(deps.conn):
            page_id = repo.insert_page(
                deps.conn,
                page_name=page_name,
                namespace=namespace,
                type_=item.type,
                embedding=vector,
            )
            _record_aliases(item, namespace, page_id, canonical=True)
            reference_number = repo.link_source(
                deps.conn, wiki_id=page_id, source_id=state["source_id"], namespace=namespace
            )

        body = deps.client.run_text(
            prompts.create_page(item, reference_number), label="create-page"
        )
        return {
            "page_id": page_id,
            "reference_number": reference_number,
            "existing_body": "",
            "body": pages.strip_references(body),
        }

    def merge_page(state: ItemState) -> ItemState:
        """Link the new source to an existing page and integrate the material.

        A single-shot draft: validating the merge and correcting any loss is the
        job of review_merge_page and its refine loop, which see the same inputs.
        """
        item = state["item"]
        namespace = state["namespace"]
        page_id = state["page_id"]

        with transaction(deps.conn):
            _record_aliases(
                item, namespace, page_id, canonical=False, how=state.get("matched_how", "alias")
            )
            reference_number = repo.link_source(
                deps.conn, wiki_id=page_id, source_id=state["source_id"], namespace=namespace
            )

        existing_body = pages.read_page(deps.wiki_dir, namespace, state["page_name"])
        merged = pages.strip_references(
            deps.client.run_text(
                prompts.merge_page(item, existing_body, reference_number),
                label=f"merge-page#{state.get('page_attempts', 0)}",
            )
        )
        return {
            "reference_number": reference_number,
            "existing_body": existing_body,
            "body": merged,
        }

    def _reference_coverage_issues(state: ItemState) -> list[str]:
        """Every source the page cites must appear as a marker in the body."""
        expected = {
            row["reference_order"] for row in repo.list_references(deps.conn, state["page_id"])
        }
        missing = expected - pages.citations(state["body"])
        if not missing:
            return []
        numbers = ", ".join(f"[{n}]" for n in sorted(missing))
        return [f"These references have no citation marker in the page: {numbers}."]

    def review_create_page(state: ItemState) -> ItemState:
        item = state["item"]
        # The one source is cited as [reference_number]; it must land in the body.
        issues: list[str] = []
        if state["reference_number"] not in pages.citations(state["body"]):
            issues.append(
                f"The source's citation [{state['reference_number']}] does not "
                "appear anywhere in the page."
            )
        review = deps.client.run_json(
            prompts.review_create_page(
                state["page_name"], state["body"], item.description, state["reference_number"]
            ),
            Review,
            label="create-review",
        )
        if review.verdict != "pass":
            issues.extend(review.issues)
        if issues:
            log.info("create review failed for %r: %s", state["page_name"], issues)
        return {"page_issues": issues}

    def review_merge_page(state: ItemState) -> ItemState:
        item = state["item"]
        existing_body = state["existing_body"]
        body = state["body"]

        # Programmatic guards against a full rewrite quietly losing content.
        issues = pages.validate_merge(existing_body, body)
        issues.extend(_reference_coverage_issues(state))
        if links.local_link_count(body) < links.local_link_count(existing_body):
            issues.append(
                "The merged page has fewer cross-page links ([text](page.md)) than "
                "the existing page; restore the dropped links."
            )

        review = deps.client.run_json(
            prompts.review_merge_page(
                state["page_name"],
                existing_body,
                item.description,
                body,
                state["reference_number"],
            ),
            Review,
            label="merge-review",
        )
        if review.verdict != "pass":
            issues.extend(review.issues)
        if issues:
            log.info("merge review failed for %r: %s", state["page_name"], issues)
        return {"page_issues": issues}

    def refine_create_page(state: ItemState) -> ItemState:
        body = deps.client.run_text(
            prompts.refine_create_page(
                state["page_name"],
                state["body"],
                state["item"].description,
                state["reference_number"],
                state["page_issues"],
            ),
            label="create-refine",
        )
        return {
            "body": pages.strip_references(body),
            "page_attempts": state.get("page_attempts", 0) + 1,
        }

    def refine_merge_page(state: ItemState) -> ItemState:
        # A refine is a re-merge: redo it from the existing page and new material
        # with the named problems, so structural losses can be reconstructed.
        attempts = state.get("page_attempts", 0) + 1
        body = deps.client.run_text(
            prompts.refine_merge_page(
                state["item"],
                state["existing_body"],
                state["reference_number"],
                state["page_issues"],
            ),
            label=f"merge-page#{attempts}",
        )
        return {"body": pages.strip_references(body), "page_attempts": attempts}

    def persist(state: ItemState) -> ItemState:
        """Write the page.

        The namespace index is not rewritten here: it covers every page, so it
        would be rewritten once per item for no reason, and concurrent items
        would clobber each other's copy. The pipeline writes it once, after all
        of a document's items are in.
        """
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
        log.info("wrote %s%s", path, " (needs review)" if needs_review else "")
        return {"written_path": str(path)}

    def after_resolve(state: ItemState) -> str:
        """Branch on the decision stage 1 already made for this item."""
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

    def flag_page(state: ItemState) -> ItemState:
        return {"needs_review": True}

    graph = StateGraph(ItemState)
    graph.add_node("create_page", create_page)
    graph.add_node("merge_page", merge_page)
    graph.add_node("review_create_page", review_create_page)
    graph.add_node("refine_create_page", refine_create_page)
    graph.add_node("review_merge_page", review_merge_page)
    graph.add_node("refine_merge_page", refine_merge_page)
    graph.add_node("flag_page", flag_page)
    graph.add_node("persist", persist)

    graph.add_conditional_edges(
        START, after_resolve, {"create": "create_page", "merge": "merge_page"}
    )
    graph.add_edge("create_page", "review_create_page")
    graph.add_edge("merge_page", "review_merge_page")

    graph.add_conditional_edges(
        "review_create_page",
        after_page_review,
        {"persist": "persist", "refine": "refine_create_page", "give_up": "flag_page"},
    )
    graph.add_conditional_edges(
        "review_merge_page",
        after_page_review,
        {"persist": "persist", "refine": "refine_merge_page", "give_up": "flag_page"},
    )
    graph.add_edge("refine_create_page", "review_create_page")
    graph.add_edge("refine_merge_page", "review_merge_page")
    graph.add_edge("flag_page", "persist")
    graph.add_edge("persist", END)

    return graph.compile()
