"""Stage 1: raw file -> SHA dedup -> entity/concept list -> page assignments.

The stage ends by resolving every extracted item to the page it belongs to, so
that stage 2 receives one assignment per *page*. Resolution has to be serial --
it is the step where "does this page already exist?" is answered, and two items
naming the same thing must not both answer "no" -- but it is also the only step
that has to be, which is what lets stage 2 fold items in concurrently.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from rapidfuzz import fuzz

from llm_wiki import embedding, loaders
from llm_wiki.db import repo
from llm_wiki.db.connection import transaction
from llm_wiki.db.repo import PageCandidate
from llm_wiki.graph.state import Deps, DocState, PageAssignment
from llm_wiki.llm import prompts
from llm_wiki.llm.schemas import ExtractedItem, ExtractionResult, ResolveDecision, Review

log = logging.getLogger(__name__)

# Pending pages -- ones this document will create, which have no row and so no
# page_id yet -- are offered to the resolve judge under synthetic negative ids.
# Real page_ids are positive, so the sign alone says which list a decision
# points into, and the judge prompt needs no changes to handle both.
_PENDING_ID_BASE = -1


def _pending_id(slot: int) -> int:
    return _PENDING_ID_BASE - slot


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine *distance*, matching pgvector's `<=>` so one threshold serves both."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 if norm == 0 else 1.0 - dot / norm


def build_doc_graph(deps: Deps):
    """Compile the stage-1 graph."""
    settings = deps.settings

    def load_document(state: DocState) -> DocState:
        document = loaders.load(Path(state["path"]))
        return {
            "filename": document.path.name,
            "text": document.text,
            "sha256": document.sha256,
            "timestamp": document.timestamp,
        }

    def check_sha(state: DocState) -> DocState:
        """Skip documents already ingested into this namespace.

        A retry arrives with ``source_id`` already set: it is re-running a
        document that is *known* to have a row, so the dedup check that exists to
        stop exactly that would defeat it. See ``pipeline.retry_failed_items``.
        """
        if state.get("source_id") is not None:
            return {"skipped": False}

        existing = repo.find_source_by_sha(deps.conn, state["namespace"], state["sha256"])
        if existing is not None:
            log.info("skip %s: already ingested as source #%s", state["filename"], existing["id"])
            return {
                "skipped": True,
                "skip_reason": f"already ingested as source #{existing['id']}",
                "source_id": int(existing["id"]),
            }
        return {"skipped": False}

    def extract(state: DocState) -> DocState:
        result = deps.client.run_json(
            prompts.extract(state["filename"], state["text"]),
            ExtractionResult,
            label="extract",
        )
        log.info("extracted %d item(s) from %s", len(result.items), state["filename"])
        return {"items": result.items}

    def review_extraction(state: DocState) -> DocState:
        """Check the whole item set against the source: recall, precision, type."""
        review = deps.client.run_json(
            prompts.review_extraction(state["filename"], state["text"], state["items"]),
            Review,
            label="extraction-review",
        )
        if review.verdict == "pass":
            return {"extraction_issues": []}
        log.info("extraction review failed for %s: %s", state["filename"], review.issues)
        return {"extraction_issues": review.issues}

    def refine_extraction(state: DocState) -> DocState:
        """Rebuild the item set from the source, adding, dropping, and fixing."""
        result = deps.client.run_json(
            prompts.refine_extraction(
                state["filename"],
                state["text"],
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

    def record_source(state: DocState) -> DocState:
        # A retry already has its row -- the one whose error_msg named the items
        # being re-run -- and must keep it, so that the re-run items cite the
        # same source and land under the same reference numbers.
        if state.get("source_id") is not None:
            return {}

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

    def _alias_keys(assignment: PageAssignment) -> set[str]:
        """The names a pending page will be findable under once it is written."""
        item = assignment.item
        return {repo.normalize_name(n) for n in (item.name, *item.aliases)}

    def _pending_candidates(
        item: ExtractedItem, vector: list[float], pending: list[PageAssignment]
    ) -> list[PageCandidate]:
        """Candidate pages from this document's own not-yet-created pages.

        The same two recall signals the database is queried with, applied to the
        pending list instead: fuzzy name match and embedding proximity, at the
        same thresholds. Without this, two items naming one new thing differently
        would each be told "no such page" and produce a duplicate.
        """
        candidates: list[PageCandidate] = []
        query = repo.normalize_name(item.name)

        for slot, entry in enumerate(pending):
            # Entries bound to a real page are already reachable through the
            # database queries; offering them again would list one page twice.
            if not entry.is_new_page:
                continue
            keys = _alias_keys(entry)
            string_score = max((fuzz.token_sort_ratio(query, key) for key in keys), default=0.0)
            distance = (
                _cosine_distance(vector, entry.name_embedding)
                if entry.name_embedding is not None
                else 1.0
            )

            # Strings first, mirroring the ordering _merge_candidates relies on.
            if string_score >= settings.string_candidate_threshold:
                how, score = "string_sim", string_score
            elif distance <= settings.embedding_candidate_threshold:
                how, score = "embedding_sim", distance
            else:
                continue

            candidates.append(
                PageCandidate(
                    page_id=_pending_id(slot),
                    page_name=entry.page_name,
                    type=entry.item.type,
                    how=how,
                    score=score,
                )
            )

        return candidates[: settings.resolve_candidate_limit]

    def _merge_candidates(
        emb: list[PageCandidate], strings: list[PageCandidate], pending: list[PageCandidate]
    ) -> list[PageCandidate]:
        """Union every signal, one entry per page, capped for the LLM."""
        seen: dict[int, PageCandidate] = {}
        for candidate in (*strings, *pending, *emb):
            seen.setdefault(candidate.page_id, candidate)
        return list(seen.values())[: settings.resolve_candidate_limit]

    def _fold_in(assignment: PageAssignment, item: ExtractedItem) -> None:
        """Fold a second item for the same thing into the assignment it matched.

        Both items describe one page, so they are folded into one before stage 2
        rather than run twice against it: one create-or-merge instead of two, and
        a body written from all of the document's material at once. The discarded
        name is kept as an alias so the next document resolves under it too.
        """
        target = assignment.item
        if item.type != target.type:
            log.warning(
                "item %r (%s) folds into %r (%s); keeping %s",
                item.name,
                item.type,
                assignment.page_name,
                target.type,
                target.type,
            )

        aliases = list(target.aliases)
        seen = {repo.normalize_name(n) for n in (target.name, *target.aliases)}
        for name in (item.name, *item.aliases):
            key = repo.normalize_name(name)
            if key not in seen:
                seen.add(key)
                aliases.append(name)

        assignment.item = target.model_copy(
            update={
                "aliases": aliases,
                "description": f"{target.description}\n\n{item.description}",
            }
        )
        assignment.merged_names.append(item.name)
        log.info("item %r folds into %r (same page)", item.name, assignment.page_name)

    def _bind_existing(
        item: ExtractedItem,
        pending: list[PageAssignment],
        page_id: int,
        page_name: str,
        matched_how: str,
        *,
        needs_review: bool,
    ) -> PageAssignment:
        """Bind an item to an existing page, folding if that page is already spoken for.

        Two items of one document can resolve to the same existing page --
        neither is in ``page_aliases`` under its own name yet, so both reach it
        through the candidate signals independently. They must end up in one
        assignment: two assignments for one page is exactly the race stage 2 is
        relieved of, and the second would also read the first's half-written body.
        """
        for entry in pending:
            if entry.page_id == page_id:
                _fold_in(entry, item)
                entry.needs_review = entry.needs_review or needs_review
                return entry

        return PageAssignment(
            item=item,
            is_new_page=False,
            page_name=page_name,
            page_id=page_id,
            matched_how=matched_how,
            needs_review=needs_review,
            merged_names=[item.name],
        )

    def _resolve(
        item: ExtractedItem, namespace: str, pending: list[PageAssignment]
    ) -> PageAssignment:
        """Find the page this item belongs to, or plan a new one.

        A funnel: an exact alias hit, then -- for everything else -- candidates
        from the vector index, from string matching, and from the pages this
        document has already planned, all handed to an LLM to judge.
        """
        # (1) Exact alias, against pages that exist and pages already planned.
        # Deterministic, no model needed.
        match = repo.find_page_by_alias(deps.conn, namespace, item.name)
        if match is not None:
            log.info("%r resolves to page #%s (alias)", item.name, match.page_id)
            return _bind_existing(
                item, pending, match.page_id, match.page_name, "alias", needs_review=False
            )

        key = repo.normalize_name(item.name)
        for entry in pending:
            if key in _alias_keys(entry):
                _fold_in(entry, item)
                return entry

        # Gather candidates from every signal (recall-oriented thresholds).
        vector = embedding.embed(
            embedding.identity_text(item.name), model_name=settings.embedding_model
        )
        candidates = _merge_candidates(
            repo.find_page_candidates_by_embedding(
                deps.conn,
                namespace,
                vector,
                settings.embedding_candidate_threshold,
                settings.resolve_candidate_limit,
            ),
            repo.find_page_candidates_by_string(
                deps.conn,
                namespace,
                item.name,
                settings.string_candidate_threshold,
                settings.resolve_candidate_limit,
            ),
            _pending_candidates(item, vector, pending),
        )

        # (2) No candidate at all -- nothing to resolve against.
        if not candidates:
            log.info("new page for %r", item.name)
            return PageAssignment(
                item=item,
                is_new_page=True,
                page_name=item.name,
                name_embedding=vector,
                merged_names=[item.name],
            )

        # (3) Let the LLM judge against name, aliases and description.
        decision = deps.client.run_json(
            prompts.resolve_entity(item, candidates), ResolveDecision, label="resolve"
        )
        chosen = next((c for c in candidates if c.page_id == decision.matched_page_id), None)
        if chosen is None:
            if decision.matched_page_id is not None:
                log.warning(
                    "resolve returned page_id %s, not among candidates for %r; new page",
                    decision.matched_page_id,
                    item.name,
                )
            log.info("new page for %r (LLM: %s)", item.name, decision.reason)
            return PageAssignment(
                item=item,
                is_new_page=True,
                page_name=item.name,
                name_embedding=vector,
                merged_names=[item.name],
            )

        # A pending page is one of this document's own, not a row to merge into.
        if chosen.page_id < 0:
            entry = pending[_PENDING_ID_BASE - chosen.page_id]
            _fold_in(entry, item)
            # An uncertain fold still merges, but a human is asked to confirm it.
            entry.needs_review = entry.needs_review or decision.confidence == "low"
            return entry

        log.info(
            "%r resolves to page #%s (LLM, %s: %s)",
            item.name,
            chosen.page_id,
            decision.confidence,
            decision.reason,
        )
        return _bind_existing(
            item,
            pending,
            chosen.page_id,
            chosen.page_name,
            "llm_sim",
            # An uncertain match still merges, but a human is asked to confirm it.
            needs_review=decision.confidence == "low",
        )

    def assign_pages(state: DocState) -> DocState:
        """Bind every extracted item to the one page it will be folded into.

        Serial on purpose, and the only part of ingest that has to be: each
        resolve is answered against both the database and the pages the items
        before it have already claimed. Nothing is written here -- the page rows
        are still created in stage 2, so an item that never gets a body does not
        leave an empty page behind.
        """
        namespace = state["namespace"]
        assignments: list[PageAssignment] = []

        for item in state.get("items", []):
            assignment = _resolve(item, namespace, assignments)
            # Identity, not equality: _resolve returns an existing entry when the
            # item folded into one, and a fresh object otherwise.
            if not any(existing is assignment for existing in assignments):
                assignments.append(assignment)

        log.info(
            "%s: %d item(s) assigned to %d page(s)",
            state["filename"],
            len(state.get("items", [])),
            len(assignments),
        )
        return {"assignments": assignments}

    def after_sha_check(state: DocState) -> str:
        return "skip" if state.get("skipped") else "continue"

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
    graph.add_node("extract", extract)
    graph.add_node("review_extraction", review_extraction)
    graph.add_node("refine_extraction", refine_extraction)
    graph.add_node("record_source", record_source)
    graph.add_node("assign_pages", assign_pages)

    graph.add_edge(START, "load_document")
    graph.add_edge("load_document", "check_sha")
    graph.add_conditional_edges(
        "check_sha", after_sha_check, {"skip": END, "continue": "extract"}
    )
    graph.add_edge("extract", "review_extraction")
    graph.add_conditional_edges(
        "review_extraction",
        after_extraction_review,
        {"accept": "record_source", "refine": "refine_extraction"},
    )
    graph.add_edge("refine_extraction", "review_extraction")
    graph.add_edge("record_source", "assign_pages")
    graph.add_edge("assign_pages", END)

    return graph.compile()
