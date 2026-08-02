"""Drives the two graphs: a document is extracted, then its items are folded in."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from llm_wiki import loaders, pages
from llm_wiki.db import repo
from llm_wiki.graph import Deps, PageAssignment, build_doc_graph, build_item_graph

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
    # Extra names that were folded into this one because the document named the
    # same thing more than once. Empty in the ordinary case.
    merged_names: list[str] = field(default_factory=list)


@dataclass
class DocumentOutcome:
    path: Path
    namespace: str
    skipped: bool = False
    skip_reason: str | None = None
    source_id: int | None = None
    # None only when the file could not be read at all. Carried out of the graph
    # so a document that failed before its source row was written can still be
    # recorded under its own hash.
    sha256: str | None = None
    items: list[ItemOutcome] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and all(item.error is None for item in self.items)


def _run_item(
    item_graph, namespace: str, source_id: int, assignment: PageAssignment
) -> ItemOutcome:
    """Fold one assignment into its page. Never raises: a bad item is reported."""
    item = assignment.item
    try:
        state = item_graph.invoke(
            {
                "namespace": namespace,
                "source_id": source_id,
                "item": item,
                "page_id": assignment.page_id,
                "page_name": assignment.page_name,
                "is_new_page": assignment.is_new_page,
                "matched_how": assignment.matched_how,
                "name_embedding": assignment.name_embedding,
                "page_attempts": 0,
                "needs_review": assignment.needs_review,
            }
        )
    except Exception as exc:
        log.error("failed to process item %r: %s", item.name, exc)
        log.debug("traceback for item %r", item.name, exc_info=True)
        return ItemOutcome(
            name=item.name,
            page_name=assignment.page_name,
            is_new_page=False,
            reference_number=0,
            needs_review=True,
            merged_names=assignment.merged_names,
            error=f"{type(exc).__name__}: {exc}",
        )

    return ItemOutcome(
        name=item.name,
        page_name=state.get("page_name", assignment.page_name),
        is_new_page=bool(state.get("is_new_page")),
        reference_number=int(state.get("reference_number", 0)),
        needs_review=bool(state.get("needs_review")),
        page_id=state.get("page_id"),
        written_path=state.get("written_path"),
        merged_names=assignment.merged_names,
    )


def _run_items(
    deps: Deps, item_graph, namespace: str, source_id: int, assignments: list[PageAssignment]
) -> list[ItemOutcome]:
    """Fold every assignment into its page, concurrently where it pays.

    Safe to parallelize because stage 1 gave every assignment a page of its own:
    no two workers write the same page row, reference number, or file. Results
    are collected in assignment order so a run reports the same way however the
    workers interleave.
    """
    workers = max(1, min(deps.settings.item_workers, len(assignments)))
    if workers == 1:
        return [_run_item(item_graph, namespace, source_id, a) for a in assignments]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda a: _run_item(item_graph, namespace, source_id, a), assignments)
        )


def ingest_document(deps: Deps, namespace: str, path: Path) -> DocumentOutcome:
    """Run a single document through both stages."""
    doc_graph = build_doc_graph(deps)
    item_graph = build_item_graph(deps)
    outcome = DocumentOutcome(path=path, namespace=namespace)
    start = time.monotonic()

    try:
        try:
            doc_state = doc_graph.invoke({"namespace": namespace, "path": str(path)})
        except Exception as exc:  # a bad document must not abort the whole run
            log.error("failed to process %s: %s", path, exc)
            log.debug("traceback for %s", path, exc_info=True)
            outcome.error = f"{type(exc).__name__}: {exc}"
            # The graph raised, so its state (and the hash it computed) is gone.
            # Re-hash the bytes directly: the caller records the failure against
            # this document, and without a hash it has no identity.
            try:
                outcome.sha256 = loaders.file_sha256(path)
            except Exception:  # unreadable file -- the failure itself, most likely
                log.debug("could not hash %s", path, exc_info=True)
            return outcome

        outcome.source_id = doc_state.get("source_id")
        outcome.sha256 = doc_state.get("sha256")

        if doc_state.get("skipped"):
            outcome.skipped = True
            outcome.skip_reason = doc_state.get("skip_reason")
            return outcome

        assignments = doc_state.get("assignments", [])
        outcome.items = _run_items(
            deps, item_graph, namespace, doc_state["source_id"], assignments
        )

        # Once, after every page is in: the index lists the whole namespace, so
        # writing it per item would rewrite the same file N times over.
        if assignments:
            pages.write_index(deps.conn, wiki_dir=deps.wiki_dir, namespace=namespace)

        return outcome
    finally:
        outcome.elapsed_seconds = time.monotonic() - start


def ingest_documents(deps: Deps, targets: list[tuple[str, Path]]) -> list[DocumentOutcome]:
    return [ingest_document(deps, namespace, path) for namespace, path in targets]


# --------------------------------------------------------------------------
# Retrying the failed items of an otherwise-ingested document
# --------------------------------------------------------------------------


@dataclass
class RetryOutcome:
    """What a targeted re-run of one document's failed items achieved."""

    source_id: int
    namespace: str
    filename: str
    # The item names read out of the source's error_msg.
    requested: list[str] = field(default_factory=list)
    path: Path | None = None
    items: list[ItemOutcome] = field(default_factory=list)
    # Requested names that this run's extraction did not produce, so nothing
    # could be re-run for them. They stay recorded as failures.
    unmatched: list[str] = field(default_factory=list)
    # Names that succeeded on the original ingest but share a page with a failed
    # one, so their material is folded in again. Those pages are flagged.
    collateral: list[str] = field(default_factory=list)
    # Set when the retry could not be attempted at all (missing file, changed
    # file, extraction failure). The source's error_msg is left untouched.
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return (
            self.error is None
            and not self.unmatched
            and all(item.error is None for item in self.items)
        )


def parse_failed_names(error_msg: str | None) -> list[str]:
    """The item names an ingest recorded as failed, in the order recorded.

    ``telemetry`` writes one ``"<name>: <error>"`` line per failed item, but the
    error text itself may wrap onto further lines, so a continuation line can
    parse into a name that was never an item. That is tolerated rather than
    guarded against: the caller keeps only the names this document's extraction
    actually produces, which drops the noise.
    """
    if not error_msg:
        return []

    names: list[str] = []
    seen: set[str] = set()
    for line in error_msg.splitlines():
        name, separator, _ = line.partition(": ")
        name = name.strip()
        if not separator or not name:
            continue
        key = repo.normalize_name(name)
        if key not in seen:
            seen.add(key)
            names.append(name)
    return names


def find_source_file(raw_dir: Path, namespace: str, filename: str) -> Path:
    """Locate the raw file a source row was ingested from.

    Only the bare filename is stored, and a namespace directory may have
    sub-directories (``iter_namespace_files`` recurses), so this searches rather
    than joining. Raises if the file is gone or if the name is ambiguous.
    """
    namespace_dir = raw_dir / namespace
    if not namespace_dir.is_dir():
        raise FileNotFoundError(f"no raw directory for namespace {namespace!r}: {namespace_dir}")

    matches = [p for p in sorted(namespace_dir.rglob(filename)) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"{filename} is no longer under {namespace_dir}")
    if len(matches) > 1:
        listed = ", ".join(str(p) for p in matches)
        raise FileNotFoundError(f"{filename} is ambiguous under {namespace_dir}: {listed}")
    return matches[0]


def _select_retry_assignments(
    assignments: list[PageAssignment], requested: list[str]
) -> tuple[list[PageAssignment], list[str], list[str]]:
    """Split a fresh extraction into what needs re-running and what is missing.

    Matching is by name because that is all ``error_msg`` preserves. An
    assignment qualifies if *any* of the names folded into it was requested --
    resolution runs again here, against a database that has since gained the
    pages the first ingest created, so an item may now share a page with one
    that already succeeded.
    """
    wanted = {repo.normalize_name(name) for name in requested}
    selected: list[PageAssignment] = []
    covered: set[str] = set()
    collateral: list[str] = []

    for assignment in assignments:
        keys = {repo.normalize_name(name) for name in assignment.merged_names}
        hits = keys & wanted
        if not hits:
            continue

        selected.append(assignment)
        covered |= hits

        extra = [n for n in assignment.merged_names if repo.normalize_name(n) not in wanted]
        if extra:
            # Their material goes into the page a second time. merge_page reads
            # the current body and the review loop guards against duplication,
            # but not reliably enough to leave unflagged.
            log.warning(
                "page %r also covers %s, which did not fail; flagging for review",
                assignment.page_name,
                ", ".join(extra),
            )
            assignment.needs_review = True
            collateral.extend(extra)

    unmatched = [name for name in requested if repo.normalize_name(name) not in covered]
    return selected, unmatched, collateral


def retry_failed_items(deps: Deps, source: repo.Row) -> RetryOutcome:
    """Re-run just the items a document's error_msg records as failed.

    For the common failure shape: a document whose pages nearly all landed, with
    a handful of items lost to a transient error. The document is re-extracted
    (error_msg keeps only names, not the extracted material), the failed names
    are picked back out, and those items alone are folded into their pages under
    the document's original source row -- so citations keep their reference
    numbers and the successful pages are never touched.
    """
    outcome = RetryOutcome(
        source_id=int(source["id"]),
        namespace=source["namespace"],
        filename=source["filename"],
        requested=parse_failed_names(source["error_msg"]),
    )
    start = time.monotonic()

    try:
        if not outcome.requested:
            outcome.error = "no failed item names recorded in error_msg"
            return outcome

        if source["sha256"] is None:
            # The document never got as far as being hashed, so it has no items
            # to speak of -- this row records a document-level failure.
            outcome.error = (
                "this document failed before it was read, so it has no failed items; "
                "delete the source row and re-run ingest instead"
            )
            return outcome

        try:
            outcome.path = find_source_file(
                deps.settings.raw_dir, outcome.namespace, outcome.filename
            )
        except FileNotFoundError as exc:
            outcome.error = str(exc)
            return outcome

        # The source row's hash is the document's identity. Re-extracting edited
        # content under it would attribute new material to the old document, so
        # a changed file is refused rather than silently ingested.
        digest = loaders.file_sha256(outcome.path)
        if digest != source["sha256"]:
            outcome.error = (
                f"{outcome.path} has changed since it was ingested "
                f"({digest[:12]} != {str(source['sha256'])[:12]}); "
                "delete this source row and re-run ingest instead"
            )
            return outcome

        try:
            doc_state = build_doc_graph(deps).invoke(
                {
                    "namespace": outcome.namespace,
                    "path": str(outcome.path),
                    # Re-use the row instead of writing a new one; this is also
                    # what tells stage 1 not to skip the document as a duplicate.
                    "source_id": outcome.source_id,
                }
            )
        except Exception as exc:
            log.error("failed to re-extract %s: %s", outcome.path, exc)
            log.debug("traceback for %s", outcome.path, exc_info=True)
            outcome.error = f"{type(exc).__name__}: {exc}"
            return outcome

        assignments, outcome.unmatched, outcome.collateral = _select_retry_assignments(
            doc_state.get("assignments", []), outcome.requested
        )
        for name in outcome.unmatched:
            log.warning("%r was not extracted from %s this time; still failed", name, outcome.path)

        if not assignments:
            # Nothing ran, so there is nothing to record. Reported as a failed
            # retry rather than a set of unmatched names, which would overwrite
            # the recorded errors with "not extracted" and lose them. Names that
            # were never items -- a document-level error parsed as one -- land
            # here too.
            outcome.unmatched = []
            outcome.error = (
                "none of the recorded failures were extracted from this document "
                f"this time ({', '.join(outcome.requested)}); nothing to retry"
            )
            return outcome

        log.info(
            "retrying %d of %d item(s) for source #%s",
            len(assignments),
            len(outcome.requested),
            outcome.source_id,
        )
        outcome.items = _run_items(
            deps, build_item_graph(deps), outcome.namespace, outcome.source_id, assignments
        )
        pages.write_index(deps.conn, wiki_dir=deps.wiki_dir, namespace=outcome.namespace)
        return outcome
    finally:
        outcome.elapsed_seconds = time.monotonic() - start


def retry_sources(deps: Deps, sources: list[repo.Row]) -> list[RetryOutcome]:
    return [retry_failed_items(deps, source) for source in sources]
