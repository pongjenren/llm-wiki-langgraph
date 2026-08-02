"""Graph state and the runtime dependencies the nodes share."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from llm_wiki.config import Settings
from llm_wiki.db.connection import Connection
from llm_wiki.llm.client import LLMClient
from llm_wiki.llm.schemas import ExtractedItem


@dataclass
class Deps:
    """Everything the nodes need that is not part of graph state.

    Passed to the graph builders rather than through state: connections and
    clients are not serializable, and checkpointing state that holds them would
    fail.
    """

    conn: Connection
    client: LLMClient
    settings: Settings

    @property
    def wiki_dir(self) -> Path:
        return self.settings.wiki_dir


@dataclass
class PageAssignment:
    """One document item bound to the single page it will be folded into.

    Stage 1 produces one of these per *page*, not per extracted item: items that
    resolve to the same thing are merged into one assignment (see
    ``merged_names``). That is what lets stage 2 run its items concurrently --
    no two assignments share a page, so no two workers contend for a page row,
    its reference numbering, or its file on disk.

    Nothing here is written to the database yet. The assignment is a plan; the
    page row and its source link are still created inside the stage-2 graph, so
    an item that never gets a body never leaves a page row behind.
    """

    item: ExtractedItem
    is_new_page: bool
    page_name: str
    # None exactly when is_new_page -- the row does not exist yet.
    page_id: int | None = None
    matched_how: str = "alias"
    # A low-confidence resolve merges anyway, but asks for a human check.
    needs_review: bool = False
    # The item name's vector, already computed while resolving. Carried so
    # create_page can index the new page without embedding the name twice.
    name_embedding: list[float] | None = None
    # Every extracted name folded into this assignment, in the order seen. One
    # entry is the common case; more means the document named one thing twice.
    merged_names: list[str] = field(default_factory=list)


class DocState(TypedDict, total=False):
    """Stage 1: a raw file becomes a list of entity/concept items."""

    # Input
    namespace: str
    path: str

    # Loaded document
    filename: str
    text: str
    sha256: str
    timestamp: str

    # Dedup outcome
    source_id: int
    skipped: bool
    skip_reason: str

    # Processing
    items: list[ExtractedItem]

    # Extraction review loop
    extraction_attempts: int
    extraction_issues: list[str]

    # Entity resolution: one entry per distinct page, ready for stage 2.
    assignments: list[PageAssignment]


class ItemState(TypedDict, total=False):
    """Stage 2: one item is folded into a new or existing page.

    Entity resolution already happened in stage 1: the fields below down to
    ``matched_how`` arrive pre-filled from a :class:`PageAssignment` and are
    never decided here.
    """

    # Input
    namespace: str
    source_id: int
    item: ExtractedItem

    # Entity resolution (decided in stage 1)
    page_id: int
    page_name: str
    is_new_page: bool
    matched_how: str
    name_embedding: list[float] | None
    reference_number: int

    # Page content
    existing_body: str
    body: str

    # Page review loop
    page_attempts: int
    page_issues: list[str]
    needs_review: bool

    # Outcome
    written_path: str
