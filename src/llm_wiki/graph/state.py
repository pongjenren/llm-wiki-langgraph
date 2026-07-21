"""Graph state and the runtime dependencies the nodes share."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from llm_wiki.config import Settings
from llm_wiki.llm.client import LLMClient
from llm_wiki.llm.schemas import ExtractedItem


@dataclass
class Deps:
    """Everything the nodes need that is not part of graph state.

    Passed to the graph builders rather than through state: connections and
    clients are not serializable, and checkpointing state that holds them would
    fail.
    """

    conn: sqlite3.Connection
    client: LLMClient
    settings: Settings

    @property
    def wiki_dir(self) -> Path:
        return self.settings.wiki_dir


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
    summarized: bool
    working_text: str
    items: list[ExtractedItem]


class ItemState(TypedDict, total=False):
    """Stage 2: one item is folded into a new or existing page."""

    # Input
    namespace: str
    source_id: int
    item: ExtractedItem

    # Item review loop
    item_attempts: int
    item_issues: list[str]

    # Entity resolution
    page_id: int
    page_name: str
    is_new_page: bool
    matched_how: str
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
