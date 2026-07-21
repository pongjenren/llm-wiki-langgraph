"""Markdown page files on disk.

Page bodies are the source of truth; the database holds metadata only. The
References section is machine-generated from wiki_source on every write, so the
LLM never writes one and never renumbers citations.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Sequence

from llm_wiki.db import repo

REFERENCES_HEADING = "## References"

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_REFERENCES_SECTION = re.compile(
    rf"^{re.escape(REFERENCES_HEADING)}\s*$.*", re.MULTILINE | re.DOTALL
)
_CITATION = re.compile(r"\[(\d+)\]")
_HEADING = re.compile(r"^#{2,6}\s+(.+?)\s*$", re.MULTILINE)

# The banner prepended to a page that failed review. Kept as a constant so the
# link pass can strip it before rewriting and let write_page re-add it.
REVIEW_BANNER = "> [!warning]\n> This page did not pass review and needs a human check.\n"
_REVIEW_BANNER_RE = re.compile(
    r"^> \[!warning\]\n> This page did not pass review and needs a human check\.\n+"
)


def slugify(page_name: str) -> str:
    """Turn a page name into a filesystem-safe basename."""
    cleaned = _INVALID_FILENAME_CHARS.sub("", page_name).strip().strip(".")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "untitled"


def page_path(wiki_dir: Path, namespace: str, page_name: str) -> Path:
    return wiki_dir / namespace / f"{slugify(page_name)}.md"


def strip_references(markdown: str) -> str:
    """Remove a trailing References section, if the model wrote one anyway."""
    return _REFERENCES_SECTION.sub("", markdown).rstrip() + "\n"


def strip_review_banner(markdown: str) -> str:
    """Remove a leading needs-review banner, if present."""
    return _REVIEW_BANNER_RE.sub("", markdown, count=1)


def citations(markdown: str) -> set[int]:
    return {int(m) for m in _CITATION.findall(markdown)}


def headings(markdown: str) -> set[str]:
    return {m.strip() for m in _HEADING.findall(markdown)}


def render_references(conn: sqlite3.Connection, page_id: int) -> str:
    rows = repo.list_references(conn, page_id)
    if not rows:
        return ""
    lines = [REFERENCES_HEADING, ""]
    lines += [f"{row['reference_order']}. {row['filename']}" for row in rows]
    return "\n".join(lines) + "\n"


def write_page(
    conn: sqlite3.Connection,
    *,
    wiki_dir: Path,
    namespace: str,
    page_name: str,
    body: str,
    needs_review: bool = False,
) -> Path:
    """Write a page body plus its generated References section."""
    path = page_path(wiki_dir, namespace, page_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    page_id_row = conn.execute(
        "SELECT page_id FROM wiki_pages WHERE namespace = ? AND page_name = ?",
        (namespace, page_name),
    ).fetchone()

    parts = []
    if needs_review:
        parts.append(REVIEW_BANNER)
    parts.append(strip_references(body).rstrip() + "\n")
    if page_id_row is not None:
        references = render_references(conn, page_id_row["page_id"])
        if references:
            parts.append(references)

    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def read_page(wiki_dir: Path, namespace: str, page_name: str) -> str:
    """Read an existing page body, without its References section.

    A missing file yields empty text: the database can outlive its file if a run
    is interrupted between the commit and the write, and a merge should rebuild
    the page rather than fail.
    """
    path = page_path(wiki_dir, namespace, page_name)
    if not path.exists():
        return ""
    return strip_references(path.read_text(encoding="utf-8"))


def write_index(conn: sqlite3.Connection, *, wiki_dir: Path, namespace: str) -> Path:
    """Regenerate the namespace index from the database."""
    rows = repo.list_pages(conn, namespace)
    path = wiki_dir / namespace / "index.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"# {namespace}", "", f"{len(rows)} page(s).", ""]
    for kind in ("entity", "concept"):
        group = [r for r in rows if r["type"] == kind]
        if not group:
            continue
        lines.append(f"## {kind.capitalize()}s")
        lines.append("")
        for row in group:
            flag = " ⚠️ needs review" if row["needs_review"] else ""
            lines.append(f"- [{row['page_name']}]({slugify(row['page_name'])}.md){flag}")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def validate_merge(old_body: str, new_body: str) -> list[str]:
    """Check a merged page did not lose what the old page carried.

    The merge prompt asks for a full rewrite, which is the failure mode worth
    guarding: an LLM rewriting a long page can quietly drop sections or
    citations. Returns a list of problems, empty when the merge is sound.
    """
    problems: list[str] = []

    lost_citations = citations(old_body) - citations(new_body)
    if lost_citations:
        numbers = ", ".join(f"[{n}]" for n in sorted(lost_citations))
        problems.append(f"These citation markers from the existing page are missing: {numbers}.")

    lost_headings = headings(old_body) - headings(new_body)
    if lost_headings:
        names = ", ".join(sorted(lost_headings))
        problems.append(f"These sections from the existing page were dropped: {names}.")

    return problems


def format_issues(issues: Sequence[str]) -> str:
    return "; ".join(issues)
