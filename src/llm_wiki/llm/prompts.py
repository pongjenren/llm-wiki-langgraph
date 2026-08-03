"""Prompt builders for each LLM step.

Citation convention: extraction produces plain facts with no reference markers.
Every fact in a single item's description comes from the one document it was
extracted from, so the reference number is uniform across the description and is
not known until the source is linked to a page. The create/merge step is told
that number and attaches it to the claims it writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from llm_wiki.llm.schemas import ExtractedItem

if TYPE_CHECKING:
    from llm_wiki.db.repo import PageCandidate
    from llm_wiki.links import LinkCandidate


def extract(filename: str, text: str) -> str:
    return f"""
Extract the entities and concepts this document supports a wiki page about.

An "entity" is a specific named thing: a person, organization, product, system,
place, or dataset. A "concept" is a general idea, method, or technique.

Rules:
- Only include items the document says something substantive about. Skip items
  merely mentioned in passing.
- Each description must stand on its own without the document at hand, and must
  only contain facts this document supports.
- State facts plainly. Do not add any citation or reference markers; references
  are attached later when the page is written.
- Prefer the most canonical form of each name; put spelling variants and
  abbreviations used in the document into `aliases`.

Filename: {filename}

Document:
---
{text}
---
""".strip()


def _render_items(items: list[ExtractedItem]) -> str:
    if not items:
        return "(no items extracted)"
    blocks = []
    for index, item in enumerate(items, 1):
        aliases = ", ".join(item.aliases) or "(none)"
        blocks.append(
            f"{index}. name: {item.name}\n"
            f"   type: {item.type}\n"
            f"   aliases: {aliases}\n"
            f"   description: {item.description}"
        )
    return "\n\n".join(blocks)


def resolve_entity(item: ExtractedItem, candidates: Sequence[PageCandidate]) -> str:
    aliases = ", ".join(item.aliases) or "(none)"
    listing = "\n".join(
        f"- page_id {c.page_id}: {c.page_name} (type: {c.type})" for c in candidates
    )
    return f"""
Decide whether a newly extracted {item.type} is the SAME real-world thing as one
of the existing wiki pages listed below, or something new.

New item:
- name: {item.name}
- type: {item.type}
- aliases: {aliases}
- description: {item.description}

Existing candidate pages:
{listing}

Rules:
- Return the page_id of the one candidate that denotes the SAME entity or concept.
- Merely related pages, or pages about the same broad topic, are NOT the same
  thing. A part is not its whole; a member is not its category.
- When no candidate is clearly the same thing, set matched_page_id to null so a
  new page is created. When in doubt, prefer null.
- Use confidence "low" when a match is plausible but you are not certain.
""".strip()


def review_extraction(filename: str, text: str, items: list[ExtractedItem]) -> str:
    return f"""
Review the entities and concepts extracted from a document, checking the list
against the document itself.

Fail the extraction if any of these hold:
- An entity or concept the document says something substantive about is missing
  from the list.
- An item is included that the document only mentions in passing, or that is not
  a nameable entity or concept.
- An item's `type` is wrong (entity vs concept).
- A description contains claims the document does not support.
- A description leaves out substantive facts the document states about that item.
- An item's aliases include a name that refers to a different entity.
- Two items are really the same thing, or the same item appears twice.

Otherwise pass it.

Filename: {filename}

Document:
---
{text}
---

Extracted items:
---
{_render_items(items)}
---
""".strip()


def refine_extraction(filename: str, text: str, items: list[ExtractedItem], issues: list[str]) -> str:
    bullets = "\n".join(f"- {issue}" for issue in issues) or "- (unspecified)"
    return f"""
Revise the list of entities and concepts extracted from a document to fix the
problems found in review. Work from the document itself.

Problems:
{bullets}

Rules:
- Add any entity or concept the document says something substantive about that is
  missing from the list.
- Remove items that are only mentioned in passing, are not nameable entities or
  concepts, or duplicate another item.
- Fix wrong `type` values and descriptions that claim more than the document
  supports.
- Extend each description with substantive facts the document states about the
  item that it currently leaves out.
- Remove any alias that refers to a different entity than the item.
- Each description must stand on its own without the document at hand, and must
  only contain facts this document supports.
- State facts plainly, with no citation or reference markers.
- Prefer the most canonical form of each name; put spelling variants and
  abbreviations used in the document into `aliases`.

Return the complete corrected list of items, not only the changes.

Filename: {filename}

Document:
---
{text}
---

Current items:
---
{_render_items(items)}
---
""".strip()


def create_page(item: ExtractedItem, reference_number: int) -> str:
    return f"""
Write a new wiki page for this {item.type}.

Format:
- Start with `# {item.name}` as the only level-1 heading.
- Open with a one-paragraph definition, then use `##` sections as the material
  warrants. Do not pad with empty sections.
- Do not write a References section; it is generated separately.

Rules:
- Use only facts from the material below. Do not add outside knowledge.
- All of the material below comes from a single source, cited as
  [{reference_number}]. Add [{reference_number}] to every claim it supports.

Material:
---
{item.description}
---

Reply with the page markdown only.
""".strip()


def merge_page(item: ExtractedItem, existing_page: str, reference_number: int) -> str:
    return f"""
Integrate new material into an existing wiki page.

Rules:
- Preserve every existing citation marker such as [1] or [2] on the claims that
  carry them. Never renumber them.
- All of the new material comes from a single source; cite it as
  [{reference_number}] on every claim it supports.
- Keep all existing sections. You may add sections, extend prose, and reorder
  for coherence, but do not drop existing content.
- Preserve every cross-page link such as [Attention](Attention.md): keep the
  link markup intact, never flatten it back to plain text.
- Where the new material conflicts with existing content, state both and
  attribute each to its citation rather than silently overwriting.
- Do not add outside knowledge, and do not write a References section.

Existing page:
---
{existing_page}
---

New material to integrate:
---
{item.description}
---

Reply with the complete merged page markdown only.
""".strip()


def review_create_page(
    page_name: str, page_markdown: str, description: str, reference_number: int
) -> str:
    return f"""
Review a newly written wiki page for "{page_name}" before publication.

You are given the source material the page was written from and the page itself.
The page must be well-formed AND cover the source material faithfully, without
losing information it provides.

Fail it if any of these hold:
- It does not open with a clear definition of "{page_name}".
- It omits substantive facts that the source material provides.
- It contradicts itself, or repeats the same claim in several places.
- It contains claims with no citation marker, or fails to cite the material as
  [{reference_number}] on the claims it supports.
- It is malformed markdown, or has more than one level-1 heading.
- It contains placeholder text or empty sections.

Otherwise pass it.

Source material (cited as [{reference_number}]):
---
{description}
---

Page:
---
{page_markdown}
---
""".strip()


def review_merge_page(
    page_name: str,
    existing_page: str,
    description: str,
    merged_page: str,
    reference_number: int,
) -> str:
    return f"""
Review a merged wiki page for "{page_name}" before publication.

The page was produced by folding new source material into an existing page. You
are given the existing (old) page, the new material, and the merged result. The
merge must keep everything the old page carried and fully integrate the new
material.

Fail it if any of these hold:
- It drops facts, sections, citation markers, or cross-page links
  ([text](page.md)) that the existing page carried.
- It omits substantive facts from the new material, or fails to cite that
  material as [{reference_number}] on the claims it supports.
- Where the old and new material conflict, it silently overwrote one instead of
  stating both and attributing each to its citation.
- It does not open with a clear definition of "{page_name}".
- It contradicts itself or repeats claims, is malformed markdown, has more than
  one level-1 heading, or contains placeholder text or empty sections.

Otherwise pass it.

Existing page:
---
{existing_page}
---

New material (cited as [{reference_number}]):
---
{description}
---

Merged page:
---
{merged_page}
---
""".strip()


def _render_link_candidates(candidates: Sequence[LinkCandidate]) -> str:
    lines = []
    for c in candidates:
        aliases = ", ".join(c.aliases) or "(none)"
        origin = "yes" if c.from_source else "no"
        seen = f' seen in text as: "{c.mention}"' if c.mention else ""
        lines.append(
            f"- [page_id={c.page_id}] {c.page_name} | aliases: {aliases} | "
            f"same source document: {origin}{seen}"
        )
        if c.mention_sentence:
            lines.append(f"    in context: {c.mention_sentence}")
    return "\n".join(lines)


def link_page(page_name: str, body: str, candidates: Sequence[LinkCandidate]) -> str:
    return f"""
Decide which mentions in a wiki page body should become cross-links to other
pages in the same knowledge base.

You are working on the page "{page_name}". Only the candidate pages listed below
exist; never link to anything else. Pages marked "same source document: yes" were
written from the same document as this page, so a mention of them here is very
likely a genuine reference.

Candidate target pages:
{_render_link_candidates(candidates)}

Page body:
---
{body}
---

Rules:
- Link a phrase only when it genuinely refers to that exact candidate's
  entity/concept. A merely related or same-topic page is NOT a link.
- A name that only shows up inside a longer name is not a mention of that page:
  the "Apple" in "Apple TV" refers to Apple TV, not to Apple. Link such a page
  only if it is also named on its own somewhere in the body.
- For each link, copy `anchor_text` verbatim from the body (same words, same
  casing). Do not invent or paraphrase the anchor.
- `target_page_id` must be one of the page_ids listed above.
- Do not link the page to itself. Link each target at most once, at its first
  genuine mention.
- If nothing should be linked, return an empty list.
""".strip()


def refine_create_page(
    page_name: str,
    page_markdown: str,
    description: str,
    reference_number: int,
    issues: list[str],
) -> str:
    bullets = "\n".join(f"- {issue}" for issue in issues) or "- (unspecified)"
    return f"""
Revise this wiki page for "{page_name}" to fix the problems found in review.

Problems:
{bullets}

The source material is provided below; use it to restore anything the page is
missing.

Rules:
- Fix every problem above.
- Use only facts from the source material; add no outside knowledge.
- Cite the material as [{reference_number}] on every claim it supports.
- Open with a clear definition, keep a single level-1 heading, and do not write
  a References section.

Source material:
---
{description}
---

Page:
---
{page_markdown}
---

Reply with the corrected page markdown only.
""".strip()


def refine_merge_page(
    item: ExtractedItem, existing_page: str, reference_number: int, issues: list[str]
) -> str:
    bullets = "\n".join(f"- {issue}" for issue in issues) or "- (unspecified)"
    return f"""
{merge_page(item, existing_page, reference_number)}

Your previous merge was rejected in review for these problems:
{bullets}

Produce the merged page again, fixing every problem above. Restore anything the
previous attempt dropped from the existing page -- sections, citation markers,
and cross-page links -- while keeping the new material fully integrated.
""".strip()
