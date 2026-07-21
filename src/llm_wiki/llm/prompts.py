"""Prompt builders for each LLM step.

Citation convention: extraction produces plain facts with no reference markers.
Every fact in a single item's description comes from the one document it was
extracted from, so the reference number is uniform across the description and is
not known until the source is linked to a page. The create/merge step is told
that number and attaches it to the claims it writes.
"""

from __future__ import annotations

from llm_wiki.llm.schemas import ExtractedItem


def summarize_decision(filename: str, text: str) -> str:
    return f"""
You are triaging a document before entity extraction.

Decide whether it should first be rewritten into descriptive prose. Rewrite when
the raw form is not self-describing — for example a spreadsheet, a table dump, a
log, or structured records — where facts are implied by structure rather than
stated in sentences. Documents that are already prose do not need rewriting.

Filename: {filename}

Document:
---
{text}
---
""".strip()


def summarize(filename: str, text: str) -> str:
    return f"""
Rewrite the following document as descriptive prose suitable for knowledge
extraction.

Rules:
- State only facts present in the document. Do not infer, estimate, or add
  outside knowledge.
- Make implicit structure explicit: name the entities the rows and columns refer
  to, and describe the quantities and relationships they encode.
- Preserve specific values, names, dates and units exactly.
- Do not editorialize or draw conclusions.

Filename: {filename}

Document:
---
{text}
---

Reply with the rewritten prose only.
""".strip()


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


def review_page(page_name: str, page_markdown: str) -> str:
    return f"""
Review a wiki page for publication.

Fail it if any of these hold:
- It does not open with a clear definition of "{page_name}".
- It contradicts itself, or repeats the same claim in several places.
- It contains claims with no citation marker.
- It is malformed markdown, or has more than one level-1 heading.
- It contains placeholder text or empty sections.

Otherwise pass it.

Page:
---
{page_markdown}
---
""".strip()


def refine_page(page_markdown: str, issues: list[str]) -> str:
    bullets = "\n".join(f"- {issue}" for issue in issues) or "- (unspecified)"
    return f"""
Revise this wiki page to fix the problems found in review.

Problems:
{bullets}

Rules:
- Fix only what the problems call for.
- Preserve every existing citation marker on the claims that carry them.
- Do not add facts or outside knowledge, and do not write a References section.

Page:
---
{page_markdown}
---

Reply with the corrected page markdown only.
""".strip()
