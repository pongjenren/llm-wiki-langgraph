"""Prompt builders for each LLM step.

Citation convention: while a document is being processed we do not yet know
which reference number it will get on a given page, so the model always writes
citations as the literal marker [CURRENT]. The pipeline substitutes the real
number once the source is linked to the page.
"""

from __future__ import annotations

from llm_wiki.llm.schemas import ExtractedItem

CITATION_RULE = (
    "Cite the document you were given as the literal marker [CURRENT]. "
    "Never invent a numeric citation such as [1]; only [CURRENT] and citation "
    "numbers already present in existing page text are allowed."
)


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
- {CITATION_RULE}
- Prefer the most canonical form of each name; put spelling variants and
  abbreviations used in the document into `aliases`.

Filename: {filename}

Document:
---
{text}
---
""".strip()


def review_item(item: ExtractedItem) -> str:
    return f"""
Review one extracted knowledge-base item for quality.

Fail it if any of these hold:
- The name is vague, generic, or is not a nameable entity or concept.
- The `type` is wrong for what the item actually is.
- The description is empty, circular, or says nothing substantive.
- The description contains claims the item's own text does not support.
- The description is missing the [CURRENT] citation marker.
- The aliases include names that refer to something different.

Otherwise pass it.

Item:
---
name: {item.name}
type: {item.type}
aliases: {", ".join(item.aliases) or "(none)"}

{item.description}
---
""".strip()


def refine_item(item: ExtractedItem, issues: list[str]) -> str:
    bullets = "\n".join(f"- {issue}" for issue in issues) or "- (unspecified)"
    return f"""
Revise this knowledge-base item to fix the problems found in review.

Problems:
{bullets}

Rules:
- Fix only what the problems call for; keep everything else as it is.
- Do not add facts beyond what the original description supports.
- {CITATION_RULE}

Item:
---
name: {item.name}
type: {item.type}
aliases: {", ".join(item.aliases) or "(none)"}

{item.description}
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
- Cite the source as [{reference_number}] on the claims it supports.

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
- Cite the new material as [{reference_number}].
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
