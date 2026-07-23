"""Cross-page linking.

When one page's body mentions another page in the same namespace, that mention
is turned into a markdown link (``TSMC`` -> ``[TSMC](TSMC.md)``) and the
relationship is recorded in the ``wiki_links`` table.

Candidates are gathered but not applied blindly. Two lists are offered to an LLM
judge: the pages that share a source document with this one (strong references),
and the pages whose aliases appear in the body (an Aho-Corasick scan of the
``page_aliases`` dictionary). The LLM decides which mentions are genuine
cross-references; a name matching an alias is no longer enough on its own.

Whatever the LLM returns is then applied deterministically, never by the model:
each approved target must be a real candidate page (checked without the LLM), and
each approved anchor is spliced into the body by the same matcher the candidate
scan uses -- case- and whitespace-insensitive, respecting word boundaries and
protected regions (code, headings, existing links), longest anchor wins, and each
target linked only at its first mention.

Re-running is safe: existing links to sibling pages are unwrapped back to plain
text before matching, so a page is always relinked from a clean body and the
``wiki_links`` rows are rebuilt to match.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import ahocorasick

from llm_wiki import pages
from llm_wiki.db import repo
from llm_wiki.db.connection import Connection, transaction
from llm_wiki.llm import prompts
from llm_wiki.llm.client import LLMClient
from llm_wiki.llm.schemas import LinkDecision

log = logging.getLogger(__name__)

# Aliases too short or too common to link: they would pepper every page with
# noise. Real entity/concept names clear this easily.
_MIN_ALIAS_LEN = 2
_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "and", "or", "in", "on", "to", "is", "it",
     "as", "at", "by", "for", "from", "with"}
)

# Regions whose contents must never be linked, matched on the page body.
_FENCED_CODE = re.compile(r"```.*?```|~~~.*?~~~", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_MD_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_HEADING = re.compile(r"(?m)^#{1,6}[ \t].*$")

# A link whose target is a sibling page file (no path, no URL scheme): exactly
# what a previous link pass emits. These are unwrapped before relinking so the
# pass is idempotent.
_LOCAL_PAGE_LINK = re.compile(r"\[([^\]]+)\]\((?!\w+://)([^)/]+\.md)\)")

# Sentence extraction for the stored context snippet. A sentence terminator only
# counts when followed by whitespace/end, which alone skips the internal periods
# of abbreviations ("e.g.", "U.S.") and decimals ("3.14"); the abbreviation set
# then covers terminal-period cases ("Inc.", "Corp."). A paragraph (blank-line
# delimited) bounds the search so a snippet never crosses markdown structure.
_SENT_END = re.compile(r"[.!?]")
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_WHITESPACE = re.compile(r"\s+")
_ABBREV = frozenset(
    {"inc.", "corp.", "co.", "ltd.", "llc.", "dr.", "mr.", "mrs.", "ms.", "prof.",
     "st.", "jr.", "sr.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "no.", "fig."}
)


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize text for matching while tracking original positions.

    Mirrors ``repo.normalize_name`` (casefold, collapse whitespace runs to a
    single space) so the normalized page text and the normalized aliases line
    up. Returns the normalized string plus, for each normalized character, the
    index in ``text`` it came from -- letting a match in normalized space map
    back to the exact original span.
    """
    norm_chars: list[str] = []
    orig_idx: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            # Collapse runs; drop leading whitespace so no space maps to nothing.
            if not prev_space and norm_chars:
                norm_chars.append(" ")
                orig_idx.append(i)
            prev_space = True
            continue
        prev_space = False
        for folded in ch.casefold():
            norm_chars.append(folded)
            orig_idx.append(i)
    return "".join(norm_chars), orig_idx


def unwrap_local_links(body: str) -> str:
    """Turn ``[text](sibling.md)`` back into ``text``, leaving other links alone."""
    return _LOCAL_PAGE_LINK.sub(r"\1", body)


def _protected_spans(body: str) -> list[tuple[int, int]]:
    """Inclusive [start, end] spans that must not be linked into."""
    spans: list[tuple[int, int]] = []
    for pattern in (_FENCED_CODE, _INLINE_CODE, _MD_LINK, _HEADING):
        for m in pattern.finditer(body):
            spans.append((m.start(), m.end() - 1))
    return spans


def build_automaton(entries: Iterable[tuple[str, int, str]]) -> ahocorasick.Automaton | None:
    """Build a matcher over (normalized alias, page_id, page_name) triples.

    Returns None when nothing is worth matching (an empty namespace, or every
    alias filtered out), since an empty automaton cannot be searched.
    """
    automaton = ahocorasick.Automaton()
    count = 0
    for query_name, page_id, page_name in entries:
        if len(query_name) < _MIN_ALIAS_LEN or query_name in _STOPWORDS or query_name.isdigit():
            continue
        # (namespace, query_name) is unique, so no alias key collides here.
        automaton.add_word(query_name, (len(query_name), page_id, page_name))
        count += 1
    if not count:
        return None
    automaton.make_automaton()
    return automaton


@dataclass
class _Match:
    start: int  # inclusive index into the original body
    end: int  # inclusive
    page_id: int
    page_name: str


def find_matches(
    body: str, automaton: ahocorasick.Automaton | None, src_page_id: int
) -> list[_Match]:
    """Locate the mentions to link, already de-conflicted.

    A page is linked at its first mention only; where aliases overlap the
    longest wins; matches inside code, existing links or headings, and mentions
    of the page itself, are dropped.
    """
    if automaton is None:
        return []

    norm, orig_idx = _normalize_with_map(body)
    protected = _protected_spans(body)

    def in_protected(start: int, end: int) -> bool:
        return any(start <= pe and ps <= end for ps, pe in protected)

    raw: list[_Match] = []
    for norm_end, (length, page_id, page_name) in automaton.iter(norm):
        norm_start = norm_end - length + 1
        start = orig_idx[norm_start]
        end = orig_idx[norm_end]
        if page_id == src_page_id:
            continue
        if start > 0 and _is_word_char(body[start - 1]):
            continue
        if end + 1 < len(body) and _is_word_char(body[end + 1]):
            continue
        if in_protected(start, end):
            continue
        raw.append(_Match(start, end, page_id, page_name))

    # Longest match first at any given start, then greedily take non-overlapping
    # matches, one per target page.
    raw.sort(key=lambda m: (m.start, -(m.end - m.start)))
    accepted: list[_Match] = []
    linked: set[int] = set()
    cursor = -1
    for m in raw:
        if m.start <= cursor or m.page_id in linked:
            continue
        accepted.append(m)
        linked.add(m.page_id)
        cursor = m.end
    accepted.sort(key=lambda m: m.start)
    return accepted


def _link_target(page_name: str) -> str:
    """The relative link target for a sibling page.

    Parentheses are percent-encoded: a raw ``)`` in a markdown link target ends
    the link early, which both breaks rendering and defeats the unwrap on the
    next pass. The slug (with raw parens) is still the file on disk; viewers
    decode the target back to it.
    """
    slug = pages.slugify(page_name).replace("(", "%28").replace(")", "%29")
    return f"{slug}.md"


def _token_before(body: str, idx: int) -> str:
    """The alnum/dot run ending just before ``idx`` -- an abbreviation candidate."""
    j = idx
    while j > 0 and (body[j - 1].isalnum() or body[j - 1] == "."):
        j -= 1
    return body[j:idx]


def _is_sentence_break(body: str, term_end: int) -> bool:
    """Whether the terminator ending at ``term_end`` really ends a sentence.

    A terminator counts only when followed by whitespace or end of text, and when
    the token it closes is not a known abbreviation.
    """
    if term_end < len(body) and not body[term_end].isspace():
        return False
    return _token_before(body, term_end).casefold() not in _ABBREV


def _sentence_span(body: str, start: int, end: int) -> str:
    """The sentence containing ``body[start:end + 1]``, collapsed to one line.

    Bounded to the mention's paragraph so the snippet never spills across markdown
    structure, then trimmed left/right to the enclosing sentence boundaries.
    """
    block_start = 0
    for m in _BLANK_LINE.finditer(body, 0, start):
        block_start = m.end()
    tail = _BLANK_LINE.search(body, end + 1)
    block_end = tail.start() if tail else len(body)

    left = block_start
    for m in _SENT_END.finditer(body, block_start, start):
        if _is_sentence_break(body, m.end()):
            left = m.end()

    right = block_end
    for m in _SENT_END.finditer(body, end + 1, block_end):
        if _is_sentence_break(body, m.end()):
            right = m.end()
            break

    return _WHITESPACE.sub(" ", body[left:right]).strip()


def _apply(body: str, matches: Sequence[_Match]) -> tuple[str, list[tuple[int, str, str]]]:
    """Splice links into the body.

    Returns the new body and (page_id, anchor, context_sentence) links: the anchor
    is the linked surface text, the sentence is the whole clause it sits in.
    """
    out: list[str] = []
    links: list[tuple[int, str, str]] = []
    cursor = 0
    for m in matches:
        out.append(body[cursor:m.start])
        anchor = body[m.start:m.end + 1]
        out.append(f"[{anchor}]({_link_target(m.page_name)})")
        links.append((m.page_id, anchor, _sentence_span(body, m.start, m.end)))
        cursor = m.end + 1
    out.append(body[cursor:])
    return "".join(out), links


@dataclass
class LinkCandidate:
    """A page the LLM judge may link the current page to.

    ``from_source`` marks the pages written from the same document (the first
    list); ``mention`` carries the surface text an alias scan already found in the
    body (the second list), or None when only the shared-source signal put the
    page here.
    """

    page_id: int
    page_name: str
    aliases: list[str]
    from_source: bool
    mention: str | None


def _gather_candidates(
    body: str,
    automaton: ahocorasick.Automaton | None,
    src_page_id: int,
    sibling_ids: set[int],
    name_by_page: dict[int, str],
    alias_by_page: dict[int, set[int]],
) -> list[LinkCandidate]:
    """The two candidate lists, unioned into one catalogue for the LLM.

    List one is the pages sharing a source with this page; list two is every page
    an alias scan of the body turns up. A page can be on both.
    """
    mention_by_page: dict[int, str] = {}
    for m in find_matches(body, automaton, src_page_id):
        mention_by_page.setdefault(m.page_id, body[m.start : m.end + 1])

    candidate_ids = set(mention_by_page) | (sibling_ids - {src_page_id})
    candidates: list[LinkCandidate] = []
    for page_id in sorted(candidate_ids):
        page_name = name_by_page.get(page_id)
        if page_name is None:  # not a real page in this namespace; skip
            continue
        candidates.append(
            LinkCandidate(
                page_id=page_id,
                page_name=page_name,
                aliases=sorted(alias_by_page.get(page_id, set())),
                from_source=page_id in sibling_ids,
                mention=mention_by_page.get(page_id),
            )
        )
    return candidates


def _approved_matches(
    body: str,
    decision: LinkDecision,
    catalog: dict[int, LinkCandidate],
    src_page_id: int,
) -> list[_Match]:
    """Turn the LLM's picks into concrete matches, dropping anything unreal.

    Existence is checked here, not by the model: a target that is not one of the
    offered candidates is discarded. The approved (anchor -> page) pairs are then
    run through the same matcher the candidate scan uses, so every deterministic
    guard (word boundaries, protected regions, first mention, longest anchor)
    still applies to the final splice.
    """
    entries: list[tuple[str, int, str]] = []
    for link in decision.links:
        candidate = catalog.get(link.target_page_id)
        if candidate is None:
            log.warning(
                "link judge chose page_id %s, not an offered candidate; dropping",
                link.target_page_id,
            )
            continue
        anchor = repo.normalize_name(link.anchor_text)
        if not anchor:
            continue
        entries.append((anchor, candidate.page_id, candidate.page_name))

    return find_matches(body, build_automaton(entries), src_page_id)


@dataclass
class LinkResult:
    page_id: int
    page_name: str
    changed: bool
    links: list[tuple[int, str, str]] = field(default_factory=list)


async def relink_body(
    body: str,
    *,
    page_name: str,
    automaton: ahocorasick.Automaton | None,
    src_page_id: int,
    sibling_ids: set[int],
    name_by_page: dict[int, str],
    alias_by_page: dict[int, set[int]],
    client: LLMClient,
) -> tuple[str, list[tuple[int, str, str]]]:
    """Relink a single page body. ``body`` must be free of banner and references."""
    unwrapped = unwrap_local_links(body)
    candidates = _gather_candidates(
        unwrapped, automaton, src_page_id, sibling_ids, name_by_page, alias_by_page
    )
    if not candidates:
        return unwrapped, []

    decision = await client.run_json(
        prompts.link_page(page_name, unwrapped, candidates), LinkDecision, label="link"
    )
    catalog = {c.page_id: c for c in candidates}
    matches = _approved_matches(unwrapped, decision, catalog, src_page_id)
    return _apply(unwrapped, matches)


async def link_pages(
    conn: Connection,
    *,
    wiki_dir: Path,
    namespace: str,
    page_ids: Sequence[int],
    client: LLMClient,
    source_siblings: dict[int, set[int]] | None = None,
    dry_run: bool = False,
) -> list[LinkResult]:
    """Relink the given pages against every page in their namespace.

    The dictionary spans the whole namespace, so a linked page reaches any page
    it mentions -- including others created in the same batch. Pages *not* in
    ``page_ids`` are left untouched; a full-namespace reconcile is what
    ``page_ids = all pages`` is for.

    ``source_siblings`` maps each page to the other pages written from the same
    document, forming the first candidate list. It is empty for a full reconcile,
    where there is no single source to group by.
    """
    dict_rows = repo.load_link_dictionary(conn, namespace)
    automaton = build_automaton(
        [(r["query_name"], r["page_id"], r["page_name"]) for r in dict_rows]
    )
    name_by_page = {r["page_id"]: r["page_name"] for r in dict_rows}
    alias_by_page: dict[int, set[int]] = {}
    for r in dict_rows:
        alias_by_page.setdefault(r["page_id"], set()).add(r["query_name"])

    source_siblings = source_siblings or {}

    results: list[LinkResult] = []
    for page_id in page_ids:
        row = repo.get_page(conn, page_id)
        if row is None:
            continue
        page_name = row["page_name"]

        # read_page already strips the References section; drop the banner too so
        # matching sees only real content.
        body = pages.strip_review_banner(pages.read_page(wiki_dir, namespace, page_name))
        new_body, links = await relink_body(
            body,
            page_name=page_name,
            automaton=automaton,
            src_page_id=page_id,
            sibling_ids=source_siblings.get(page_id, set()),
            name_by_page=name_by_page,
            alias_by_page=alias_by_page,
            client=client,
        )
        changed = new_body != body
        results.append(LinkResult(page_id, page_name, changed, links))

        if dry_run:
            continue

        # The table is authoritative even when the body text did not move (e.g.
        # links were already correct): rewrite the rows every time.
        with transaction(conn):
            repo.replace_page_links(conn, src_page_id=page_id, namespace=namespace, links=links)
        if changed:
            pages.write_page(
                conn,
                wiki_dir=wiki_dir,
                namespace=namespace,
                page_name=page_name,
                body=new_body,
                needs_review=bool(row["needs_review"]),
            )
            log.info("relinked %s (%d link(s))", page_name, len(links))

    return results
