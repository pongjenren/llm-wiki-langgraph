"""Cross-page linking: candidate gathering, the LLM judge, and deterministic splice."""

from __future__ import annotations

import re

from llm_wiki import links, pages
from llm_wiki.db import repo
from llm_wiki.llm.schemas import LinkDecision, ProposedLink

NS = "demo"

# Candidate lines the prompt emits: "- [page_id=3] Beta | aliases: ...".
_CANDIDATE = re.compile(r"\[page_id=(\d+)\]\s*(.+?)\s*\|")


class ApproveAllLinkClient:
    """Links every offered candidate, anchoring on its page name.

    Stands in for the LLM judge so the deterministic guards (first mention,
    protected regions, longest anchor, self-link) can be tested without a model.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def _links(self, prompt: str) -> list[ProposedLink]:
        return [
            ProposedLink(target_page_id=int(pid), anchor_text=name)
            for pid, name in _CANDIDATE.findall(prompt)
        ]

    async def run_json(self, prompt: str, schema, *, label: str = "step") -> LinkDecision:
        self.prompts.append(prompt)
        return LinkDecision(links=self._links(prompt))

    async def run_text(self, prompt: str, *, label: str = "step") -> str:
        raise AssertionError("linking must not call run_text")


class FixedDecisionClient(ApproveAllLinkClient):
    """Returns a decision the test dictates, ignoring the candidates."""

    def __init__(self, decision: LinkDecision) -> None:
        super().__init__()
        self._decision = decision

    async def run_json(self, prompt: str, schema, *, label: str = "step") -> LinkDecision:
        self.prompts.append(prompt)
        return self._decision


def _page(conn, settings, name: str, body: str, *, type_: str = "concept", aliases=()) -> int:
    """Create a page: DB row, canonical + extra aliases, and a body file."""
    page_id = repo.insert_page(conn, page_name=name, namespace=NS, type_=type_)
    repo.upsert_alias(conn, namespace=NS, name=name, page_id=page_id, type_="canonical")
    for alias in aliases:
        repo.upsert_alias(conn, namespace=NS, name=alias, page_id=page_id, type_="manual")
    pages.write_page(conn, wiki_dir=settings.wiki_dir, namespace=NS, page_name=name, body=body)
    return page_id


def _body(settings, name: str) -> str:
    return pages.read_page(settings.wiki_dir, NS, name)


async def _all(conn, settings, *, client=None, dry_run: bool = False, source_siblings=None):
    ids = [row["page_id"] for row in repo.list_pages(conn, NS)]
    return await links.link_pages(
        conn,
        wiki_dir=settings.wiki_dir,
        namespace=NS,
        page_ids=ids,
        client=client or ApproveAllLinkClient(),
        source_siblings=source_siblings,
        dry_run=dry_run,
    )


async def test_links_first_mention_only(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nA model.\n")
    _page(conn, settings, "attention mechanisms", "# attention mechanisms\n\nA concept.\n")
    src = _page(
        conn,
        settings,
        "Encoder",
        "# Encoder\n\nAn Encoder feeds a Transformer. A Transformer uses attention mechanisms.\n",
    )

    await _all(conn, settings)
    body = _body(settings, "Encoder")

    assert body.count("[Transformer](Transformer.md)") == 1
    # The second "Transformer" stays plain text.
    assert body.count("Transformer") == 3  # heading + link + plain
    assert "[attention mechanisms](attention_mechanisms.md)" in body

    out = {row["page_name"] for row in repo.list_outgoing_links(conn, src)}
    assert out == {"Transformer", "attention mechanisms"}


async def test_longest_alias_wins(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    long_id = _page(
        conn, settings, "Transformer (machine learning model)", "# T\n\nx\n"
    )
    src = _page(
        conn,
        settings,
        "Survey",
        "# Survey\n\nThe Transformer (machine learning model) changed everything.\n",
    )

    await _all(conn, settings)
    body = _body(settings, "Survey")

    # The full name is linked as one anchor; no nested short "Transformer" link.
    assert "[Transformer (machine learning model)](Transformer_%28machine_learning_model%29.md)" in body
    out = {row["dst_page_id"] for row in repo.list_outgoing_links(conn, src)}
    assert out == {long_id}


async def test_never_links_to_itself(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nThe Transformer is a Transformer.\n")

    await _all(conn, settings)
    body = _body(settings, "Transformer")

    assert "](Transformer.md)" not in body
    assert repo.list_outgoing_links(conn, repo.list_pages(conn, NS)[0]["page_id"]) == []


async def test_skips_code_headings_and_existing_links(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    src = _page(
        conn,
        settings,
        "Notes",
        "# Notes\n\n"
        "## Transformer heading\n\n"
        "Inline `Transformer` code and a [Transformer](https://x.test) link.\n\n"
        "```\nTransformer in a fence\n```\n\n"
        "But a plain Transformer here.\n",
    )

    await _all(conn, settings)
    body = _body(settings, "Notes")

    # Only the last, plain occurrence is linked.
    assert body.count("[Transformer](Transformer.md)") == 1
    assert "## Transformer heading" in body  # heading untouched
    assert "`Transformer`" in body  # inline code untouched
    assert "[Transformer](https://x.test)" in body  # external link untouched
    assert "Transformer in a fence" in body  # fenced code untouched
    assert len(repo.list_outgoing_links(conn, src)) == 1


async def test_relinking_is_idempotent(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    _page(conn, settings, "Decoder", "# Decoder\n\nA Decoder mirrors a Transformer.\n")

    first = await _all(conn, settings)
    body_after_first = _body(settings, "Decoder")

    second = await _all(conn, settings)
    body_after_second = _body(settings, "Decoder")

    assert body_after_first == body_after_second
    # First pass links Decoder; second pass finds it already correct.
    assert any(r.page_name == "Decoder" and r.changed for r in first)
    assert all(not r.changed for r in second)


async def test_incremental_scope_leaves_untouched_pages_alone(conn, settings):
    a = _page(conn, settings, "Alpha", "# Alpha\n\nAlpha references Beta.\n")
    b = _page(conn, settings, "Beta", "# Beta\n\nBeta references Alpha.\n")

    # Link only Alpha: Beta must not be rewritten even though it mentions Alpha.
    await links.link_pages(
        conn, wiki_dir=settings.wiki_dir, namespace=NS, page_ids=[a], client=ApproveAllLinkClient()
    )
    assert "[Beta](Beta.md)" in _body(settings, "Alpha")
    assert "](Alpha.md)" not in _body(settings, "Beta")
    assert repo.list_outgoing_links(conn, b) == []

    # A full reconcile backfills the Beta -> Alpha link.
    await _all(conn, settings)
    assert "[Alpha](Alpha.md)" in _body(settings, "Beta")
    assert {row["page_name"] for row in repo.list_backlinks(conn, a)} == {"Beta"}


async def test_matches_are_case_and_alias_insensitive(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n", aliases=["transformer model"])
    src = _page(
        conn,
        settings,
        "Ref",
        "# Ref\n\nA TRANSFORMER and a transformer model both count.\n",
    )

    await _all(conn, settings)
    body = _body(settings, "Ref")

    # Case-insensitive match keeps the original surface text as the anchor.
    assert "[TRANSFORMER](Transformer.md)" in body
    # Only the first mention of the target links (alias is the same page).
    assert len(repo.list_outgoing_links(conn, src)) == 1


async def test_llm_can_reject_a_candidate(conn, settings):
    """An alias match is only a candidate now; the judge may decline to link it."""
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    src = _page(conn, settings, "Doc", "# Doc\n\nA Transformer appears.\n")

    # The judge returns no links even though "Transformer" matches an alias.
    await _all(conn, settings, client=FixedDecisionClient(LinkDecision(links=[])))

    assert "](Transformer.md)" not in _body(settings, "Doc")
    assert repo.list_outgoing_links(conn, src) == []


async def test_unreal_target_is_dropped_without_llm(conn, settings):
    """A page_id the judge invents is verified away, not written."""
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    src = _page(conn, settings, "Doc", "# Doc\n\nA Transformer appears.\n")

    bogus = LinkDecision(links=[ProposedLink(target_page_id=999999, anchor_text="Transformer")])
    await _all(conn, settings, client=FixedDecisionClient(bogus))

    assert "](Transformer.md)" not in _body(settings, "Doc")
    assert repo.list_outgoing_links(conn, src) == []


async def test_source_siblings_are_offered_as_candidates(conn, settings):
    """Pages sharing a source are handed to the judge flagged as same-source."""
    t = _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    src = _page(conn, settings, "Encoder", "# Encoder\n\nAn Encoder feeds a Transformer.\n")

    client = ApproveAllLinkClient()
    await links.link_pages(
        conn,
        wiki_dir=settings.wiki_dir,
        namespace=NS,
        page_ids=[src],
        client=client,
        source_siblings={src: {t}},
    )

    prompt = next(p for p in client.prompts if "Encoder" in p or "Transformer" in p)
    assert f"[page_id={t}] Transformer" in prompt
    assert "same source document: yes" in prompt
    assert "[Transformer](Transformer.md)" in _body(settings, "Encoder")


def test_sentence_span_bounds_and_abbreviations():
    # A terminal-period abbreviation ("Inc.") must not cut the sentence short.
    body = "# H\n\nFoo Inc. makes the Widget here. Another one.\n"
    i = body.index("Widget")
    span = links._sentence_span(body, i, i + len("Widget") - 1)
    assert span == "Foo Inc. makes the Widget here."


def test_sentence_span_stays_within_paragraph():
    # The snippet must not reach back into the previous paragraph.
    body = "First para ends here.\n\nSecond has the Widget inside.\n"
    i = body.index("Widget")
    span = links._sentence_span(body, i, i + len("Widget") - 1)
    assert span == "Second has the Widget inside."


async def test_context_sentence_is_stored(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    src = _page(
        conn,
        settings,
        "Encoder",
        "# Encoder\n\nAn Encoder feeds a Transformer. A Transformer uses attention.\n",
    )

    await _all(conn, settings)

    row = next(
        r for r in repo.list_outgoing_links(conn, src) if r["page_name"] == "Transformer"
    )
    # anchor stays the linked name; context is the whole first-mention sentence.
    assert row["anchor_text"] == "Transformer"
    assert row["context_sentence"] == "An Encoder feeds a Transformer."


async def test_dry_run_reports_without_writing(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    src = _page(conn, settings, "Doc", "# Doc\n\nA Transformer appears.\n")

    results = await _all(conn, settings, dry_run=True)

    assert any(r.changed for r in results)
    assert "](Transformer.md)" not in _body(settings, "Doc")  # file untouched
    assert repo.list_outgoing_links(conn, src) == []  # table untouched
