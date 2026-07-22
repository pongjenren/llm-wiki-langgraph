"""Cross-page linking: detection, replacement, scope, and idempotency."""

from __future__ import annotations

from pathlib import Path

from llm_wiki import links, pages
from llm_wiki.db import repo

NS = "demo"


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


def _all(conn, settings, dry_run: bool = False):
    ids = [row["page_id"] for row in repo.list_pages(conn, NS)]
    return links.link_pages(
        conn, wiki_dir=settings.wiki_dir, namespace=NS, page_ids=ids, dry_run=dry_run
    )


def test_links_first_mention_only(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nA model.\n")
    _page(conn, settings, "attention mechanisms", "# attention mechanisms\n\nA concept.\n")
    src = _page(
        conn,
        settings,
        "Encoder",
        "# Encoder\n\nAn Encoder feeds a Transformer. A Transformer uses attention mechanisms.\n",
    )

    _all(conn, settings)
    body = _body(settings, "Encoder")

    assert body.count("[Transformer](Transformer.md)") == 1
    # The second "Transformer" stays plain text.
    assert body.count("Transformer") == 3  # heading + link + plain
    assert "[attention mechanisms](attention_mechanisms.md)" in body

    out = {row["page_name"] for row in repo.list_outgoing_links(conn, src)}
    assert out == {"Transformer", "attention mechanisms"}


def test_longest_alias_wins(conn, settings):
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

    _all(conn, settings)
    body = _body(settings, "Survey")

    # The full name is linked as one anchor; no nested short "Transformer" link.
    assert "[Transformer (machine learning model)](Transformer_%28machine_learning_model%29.md)" in body
    out = {row["dst_page_id"] for row in repo.list_outgoing_links(conn, src)}
    assert out == {long_id}


def test_never_links_to_itself(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nThe Transformer is a Transformer.\n")

    _all(conn, settings)
    body = _body(settings, "Transformer")

    assert "](Transformer.md)" not in body
    assert repo.list_outgoing_links(conn, repo.list_pages(conn, NS)[0]["page_id"]) == []


def test_skips_code_headings_and_existing_links(conn, settings):
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

    _all(conn, settings)
    body = _body(settings, "Notes")

    # Only the last, plain occurrence is linked.
    assert body.count("[Transformer](Transformer.md)") == 1
    assert "## Transformer heading" in body  # heading untouched
    assert "`Transformer`" in body  # inline code untouched
    assert "[Transformer](https://x.test)" in body  # external link untouched
    assert "Transformer in a fence" in body  # fenced code untouched
    assert len(repo.list_outgoing_links(conn, src)) == 1


def test_relinking_is_idempotent(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    _page(conn, settings, "Decoder", "# Decoder\n\nA Decoder mirrors a Transformer.\n")

    first = _all(conn, settings)
    body_after_first = _body(settings, "Decoder")

    second = _all(conn, settings)
    body_after_second = _body(settings, "Decoder")

    assert body_after_first == body_after_second
    # First pass links Decoder; second pass finds it already correct.
    assert any(r.page_name == "Decoder" and r.changed for r in first)
    assert all(not r.changed for r in second)


def test_incremental_scope_leaves_untouched_pages_alone(conn, settings):
    a = _page(conn, settings, "Alpha", "# Alpha\n\nAlpha references Beta.\n")
    b = _page(conn, settings, "Beta", "# Beta\n\nBeta references Alpha.\n")

    # Link only Alpha: Beta must not be rewritten even though it mentions Alpha.
    links.link_pages(conn, wiki_dir=settings.wiki_dir, namespace=NS, page_ids=[a])
    assert "[Beta](Beta.md)" in _body(settings, "Alpha")
    assert "](Alpha.md)" not in _body(settings, "Beta")
    assert repo.list_outgoing_links(conn, b) == []

    # A full reconcile backfills the Beta -> Alpha link.
    _all(conn, settings)
    assert "[Alpha](Alpha.md)" in _body(settings, "Beta")
    assert {row["page_name"] for row in repo.list_backlinks(conn, a)} == {"Beta"}


def test_matches_are_case_and_alias_insensitive(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n", aliases=["transformer model"])
    src = _page(
        conn,
        settings,
        "Ref",
        "# Ref\n\nA TRANSFORMER and a transformer model both count.\n",
    )

    _all(conn, settings)
    body = _body(settings, "Ref")

    # Case-insensitive match keeps the original surface text as the anchor.
    assert "[TRANSFORMER](Transformer.md)" in body
    # Only the first mention of the target links (alias is the same page).
    assert len(repo.list_outgoing_links(conn, src)) == 1


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


def test_context_sentence_is_stored(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    src = _page(
        conn,
        settings,
        "Encoder",
        "# Encoder\n\nAn Encoder feeds a Transformer. A Transformer uses attention.\n",
    )

    _all(conn, settings)

    row = next(
        r for r in repo.list_outgoing_links(conn, src) if r["page_name"] == "Transformer"
    )
    # anchor stays the linked name; context is the whole first-mention sentence.
    assert row["anchor_text"] == "Transformer"
    assert row["context_sentence"] == "An Encoder feeds a Transformer."


def test_dry_run_reports_without_writing(conn, settings):
    _page(conn, settings, "Transformer", "# Transformer\n\nx\n")
    src = _page(conn, settings, "Doc", "# Doc\n\nA Transformer appears.\n")

    results = _all(conn, settings, dry_run=True)

    assert any(r.changed for r in results)
    assert "](Transformer.md)" not in _body(settings, "Doc")  # file untouched
    assert repo.list_outgoing_links(conn, src) == []  # table untouched
