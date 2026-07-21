"""Unit coverage for the pieces the pipeline tests exercise only indirectly."""

from __future__ import annotations

from pathlib import Path

import pytest
from openai import OpenAIError
from openpyxl import Workbook

from llm_wiki import loaders, pages
from llm_wiki.db import repo
from llm_wiki.llm.client import LLMClient, LLMError, _extract_json, query_LLM
from llm_wiki.llm.schemas import Review


# ------------------------------------------------------------------ loaders
def test_xlsx_becomes_a_markdown_table(tmp_path: Path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Q3"
    sheet.append(["Region", "Revenue"])
    sheet.append(["APAC", 1200])
    path = tmp_path / "r.xlsx"
    workbook.save(path)

    text = loaders.load(path).text
    lines = text.splitlines()
    assert "## Sheet: Q3" in lines
    header = lines.index("| Region | Revenue |")
    # Table rows must be on consecutive lines or the markdown table breaks.
    assert lines[header + 1] == "| --- | --- |"
    assert lines[header + 2] == "| APAC | 1200 |"


def test_hash_covers_raw_bytes_not_extracted_text(tmp_path: Path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("same", encoding="utf-8")
    b.write_text("same", encoding="utf-8")
    assert loaders.load(a).sha256 == loaders.load(b).sha256

    b.write_text("different", encoding="utf-8")
    assert loaders.load(a).sha256 != loaders.load(b).sha256


def test_unsupported_extension_is_rejected(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"\x00")
    with pytest.raises(loaders.UnsupportedFileType):
        loaders.load(path)


def test_namespace_comes_from_directory_name(tmp_path: Path):
    (tmp_path / "ml").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / "ml" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "ops" / "b.txt").write_text("y", encoding="utf-8")
    (tmp_path / "ml" / "skip.bin").write_bytes(b"\x00")

    found = loaders.iter_raw_files(tmp_path)
    assert [(ns, p.name) for ns, p in found] == [("ml", "a.md"), ("ops", "b.txt")]


# -------------------------------------------------------------------- pages
def test_substitute_current_replaces_every_marker():
    text = "A [CURRENT] and B [CURRENT]."
    assert pages.substitute_current(text, 3) == "A [3] and B [3]."


def test_strip_references_removes_a_model_written_section():
    body = "# T\n\nBody [1].\n\n## References\n\n1. doc.md\n"
    assert "## References" not in pages.strip_references(body)
    assert "Body [1]." in pages.strip_references(body)


def test_validate_merge_detects_lost_citations_and_sections():
    old = "# T\n\nA [1].\n\n## Architecture\n\nB [2].\n"
    assert pages.validate_merge(old, old) == []

    problems = pages.validate_merge(old, "# T\n\nA [1].\n")
    assert any("[2]" in p for p in problems)
    assert any("Architecture" in p for p in problems)


def test_slugify_makes_filesystem_safe_names():
    assert pages.slugify("Self-attention network") == "Self-attention_network"
    assert pages.slugify("A/B testing") == "AB_testing"
    assert pages.slugify("   ") == "untitled"


# ------------------------------------------------------------------- naming
def test_alias_normalization_is_case_and_space_insensitive():
    assert repo.normalize_name("  Self-Attention   Network ") == "self-attention network"


# --------------------------------------------------------------- llm client
@pytest.mark.parametrize(
    "raw",
    [
        '{"verdict": "pass", "issues": []}',
        '```json\n{"verdict": "pass", "issues": []}\n```',
        'Sure:\n```\n{"verdict": "pass", "issues": []}\n```\nDone.',
        '{"verdict": "pass", "issues": [],}',
        '{"verdict": "pass", "issues": []',
    ],
)
def test_extract_json_tolerates_common_model_output(raw: str):
    assert Review.model_validate(_extract_json(raw)).verdict == "pass"


class _ScriptedLLM:
    """Stands in for query_LLM: returns each reply in turn, recording prompts."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.prompts: list[str] = []

    async def __call__(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]


async def test_run_json_retries_and_feeds_back_the_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = _ScriptedLLM(["not json", '{"verdict": "maybe"}', '{"verdict": "pass", "issues": []}'])
    monkeypatch.setattr("llm_wiki.llm.client.query_LLM", llm)

    client = LLMClient(retries=2)
    result = await client.run_json("review", Review, label="t")

    assert result.verdict == "pass"
    assert len(llm.prompts) == 3
    assert "was rejected" in llm.prompts[1], "the retry must tell the model what was wrong"


async def test_run_json_gives_up_after_the_budget(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("llm_wiki.llm.client.query_LLM", _ScriptedLLM(["nope"]))

    client = LLMClient(retries=1)
    with pytest.raises(LLMError, match="no valid JSON"):
        await client.run_json("review", Review, label="t")


async def test_query_llm_maps_provider_errors():
    class FailingCompletions:
        async def create(self, **kwargs):
            raise OpenAIError("402 insufficient credits")

    class FailingClient:
        chat = type("Chat", (), {"completions": FailingCompletions()})()

    with pytest.raises(LLMError, match="insufficient credits"):
        await query_LLM("hi", client=FailingClient(), model="m", temperature=0.1, max_tokens=8, label="t")
