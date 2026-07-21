"""Runtime configuration, loaded from the environment (and an optional .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # Paths
    raw_dir: Path = field(default_factory=lambda: _env_path("LLM_WIKI_RAW_DIR", PROJECT_ROOT / "raw"))
    wiki_dir: Path = field(default_factory=lambda: _env_path("LLM_WIKI_WIKI_DIR", PROJECT_ROOT / "wiki"))
    db_path: Path = field(default_factory=lambda: _env_path("LLM_WIKI_DB", PROJECT_ROOT / "llm_wiki.db"))

    # LLM
    model: str = field(default_factory=lambda: os.getenv("LLM_WIKI_MODEL", "openai/gpt-oss-120b"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_WIKI_TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_WIKI_MAX_TOKENS", 8192))
    json_retries: int = field(default_factory=lambda: _env_int("LLM_WIKI_JSON_RETRIES", 2))

    # Review loops: how many refine attempts before we give up and flag needs_review.
    review_retries: int = field(default_factory=lambda: _env_int("LLM_WIKI_REVIEW_RETRIES", 2))

    # Entity resolution
    #
    # Resolution is a funnel: an exact alias hit, then a cheap "strong signal"
    # auto-accept, then — only for the genuinely ambiguous cases — candidates
    # from both the vector index and string matching are handed to an LLM judge.
    embedding_model: str = field(
        default_factory=lambda: os.getenv("LLM_WIKI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # --- candidate retrieval (recall-oriented; the LLM makes the final call) ---
    # Cosine distance under which a page is pulled as a candidate. sqlite-vec
    # returns distance, not similarity, so smaller means closer.
    embedding_candidate_threshold: float = field(
        default_factory=lambda: _env_float("LLM_WIKI_EMBEDDING_CANDIDATE_THRESHOLD", 0.45)
    )
    # token_sort_ratio (0-100) at or above which an alias is pulled as a candidate.
    string_candidate_threshold: float = field(
        default_factory=lambda: _env_float("LLM_WIKI_STRING_CANDIDATE_THRESHOLD", 80.0)
    )
    # How many candidates (at most) to hand the LLM judge.
    resolve_candidate_limit: int = field(
        default_factory=lambda: _env_int("LLM_WIKI_RESOLVE_CANDIDATE_LIMIT", 5)
    )

    # --- strong-signal auto-accept (precision-oriented; skips the LLM) ---
    # A near-exact match on either signal is accepted without asking the LLM.
    embedding_autoaccept_threshold: float = field(
        default_factory=lambda: _env_float("LLM_WIKI_EMBEDDING_AUTOACCEPT_THRESHOLD", 0.10)
    )
    string_autoaccept_threshold: float = field(
        default_factory=lambda: _env_float("LLM_WIKI_STRING_AUTOACCEPT_THRESHOLD", 95.0)
    )

    @property
    def openrouter_api_key(self) -> str | None:
        return os.getenv("OPENROUTER_API_KEY")


settings = Settings()
