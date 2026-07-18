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
    nanobot_config: Path = field(
        default_factory=lambda: _env_path("LLM_WIKI_NANOBOT_CONFIG", PROJECT_ROOT / "nanobot.config.json")
    )
    nanobot_workspace: Path = field(
        default_factory=lambda: _env_path("LLM_WIKI_NANOBOT_WORKSPACE", PROJECT_ROOT / ".nanobot-workspace")
    )

    # LLM
    model: str = field(default_factory=lambda: os.getenv("LLM_WIKI_MODEL", "openai/gpt-oss-120b"))
    json_retries: int = field(default_factory=lambda: _env_int("LLM_WIKI_JSON_RETRIES", 2))

    # Review loops: how many refine attempts before we give up and flag needs_review.
    review_retries: int = field(default_factory=lambda: _env_int("LLM_WIKI_REVIEW_RETRIES", 2))

    # Entity resolution
    embedding_model: str = field(
        default_factory=lambda: os.getenv("LLM_WIKI_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    # Cosine distance below which an existing page is considered the same entity.
    # sqlite-vec returns distance, not similarity, so smaller means closer.
    similarity_threshold: float = field(
        default_factory=lambda: _env_float("LLM_WIKI_SIMILARITY_THRESHOLD", 0.25)
    )

    @property
    def openrouter_api_key(self) -> str | None:
        return os.getenv("OPENROUTER_API_KEY")


settings = Settings()
