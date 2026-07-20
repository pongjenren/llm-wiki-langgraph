"""Read-only web dashboard for ingest status, timing, and logs."""

from __future__ import annotations

from llm_wiki.dashboard.app import create_app, serve

__all__ = ["create_app", "serve"]
