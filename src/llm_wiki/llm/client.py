"""nanobot-backed LLM client.

Every LLM step in the ingest pipeline goes through here. nanobot has no
structured-output mode, so `run_json` asks for JSON in the prompt, validates the
reply against a Pydantic model, and re-prompts with the validation error when it
does not conform.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, TypeVar

import json_repair
from pydantic import BaseModel, ValidationError

from llm_wiki.config import settings

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_JSON_INSTRUCTIONS = """
Reply with a single JSON value and nothing else. No prose, no explanation, no
code fence. It must validate against this JSON schema:

{schema}
""".strip()


class LLMError(RuntimeError):
    """A nanobot run failed, or never produced a valid response."""


def _extract_json(text: str) -> Any:
    """Pull a JSON value out of a model reply.

    Models wrap JSON in fences or prose even when told not to, so this unwraps a
    fenced block if present and falls back to a lenient repair parse.
    """
    candidate = text.strip()
    if match := _FENCE.search(candidate):
        candidate = match.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = json_repair.loads(candidate)
        if repaired == "" or repaired is None:
            raise ValueError("no JSON value found in reply")
        return repaired


class NanobotClient:
    """Async wrapper over the nanobot SDK.

    Each call runs with `ephemeral=True` under a fresh session key: pipeline
    steps must not see each other's history, or an earlier document's entities
    leak into a later extraction.
    """

    def __init__(
        self,
        *,
        config_path: Any = None,
        workspace: Any = None,
        model: str | None = None,
        retries: int | None = None,
    ) -> None:
        self._config_path = config_path or settings.nanobot_config
        self._workspace = workspace or settings.nanobot_workspace
        self._model = model or settings.model
        self._retries = settings.json_retries if retries is None else retries
        self._bot: Any = None

    async def __aenter__(self) -> "NanobotClient":
        from nanobot.nanobot import Nanobot

        self._workspace.mkdir(parents=True, exist_ok=True)
        self._bot = Nanobot.from_config(self._config_path, workspace=self._workspace)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._bot is not None:
            await self._bot.aclose()
            self._bot = None

    async def run_text(self, prompt: str, *, label: str = "step") -> str:
        if self._bot is None:
            raise LLMError("NanobotClient used outside its async context manager")

        result = await self._bot.run(
            prompt,
            session_key=f"llm-wiki:{label}:{uuid.uuid4()}",
            ephemeral=True,
            model=self._model,
        )
        if result.error:
            raise LLMError(f"{label}: nanobot run failed: {result.error}")
        if not result.content or not result.content.strip():
            raise LLMError(f"{label}: nanobot returned an empty response")
        return result.content.strip()

    async def run_json(self, prompt: str, schema: type[T], *, label: str = "step") -> T:
        """Run a prompt and parse the reply into `schema`, retrying on invalid output."""
        request = f"{prompt}\n\n{_JSON_INSTRUCTIONS.format(schema=json.dumps(schema.model_json_schema(), indent=2))}"
        last_error: str | None = None

        for attempt in range(self._retries + 1):
            if last_error is not None:
                request = (
                    f"{prompt}\n\n{_JSON_INSTRUCTIONS.format(schema=json.dumps(schema.model_json_schema(), indent=2))}"
                    f"\n\nYour previous reply was rejected: {last_error}\n"
                    "Return corrected JSON only."
                )

            raw = await self.run_text(request, label=f"{label}#{attempt}")
            try:
                return schema.model_validate(_extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:1000]

        raise LLMError(f"{label}: no valid JSON after {self._retries + 1} attempts: {last_error}")
