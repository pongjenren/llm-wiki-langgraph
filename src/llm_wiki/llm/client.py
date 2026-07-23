"""Direct OpenRouter-backed LLM client.

Every LLM step in the ingest pipeline goes through here. `query_LLM` makes a
single OpenRouter chat/completions call; `LLMClient` wraps it with the
pipeline's cross-cutting concerns. OpenRouter's `openai/gpt-oss-120b` is not
prompted for structured output, so `run_json` asks for JSON in the prompt,
validates the reply against a Pydantic model, and re-prompts with the
validation error when it does not conform.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

import json_repair
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from llm_wiki.config import settings

T = TypeVar("T", bound=BaseModel)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_JSON_INSTRUCTIONS = """
Reply with a single JSON value and nothing else. No prose, no explanation, no
code fence. It must validate against this JSON schema:

{schema}
""".strip()


class LLMError(RuntimeError):
    """An LLM call failed, or never produced a valid response."""


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


def query_LLM(
    prompt: str,
    *,
    client: OpenAI,
    model: str,
    temperature: float,
    max_tokens: int,
    label: str = "step",
) -> str:
    """Make one OpenRouter chat/completions call and return the reply text.

    Pipeline steps must not see each other's history, so each call is a fresh,
    single-message request with no shared conversation state.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except OpenAIError as exc:
        raise LLMError(f"{label}: LLM call failed: {exc}") from exc

    content = response.choices[0].message.content if response.choices else None
    if not content or not content.strip():
        raise LLMError(f"{label}: LLM returned an empty response")
    return content.strip()


class LLMClient:
    """Client over OpenRouter's chat/completions API.

    Holds one `OpenAI` instance (and its connection pool) for reuse across
    pipeline steps. `run_json` and `run_text` are the interface the graph nodes
    depend on; tests swap this object out wholesale.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int | None = None,
    ) -> None:
        self._model = model or settings.model
        self._temperature = settings.temperature if temperature is None else temperature
        self._max_tokens = settings.max_tokens if max_tokens is None else max_tokens
        self._retries = settings.json_retries if retries is None else retries
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=settings.openrouter_api_key,
        )

    def run_text(self, prompt: str, *, label: str = "step") -> str:
        return query_LLM(
            prompt,
            client=self._client,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            label=label,
        )

    def run_json(self, prompt: str, schema: type[T], *, label: str = "step") -> T:
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

            raw = self.run_text(request, label=f"{label}#{attempt}")
            try:
                return schema.model_validate(_extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:1000]

        raise LLMError(f"{label}: no valid JSON after {self._retries + 1} attempts: {last_error}")
