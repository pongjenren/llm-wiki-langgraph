"""Page-name embeddings via an OpenAI-compatible embeddings API.

Embeddings serve entity resolution only (see :func:`identity_text`): a page is
indexed under a vector of its name so an incoming item can find the pages that
might be the same entity. The vectors come from an external OpenAI-compatible
``/v1/embeddings`` endpoint (e.g. an NV-Embed-v2 deployment on vLLM / TEI / NIM),
so this module holds no model weights. The client is created lazily and cached,
keeping import cheap for CLI paths that never embed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from llm_wiki.config import settings

if TYPE_CHECKING:
    from openai import OpenAI


@lru_cache(maxsize=1)
def _client(base_url: str, api_key: str) -> "OpenAI":
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def embed(text: str, *, model_name: str | None = None) -> list[float]:
    """Embed a single string into a vector via the embeddings API."""
    # The SDK rejects an empty key; unauthenticated endpoints ignore the value.
    client = _client(settings.embedding_base_url, settings.embedding_api_key or "EMPTY")
    response = client.embeddings.create(
        model=model_name or settings.embedding_model,
        input=text,
    )
    return [float(x) for x in response.data[0].embedding]


def identity_text(name: str) -> str:
    """Build the text a page is indexed under for entity resolution.

    The name alone, deliberately. This vector answers "is this the same entity?",
    and two documents describing one entity differently must still match. Mixing
    the description in measures topical similarity instead: in testing it pushed
    "Self-attention network" and "Self-attention networks" from 0.06 to 0.26
    cosine distance, far enough apart to create a duplicate page.
    """
    return name.strip()
