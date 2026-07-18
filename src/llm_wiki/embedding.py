"""Local sentence-transformers embeddings.

The model is loaded lazily and cached: importing this module must stay cheap,
since loading weights takes seconds and most CLI paths never need them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from llm_wiki.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _model(name: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(text: str, *, model_name: str | None = None) -> list[float]:
    """Embed a single string into a normalized vector."""
    model = _model(model_name or settings.embedding_model)
    vector = model.encode(text, normalize_embeddings=True)
    return [float(x) for x in vector]


def embedding_dim(*, model_name: str | None = None) -> int:
    model = _model(model_name or settings.embedding_model)
    # Renamed in sentence-transformers 5.x; keep working on older releases too.
    getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(getter())


def identity_text(name: str) -> str:
    """Build the text a page is indexed under for entity resolution.

    The name alone, deliberately. This vector answers "is this the same entity?",
    and two documents describing one entity differently must still match. Mixing
    the description in measures topical similarity instead: in testing it pushed
    "Self-attention network" and "Self-attention networks" from 0.06 to 0.26
    cosine distance, far enough apart to create a duplicate page.
    """
    return name.strip()
