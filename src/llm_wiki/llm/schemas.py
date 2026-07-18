"""Pydantic models the LLM steps must produce."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PageType = Literal["entity", "concept"]
Verdict = Literal["pass", "fail"]


class SummarizeDecision(BaseModel):
    """Whether a raw document should be rewritten before entity extraction."""

    should_summarize: bool
    reason: str = Field(description="One sentence explaining the decision.")


class ExtractedItem(BaseModel):
    """One entity or concept found in a document."""

    name: str = Field(description="Canonical display name.")
    type: PageType
    description: str = Field(
        description="Self-contained description of this item as supported by the document, "
        "citing the source as [CURRENT]."
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names the document uses for this same item.",
    )


class ExtractionResult(BaseModel):
    items: list[ExtractedItem] = Field(default_factory=list)


class Review(BaseModel):
    """Outcome of a review step."""

    verdict: Verdict
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete problems to fix. Required when the verdict is 'fail'.",
    )
