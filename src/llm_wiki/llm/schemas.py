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
        description="Self-contained description of this item, containing only facts the "
        "document supports and no citation markers."
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names the document uses for this same item.",
    )


class ExtractionResult(BaseModel):
    items: list[ExtractedItem] = Field(default_factory=list)


class ResolveDecision(BaseModel):
    """Whether a newly extracted item is one of the existing candidate pages."""

    matched_page_id: int | None = Field(
        default=None,
        description="page_id of the candidate that is the SAME entity/concept, "
        "or null to create a new page.",
    )
    confidence: Literal["high", "low"] = Field(
        default="high",
        description="'low' when the match is plausible but uncertain; it flags the page for review.",
    )
    reason: str = Field(description="One sentence justification.")


class Review(BaseModel):
    """Outcome of a review step."""

    verdict: Verdict
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete problems to fix. Required when the verdict is 'fail'.",
    )
