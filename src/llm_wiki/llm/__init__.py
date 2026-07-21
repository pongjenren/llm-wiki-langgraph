"""LLM layer: OpenRouter client, prompts, and the schemas its replies must satisfy."""

from llm_wiki.llm.client import LLMClient, LLMError, query_LLM

__all__ = ["LLMClient", "LLMError", "query_LLM"]
