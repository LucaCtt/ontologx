"""Factories for creating backend components."""

from ontologx.backend.embeddings import EmbeddingsFactory
from ontologx.backend.llm import LLMFactory

__all__ = [
    "EmbeddingsFactory",
    "LLMFactory",
]
