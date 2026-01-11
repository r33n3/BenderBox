"""
BenderBox NLP Module - Natural Language Processing for AI Security Analysis

This module provides:
- LocalLLMEngine: Local LLM inference using llama-cpp-python
- ConversationManager: Multi-turn conversation handling
- IntentRouter: Intent classification and routing
- EmbeddingModel: Text embedding generation
"""

# Lazy imports to avoid circular dependencies and missing deps at import time
__all__ = [
    "LocalLLMEngine",
    "LlamaModel",
    "LLMEngineError",
    "ModelNotFoundError",
    "EmbeddingModel",
    "EmbeddingCache",
    "EmbeddingError",
]


def __getattr__(name):
    """Lazy import for module attributes."""
    if name in ("LocalLLMEngine", "LlamaModel", "LLMEngineError", "ModelNotFoundError"):
        from benderbox.nlp.llm_engine import (
            LocalLLMEngine,
            LlamaModel,
            LLMEngineError,
            ModelNotFoundError,
        )
        return locals()[name]
    elif name in ("EmbeddingModel", "EmbeddingCache", "EmbeddingError"):
        from benderbox.nlp.embeddings import EmbeddingModel, EmbeddingCache, EmbeddingError
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
