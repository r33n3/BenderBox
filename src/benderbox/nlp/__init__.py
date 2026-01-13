"""
BenderBox NLP Module - Natural Language Processing for AI Security Analysis

This module provides:
- LocalLLMEngine: Local LLM inference using llama-cpp-python
- EmbeddingModel: Text embedding generation
- IntentRouter: Intent classification and routing
- ResponseGenerator: Natural language response generation
- ConversationManager: Multi-turn conversation handling
- ContextManager: Conversation context tracking
- AnalysisBridge: Bridge to analysis engine
- RLMController: Recursive Language Model processing
- CodebaseAnalyzer: Codebase-scale security analysis
- BatchResponseAnalyzer: Large-scale response analysis
"""

# Lazy imports to avoid circular dependencies and missing deps at import time
__all__ = [
    # Phase 1: LLM Engine
    "LocalLLMEngine",
    "LlamaModel",
    "LLMEngineError",
    "ModelNotFoundError",
    # Phase 1: Embeddings
    "EmbeddingModel",
    "EmbeddingCache",
    "EmbeddingError",
    # Phase 2: Intent
    "IntentRouter",
    "Intent",
    "IntentType",
    # Phase 2: Response
    "ResponseGenerator",
    "ResponseContext",
    "Message",
    # Phase 2: Conversation
    "ConversationManager",
    "Response",
    # Phase 2: Context
    "ContextManager",
    "ConversationContext",
    # Phase 2: Analysis Bridge
    "AnalysisBridge",
    "AnalysisBridgeError",
    # RLM: Recursive Language Model
    "RLMController",
    "RLMConfig",
    "RLMContext",
    "RLMStats",
    "CodebaseAnalyzer",
    "CodebaseAnalysisResult",
    "BatchResponseAnalyzer",
    "BatchResponseResult",
    "analyze_codebase",
    "analyze_responses_batch",
]


def __getattr__(name):
    """Lazy import for module attributes."""
    # Phase 1: LLM Engine
    if name in ("LocalLLMEngine", "LlamaModel", "LLMEngineError", "ModelNotFoundError"):
        from benderbox.nlp.llm_engine import (
            LocalLLMEngine,
            LlamaModel,
            LLMEngineError,
            ModelNotFoundError,
        )
        return locals()[name]

    # Phase 1: Embeddings
    if name in ("EmbeddingModel", "EmbeddingCache", "EmbeddingError"):
        from benderbox.nlp.embeddings import EmbeddingModel, EmbeddingCache, EmbeddingError
        return locals()[name]

    # Phase 2: Intent
    if name in ("IntentRouter", "Intent", "IntentType"):
        from benderbox.nlp.intent import IntentRouter, Intent, IntentType
        return locals()[name]

    # Phase 2: Response
    if name in ("ResponseGenerator", "ResponseContext", "Message"):
        from benderbox.nlp.response import ResponseGenerator, ResponseContext, Message
        return locals()[name]

    # Phase 2: Conversation
    if name in ("ConversationManager", "Response"):
        from benderbox.nlp.conversation import ConversationManager, Response
        return locals()[name]

    # Phase 2: Context
    if name in ("ContextManager", "ConversationContext"):
        from benderbox.nlp.context import ContextManager, ConversationContext
        return locals()[name]

    # Phase 2: Analysis Bridge
    if name in ("AnalysisBridge", "AnalysisBridgeError"):
        from benderbox.nlp.analysis_bridge import AnalysisBridge, AnalysisBridgeError
        return locals()[name]

    # RLM: Recursive Language Model
    if name in (
        "RLMController",
        "RLMConfig",
        "RLMContext",
        "RLMStats",
        "CodebaseAnalyzer",
        "CodebaseAnalysisResult",
        "BatchResponseAnalyzer",
        "BatchResponseResult",
        "analyze_codebase",
        "analyze_responses_batch",
    ):
        from benderbox.nlp.rlm import (
            RLMController,
            RLMConfig,
            RLMContext,
            RLMStats,
            CodebaseAnalyzer,
            CodebaseAnalysisResult,
            BatchResponseAnalyzer,
            BatchResponseResult,
            analyze_codebase,
            analyze_responses_batch,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
