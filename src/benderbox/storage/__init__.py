"""
BenderBox Storage Module - Data Persistence and Vector Store

This module provides:
- VectorStore: ChromaDB-based vector storage for semantic search
- ReportDatabase: SQLite database for report storage
- KnowledgeBase: Threat knowledge management
"""

# Lazy imports to avoid circular dependencies and missing deps at import time
__all__ = [
    "VectorStore",
    "Document",
    "SearchResult",
    "VectorStoreError",
    "ReportDatabase",
    "ReportFilters",
    "ReportSummary",
    "Finding",
    "ReportDatabaseError",
    "KnowledgeBase",
    "KnowledgeEntry",
    "JailbreakTechnique",
    "VulnerabilityPattern",
    "CodePattern",
]


def __getattr__(name):
    """Lazy import for module attributes."""
    if name in ("VectorStore", "Document", "SearchResult", "VectorStoreError"):
        from benderbox.storage.vector_store import (
            VectorStore,
            Document,
            SearchResult,
            VectorStoreError,
        )
        return locals()[name]
    elif name in ("ReportDatabase", "ReportFilters", "ReportSummary", "Finding", "ReportDatabaseError"):
        from benderbox.storage.report_db import (
            ReportDatabase,
            ReportFilters,
            ReportSummary,
            Finding,
            ReportDatabaseError,
        )
        return locals()[name]
    elif name in ("KnowledgeBase", "KnowledgeEntry", "JailbreakTechnique", "VulnerabilityPattern", "CodePattern"):
        from benderbox.storage.knowledge_base import (
            KnowledgeBase,
            KnowledgeEntry,
            JailbreakTechnique,
            VulnerabilityPattern,
            CodePattern,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
