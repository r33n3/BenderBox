"""
BenderBox Reporting Module - Report Generation and Export

This module provides:
- ReportGenerator: Multi-format report generation (Markdown, JSON, HTML)
- ModelCardGenerator: AI model safety/ethics cards
- ExportManager: Export operations and batch processing
"""

__all__ = [
    # Report Generator
    "ReportGenerator",
    "ReportSection",
    "ReportMetadata",
    # Model Card
    "ModelCardGenerator",
    "ModelCard",
    "SafetyRating",
    # Export Manager
    "ExportManager",
    "ExportFormat",
    "ExportResult",
    "BatchExportResult",
]


def __getattr__(name):
    """Lazy import for module attributes."""
    # Report Generator
    if name in ("ReportGenerator", "ReportSection", "ReportMetadata"):
        from benderbox.reporting.report_generator import (
            ReportGenerator,
            ReportSection,
            ReportMetadata,
        )
        return locals()[name]

    # Model Card
    if name in ("ModelCardGenerator", "ModelCard", "SafetyRating"):
        from benderbox.reporting.model_card import (
            ModelCardGenerator,
            ModelCard,
            SafetyRating,
        )
        return locals()[name]

    # Export Manager
    if name in ("ExportManager", "ExportFormat", "ExportResult", "BatchExportResult"):
        from benderbox.reporting.export import (
            ExportManager,
            ExportFormat,
            ExportResult,
            BatchExportResult,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
