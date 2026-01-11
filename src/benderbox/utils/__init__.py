"""
BenderBox Utils Module - Utility Functions and Handlers

This module provides:
- ModelSourceHandler: Resolve models from URLs, Hugging Face, or local files
- ModelSource: Enum for source types
- ResolvedModel: Result of model resolution
- DownloadProgress: Progress information for downloads
"""

from benderbox.utils.model_source import (
    ModelSource,
    ModelSourceHandler,
    ModelSourceError,
    ResolvedModel,
    DownloadProgress,
)

__all__ = [
    "ModelSource",
    "ModelSourceHandler",
    "ModelSourceError",
    "ResolvedModel",
    "DownloadProgress",
]
