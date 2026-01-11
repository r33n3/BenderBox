"""
BenderBox Utils Module - Utility Functions and Handlers

This module provides:
- ModelSourceHandler: Resolve models from URLs, Hugging Face, or local files
- ModelSource: Enum for source types
- ResolvedModel: Result of model resolution
- DownloadProgress: Progress information for downloads
- PrerequisiteManager: Manage external tool dependencies
- PackageDefinition: Define installable packages
"""

from benderbox.utils.model_source import (
    ModelSource,
    ModelSourceHandler,
    ModelSourceError,
    ResolvedModel,
    DownloadProgress,
)

from benderbox.utils.prerequisites import (
    PrerequisiteManager,
    PackageDefinition,
    PackageStatus,
    PackageCheckResult,
    InstallResult,
    InstallMethod,
    InstallConfig,
    VersionInfo,
)

__all__ = [
    # Model source handling
    "ModelSource",
    "ModelSourceHandler",
    "ModelSourceError",
    "ResolvedModel",
    "DownloadProgress",
    # Prerequisites management
    "PrerequisiteManager",
    "PackageDefinition",
    "PackageStatus",
    "PackageCheckResult",
    "InstallResult",
    "InstallMethod",
    "InstallConfig",
    "VersionInfo",
]
