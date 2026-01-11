"""
BenderBox Utils Module - Utility Functions and Handlers

This module provides:
- ModelSourceHandler: Resolve models from URLs, Hugging Face, or local files
- ModelSource: Enum for source types
- ResolvedModel: Result of model resolution
- DownloadProgress: Progress information for downloads
- PrerequisiteManager: Manage external tool dependencies
- PackageDefinition: Define installable packages
- SecretsManager: Secure storage for API keys and secrets
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

from benderbox.utils.secrets import (
    SecretsManager,
    get_secrets_manager,
    mask_key,
    API_PROVIDERS,
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
    # Secrets management
    "SecretsManager",
    "get_secrets_manager",
    "mask_key",
    "API_PROVIDERS",
]
