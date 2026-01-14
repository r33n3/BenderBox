"""
BenderBox Configuration System

Provides centralized configuration management with support for:
- YAML configuration files
- Environment variable overrides
- Sensible defaults for offline operation

DEFAULT PATHS
=============

BenderBox stores all data in the deployment folder (where bb.py is located):

    BenderBox/
    ├── bb.py                  # Main entry point
    ├── models/
    │   ├── nlp/               # BenderBox's own chat model (TinyLlama)
    │   └── analysis/          # Models to be analyzed/interrogated
    ├── data/
    │   ├── benderbox.db       # SQLite database (reports, sessions)
    │   ├── chromadb/          # Vector store for semantic search
    │   ├── knowledge/         # Knowledge base entries
    │   ├── reports/           # Generated analysis reports
    │   └── secrets.yaml       # API keys (gitignored)
    └── tools/                 # External tools (llama-cli, etc.)

This can be customized with the BENDERBOX_HOME environment variable.

ENVIRONMENT VARIABLES
=====================

Set BENDERBOX_HOME to change the base directory:
    export BENDERBOX_HOME=/path/to/custom/location

Override specific settings with BENDERBOX_* variables:
    BENDERBOX_LLM_ANALYSIS_MODEL  - Path to analysis model
    BENDERBOX_LLM_NLP_MODEL       - Path to NLP/chat model
    BENDERBOX_LLM_THREADS         - Number of CPU threads
    BENDERBOX_LLM_GPU_LAYERS      - GPU layers for acceleration
    OPENAI_API_KEY                - OpenAI API key
    ANTHROPIC_API_KEY             - Anthropic API key

CONFIGURATION FILE
==================

Create BenderBox/data/config.yaml to customize settings:

    llm:
      context_length: 8192
      threads: 8
      gpu_layers: 32

    storage:
      reports_path: "/custom/reports/path"

    ui:
      color_output: true
      progress_indicators: true

API ACCESS
==========

Use get_config_info() to see all paths and settings.
Use get_benderbox_home() to get the base directory.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml


@dataclass
class LLMConfig:
    """Configuration for Local LLM Engine."""

    # Model paths (relative to base directory or absolute)
    # - analysis: Models for interrogating/analyzing other models
    # - nlp: Models for BenderBox's own chat/NLP features
    # - code: Models for code generation/analysis
    analysis_model_path: str = "models/analysis/model.gguf"
    nlp_model_path: str = "models/nlp/model.gguf"
    code_model_path: str = "models/code/model.gguf"

    # Model parameters
    context_length: int = 4096
    threads: int = 4
    gpu_layers: int = 0  # 0 = CPU only

    # Generation defaults
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95

    # Memory management
    max_loaded_models: int = 2  # LRU eviction when exceeded


@dataclass
class StorageConfig:
    """Configuration for Storage Layer."""

    # Vector store (ChromaDB)
    vector_store_path: str = "data/chromadb"

    # SQLite database
    db_path: str = "data/benderbox.db"

    # Knowledge base
    knowledge_path: str = "data/knowledge"

    # Report storage
    reports_path: str = "data/reports"

    # Model cache for downloaded models (URLs, Hugging Face)
    model_cache_path: str = "data/models"
    model_cache_ttl_days: int = 30
    download_timeout_seconds: int = 600
    max_download_size_gb: float = 50.0


@dataclass
class EmbeddingConfig:
    """Configuration for Embedding Model."""

    # Model name or local path
    model_name_or_path: str = "all-MiniLM-L6-v2"

    # Local cache directory
    cache_dir: str = "models/embeddings"

    # Use local model only (no download)
    offline_mode: bool = True

    # Embedding cache
    enable_cache: bool = True
    cache_size: int = 10000


@dataclass
class AnalysisConfig:
    """Configuration for Analysis Engine."""

    # Default analysis profile
    default_profile: str = "standard"

    # Cache settings
    cache_ttl_seconds: int = 3600  # 1 hour

    # Concurrency
    max_concurrent_analyses: int = 2

    # Semantic analysis
    enable_semantic_analysis: bool = True
    semantic_chunk_size: int = 2000  # Characters per chunk


@dataclass
class UIConfig:
    """Configuration for User Interfaces."""

    # Web UI
    web_enabled: bool = False
    web_host: str = "127.0.0.1"
    web_port: int = 8080

    # TUI
    tui_theme: str = "dark"

    # CLI
    color_output: bool = True
    progress_indicators: bool = True


@dataclass
class APIConfig:
    """Configuration for API-based model runners."""

    # OpenAI
    openai_api_key: str = ""  # Or from OPENAI_API_KEY env var
    openai_base_url: str = "https://api.openai.com/v1"

    # Anthropic
    anthropic_api_key: str = ""  # Or from ANTHROPIC_API_KEY env var
    anthropic_base_url: str = "https://api.anthropic.com"

    # Google (Gemini)
    google_api_key: str = ""  # Or from GOOGLE_API_KEY env var

    # xAI (Grok)
    xai_api_key: str = ""  # Or from XAI_API_KEY env var
    xai_base_url: str = "https://api.x.ai/v1"

    # Common settings
    api_timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class Config:
    """Main configuration container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Base paths (resolved at load time)
    base_path: str = ""


def _resolve_path(path: str, base_path: Path) -> str:
    """Resolve a path relative to base_path if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(base_path / p)


def _apply_env_overrides(config: Config) -> None:
    """Apply environment variable overrides to config."""
    env_mappings = {
        # LLM
        "BENDERBOX_LLM_ANALYSIS_MODEL": ("llm", "analysis_model_path"),
        "BENDERBOX_LLM_NLP_MODEL": ("llm", "nlp_model_path"),
        "BENDERBOX_LLM_CODE_MODEL": ("llm", "code_model_path"),
        "BENDERBOX_LLM_CONTEXT_LENGTH": ("llm", "context_length", int),
        "BENDERBOX_LLM_THREADS": ("llm", "threads", int),
        "BENDERBOX_LLM_GPU_LAYERS": ("llm", "gpu_layers", int),

        # Storage
        "BENDERBOX_STORAGE_VECTOR_PATH": ("storage", "vector_store_path"),
        "BENDERBOX_STORAGE_DB_PATH": ("storage", "db_path"),
        "BENDERBOX_STORAGE_KNOWLEDGE_PATH": ("storage", "knowledge_path"),

        # Embedding
        "BENDERBOX_EMBEDDING_MODEL": ("embedding", "model_name_or_path"),
        "BENDERBOX_EMBEDDING_OFFLINE": ("embedding", "offline_mode", lambda x: x.lower() == "true"),

        # Analysis
        "BENDERBOX_ANALYSIS_PROFILE": ("analysis", "default_profile"),
        "BENDERBOX_ANALYSIS_SEMANTIC": ("analysis", "enable_semantic_analysis", lambda x: x.lower() == "true"),

        # UI
        "BENDERBOX_UI_WEB_ENABLED": ("ui", "web_enabled", lambda x: x.lower() == "true"),
        "BENDERBOX_UI_WEB_HOST": ("ui", "web_host"),
        "BENDERBOX_UI_WEB_PORT": ("ui", "web_port", int),

        # API (also check standard provider env vars)
        "BENDERBOX_API_TIMEOUT": ("api", "api_timeout_seconds", int),
        "BENDERBOX_API_MAX_RETRIES": ("api", "max_retries", int),
    }

    # Check standard API key environment variables
    standard_api_keys = {
        "OPENAI_API_KEY": ("api", "openai_api_key"),
        "ANTHROPIC_API_KEY": ("api", "anthropic_api_key"),
        "GOOGLE_API_KEY": ("api", "google_api_key"),
        "XAI_API_KEY": ("api", "xai_api_key"),
    }
    env_mappings.update(standard_api_keys)

    for env_var, mapping in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            section_name = mapping[0]
            attr_name = mapping[1]
            converter = mapping[2] if len(mapping) > 2 else str

            section = getattr(config, section_name)
            try:
                setattr(section, attr_name, converter(value))
            except (ValueError, TypeError):
                pass  # Ignore invalid env values


def _apply_secrets(config: Config) -> None:
    """Apply API keys from secrets manager to config."""
    try:
        from benderbox.utils.secrets import get_secrets_manager

        secrets = get_secrets_manager()

        # Map provider keys to config attributes
        api_key_mappings = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "google": "google_api_key",
            "xai": "xai_api_key",
        }

        for provider, attr_name in api_key_mappings.items():
            # Only set if not already set (env vars take precedence)
            current_value = getattr(config.api, attr_name, "")
            if not current_value:
                secret_value = secrets.get_api_key(provider)
                if secret_value:
                    setattr(config.api, attr_name, secret_value)

    except ImportError:
        pass  # Secrets module not available
    except Exception:
        pass  # Ignore errors loading secrets


def _dict_to_config(data: Dict[str, Any]) -> Config:
    """Convert a dictionary to a Config object."""
    config = Config()

    if "llm" in data:
        for key, value in data["llm"].items():
            if hasattr(config.llm, key):
                setattr(config.llm, key, value)

    if "storage" in data:
        for key, value in data["storage"].items():
            if hasattr(config.storage, key):
                setattr(config.storage, key, value)

    if "embedding" in data:
        for key, value in data["embedding"].items():
            if hasattr(config.embedding, key):
                setattr(config.embedding, key, value)

    if "analysis" in data:
        for key, value in data["analysis"].items():
            if hasattr(config.analysis, key):
                setattr(config.analysis, key, value)

    if "ui" in data:
        for key, value in data["ui"].items():
            if hasattr(config.ui, key):
                setattr(config.ui, key, value)

    if "api" in data:
        for key, value in data["api"].items():
            if hasattr(config.api, key):
                setattr(config.api, key, value)

    return config


def _ensure_directories(config: "Config") -> None:
    """
    Ensure all required directories exist.

    Creates directories for:
    - Model storage (analysis, code, NLP models)
    - Data storage (vector DB, reports, knowledge base)
    - Cache directories (model downloads, embeddings)
    """
    directories = [
        # Model directories
        Path(config.llm.analysis_model_path).parent,
        Path(config.llm.nlp_model_path).parent,
        Path(config.llm.code_model_path).parent,
        # Storage directories
        Path(config.storage.vector_store_path),
        Path(config.storage.db_path).parent,
        Path(config.storage.knowledge_path),
        Path(config.storage.reports_path),
        Path(config.storage.model_cache_path),
        # Cache directories
        Path(config.embedding.cache_dir),
    ]

    for dir_path in directories:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # Ignore errors, will fail later if needed


def _get_deployment_dir() -> Path:
    """
    Get the BenderBox deployment directory (where bb.py is located).

    This finds the root BenderBox folder by looking for bb.py.
    """
    # Start from this file's location and go up to find bb.py
    current = Path(__file__).resolve()

    # Go up the directory tree looking for bb.py
    for parent in [current] + list(current.parents):
        if (parent / "bb.py").exists():
            return parent

    # Fallback to current working directory
    cwd = Path.cwd()
    if (cwd / "bb.py").exists():
        return cwd

    # Last resort: use the src parent
    return current.parent.parent.parent


def get_benderbox_home() -> Path:
    """
    Get the BenderBox home directory.

    Priority:
    1. BENDERBOX_HOME environment variable if set
    2. Deployment directory (where bb.py is located)

    All data (models, reports, etc.) is stored in this directory.
    """
    env_home = os.environ.get("BENDERBOX_HOME")
    if env_home:
        return Path(env_home)
    return _get_deployment_dir()


def get_legacy_home() -> Path:
    """Get the legacy ~/.benderbox/ directory for migration."""
    return Path.home() / ".benderbox"


def migrate_from_legacy(force: bool = False) -> Dict[str, Any]:
    """
    Migrate data from legacy ~/.benderbox/ to deployment folder.

    Args:
        force: If True, overwrite existing files in destination.

    Returns:
        Dictionary with migration results:
        - migrated: List of files migrated
        - skipped: List of files skipped (already exist)
        - errors: List of errors encountered
    """
    import shutil

    legacy = get_legacy_home()
    deployment = get_benderbox_home()

    results = {
        "migrated": [],
        "skipped": [],
        "errors": [],
        "legacy_path": str(legacy),
        "deployment_path": str(deployment),
    }

    if not legacy.exists():
        return results

    # Migration mappings: (legacy_subpath, deployment_subpath)
    migrations = [
        ("models", "models"),
        ("data", "data"),
        ("tools", "tools"),
        ("secrets.yaml", "data/secrets.yaml"),
    ]

    for legacy_sub, deploy_sub in migrations:
        src = legacy / legacy_sub
        dst = deployment / deploy_sub

        if not src.exists():
            continue

        try:
            if src.is_file():
                # File migration
                if dst.exists() and not force:
                    results["skipped"].append(str(src))
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                results["migrated"].append(f"{src} -> {dst}")
            else:
                # Directory migration
                for item in src.rglob("*"):
                    if item.is_file():
                        rel_path = item.relative_to(src)
                        dest_file = dst / rel_path

                        if dest_file.exists() and not force:
                            results["skipped"].append(str(item))
                            continue

                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dest_file)
                        results["migrated"].append(f"{item} -> {dest_file}")
        except Exception as e:
            results["errors"].append(f"{src}: {e}")

    return results


def check_migration_needed() -> bool:
    """Check if migration from legacy ~/.benderbox/ is needed."""
    legacy = get_legacy_home()
    deployment = get_benderbox_home()

    # No legacy folder exists
    if not legacy.exists():
        return False

    # Check if legacy has content that deployment doesn't
    legacy_has_models = (legacy / "models").exists() and any((legacy / "models").rglob("*.gguf"))
    legacy_has_data = (legacy / "data").exists()

    deploy_has_models = (deployment / "models").exists() and any((deployment / "models").rglob("*.gguf"))

    # Migration needed if legacy has stuff we don't
    return legacy_has_models and not deploy_has_models


def load_config(config_path: Optional[str] = None, base_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to config file. If None, looks for data/config.yaml
        base_path: Base path for resolving relative paths. If None, uses deployment folder.

    Returns:
        Config object with all settings loaded.
    """
    # Determine base path - default to deployment folder for consistent storage
    if base_path:
        base = Path(base_path)
    else:
        base = get_benderbox_home()

    # Ensure base directory exists
    base.mkdir(parents=True, exist_ok=True)

    # Determine config file path
    if config_path:
        config_file = Path(config_path)
    else:
        # Check new location first: data/config.yaml
        config_file = base / "data" / "config.yaml"
        if not config_file.exists():
            # Fall back to legacy location
            legacy_config = base / "config" / "benderbox.yaml"
            if legacy_config.exists():
                config_file = legacy_config

    # Load config from file if it exists
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = _dict_to_config(data)
    else:
        config = Config()

    # Set base path
    config.base_path = str(base)

    # Resolve relative paths to absolute paths under base directory
    config.llm.analysis_model_path = _resolve_path(config.llm.analysis_model_path, base)
    config.llm.nlp_model_path = _resolve_path(config.llm.nlp_model_path, base)
    config.llm.code_model_path = _resolve_path(config.llm.code_model_path, base)
    config.storage.vector_store_path = _resolve_path(config.storage.vector_store_path, base)
    config.storage.db_path = _resolve_path(config.storage.db_path, base)
    config.storage.knowledge_path = _resolve_path(config.storage.knowledge_path, base)
    config.storage.reports_path = _resolve_path(config.storage.reports_path, base)
    config.storage.model_cache_path = _resolve_path(config.storage.model_cache_path, base)
    config.embedding.cache_dir = _resolve_path(config.embedding.cache_dir, base)

    # Ensure critical directories exist
    _ensure_directories(config)

    # Apply environment variable overrides
    _apply_env_overrides(config)

    # Apply secrets from secrets manager (lowest priority - env vars override)
    _apply_secrets(config)

    return config


def get_default_config() -> Config:
    """Get a Config object with all default values."""
    return Config()


# Global config instance (lazy loaded)
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Loads config on first call, returns cached instance thereafter.
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Reload the global configuration.

    Args:
        config_path: Optional path to config file.

    Returns:
        Newly loaded Config object.
    """
    global _global_config
    _global_config = load_config(config_path)
    return _global_config


def get_config_info() -> Dict[str, Any]:
    """
    Get a dictionary with all configuration paths and settings.

    Useful for debugging and displaying configuration to users.

    Returns:
        Dictionary with keys:
        - home: BenderBox home directory
        - config_file: Path to config file (may not exist)
        - paths: Dictionary of all configured paths
        - settings: Dictionary of key settings
        - env_vars: List of environment variables that can override settings
    """
    config = get_config()
    home = get_benderbox_home()

    return {
        "home": str(home),
        "config_file": str(home / "data" / "config.yaml"),
        "paths": {
            "models": {
                "analysis": config.llm.analysis_model_path,
                "nlp": config.llm.nlp_model_path,
                "code": config.llm.code_model_path,
                "embeddings": config.embedding.cache_dir,
            },
            "data": {
                "database": config.storage.db_path,
                "vector_store": config.storage.vector_store_path,
                "knowledge": config.storage.knowledge_path,
                "reports": config.storage.reports_path,
                "model_cache": config.storage.model_cache_path,
            },
        },
        "settings": {
            "llm_context_length": config.llm.context_length,
            "llm_threads": config.llm.threads,
            "llm_gpu_layers": config.llm.gpu_layers,
            "default_profile": config.analysis.default_profile,
            "semantic_analysis": config.analysis.enable_semantic_analysis,
        },
        "env_vars": [
            "BENDERBOX_HOME",
            "BENDERBOX_LLM_ANALYSIS_MODEL",
            "BENDERBOX_LLM_NLP_MODEL",
            "BENDERBOX_LLM_THREADS",
            "BENDERBOX_LLM_GPU_LAYERS",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
        ],
    }


def print_config_paths() -> None:
    """Print configuration paths to stdout in a user-friendly format."""
    info = get_config_info()

    print("\nBenderBox Configuration")
    print("=" * 50)
    print(f"\nHome Directory: {info['home']}")
    print(f"Config File:    {info['config_file']}")

    print("\nModel Paths:")
    for name, path in info["paths"]["models"].items():
        exists = "OK" if Path(path).exists() else "not found"
        print(f"  {name:12} {path} [{exists}]")

    print("\nData Paths:")
    for name, path in info["paths"]["data"].items():
        exists = "OK" if Path(path).exists() else "not found"
        print(f"  {name:12} {path} [{exists}]")

    print("\nSettings:")
    for name, value in info["settings"].items():
        print(f"  {name}: {value}")

    print("\nEnvironment Variables (set these to override):")
    for var in info["env_vars"][:5]:  # First 5
        print(f"  {var}")
    print("  ...")
    print()
