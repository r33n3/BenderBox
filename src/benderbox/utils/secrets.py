"""
Secure secrets management for BenderBox.

Handles storage and retrieval of sensitive data like API keys.
Stores secrets in BenderBox/data/secrets.yaml with restricted permissions.
"""

import logging
import os
import platform
import stat
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


# Supported API providers
API_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "key_prefix": "sk-",
        "description": "OpenAI API key for GPT models",
        "url": "https://platform.openai.com/api-keys",
    },
    "anthropic": {
        "name": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "key_prefix": "sk-ant-",
        "description": "Anthropic API key for Claude models",
        "url": "https://console.anthropic.com/settings/keys",
    },
    "google": {
        "name": "Google",
        "env_var": "GOOGLE_API_KEY",
        "key_prefix": "",
        "description": "Google API key for Gemini models",
        "url": "https://aistudio.google.com/app/apikey",
    },
    "xai": {
        "name": "xAI",
        "env_var": "XAI_API_KEY",
        "key_prefix": "xai-",
        "description": "xAI API key for Grok models",
        "url": "https://console.x.ai/",
    },
}


def get_secrets_path() -> Path:
    """Get the path to the secrets file."""
    from benderbox.config import get_benderbox_home
    return get_benderbox_home() / "data" / "secrets.yaml"


def mask_key(key: str, visible_chars: int = 4) -> str:
    """
    Mask an API key for display.

    Args:
        key: The API key to mask
        visible_chars: Number of characters to show at start and end

    Returns:
        Masked key like "sk-abc...xyz"
    """
    if not key:
        return "(not set)"

    if len(key) <= visible_chars * 2:
        return "*" * len(key)

    return f"{key[:visible_chars]}...{key[-visible_chars:]}"


def _set_file_permissions(path: Path) -> None:
    """
    Set restrictive permissions on the secrets file.

    On Unix: 600 (owner read/write only)
    On Windows: Uses icacls to restrict access
    """
    try:
        if platform.system() == "Windows":
            # Windows: Use icacls to set permissions
            import subprocess
            # Remove inherited permissions and set owner-only access
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{os.environ.get('USERNAME', 'User')}:F"],
                capture_output=True,
                check=False,
            )
        else:
            # Unix: Set 600 permissions
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception as e:
        logger.warning(f"Could not set file permissions: {e}")


class SecretsManager:
    """
    Manages secure storage of API keys and other secrets.

    Secrets are stored in BenderBox/data/secrets.yaml with restricted permissions.
    Environment variables take precedence over stored secrets.
    """

    def __init__(self, secrets_path: Optional[Path] = None):
        """
        Initialize the secrets manager.

        Args:
            secrets_path: Custom path for secrets file (default: BenderBox/data/secrets.yaml)
        """
        self._path = secrets_path or get_secrets_path()
        self._secrets: Dict[str, Any] = {}
        self._loaded = False

    @property
    def path(self) -> Path:
        """Path to the secrets file."""
        return self._path

    def _ensure_directory(self) -> None:
        """Ensure the secrets directory exists with proper permissions."""
        directory = self._path.parent
        if not directory.exists():
            directory.mkdir(parents=True, mode=0o700)
            logger.info(f"Created secrets directory: {directory}")

    def load(self) -> Dict[str, Any]:
        """
        Load secrets from file.

        Returns:
            Dictionary of secrets
        """
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._secrets = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load secrets: {e}")
                self._secrets = {}
        else:
            self._secrets = {}

        self._loaded = True
        return self._secrets

    def save(self) -> bool:
        """
        Save secrets to file with restricted permissions.

        Returns:
            True if successful
        """
        try:
            self._ensure_directory()

            with open(self._path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._secrets, f, default_flow_style=False)

            _set_file_permissions(self._path)
            logger.info(f"Secrets saved to: {self._path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            return False

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider.

        Checks environment variable first, then stored secrets.

        Args:
            provider: Provider key (openai, anthropic, google, xai)

        Returns:
            API key or None if not found
        """
        provider_info = API_PROVIDERS.get(provider)
        if not provider_info:
            return None

        # Check environment variable first
        env_var = provider_info["env_var"]
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value

        # Check stored secrets
        if not self._loaded:
            self.load()

        api_keys = self._secrets.get("api_keys", {})
        return api_keys.get(provider)

    def set_api_key(self, provider: str, key: str) -> bool:
        """
        Set API key for a provider.

        Args:
            provider: Provider key (openai, anthropic, google, xai)
            key: The API key

        Returns:
            True if successful
        """
        if provider not in API_PROVIDERS:
            logger.error(f"Unknown provider: {provider}")
            return False

        if not self._loaded:
            self.load()

        if "api_keys" not in self._secrets:
            self._secrets["api_keys"] = {}

        self._secrets["api_keys"][provider] = key
        return self.save()

    def remove_api_key(self, provider: str) -> bool:
        """
        Remove API key for a provider.

        Args:
            provider: Provider key

        Returns:
            True if successful
        """
        if not self._loaded:
            self.load()

        api_keys = self._secrets.get("api_keys", {})
        if provider in api_keys:
            del api_keys[provider]
            return self.save()

        return True

    def list_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        List all API key statuses.

        Returns:
            Dictionary with provider info and masked keys
        """
        if not self._loaded:
            self.load()

        result = {}
        for provider, info in API_PROVIDERS.items():
            key = self.get_api_key(provider)
            source = None

            if key:
                # Determine source
                env_var = info["env_var"]
                if os.environ.get(env_var):
                    source = "environment"
                else:
                    source = "secrets file"

            result[provider] = {
                "name": info["name"],
                "configured": bool(key),
                "masked_key": mask_key(key) if key else "(not set)",
                "source": source,
                "env_var": info["env_var"],
                "url": info["url"],
            }

        return result

    def clear_all(self) -> bool:
        """
        Clear all stored secrets.

        Returns:
            True if successful
        """
        self._secrets = {}
        if self._path.exists():
            try:
                self._path.unlink()
                logger.info("Secrets file deleted")
                return True
            except Exception as e:
                logger.error(f"Failed to delete secrets file: {e}")
                return False
        return True


# Global instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager
