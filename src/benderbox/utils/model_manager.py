"""
Model Manager for BenderBox

Handles downloading, configuring, and managing local GGUF models
for NLP and analysis features.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RecommendedModel:
    """A recommended model configuration."""

    name: str
    description: str
    huggingface_repo: str
    filename: str
    size_mb: int
    min_ram_gb: int
    quality: str  # "basic", "good", "best"
    use_case: str  # "analysis", "code", "general"


# Recommended models for different resource levels
# Prioritizing small, efficient models that work on limited hardware
RECOMMENDED_MODELS: Dict[str, RecommendedModel] = {
    # Tiny models (< 1GB, works on any system)
    "tinyllama": RecommendedModel(
        name="TinyLlama-1.1B-Chat",
        description="Tiny but capable chat model. Best for limited resources.",
        huggingface_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        size_mb=700,
        min_ram_gb=4,
        quality="basic",
        use_case="general",
    ),
    "tinyllama-small": RecommendedModel(
        name="TinyLlama-1.1B-Chat (Q2_K)",
        description="Smallest quantization. Fastest but lower quality.",
        huggingface_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        size_mb=500,
        min_ram_gb=2,
        quality="basic",
        use_case="general",
    ),
    # Small models (1-2GB, good for most systems)
    "qwen2-1.5b": RecommendedModel(
        name="Qwen2-1.5B-Instruct",
        description="Excellent small model. Great quality for its size.",
        huggingface_repo="Qwen/Qwen2-1.5B-Instruct-GGUF",
        filename="qwen2-1_5b-instruct-q4_k_m.gguf",
        size_mb=1000,
        min_ram_gb=6,
        quality="good",
        use_case="general",
    ),
    "phi2": RecommendedModel(
        name="Phi-2",
        description="Microsoft's efficient 2.7B model. Very capable.",
        huggingface_repo="TheBloke/phi-2-GGUF",
        filename="phi-2.Q4_K_M.gguf",
        size_mb=1700,
        min_ram_gb=8,
        quality="good",
        use_case="analysis",
    ),
    # Medium models (3-5GB, for systems with 16GB+ RAM)
    "mistral-7b": RecommendedModel(
        name="Mistral-7B-Instruct",
        description="High quality 7B model. Requires 16GB+ RAM.",
        huggingface_repo="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        size_mb=4400,
        min_ram_gb=12,
        quality="best",
        use_case="analysis",
    ),
}

# Default model for resource-constrained systems
DEFAULT_MODEL = "tinyllama"


class ModelManager:
    """
    Manages local GGUF models for BenderBox.

    Handles:
    - Listing available and downloaded models
    - Downloading recommended models
    - Setting default models for NLP features
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize ModelManager.

        Args:
            base_path: Base path for BenderBox. Defaults to cwd.
        """
        self.base_path = base_path or Path.cwd()
        self.models_dir = self.base_path / "models"
        self.data_models_dir = self.base_path / "data" / "models"
        self.analysis_model_dir = self.models_dir / "analysis"
        self.code_model_dir = self.models_dir / "code"

    def get_downloaded_models(self) -> List[Dict]:
        """
        Get list of downloaded GGUF models.

        Returns:
            List of model info dictionaries.
        """
        models = []

        # Check models/ directory
        if self.models_dir.exists():
            for gguf_file in self.models_dir.rglob("*.gguf"):
                models.append({
                    "path": str(gguf_file),
                    "name": gguf_file.name,
                    "size_mb": gguf_file.stat().st_size // (1024 * 1024),
                    "location": "models/",
                })

        # Check data/models/ directory (HuggingFace downloads)
        if self.data_models_dir.exists():
            for gguf_file in self.data_models_dir.rglob("*.gguf"):
                models.append({
                    "path": str(gguf_file),
                    "name": gguf_file.name,
                    "size_mb": gguf_file.stat().st_size // (1024 * 1024),
                    "location": "data/models/",
                })

        return models

    def get_recommended_models(self, max_ram_gb: int = 8) -> List[RecommendedModel]:
        """
        Get recommended models for the given RAM constraint.

        Args:
            max_ram_gb: Maximum RAM available in GB.

        Returns:
            List of recommended models that fit the RAM constraint.
        """
        suitable = []
        for model in RECOMMENDED_MODELS.values():
            if model.min_ram_gb <= max_ram_gb:
                suitable.append(model)

        # Sort by quality and size
        quality_order = {"basic": 0, "good": 1, "best": 2}
        suitable.sort(key=lambda m: (quality_order.get(m.quality, 0), m.size_mb))

        return suitable

    def get_default_model_path(self) -> Optional[Path]:
        """
        Get the path to the default analysis model.

        Checks in order:
        1. models/analysis/model.gguf (configured location)
        2. Any downloaded TinyLlama model
        3. Any downloaded .gguf file

        Returns:
            Path to model file, or None if not found.
        """
        # Check configured location
        default_path = self.analysis_model_dir / "model.gguf"
        if default_path.exists():
            return default_path

        # Check for any model in analysis dir
        if self.analysis_model_dir.exists():
            for gguf in self.analysis_model_dir.glob("*.gguf"):
                return gguf

        # Check for TinyLlama in data/models
        tinyllama_pattern = self.data_models_dir / "huggingface" / "*TinyLlama*" / "*.gguf"
        for gguf in self.base_path.glob(str(tinyllama_pattern.relative_to(self.base_path))):
            return gguf

        # Fall back to any downloaded model
        downloaded = self.get_downloaded_models()
        if downloaded:
            return Path(downloaded[0]["path"])

        return None

    def setup_default_model(self, model_path: Path) -> Tuple[bool, str]:
        """
        Set up a model as the default analysis model.

        Creates a symlink or copies the model to models/analysis/model.gguf

        Args:
            model_path: Path to the model file.

        Returns:
            Tuple of (success, message).
        """
        if not model_path.exists():
            return False, f"Model not found: {model_path}"

        # Create analysis directory
        self.analysis_model_dir.mkdir(parents=True, exist_ok=True)

        target_path = self.analysis_model_dir / "model.gguf"

        # Remove existing if present
        if target_path.exists():
            target_path.unlink()

        try:
            # Try to create symlink first (saves space)
            target_path.symlink_to(model_path.resolve())
            return True, f"Linked {model_path.name} as default model"
        except OSError:
            # Fall back to copy on Windows if symlinks not supported
            try:
                shutil.copy2(model_path, target_path)
                return True, f"Copied {model_path.name} as default model"
            except Exception as e:
                return False, f"Failed to set up model: {e}"

    async def download_model(
        self,
        model_key: str,
        progress_callback=None,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Download a recommended model from HuggingFace.

        Args:
            model_key: Key from RECOMMENDED_MODELS.
            progress_callback: Optional callback(message, progress_pct).

        Returns:
            Tuple of (success, message, path).
        """
        if model_key not in RECOMMENDED_MODELS:
            return False, f"Unknown model: {model_key}", None

        model = RECOMMENDED_MODELS[model_key]

        try:
            from benderbox.utils.model_source import ModelSourceHandler

            handler = ModelSourceHandler(
                cache_dir=str(self.data_models_dir),
            )

            # Download from HuggingFace
            if progress_callback:
                progress_callback(f"Downloading {model.name}...", 0)

            result = await handler.resolve(
                model.huggingface_repo,
                preferred_file=model.filename,
            )

            if result.success and result.local_path:
                if progress_callback:
                    progress_callback(f"Downloaded {model.name}", 100)
                return True, f"Downloaded {model.name}", Path(result.local_path)
            else:
                return False, f"Download failed: {result.error}", None

        except ImportError as e:
            return False, f"Missing dependency: {e}", None
        except Exception as e:
            return False, f"Download error: {e}", None

    def get_system_ram_gb(self) -> int:
        """Get system RAM in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total // (1024 ** 3)
        except ImportError:
            # Default assumption
            return 8


def get_model_manager(base_path: Optional[Path] = None) -> ModelManager:
    """Get a ModelManager instance."""
    return ModelManager(base_path)
