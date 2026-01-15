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
            base_path: Base path for BenderBox. Defaults to deployment folder.
        """
        if base_path is None:
            from benderbox.config import get_benderbox_home
            base_path = get_benderbox_home()
        self.base_path = base_path

        # Define directory structure
        self.models_dir = self.base_path / "models"
        self.data_models_dir = self.base_path / "data" / "models"
        self.analysis_model_dir = self.models_dir / "analysis"
        self.code_model_dir = self.models_dir / "code"
        self.nlp_model_dir = self.models_dir / "nlp"

        # Create all required directories on initialization
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.base_path,
            self.models_dir,
            self.data_models_dir,
            self.analysis_model_dir,
            self.code_model_dir,
            self.nlp_model_dir,
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

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

    def get_model_dir_for_purpose(self, purpose: str) -> Path:
        """
        Get the model directory for a given purpose.

        Args:
            purpose: One of "analysis", "nlp", or "code"

        Returns:
            Path to the appropriate model directory.
        """
        purpose_dirs = {
            "analysis": self.analysis_model_dir,
            "nlp": self.nlp_model_dir,
            "code": self.code_model_dir,
        }
        if purpose not in purpose_dirs:
            raise ValueError(f"Unknown purpose: {purpose}. Must be one of: {list(purpose_dirs.keys())}")
        return purpose_dirs[purpose]

    def list_analysis_models(self) -> List[Dict]:
        """
        List all models available for analysis in the analysis folder.

        Returns:
            List of model info dictionaries with name, path, and size.
        """
        models = []
        if self.analysis_model_dir.exists():
            for gguf in self.analysis_model_dir.glob("*.gguf"):
                models.append({
                    "name": gguf.stem,  # filename without extension
                    "filename": gguf.name,
                    "path": str(gguf),
                    "size_mb": gguf.stat().st_size // (1024 * 1024),
                    "purpose": "analysis",
                })
        return sorted(models, key=lambda m: m["name"].lower())

    def list_nlp_models(self) -> List[Dict]:
        """
        List all models available for NLP/chat in the nlp folder.

        Returns:
            List of model info dictionaries with name, path, and size.
        """
        models = []
        if self.nlp_model_dir.exists():
            for gguf in self.nlp_model_dir.glob("*.gguf"):
                models.append({
                    "name": gguf.stem,
                    "filename": gguf.name,
                    "path": str(gguf),
                    "size_mb": gguf.stat().st_size // (1024 * 1024),
                    "purpose": "nlp",
                })
        return sorted(models, key=lambda m: m["name"].lower())

    def find_model_by_name(self, name: str, purpose: str = "analysis") -> Optional[Path]:
        """
        Find a model by name with fuzzy matching across all model locations.

        Args:
            name: Model name (partial match supported, case-insensitive)
            purpose: "analysis" or "nlp" (affects search order, not exclusivity)

        Returns:
            Path to the model if found, None otherwise.

        Search order:
        1. Purpose-specific directory (models/analysis/ or models/nlp/)
        2. Downloaded models (data/models/huggingface/)
        3. All models directories (models/)

        For each location:
        - Exact match (filename without extension)
        - Exact match (full filename with .gguf)
        - Fuzzy match (name contained in filename)
        """
        name_lower = name.lower().strip()

        # Build search directories in priority order
        search_dirs = []

        # 1. Purpose-specific directory first
        try:
            purpose_dir = self.get_model_dir_for_purpose(purpose)
            if purpose_dir.exists():
                search_dirs.append(purpose_dir)
        except ValueError:
            pass

        # 2. Downloaded models (HuggingFace)
        hf_dir = self.data_models_dir / "huggingface"
        if hf_dir.exists():
            search_dirs.append(hf_dir)

        # 3. URL downloaded models
        url_dir = self.data_models_dir / "url"
        if url_dir.exists():
            search_dirs.append(url_dir)

        # 4. All model directories as fallback
        if self.models_dir.exists():
            search_dirs.append(self.models_dir)

        # Search each directory
        all_matches = []

        for search_dir in search_dirs:
            # Recursively find all .gguf files
            for gguf in search_dir.rglob("*.gguf"):
                # 1. Exact match (stem)
                if gguf.stem.lower() == name_lower:
                    return gguf

                # 2. Exact match (full filename)
                if not name_lower.endswith(".gguf"):
                    name_with_ext = name_lower + ".gguf"
                else:
                    name_with_ext = name_lower

                if gguf.name.lower() == name_with_ext:
                    return gguf

                # 3. Fuzzy match (name contained in filename) - collect for later
                if name_lower in gguf.stem.lower():
                    all_matches.append(gguf)

        # Return best fuzzy match
        if len(all_matches) == 1:
            return all_matches[0]
        elif len(all_matches) > 1:
            # Deduplicate by resolved path
            unique_matches = list({p.resolve(): p for p in all_matches}.values())
            if len(unique_matches) == 1:
                return unique_matches[0]
            # Return shortest stem (most specific match)
            return min(unique_matches, key=lambda p: len(p.stem))

        return None

    def get_all_models(self) -> List[Dict]:
        """
        Get all available GGUF models from all locations.

        Returns:
            List of model info dictionaries with deduplication.
        """
        models = []
        seen_paths = set()

        # Check all locations
        for gguf_file in self._iter_all_models():
            resolved = gguf_file.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            # Determine location category
            rel_path = str(gguf_file)
            if "data/models" in rel_path or "data\\models" in rel_path:
                location = "downloaded"
            elif "analysis" in rel_path:
                location = "analysis"
            elif "nlp" in rel_path:
                location = "nlp"
            elif "code" in rel_path:
                location = "code"
            else:
                location = "other"

            models.append({
                "path": str(gguf_file),
                "name": gguf_file.stem,
                "filename": gguf_file.name,
                "size_mb": gguf_file.stat().st_size // (1024 * 1024),
                "location": location,
            })

        return sorted(models, key=lambda m: m["name"].lower())

    def _iter_all_models(self):
        """Iterate over all GGUF files in all model locations."""
        # models/ directory
        if self.models_dir.exists():
            yield from self.models_dir.rglob("*.gguf")

        # data/models/ directory
        if self.data_models_dir.exists():
            yield from self.data_models_dir.rglob("*.gguf")

    def get_model_suggestions(self, partial_name: str, purpose: str = "analysis") -> List[str]:
        """
        Get suggestions for model names based on partial input.

        Args:
            partial_name: Partial model name to match
            purpose: "analysis" or "nlp"

        Returns:
            List of matching model names.
        """
        model_dir = self.get_model_dir_for_purpose(purpose)
        if not model_dir.exists():
            return []

        partial_lower = partial_name.lower().strip()
        suggestions = []

        for gguf in model_dir.glob("*.gguf"):
            if partial_lower in gguf.stem.lower():
                suggestions.append(gguf.stem)

        return sorted(suggestions)

    def setup_default_model(
        self, model_path: Path, purpose: str = "analysis"
    ) -> Tuple[bool, str]:
        """
        Set up a model as the default for a specific purpose.

        Creates a symlink or copies the model to the appropriate directory:
        - analysis: models/analysis/model.gguf (for interrogating other models)
        - nlp: models/nlp/model.gguf (for BenderBox's chat features)
        - code: models/code/model.gguf (for code generation)

        Args:
            model_path: Path to the model file.
            purpose: One of "analysis", "nlp", or "code". Default is "analysis".

        Returns:
            Tuple of (success, message).
        """
        if not model_path.exists():
            return False, f"Model not found: {model_path}"

        try:
            target_dir = self.get_model_dir_for_purpose(purpose)
        except ValueError as e:
            return False, str(e)

        # Create target directory
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / "model.gguf"

        # Remove existing if present
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()

        try:
            # Try to create symlink first (saves space)
            target_path.symlink_to(model_path.resolve())
            return True, f"Linked {model_path.name} as default {purpose} model"
        except OSError:
            # Fall back to copy on Windows if symlinks not supported
            try:
                shutil.copy2(model_path, target_path)
                return True, f"Copied {model_path.name} as default {purpose} model"
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
                cache_path=self.data_models_dir,
            )

            # Download from HuggingFace
            # Format target as repo/filename for specific file download
            if progress_callback:
                progress_callback(f"Downloading {model.name}...", 0)

            target = f"{model.huggingface_repo}/{model.filename}"
            result = await handler.resolve(target)

            if result.local_path and result.local_path.exists():
                if progress_callback:
                    progress_callback(f"Downloaded {model.name}", 100)
                return True, f"Downloaded {model.name}", result.local_path
            else:
                return False, "Download failed: file not found after download", None

        except ImportError as e:
            return False, f"Missing dependency: {e}", None
        except Exception as e:
            return False, f"Download error: {e}", None

    def get_system_ram_gb(self) -> int:
        """
        Get system RAM in GB using multiple detection methods.

        Uses cross-validation to ensure accurate detection on Windows.
        Falls back gracefully if dependencies are missing.
        """
        detected_values = []

        # Method 1: psutil (cross-platform, most reliable)
        try:
            import psutil
            ram_bytes = psutil.virtual_memory().total
            ram_gb = ram_bytes // (1024 ** 3)
            detected_values.append(("psutil", ram_gb))
            logger.debug(f"psutil detected {ram_gb}GB RAM ({ram_bytes} bytes)")
        except ImportError:
            logger.debug("psutil not available for RAM detection")
        except Exception as e:
            logger.debug(f"psutil RAM detection failed: {e}")

        # Method 2: Windows-specific via ctypes (fallback)
        if os.name == "nt":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                mem_status = MEMORYSTATUSEX()
                mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
                    ram_bytes = mem_status.ullTotalPhys
                    ram_gb = ram_bytes // (1024 ** 3)
                    detected_values.append(("ctypes", ram_gb))
                    logger.debug(f"ctypes detected {ram_gb}GB RAM ({ram_bytes} bytes)")
            except Exception as e:
                logger.debug(f"ctypes RAM detection failed: {e}")

        # Cross-validate results
        if detected_values:
            # Log all detected values for debugging
            for method, value in detected_values:
                logger.debug(f"RAM detection [{method}]: {value}GB")

            # If we have multiple values, check for consistency
            if len(detected_values) > 1:
                values = [v for _, v in detected_values]
                # If values differ significantly (>10%), prefer ctypes on Windows
                if max(values) > min(values) * 1.1:
                    logger.warning(
                        f"RAM detection inconsistency: {detected_values}. "
                        "Using highest value."
                    )
                    return max(values)

            # Return the first (most reliable) value
            return detected_values[0][1]

        # Fallback: conservative default
        logger.warning("Could not detect system RAM, defaulting to 8GB")
        return 8

    def get_system_ram_info(self) -> Dict:
        """
        Get detailed RAM information for diagnostics.

        Returns:
            Dictionary with RAM details from all detection methods.
        """
        info = {
            "detected_gb": self.get_system_ram_gb(),
            "methods": {},
        }

        # psutil method
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["methods"]["psutil"] = {
                "total_bytes": mem.total,
                "total_gb": mem.total // (1024 ** 3),
                "available_gb": mem.available // (1024 ** 3),
                "percent_used": mem.percent,
            }
        except Exception as e:
            info["methods"]["psutil"] = {"error": str(e)}

        # Windows ctypes method
        if os.name == "nt":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                mem_status = MEMORYSTATUSEX()
                mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
                    info["methods"]["ctypes"] = {
                        "total_bytes": mem_status.ullTotalPhys,
                        "total_gb": mem_status.ullTotalPhys // (1024 ** 3),
                        "available_gb": mem_status.ullAvailPhys // (1024 ** 3),
                        "memory_load": mem_status.dwMemoryLoad,
                    }
            except Exception as e:
                info["methods"]["ctypes"] = {"error": str(e)}

        return info


def get_model_manager(base_path: Optional[Path] = None) -> ModelManager:
    """Get a ModelManager instance."""
    return ModelManager(base_path)
