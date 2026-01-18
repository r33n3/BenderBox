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
            List of model info dictionaries (deduplicated by filename).
        """
        models = []
        seen_names = set()  # Track filenames to avoid duplicates

        # Check models/ directory
        if self.models_dir.exists():
            for gguf_file in self.models_dir.rglob("*.gguf"):
                if gguf_file.name not in seen_names:
                    seen_names.add(gguf_file.name)
                    models.append({
                        "path": str(gguf_file),
                        "name": gguf_file.name,
                        "size_mb": gguf_file.stat().st_size // (1024 * 1024),
                        "location": "models/",
                    })

        # Check data/models/ directory (HuggingFace downloads)
        if self.data_models_dir.exists():
            for gguf_file in self.data_models_dir.rglob("*.gguf"):
                if gguf_file.name not in seen_names:
                    seen_names.add(gguf_file.name)
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

    def get_default_model_path(self, purpose: str = "analysis") -> Optional[Path]:
        """
        Get the path to the default model for a given purpose.

        Checks in order:
        1. Model path/name specified in default.txt marker file
        2. Legacy models/analysis/model.gguf (for backwards compatibility)
        3. Any .gguf file in the purpose directory
        4. Any downloaded TinyLlama model
        5. Any downloaded .gguf file

        Args:
            purpose: One of "analysis", "nlp", or "code". Default is "analysis".

        Returns:
            Path to model file, or None if not found.
        """
        try:
            model_dir = self.get_model_dir_for_purpose(purpose)
        except ValueError:
            model_dir = self.analysis_model_dir

        # Check marker file first
        default_value = self._get_default_marker(model_dir)
        if default_value:
            # default.txt may contain a full path or just a filename
            default_path = Path(default_value)
            if default_path.is_absolute() and default_path.exists():
                # Full path stored in marker file
                return default_path
            # Legacy: relative filename in the purpose directory
            relative_path = model_dir / default_value
            if relative_path.exists():
                return relative_path

        # Legacy: check for model.gguf (backwards compatibility)
        legacy_path = model_dir / "model.gguf"
        if legacy_path.exists():
            return legacy_path

        # Check for any model in the purpose directory
        if model_dir.exists():
            for gguf in model_dir.glob("*.gguf"):
                return gguf

        # Only for analysis purpose: fall back to downloaded models
        if purpose == "analysis":
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
        self, model_path: Path, purpose: str = "analysis", use_link: bool = True
    ) -> Tuple[bool, str]:
        """
        Set up a model as the default for a specific purpose.

        Simply stores the full path to the model in a 'default.txt' marker file.
        No copying or symlinking - this saves disk space and avoids duplicates.

        Args:
            model_path: Path to the model file.
            purpose: One of "analysis", "nlp", or "code". Default is "analysis".
            use_link: Ignored (kept for API compatibility).

        Returns:
            Tuple of (success, message).
        """
        if not model_path.exists():
            return False, f"Model not found: {model_path}"

        try:
            target_dir = self.get_model_dir_for_purpose(purpose)
        except ValueError as e:
            return False, str(e)

        # Create target directory for the marker file
        target_dir.mkdir(parents=True, exist_ok=True)

        # Store the full path to the model in default.txt (no copying needed)
        self._set_default_marker(target_dir, str(model_path.resolve()))
        return True, f"Set {model_path.name} as default {purpose} model"

    def _set_default_marker(self, model_dir: Path, model_filename: str) -> None:
        """
        Set a marker file indicating which model is the default.

        Args:
            model_dir: Directory containing the models.
            model_filename: Filename of the model to set as default.
        """
        marker_path = model_dir / "default.txt"
        marker_path.write_text(model_filename, encoding="utf-8")

    def _get_default_marker(self, model_dir: Path) -> Optional[str]:
        """
        Get the default model filename from marker file.

        Args:
            model_dir: Directory containing the models.

        Returns:
            Filename of default model, or None if not set.
        """
        marker_path = model_dir / "default.txt"
        if marker_path.exists():
            return marker_path.read_text(encoding="utf-8").strip()
        return None

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


    def get_gpu_info(self) -> Dict:
        """
        Detect GPU information including NVIDIA/CUDA support.

        Returns:
            Dictionary with GPU details:
            - has_nvidia: bool
            - gpu_name: str or None
            - vram_gb: int or None
            - cuda_available: bool
            - recommended_layers: int
        """
        info = {
            "has_nvidia": False,
            "gpu_name": None,
            "vram_gb": None,
            "cuda_available": False,
            "recommended_layers": 0,
        }

        # Method 1: Check nvidia-smi (works without Python CUDA bindings)
        if shutil.which("nvidia-smi"):
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split("\n")
                    if lines:
                        parts = lines[0].split(", ")
                        info["has_nvidia"] = True
                        info["gpu_name"] = parts[0].strip() if len(parts) > 0 else "Unknown NVIDIA"
                        if len(parts) > 1:
                            try:
                                vram_mb = int(parts[1].strip())
                                info["vram_gb"] = vram_mb // 1024
                            except ValueError:
                                pass
                        logger.debug(f"nvidia-smi detected: {info['gpu_name']}, {info['vram_gb']}GB VRAM")
            except Exception as e:
                logger.debug(f"nvidia-smi detection failed: {e}")

        # Method 2: Check if CUDA is available via llama-cpp-python
        try:
            from llama_cpp import llama_supports_gpu_offload
            info["cuda_available"] = llama_supports_gpu_offload()
        except ImportError:
            # llama-cpp-python not installed
            pass
        except Exception as e:
            logger.debug(f"CUDA detection via llama-cpp failed: {e}")

        # Method 3: Try torch (if installed for other features)
        if not info["has_nvidia"]:
            try:
                import torch
                if torch.cuda.is_available():
                    info["has_nvidia"] = True
                    info["cuda_available"] = True
                    info["gpu_name"] = torch.cuda.get_device_name(0)
                    vram_bytes = torch.cuda.get_device_properties(0).total_memory
                    info["vram_gb"] = vram_bytes // (1024 ** 3)
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"torch CUDA detection failed: {e}")

        # Calculate recommended GPU layers based on VRAM
        if info["vram_gb"]:
            # Rule of thumb: ~1GB VRAM per 4-8 layers for 7B model
            # Conservative estimate for safety
            info["recommended_layers"] = min(info["vram_gb"] * 4, 35)

        return info

    def get_cpu_info(self) -> Dict:
        """
        Get CPU information.

        Returns:
            Dictionary with CPU details.
        """
        import platform

        info = {
            "processor": platform.processor() or "Unknown",
            "cores": 1,
            "threads": 1,
        }

        try:
            import psutil
            info["cores"] = psutil.cpu_count(logical=False) or 1
            info["threads"] = psutil.cpu_count(logical=True) or 1
        except ImportError:
            try:
                import os
                info["threads"] = os.cpu_count() or 1
            except Exception:
                pass

        return info

    def extract_parameter_count(self, filename: str) -> Optional[float]:
        """
        Extract parameter count (in billions) from model filename.

        Recognizes patterns like:
        - llama-2-7b, llama-7b, llama2-7b
        - mistral-7b-instruct
        - phi-2 (implies 2.7B), phi-3
        - tinyllama-1.1b
        - mixtral-8x7b (MoE models)
        - 70b, 13b, 30b, 120b

        Args:
            filename: Model filename or path.

        Returns:
            Parameter count in billions, or None if not detected.
        """
        import re

        name = Path(filename).stem.lower()

        # Pattern for explicit billion params: 7b, 13b, 70b, 1.1b, etc.
        patterns = [
            r'(\d+\.?\d*)b(?:illion)?(?:[-_\.]|$)',  # 7b, 13b, 70b, 1.1b
            r'[-_](\d+\.?\d*)b[-_]',  # -7b- or _7b_
            r'^(\d+\.?\d*)b[-_]',  # 7b- at start
            r'(\d+)x(\d+)b',  # MoE: 8x7b = 56b effective
        ]

        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:  # MoE pattern
                    return float(groups[0]) * float(groups[1])
                return float(groups[0])

        # Special cases for known model families
        known_params = {
            'phi-2': 2.7,
            'phi2': 2.7,
            'phi-3-mini': 3.8,
            'phi3-mini': 3.8,
            'phi-3-small': 7.0,
            'phi-3-medium': 14.0,
            'tinyllama': 1.1,
            'tiny-llama': 1.1,
            'gemma-2b': 2.0,
            'gemma-7b': 7.0,
            'qwen-1.5': 1.5,
            'stablelm-2': 1.6,
        }

        for key, params in known_params.items():
            if key in name:
                return params

        return None

    def estimate_ram_requirement(
        self,
        file_size_mb: int,
        param_billions: Optional[float] = None
    ) -> Dict:
        """
        Estimate RAM requirements for a model.

        Uses both file size and parameter count for better estimates.
        GGUF models typically need 1.1-1.3x file size in RAM.
        Parameter count helps estimate context memory needs.

        Args:
            file_size_mb: Model file size in MB.
            param_billions: Parameter count in billions (optional).

        Returns:
            Dictionary with:
            - estimated_ram_gb: Estimated RAM needed
            - min_ram_gb: Minimum RAM to attempt loading
            - recommended_ram_gb: Recommended RAM for smooth operation
            - confidence: "high", "medium", or "low"
        """
        # Base estimate from file size (1.2x multiplier for loading overhead)
        base_ram_gb = (file_size_mb * 1.2) / 1024

        # If we have parameter count, refine the estimate
        # Larger models need more context memory
        if param_billions:
            # Context memory scales with parameters
            # ~0.5GB per billion parameters for context at 2048 tokens
            context_overhead_gb = param_billions * 0.5
            estimated_ram_gb = base_ram_gb + context_overhead_gb
            confidence = "high"
        else:
            estimated_ram_gb = base_ram_gb * 1.1  # Add 10% uncertainty
            confidence = "medium" if file_size_mb > 1000 else "low"

        # Calculate min and recommended
        min_ram_gb = max(estimated_ram_gb * 0.9, 2)  # At least 2GB
        recommended_ram_gb = estimated_ram_gb * 1.3  # 30% headroom

        return {
            "estimated_ram_gb": round(estimated_ram_gb, 1),
            "min_ram_gb": round(min_ram_gb, 1),
            "recommended_ram_gb": round(recommended_ram_gb, 1),
            "confidence": confidence,
            "param_billions": param_billions,
            "file_size_mb": file_size_mb,
        }

    def get_model_requirements(self, model_path: str) -> Dict:
        """
        Get complete requirements info for a model.

        Args:
            model_path: Path to model file.

        Returns:
            Dictionary with size, parameters, RAM requirements, and warnings.
        """
        path = Path(model_path)

        if not path.exists():
            return {"error": f"Model not found: {model_path}"}

        file_size_mb = path.stat().st_size // (1024 * 1024)
        param_billions = self.extract_parameter_count(path.name)
        ram_req = self.estimate_ram_requirement(file_size_mb, param_billions)

        # Check against system RAM
        system_ram_gb = self.get_system_ram_gb()
        available_ram_gb = max(system_ram_gb - 4, 2)  # Leave 4GB for OS

        warnings = []
        can_load = True

        if ram_req["estimated_ram_gb"] > available_ram_gb:
            warnings.append(
                f"Model needs ~{ram_req['estimated_ram_gb']}GB RAM, "
                f"but only {available_ram_gb}GB available"
            )
            can_load = False
        elif ram_req["estimated_ram_gb"] > available_ram_gb * 0.8:
            warnings.append(
                f"Model will use most available RAM ({ram_req['estimated_ram_gb']}GB of {available_ram_gb}GB)"
            )

        # Parameter-based warnings
        if param_billions:
            if param_billions >= 30:
                warnings.append(f"{param_billions}B model - requires high-end system or GPU offloading")
            elif param_billions >= 13:
                warnings.append(f"{param_billions}B model - may be slow without GPU")

        return {
            "name": path.stem,
            "filename": path.name,
            "path": str(path),
            "file_size_mb": file_size_mb,
            "param_billions": param_billions,
            "param_display": f"{param_billions}B" if param_billions else "Unknown",
            "ram_requirements": ram_req,
            "system_ram_gb": system_ram_gb,
            "available_ram_gb": available_ram_gb,
            "can_load": can_load,
            "warnings": warnings,
        }

    def get_system_assessment(self) -> Dict:
        """
        Comprehensive system assessment for model selection.

        Returns:
            Dictionary with full system assessment including:
            - RAM, GPU, CPU info
            - Model recommendations for NLP vs analysis
            - Warnings and notes
        """
        ram_gb = self.get_system_ram_gb()
        gpu_info = self.get_gpu_info()
        cpu_info = self.get_cpu_info()

        assessment = {
            "ram_gb": ram_gb,
            "gpu": gpu_info,
            "cpu": cpu_info,
            "nlp_recommendations": [],
            "analysis_recommendations": [],
            "warnings": [],
            "notes": [],
        }

        # NLP model recommendations (stays loaded in memory)
        # Need to leave room for OS and other processes (~4GB overhead)
        available_for_nlp = max(ram_gb - 4, 2)

        for model_id, model in RECOMMENDED_MODELS.items():
            model_info = {
                "id": model_id,
                "name": model.name,
                "size_mb": model.size_mb,
                "min_ram_gb": model.min_ram_gb,
                "quality": model.quality,
                "fits": model.min_ram_gb <= available_for_nlp,
            }

            # NLP models need RAM headroom since they stay loaded
            if model.min_ram_gb <= available_for_nlp:
                assessment["nlp_recommendations"].append(model_info)

            # Analysis models can use more RAM since they're loaded temporarily
            if model.min_ram_gb <= ram_gb:
                assessment["analysis_recommendations"].append(model_info)

        # Sort by quality (best first)
        quality_order = {"best": 0, "good": 1, "basic": 2}
        assessment["nlp_recommendations"].sort(
            key=lambda x: (quality_order.get(x["quality"], 99), -x["size_mb"])
        )
        assessment["analysis_recommendations"].sort(
            key=lambda x: (quality_order.get(x["quality"], 99), -x["size_mb"])
        )

        # Generate warnings
        if ram_gb < 4:
            assessment["warnings"].append(
                f"Low RAM detected ({ram_gb}GB). Only smallest models will work."
            )
        elif ram_gb < 8:
            assessment["warnings"].append(
                f"Limited RAM ({ram_gb}GB). Larger models may be slow or fail."
            )

        # GPU notes
        if gpu_info["has_nvidia"]:
            assessment["notes"].append(
                f"NVIDIA GPU detected: {gpu_info['gpu_name']} ({gpu_info['vram_gb']}GB VRAM)"
            )
            if gpu_info["cuda_available"]:
                assessment["notes"].append(
                    f"GPU acceleration available. Recommended layers: {gpu_info['recommended_layers']}"
                )
            else:
                assessment["notes"].append(
                    "Install llama-cpp-python with CUDA support for GPU acceleration."
                )
        else:
            assessment["notes"].append(
                "No NVIDIA GPU detected. Models will run on CPU only."
            )

        return assessment

    def get_best_model_for_system(self, purpose: str = "nlp") -> Optional[str]:
        """
        Get the best model ID for this system based on hardware.

        Args:
            purpose: "nlp" (stays loaded) or "analysis" (temporary)

        Returns:
            Model ID string or None if no suitable model.
        """
        assessment = self.get_system_assessment()

        if purpose == "nlp":
            recommendations = assessment["nlp_recommendations"]
        else:
            recommendations = assessment["analysis_recommendations"]

        if recommendations:
            return recommendations[0]["id"]
        return None

    def download_huggingface_model_sync(
        self,
        repo_id: str,
        filename: str,
        purpose: str = "nlp",
        show_progress: bool = True,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Synchronously download a GGUF model from HuggingFace with progress.

        Args:
            repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            filename: GGUF filename (e.g., "llama-2-7b.Q4_K_M.gguf")
            purpose: "nlp", "analysis", or "code"
            show_progress: Whether to show download progress bar

        Returns:
            Tuple of (success, message, path_if_success)
        """
        try:
            from huggingface_hub import hf_hub_download

            # Determine target directory based on purpose
            if purpose == "nlp":
                target_dir = self.nlp_model_dir
            elif purpose == "analysis":
                target_dir = self.analysis_model_dir
            elif purpose == "code":
                target_dir = self.code_model_dir
            else:
                target_dir = self.models_dir

            target_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {repo_id}/{filename}...")

            # Download with progress display
            # huggingface_hub shows its own progress by default
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.data_models_dir / "huggingface",
                local_dir_use_symlinks=False,
            )

            # Copy to purpose-specific directory
            final_path = target_dir / filename
            if not final_path.exists():
                if show_progress:
                    print(f"Copying to {target_dir.name}/...")
                shutil.copy2(downloaded_path, final_path)

            return True, f"Downloaded to {final_path}", final_path

        except ImportError:
            return False, "huggingface_hub not installed. Run: pip install huggingface-hub", None
        except Exception as e:
            return False, f"Download failed: {e}", None

    async def download_huggingface_model(
        self,
        repo_id: str,
        filename: str,
        purpose: str = "nlp",
        progress_callback=None,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Async wrapper for download_huggingface_model_sync.

        Args:
            repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            filename: GGUF filename (e.g., "llama-2-7b.Q4_K_M.gguf")
            purpose: "nlp", "analysis", or "code"
            progress_callback: Optional callback for download progress

        Returns:
            Tuple of (success, message, path_if_success)
        """
        import asyncio
        return await asyncio.to_thread(
            self.download_huggingface_model_sync,
            repo_id, filename, purpose, progress_callback is None
        )

    def parse_huggingface_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse a HuggingFace URL to extract repo_id and filename.

        Args:
            url: HuggingFace URL like:
                 - https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf
                 - TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf
                 - TheBloke/Llama-2-7B-GGUF

        Returns:
            Tuple of (repo_id, filename) or (None, None) if invalid
        """
        import re

        # Remove common URL prefixes
        url = url.strip()
        url = re.sub(r'^https?://huggingface\.co/', '', url)
        url = re.sub(r'^hf://', '', url)

        # Handle blob/main or resolve/main in path
        url = re.sub(r'/blob/main/', '/', url)
        url = re.sub(r'/resolve/main/', '/', url)

        parts = url.split('/')

        if len(parts) >= 2:
            repo_id = f"{parts[0]}/{parts[1]}"

            # Look for .gguf filename
            filename = None
            for part in parts[2:]:
                if part.endswith('.gguf'):
                    filename = part
                    break

            return repo_id, filename

        return None, None

    def validate_gguf_file(self, file_path: Path) -> Tuple[bool, str, Dict]:
        """
        Validate that a file is a valid GGUF model by checking the magic bytes.

        GGUF files start with magic bytes: GGUF (0x47 0x47 0x55 0x46)
        followed by version number (uint32).

        Args:
            file_path: Path to the file to validate

        Returns:
            Tuple of (is_valid, message, metadata_dict)
        """
        import struct

        GGUF_MAGIC = b'GGUF'

        metadata = {
            "file_path": str(file_path),
            "file_size_mb": 0,
            "is_gguf": False,
            "gguf_version": None,
        }

        try:
            if not file_path.exists():
                return False, f"File not found: {file_path}", metadata

            file_size = file_path.stat().st_size
            metadata["file_size_mb"] = round(file_size / (1024 * 1024), 2)

            if file_size < 8:
                return False, "File too small to be a GGUF model", metadata

            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic != GGUF_MAGIC:
                    # Check for other common model formats
                    f.seek(0)
                    header = f.read(16)

                    if header.startswith(b'PK'):
                        return False, "File appears to be a ZIP archive (possibly SafeTensors or PyTorch)", metadata
                    elif header.startswith(b'\x89PNG'):
                        return False, "File is a PNG image, not a model", metadata
                    elif b'pytorch' in header.lower() or header[:2] == b'\x80\x02':
                        return False, "File appears to be a PyTorch model (.pt/.pth), not GGUF", metadata

                    return False, f"Not a GGUF file (magic bytes: {magic.hex()})", metadata

                # Read GGUF version (uint32, little endian)
                version_bytes = f.read(4)
                if len(version_bytes) == 4:
                    version = struct.unpack('<I', version_bytes)[0]
                    metadata["gguf_version"] = version
                    metadata["is_gguf"] = True

                    if version < 1 or version > 3:
                        return False, f"Unsupported GGUF version: {version}", metadata

            return True, f"Valid GGUF file (version {metadata['gguf_version']}, {metadata['file_size_mb']} MB)", metadata

        except Exception as e:
            return False, f"Error validating file: {e}", metadata

    def is_direct_download_url(self, url: str) -> bool:
        """
        Check if a URL is a direct download link (not a HuggingFace repo page).

        Returns True for URLs that point directly to files.
        """
        url = url.lower().strip()

        # Direct file URLs typically end with file extensions
        direct_extensions = ['.gguf', '.bin', '.safetensors', '.pt', '.pth', '.onnx']
        if any(url.endswith(ext) for ext in direct_extensions):
            return True

        # HuggingFace resolve URLs are direct downloads
        if '/resolve/' in url:
            return True

        # Check for common CDN/file hosting patterns
        direct_patterns = [
            'github.com' in url and '/releases/download/' in url,
            'raw.githubusercontent.com' in url,
            '/download/' in url and any(ext in url for ext in direct_extensions),
        ]

        return any(direct_patterns)

    def download_from_url(
        self,
        url: str,
        purpose: str = "analysis",
        filename: Optional[str] = None,
        show_progress: bool = True,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Download a model file from any URL.

        Args:
            url: Direct download URL
            purpose: "nlp", "analysis", or "code"
            filename: Optional filename (auto-detected from URL if not provided)
            show_progress: Whether to show download progress

        Returns:
            Tuple of (success, message, path_if_success)
        """
        import re
        import urllib.request
        import urllib.error

        # Convert HuggingFace /blob/ URLs to /resolve/ URLs
        # /blob/ returns HTML page, /resolve/ returns actual file
        if 'huggingface.co' in url and '/blob/' in url:
            url = url.replace('/blob/', '/resolve/')
            logger.info(f"Converted HuggingFace blob URL to resolve URL")

        # Determine target directory based on purpose
        if purpose == "nlp":
            target_dir = self.nlp_model_dir
        elif purpose == "analysis":
            target_dir = self.analysis_model_dir
        elif purpose == "code":
            target_dir = self.code_model_dir
        else:
            target_dir = self.models_dir

        target_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename from URL if not provided
        if not filename:
            # Try to get filename from URL path
            url_path = url.split('?')[0]  # Remove query params
            filename = url_path.split('/')[-1]

            # If filename doesn't look valid, generate one
            if not filename or '.' not in filename:
                filename = f"model_{hash(url) % 10000}.gguf"

        # Ensure .gguf extension
        if not filename.endswith('.gguf'):
            filename = filename + '.gguf'

        target_path = target_dir / filename
        temp_path = target_dir / f".{filename}.download"

        logger.info(f"Downloading {url} to {target_path}")

        try:
            # Set up request with user agent
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'BenderBox/4.0 (Model Downloader)',
                }
            )

            # Download with progress
            with urllib.request.urlopen(req, timeout=300) as response:
                total_size = response.headers.get('content-length')
                if total_size:
                    total_size = int(total_size)

                downloaded = 0
                block_size = 8192

                with open(temp_path, 'wb') as f:
                    while True:
                        block = response.read(block_size)
                        if not block:
                            break
                        f.write(block)
                        downloaded += len(block)

                        if show_progress and total_size:
                            pct = (downloaded / total_size) * 100
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r  Downloading: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.1f}%)", end='', flush=True)

                if show_progress:
                    print()  # Newline after progress

            # Validate the downloaded file
            is_valid, msg, metadata = self.validate_gguf_file(temp_path)

            if not is_valid:
                temp_path.unlink(missing_ok=True)
                return False, f"Downloaded file is not a valid GGUF model: {msg}", None

            # Move to final location
            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)

            return True, f"Downloaded {metadata['file_size_mb']} MB to {target_path}", target_path

        except urllib.error.HTTPError as e:
            temp_path.unlink(missing_ok=True)
            return False, f"HTTP error {e.code}: {e.reason}", None
        except urllib.error.URLError as e:
            temp_path.unlink(missing_ok=True)
            return False, f"URL error: {e.reason}", None
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            return False, f"Download failed: {e}", None

    async def download_from_url_async(
        self,
        url: str,
        purpose: str = "analysis",
        filename: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Path]]:
        """Async wrapper for download_from_url."""
        import asyncio
        return await asyncio.to_thread(
            self.download_from_url, url, purpose, filename, True
        )


def get_model_manager(base_path: Optional[Path] = None) -> ModelManager:
    """Get a ModelManager instance."""
    return ModelManager(base_path)
