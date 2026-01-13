"""
Local LLM Engine for BenderBox

Provides local LLM inference using llama-cpp-python with:
- Lazy model loading
- LRU eviction for memory management
- Async generation support
- Graceful handling of missing models
"""

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from benderbox.config import LLMConfig, get_config

logger = logging.getLogger(__name__)


def is_llama_cpp_available() -> bool:
    """Check if llama-cpp-python is installed and importable."""
    try:
        from llama_cpp import Llama
        return True
    except ImportError:
        return False


# Cache the availability check
_LLAMA_CPP_AVAILABLE: Optional[bool] = None


def check_llama_cpp_available() -> bool:
    """Check availability with caching."""
    global _LLAMA_CPP_AVAILABLE
    if _LLAMA_CPP_AVAILABLE is None:
        _LLAMA_CPP_AVAILABLE = is_llama_cpp_available()
    return _LLAMA_CPP_AVAILABLE


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    path: str
    context_length: int
    loaded: bool
    memory_usage_mb: float = 0.0


class LlamaModel:
    """
    Wrapper for llama-cpp-python Llama model.

    Provides a simplified interface for text generation with
    proper resource management.
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 4096,
        threads: int = 4,
        gpu_layers: int = 0,
    ):
        """
        Initialize LlamaModel wrapper.

        Args:
            model_path: Path to GGUF model file.
            context_length: Context window size in tokens.
            threads: Number of CPU threads for inference.
            gpu_layers: Number of layers to offload to GPU (0 = CPU only).
        """
        self.model_path = model_path
        self.context_length = context_length
        self.threads = threads
        self.gpu_layers = gpu_layers
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded, raising informative error if not possible."""
        if self._loaded and self._model is not None:
            return

        path = Path(self.model_path)
        if not path.exists():
            raise ModelNotFoundError(
                f"Model file not found: {self.model_path}\n\n"
                f"To use BenderBox's NLP features, download a model:\n\n"
                f"  python bb.py models list              # See available models for your system\n"
                f"  python bb.py models download <id>     # Download a model\n"
                f"  python bb.py models setup             # Configure default model\n\n"
                f"Quick start (recommended for most systems):\n"
                f"  python bb.py models download tinyllama\n"
                f"  python bb.py models setup"
            )

        try:
            from llama_cpp import Llama

            logger.info(f"Loading model from {self.model_path}")
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_length,
                n_threads=self.threads,
                n_gpu_layers=self.gpu_layers,
                verbose=False,
            )
            self._loaded = True
            logger.info(f"Model loaded successfully: {self.model_path}")
        except ImportError:
            raise LLMEngineError(
                "llama-cpp-python is not installed. Install with:\n"
                "  pip install llama-cpp-python\n\n"
                "For GPU support, see: https://github.com/abetlen/llama-cpp-python#installation"
            )
        except Exception as e:
            raise LLMEngineError(f"Failed to load model: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text completion for the given prompt.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            top_p: Nucleus sampling threshold.
            stop: List of stop sequences.

        Returns:
            Generated text completion.
        """
        self._ensure_loaded()

        result = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
        )

        return result["choices"][0]["text"]

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ):
        """
        Generate text completion with streaming output.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: List of stop sequences.

        Yields:
            Generated text chunks.
        """
        self._ensure_loaded()

        for chunk in self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
            stream=True,
        ):
            yield chunk["choices"][0]["text"]

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs.

        Args:
            text: Input text.

        Returns:
            List of token IDs.
        """
        self._ensure_loaded()
        return self._model.tokenize(text.encode("utf-8"))

    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.
        """
        self._ensure_loaded()
        return self._model.detokenize(tokens).decode("utf-8")

    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        logger.info(f"Model unloaded: {self.model_path}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._loaded


class LocalLLMEngine:
    """
    Local LLM Engine for BenderBox.

    Manages multiple LLM models with:
    - Lazy loading on first use
    - LRU eviction when memory constrained
    - Async interface for non-blocking operations
    """

    MODEL_TYPES = {"analysis", "code"}

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LocalLLMEngine.

        Args:
            config: LLM configuration. Uses global config if not provided.
        """
        if config is None:
            config = get_config().llm
        self.config = config

        # Check if llama-cpp-python is available
        self._llama_available = check_llama_cpp_available()
        if not self._llama_available:
            logger.warning(
                "llama-cpp-python is not installed. LLM features will be disabled. "
                "Install with: pip install llama-cpp-python"
            )

        # Model instances (LRU ordered)
        self._loaded_models: OrderedDict[str, LlamaModel] = OrderedDict()

        # Model paths
        self._model_paths = {
            "analysis": config.analysis_model_path,
            "code": config.code_model_path,
        }

    @property
    def is_available(self) -> bool:
        """Check if the LLM engine has llama-cpp-python available."""
        return self._llama_available

    def _get_or_create_model(self, model_type: str) -> LlamaModel:
        """Get or create a model instance."""
        if not self._llama_available:
            raise LLMEngineError(
                "llama-cpp-python is not installed. Install with:\n"
                "  pip install llama-cpp-python\n\n"
                "For GPU support, see: https://github.com/abetlen/llama-cpp-python#installation"
            )

        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Valid types: {self.MODEL_TYPES}")

        # Check if already loaded
        if model_type in self._loaded_models:
            # Move to end (most recently used)
            self._loaded_models.move_to_end(model_type)
            return self._loaded_models[model_type]

        # Check if we need to evict
        while len(self._loaded_models) >= self.config.max_loaded_models:
            # Evict least recently used
            oldest_type, oldest_model = self._loaded_models.popitem(last=False)
            oldest_model.unload()
            logger.info(f"Evicted model due to memory limit: {oldest_type}")

        # Create new model
        model_path = self._model_paths[model_type]
        model = LlamaModel(
            model_path=model_path,
            context_length=self.config.context_length,
            threads=self.config.threads,
            gpu_layers=self.config.gpu_layers,
        )

        self._loaded_models[model_type] = model
        return model

    async def load_model(self, model_type: str) -> None:
        """
        Pre-load a model into memory.

        Args:
            model_type: Type of model to load ("analysis" or "code").
        """
        model = self._get_or_create_model(model_type)
        # Trigger lazy loading in thread pool
        await asyncio.get_event_loop().run_in_executor(
            None, model._ensure_loaded
        )

    async def unload_model(self, model_type: str) -> None:
        """
        Unload a model from memory.

        Args:
            model_type: Type of model to unload.
        """
        if model_type in self._loaded_models:
            model = self._loaded_models.pop(model_type)
            await asyncio.get_event_loop().run_in_executor(
                None, model.unload
            )

    async def generate(
        self,
        prompt: str,
        model_type: str = "analysis",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text completion asynchronously.

        Args:
            prompt: Input prompt text.
            model_type: Model to use ("analysis" or "code").
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: List of stop sequences.

        Returns:
            Generated text completion.
        """
        model = self._get_or_create_model(model_type)

        # Use config defaults if not specified
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        # Run generation in thread pool
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            ),
        )

    async def generate_stream(
        self,
        prompt: str,
        model_type: str = "analysis",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """
        Generate text completion with streaming output.

        Args:
            prompt: Input prompt text.
            model_type: Model to use.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: List of stop sequences.

        Yields:
            Generated text chunks.
        """
        model = self._get_or_create_model(model_type)

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p

        # Create sync generator
        def sync_stream():
            yield from model.generate_stream(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )

        # Wrap in async iterator
        loop = asyncio.get_event_loop()
        gen = sync_stream()

        while True:
            try:
                chunk = await loop.run_in_executor(None, next, gen)
                yield chunk
            except StopIteration:
                break

    async def is_model_loaded(self, model_type: str) -> bool:
        """
        Check if a model is currently loaded.

        Args:
            model_type: Model type to check.

        Returns:
            True if model is loaded and ready.
        """
        if model_type not in self._loaded_models:
            return False
        return self._loaded_models[model_type].is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about available and loaded models.

        Returns:
            Dictionary with model status information.
        """
        info = {
            "loaded_models": list(self._loaded_models.keys()),
            "max_loaded_models": self.config.max_loaded_models,
            "models": {},
        }

        for model_type, path in self._model_paths.items():
            model_exists = Path(path).exists()
            is_loaded = model_type in self._loaded_models

            info["models"][model_type] = {
                "path": path,
                "exists": model_exists,
                "loaded": is_loaded,
            }

        return info

    def unload_all(self) -> None:
        """Unload all models from memory."""
        for model_type in list(self._loaded_models.keys()):
            model = self._loaded_models.pop(model_type)
            model.unload()


class LLMEngineError(Exception):
    """Base exception for LLM engine errors."""

    pass


class ModelNotFoundError(LLMEngineError):
    """Raised when a model file is not found."""

    pass
