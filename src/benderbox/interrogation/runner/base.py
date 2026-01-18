"""
Base model runner protocol and common types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


class FinishReason(Enum):
    """Reason why generation finished."""

    STOP = "stop"  # Normal completion
    LENGTH = "length"  # Max tokens reached
    TIMEOUT = "timeout"  # Timed out
    ERROR = "error"  # Error occurred


@dataclass
class GenerationResult:
    """Result of a model generation."""

    response: str
    finish_reason: FinishReason
    tokens_generated: int = 0
    generation_time: float = 0.0
    prompt_tokens: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if generation was successful."""
        return self.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "response": self.response,
            "finish_reason": self.finish_reason.value,
            "tokens_generated": self.tokens_generated,
            "generation_time": self.generation_time,
            "prompt_tokens": self.prompt_tokens,
            "error": self.error,
        }


@dataclass
class RunnerConfig:
    """Configuration for model runner."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    timeout_seconds: float = 60.0
    gpu_layers: int = 0  # 0 = CPU only
    context_size: int = 2048
    threads: int = 4
    seed: int = -1  # -1 = random

    def to_cli_args(self) -> list:
        """Convert to llama-cli arguments."""
        args = [
            "-n", str(self.max_tokens),
            "--temp", str(self.temperature),
            "--top-p", str(self.top_p),
            "--top-k", str(self.top_k),
            "--repeat-penalty", str(self.repeat_penalty),
            "-ngl", str(self.gpu_layers),
            "-c", str(self.context_size),
            "-t", str(self.threads),
            "--log-disable",  # Suppress llama.cpp logging (context size warnings, etc.)
        ]
        if self.seed >= 0:
            args.extend(["-s", str(self.seed)])
        return args


@runtime_checkable
class ModelRunner(Protocol):
    """Protocol for model runners."""

    @property
    def model_path(self) -> Path:
        """Path to the model file."""
        ...

    @property
    def is_ready(self) -> bool:
        """Check if runner is ready to generate."""
        ...

    async def generate(
        self,
        prompt: str,
        config: Optional[RunnerConfig] = None,
    ) -> GenerationResult:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt
            config: Generation configuration (uses defaults if None)

        Returns:
            GenerationResult with response and metadata
        """
        ...

    async def batch_generate(
        self,
        prompts: list[str],
        config: Optional[RunnerConfig] = None,
    ) -> list[GenerationResult]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of GenerationResults
        """
        ...


class BaseModelRunner(ABC):
    """Base class for model runners with common functionality."""

    def __init__(self, model_path: Path, config: Optional[RunnerConfig] = None):
        self._model_path = Path(model_path)
        self._config = config or RunnerConfig()
        self._ready = False

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def config(self) -> RunnerConfig:
        return self._config

    @property
    def is_ready(self) -> bool:
        return self._ready

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[RunnerConfig] = None,
    ) -> GenerationResult:
        """Generate a response."""
        pass

    async def batch_generate(
        self,
        prompts: list[str],
        config: Optional[RunnerConfig] = None,
    ) -> list[GenerationResult]:
        """Generate responses for multiple prompts sequentially."""
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, config)
            results.append(result)
        return results
