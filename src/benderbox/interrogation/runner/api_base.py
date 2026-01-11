"""
Base class for API-based model runners.

Provides common functionality for making HTTP requests to LLM APIs,
handling retries, rate limiting, and cost tracking.
"""

import asyncio
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx

from .base import BaseModelRunner, FinishReason, GenerationResult, RunnerConfig

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Exception raised for API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class BaseAPIRunner(BaseModelRunner):
    """
    Base class for API-based model runners.

    Provides common HTTP request handling, retry logic, and cost tracking.
    Subclasses implement provider-specific API calls.
    """

    # Override in subclasses with provider-specific pricing
    # Format: {model_name: {"input": cost_per_1k, "output": cost_per_1k}}
    PRICING: Dict[str, Dict[str, float]] = {}

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        config: Optional[RunnerConfig] = None,
    ):
        """
        Initialize the API runner.

        Args:
            model_name: Name of the model to use (e.g., "gpt-4-turbo")
            api_key: API key for authentication
            base_url: Base URL for the API
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
            config: Runner configuration for generation parameters
        """
        # API runners don't use a local model path
        super().__init__(Path("/api"), config)

        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Cost tracking
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0

        # Validate API key
        if not api_key:
            raise ValueError(f"API key required for {self.__class__.__name__}")

        self._ready = True
        logger.info(f"{self.__class__.__name__} initialized: {model_name}")

    @property
    def model_name(self) -> str:
        """Name of the model being used."""
        return self._model_name

    @property
    def model_path(self) -> Path:
        """Override to return a descriptive path for API models."""
        return Path(f"/api/{self._model_name}")

    @property
    def tokens_used(self) -> int:
        """Total tokens used across all requests."""
        return self._total_prompt_tokens + self._total_completion_tokens

    @property
    def prompt_tokens_used(self) -> int:
        """Total prompt/input tokens used."""
        return self._total_prompt_tokens

    @property
    def completion_tokens_used(self) -> int:
        """Total completion/output tokens used."""
        return self._total_completion_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimated total cost in USD."""
        return self._total_cost

    @property
    def provider_name(self) -> str:
        """Name of the API provider (override in subclasses)."""
        return "API"

    @abstractmethod
    async def _call_api(
        self,
        prompt: str,
        config: RunnerConfig,
    ) -> dict:
        """
        Make the actual API call.

        Args:
            prompt: The prompt to send
            config: Generation configuration

        Returns:
            Raw API response as a dictionary

        Raises:
            APIError: If the API call fails
        """
        pass

    @abstractmethod
    def _parse_response(self, response: dict) -> Tuple[str, int, int]:
        """
        Parse the API response.

        Args:
            response: Raw API response dictionary

        Returns:
            Tuple of (response_text, prompt_tokens, completion_tokens)
        """
        pass

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Estimate cost for this request.

        Args:
            prompt_tokens: Number of prompt/input tokens
            completion_tokens: Number of completion/output tokens

        Returns:
            Estimated cost in USD
        """
        pricing = self.PRICING.get(self._model_name)
        if not pricing:
            # Try to find a matching prefix (e.g., "gpt-4" for "gpt-4-turbo-2024-01-01")
            for model_prefix, prices in self.PRICING.items():
                if self._model_name.startswith(model_prefix):
                    pricing = prices
                    break

        if not pricing:
            return 0.0

        input_cost = (prompt_tokens / 1000) * pricing.get("input", 0)
        output_cost = (completion_tokens / 1000) * pricing.get("output", 0)
        return input_cost + output_cost

    async def generate(
        self,
        prompt: str,
        config: Optional[RunnerConfig] = None,
    ) -> GenerationResult:
        """
        Generate a response using the API.

        Args:
            prompt: The input prompt
            config: Generation configuration (uses instance config if None)

        Returns:
            GenerationResult with response and metadata
        """
        cfg = config or self._config
        start_time = time.time()
        last_error = None

        # Retry logic with exponential backoff
        for attempt in range(self._max_retries):
            try:
                response = await self._call_api(prompt, cfg)
                elapsed = time.time() - start_time

                # Parse response
                text, prompt_tokens, completion_tokens = self._parse_response(response)

                # Update totals
                self._total_prompt_tokens += prompt_tokens
                self._total_completion_tokens += completion_tokens
                cost = self._estimate_cost(prompt_tokens, completion_tokens)
                self._total_cost += cost

                return GenerationResult(
                    response=text,
                    finish_reason=FinishReason.STOP,
                    tokens_generated=completion_tokens,
                    generation_time=elapsed,
                    prompt_tokens=prompt_tokens,
                    error=None,
                )

            except httpx.TimeoutException as e:
                last_error = f"Request timed out: {e}"
                logger.warning(f"Attempt {attempt + 1}/{self._max_retries}: {last_error}")

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:  # Rate limited
                    last_error = "Rate limited by API"
                    logger.warning(f"Rate limited, waiting before retry...")
                elif status >= 500:  # Server error
                    last_error = f"Server error: {status}"
                else:
                    # Client error - don't retry
                    elapsed = time.time() - start_time
                    error_msg = f"API error {status}: {e.response.text[:200]}"
                    return GenerationResult(
                        response="",
                        finish_reason=FinishReason.ERROR,
                        generation_time=elapsed,
                        error=error_msg,
                    )

            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt + 1}/{self._max_retries}: {last_error}")

            # Wait before retry (exponential backoff)
            if attempt < self._max_retries - 1:
                delay = self._retry_delay * (2**attempt)
                await asyncio.sleep(delay)

        # All retries exhausted
        elapsed = time.time() - start_time
        return GenerationResult(
            response="",
            finish_reason=FinishReason.ERROR,
            generation_time=elapsed,
            error=f"Failed after {self._max_retries} attempts: {last_error}",
        )

    def get_cost_estimate(self, num_prompts: int, avg_prompt_tokens: int = 100, avg_response_tokens: int = 150) -> Tuple[float, float]:
        """
        Estimate cost for running interrogation.

        Args:
            num_prompts: Number of prompts to send
            avg_prompt_tokens: Estimated average tokens per prompt
            avg_response_tokens: Estimated average tokens per response

        Returns:
            Tuple of (min_cost, max_cost) in USD
        """
        total_prompt_tokens = num_prompts * avg_prompt_tokens
        min_response_tokens = num_prompts * (avg_response_tokens // 2)
        max_response_tokens = num_prompts * (avg_response_tokens * 2)

        min_cost = self._estimate_cost(total_prompt_tokens, min_response_tokens)
        max_cost = self._estimate_cost(total_prompt_tokens, max_response_tokens)

        return min_cost, max_cost

    def reset_cost_tracking(self) -> None:
        """Reset cost tracking counters."""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0
