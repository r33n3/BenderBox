"""
OpenAI API model runner.

Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo, and other OpenAI chat models.
"""

import logging
from typing import Dict, Optional, Tuple

import httpx

from .api_base import BaseAPIRunner
from .base import RunnerConfig

logger = logging.getLogger(__name__)


class OpenAIRunner(BaseAPIRunner):
    """
    Model runner for OpenAI API.

    Supports all OpenAI chat completion models including GPT-4 and GPT-3.5.
    """

    # Pricing per 1K tokens (as of 2024)
    PRICING: Dict[str, Dict[str, float]] = {
        # GPT-4 Turbo
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-2024": {"input": 0.01, "output": 0.03},
        # GPT-4
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-0613": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        # GPT-4o
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        # GPT-3.5
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    }

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        config: Optional[RunnerConfig] = None,
    ):
        """
        Initialize the OpenAI runner.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4-turbo", "gpt-3.5-turbo")
            api_key: OpenAI API key
            base_url: Base URL for API (default: OpenAI's API)
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
            config: Generation configuration
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            config=config,
        )

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    async def _call_api(
        self,
        prompt: str,
        config: RunnerConfig,
    ) -> dict:
        """Make API call to OpenAI chat completions endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()

    def _parse_response(self, response: dict) -> Tuple[str, int, int]:
        """Parse OpenAI API response."""
        # Extract text from response
        choices = response.get("choices", [])
        if not choices:
            return "", 0, 0

        text = choices[0].get("message", {}).get("content", "")

        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        return text, prompt_tokens, completion_tokens
