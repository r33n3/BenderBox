"""
xAI Grok API model runner.

Supports Grok-2, Grok-2 mini, and other Grok models.
Uses OpenAI-compatible API format.
"""

import logging
from typing import Dict, Optional, Tuple

import httpx

from .api_base import BaseAPIRunner
from .base import RunnerConfig

logger = logging.getLogger(__name__)


class GrokRunner(BaseAPIRunner):
    """
    Model runner for xAI Grok API.

    Supports Grok-2 and Grok-2 mini models.
    Uses OpenAI-compatible chat completions API format.
    """

    # Pricing per 1K tokens (as of 2024 - approximate)
    PRICING: Dict[str, Dict[str, float]] = {
        "grok-2": {"input": 0.002, "output": 0.010},
        "grok-2-latest": {"input": 0.002, "output": 0.010},
        "grok-2-mini": {"input": 0.0002, "output": 0.001},
        "grok-2-mini-latest": {"input": 0.0002, "output": 0.001},
        # Grok Beta (if still available)
        "grok-beta": {"input": 0.005, "output": 0.015},
    }

    def __init__(
        self,
        model_name: str = "grok-2",
        api_key: str = "",
        base_url: str = "https://api.x.ai/v1",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        config: Optional[RunnerConfig] = None,
    ):
        """
        Initialize the Grok runner.

        Args:
            model_name: Grok model name (e.g., "grok-2", "grok-2-mini")
            api_key: xAI API key
            base_url: Base URL for API (default: xAI's API)
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
        return "xAI"

    async def _call_api(
        self,
        prompt: str,
        config: RunnerConfig,
    ) -> dict:
        """Make API call to Grok chat completions endpoint (OpenAI-compatible)."""
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
        """Parse Grok API response (OpenAI-compatible format)."""
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
