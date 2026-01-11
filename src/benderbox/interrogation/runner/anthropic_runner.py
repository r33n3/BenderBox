"""
Anthropic API model runner.

Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, and other Claude models.
"""

import logging
from typing import Dict, Optional, Tuple

import httpx

from .api_base import BaseAPIRunner
from .base import RunnerConfig

logger = logging.getLogger(__name__)


class AnthropicRunner(BaseAPIRunner):
    """
    Model runner for Anthropic API.

    Supports all Claude models including Claude 3.5 and Claude 3.
    """

    # Pricing per 1K tokens (as of 2024)
    PRICING: Dict[str, Dict[str, float]] = {
        # Claude 3.5
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet-latest": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
        "claude-3-5-haiku-latest": {"input": 0.001, "output": 0.005},
        # Claude 3
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-opus-latest": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        # Claude 2 (legacy)
        "claude-2.1": {"input": 0.008, "output": 0.024},
        "claude-2.0": {"input": 0.008, "output": 0.024},
    }

    # API version header
    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str = "",
        base_url: str = "https://api.anthropic.com",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        config: Optional[RunnerConfig] = None,
    ):
        """
        Initialize the Anthropic runner.

        Args:
            model_name: Claude model name (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key
            base_url: Base URL for API (default: Anthropic's API)
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
        return "Anthropic"

    async def _call_api(
        self,
        prompt: str,
        config: RunnerConfig,
    ) -> dict:
        """Make API call to Anthropic messages endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._base_url}/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": self.ANTHROPIC_VERSION,
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "max_tokens": config.max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()

    def _parse_response(self, response: dict) -> Tuple[str, int, int]:
        """Parse Anthropic API response."""
        # Extract text from response
        content = response.get("content", [])
        text = ""
        for block in content:
            if block.get("type") == "text":
                text += block.get("text", "")

        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return text, prompt_tokens, completion_tokens
