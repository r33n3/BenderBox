"""
Google Gemini API model runner.

Supports Gemini 1.5 Pro, Gemini 1.5 Flash, and other Gemini models.
"""

import logging
from typing import Dict, Optional, Tuple

import httpx

from .api_base import BaseAPIRunner
from .base import RunnerConfig

logger = logging.getLogger(__name__)


class GeminiRunner(BaseAPIRunner):
    """
    Model runner for Google Gemini API.

    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, and other models.
    Uses the generativelanguage.googleapis.com API.
    """

    # Pricing per 1K tokens (as of 2024)
    PRICING: Dict[str, Dict[str, float]] = {
        # Gemini 1.5
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-pro-latest": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-flash-latest": {"input": 0.000075, "output": 0.0003},
        # Gemini 1.0 (legacy)
        "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        api_key: str = "",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        config: Optional[RunnerConfig] = None,
    ):
        """
        Initialize the Gemini runner.

        Args:
            model_name: Gemini model name (e.g., "gemini-1.5-pro")
            api_key: Google API key
            base_url: Base URL for API
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
        return "Google"

    async def _call_api(
        self,
        prompt: str,
        config: RunnerConfig,
    ) -> dict:
        """Make API call to Gemini generateContent endpoint."""
        async with httpx.AsyncClient() as client:
            # Gemini uses a different URL structure with API key as query param
            url = f"{self._base_url}/models/{self._model_name}:generateContent"

            response = await client.post(
                url,
                params={"key": self._api_key},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ],
                    "generationConfig": {
                        "maxOutputTokens": config.max_tokens,
                        "temperature": config.temperature,
                        "topP": config.top_p,
                        "topK": config.top_k,
                    },
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            return response.json()

    def _parse_response(self, response: dict) -> Tuple[str, int, int]:
        """Parse Gemini API response."""
        # Extract text from response
        text = ""
        candidates = response.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    text += part["text"]

        # Extract token counts from usageMetadata
        usage = response.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        return text, prompt_tokens, completion_tokens
