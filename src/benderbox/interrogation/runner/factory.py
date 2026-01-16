"""
Runner factory for creating appropriate model runners.

Automatically selects the correct runner based on target format:
- Local GGUF files: LlamaCppRunner
- HuggingFace models: Download + LlamaCppRunner
- OpenAI API: OpenAIRunner
- Anthropic API: AnthropicRunner
- Gemini API: GeminiRunner
- Grok API: GrokRunner
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

from .base import ModelRunner, RunnerConfig
from .llama_cpp import LlamaCppRunner
from .openai_runner import OpenAIRunner
from .anthropic_runner import AnthropicRunner
from .gemini_runner import GeminiRunner
from .grok_runner import GrokRunner

logger = logging.getLogger(__name__)


# API prefixes and their runners
API_PREFIXES = {
    "openai:": ("openai", OpenAIRunner),
    "anthropic:": ("anthropic", AnthropicRunner),
    "gemini:": ("google", GeminiRunner),
    "grok:": ("xai", GrokRunner),
}


class RunnerFactory:
    """Factory for creating model runners based on target format."""

    @staticmethod
    def is_api_target(target: str) -> bool:
        """Check if target is an API target (requires API key)."""
        return any(target.startswith(prefix) for prefix in API_PREFIXES.keys())

    @staticmethod
    def parse_api_target(target: str) -> Tuple[str, str, type]:
        """
        Parse API target into components.

        Args:
            target: Target string (e.g., "openai:gpt-4-turbo")

        Returns:
            Tuple of (provider_key, model_name, runner_class)

        Raises:
            ValueError: If target is not a valid API target
        """
        for prefix, (provider_key, runner_class) in API_PREFIXES.items():
            if target.startswith(prefix):
                model_name = target[len(prefix):]
                return provider_key, model_name, runner_class

        raise ValueError(f"Unknown API target format: {target}")

    @staticmethod
    def get_api_key(config, provider_key: str) -> str:
        """
        Get API key for provider from config.

        Args:
            config: BenderBox Config object
            provider_key: Provider key ("openai", "anthropic", "google", "xai")

        Returns:
            API key string

        Raises:
            ValueError: If API key is not configured
        """
        key_mapping = {
            "openai": "openai_api_key",
            "anthropic": "anthropic_api_key",
            "google": "google_api_key",
            "xai": "xai_api_key",
        }

        env_var_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "xai": "XAI_API_KEY",
        }

        attr_name = key_mapping.get(provider_key)
        if not attr_name:
            raise ValueError(f"Unknown provider: {provider_key}")

        api_key = getattr(config.api, attr_name, "")
        if not api_key:
            env_var = env_var_mapping.get(provider_key, "")
            raise ValueError(
                f"{provider_key.upper()} API key not configured. "
                f"Set {env_var} environment variable."
            )

        return api_key

    @staticmethod
    def get_base_url(config, provider_key: str) -> str:
        """Get base URL for provider from config."""
        url_mapping = {
            "openai": "openai_base_url",
            "anthropic": "anthropic_base_url",
            "xai": "xai_base_url",
        }

        attr_name = url_mapping.get(provider_key)
        if attr_name:
            return getattr(config.api, attr_name, "")

        # Default URLs for providers without configurable base URL
        defaults = {
            "google": "https://generativelanguage.googleapis.com/v1beta",
        }
        return defaults.get(provider_key, "")

    @staticmethod
    def create_runner(
        target: str,
        config,
        runner_config: Optional[RunnerConfig] = None,
    ) -> ModelRunner:
        """
        Create appropriate runner based on target.

        Target formats:
        - Local GGUF: ./model.gguf, data/models/file.gguf
        - OpenAI: openai:gpt-4-turbo, openai:gpt-3.5-turbo
        - Anthropic: anthropic:claude-3-5-sonnet-20241022
        - Google: gemini:gemini-1.5-pro
        - xAI: grok:grok-2

        Args:
            target: Model target string
            config: BenderBox Config object
            runner_config: Generation configuration

        Returns:
            Appropriate ModelRunner instance

        Raises:
            ValueError: If target format is invalid or API key missing
        """
        # Check for API targets
        if RunnerFactory.is_api_target(target):
            provider_key, model_name, runner_class = RunnerFactory.parse_api_target(target)

            # Get API key and base URL
            api_key = RunnerFactory.get_api_key(config, provider_key)
            base_url = RunnerFactory.get_base_url(config, provider_key)

            # Get timeout settings
            timeout = config.api.api_timeout_seconds
            max_retries = config.api.max_retries

            logger.info(f"Creating {runner_class.__name__} for {model_name}")

            return runner_class(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                timeout_seconds=timeout,
                max_retries=max_retries,
                config=runner_config,
            )

        # Default: local GGUF model
        model_path = Path(target)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {target}")

        logger.info(f"Creating LlamaCppRunner for {model_path.name}")
        return LlamaCppRunner(model_path, config=runner_config)

    @staticmethod
    def get_prompt_count_for_profile(profile: str) -> int:
        """Get estimated prompt count for a profile."""
        counts = {
            "quick": 16,
            "standard": 50,
            "full": 100,
            "adversarial": 64,  # Jailbreak/security focused
        }
        return counts.get(profile, 50)

    @staticmethod
    def estimate_api_cost(
        target: str,
        profile: str = "quick",
        avg_prompt_tokens: int = 100,
        avg_response_tokens: int = 150,
    ) -> Tuple[float, float]:
        """
        Estimate cost for API interrogation.

        Args:
            target: API target string (e.g., "openai:gpt-4-turbo")
            profile: Interrogation profile
            avg_prompt_tokens: Average tokens per prompt
            avg_response_tokens: Average tokens per response

        Returns:
            Tuple of (min_cost, max_cost) in USD
        """
        if not RunnerFactory.is_api_target(target):
            return 0.0, 0.0

        provider_key, model_name, runner_class = RunnerFactory.parse_api_target(target)
        num_prompts = RunnerFactory.get_prompt_count_for_profile(profile)

        # Get pricing from runner class
        pricing = runner_class.PRICING.get(model_name)
        if not pricing:
            # Try prefix matching
            for model_prefix, prices in runner_class.PRICING.items():
                if model_name.startswith(model_prefix):
                    pricing = prices
                    break

        if not pricing:
            return 0.0, 0.0

        total_prompt_tokens = num_prompts * avg_prompt_tokens
        min_response_tokens = num_prompts * (avg_response_tokens // 2)
        max_response_tokens = num_prompts * (avg_response_tokens * 2)

        input_cost_per_1k = pricing.get("input", 0)
        output_cost_per_1k = pricing.get("output", 0)

        min_cost = (
            (total_prompt_tokens / 1000) * input_cost_per_1k +
            (min_response_tokens / 1000) * output_cost_per_1k
        )
        max_cost = (
            (total_prompt_tokens / 1000) * input_cost_per_1k +
            (max_response_tokens / 1000) * output_cost_per_1k
        )

        return min_cost, max_cost
