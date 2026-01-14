"""
llama.cpp model runner for local GGUF models.
"""

import asyncio
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from .base import BaseModelRunner, FinishReason, GenerationResult, RunnerConfig

logger = logging.getLogger(__name__)


def find_llama_cli() -> Optional[Path]:
    """
    Find llama-cli executable.

    Checks:
    1. BenderBox tools directory (./tools/)
    2. llama.cpp/build*/bin/llama-cli (local build)
    3. System PATH
    """
    import shutil
    from benderbox.config import get_benderbox_home

    # Check BenderBox tools directory first
    tools_dir = get_benderbox_home() / "tools"
    if tools_dir.exists():
        for exe_name in ["llama-cli.exe", "llama-cli"]:
            tool_path = tools_dir / exe_name
            if tool_path.exists():
                return tool_path.resolve()

    # Check local build directories
    local_patterns = [
        Path("llama.cpp/build/bin/llama-cli"),
        Path("llama.cpp/build/bin/llama-cli.exe"),
        Path("llama.cpp/build-release/bin/llama-cli"),
        Path("llama.cpp/build-release/bin/llama-cli.exe"),
    ]
    for pattern in local_patterns:
        if pattern.exists():
            return pattern.resolve()

    # Check system PATH
    system_cli = shutil.which("llama-cli")
    if system_cli:
        return Path(system_cli)

    return None


class LlamaCppRunner(BaseModelRunner):
    """
    Model runner using llama.cpp CLI.

    Executes prompts against GGUF models using llama-cli.
    """

    def __init__(
        self,
        model_path: Path,
        llama_cli_path: Optional[Path] = None,
        config: Optional[RunnerConfig] = None,
    ):
        """
        Initialize the llama.cpp runner.

        Args:
            model_path: Path to the GGUF model file
            llama_cli_path: Path to llama-cli executable (auto-detected if None)
            config: Runner configuration
        """
        super().__init__(model_path, config)

        self._llama_cli = llama_cli_path or find_llama_cli()
        if self._llama_cli is None:
            raise RuntimeError(
                "llama-cli not found. Install with: benderbox prerequisites install llama-cli"
            )

        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        self._ready = True
        logger.info(f"LlamaCppRunner initialized: {self._model_path.name}")

    @property
    def llama_cli_path(self) -> Path:
        """Path to llama-cli executable."""
        return self._llama_cli

    async def generate(
        self,
        prompt: str,
        config: Optional[RunnerConfig] = None,
    ) -> GenerationResult:
        """
        Generate a response using llama-cli.

        Args:
            prompt: The input prompt
            config: Generation configuration (uses instance config if None)

        Returns:
            GenerationResult with response and metadata
        """
        cfg = config or self._config
        start_time = time.time()

        try:
            # Build command
            cmd = [
                str(self._llama_cli),
                "-m", str(self._model_path),
                "-p", prompt,
                "--single-turn",  # Exit after generating
                "--no-display-prompt",  # Don't echo the prompt
            ]
            cmd.extend(cfg.to_cli_args())

            logger.debug(f"Running: {' '.join(cmd[:6])}...")

            # Run in thread pool to not block async
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=cfg.timeout_seconds,
                ),
            )

            elapsed = time.time() - start_time
            output = result.stdout + result.stderr

            # Parse the response
            response = self._parse_response(output, prompt)

            # Determine finish reason
            if result.returncode != 0:
                finish_reason = FinishReason.ERROR
                error = f"llama-cli exited with code {result.returncode}"
            elif len(response) >= cfg.max_tokens * 4:  # Rough estimate
                finish_reason = FinishReason.LENGTH
                error = None
            else:
                finish_reason = FinishReason.STOP
                error = None

            # Estimate tokens (rough)
            tokens_generated = len(response.split())

            return GenerationResult(
                response=response,
                finish_reason=finish_reason,
                tokens_generated=tokens_generated,
                generation_time=elapsed,
                error=error,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            return GenerationResult(
                response="",
                finish_reason=FinishReason.TIMEOUT,
                generation_time=elapsed,
                error=f"Generation timed out after {cfg.timeout_seconds}s",
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                response="",
                finish_reason=FinishReason.ERROR,
                generation_time=elapsed,
                error=str(e),
            )

    def _parse_response(self, output: str, prompt: str) -> str:
        """
        Parse llama-cli output to extract the model's response.

        Args:
            output: Raw output from llama-cli
            prompt: The original prompt (to remove if echoed)

        Returns:
            The extracted response text
        """
        # Remove ANSI escape codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        output = ansi_escape.sub('', output)

        # Remove the llama.cpp banner/logo
        lines = output.split('\n')
        response_lines = []
        in_response = False
        skip_patterns = [
            'Loading model',
            'build      :',
            'model      :',
            'modalities :',
            'available commands:',
            '/exit or Ctrl+C',
            '/regen',
            '/clear',
            '/read',
            'load_backend:',
            'Exiting...',
            '[ Prompt:',
            'llama_memory',
        ]

        for line in lines:
            # Skip banner and metadata lines
            skip = False
            for pattern in skip_patterns:
                if pattern in line:
                    skip = True
                    break

            # Skip lines that are just the prompt echo
            if line.strip() == '>' or line.strip().startswith('> '):
                in_response = True
                # Get the part after "> "
                if line.strip().startswith('> '):
                    remaining = line.strip()[2:].strip()
                    if remaining and remaining != prompt.strip():
                        response_lines.append(remaining)
                continue

            if skip:
                continue

            # Skip empty lines at the start
            if not in_response and not line.strip():
                continue

            # Once we see ">", we're in response mode
            if in_response or (line.strip() and not any(p in line for p in skip_patterns)):
                in_response = True
                response_lines.append(line)

        response = '\n'.join(response_lines).strip()

        # Remove the original prompt if it appears at the start
        if response.startswith(prompt.strip()):
            response = response[len(prompt.strip()):].strip()

        return response

    def generate_sync(
        self,
        prompt: str,
        config: Optional[RunnerConfig] = None,
    ) -> GenerationResult:
        """
        Synchronous version of generate for non-async contexts.
        """
        return asyncio.run(self.generate(prompt, config))
