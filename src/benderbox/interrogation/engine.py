"""
Main interrogation engine that orchestrates model testing.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional

from .analyzer.results import AnalysisResult, ResponseAnalyzer
from .prompts.loader import PromptLibrary
from .prompts.schema import TestPrompt
from .reports.generator import InterrogationReport, ReportGenerator
from .runner.base import ModelRunner, RunnerConfig
from .runner.llama_cpp import LlamaCppRunner
from .scoring.risk import InterrogationRiskScore, RiskScorer
from .validator.censorship import CensorshipValidator, MislabelingReport

logger = logging.getLogger(__name__)


class InterrogationEngine:
    """
    Main engine for model interrogation.

    Orchestrates prompt loading, model execution, analysis,
    and report generation.
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
    ):
        """
        Initialize the interrogation engine.

        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        self.prompt_library = PromptLibrary(prompts_dir)
        self.analyzer = ResponseAnalyzer()
        self.scorer = RiskScorer()
        self.censorship_validator = CensorshipValidator()
        self.report_generator = ReportGenerator()

    async def interrogate(
        self,
        model_path: Path,
        profile: str = "standard",
        claimed_censorship: str = "unknown",
        validate_censorship: bool = True,
        runner_config: Optional[RunnerConfig] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> InterrogationReport:
        """
        Run full model interrogation.

        Args:
            model_path: Path to the GGUF model file
            profile: Interrogation profile ("quick", "standard", "full")
            claimed_censorship: Claimed censorship level
            validate_censorship: Whether to run censorship validation
            runner_config: Configuration for model runner
            progress_callback: Callback for progress updates (message, percent)

        Returns:
            Complete InterrogationReport
        """
        start_time = time.time()
        model_path = Path(model_path)

        logger.info(f"Starting interrogation of {model_path.name}")
        logger.info(f"Profile: {profile}, Validate censorship: {validate_censorship}")

        # Initialize runner
        if progress_callback:
            progress_callback("Initializing model runner...", 0.0)

        try:
            runner = LlamaCppRunner(model_path, config=runner_config)
        except Exception as e:
            logger.error(f"Failed to initialize runner: {e}")
            raise

        # Load prompts
        if progress_callback:
            progress_callback("Loading prompts...", 0.05)

        self.prompt_library.load()
        prompts = self.prompt_library.get_for_profile(profile)
        logger.info(f"Loaded {len(prompts)} prompts for {profile} profile")

        # Run interrogation
        results = await self._run_prompts(
            runner=runner,
            prompts=prompts,
            config=runner_config,
            progress_callback=progress_callback,
            progress_start=0.1,
            progress_end=0.8,
        )

        # Run censorship validation if requested
        censorship_report: Optional[MislabelingReport] = None
        if validate_censorship:
            if progress_callback:
                progress_callback("Validating censorship level...", 0.85)

            try:
                censorship_report = await self.censorship_validator.validate(
                    runner=runner,
                    claimed_censorship=claimed_censorship,
                    config=runner_config,
                )
            except Exception as e:
                logger.error(f"Censorship validation failed: {e}")

        # Calculate risk score
        if progress_callback:
            progress_callback("Calculating risk score...", 0.95)

        duration = time.time() - start_time
        risk_score = self.scorer.calculate(results, duration)

        # Generate report
        if progress_callback:
            progress_callback("Generating report...", 0.98)

        report = self.report_generator.generate(
            results=results,
            risk_score=risk_score,
            model_path=str(model_path),
            claimed_censorship=claimed_censorship,
            censorship_report=censorship_report,
            profile=profile,
            duration=duration,
        )

        if progress_callback:
            progress_callback("Complete", 1.0)

        logger.info(
            f"Interrogation complete: {risk_score.passed_count}/{len(results)} passed, "
            f"risk level: {risk_score.risk_level.value}"
        )

        return report

    async def _run_prompts(
        self,
        runner: ModelRunner,
        prompts: List[TestPrompt],
        config: Optional[RunnerConfig],
        progress_callback: Optional[Callable[[str, float], None]],
        progress_start: float,
        progress_end: float,
    ) -> List[AnalysisResult]:
        """Run prompts against the model and analyze results."""
        results = []
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            # Update progress
            if progress_callback:
                progress = progress_start + (progress_end - progress_start) * (i / total)
                progress_callback(
                    f"Testing prompt {i+1}/{total}: {prompt.id}",
                    progress,
                )

            try:
                # Generate response
                generation = await runner.generate(prompt.prompt, config)

                # Analyze response
                analysis = self.analyzer.analyze(prompt, generation)
                results.append(analysis)

                logger.debug(
                    f"[{i+1}/{total}] {prompt.id}: "
                    f"{analysis.classification.value} (passed: {analysis.passed})"
                )

            except Exception as e:
                logger.error(f"Failed to test prompt {prompt.id}: {e}")
                # Continue with other prompts

        return results

    def interrogate_sync(
        self,
        model_path: Path,
        profile: str = "standard",
        claimed_censorship: str = "unknown",
        validate_censorship: bool = True,
        runner_config: Optional[RunnerConfig] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> InterrogationReport:
        """Synchronous version of interrogate."""
        return asyncio.run(
            self.interrogate(
                model_path=model_path,
                profile=profile,
                claimed_censorship=claimed_censorship,
                validate_censorship=validate_censorship,
                runner_config=runner_config,
                progress_callback=progress_callback,
            )
        )


# Convenience function for quick interrogation
async def interrogate_model(
    model_path: Path,
    profile: str = "quick",
    claimed_censorship: str = "unknown",
) -> InterrogationReport:
    """
    Quick model interrogation with default settings.

    Args:
        model_path: Path to GGUF model
        profile: "quick", "standard", or "full"
        claimed_censorship: "censored", "uncensored", or "unknown"

    Returns:
        InterrogationReport
    """
    engine = InterrogationEngine()
    return await engine.interrogate(
        model_path=model_path,
        profile=profile,
        claimed_censorship=claimed_censorship,
    )
