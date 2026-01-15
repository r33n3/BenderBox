"""
Main interrogation engine that orchestrates model testing.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Union

from .analyzer.results import AnalysisResult, ResponseAnalyzer
from .prompts.loader import PromptLibrary
from .prompts.schema import TestPrompt
from .reports.generator import InterrogationReport, ReportGenerator
from .runner.base import ModelRunner, RunnerConfig
from .runner.llama_cpp import LlamaCppRunner
from .runner.api_base import BaseAPIRunner
from .scoring.risk import InterrogationRiskScore, RiskScorer
from .validator.censorship import CensorshipValidator, MislabelingReport

# Import behavior analyzer for optional behavior analysis
try:
    from benderbox.analyzers.behavior import BehaviorAnalyzer, BehaviorProfile
    HAS_BEHAVIOR_ANALYZER = True
except ImportError:
    HAS_BEHAVIOR_ANALYZER = False
    BehaviorProfile = None

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
        enable_behavior_analysis: bool = True,
    ):
        """
        Initialize the interrogation engine.

        Args:
            prompts_dir: Directory containing prompt YAML files
            enable_behavior_analysis: Whether to run behavior analysis
        """
        self.prompt_library = PromptLibrary(prompts_dir)
        self.analyzer = ResponseAnalyzer()
        self.scorer = RiskScorer()
        self.censorship_validator = CensorshipValidator()
        self.report_generator = ReportGenerator()

        # Initialize behavior analyzer if available and enabled
        self._behavior_analyzer = None
        self._enable_behavior_analysis = enable_behavior_analysis
        if enable_behavior_analysis and HAS_BEHAVIOR_ANALYZER:
            self._behavior_analyzer = BehaviorAnalyzer()

    def load_custom_tests(self, tests_file: Union[Path, str]) -> tuple:
        """
        Load custom interrogation tests from a file.

        Args:
            tests_file: Path to custom tests file (.md, .yaml, .yml)

        Returns:
            Tuple of (count loaded, list of errors)
        """
        self.prompt_library.load()  # Ensure library is loaded first
        return self.prompt_library.load_custom_file(Path(tests_file))

    async def interrogate(
        self,
        model_path: Union[Path, str],
        profile: str = "standard",
        claimed_censorship: str = "unknown",
        validate_censorship: bool = True,
        runner_config: Optional[RunnerConfig] = None,
        runner: Optional[ModelRunner] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        custom_tests_file: Optional[Union[Path, str]] = None,
    ) -> InterrogationReport:
        """
        Run full model interrogation.

        Args:
            model_path: Path to the GGUF model file or model identifier
            profile: Interrogation profile ("quick", "standard", "full")
            claimed_censorship: Claimed censorship level
            validate_censorship: Whether to run censorship validation
            runner_config: Configuration for model runner
            runner: Pre-initialized model runner (if None, creates LlamaCppRunner)
            progress_callback: Callback for progress updates (message, percent)
            custom_tests_file: Optional path to custom tests file (.md, .yaml, .yml)

        Returns:
            Complete InterrogationReport
        """
        start_time = time.time()

        # Determine model name for logging
        if runner is not None:
            if isinstance(runner, BaseAPIRunner):
                model_name = f"{runner.provider_name}:{runner.model_name}"
            else:
                model_name = str(runner.model_path.name)
        else:
            model_path = Path(model_path)
            model_name = model_path.name

        logger.info(f"Starting interrogation of {model_name}")
        logger.info(f"Profile: {profile}, Validate censorship: {validate_censorship}")

        # Initialize runner if not provided
        if progress_callback:
            progress_callback("Initializing model runner...", 0.0)

        if runner is None:
            try:
                model_path = Path(model_path)
                runner = LlamaCppRunner(model_path, config=runner_config)
            except Exception as e:
                logger.error(f"Failed to initialize runner: {e}")
                raise

        # Load prompts
        if progress_callback:
            progress_callback("Loading prompts...", 0.05)

        self.prompt_library.load()

        # Load custom tests from file if provided
        if custom_tests_file:
            custom_path = Path(custom_tests_file)
            if progress_callback:
                progress_callback(f"Loading custom tests from {custom_path.name}...", 0.06)
            count, errors = self.prompt_library.load_custom_file(custom_path)
            if count > 0:
                logger.info(f"Loaded {count} custom prompts from {custom_path}")
            for error in errors:
                logger.warning(error)

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

        # Run behavior analysis if enabled
        behavior_profile = None
        if self._behavior_analyzer and self._enable_behavior_analysis:
            if progress_callback:
                progress_callback("Running behavior analysis...", 0.90)

            try:
                # Build prompt-response pairs from results
                prompt_response_pairs = [
                    (r.prompt.prompt, r.response)
                    for r in results
                    if r.response
                ]
                behavior_profile = await self._behavior_analyzer.build_profile(
                    [
                        await self._behavior_analyzer.analyze_output(p, r)
                        for p, r in prompt_response_pairs
                    ],
                    model_name=model_name,
                )
            except Exception as e:
                logger.warning(f"Behavior analysis failed: {e}")

        # Calculate risk score
        if progress_callback:
            progress_callback("Calculating risk score...", 0.95)

        duration = time.time() - start_time
        risk_score = self.scorer.calculate(results, duration)

        # Generate report
        if progress_callback:
            progress_callback("Generating report...", 0.98)

        # Add API cost info to report if available
        api_cost_info = None
        if isinstance(runner, BaseAPIRunner):
            api_cost_info = {
                "provider": runner.provider_name,
                "model": runner.model_name,
                "prompt_tokens": runner.prompt_tokens_used,
                "completion_tokens": runner.completion_tokens_used,
                "total_tokens": runner.tokens_used,
                "estimated_cost_usd": runner.estimated_cost,
            }

        report = self.report_generator.generate(
            results=results,
            risk_score=risk_score,
            model_path=model_name,
            claimed_censorship=claimed_censorship,
            censorship_report=censorship_report,
            profile=profile,
            duration=duration,
            api_cost_info=api_cost_info,
            behavior_profile=behavior_profile,
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
        model_path: Union[Path, str],
        profile: str = "standard",
        claimed_censorship: str = "unknown",
        validate_censorship: bool = True,
        runner_config: Optional[RunnerConfig] = None,
        runner: Optional[ModelRunner] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        custom_tests_file: Optional[Union[Path, str]] = None,
    ) -> InterrogationReport:
        """Synchronous version of interrogate."""
        return asyncio.run(
            self.interrogate(
                model_path=model_path,
                profile=profile,
                claimed_censorship=claimed_censorship,
                validate_censorship=validate_censorship,
                runner_config=runner_config,
                runner=runner,
                progress_callback=progress_callback,
                custom_tests_file=custom_tests_file,
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
