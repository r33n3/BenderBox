"""
Analysis Bridge for BenderBox

Bridges the NLP layer to the existing analysis engine (sandbox_cli),
providing async wrappers and structured result handling.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from benderbox.utils.model_source import DownloadProgress

logger = logging.getLogger(__name__)


@dataclass
class TestInfo:
    """Information about an available test."""

    name: str
    category: str
    description: str = ""


@dataclass
class ProfileInfo:
    """Information about an analysis profile."""

    name: str
    description: str
    tests: List[str]
    target_type: str  # "model", "infrastructure", "both"


# Profile descriptions
PROFILE_INFO: Dict[str, ProfileInfo] = {
    "quick": ProfileInfo(
        name="quick",
        description="Fast GGUF metadata extraction only",
        tests=["gguf_metadata_sanity"],
        target_type="model",
    ),
    "standard": ProfileInfo(
        name="standard",
        description="Common static tests including GGUF analysis",
        tests=["static_metadata_basic", "gguf_metadata_sanity"],
        target_type="model",
    ),
    "deep": ProfileInfo(
        name="deep",
        description="All available tests (static + dynamic)",
        tests=["static_metadata_basic", "gguf_metadata_sanity"],
        target_type="model",
    ),
    "attack": ProfileInfo(
        name="attack",
        description="Security/jailbreak focused tests only",
        tests=[],
        target_type="model",
    ),
    "infra-quick": ProfileInfo(
        name="infra-quick",
        description="Quick infrastructure scan",
        tests=[],
        target_type="infrastructure",
    ),
    "infra-standard": ProfileInfo(
        name="infra-standard",
        description="Standard infrastructure security tests",
        tests=[],
        target_type="infrastructure",
    ),
    "infra-deep": ProfileInfo(
        name="infra-deep",
        description="Comprehensive infrastructure analysis",
        tests=[],
        target_type="infrastructure",
    ),
}


@dataclass
class ComparisonResult:
    """Result of comparing multiple targets."""

    targets: List[str]
    comparison_type: str
    results: List[Dict[str, Any]]
    summary: Dict[str, Any] = field(default_factory=dict)


class AnalysisBridge:
    """
    Bridges NLP layer to existing analysis engine.

    Provides:
    - Async wrappers for sandbox_cli functions
    - Result conversion to NLP-friendly format
    - Analysis coordination
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        model_cache_path: Optional[str] = None,
        cache_ttl_days: int = 30,
        download_timeout: int = 600,
        max_download_size_gb: float = 50.0,
    ):
        """
        Initialize AnalysisBridge.

        Args:
            log_dir: Directory for analysis logs. Defaults to ./sandbox_logs.
            model_cache_path: Path for caching downloaded models.
            cache_ttl_days: Days to keep cached models.
            download_timeout: Download timeout in seconds.
            max_download_size_gb: Maximum download size in GB.
        """
        self.log_dir = Path(log_dir or "./sandbox_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Model source handler config
        self._model_cache_path = Path(model_cache_path) if model_cache_path else Path("data/models")
        self._cache_ttl_days = cache_ttl_days
        self._download_timeout = download_timeout
        self._max_download_size_gb = max_download_size_gb

        # Lazy import of sandbox_cli components
        self._sandbox_analyze = None
        self._test_registry = None
        self._profile_tests = None
        self._model_source_handler = None

    def _ensure_imported(self) -> None:
        """Ensure sandbox_cli components are imported."""
        if self._sandbox_analyze is not None:
            return

        try:
            from benderbox.sandbox_cli import (
                sandbox_analyze,
                TEST_REGISTRY,
                PROFILE_TESTS,
            )
            self._sandbox_analyze = sandbox_analyze
            self._test_registry = TEST_REGISTRY
            self._profile_tests = PROFILE_TESTS
        except ImportError as e:
            logger.error(f"Failed to import sandbox_cli: {e}")
            raise AnalysisBridgeError(f"Analysis engine not available: {e}")

    def _get_model_source_handler(self):
        """Get or create the model source handler."""
        if self._model_source_handler is None:
            from benderbox.utils.model_source import ModelSourceHandler
            self._model_source_handler = ModelSourceHandler(
                cache_path=self._model_cache_path,
                cache_ttl_days=self._cache_ttl_days,
                download_timeout=self._download_timeout,
                max_size_gb=self._max_download_size_gb,
            )
        return self._model_source_handler

    async def resolve_model_source(
        self,
        target: str,
        progress_callback: Optional[Callable[["DownloadProgress"], None]] = None,
    ) -> Path:
        """
        Resolve a model target to a local path.

        Supports:
        - Local file paths (returned as-is if exists)
        - HTTP/HTTPS URLs (downloaded and cached)
        - Hugging Face model IDs (downloaded and cached)

        Args:
            target: Model target (path, URL, or HF model ID).
            progress_callback: Optional callback for download progress.

        Returns:
            Path to the local model file.

        Raises:
            AnalysisBridgeError: If resolution fails.
        """
        from benderbox.utils.model_source import ModelSource, ModelSourceError

        handler = self._get_model_source_handler()
        source_type = handler.detect_source(target)

        logger.info(f"Resolving model target: {target} (source: {source_type.value})")

        try:
            result = await handler.resolve(target, progress_callback)

            if result.cached:
                logger.info(f"Using cached model: {result.local_path}")
            else:
                logger.info(f"Model resolved to: {result.local_path}")

            return result.local_path

        except ModelSourceError as e:
            raise AnalysisBridgeError(str(e)) from e

    async def analyze_model(
        self,
        model_path: str,
        profile: str = "standard",
        progress_callback: Optional[Callable[[str], None]] = None,
        download_progress_callback: Optional[Callable[["DownloadProgress"], None]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a GGUF model file.

        Args:
            model_path: Path, URL, or Hugging Face model ID.
            profile: Analysis profile (quick, standard, deep, attack).
            progress_callback: Optional callback for progress updates.
            download_progress_callback: Optional callback for download progress.

        Returns:
            Analysis result dictionary.
        """
        self._ensure_imported()

        # Resolve model source (handles URLs, HF, and local paths)
        if progress_callback:
            progress_callback(f"Resolving model source: {model_path}...")

        path = await self.resolve_model_source(model_path, download_progress_callback)

        if not path.exists():
            raise AnalysisBridgeError(f"Model not found: {model_path}")

        if progress_callback:
            progress_callback(f"Starting analysis of {path.name}...")

        # Run analysis in thread pool
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._run_analysis(
                model_path=path,
                profile=profile,
                mcp_server_path=None,
                skill_path=None,
            ),
        )

        if progress_callback:
            progress_callback("Analysis complete.")

        return self._convert_result(result, "model", model_path)

    async def analyze_mcp_server(
        self,
        server_path: str,
        profile: str = "infra-standard",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze an MCP server file.

        Args:
            server_path: Path to the MCP server Python file.
            profile: Analysis profile (infra-quick, infra-standard, infra-deep).
            progress_callback: Optional callback for progress updates.

        Returns:
            Analysis result dictionary.
        """
        self._ensure_imported()

        path = Path(server_path)
        if not path.exists():
            raise AnalysisBridgeError(f"Server file not found: {server_path}")

        if progress_callback:
            progress_callback(f"Starting analysis of {path.name}...")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._run_analysis(
                model_path=None,
                profile=profile,
                mcp_server_path=path,
                skill_path=None,
            ),
        )

        if progress_callback:
            progress_callback("Analysis complete.")

        return self._convert_result(result, "mcp_server", server_path)

    async def analyze_skill(
        self,
        skill_path: str,
        profile: str = "infra-quick",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a skill definition file.

        Args:
            skill_path: Path to the skill Markdown file.
            profile: Analysis profile.
            progress_callback: Optional callback for progress updates.

        Returns:
            Analysis result dictionary.
        """
        self._ensure_imported()

        path = Path(skill_path)
        if not path.exists():
            raise AnalysisBridgeError(f"Skill file not found: {skill_path}")

        if progress_callback:
            progress_callback(f"Starting analysis of {path.name}...")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._run_analysis(
                model_path=None,
                profile=profile,
                mcp_server_path=None,
                skill_path=path,
            ),
        )

        if progress_callback:
            progress_callback("Analysis complete.")

        return self._convert_result(result, "skill", skill_path)

    async def compare_targets(
        self,
        targets: List[str],
        comparison_type: str = "safety",
    ) -> ComparisonResult:
        """
        Compare multiple analysis targets.

        Args:
            targets: List of target paths to compare.
            comparison_type: Type of comparison (safety, performance, etc.).

        Returns:
            ComparisonResult with analysis results.
        """
        if len(targets) < 2:
            raise AnalysisBridgeError("Comparison requires at least 2 targets")

        # Analyze each target
        results = []
        for target in targets:
            path = Path(target)
            if path.suffix == ".gguf":
                result = await self.analyze_model(target, profile="quick")
            elif path.suffix in (".py", ".js", ".ts"):
                result = await self.analyze_mcp_server(target, profile="infra-quick")
            elif path.suffix in (".md", ".yaml", ".yml"):
                result = await self.analyze_skill(target, profile="infra-quick")
            else:
                raise AnalysisBridgeError(f"Unknown target type: {target}")

            results.append(result)

        # Build comparison summary
        summary = self._build_comparison_summary(results, comparison_type)

        return ComparisonResult(
            targets=targets,
            comparison_type=comparison_type,
            results=results,
            summary=summary,
        )

    def _run_analysis(
        self,
        model_path: Optional[Path],
        profile: str,
        mcp_server_path: Optional[Path],
        skill_path: Optional[Path],
    ) -> Optional[Path]:
        """
        Run sandbox analysis synchronously.

        Returns path to JSON result file.
        """
        # Run analysis (writes JSON to log_dir)
        self._sandbox_analyze(
            model_path=model_path,
            profile=profile,
            log_dir=self.log_dir,
            tests_override=None,
            format_mode="json",  # Only JSON output
            no_fail_on_test_errors=True,
            mcp_server_path=mcp_server_path,
            skill_path=skill_path,
        )

        # Find the most recent result file
        result_files = sorted(
            self.log_dir.glob("benderbox_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return result_files[0] if result_files else None

    def _convert_result(
        self,
        result_path: Optional[Path],
        target_type: str,
        target_path: str,
    ) -> Dict[str, Any]:
        """Convert sandbox result to NLP-friendly format."""
        if not result_path or not result_path.exists():
            return {
                "error": "Analysis produced no results",
                "target_name": Path(target_path).name,
                "target_type": target_type,
                "target_path": target_path,
            }

        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to standard format
        return {
            "run_id": data.get("run_id", ""),
            "timestamp": data.get("timestamp_utc", datetime.now().isoformat()),
            "profile": data.get("profile", "unknown"),
            "target_type": target_type,
            "target_name": Path(target_path).name,
            "target_path": target_path,
            "summary": {
                "risk": {
                    "level": data.get("overall_risk", {}).get("level", "unknown"),
                    "score": data.get("overall_risk", {}).get("score", 0),
                    "factors": data.get("overall_risk", {}).get("primary_factors", []),
                },
                "safety": data.get("safety", {}),
                "capability": data.get("capability_fingerprint", {}),
            },
            "results": [
                {
                    "test_name": t.get("name", "unknown"),
                    "category": t.get("category", "unknown"),
                    "status": t.get("status", "unknown").lower(),
                    "severity": t.get("severity", "info").lower(),
                    "details": {
                        "message": t.get("details", ""),
                        "metrics": t.get("metrics", {}),
                    },
                    "artifacts": t.get("artifacts", []),
                }
                for t in data.get("tests", [])
            ],
            "errors": data.get("errors", []),
            "model_info": data.get("model", {}),
            "environment": data.get("environment", {}),
        }

    def _build_comparison_summary(
        self,
        results: List[Dict[str, Any]],
        comparison_type: str,
    ) -> Dict[str, Any]:
        """Build comparison summary from results."""
        summary = {
            "comparison_type": comparison_type,
            "target_count": len(results),
        }

        if comparison_type == "safety":
            # Compare risk scores
            scores = [
                (r.get("target_name"), r.get("summary", {}).get("risk", {}).get("score", 0))
                for r in results
            ]
            scores.sort(key=lambda x: x[1])

            summary["safest"] = scores[0][0] if scores else None
            summary["riskiest"] = scores[-1][0] if scores else None
            summary["score_range"] = {
                "min": scores[0][1] if scores else 0,
                "max": scores[-1][1] if scores else 0,
            }
            summary["ranking"] = [{"name": name, "score": score} for name, score in scores]

        return summary

    async def get_available_tests(self) -> List[TestInfo]:
        """Get list of available tests."""
        self._ensure_imported()

        tests = []
        for name, factory in self._test_registry.items():
            try:
                test = factory()
                tests.append(TestInfo(
                    name=name,
                    category=getattr(test, "category", "unknown"),
                    description=getattr(test, "__doc__", "") or "",
                ))
            except Exception:
                tests.append(TestInfo(name=name, category="unknown"))

        return tests

    async def get_available_profiles(self) -> List[ProfileInfo]:
        """Get list of available analysis profiles."""
        return list(PROFILE_INFO.values())

    async def get_latest_report(self, target_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most recent analysis report.

        Args:
            target_name: Optional filter by target name.

        Returns:
            Report dictionary or None.
        """
        result_files = sorted(
            self.log_dir.glob("benderbox_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for path in result_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if target_name is None:
                    return self._convert_result(path, "unknown", data.get("model", {}).get("path", ""))

                if data.get("model", {}).get("name") == target_name:
                    return self._convert_result(path, "model", data.get("model", {}).get("path", ""))
            except Exception:
                continue

        return None

    async def get_report_list(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of recent reports.

        Args:
            limit: Maximum number of reports.

        Returns:
            List of report summaries.
        """
        result_files = sorted(
            self.log_dir.glob("benderbox_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

        reports = []
        for path in result_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                reports.append({
                    "run_id": data.get("run_id", ""),
                    "timestamp": data.get("timestamp_utc", ""),
                    "profile": data.get("profile", ""),
                    "target_name": data.get("model", {}).get("name", "infrastructure"),
                    "risk_level": data.get("overall_risk", {}).get("level", "unknown"),
                    "risk_score": data.get("overall_risk", {}).get("score", 0),
                })
            except Exception:
                continue

        return reports

    def get_status(self) -> Dict[str, Any]:
        """Get analysis bridge status."""
        self._ensure_imported()

        return {
            "version": "3.0.0-alpha",
            "log_dir": str(self.log_dir),
            "available_tests": len(self._test_registry) if self._test_registry else 0,
            "available_profiles": len(PROFILE_INFO),
            "report_count": len(list(self.log_dir.glob("benderbox_*.json"))),
        }


class AnalysisBridgeError(Exception):
    """Exception for analysis bridge errors."""

    pass
