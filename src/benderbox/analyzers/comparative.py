"""
Comparative Analyzer for BenderBox

Compares analysis results across multiple targets to identify
patterns, regressions, and relative safety rankings.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from benderbox.analyzers.semantic import SemanticAnalysisResult, FindingSeverity
from benderbox.analyzers.behavior import BehaviorProfile, RiskLevel

logger = logging.getLogger(__name__)


class ComparisonType(Enum):
    """Types of comparisons."""

    SAFETY = "safety"
    RISK_SCORE = "risk_score"
    FINDINGS = "findings"
    BEHAVIOR = "behavior"
    COMPREHENSIVE = "comprehensive"


class TrendDirection(Enum):
    """Direction of trend between comparisons."""

    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class ComparisonMetric:
    """A single comparison metric."""

    name: str
    target_a_value: Any
    target_b_value: Any
    difference: float
    winner: Optional[str] = None  # Name of better target
    trend: TrendDirection = TrendDirection.UNKNOWN
    notes: str = ""


@dataclass
class FindingDiff:
    """Difference in findings between targets."""

    only_in_a: List[str]  # Finding titles only in target A
    only_in_b: List[str]  # Finding titles only in target B
    common: List[str]  # Finding titles in both
    severity_changes: List[Dict[str, Any]]  # Findings with different severity


@dataclass
class ComparisonResult:
    """Result of comparing two analysis targets."""

    target_a: str
    target_b: str
    comparison_type: ComparisonType
    metrics: List[ComparisonMetric]
    winner: Optional[str] = None
    finding_diff: Optional[FindingDiff] = None
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RankingEntry:
    """Entry in a ranked list of targets."""

    rank: int
    target: str
    score: float
    risk_level: str
    findings_count: int
    critical_count: int
    notes: str = ""


@dataclass
class RankingResult:
    """Result of ranking multiple targets."""

    comparison_type: ComparisonType
    rankings: List[RankingEntry]
    best: str
    worst: str
    score_range: Tuple[float, float]
    summary: str = ""


class ComparativeAnalyzer:
    """
    Compares analysis results across multiple targets.

    Provides:
    - Pairwise comparison of models/servers
    - Ranking of multiple targets
    - Trend analysis over time
    - Regression detection
    """

    def __init__(self, llm_engine=None):
        """
        Initialize ComparativeAnalyzer.

        Args:
            llm_engine: LocalLLMEngine for generating summaries.
        """
        self._llm_engine = llm_engine

    def _set_llm_engine(self, llm_engine) -> None:
        """Set the LLM engine."""
        self._llm_engine = llm_engine

    async def compare_analyses(
        self,
        result_a: Dict[str, Any],
        result_b: Dict[str, Any],
        comparison_type: ComparisonType = ComparisonType.COMPREHENSIVE,
    ) -> ComparisonResult:
        """
        Compare two analysis results.

        Args:
            result_a: First analysis result (from AnalysisBridge).
            result_b: Second analysis result.
            comparison_type: Type of comparison.

        Returns:
            ComparisonResult with comparison metrics.
        """
        target_a = result_a.get("target_name", "Target A")
        target_b = result_b.get("target_name", "Target B")

        metrics = []
        recommendations = []

        # Risk score comparison
        score_a = result_a.get("summary", {}).get("risk", {}).get("score", 0)
        score_b = result_b.get("summary", {}).get("risk", {}).get("score", 0)
        score_diff = score_b - score_a

        metrics.append(ComparisonMetric(
            name="risk_score",
            target_a_value=score_a,
            target_b_value=score_b,
            difference=score_diff,
            winner=target_a if score_a < score_b else target_b if score_b < score_a else None,
            trend=TrendDirection.IMPROVING if score_diff < 0 else TrendDirection.DEGRADING if score_diff > 0 else TrendDirection.STABLE,
            notes=f"Risk score difference: {abs(score_diff):.1f} points",
        ))

        # Risk level comparison
        level_a = result_a.get("summary", {}).get("risk", {}).get("level", "unknown")
        level_b = result_b.get("summary", {}).get("risk", {}).get("level", "unknown")
        level_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "unknown": 0}
        level_diff = level_order.get(level_b.lower(), 0) - level_order.get(level_a.lower(), 0)

        metrics.append(ComparisonMetric(
            name="risk_level",
            target_a_value=level_a,
            target_b_value=level_b,
            difference=level_diff,
            winner=target_a if level_diff > 0 else target_b if level_diff < 0 else None,
            trend=TrendDirection.IMPROVING if level_diff < 0 else TrendDirection.DEGRADING if level_diff > 0 else TrendDirection.STABLE,
        ))

        # Findings comparison
        findings_a = result_a.get("results", [])
        findings_b = result_b.get("results", [])

        finding_diff = self._compare_findings(findings_a, findings_b)

        metrics.append(ComparisonMetric(
            name="total_findings",
            target_a_value=len(findings_a),
            target_b_value=len(findings_b),
            difference=len(findings_b) - len(findings_a),
            winner=target_a if len(findings_a) < len(findings_b) else target_b if len(findings_b) < len(findings_a) else None,
        ))

        # Critical/high findings
        critical_a = sum(1 for f in findings_a if f.get("severity", "").lower() in ("critical", "high"))
        critical_b = sum(1 for f in findings_b if f.get("severity", "").lower() in ("critical", "high"))

        metrics.append(ComparisonMetric(
            name="critical_high_findings",
            target_a_value=critical_a,
            target_b_value=critical_b,
            difference=critical_b - critical_a,
            winner=target_a if critical_a < critical_b else target_b if critical_b < critical_a else None,
        ))

        # Determine overall winner
        winner_votes = {target_a: 0, target_b: 0}
        for m in metrics:
            if m.winner:
                winner_votes[m.winner] += 1

        overall_winner = None
        if winner_votes[target_a] > winner_votes[target_b]:
            overall_winner = target_a
        elif winner_votes[target_b] > winner_votes[target_a]:
            overall_winner = target_b

        # Generate recommendations
        if score_diff > 20:
            recommendations.append(f"{target_b} has significantly higher risk - review findings")
        if score_diff < -20:
            recommendations.append(f"{target_b} shows significant improvement in safety")
        if finding_diff.only_in_b:
            recommendations.append(f"New issues in {target_b}: {', '.join(finding_diff.only_in_b[:3])}")
        if finding_diff.only_in_a:
            recommendations.append(f"Issues resolved in {target_b}: {', '.join(finding_diff.only_in_a[:3])}")

        # Build summary
        summary = await self._generate_comparison_summary(
            target_a, target_b, metrics, finding_diff, overall_winner
        )

        return ComparisonResult(
            target_a=target_a,
            target_b=target_b,
            comparison_type=comparison_type,
            metrics=metrics,
            winner=overall_winner,
            finding_diff=finding_diff,
            summary=summary,
            recommendations=recommendations,
        )

    async def compare_semantic_results(
        self,
        result_a: SemanticAnalysisResult,
        result_b: SemanticAnalysisResult,
    ) -> ComparisonResult:
        """
        Compare two semantic analysis results.

        Args:
            result_a: First semantic analysis result.
            result_b: Second semantic analysis result.

        Returns:
            ComparisonResult for semantic analyses.
        """
        metrics = []

        # Risk scores
        metrics.append(ComparisonMetric(
            name="risk_score",
            target_a_value=result_a.risk_score,
            target_b_value=result_b.risk_score,
            difference=result_b.risk_score - result_a.risk_score,
            winner=result_a.target if result_a.risk_score < result_b.risk_score else result_b.target,
        ))

        # Finding counts by severity
        for severity in FindingSeverity:
            count_a = sum(1 for f in result_a.findings if f.severity == severity)
            count_b = sum(1 for f in result_b.findings if f.severity == severity)
            metrics.append(ComparisonMetric(
                name=f"{severity.value}_findings",
                target_a_value=count_a,
                target_b_value=count_b,
                difference=count_b - count_a,
            ))

        # Build finding diff
        titles_a = {f.title for f in result_a.findings}
        titles_b = {f.title for f in result_b.findings}

        finding_diff = FindingDiff(
            only_in_a=list(titles_a - titles_b),
            only_in_b=list(titles_b - titles_a),
            common=list(titles_a & titles_b),
            severity_changes=[],
        )

        winner = result_a.target if result_a.risk_score < result_b.risk_score else result_b.target

        summary = f"Comparing {result_a.target} vs {result_b.target}: "
        if result_a.risk_score < result_b.risk_score:
            summary += f"{result_a.target} is safer (score: {result_a.risk_score} vs {result_b.risk_score})"
        else:
            summary += f"{result_b.target} is safer (score: {result_b.risk_score} vs {result_a.risk_score})"

        return ComparisonResult(
            target_a=result_a.target,
            target_b=result_b.target,
            comparison_type=ComparisonType.FINDINGS,
            metrics=metrics,
            winner=winner,
            finding_diff=finding_diff,
            summary=summary,
        )

    async def compare_behavior_profiles(
        self,
        profile_a: BehaviorProfile,
        profile_b: BehaviorProfile,
    ) -> ComparisonResult:
        """
        Compare two behavior profiles.

        Args:
            profile_a: First behavior profile.
            profile_b: Second behavior profile.

        Returns:
            ComparisonResult for behavior profiles.
        """
        metrics = []

        # Safety scores
        metrics.append(ComparisonMetric(
            name="safety_score",
            target_a_value=profile_a.overall_safety_score,
            target_b_value=profile_b.overall_safety_score,
            difference=profile_b.overall_safety_score - profile_a.overall_safety_score,
            winner=profile_a.model_name if profile_a.overall_safety_score > profile_b.overall_safety_score else profile_b.model_name,
            notes="Higher is safer",
        ))

        # Jailbreak rates
        metrics.append(ComparisonMetric(
            name="jailbreak_success_rate",
            target_a_value=profile_a.jailbreak_success_rate,
            target_b_value=profile_b.jailbreak_success_rate,
            difference=profile_b.jailbreak_success_rate - profile_a.jailbreak_success_rate,
            winner=profile_a.model_name if profile_a.jailbreak_success_rate < profile_b.jailbreak_success_rate else profile_b.model_name,
            notes="Lower is better",
        ))

        # Unsafe outputs
        metrics.append(ComparisonMetric(
            name="unsafe_output_ratio",
            target_a_value=profile_a.unsafe_outputs / max(profile_a.total_outputs, 1),
            target_b_value=profile_b.unsafe_outputs / max(profile_b.total_outputs, 1),
            difference=(profile_b.unsafe_outputs / max(profile_b.total_outputs, 1)) - (profile_a.unsafe_outputs / max(profile_a.total_outputs, 1)),
        ))

        winner = profile_a.model_name if profile_a.overall_safety_score > profile_b.overall_safety_score else profile_b.model_name

        summary = f"{winner} has better behavioral safety profile"

        return ComparisonResult(
            target_a=profile_a.model_name,
            target_b=profile_b.model_name,
            comparison_type=ComparisonType.BEHAVIOR,
            metrics=metrics,
            winner=winner,
            summary=summary,
        )

    async def rank_targets(
        self,
        results: List[Dict[str, Any]],
        ranking_type: ComparisonType = ComparisonType.SAFETY,
    ) -> RankingResult:
        """
        Rank multiple targets by safety.

        Args:
            results: List of analysis results.
            ranking_type: Type of ranking.

        Returns:
            RankingResult with ordered rankings.
        """
        if not results:
            return RankingResult(
                comparison_type=ranking_type,
                rankings=[],
                best="",
                worst="",
                score_range=(0, 0),
                summary="No targets to rank",
            )

        # Extract scores and build ranking entries
        entries = []
        for result in results:
            target = result.get("target_name", "Unknown")
            score = result.get("summary", {}).get("risk", {}).get("score", 0)
            level = result.get("summary", {}).get("risk", {}).get("level", "unknown")
            findings = result.get("results", [])
            critical = sum(1 for f in findings if f.get("severity", "").lower() == "critical")

            entries.append({
                "target": target,
                "score": score,
                "level": level,
                "findings_count": len(findings),
                "critical_count": critical,
            })

        # Sort by risk score (lower is better)
        entries.sort(key=lambda x: x["score"])

        # Build ranking entries
        rankings = []
        for i, entry in enumerate(entries, 1):
            rankings.append(RankingEntry(
                rank=i,
                target=entry["target"],
                score=entry["score"],
                risk_level=entry["level"],
                findings_count=entry["findings_count"],
                critical_count=entry["critical_count"],
            ))

        scores = [e["score"] for e in entries]
        score_range = (min(scores), max(scores))

        summary = f"Ranked {len(rankings)} targets. "
        summary += f"Best: {rankings[0].target} (score: {rankings[0].score}). "
        if len(rankings) > 1:
            summary += f"Worst: {rankings[-1].target} (score: {rankings[-1].score})."

        return RankingResult(
            comparison_type=ranking_type,
            rankings=rankings,
            best=rankings[0].target,
            worst=rankings[-1].target if rankings else "",
            score_range=score_range,
            summary=summary,
        )

    async def detect_regression(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        threshold: float = 10.0,
    ) -> Tuple[bool, str]:
        """
        Detect if safety regressed between two analyses.

        Args:
            before: Earlier analysis result.
            after: Later analysis result.
            threshold: Score difference threshold for regression.

        Returns:
            Tuple of (has_regression, explanation).
        """
        comparison = await self.compare_analyses(before, after)

        risk_metric = next((m for m in comparison.metrics if m.name == "risk_score"), None)
        if not risk_metric:
            return False, "Could not compare risk scores"

        diff = risk_metric.difference  # Positive means after is worse

        if diff > threshold:
            return True, f"Safety regression detected: risk score increased by {diff:.1f} points"

        critical_metric = next((m for m in comparison.metrics if m.name == "critical_high_findings"), None)
        if critical_metric and critical_metric.difference > 0:
            return True, f"Regression: {int(critical_metric.difference)} new critical/high findings"

        if comparison.finding_diff and comparison.finding_diff.only_in_b:
            new_issues = len(comparison.finding_diff.only_in_b)
            if new_issues > 3:
                return True, f"Potential regression: {new_issues} new issues detected"

        return False, "No significant regression detected"

    def _compare_findings(
        self,
        findings_a: List[Dict[str, Any]],
        findings_b: List[Dict[str, Any]],
    ) -> FindingDiff:
        """Compare findings between two results."""
        # Extract finding identifiers (test_name)
        names_a = {f.get("test_name", "") for f in findings_a if f.get("test_name")}
        names_b = {f.get("test_name", "") for f in findings_b if f.get("test_name")}

        only_in_a = list(names_a - names_b)
        only_in_b = list(names_b - names_a)
        common = list(names_a & names_b)

        # Check for severity changes in common findings
        severity_changes = []
        a_by_name = {f.get("test_name"): f for f in findings_a}
        b_by_name = {f.get("test_name"): f for f in findings_b}

        for name in common:
            if name in a_by_name and name in b_by_name:
                sev_a = a_by_name[name].get("severity", "")
                sev_b = b_by_name[name].get("severity", "")
                if sev_a != sev_b:
                    severity_changes.append({
                        "finding": name,
                        "before": sev_a,
                        "after": sev_b,
                    })

        return FindingDiff(
            only_in_a=only_in_a,
            only_in_b=only_in_b,
            common=common,
            severity_changes=severity_changes,
        )

    async def _generate_comparison_summary(
        self,
        target_a: str,
        target_b: str,
        metrics: List[ComparisonMetric],
        finding_diff: FindingDiff,
        winner: Optional[str],
    ) -> str:
        """Generate a summary of the comparison."""
        if self._llm_engine:
            try:
                metrics_str = "\n".join([
                    f"- {m.name}: {m.target_a_value} vs {m.target_b_value}"
                    for m in metrics
                ])

                prompt = f"""Summarize this security comparison in 2-3 sentences:

Comparing: {target_a} vs {target_b}

Metrics:
{metrics_str}

New issues in {target_b}: {len(finding_diff.only_in_b)}
Resolved issues: {len(finding_diff.only_in_a)}
Winner: {winner or 'tie'}

Summary:"""

                return await self._llm_engine.generate(
                    prompt=prompt,
                    model_type="analysis",
                    max_tokens=150,
                    temperature=0.5,
                )
            except Exception as e:
                logger.warning(f"LLM summary generation failed: {e}")

        # Fallback template summary
        if winner:
            return f"{winner} has better security posture. New issues: {len(finding_diff.only_in_b)}, Resolved: {len(finding_diff.only_in_a)}."
        return f"Security comparison between {target_a} and {target_b} shows similar risk levels."

    async def compare_models(
        self,
        model_paths: List[str],
        analysis_bridge=None,
        profile: str = "quick",
    ) -> RankingResult:
        """
        Compare multiple model files by running analysis on each.

        Args:
            model_paths: List of paths to model files.
            analysis_bridge: AnalysisBridge for running analyses.
            profile: Analysis profile to use.

        Returns:
            RankingResult with models ranked by safety.
        """
        if not analysis_bridge:
            raise ValueError("analysis_bridge is required for model comparison")

        results = []
        for path in model_paths:
            try:
                result = await analysis_bridge.analyze_model(path, profile)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze {path}: {e}")
                # Add placeholder result
                results.append({
                    "target_name": path,
                    "summary": {"risk": {"score": 100, "level": "unknown"}},
                    "results": [],
                    "error": str(e),
                })

        return await self.rank_targets(results)

    async def compare_reports(
        self,
        report_ids: List[str],
        report_db=None,
    ) -> RankingResult:
        """
        Compare multiple stored reports.

        Args:
            report_ids: List of report IDs to compare.
            report_db: ReportDatabase for retrieving reports.

        Returns:
            RankingResult with reports ranked.
        """
        if not report_db:
            raise ValueError("report_db is required for report comparison")

        results = []
        for report_id in report_ids:
            try:
                report = await report_db.get_report(report_id)
                if report:
                    results.append(report)
            except Exception as e:
                logger.warning(f"Failed to get report {report_id}: {e}")

        return await self.rank_targets(results)

    async def detect_drift(
        self,
        current_result: Dict[str, Any],
        baseline_id: str,
        report_db=None,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Detect behavioral drift from a baseline analysis.

        Compares current analysis against a historical baseline to identify:
        - Score changes beyond threshold
        - New/resolved findings
        - Category distribution changes
        - Trend direction

        Args:
            current_result: Current analysis result.
            baseline_id: Report ID of the baseline analysis.
            report_db: ReportDatabase for retrieving baseline.
            thresholds: Custom thresholds for drift detection.

        Returns:
            Dict with drift analysis results.
        """
        if not report_db:
            raise ValueError("report_db is required for drift detection")

        # Default thresholds
        default_thresholds = {
            "score_change": 10.0,  # Points
            "new_findings_ratio": 0.2,  # 20% new findings
            "category_change_count": 2,  # Number of new categories
        }
        thresholds = {**default_thresholds, **(thresholds or {})}

        # Get baseline
        baseline = await report_db.get_report(baseline_id)
        if not baseline:
            return {
                "has_drift": False,
                "error": f"Baseline report {baseline_id} not found",
                "drift_indicators": [],
            }

        # Compare
        comparison = await self.compare_analyses(baseline, current_result)

        drift_indicators = []

        # Check score drift
        risk_metric = next(
            (m for m in comparison.metrics if m.name == "risk_score"), None
        )
        if risk_metric:
            score_change = abs(risk_metric.difference)
            if score_change >= thresholds["score_change"]:
                direction = "increased" if risk_metric.difference > 0 else "decreased"
                drift_indicators.append({
                    "type": "score_drift",
                    "description": f"Risk score {direction} by {score_change:.1f} points",
                    "baseline_value": risk_metric.target_a_value,
                    "current_value": risk_metric.target_b_value,
                    "change": risk_metric.difference,
                    "severity": "high" if score_change >= 20 else "medium",
                    "trend": risk_metric.trend.value,
                })

        # Check findings drift
        if comparison.finding_diff:
            baseline_count = len(comparison.finding_diff.common) + len(comparison.finding_diff.only_in_a)
            new_ratio = len(comparison.finding_diff.only_in_b) / max(baseline_count, 1)

            if new_ratio >= thresholds["new_findings_ratio"]:
                drift_indicators.append({
                    "type": "findings_drift",
                    "description": f"{len(comparison.finding_diff.only_in_b)} new findings detected",
                    "new_findings": comparison.finding_diff.only_in_b[:5],
                    "resolved_findings": comparison.finding_diff.only_in_a[:5],
                    "severity": "high" if new_ratio >= 0.4 else "medium",
                })

            # Check for severity changes
            if comparison.finding_diff.severity_changes:
                escalations = [
                    c for c in comparison.finding_diff.severity_changes
                    if self._is_severity_escalation(c["before"], c["after"])
                ]
                if escalations:
                    drift_indicators.append({
                        "type": "severity_escalation",
                        "description": f"{len(escalations)} findings escalated in severity",
                        "changes": escalations[:3],
                        "severity": "high",
                    })

        # Statistical significance (simple check)
        is_significant = len(drift_indicators) > 0 and any(
            d.get("severity") == "high" for d in drift_indicators
        )

        # Trend detection
        trend = TrendDirection.STABLE
        if drift_indicators:
            if risk_metric and risk_metric.difference > 0:
                trend = TrendDirection.DEGRADING
            elif risk_metric and risk_metric.difference < 0:
                trend = TrendDirection.IMPROVING

        return {
            "has_drift": len(drift_indicators) > 0,
            "is_significant": is_significant,
            "drift_indicators": drift_indicators,
            "trend": trend.value,
            "baseline_id": baseline_id,
            "baseline_target": baseline.get("target_name", "unknown"),
            "current_target": current_result.get("target_name", "unknown"),
            "comparison": {
                "score_change": risk_metric.difference if risk_metric else 0,
                "new_findings": len(comparison.finding_diff.only_in_b) if comparison.finding_diff else 0,
                "resolved_findings": len(comparison.finding_diff.only_in_a) if comparison.finding_diff else 0,
            },
            "recommendations": comparison.recommendations,
        }

    async def get_historical_trend(
        self,
        target_name: str,
        report_db=None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Analyze trend over historical reports for a target.

        Args:
            target_name: Name of the target.
            report_db: ReportDatabase for retrieving reports.
            limit: Maximum number of historical reports.

        Returns:
            Dict with trend analysis.
        """
        if not report_db:
            raise ValueError("report_db is required for trend analysis")

        # Get report history
        from benderbox.storage.report_db import ReportFilters
        filters = ReportFilters(target_name=target_name, limit=limit)
        summaries = await report_db.list_reports(filters)

        if len(summaries) < 2:
            return {
                "has_trend": False,
                "message": "Not enough historical data for trend analysis",
                "data_points": len(summaries),
            }

        # Extract scores over time
        data_points = [
            {
                "timestamp": s.timestamp.isoformat(),
                "score": s.risk_score,
                "level": s.risk_level,
                "findings": s.finding_count,
            }
            for s in summaries
        ]

        # Calculate trend
        scores = [s.risk_score for s in summaries]
        first_half = sum(scores[:len(scores)//2]) / (len(scores)//2)
        second_half = sum(scores[len(scores)//2:]) / max(len(scores) - len(scores)//2, 1)

        if second_half < first_half - 5:
            trend = TrendDirection.IMPROVING
            trend_description = "Security posture is improving over time"
        elif second_half > first_half + 5:
            trend = TrendDirection.DEGRADING
            trend_description = "Security posture is degrading over time"
        else:
            trend = TrendDirection.STABLE
            trend_description = "Security posture is stable"

        # Score statistics
        import statistics
        score_stats = {
            "min": min(scores),
            "max": max(scores),
            "mean": statistics.mean(scores),
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
        }

        return {
            "has_trend": True,
            "target_name": target_name,
            "trend": trend.value,
            "trend_description": trend_description,
            "data_points": data_points,
            "score_statistics": score_stats,
            "total_reports": len(summaries),
            "first_report": data_points[-1]["timestamp"],
            "latest_report": data_points[0]["timestamp"],
        }

    def _is_severity_escalation(self, before: str, after: str) -> bool:
        """Check if severity changed to a higher level."""
        order = ["info", "low", "medium", "high", "critical"]
        try:
            before_idx = order.index(before.lower())
            after_idx = order.index(after.lower())
            return after_idx > before_idx
        except ValueError:
            return False
