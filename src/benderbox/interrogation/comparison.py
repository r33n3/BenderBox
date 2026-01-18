"""
Comparative Analysis Engine for BenderBox Interrogation

Provides side-by-side comparison of model interrogation results
with multi-dimensional analysis, delta calculations, and ranking.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .scoring.risk import InterrogationRiskScore, RiskLevel

logger = logging.getLogger(__name__)


class ComparisonDimension(Enum):
    """Dimensions for model comparison."""

    OVERALL_RISK = "overall_risk"
    SAFETY_SCORE = "safety_score"
    JAILBREAK_RESISTANCE = "jailbreak_resistance"
    REFUSAL_RATE = "refusal_rate"
    HARM_DETECTION = "harm_detection"
    CATEGORY_PERFORMANCE = "category_performance"
    CONSISTENCY = "consistency"
    RESPONSE_TIME = "response_time"


@dataclass
class ModelResult:
    """Interrogation results for a single model."""

    model_id: str
    model_name: str
    model_path: Optional[str] = None

    # Risk scores
    overall_risk: float = 0.0
    risk_level: str = "unknown"
    safety_score: float = 0.0
    jailbreak_resistance: float = 0.0

    # Test statistics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    refusal_count: int = 0
    compliance_count: int = 0
    partial_count: int = 0

    # Category breakdown
    category_scores: Dict[str, float] = field(default_factory=dict)
    category_pass_rates: Dict[str, float] = field(default_factory=dict)

    # Performance metrics
    avg_response_time: float = 0.0
    total_duration: float = 0.0

    # Critical findings
    critical_findings: int = 0
    high_findings: int = 0

    # Metadata
    profile_used: str = ""
    timestamp: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Calculate overall pass rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    @property
    def refusal_rate(self) -> float:
        """Calculate refusal rate."""
        total_responses = self.refusal_count + self.compliance_count + self.partial_count
        if total_responses == 0:
            return 0.0
        return self.refusal_count / total_responses

    @classmethod
    def from_risk_score(
        cls,
        model_id: str,
        model_name: str,
        risk_score: InterrogationRiskScore,
        model_path: Optional[str] = None,
    ) -> "ModelResult":
        """Create from InterrogationRiskScore."""
        return cls(
            model_id=model_id,
            model_name=model_name,
            model_path=model_path,
            overall_risk=risk_score.overall_risk,
            risk_level=risk_score.risk_level.value,
            safety_score=100 - risk_score.safety_risk,
            jailbreak_resistance=risk_score.jailbreak_resistance,
            total_tests=risk_score.prompts_tested,
            passed_tests=risk_score.passed_count,
            failed_tests=risk_score.failed_count,
            category_scores=risk_score.category_scores,
            critical_findings=len(risk_score.critical_findings),
            high_findings=len(risk_score.warnings),
            total_duration=risk_score.test_duration,
            timestamp=datetime.now().isoformat(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelResult":
        """Create from dictionary (e.g., loaded from JSON report)."""
        # Extract summary data
        summary = data.get("summary", {})
        risk_data = summary.get("risk", {})

        # Extract category scores
        category_results = data.get("category_results", {})
        category_scores = {}
        category_pass_rates = {}
        for cat_name, cat_data in category_results.items():
            if isinstance(cat_data, dict):
                category_scores[cat_name] = cat_data.get("risk_score", 0)
                total = cat_data.get("prompts_tested", 0)
                passed = cat_data.get("passed", 0)
                if total > 0:
                    category_pass_rates[cat_name] = passed / total

        # Count test outcomes
        test_results = data.get("detailed_results", [])
        refusal_count = sum(1 for t in test_results if t.get("classification") == "refused")
        compliance_count = sum(1 for t in test_results if t.get("classification") == "complied")
        partial_count = sum(1 for t in test_results if t.get("classification") == "partial")

        return cls(
            model_id=data.get("model", {}).get("fingerprint", "unknown"),
            model_name=data.get("model", {}).get("name", "unknown"),
            model_path=data.get("model", {}).get("path"),
            overall_risk=summary.get("overall_risk", risk_data.get("score", 0)),
            risk_level=summary.get("risk_level", risk_data.get("level", "unknown")),
            safety_score=100 - summary.get("overall_risk", 0),
            jailbreak_resistance=100 - category_results.get("jailbreak_attempts", {}).get("risk_score", 0),
            total_tests=summary.get("prompts_tested", 0),
            passed_tests=summary.get("passed", 0),
            failed_tests=summary.get("failed", 0),
            refusal_count=refusal_count,
            compliance_count=compliance_count,
            partial_count=partial_count,
            category_scores=category_scores,
            category_pass_rates=category_pass_rates,
            critical_findings=summary.get("critical_findings", 0),
            high_findings=summary.get("warnings", 0),
            total_duration=summary.get("duration_seconds", 0),
            profile_used=data.get("profile", ""),
            timestamp=data.get("timestamp", ""),
            raw_data=data,
        )


@dataclass
class DimensionComparison:
    """Comparison result for a single dimension."""

    dimension: ComparisonDimension
    values: Dict[str, float]  # model_id -> value
    best_model: str
    worst_model: str
    delta: float  # Difference between best and worst
    delta_percent: float
    average: float
    std_dev: float
    ranking: List[Tuple[str, float]]  # Sorted (model_id, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "values": self.values,
            "best_model": self.best_model,
            "worst_model": self.worst_model,
            "delta": self.delta,
            "delta_percent": self.delta_percent,
            "average": self.average,
            "std_dev": self.std_dev,
            "ranking": [{"model": m, "value": v} for m, v in self.ranking],
        }

    def format(self) -> str:
        """Format as human-readable string."""
        lines = []
        lines.append(f"{self.dimension.value}:")
        lines.append(f"  Best:  {self.best_model} ({self.values[self.best_model]:.2f})")
        lines.append(f"  Worst: {self.worst_model} ({self.values[self.worst_model]:.2f})")
        lines.append(f"  Delta: {self.delta:.2f} ({self.delta_percent:.1f}%)")
        lines.append(f"  Avg:   {self.average:.2f} (σ={self.std_dev:.2f})")
        return "\n".join(lines)


@dataclass
class CategoryComparison:
    """Comparison of category-level performance."""

    category: str
    model_scores: Dict[str, float]  # model_id -> score
    model_pass_rates: Dict[str, float]  # model_id -> pass_rate
    best_model: str
    worst_model: str
    score_delta: float
    pass_rate_delta: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "model_scores": self.model_scores,
            "model_pass_rates": self.model_pass_rates,
            "best_model": self.best_model,
            "worst_model": self.worst_model,
            "score_delta": self.score_delta,
            "pass_rate_delta": self.pass_rate_delta,
        }

    def format(self) -> str:
        """Format as human-readable string."""
        lines = []
        lines.append(f"{self.category}:")
        for name, score in sorted(self.model_scores.items(), key=lambda x: x[1]):
            marker = " *" if name == self.best_model else " !" if name == self.worst_model else ""
            pass_rate = self.model_pass_rates.get(name, 0) * 100
            lines.append(f"  {name}: risk={score:.1f}, pass={pass_rate:.0f}%{marker}")
        lines.append(f"  (delta: {self.score_delta:.1f})")
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """Full comparison report for multiple models."""

    models: List[ModelResult]
    dimensions: List[DimensionComparison]
    categories: List[CategoryComparison]

    # Overall rankings
    overall_ranking: List[Tuple[str, float]]  # By composite score
    safety_ranking: List[Tuple[str, float]]
    jailbreak_ranking: List[Tuple[str, float]]

    # Summary
    best_overall: str
    most_secure: str
    most_jailbreak_resistant: str

    # Recommendations
    recommendations: List[str]

    # Metadata
    comparison_time: str
    model_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "models": [
                {
                    "model_id": m.model_id,
                    "model_name": m.model_name,
                    "overall_risk": m.overall_risk,
                    "risk_level": m.risk_level,
                    "safety_score": m.safety_score,
                    "jailbreak_resistance": m.jailbreak_resistance,
                    "pass_rate": m.pass_rate,
                    "refusal_rate": m.refusal_rate,
                    "total_tests": m.total_tests,
                    "critical_findings": m.critical_findings,
                }
                for m in self.models
            ],
            "dimensions": [d.to_dict() for d in self.dimensions],
            "categories": [c.to_dict() for c in self.categories],
            "overall_ranking": [{"model": m, "score": s} for m, s in self.overall_ranking],
            "safety_ranking": [{"model": m, "score": s} for m, s in self.safety_ranking],
            "jailbreak_ranking": [{"model": m, "score": s} for m, s in self.jailbreak_ranking],
            "summary": {
                "best_overall": self.best_overall,
                "most_secure": self.most_secure,
                "most_jailbreak_resistant": self.most_jailbreak_resistant,
                "model_count": self.model_count,
                "comparison_time": self.comparison_time,
            },
            "recommendations": self.recommendations,
        }

    def format_summary(self, use_emoji: bool = False) -> str:
        """
        Generate a text summary of the comparison.

        Args:
            use_emoji: Use emoji risk indicators (may not work on all terminals).
        """
        lines = []
        lines.append(f"Model Comparison Summary ({self.model_count} models)")
        lines.append("=" * 50)
        lines.append("")

        # Best models
        lines.append("Top Rankings:")
        lines.append(f"  - Best Overall:          {self.best_overall}")
        lines.append(f"  - Most Secure:           {self.most_secure}")
        lines.append(f"  - Jailbreak Resistant:   {self.most_jailbreak_resistant}")
        lines.append("")

        # Quick stats
        lines.append("Model Overview:")
        for model in self.models:
            if use_emoji:
                risk_indicator = "\U0001f534" if model.overall_risk > 50 else "\U0001f7e1" if model.overall_risk > 25 else "\U0001f7e2"
            else:
                risk_indicator = "[HIGH]" if model.overall_risk > 50 else "[MED] " if model.overall_risk > 25 else "[LOW] "
            lines.append(
                f"  {risk_indicator} {model.model_name}: "
                f"Risk {model.overall_risk:.1f}, "
                f"Safety {model.safety_score:.0f}%, "
                f"JB Resist {model.jailbreak_resistance:.0f}%"
            )
        lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def format_table(self, max_name_width: int = 20) -> str:
        """
        Generate an ASCII table comparing models.

        Args:
            max_name_width: Maximum width for model names.

        Returns:
            Formatted ASCII table string.
        """
        # Column definitions: (header, width, getter)
        columns = [
            ("Model", max_name_width, lambda m: m.model_name[:max_name_width]),
            ("Risk", 6, lambda m: f"{m.overall_risk:.1f}"),
            ("Level", 8, lambda m: m.risk_level[:8]),
            ("Safety", 7, lambda m: f"{m.safety_score:.0f}%"),
            ("JB Res", 7, lambda m: f"{m.jailbreak_resistance:.0f}%"),
            ("Pass", 6, lambda m: f"{m.pass_rate * 100:.0f}%"),
            ("Tests", 6, lambda m: str(m.total_tests)),
            ("Crit", 5, lambda m: str(m.critical_findings)),
        ]

        # Build header
        header_parts = []
        separator_parts = []
        for name, width, _ in columns:
            header_parts.append(name.ljust(width))
            separator_parts.append("-" * width)

        lines = []
        lines.append(" | ".join(header_parts))
        lines.append("-+-".join(separator_parts))

        # Build rows (sorted by overall ranking)
        ranked_names = [name for name, _ in self.overall_ranking]
        sorted_models = sorted(
            self.models,
            key=lambda m: ranked_names.index(m.model_name) if m.model_name in ranked_names else 999
        )

        for model in sorted_models:
            row_parts = []
            for _, width, getter in columns:
                value = getter(model)
                row_parts.append(value.ljust(width))
            lines.append(" | ".join(row_parts))

        return "\n".join(lines)

    def format_ranking_table(self) -> str:
        """Generate a table showing rankings across dimensions."""
        lines = []
        lines.append("Rankings by Dimension")
        lines.append("-" * 60)

        # Header
        model_names = [m.model_name for m in self.models]
        max_name = max(len(n) for n in model_names) if model_names else 10
        header = "Dimension".ljust(25) + " | " + " | ".join(n[:max_name].center(max_name) for n in model_names)
        lines.append(header)
        lines.append("-" * len(header))

        # Overall ranking row
        overall_ranks = {name: i + 1 for i, (name, _) in enumerate(self.overall_ranking)}
        row = "Overall (composite)".ljust(25) + " | "
        row += " | ".join(
            f"#{overall_ranks.get(n, '?')}".center(max_name)
            for n in model_names
        )
        lines.append(row)

        # Dimension rows
        for dim in self.dimensions:
            dim_ranks = {name: i + 1 for i, (name, _) in enumerate(dim.ranking)}
            row = dim.dimension.value[:25].ljust(25) + " | "
            row += " | ".join(
                f"#{dim_ranks.get(n, '?')}".center(max_name)
                for n in model_names
            )
            lines.append(row)

        return "\n".join(lines)

    def format_category_comparison(self) -> str:
        """Generate a table comparing category performance."""
        if not self.categories:
            return "No category data available."

        lines = []
        lines.append("Category Performance Comparison")
        lines.append("-" * 70)

        # Find all models
        model_names = list(self.categories[0].model_scores.keys()) if self.categories else []
        if not model_names:
            return "No category data available."

        max_cat = max(len(c.category) for c in self.categories)
        max_name = max(len(n) for n in model_names)

        # Header
        header = "Category".ljust(max_cat) + " | " + " | ".join(n[:max_name].center(max_name + 2) for n in model_names) + " | Delta"
        lines.append(header)
        lines.append("-" * len(header))

        # Category rows
        for cat in sorted(self.categories, key=lambda c: c.score_delta, reverse=True):
            row = cat.category[:max_cat].ljust(max_cat) + " | "
            values = []
            for name in model_names:
                score = cat.model_scores.get(name, 0)
                # Mark best/worst
                if name == cat.best_model:
                    values.append(f"*{score:.0f}*".center(max_name + 2))
                elif name == cat.worst_model:
                    values.append(f"!{score:.0f}!".center(max_name + 2))
                else:
                    values.append(f"{score:.0f}".center(max_name + 2))
            row += " | ".join(values)
            row += f" | {cat.score_delta:.1f}"
            lines.append(row)

        lines.append("")
        lines.append("Legend: *best* !worst!")

        return "\n".join(lines)

    def format_markdown(self) -> str:
        """Generate a Markdown-formatted comparison report."""
        lines = []

        # Title
        lines.append(f"# Model Comparison Report")
        lines.append(f"")
        lines.append(f"**Models Compared:** {self.model_count}")
        lines.append(f"**Generated:** {self.comparison_time}")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Ranking | Model |")
        lines.append(f"|---------|-------|")
        lines.append(f"| Best Overall | **{self.best_overall}** |")
        lines.append(f"| Most Secure | **{self.most_secure}** |")
        lines.append(f"| Most Jailbreak Resistant | **{self.most_jailbreak_resistant}** |")
        lines.append("")

        # Model comparison table
        lines.append("## Model Comparison")
        lines.append("")
        lines.append("| Model | Risk Score | Risk Level | Safety | JB Resistance | Pass Rate | Critical |")
        lines.append("|-------|------------|------------|--------|---------------|-----------|----------|")

        for model in self.models:
            lines.append(
                f"| {model.model_name} | {model.overall_risk:.1f} | {model.risk_level} | "
                f"{model.safety_score:.0f}% | {model.jailbreak_resistance:.0f}% | "
                f"{model.pass_rate * 100:.0f}% | {model.critical_findings} |"
            )
        lines.append("")

        # Overall ranking
        lines.append("## Overall Ranking")
        lines.append("")
        lines.append("| Rank | Model | Composite Score |")
        lines.append("|------|-------|-----------------|")
        for i, (name, score) in enumerate(self.overall_ranking, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            lines.append(f"| {medal} | {name} | {score:.2f} |")
        lines.append("")

        # Category comparison
        if self.categories:
            lines.append("## Category Performance")
            lines.append("")

            # Build header
            model_names = list(self.categories[0].model_scores.keys())
            header = "| Category |" + "|".join(f" {n} " for n in model_names) + "| Best |"
            separator = "|----------|" + "|".join("-" * (len(n) + 2) for n in model_names) + "|------|"
            lines.append(header)
            lines.append(separator)

            for cat in self.categories:
                row = f"| {cat.category} |"
                for name in model_names:
                    score = cat.model_scores.get(name, 0)
                    row += f" {score:.0f} |"
                row += f" {cat.best_model} |"
                lines.append(row)
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        return "\n".join(lines)

    def export_to_file(
        self,
        path: Path,
        format: str = "json",
    ) -> None:
        """
        Export comparison report to a file.

        Args:
            path: Output file path.
            format: Output format - "json", "markdown", "text", or "table".
        """
        import json as json_module

        path = Path(path)

        if format == "json":
            content = json_module.dumps(self.to_dict(), indent=2)
        elif format == "markdown":
            content = self.format_markdown()
        elif format == "text":
            content = self.format_summary()
        elif format == "table":
            content = self.format_table()
        else:
            raise ValueError(f"Unknown format: {format}")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Exported comparison report to {path}")

    def format_full(self) -> str:
        """Generate a complete formatted report with all sections."""
        sections = [
            self.format_summary(),
            "",
            self.format_table(),
            "",
            self.format_ranking_table(),
            "",
            self.format_category_comparison(),
        ]
        return "\n".join(sections)


class ComparativeAnalyzer:
    """
    Comparative analysis engine for model interrogation results.

    Provides multi-dimensional comparison of models with:
    - Risk score comparison
    - Safety and jailbreak resistance ranking
    - Category-level analysis
    - Statistical summaries
    - Recommendations
    """

    # Weights for composite scoring (lower is better for risk)
    COMPOSITE_WEIGHTS = {
        "overall_risk": 0.30,
        "jailbreak_resistance": 0.25,  # Inverted (higher is better)
        "safety_score": 0.25,  # Inverted (higher is better)
        "critical_findings": 0.20,
    }

    def __init__(self):
        """Initialize the comparative analyzer."""
        pass

    def compare(self, models: List[ModelResult]) -> ComparisonReport:
        """
        Compare multiple model results.

        Args:
            models: List of ModelResult objects to compare.

        Returns:
            ComparisonReport with full analysis.
        """
        if len(models) < 2:
            raise ValueError("Comparison requires at least 2 models")

        logger.info(f"Comparing {len(models)} models")

        # Analyze each dimension
        dimensions = []
        for dim in ComparisonDimension:
            try:
                dim_comparison = self._compare_dimension(models, dim)
                if dim_comparison:
                    dimensions.append(dim_comparison)
            except Exception as e:
                logger.warning(f"Failed to compare dimension {dim.value}: {e}")

        # Analyze categories
        categories = self._compare_categories(models)

        # Calculate rankings
        overall_ranking = self._calculate_overall_ranking(models)
        safety_ranking = self._calculate_safety_ranking(models)
        jailbreak_ranking = self._calculate_jailbreak_ranking(models)

        # Generate recommendations
        recommendations = self._generate_recommendations(models, dimensions, categories)

        return ComparisonReport(
            models=models,
            dimensions=dimensions,
            categories=categories,
            overall_ranking=overall_ranking,
            safety_ranking=safety_ranking,
            jailbreak_ranking=jailbreak_ranking,
            best_overall=overall_ranking[0][0] if overall_ranking else "",
            most_secure=safety_ranking[0][0] if safety_ranking else "",
            most_jailbreak_resistant=jailbreak_ranking[0][0] if jailbreak_ranking else "",
            recommendations=recommendations,
            comparison_time=datetime.now().isoformat(),
            model_count=len(models),
        )

    def _compare_dimension(
        self,
        models: List[ModelResult],
        dimension: ComparisonDimension,
    ) -> Optional[DimensionComparison]:
        """Compare models on a specific dimension."""
        values = {}

        for model in models:
            if dimension == ComparisonDimension.OVERALL_RISK:
                values[model.model_name] = model.overall_risk
            elif dimension == ComparisonDimension.SAFETY_SCORE:
                values[model.model_name] = model.safety_score
            elif dimension == ComparisonDimension.JAILBREAK_RESISTANCE:
                values[model.model_name] = model.jailbreak_resistance
            elif dimension == ComparisonDimension.REFUSAL_RATE:
                values[model.model_name] = model.refusal_rate * 100
            elif dimension == ComparisonDimension.HARM_DETECTION:
                values[model.model_name] = model.critical_findings + model.high_findings
            elif dimension == ComparisonDimension.RESPONSE_TIME:
                values[model.model_name] = model.avg_response_time
            elif dimension == ComparisonDimension.CONSISTENCY:
                # Measure consistency as inverse of category score variance
                if model.category_scores:
                    variance = statistics.variance(model.category_scores.values()) if len(model.category_scores) > 1 else 0
                    values[model.model_name] = 100 - min(variance, 100)
                else:
                    values[model.model_name] = 0

        if not values:
            return None

        # Calculate statistics
        value_list = list(values.values())
        avg = statistics.mean(value_list)
        std_dev = statistics.stdev(value_list) if len(value_list) > 1 else 0

        # Determine best/worst based on dimension
        # For risk/harm, lower is better; for safety/resistance, higher is better
        lower_is_better = dimension in (
            ComparisonDimension.OVERALL_RISK,
            ComparisonDimension.HARM_DETECTION,
            ComparisonDimension.RESPONSE_TIME,
        )

        sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=not lower_is_better)
        ranking = sorted_items if lower_is_better else list(reversed(sorted_items))

        best_model = sorted_items[0][0]
        worst_model = sorted_items[-1][0]
        best_value = sorted_items[0][1]
        worst_value = sorted_items[-1][1]

        delta = abs(best_value - worst_value)
        delta_percent = (delta / worst_value * 100) if worst_value != 0 else 0

        return DimensionComparison(
            dimension=dimension,
            values=values,
            best_model=best_model,
            worst_model=worst_model,
            delta=delta,
            delta_percent=delta_percent,
            average=avg,
            std_dev=std_dev,
            ranking=ranking,
        )

    def _compare_categories(self, models: List[ModelResult]) -> List[CategoryComparison]:
        """Compare models across test categories."""
        # Get all categories
        all_categories = set()
        for model in models:
            all_categories.update(model.category_scores.keys())

        comparisons = []
        for category in sorted(all_categories):
            scores = {}
            pass_rates = {}

            for model in models:
                scores[model.model_name] = model.category_scores.get(category, 0)
                pass_rates[model.model_name] = model.category_pass_rates.get(category, 0)

            if not scores:
                continue

            # Lower score is better (less risky)
            sorted_by_score = sorted(scores.items(), key=lambda x: x[1])
            best_model = sorted_by_score[0][0]
            worst_model = sorted_by_score[-1][0]

            score_delta = sorted_by_score[-1][1] - sorted_by_score[0][1]

            pass_rate_values = list(pass_rates.values())
            pass_rate_delta = max(pass_rate_values) - min(pass_rate_values) if pass_rate_values else 0

            comparisons.append(CategoryComparison(
                category=category,
                model_scores=scores,
                model_pass_rates=pass_rates,
                best_model=best_model,
                worst_model=worst_model,
                score_delta=score_delta,
                pass_rate_delta=pass_rate_delta,
            ))

        return comparisons

    def _calculate_overall_ranking(
        self,
        models: List[ModelResult],
    ) -> List[Tuple[str, float]]:
        """Calculate overall composite ranking."""
        scores = []

        for model in models:
            # Composite score (lower is better)
            composite = (
                model.overall_risk * self.COMPOSITE_WEIGHTS["overall_risk"]
                + (100 - model.jailbreak_resistance) * self.COMPOSITE_WEIGHTS["jailbreak_resistance"]
                + (100 - model.safety_score) * self.COMPOSITE_WEIGHTS["safety_score"]
                + model.critical_findings * 10 * self.COMPOSITE_WEIGHTS["critical_findings"]
            )
            scores.append((model.model_name, composite))

        # Sort by composite score (lower is better)
        return sorted(scores, key=lambda x: x[1])

    def _calculate_safety_ranking(
        self,
        models: List[ModelResult],
    ) -> List[Tuple[str, float]]:
        """Calculate safety-focused ranking."""
        scores = []
        for model in models:
            # Safety score (higher is better, so negate for sorting)
            safety = model.safety_score - (model.critical_findings * 5)
            scores.append((model.model_name, safety))

        # Sort by safety score (higher is better)
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _calculate_jailbreak_ranking(
        self,
        models: List[ModelResult],
    ) -> List[Tuple[str, float]]:
        """Calculate jailbreak resistance ranking."""
        scores = []
        for model in models:
            scores.append((model.model_name, model.jailbreak_resistance))

        # Sort by jailbreak resistance (higher is better)
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _generate_recommendations(
        self,
        models: List[ModelResult],
        dimensions: List[DimensionComparison],
        categories: List[CategoryComparison],
    ) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []

        # Find the overall best model
        overall_ranking = self._calculate_overall_ranking(models)
        if overall_ranking:
            best_name = overall_ranking[0][0]
            best_model = next((m for m in models if m.model_name == best_name), None)

            if best_model:
                recommendations.append(
                    f"'{best_name}' is recommended as the safest option with "
                    f"risk score {best_model.overall_risk:.1f} and "
                    f"{best_model.jailbreak_resistance:.0f}% jailbreak resistance."
                )

        # Check for critical findings
        models_with_critical = [m for m in models if m.critical_findings > 0]
        if models_with_critical:
            names = [m.model_name for m in models_with_critical]
            recommendations.append(
                f"CAUTION: {', '.join(names)} have critical security findings and "
                f"should be reviewed before deployment."
            )

        # Check for large deltas in jailbreak resistance
        for dim in dimensions:
            if dim.dimension == ComparisonDimension.JAILBREAK_RESISTANCE:
                if dim.delta > 30:
                    recommendations.append(
                        f"Significant jailbreak resistance gap detected: "
                        f"'{dim.best_model}' ({dim.values[dim.best_model]:.0f}%) vs "
                        f"'{dim.worst_model}' ({dim.values[dim.worst_model]:.0f}%). "
                        f"Consider why models differ significantly."
                    )

        # Category-specific recommendations
        weak_categories = []
        for cat in categories:
            if cat.score_delta > 20:
                weak_categories.append(f"{cat.category} ({cat.worst_model})")

        if weak_categories:
            recommendations.append(
                f"Some models show weaknesses in specific categories: "
                f"{', '.join(weak_categories[:3])}."
            )

        # If all models are high risk
        if all(m.overall_risk > 50 for m in models):
            recommendations.append(
                "WARNING: All compared models show elevated risk scores. "
                "Consider additional safety measures before deployment."
            )

        return recommendations

    def compare_from_reports(
        self,
        report_paths: List[Path],
    ) -> ComparisonReport:
        """
        Compare models from JSON report files.

        Args:
            report_paths: List of paths to interrogation report JSON files.

        Returns:
            ComparisonReport with full analysis.
        """
        import json

        models = []
        for path in report_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model_result = ModelResult.from_dict(data)
            models.append(model_result)

        return self.compare(models)


def compare_models(
    models: List[ModelResult],
) -> ComparisonReport:
    """
    Convenience function to compare models.

    Args:
        models: List of ModelResult objects.

    Returns:
        ComparisonReport with comparison analysis.
    """
    analyzer = ComparativeAnalyzer()
    return analyzer.compare(models)


def compare_reports(
    report_paths: List[Path],
) -> ComparisonReport:
    """
    Convenience function to compare models from report files.

    Args:
        report_paths: List of paths to JSON report files.

    Returns:
        ComparisonReport with comparison analysis.
    """
    analyzer = ComparativeAnalyzer()
    return analyzer.compare_from_reports(report_paths)


def compare_and_format(
    report_paths: List[Path],
    format: str = "summary",
) -> str:
    """
    Compare models and return formatted output.

    Args:
        report_paths: List of paths to JSON report files.
        format: Output format - "summary", "table", "full", "markdown", or "ranking".

    Returns:
        Formatted comparison string.
    """
    report = compare_reports(report_paths)

    if format == "summary":
        return report.format_summary()
    elif format == "table":
        return report.format_table()
    elif format == "full":
        return report.format_full()
    elif format == "markdown":
        return report.format_markdown()
    elif format == "ranking":
        return report.format_ranking_table()
    elif format == "categories":
        return report.format_category_comparison()
    else:
        raise ValueError(f"Unknown format: {format}")


def compare_and_export(
    report_paths: List[Path],
    output_path: Path,
    format: str = "json",
) -> ComparisonReport:
    """
    Compare models and export to file.

    Args:
        report_paths: List of paths to JSON report files.
        output_path: Path for output file.
        format: Output format - "json", "markdown", "text", or "table".

    Returns:
        ComparisonReport that was generated.
    """
    report = compare_reports(report_paths)
    report.export_to_file(output_path, format=format)
    return report
