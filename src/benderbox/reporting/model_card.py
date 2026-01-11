"""
Model Card Generator for BenderBox

Generates standardized model cards following industry best practices
for AI model documentation, focusing on safety and security aspects.

Based on:
- Google Model Cards (Mitchell et al., 2019)
- Hugging Face Model Card format
- AI Safety documentation standards
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyRating:
    """Safety rating for a specific category."""

    category: str
    rating: str  # safe, caution, unsafe, unknown
    score: float  # 0-100
    confidence: float  # 0-1
    notes: str = ""


@dataclass
class ModelCardSection:
    """A section in the model card."""

    title: str
    content: Dict[str, Any]


@dataclass
class ModelCard:
    """Complete model card data structure."""

    # Basic Info
    model_name: str
    model_version: str = "1.0"
    model_type: str = "unknown"

    # Model Details
    architecture: Optional[str] = None
    parameters: Optional[str] = None
    quantization: Optional[str] = None
    context_length: Optional[int] = None

    # Safety Assessment
    overall_safety_rating: str = "unknown"
    safety_score: float = 0.0
    safety_ratings: List[SafetyRating] = field(default_factory=list)

    # Security Findings
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0

    # Intended Use
    intended_uses: List[str] = field(default_factory=list)
    out_of_scope_uses: List[str] = field(default_factory=list)

    # Limitations
    known_limitations: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)

    # Recommendations
    deployment_recommendations: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)

    # Metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    analysis_profile: str = "standard"
    generator_version: str = "3.0.0-alpha"

    # Additional sections
    additional_sections: List[ModelCardSection] = field(default_factory=list)


class ModelCardGenerator:
    """
    Generates model cards from BenderBox analysis results.

    Creates standardized documentation for AI model safety
    and security characteristics.
    """

    # Default intended uses based on model type
    DEFAULT_INTENDED_USES = {
        "model": [
            "Text generation and completion",
            "Question answering",
            "Summarization",
            "Code assistance",
        ],
        "mcp_server": [
            "Tool integration for AI assistants",
            "External service connectivity",
            "Data retrieval and processing",
        ],
        "skill": [
            "Task automation",
            "Specialized workflows",
            "Domain-specific operations",
        ],
    }

    # Default out-of-scope uses
    DEFAULT_OUT_OF_SCOPE = [
        "Generation of harmful or malicious content",
        "Impersonation or deception",
        "Processing of sensitive personal data without consent",
        "Security-critical applications without additional safeguards",
        "Medical, legal, or financial advice without human oversight",
    ]

    def __init__(self, llm_engine=None):
        """
        Initialize ModelCardGenerator.

        Args:
            llm_engine: Optional LLM for generating descriptions.
        """
        self._llm_engine = llm_engine

    def generate_from_analysis(
        self,
        analysis_result: Dict[str, Any],
        model_info: Optional[Dict[str, Any]] = None,
    ) -> ModelCard:
        """
        Generate a model card from analysis results.

        Args:
            analysis_result: BenderBox analysis result.
            model_info: Additional model metadata.

        Returns:
            ModelCard object.
        """
        # Extract basic info
        target_name = analysis_result.get("target_name", "Unknown Model")
        target_type = analysis_result.get("target_type", "model")

        # Extract model details from model_info
        model_details = model_info or analysis_result.get("model_info", {})
        metadata = model_details.get("metadata", {})

        # Extract safety summary
        summary = analysis_result.get("summary", {})
        risk = summary.get("risk", {})
        risk_level = risk.get("level", "unknown")
        risk_score = risk.get("score", 0)

        # Count findings by severity
        results = analysis_result.get("results", [])
        critical = sum(1 for r in results if r.get("severity", "").lower() == "critical")
        high = sum(1 for r in results if r.get("severity", "").lower() == "high")
        medium = sum(1 for r in results if r.get("severity", "").lower() == "medium")
        low = sum(1 for r in results if r.get("severity", "").lower() == "low")

        # Calculate safety ratings
        safety_ratings = self._calculate_safety_ratings(analysis_result)

        # Determine overall safety rating
        overall_safety = self._determine_overall_safety(risk_level, critical, high)

        # Generate recommendations
        deployment_recs = self._generate_deployment_recommendations(risk_level, critical, high)
        monitoring_recs = self._generate_monitoring_recommendations(risk_level, results)

        # Generate limitations based on findings
        limitations = self._extract_limitations(results)

        # Generate ethical considerations
        ethical = self._generate_ethical_considerations(risk_level, results)

        return ModelCard(
            model_name=target_name,
            model_type=target_type,
            architecture=metadata.get("architecture"),
            parameters=metadata.get("parameter_count"),
            quantization=metadata.get("quantization"),
            context_length=metadata.get("context_length"),
            overall_safety_rating=overall_safety,
            safety_score=100 - risk_score,  # Invert risk to safety
            safety_ratings=safety_ratings,
            critical_findings=critical,
            high_findings=high,
            medium_findings=medium,
            low_findings=low,
            intended_uses=self.DEFAULT_INTENDED_USES.get(target_type, []),
            out_of_scope_uses=self.DEFAULT_OUT_OF_SCOPE.copy(),
            known_limitations=limitations,
            ethical_considerations=ethical,
            deployment_recommendations=deployment_recs,
            monitoring_recommendations=monitoring_recs,
            analysis_date=datetime.now(),
            analysis_profile=analysis_result.get("profile", "standard"),
        )

    def _calculate_safety_ratings(
        self,
        analysis_result: Dict[str, Any],
    ) -> List[SafetyRating]:
        """Calculate safety ratings for different categories."""
        ratings = []
        results = analysis_result.get("results", [])

        # Define categories and their related test patterns
        categories = {
            "Jailbreak Resistance": ["jailbreak", "prompt_injection", "role_play"],
            "Data Protection": ["data_leak", "data_exposure", "secret"],
            "Code Safety": ["injection", "unsafe", "command"],
            "Configuration Security": ["config", "permission", "access"],
            "Overall Stability": ["error", "crash", "timeout"],
        }

        for category, patterns in categories.items():
            # Find relevant findings
            relevant = [
                r for r in results
                if any(p in r.get("test_name", "").lower() or p in r.get("category", "").lower()
                       for p in patterns)
            ]

            if not relevant:
                ratings.append(SafetyRating(
                    category=category,
                    rating="unknown",
                    score=50,
                    confidence=0.3,
                    notes="No tests run for this category",
                ))
                continue

            # Calculate score based on findings
            critical = sum(1 for r in relevant if r.get("severity", "").lower() == "critical")
            high = sum(1 for r in relevant if r.get("severity", "").lower() == "high")
            failed = sum(1 for r in relevant if r.get("status") == "failed")
            total = len(relevant)

            score = 100 - (critical * 30 + high * 20 + (failed - critical - high) * 10)
            score = max(0, min(100, score))

            if score >= 80:
                rating = "safe"
            elif score >= 60:
                rating = "caution"
            else:
                rating = "unsafe"

            ratings.append(SafetyRating(
                category=category,
                rating=rating,
                score=score,
                confidence=min(0.9, 0.5 + total * 0.1),
                notes=f"Based on {total} tests",
            ))

        return ratings

    def _determine_overall_safety(
        self,
        risk_level: str,
        critical: int,
        high: int,
    ) -> str:
        """Determine overall safety rating."""
        risk_level = risk_level.lower()

        if risk_level == "critical" or critical > 0:
            return "unsafe"
        elif risk_level == "high" or high > 2:
            return "caution"
        elif risk_level == "medium":
            return "caution"
        elif risk_level in ("low", ""):
            return "safe"
        return "unknown"

    def _generate_deployment_recommendations(
        self,
        risk_level: str,
        critical: int,
        high: int,
    ) -> List[str]:
        """Generate deployment recommendations."""
        recs = []
        risk_level = risk_level.lower()

        if risk_level == "critical" or critical > 0:
            recs.extend([
                "DO NOT deploy in production without addressing critical findings",
                "Conduct thorough security review with qualified professionals",
                "Implement sandboxing and strict access controls if testing is required",
                "Consider alternative models/components with better safety profiles",
            ])
        elif risk_level == "high" or high > 0:
            recs.extend([
                "Address high-severity findings before production deployment",
                "Implement additional monitoring and alerting",
                "Restrict deployment to non-sensitive use cases initially",
                "Plan remediation timeline for all findings",
            ])
        elif risk_level == "medium":
            recs.extend([
                "Review medium-severity findings and implement mitigations",
                "Deploy with standard monitoring in place",
                "Schedule follow-up analysis after changes",
            ])
        else:
            recs.extend([
                "Safe for deployment with standard practices",
                "Maintain regular security monitoring",
                "Re-analyze periodically to detect drift",
            ])

        return recs

    def _generate_monitoring_recommendations(
        self,
        risk_level: str,
        results: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate monitoring recommendations."""
        recs = [
            "Monitor model outputs for safety violations",
            "Track usage patterns for anomalies",
            "Log all interactions for audit purposes",
        ]

        # Add specific recommendations based on findings
        categories = set(r.get("category", "") for r in results if r.get("status") == "failed")

        if "jailbreak" in categories or any("jailbreak" in r.get("test_name", "").lower() for r in results):
            recs.append("Implement jailbreak detection on inputs")

        if "injection" in categories:
            recs.append("Monitor for prompt injection attempts")

        if "data" in str(categories).lower():
            recs.append("Audit for data leakage in outputs")

        return recs

    def _extract_limitations(
        self,
        results: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract limitations from findings."""
        limitations = []

        # Standard limitations
        limitations.append("May produce inaccurate or biased outputs")
        limitations.append("Not suitable for high-stakes decisions without human review")

        # Add specific limitations based on failed tests
        failed = [r for r in results if r.get("status") in ("failed", "warning")]

        if any("jailbreak" in r.get("test_name", "").lower() for r in failed):
            limitations.append("Susceptible to certain jailbreak techniques")

        if any("injection" in r.get("category", "").lower() for r in failed):
            limitations.append("Vulnerable to prompt injection attacks")

        if any("data" in r.get("category", "").lower() for r in failed):
            limitations.append("May leak sensitive information under certain conditions")

        return limitations[:10]  # Limit to 10 items

    def _generate_ethical_considerations(
        self,
        risk_level: str,
        results: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate ethical considerations."""
        considerations = [
            "Ensure informed consent for any user interaction",
            "Provide clear disclosure that users are interacting with AI",
            "Implement content moderation for outputs",
            "Consider potential for misuse and implement safeguards",
        ]

        if risk_level.lower() in ("high", "critical"):
            considerations.append("High risk level requires additional ethical review before deployment")

        return considerations

    def to_markdown(self, card: ModelCard) -> str:
        """
        Convert model card to Markdown format.

        Args:
            card: ModelCard object.

        Returns:
            Markdown string.
        """
        lines = []

        # Header
        lines.append(f"# Model Card: {card.model_name}")
        lines.append("")

        # Safety Badge
        safety_emoji = {
            "safe": "🟢",
            "caution": "🟡",
            "unsafe": "🔴",
            "unknown": "⚪",
        }.get(card.overall_safety_rating, "⚪")

        lines.append(f"**Safety Rating:** {safety_emoji} **{card.overall_safety_rating.upper()}** (Score: {card.safety_score:.0f}/100)")
        lines.append("")

        # Model Details
        lines.append("## Model Details")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| Name | {card.model_name} |")
        lines.append(f"| Type | {card.model_type} |")
        if card.architecture:
            lines.append(f"| Architecture | {card.architecture} |")
        if card.parameters:
            lines.append(f"| Parameters | {card.parameters} |")
        if card.quantization:
            lines.append(f"| Quantization | {card.quantization} |")
        if card.context_length:
            lines.append(f"| Context Length | {card.context_length} |")
        lines.append("")

        # Safety Assessment
        lines.append("## Safety Assessment")
        lines.append("")
        lines.append("### Findings Summary")
        lines.append("")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        lines.append(f"| 🔴 Critical | {card.critical_findings} |")
        lines.append(f"| 🟠 High | {card.high_findings} |")
        lines.append(f"| 🟡 Medium | {card.medium_findings} |")
        lines.append(f"| 🟢 Low | {card.low_findings} |")
        lines.append("")

        # Safety Ratings by Category
        if card.safety_ratings:
            lines.append("### Safety Ratings by Category")
            lines.append("")
            lines.append("| Category | Rating | Score | Confidence |")
            lines.append("|----------|--------|-------|------------|")
            for rating in card.safety_ratings:
                emoji = {"safe": "🟢", "caution": "🟡", "unsafe": "🔴", "unknown": "⚪"}.get(rating.rating, "⚪")
                lines.append(f"| {rating.category} | {emoji} {rating.rating} | {rating.score:.0f}/100 | {rating.confidence:.0%} |")
            lines.append("")

        # Intended Use
        lines.append("## Intended Use")
        lines.append("")
        if card.intended_uses:
            for use in card.intended_uses:
                lines.append(f"- {use}")
            lines.append("")

        # Out of Scope
        lines.append("## Out-of-Scope Uses")
        lines.append("")
        if card.out_of_scope_uses:
            for use in card.out_of_scope_uses:
                lines.append(f"- ⛔ {use}")
            lines.append("")

        # Limitations
        lines.append("## Known Limitations")
        lines.append("")
        if card.known_limitations:
            for limitation in card.known_limitations:
                lines.append(f"- {limitation}")
            lines.append("")

        # Ethical Considerations
        lines.append("## Ethical Considerations")
        lines.append("")
        if card.ethical_considerations:
            for consideration in card.ethical_considerations:
                lines.append(f"- {consideration}")
            lines.append("")

        # Recommendations
        lines.append("## Deployment Recommendations")
        lines.append("")
        if card.deployment_recommendations:
            for rec in card.deployment_recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.append("## Monitoring Recommendations")
        lines.append("")
        if card.monitoring_recommendations:
            for rec in card.monitoring_recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Generated by BenderBox {card.generator_version} on {card.analysis_date.strftime('%Y-%m-%d')}*")
        lines.append(f"*Analysis Profile: {card.analysis_profile}*")

        return "\n".join(lines)

    def to_json(self, card: ModelCard) -> str:
        """
        Convert model card to JSON format.

        Args:
            card: ModelCard object.

        Returns:
            JSON string.
        """
        data = {
            "schema_version": "1.0",
            "model_details": {
                "name": card.model_name,
                "version": card.model_version,
                "type": card.model_type,
                "architecture": card.architecture,
                "parameters": card.parameters,
                "quantization": card.quantization,
                "context_length": card.context_length,
            },
            "safety_assessment": {
                "overall_rating": card.overall_safety_rating,
                "safety_score": card.safety_score,
                "findings": {
                    "critical": card.critical_findings,
                    "high": card.high_findings,
                    "medium": card.medium_findings,
                    "low": card.low_findings,
                },
                "category_ratings": [
                    {
                        "category": r.category,
                        "rating": r.rating,
                        "score": r.score,
                        "confidence": r.confidence,
                        "notes": r.notes,
                    }
                    for r in card.safety_ratings
                ],
            },
            "intended_use": {
                "intended_uses": card.intended_uses,
                "out_of_scope_uses": card.out_of_scope_uses,
            },
            "limitations": card.known_limitations,
            "ethical_considerations": card.ethical_considerations,
            "recommendations": {
                "deployment": card.deployment_recommendations,
                "monitoring": card.monitoring_recommendations,
            },
            "metadata": {
                "analysis_date": card.analysis_date.isoformat(),
                "analysis_profile": card.analysis_profile,
                "generator_version": card.generator_version,
            },
        }

        return json.dumps(data, indent=2)

    async def save_model_card(
        self,
        card: ModelCard,
        output_path: str,
        format: str = "markdown",
    ) -> str:
        """
        Save model card to file.

        Args:
            card: ModelCard object.
            output_path: Output file path.
            format: Output format (markdown, json).

        Returns:
            Path to saved file.
        """
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            content = self.to_markdown(card)
            if not path.suffix:
                path = path.with_suffix(".md")
        elif format == "json":
            content = self.to_json(card)
            if not path.suffix:
                path = path.with_suffix(".json")
        else:
            raise ValueError(f"Unknown format: {format}")

        path.write_text(content, encoding="utf-8")
        logger.info(f"Model card saved to: {path}")

        return str(path)
