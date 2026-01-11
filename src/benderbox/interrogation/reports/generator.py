"""
Report generation for model interrogation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..analyzer.results import AnalysisResult
from ..scoring.risk import InterrogationRiskScore
from ..validator.censorship import MislabelingReport


@dataclass
class InterrogationReport:
    """Complete interrogation report."""

    # Metadata
    schema_version: str = "1.0.0"
    report_type: str = "model_interrogation"
    timestamp: str = ""
    run_id: str = ""

    # Model info
    model_path: str = ""
    model_name: str = ""
    model_fingerprint: str = ""
    claimed_censorship: str = "unknown"

    # Scores
    risk_score: Optional[InterrogationRiskScore] = None
    censorship_report: Optional[MislabelingReport] = None

    # Results
    results: List[AnalysisResult] = field(default_factory=list)

    # Profile
    profile: str = "standard"
    prompts_tested: int = 0
    duration_seconds: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.run_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.run_id = f"interrogation_{ts}"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "report_type": self.report_type,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "model": {
                "path": self.model_path,
                "name": self.model_name,
                "fingerprint": self.model_fingerprint,
                "claimed_censorship": self.claimed_censorship,
            },
            "summary": {
                "overall_risk": self.risk_score.overall_risk if self.risk_score else 0,
                "risk_level": self.risk_score.risk_level.value if self.risk_score else "unknown",
                "prompts_tested": self.prompts_tested,
                "passed": self.risk_score.passed_count if self.risk_score else 0,
                "failed": self.risk_score.failed_count if self.risk_score else 0,
                "duration_seconds": self.duration_seconds,
                "profile": self.profile,
            },
            "risk_assessment": self.risk_score.to_dict() if self.risk_score else None,
            "censorship_validation": self.censorship_report.to_dict() if self.censorship_report else None,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path) -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def summary_text(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "MODEL INTERROGATION REPORT",
            "=" * 60,
            f"Model: {self.model_name}",
            f"Profile: {self.profile}",
            f"Timestamp: {self.timestamp}",
            "",
        ]

        if self.risk_score:
            lines.extend([
                "RISK ASSESSMENT",
                "-" * 40,
                self.risk_score.summary(),
                "",
            ])

        if self.censorship_report:
            lines.extend([
                "CENSORSHIP VALIDATION",
                "-" * 40,
                self.censorship_report.summary(),
                "",
            ])

        if self.risk_score and self.risk_score.critical_findings:
            lines.extend([
                "CRITICAL FINDINGS",
                "-" * 40,
            ])
            for finding in self.risk_score.critical_findings[:5]:
                lines.append(f"  - [{finding.category}] {finding.description}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


class ReportGenerator:
    """Generates interrogation reports."""

    def generate(
        self,
        results: List[AnalysisResult],
        risk_score: InterrogationRiskScore,
        model_path: str,
        model_name: str = "",
        model_fingerprint: str = "",
        claimed_censorship: str = "unknown",
        censorship_report: Optional[MislabelingReport] = None,
        profile: str = "standard",
        duration: float = 0.0,
    ) -> InterrogationReport:
        """
        Generate a complete interrogation report.

        Args:
            results: Analysis results from interrogation
            risk_score: Calculated risk score
            model_path: Path to model file
            model_name: Name of model
            model_fingerprint: SHA256 fingerprint
            claimed_censorship: Claimed censorship level
            censorship_report: Censorship validation report
            profile: Interrogation profile used
            duration: Total duration in seconds

        Returns:
            InterrogationReport
        """
        return InterrogationReport(
            model_path=model_path,
            model_name=model_name or Path(model_path).name,
            model_fingerprint=model_fingerprint,
            claimed_censorship=claimed_censorship,
            risk_score=risk_score,
            censorship_report=censorship_report,
            results=results,
            profile=profile,
            prompts_tested=len(results),
            duration_seconds=duration,
        )
