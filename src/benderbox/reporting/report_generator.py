"""
Report Generator for BenderBox

Generates comprehensive security analysis reports in multiple formats:
- Markdown (human-readable)
- JSON (machine-readable)
- HTML (web viewing)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """A section in the report."""

    title: str
    content: str
    level: int = 2  # Heading level (1-6)
    subsections: List["ReportSection"] = field(default_factory=list)


@dataclass
class ReportMetadata:
    """Metadata for a report."""

    title: str
    target_name: str
    target_type: str
    generated_at: datetime = field(default_factory=datetime.now)
    generator_version: str = "3.0.0-alpha"
    profile: str = "standard"
    author: str = "BenderBox"


class ReportGenerator:
    """
    Generates security analysis reports in multiple formats.

    Supports:
    - Markdown for documentation
    - JSON for programmatic access
    - HTML for web viewing
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize ReportGenerator.

        Args:
            template_dir: Directory containing report templates.
        """
        self.template_dir = Path(template_dir) if template_dir else None

    def generate_markdown(
        self,
        analysis_result: Dict[str, Any],
        include_details: bool = True,
        include_recommendations: bool = True,
    ) -> str:
        """
        Generate a Markdown report from analysis results.

        Args:
            analysis_result: Analysis result dictionary.
            include_details: Include detailed findings.
            include_recommendations: Include recommendations section.

        Returns:
            Markdown formatted report string.
        """
        lines = []

        # Header
        target_name = analysis_result.get("target_name", "Unknown Target")
        lines.append(f"# Security Analysis Report: {target_name}")
        lines.append("")

        # Metadata
        lines.append("## Report Information")
        lines.append("")
        lines.append(f"| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| **Target** | {target_name} |")
        lines.append(f"| **Type** | {analysis_result.get('target_type', 'unknown')} |")
        lines.append(f"| **Profile** | {analysis_result.get('profile', 'standard')} |")
        lines.append(f"| **Date** | {analysis_result.get('timestamp', datetime.now().isoformat())} |")
        lines.append(f"| **Run ID** | {analysis_result.get('run_id', 'N/A')} |")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        summary = analysis_result.get("summary", {})
        risk = summary.get("risk", {})
        risk_level = risk.get("level", "unknown").upper()
        risk_score = risk.get("score", 0)

        # Risk badge
        risk_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
        }.get(risk_level, "⚪")

        lines.append(f"**Risk Level:** {risk_emoji} **{risk_level}** (Score: {risk_score}/100)")
        lines.append("")

        # Results summary
        results = analysis_result.get("results", [])
        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        warnings = sum(1 for r in results if r.get("status") == "warning")
        errors = sum(1 for r in results if r.get("status") == "error")

        lines.append("### Test Results Summary")
        lines.append("")
        lines.append(f"| Status | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| ✅ Passed | {passed} |")
        lines.append(f"| ❌ Failed | {failed} |")
        lines.append(f"| ⚠️ Warnings | {warnings} |")
        lines.append(f"| 🔧 Errors | {errors} |")
        lines.append(f"| **Total** | {len(results)} |")
        lines.append("")

        # Risk factors
        factors = risk.get("factors", [])
        if factors:
            lines.append("### Primary Risk Factors")
            lines.append("")
            for factor in factors[:5]:
                lines.append(f"- {factor}")
            lines.append("")

        # Findings by severity
        if include_details and results:
            lines.append("## Detailed Findings")
            lines.append("")

            # Group by severity
            severity_order = ["critical", "high", "medium", "low", "info"]
            for severity in severity_order:
                severity_findings = [
                    r for r in results
                    if r.get("severity", "").lower() == severity and r.get("status") in ("failed", "warning")
                ]

                if severity_findings:
                    severity_emoji = {
                        "critical": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🟢",
                        "info": "🤖",
                    }.get(severity, "")

                    lines.append(f"### {severity_emoji} {severity.upper()} Severity ({len(severity_findings)})")
                    lines.append("")

                    for finding in severity_findings:
                        lines.append(f"#### {finding.get('test_name', 'Unknown Test')}")
                        lines.append("")
                        lines.append(f"**Category:** {finding.get('category', 'N/A')}")
                        lines.append(f"**Status:** {finding.get('status', 'N/A')}")
                        lines.append("")

                        details = finding.get("details", {})
                        if isinstance(details, dict):
                            message = details.get("message", "")
                            if message:
                                lines.append(f"**Details:**")
                                lines.append(f"> {message[:500]}")
                                lines.append("")

                            metrics = details.get("metrics", {})
                            if metrics:
                                lines.append("**Metrics:**")
                                for key, value in list(metrics.items())[:5]:
                                    lines.append(f"- {key}: {value}")
                                lines.append("")
                        elif isinstance(details, str) and details:
                            lines.append(f"**Details:** {details[:500]}")
                            lines.append("")

        # Recommendations
        if include_recommendations:
            lines.append("## Recommendations")
            lines.append("")

            if risk_level == "CRITICAL":
                lines.append("### ⛔ Critical Actions Required")
                lines.append("")
                lines.append("1. **Do not deploy** this target in its current state")
                lines.append("2. Address all critical findings immediately")
                lines.append("3. Conduct thorough security review before any deployment")
                lines.append("4. Consider engaging security specialists")
                lines.append("")
            elif risk_level == "HIGH":
                lines.append("### 🚨 High Priority Actions")
                lines.append("")
                lines.append("1. Address all critical and high severity issues before deployment")
                lines.append("2. Implement additional monitoring if deployment is necessary")
                lines.append("3. Plan for immediate remediation timeline")
                lines.append("")
            elif risk_level == "MEDIUM":
                lines.append("### ⚠️ Recommended Actions")
                lines.append("")
                lines.append("1. Review and address findings before production deployment")
                lines.append("2. Implement mitigations where possible")
                lines.append("3. Schedule follow-up analysis after changes")
                lines.append("")
            else:
                lines.append("### ✅ Maintenance Actions")
                lines.append("")
                lines.append("1. Continue standard security monitoring")
                lines.append("2. Re-analyze periodically to detect drift")
                lines.append("3. Review any warnings for potential improvements")
                lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Report generated by BenderBox v3.0.0-alpha on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def generate_json(
        self,
        analysis_result: Dict[str, Any],
        pretty: bool = True,
    ) -> str:
        """
        Generate a JSON report from analysis results.

        Args:
            analysis_result: Analysis result dictionary.
            pretty: Pretty-print the JSON.

        Returns:
            JSON formatted report string.
        """
        report = {
            "report_version": "1.0",
            "generator": "BenderBox",
            "generator_version": "3.0.0-alpha",
            "generated_at": datetime.now().isoformat(),
            "analysis": analysis_result,
        }

        if pretty:
            return json.dumps(report, indent=2, default=str)
        return json.dumps(report, default=str)

    def generate_html(
        self,
        analysis_result: Dict[str, Any],
        include_styles: bool = True,
    ) -> str:
        """
        Generate an HTML report from analysis results.

        Args:
            analysis_result: Analysis result dictionary.
            include_styles: Include embedded CSS styles.

        Returns:
            HTML formatted report string.
        """
        target_name = analysis_result.get("target_name", "Unknown Target")
        summary = analysis_result.get("summary", {})
        risk = summary.get("risk", {})
        risk_level = risk.get("level", "unknown").upper()
        risk_score = risk.get("score", 0)
        results = analysis_result.get("results", [])

        # CSS styles
        styles = """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .report { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #1a1a2e; border-bottom: 3px solid #4a4a8a; padding-bottom: 10px; }
            h2 { color: #16213e; margin-top: 30px; }
            h3 { color: #0f3460; }
            .risk-badge { display: inline-block; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
            .risk-critical { background: #ff4444; color: white; }
            .risk-high { background: #ff8800; color: white; }
            .risk-medium { background: #ffcc00; color: #333; }
            .risk-low { background: #44cc44; color: white; }
            .summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .summary-table th, .summary-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            .summary-table th { background: #4a4a8a; color: white; }
            .finding { background: #f9f9f9; border-left: 4px solid #ccc; padding: 15px; margin: 10px 0; }
            .finding-critical { border-left-color: #ff4444; }
            .finding-high { border-left-color: #ff8800; }
            .finding-medium { border-left-color: #ffcc00; }
            .finding-low { border-left-color: #44cc44; }
            .finding-title { font-weight: bold; margin-bottom: 8px; }
            .finding-meta { color: #666; font-size: 0.9em; }
            .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }
        </style>
        """ if include_styles else ""

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>Security Report: {target_name}</title>",
            styles,
            "</head>",
            "<body>",
            "<div class='report'>",
            f"<h1>Security Analysis Report: {target_name}</h1>",
        ]

        # Risk badge
        risk_class = f"risk-{risk_level.lower()}"
        html_parts.append(f"<p><span class='risk-badge {risk_class}'>Risk: {risk_level} ({risk_score}/100)</span></p>")

        # Summary table
        html_parts.append("<h2>Summary</h2>")
        html_parts.append("<table class='summary-table'>")
        html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
        html_parts.append(f"<tr><td>Target</td><td>{target_name}</td></tr>")
        html_parts.append(f"<tr><td>Type</td><td>{analysis_result.get('target_type', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Profile</td><td>{analysis_result.get('profile', 'N/A')}</td></tr>")
        html_parts.append(f"<tr><td>Total Tests</td><td>{len(results)}</td></tr>")

        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        html_parts.append(f"<tr><td>Passed</td><td>{passed}</td></tr>")
        html_parts.append(f"<tr><td>Failed</td><td>{failed}</td></tr>")
        html_parts.append("</table>")

        # Findings
        failed_findings = [r for r in results if r.get("status") in ("failed", "warning")]
        if failed_findings:
            html_parts.append("<h2>Findings</h2>")
            for finding in failed_findings:
                severity = finding.get("severity", "info").lower()
                html_parts.append(f"<div class='finding finding-{severity}'>")
                html_parts.append(f"<div class='finding-title'>{finding.get('test_name', 'Unknown')}</div>")
                html_parts.append(f"<div class='finding-meta'>Severity: {severity.upper()} | Category: {finding.get('category', 'N/A')}</div>")

                details = finding.get("details", {})
                if isinstance(details, dict) and details.get("message"):
                    html_parts.append(f"<p>{details['message'][:500]}</p>")
                elif isinstance(details, str):
                    html_parts.append(f"<p>{details[:500]}</p>")

                html_parts.append("</div>")

        # Footer
        html_parts.append("<div class='footer'>")
        html_parts.append(f"Generated by BenderBox v3.0.0-alpha on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        html_parts.append("</div>")
        html_parts.append("</div>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        return "\n".join(html_parts)

    def generate_summary(
        self,
        analysis_result: Dict[str, Any],
    ) -> str:
        """
        Generate a brief text summary of the analysis.

        Args:
            analysis_result: Analysis result dictionary.

        Returns:
            Brief text summary.
        """
        target = analysis_result.get("target_name", "Unknown")
        summary = analysis_result.get("summary", {})
        risk = summary.get("risk", {})
        risk_level = risk.get("level", "unknown").upper()
        risk_score = risk.get("score", 0)
        results = analysis_result.get("results", [])

        critical = sum(1 for r in results if r.get("severity", "").lower() == "critical")
        high = sum(1 for r in results if r.get("severity", "").lower() == "high")

        lines = [
            f"Analysis Summary: {target}",
            f"Risk: {risk_level} ({risk_score}/100)",
            f"Tests: {len(results)} total",
        ]

        if critical or high:
            lines.append(f"Critical findings: {critical}, High findings: {high}")

        return " | ".join(lines)

    async def save_report(
        self,
        analysis_result: Dict[str, Any],
        output_path: str,
        format: str = "markdown",
    ) -> str:
        """
        Save report to file.

        Args:
            analysis_result: Analysis result dictionary.
            output_path: Output file path.
            format: Output format (markdown, json, html).

        Returns:
            Path to saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            content = self.generate_markdown(analysis_result)
            if not path.suffix:
                path = path.with_suffix(".md")
        elif format == "json":
            content = self.generate_json(analysis_result)
            if not path.suffix:
                path = path.with_suffix(".json")
        elif format == "html":
            content = self.generate_html(analysis_result)
            if not path.suffix:
                path = path.with_suffix(".html")
        else:
            raise ValueError(f"Unknown format: {format}")

        path.write_text(content, encoding="utf-8")
        logger.info(f"Report saved to: {path}")

        return str(path)

    async def save_all_formats(
        self,
        analysis_result: Dict[str, Any],
        output_dir: str,
        base_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save report in all formats.

        Args:
            analysis_result: Analysis result dictionary.
            output_dir: Output directory.
            base_name: Base filename (without extension).

        Returns:
            Dictionary mapping format to file path.
        """
        dir_path = Path(output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        if base_name is None:
            target = analysis_result.get("target_name", "report")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"benderbox_{target}_{timestamp}"

        # Clean filename
        base_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)

        paths = {}
        for fmt in ["markdown", "json", "html"]:
            ext = {"markdown": ".md", "json": ".json", "html": ".html"}[fmt]
            file_path = dir_path / f"{base_name}{ext}"
            paths[fmt] = await self.save_report(analysis_result, str(file_path), fmt)

        return paths

    def generate_comparison_markdown(
        self,
        comparison_result: Dict[str, Any],
    ) -> str:
        """
        Generate a Markdown comparison report.

        Args:
            comparison_result: Comparison result dictionary.

        Returns:
            Markdown formatted comparison report.
        """
        lines = []
        targets = comparison_result.get("targets", [])
        results = comparison_result.get("results", [])

        lines.append("# Model Comparison Report")
        lines.append("")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Targets Compared:** {len(targets)}")
        lines.append("")

        # Comparison table
        lines.append("## Side-by-Side Comparison")
        lines.append("")

        # Build table header
        header = "| Metric |"
        separator = "|--------|"
        for target in targets:
            name = target[:20] if len(target) > 20 else target
            header += f" {name} |"
            separator += "--------|"

        lines.append(header)
        lines.append(separator)

        # Add metric rows
        metrics = [
            ("Risk Score", "risk_score", lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)),
            ("Risk Level", "risk_level", lambda x: x.upper() if x else "N/A"),
            ("Finding Count", "finding_count", str),
            ("Critical Findings", "critical_count", str),
            ("High Findings", "high_count", str),
        ]

        for label, key, formatter in metrics:
            row = f"| **{label}** |"
            for result in results:
                value = result.get(key, "N/A")
                row += f" {formatter(value)} |"
            lines.append(row)

        lines.append("")

        # Summary
        summary = comparison_result.get("summary", {})
        if summary:
            lines.append("## Summary")
            lines.append("")
            if summary.get("safest"):
                lines.append(f"- **Safest Model:** {summary['safest']}")
            if summary.get("riskiest"):
                lines.append(f"- **Riskiest Model:** {summary['riskiest']}")
            lines.append("")

        # Recommendation
        recommendation = comparison_result.get("recommendation", "")
        if recommendation:
            lines.append("## Recommendation")
            lines.append("")
            lines.append(recommendation)
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated by BenderBox v3.0.0-alpha*")

        return "\n".join(lines)

    def generate_comparison_html(
        self,
        comparison_result: Dict[str, Any],
    ) -> str:
        """
        Generate an HTML comparison report.

        Args:
            comparison_result: Comparison result dictionary.

        Returns:
            HTML formatted comparison report.
        """
        targets = comparison_result.get("targets", [])
        results = comparison_result.get("results", [])
        summary = comparison_result.get("summary", {})

        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<title>Model Comparison Report</title>",
            "<style>",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;",
            "       max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }",
            ".report { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "h1 { color: #1a1a2e; border-bottom: 3px solid #4a4a8a; padding-bottom: 10px; }",
            "h2 { color: #16213e; margin-top: 30px; }",
            ".comparison-table { width: 100%; border-collapse: collapse; margin: 20px 0; }",
            ".comparison-table th, .comparison-table td { padding: 12px; text-align: left; border: 1px solid #ddd; }",
            ".comparison-table th { background: #4a4a8a; color: white; }",
            ".comparison-table tr:nth-child(even) { background: #f9f9f9; }",
            ".risk-critical { color: #ff4444; font-weight: bold; }",
            ".risk-high { color: #ff8800; font-weight: bold; }",
            ".risk-medium { color: #ffcc00; }",
            ".risk-low { color: #44cc44; }",
            ".summary-box { background: #e8f4fd; padding: 15px; border-radius: 4px; margin: 20px 0; }",
            ".recommendation { background: #fff3cd; padding: 15px; border-radius: 4px; border-left: 4px solid #ffc107; }",
            ".footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='report'>",
            "<h1>Model Comparison Report</h1>",
            f"<p>Comparing {len(targets)} targets</p>",
        ]

        # Comparison table
        html_parts.append("<h2>Side-by-Side Comparison</h2>")
        html_parts.append("<table class='comparison-table'>")
        html_parts.append("<tr><th>Metric</th>")
        for target in targets:
            name = target[:25] if len(target) > 25 else target
            html_parts.append(f"<th>{name}</th>")
        html_parts.append("</tr>")

        # Add rows
        metrics = [
            ("Risk Score", "risk_score"),
            ("Risk Level", "risk_level"),
            ("Finding Count", "finding_count"),
            ("Critical Findings", "critical_count"),
            ("High Findings", "high_count"),
        ]

        for label, key in metrics:
            html_parts.append(f"<tr><td><strong>{label}</strong></td>")
            for result in results:
                value = result.get(key, "N/A")
                if key == "risk_level" and value:
                    css_class = f"risk-{value.lower()}"
                    html_parts.append(f"<td class='{css_class}'>{value.upper()}</td>")
                elif key == "risk_score" and isinstance(value, (int, float)):
                    html_parts.append(f"<td>{value:.1f}</td>")
                else:
                    html_parts.append(f"<td>{value}</td>")
            html_parts.append("</tr>")

        html_parts.append("</table>")

        # Summary
        if summary:
            html_parts.append("<div class='summary-box'>")
            html_parts.append("<h3>Summary</h3>")
            if summary.get("safest"):
                html_parts.append(f"<p><strong>Safest Model:</strong> {summary['safest']}</p>")
            if summary.get("riskiest"):
                html_parts.append(f"<p><strong>Riskiest Model:</strong> {summary['riskiest']}</p>")
            html_parts.append("</div>")

        # Recommendation
        recommendation = comparison_result.get("recommendation", "")
        if recommendation:
            html_parts.append("<div class='recommendation'>")
            html_parts.append(f"<h3>Recommendation</h3>")
            html_parts.append(f"<p>{recommendation}</p>")
            html_parts.append("</div>")

        # Footer
        html_parts.append("<div class='footer'>")
        html_parts.append(f"Generated by BenderBox v3.0.0-alpha on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        html_parts.append("</div>")
        html_parts.append("</div>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        return "\n".join(html_parts)
