#!/usr/bin/env python3
"""
AegisML Infrastructure Security Tests

Test plugins for analyzing AI agent infrastructure (MCP servers, skills, protocols).
Integrates with the main Sandbox CLI via the test registry.

Test Categories:
  - mcp_static: Static analysis of MCP server code
  - mcp_security: Security testing of MCP servers
  - skill_static: Static analysis of skill files
  - skill_security: Security testing of skills
"""

import json
from pathlib import Path
from typing import Dict, Any

from aegisml_sandbox_cli import TestResult, SandboxTest
from aegisml_mcp_analyzer import analyze_mcp_server
from aegisml_skill_analyzer import analyze_skill


# ---------- MCP Server Tests ----------

class MCPStaticAnalysisTest(SandboxTest):
    """
    Static security analysis of MCP server Python code.

    Detects:
      - Command injection vulnerabilities
      - Data exfiltration patterns
      - Unsafe file operations
      - Credential access
      - Dangerous imports
      - Obfuscation
      - Backdoor patterns
    """

    name = "mcp_static_analysis"
    category = "mcp_security"

    def run(self, context: Dict[str, Any]) -> TestResult:
        mcp_server_path = context.get("mcp_server_path")

        if not mcp_server_path:
            return TestResult(
                name=self.name,
                category=self.category,
                status="SKIP",
                severity="INFO",
                details="No MCP server specified for analysis (use --mcp-server flag)",
            )

        server_path = Path(mcp_server_path)

        if not server_path.exists():
            return TestResult(
                name=self.name,
                category=self.category,
                status="ERROR",
                severity="HIGH",
                details=f"MCP server file not found: {mcp_server_path}",
            )

        # Run static analysis
        result = analyze_mcp_server(server_path)

        if result["status"] != "success":
            return TestResult(
                name=self.name,
                category=self.category,
                status="ERROR",
                severity="MEDIUM",
                details=f"Analysis failed: {result.get('error', 'Unknown error')}",
            )

        # Convert findings to test result
        risk_level = result["risk_level"]
        severity_counts = result["severity_counts"]
        findings = result["findings"]

        # Determine test status
        if risk_level == "CRITICAL":
            status = "FAIL"
            severity = "CRITICAL"
        elif risk_level == "HIGH":
            status = "FAIL"
            severity = "HIGH"
        elif risk_level == "MEDIUM":
            status = "WARN"
            severity = "MEDIUM"
        elif risk_level == "LOW":
            status = "WARN"
            severity = "LOW"
        else:
            status = "PASS"
            severity = "INFO"

        # Build detailed report
        details_parts = [
            f"=== MCP Server Security Analysis ===",
            f"Server: {server_path.name}",
            f"Risk Level: {risk_level}",
            f"Total Findings: {result['total_findings']}",
            "",
            f"Severity Breakdown:",
            f"  CRITICAL: {severity_counts['CRITICAL']}",
            f"  HIGH:     {severity_counts['HIGH']}",
            f"  MEDIUM:   {severity_counts['MEDIUM']}",
            f"  LOW:      {severity_counts['LOW']}",
            f"  INFO:     {severity_counts['INFO']}",
            "",
        ]

        if findings:
            details_parts.append("=== Top Security Findings ===")

            # Show top 10 findings
            for i, finding in enumerate(findings[:10], 1):
                details_parts.extend([
                    f"\n{i}. [{finding['severity']}] {finding['title']}",
                    f"   Category: {finding['category']}",
                    f"   Location: {finding['location']}",
                    f"   {finding['description']}",
                ])

                if finding.get('code_snippet'):
                    details_parts.append(f"   Code:\n{finding['code_snippet']}")

                if finding.get('remediation'):
                    details_parts.append(f"   Fix: {finding['remediation']}")

            if len(findings) > 10:
                details_parts.append(f"\n... and {len(findings) - 10} more findings (see JSON report)")

        # Metrics
        metrics = {
            "risk_level": risk_level,
            "total_findings": result["total_findings"],
            **{f"severity_{k.lower()}": v for k, v in severity_counts.items()}
        }

        return TestResult(
            name=self.name,
            category=self.category,
            status=status,
            severity=severity,
            details="\n".join(details_parts),
            metrics=metrics,
            artifacts=[
                {
                    "type": "mcp_security_analysis",
                    "description": "Complete MCP server security findings",
                    "data": json.dumps(result, indent=2),
                }
            ],
        )


# ---------- Skill Tests ----------

class SkillStaticAnalysisTest(SandboxTest):
    """
    Static security analysis of Markdown skill files.

    Detects:
      - Prompt injection patterns
      - Credential harvesting
      - Data exfiltration instructions
      - Malicious tool chains
      - Social engineering
      - Missing security constraints
      - Obfuscation
    """

    name = "skill_static_analysis"
    category = "skill_security"

    def run(self, context: Dict[str, Any]) -> TestResult:
        skill_path = context.get("skill_path")

        if not skill_path:
            return TestResult(
                name=self.name,
                category=self.category,
                status="SKIP",
                severity="INFO",
                details="No skill specified for analysis (use --skill flag)",
            )

        skill_file = Path(skill_path)

        if not skill_file.exists():
            return TestResult(
                name=self.name,
                category=self.category,
                status="ERROR",
                severity="HIGH",
                details=f"Skill file not found: {skill_path}",
            )

        # Run static analysis
        result = analyze_skill(skill_file)

        if result["status"] != "success":
            return TestResult(
                name=self.name,
                category=self.category,
                status="ERROR",
                severity="MEDIUM",
                details=f"Analysis failed: {result.get('error', 'Unknown error')}",
            )

        # Convert findings to test result
        risk_level = result["risk_level"]
        severity_counts = result["severity_counts"]
        findings = result["findings"]

        # Determine test status
        if risk_level == "CRITICAL":
            status = "FAIL"
            severity = "CRITICAL"
        elif risk_level == "HIGH":
            status = "FAIL"
            severity = "HIGH"
        elif risk_level == "MEDIUM":
            status = "WARN"
            severity = "MEDIUM"
        elif risk_level == "LOW":
            status = "WARN"
            severity = "LOW"
        else:
            status = "PASS"
            severity = "INFO"

        # Build detailed report
        details_parts = [
            f"=== Skill Security Analysis ===",
            f"Skill: {skill_file.name}",
            f"Risk Level: {risk_level}",
            f"Total Findings: {result['total_findings']}",
            f"Sections Found: {len(result['sections_found'])}",
            "",
            f"Severity Breakdown:",
            f"  CRITICAL: {severity_counts['CRITICAL']}",
            f"  HIGH:     {severity_counts['HIGH']}",
            f"  MEDIUM:   {severity_counts['MEDIUM']}",
            f"  LOW:      {severity_counts['LOW']}",
            f"  INFO:     {severity_counts['INFO']}",
            "",
        ]

        if findings:
            details_parts.append("=== Top Security Findings ===")

            # Show top 10 findings
            for i, finding in enumerate(findings[:10], 1):
                details_parts.extend([
                    f"\n{i}. [{finding['severity']}] {finding['title']}",
                    f"   Category: {finding['category']}",
                    f"   Location: {finding['location']}",
                    f"   {finding['description']}",
                ])

                if finding.get('text_snippet'):
                    snippet = finding['text_snippet'][:200]
                    details_parts.append(f"   Snippet: {snippet}...")

                if finding.get('remediation'):
                    details_parts.append(f"   Fix: {finding['remediation']}")

            if len(findings) > 10:
                details_parts.append(f"\n... and {len(findings) - 10} more findings (see JSON report)")

        # Metrics
        metrics = {
            "risk_level": risk_level,
            "total_findings": result["total_findings"],
            "sections_found": len(result["sections_found"]),
            **{f"severity_{k.lower()}": v for k, v in severity_counts.items()}
        }

        return TestResult(
            name=self.name,
            category=self.category,
            status=status,
            severity=severity,
            details="\n".join(details_parts),
            metrics=metrics,
            artifacts=[
                {
                    "type": "skill_security_analysis",
                    "description": "Complete skill security findings",
                    "data": json.dumps(result, indent=2),
                }
            ],
        )


# ---------- Combined Infrastructure Test ----------

class InfrastructureComprehensiveTest(SandboxTest):
    """
    Comprehensive infrastructure security test.

    Runs both MCP and skill analysis if provided.
    """

    name = "infrastructure_comprehensive"
    category = "infrastructure"

    def run(self, context: Dict[str, Any]) -> TestResult:
        mcp_result = None
        skill_result = None

        # Run MCP analysis if provided
        if context.get("mcp_server_path"):
            mcp_test = MCPStaticAnalysisTest()
            mcp_result = mcp_test.run(context)

        # Run skill analysis if provided
        if context.get("skill_path"):
            skill_test = SkillStaticAnalysisTest()
            skill_result = skill_test.run(context)

        # If neither provided, skip
        if not mcp_result and not skill_result:
            return TestResult(
                name=self.name,
                category=self.category,
                status="SKIP",
                severity="INFO",
                details="No infrastructure components specified (use --mcp-server or --skill)",
            )

        # Aggregate results
        details_parts = ["=== Infrastructure Comprehensive Analysis ===", ""]

        overall_status = "PASS"
        overall_severity = "INFO"
        total_findings = 0

        if mcp_result:
            details_parts.append("--- MCP Server ---")
            details_parts.append(f"Status: {mcp_result.status}")
            details_parts.append(f"Risk: {mcp_result.metrics.get('risk_level', 'N/A')}")
            details_parts.append(f"Findings: {mcp_result.metrics.get('total_findings', 0)}")
            details_parts.append("")

            total_findings += mcp_result.metrics.get('total_findings', 0)

            # Update overall status
            if mcp_result.status in ["FAIL", "ERROR"]:
                overall_status = "FAIL"
            elif mcp_result.status == "WARN" and overall_status == "PASS":
                overall_status = "WARN"

            # Update overall severity
            severity_order = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            if severity_order.index(mcp_result.severity) > severity_order.index(overall_severity):
                overall_severity = mcp_result.severity

        if skill_result:
            details_parts.append("--- Skill ---")
            details_parts.append(f"Status: {skill_result.status}")
            details_parts.append(f"Risk: {skill_result.metrics.get('risk_level', 'N/A')}")
            details_parts.append(f"Findings: {skill_result.metrics.get('total_findings', 0)}")
            details_parts.append("")

            total_findings += skill_result.metrics.get('total_findings', 0)

            # Update overall status
            if skill_result.status in ["FAIL", "ERROR"]:
                overall_status = "FAIL"
            elif skill_result.status == "WARN" and overall_status == "PASS":
                overall_status = "WARN"

            # Update overall severity
            severity_order = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            if severity_order.index(skill_result.severity) > severity_order.index(overall_severity):
                overall_severity = skill_result.severity

        details_parts.append(f"=== Summary ===")
        details_parts.append(f"Overall Status: {overall_status}")
        details_parts.append(f"Total Findings: {total_findings}")

        return TestResult(
            name=self.name,
            category=self.category,
            status=overall_status,
            severity=overall_severity,
            details="\n".join(details_parts),
            metrics={
                "total_findings": total_findings,
                "mcp_analyzed": mcp_result is not None,
                "skill_analyzed": skill_result is not None,
            },
        )


# Export test classes for registration
__all__ = [
    "MCPStaticAnalysisTest",
    "SkillStaticAnalysisTest",
    "InfrastructureComprehensiveTest",
]
