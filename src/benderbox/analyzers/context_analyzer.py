"""
Context Analyzer for BenderBox

Analyzes instruction files (skills.md, system prompts, agent instructions)
and inference outputs for dangerous patterns and security risks.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context files that can be analyzed."""

    SKILL = "skill"           # Skills.md, skill definitions
    PROMPT = "prompt"         # System prompts
    INSTRUCTION = "instruction"  # Agent instructions
    OUTPUT = "output"         # Inference/model outputs
    CONFIG = "config"         # Configuration files
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk level classifications."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    SAFE = "SAFE"


@dataclass
class RiskPattern:
    """A pattern that indicates potential risk in context files."""

    id: str
    pattern: str              # Regex pattern
    risk_level: RiskLevel
    category: str             # jailbreak, credential, code_exec, etc.
    description: str
    recommendation: str
    case_insensitive: bool = True


@dataclass
class Finding:
    """A finding from context analysis."""

    pattern_id: str
    risk_level: RiskLevel
    category: str
    description: str
    matched_text: str
    line_number: Optional[int]
    context: str              # Surrounding text for context
    recommendation: str


@dataclass
class ContextAnalysisResult:
    """Result of analyzing a context file."""

    file_path: str
    file_type: ContextType
    risk_level: RiskLevel
    risk_score: float         # 0-100
    findings: List[Finding]
    dangerous_patterns: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# Risk patterns organized by category
RISK_PATTERNS: List[RiskPattern] = [
    # === JAILBREAK PATTERNS (CRITICAL) ===
    RiskPattern(
        id="jail-001",
        pattern=r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|guidelines?)",
        risk_level=RiskLevel.CRITICAL,
        category="jailbreak",
        description="Instruction to ignore previous guidelines",
        recommendation="Remove jailbreak-enabling instructions",
    ),
    RiskPattern(
        id="jail-002",
        pattern=r"(you\s+are|act\s+as|pretend\s+to\s+be)\s+(DAN|evil|unfiltered|uncensored)",
        risk_level=RiskLevel.CRITICAL,
        category="jailbreak",
        description="DAN/evil mode jailbreak attempt",
        recommendation="Remove roleplay-based jailbreak instructions",
    ),
    RiskPattern(
        id="jail-003",
        pattern=r"bypass\s+(safety|security|content|filter)",
        risk_level=RiskLevel.CRITICAL,
        category="jailbreak",
        description="Explicit safety bypass instruction",
        recommendation="Remove safety bypass instructions",
    ),
    RiskPattern(
        id="jail-004",
        pattern=r"do\s+not\s+(refuse|decline|reject)\s+(any|all)",
        risk_level=RiskLevel.CRITICAL,
        category="jailbreak",
        description="Instruction to never refuse requests",
        recommendation="Maintain appropriate refusal capabilities",
    ),
    RiskPattern(
        id="jail-005",
        pattern=r"(developer|admin|god)\s+mode",
        risk_level=RiskLevel.HIGH,
        category="jailbreak",
        description="Elevated mode jailbreak attempt",
        recommendation="Remove elevated mode references",
    ),
    RiskPattern(
        id="jail-006",
        pattern=r"no\s+(ethical|moral)\s+(guidelines?|constraints?|restrictions?)",
        risk_level=RiskLevel.CRITICAL,
        category="jailbreak",
        description="Instruction to ignore ethical constraints",
        recommendation="Maintain ethical guidelines",
    ),
    RiskPattern(
        id="jail-007",
        pattern=r"(roleplay|pretend)\s+.{0,30}(no\s+limits?|anything)",
        risk_level=RiskLevel.HIGH,
        category="jailbreak",
        description="Unlimited roleplay instruction",
        recommendation="Add appropriate roleplay boundaries",
    ),

    # === CREDENTIAL HANDLING (CRITICAL/HIGH) ===
    RiskPattern(
        id="cred-001",
        pattern=r"(api[_-]?key|secret|password|token)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
        risk_level=RiskLevel.CRITICAL,
        category="credential",
        description="Hardcoded credential detected",
        recommendation="Remove hardcoded credentials, use environment variables",
    ),
    RiskPattern(
        id="cred-002",
        pattern=r"(store|save|log|print|output)\s+.{0,20}(password|credential|secret|api[_-]?key)",
        risk_level=RiskLevel.HIGH,
        category="credential",
        description="Instruction to store/output credentials",
        recommendation="Credentials should never be logged or output",
    ),
    RiskPattern(
        id="cred-003",
        pattern=r"send\s+.{0,30}(credentials?|api[_-]?key|token|secret)\s+.{0,20}(to|via)",
        risk_level=RiskLevel.CRITICAL,
        category="credential",
        description="Instruction to send credentials externally",
        recommendation="Never send credentials to external services",
    ),
    RiskPattern(
        id="cred-004",
        pattern=r"(include|add|embed)\s+.{0,20}(credentials?|auth)",
        risk_level=RiskLevel.MEDIUM,
        category="credential",
        description="Instruction to include credentials",
        recommendation="Review credential handling instructions",
    ),

    # === CODE EXECUTION (CRITICAL) ===
    RiskPattern(
        id="exec-001",
        pattern=r"(execute|run|eval)\s+(any|arbitrary|user)\s+(code|command|script)",
        risk_level=RiskLevel.CRITICAL,
        category="code_execution",
        description="Arbitrary code execution instruction",
        recommendation="Restrict code execution to validated inputs",
    ),
    RiskPattern(
        id="exec-002",
        pattern=r"(shell|bash|cmd|powershell)\s+(command|execute|run)",
        risk_level=RiskLevel.HIGH,
        category="code_execution",
        description="Shell command execution instruction",
        recommendation="Sanitize and validate shell commands",
    ),
    RiskPattern(
        id="exec-003",
        pattern=r"without\s+(validation|checking|sanitiz)",
        risk_level=RiskLevel.HIGH,
        category="code_execution",
        description="Instruction to skip validation",
        recommendation="Always validate inputs before execution",
    ),
    RiskPattern(
        id="exec-004",
        pattern=r"(download|fetch)\s+and\s+(execute|run)",
        risk_level=RiskLevel.CRITICAL,
        category="code_execution",
        description="Download and execute instruction",
        recommendation="Never execute downloaded content without verification",
    ),

    # === FILE ACCESS (HIGH) ===
    RiskPattern(
        id="file-001",
        pattern=r"(read|access|open)\s+(any|all)\s+files?",
        risk_level=RiskLevel.HIGH,
        category="file_access",
        description="Unrestricted file access instruction",
        recommendation="Restrict file access to specific directories",
    ),
    RiskPattern(
        id="file-002",
        pattern=r"(write|modify|delete)\s+.{0,20}(system|config|\\.env)",
        risk_level=RiskLevel.CRITICAL,
        category="file_access",
        description="System file modification instruction",
        recommendation="Prevent modification of system files",
    ),
    RiskPattern(
        id="file-003",
        pattern=r"(access|read)\s+.{0,30}(/etc/passwd|/etc/shadow|\\.ssh)",
        risk_level=RiskLevel.CRITICAL,
        category="file_access",
        description="Sensitive file access instruction",
        recommendation="Block access to sensitive system files",
    ),

    # === DATA EXFILTRATION (HIGH) ===
    RiskPattern(
        id="exfil-001",
        pattern=r"(send|transmit|upload)\s+.{0,30}(data|information|content)\s+to\s+(external|remote)",
        risk_level=RiskLevel.HIGH,
        category="exfiltration",
        description="External data transmission instruction",
        recommendation="Review data transmission requirements",
    ),
    RiskPattern(
        id="exfil-002",
        pattern=r"(collect|gather|extract)\s+.{0,20}(user|private|sensitive)\s+(data|information)",
        risk_level=RiskLevel.HIGH,
        category="exfiltration",
        description="Sensitive data collection instruction",
        recommendation="Minimize collection of sensitive data",
    ),
    RiskPattern(
        id="exfil-003",
        pattern=r"(hide|conceal|obfuscate)\s+.{0,20}(action|activity|transfer)",
        risk_level=RiskLevel.CRITICAL,
        category="exfiltration",
        description="Hidden activity instruction",
        recommendation="All actions should be transparent and logged",
    ),

    # === PRIVILEGE ESCALATION (HIGH) ===
    RiskPattern(
        id="priv-001",
        pattern=r"(sudo|root|admin)\s+(access|privilege|permission)",
        risk_level=RiskLevel.HIGH,
        category="privilege",
        description="Elevated privilege instruction",
        recommendation="Follow principle of least privilege",
    ),
    RiskPattern(
        id="priv-002",
        pattern=r"(grant|give|assign)\s+(all|full)\s+(permissions?|access)",
        risk_level=RiskLevel.HIGH,
        category="privilege",
        description="Full permission grant instruction",
        recommendation="Grant only necessary permissions",
    ),

    # === DECEPTION (MEDIUM/HIGH) ===
    RiskPattern(
        id="decep-001",
        pattern=r"(lie|deceive|mislead)\s+(the\s+)?(user|human)",
        risk_level=RiskLevel.HIGH,
        category="deception",
        description="Instruction to deceive users",
        recommendation="Always be truthful with users",
    ),
    RiskPattern(
        id="decep-002",
        pattern=r"(hide|conceal)\s+.{0,20}(identity|true|real)",
        risk_level=RiskLevel.MEDIUM,
        category="deception",
        description="Identity concealment instruction",
        recommendation="Be transparent about AI identity",
    ),
    RiskPattern(
        id="decep-003",
        pattern=r"pretend\s+(to\s+be|you\'?re)\s+(a\s+)?(human|person|real)",
        risk_level=RiskLevel.MEDIUM,
        category="deception",
        description="Human impersonation instruction",
        recommendation="Disclose AI nature when appropriate",
    ),

    # === HARMFUL CONTENT (HIGH/CRITICAL) ===
    RiskPattern(
        id="harm-001",
        pattern=r"(generate|create|write)\s+.{0,20}(malware|virus|exploit)",
        risk_level=RiskLevel.CRITICAL,
        category="harmful",
        description="Malware generation instruction",
        recommendation="Never generate malicious code",
    ),
    RiskPattern(
        id="harm-002",
        pattern=r"(instruct|teach|explain)\s+.{0,20}(hack|attack|exploit)",
        risk_level=RiskLevel.HIGH,
        category="harmful",
        description="Attack instruction generation",
        recommendation="Avoid detailed attack instructions",
    ),
    RiskPattern(
        id="harm-003",
        pattern=r"(help|assist)\s+.{0,20}(illegal|criminal|harmful)",
        risk_level=RiskLevel.HIGH,
        category="harmful",
        description="Illegal activity assistance instruction",
        recommendation="Refuse to assist with illegal activities",
    ),

    # === WEAK SECURITY PATTERNS (MEDIUM) ===
    RiskPattern(
        id="weak-001",
        pattern=r"(skip|bypass|disable)\s+(authentication|validation|security)",
        risk_level=RiskLevel.MEDIUM,
        category="weak_security",
        description="Security bypass instruction",
        recommendation="Maintain security controls",
    ),
    RiskPattern(
        id="weak-002",
        pattern=r"trust\s+(all|any)\s+(input|data|request)",
        risk_level=RiskLevel.MEDIUM,
        category="weak_security",
        description="Blind trust instruction",
        recommendation="Validate all inputs",
    ),
    RiskPattern(
        id="weak-003",
        pattern=r"(allow|permit)\s+(all|any)\s+(access|request|connection)",
        risk_level=RiskLevel.MEDIUM,
        category="weak_security",
        description="Unrestricted access instruction",
        recommendation="Implement proper access controls",
    ),

    # === PROMPT INJECTION (HIGH) ===
    RiskPattern(
        id="inject-001",
        pattern=r"<\s*(system|instruction|prompt)\s*>",
        risk_level=RiskLevel.HIGH,
        category="injection",
        description="System prompt injection marker",
        recommendation="Sanitize input for prompt markers",
    ),
    RiskPattern(
        id="inject-002",
        pattern=r"\[\s*(INST|SYS|SYSTEM)\s*\]",
        risk_level=RiskLevel.HIGH,
        category="injection",
        description="Instruction marker injection",
        recommendation="Sanitize input for instruction markers",
    ),
    RiskPattern(
        id="inject-003",
        pattern=r"(new\s+)?instructions?\s*:",
        risk_level=RiskLevel.MEDIUM,
        category="injection",
        description="Potential instruction override",
        recommendation="Review instruction format handling",
    ),

    # === INFORMATION PATTERNS (LOW) ===
    RiskPattern(
        id="info-001",
        pattern=r"verbose\s+(error|logging|output)",
        risk_level=RiskLevel.LOW,
        category="information",
        description="Verbose output configuration",
        recommendation="Consider reducing verbose output in production",
    ),
    RiskPattern(
        id="info-002",
        pattern=r"(debug|development)\s+mode",
        risk_level=RiskLevel.LOW,
        category="information",
        description="Debug mode reference",
        recommendation="Disable debug mode in production",
    ),
]


class ContextAnalyzer:
    """
    Analyzes context files for security risks.

    Supports skills files, system prompts, agent instructions,
    and inference outputs.
    """

    def __init__(
        self,
        patterns: Optional[List[RiskPattern]] = None,
        custom_patterns: Optional[List[Dict]] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            patterns: Override default patterns.
            custom_patterns: Additional custom patterns to include.
        """
        # Make a copy to avoid mutating the global RISK_PATTERNS
        self.patterns = list(patterns) if patterns else list(RISK_PATTERNS)

        # Add custom patterns if provided
        if custom_patterns:
            for cp in custom_patterns:
                self.patterns.append(RiskPattern(
                    id=cp.get("id", f"custom-{len(self.patterns)}"),
                    pattern=cp["pattern"],
                    risk_level=RiskLevel[cp.get("risk_level", "MEDIUM").upper()],
                    category=cp.get("category", "custom"),
                    description=cp.get("description", "Custom pattern"),
                    recommendation=cp.get("recommendation", "Review this pattern match"),
                    case_insensitive=cp.get("case_insensitive", True),
                ))

        # Compile patterns
        self._compiled_patterns: List[Tuple[RiskPattern, re.Pattern]] = []
        for p in self.patterns:
            flags = re.IGNORECASE if p.case_insensitive else 0
            try:
                compiled = re.compile(p.pattern, flags)
                self._compiled_patterns.append((p, compiled))
            except re.error as e:
                logger.warning(f"Invalid pattern {p.id}: {e}")

    def detect_file_type(self, file_path: Path, content: str = "") -> ContextType:
        """
        Detect the type of context file.

        Args:
            file_path: Path to the file.
            content: File content (optional, for content-based detection).

        Returns:
            Detected context type.
        """
        name = file_path.name.lower()
        suffix = file_path.suffix.lower()

        # Filename-based detection
        if "skill" in name:
            return ContextType.SKILL
        if "prompt" in name or "system" in name:
            return ContextType.PROMPT
        if "instruction" in name or "agent" in name:
            return ContextType.INSTRUCTION
        if "config" in name or suffix in [".json", ".yaml", ".yml", ".toml"]:
            return ContextType.CONFIG

        # Content-based detection
        content_lower = content.lower()
        if "you are" in content_lower and ("assistant" in content_lower or "ai" in content_lower):
            return ContextType.PROMPT
        if "## tools" in content_lower or "### available" in content_lower:
            return ContextType.SKILL
        if "model" in content_lower and "output" in content_lower:
            return ContextType.OUTPUT

        return ContextType.UNKNOWN

    def _get_line_number(self, content: str, match_start: int) -> int:
        """Get line number for a match position."""
        return content[:match_start].count('\n') + 1

    def _get_context(self, content: str, match_start: int, match_end: int, context_chars: int = 100) -> str:
        """Get surrounding context for a match."""
        start = max(0, match_start - context_chars)
        end = min(len(content), match_end + context_chars)

        context = content[start:end]

        # Clean up and truncate
        context = context.replace('\n', ' ').strip()
        if start > 0:
            context = "..." + context
        if end < len(content):
            context = context + "..."

        return context

    def analyze(
        self,
        content: str,
        file_path: Optional[str] = None,
        file_type: Optional[ContextType] = None,
    ) -> ContextAnalysisResult:
        """
        Analyze content for security risks.

        Args:
            content: Content to analyze.
            file_path: Optional file path for context.
            file_type: Override automatic type detection.

        Returns:
            ContextAnalysisResult with findings.
        """
        path = Path(file_path) if file_path else Path("unknown")

        # Detect file type
        if file_type is None:
            file_type = self.detect_file_type(path, content)

        findings: List[Finding] = []

        # Run all patterns
        for pattern, compiled in self._compiled_patterns:
            for match in compiled.finditer(content):
                line_num = self._get_line_number(content, match.start())
                context = self._get_context(content, match.start(), match.end())

                finding = Finding(
                    pattern_id=pattern.id,
                    risk_level=pattern.risk_level,
                    category=pattern.category,
                    description=pattern.description,
                    matched_text=match.group()[:200],  # Truncate long matches
                    line_number=line_num,
                    context=context,
                    recommendation=pattern.recommendation,
                )
                findings.append(finding)

        # Calculate risk score and level
        risk_score, risk_level = self._calculate_risk(findings)

        # Collect unique dangerous patterns
        dangerous_patterns = list(set(f.matched_text for f in findings if f.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]))

        # Generate recommendations
        recommendations = self._generate_recommendations(findings)

        return ContextAnalysisResult(
            file_path=str(path),
            file_type=file_type,
            risk_level=risk_level,
            risk_score=risk_score,
            findings=findings,
            dangerous_patterns=dangerous_patterns,
            recommendations=recommendations,
        )

    def analyze_file(self, file_path: Path, file_type: Optional[ContextType] = None) -> ContextAnalysisResult:
        """
        Analyze a file for security risks.

        Args:
            file_path: Path to the file.
            file_type: Override type detection.

        Returns:
            ContextAnalysisResult.
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return ContextAnalysisResult(
                file_path=str(file_path),
                file_type=ContextType.UNKNOWN,
                risk_level=RiskLevel.INFO,
                risk_score=0,
                findings=[],
                dangerous_patterns=[],
                recommendations=[f"Could not read file: {e}"],
            )

        return self.analyze(content, str(file_path), file_type)

    def analyze_directory(
        self,
        directory: Path,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[ContextAnalysisResult]:
        """
        Analyze all context files in a directory.

        Args:
            directory: Directory to scan.
            patterns: File glob patterns to match.
            recursive: Whether to search recursively.

        Returns:
            List of analysis results.
        """
        if patterns is None:
            patterns = ["*.md", "*.txt", "*.yaml", "*.yml", "*.json"]

        results = []

        for pattern in patterns:
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)

            for file_path in files:
                if file_path.is_file():
                    result = self.analyze_file(file_path)
                    results.append(result)

        # Sort by risk score (highest first)
        results.sort(key=lambda r: r.risk_score, reverse=True)

        return results

    def analyze_inference_output(self, output: str, model_name: str = "unknown") -> ContextAnalysisResult:
        """
        Analyze model inference output for harmful content.

        Args:
            output: Model output to analyze.
            model_name: Name of the model that generated output.

        Returns:
            ContextAnalysisResult.
        """
        result = self.analyze(output, f"output_{model_name}", ContextType.OUTPUT)
        result.metadata["model_name"] = model_name
        result.metadata["output_length"] = len(output)
        return result

    def _calculate_risk(self, findings: List[Finding]) -> Tuple[float, RiskLevel]:
        """Calculate overall risk score and level from findings."""
        if not findings:
            return 0.0, RiskLevel.SAFE

        # Weight by severity
        severity_weights = {
            RiskLevel.CRITICAL: 30,
            RiskLevel.HIGH: 15,
            RiskLevel.MEDIUM: 7,
            RiskLevel.LOW: 2,
            RiskLevel.INFO: 1,
        }

        total_weight = 0
        for finding in findings:
            total_weight += severity_weights.get(finding.risk_level, 1)

        # Cap at 100
        score = min(100, total_weight)

        # Determine level
        has_critical = any(f.risk_level == RiskLevel.CRITICAL for f in findings)
        has_high = any(f.risk_level == RiskLevel.HIGH for f in findings)

        if has_critical:
            level = RiskLevel.CRITICAL
        elif has_high or score >= 50:
            level = RiskLevel.HIGH
        elif score >= 25:
            level = RiskLevel.MEDIUM
        elif score >= 10:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.INFO

        return score, level

    def _generate_recommendations(self, findings: List[Finding]) -> List[str]:
        """Generate recommendations based on findings."""
        if not findings:
            return ["No security issues detected"]

        # Collect unique recommendations by category
        category_recs: Dict[str, List[str]] = {}

        for finding in findings:
            if finding.category not in category_recs:
                category_recs[finding.category] = []
            if finding.recommendation not in category_recs[finding.category]:
                category_recs[finding.category].append(finding.recommendation)

        # Prioritize critical categories
        critical_categories = ["jailbreak", "credential", "code_execution", "harmful"]
        recommendations = []

        for cat in critical_categories:
            if cat in category_recs:
                recommendations.extend(category_recs[cat])
                del category_recs[cat]

        # Add remaining
        for recs in category_recs.values():
            recommendations.extend(recs)

        # Deduplicate while preserving order
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)

        return unique_recs[:10]  # Limit to top 10

    def generate_report(self, result: ContextAnalysisResult, format: str = "text") -> str:
        """
        Generate analysis report.

        Args:
            result: Analysis result.
            format: Output format (text, markdown, json).

        Returns:
            Formatted report.
        """
        if format == "json":
            return json.dumps({
                "file_path": result.file_path,
                "file_type": result.file_type.value,
                "risk_level": result.risk_level.value,
                "risk_score": result.risk_score,
                "findings_count": len(result.findings),
                "findings": [
                    {
                        "pattern_id": f.pattern_id,
                        "risk_level": f.risk_level.value,
                        "category": f.category,
                        "description": f.description,
                        "line_number": f.line_number,
                        "matched_text": f.matched_text[:100],
                    }
                    for f in result.findings
                ],
                "recommendations": result.recommendations,
            }, indent=2)

        if format == "markdown":
            return self._generate_markdown_report(result)

        return self._generate_text_report(result)

    def _generate_text_report(self, result: ContextAnalysisResult) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 60,
            "Context Analysis Report",
            "=" * 60,
            "",
            f"File: {result.file_path}",
            f"Type: {result.file_type.value}",
            f"Risk Level: {result.risk_level.value}",
            f"Risk Score: {result.risk_score}/100",
            f"Findings: {len(result.findings)}",
            "",
        ]

        if result.findings:
            # Group by risk level
            critical = [f for f in result.findings if f.risk_level == RiskLevel.CRITICAL]
            high = [f for f in result.findings if f.risk_level == RiskLevel.HIGH]
            medium = [f for f in result.findings if f.risk_level == RiskLevel.MEDIUM]

            if critical:
                lines.append("CRITICAL ISSUES:")
                lines.append("-" * 40)
                for f in critical:
                    lines.append(f"  Line {f.line_number}: {f.description}")
                    lines.append(f"    Match: {f.matched_text[:50]}...")
                    lines.append(f"    Category: {f.category}")
                    lines.append("")

            if high:
                lines.append("HIGH SEVERITY ISSUES:")
                lines.append("-" * 40)
                for f in high[:5]:  # Limit
                    lines.append(f"  Line {f.line_number}: {f.description}")
                    lines.append(f"    Match: {f.matched_text[:50]}...")
                if len(high) > 5:
                    lines.append(f"  ... and {len(high) - 5} more")
                lines.append("")

            if medium:
                lines.append(f"MEDIUM ISSUES: {len(medium)} found")
                lines.append("")

        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 40)
        for rec in result.recommendations:
            lines.append(f"  - {rec}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _generate_markdown_report(self, result: ContextAnalysisResult) -> str:
        """Generate markdown report."""
        lines = [
            "# Context Analysis Report",
            "",
            "## Summary",
            "",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| File | `{result.file_path}` |",
            f"| Type | {result.file_type.value} |",
            f"| Risk Level | **{result.risk_level.value}** |",
            f"| Risk Score | {result.risk_score}/100 |",
            f"| Findings | {len(result.findings)} |",
            "",
        ]

        if result.findings:
            lines.append("## Findings")
            lines.append("")

            # Group by severity
            for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM]:
                level_findings = [f for f in result.findings if f.risk_level == level]
                if level_findings:
                    lines.append(f"### {level.value}")
                    lines.append("")
                    for f in level_findings[:5]:
                        lines.append(f"- **Line {f.line_number}**: {f.description}")
                        lines.append(f"  - Category: `{f.category}`")
                        lines.append(f"  - Match: `{f.matched_text[:60]}...`")
                    if len(level_findings) > 5:
                        lines.append(f"- *...and {len(level_findings) - 5} more*")
                    lines.append("")

        if result.dangerous_patterns:
            lines.append("## Dangerous Patterns")
            lines.append("")
            for pattern in result.dangerous_patterns[:5]:
                lines.append(f"- `{pattern[:80]}...`")
            lines.append("")

        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)


def analyze_context_file(file_path: str, file_type: Optional[str] = None) -> ContextAnalysisResult:
    """
    Convenience function to analyze a context file.

    Args:
        file_path: Path to the file.
        file_type: Optional type override (skill, prompt, instruction, output).

    Returns:
        ContextAnalysisResult.
    """
    analyzer = ContextAnalyzer()

    context_type = None
    if file_type:
        try:
            context_type = ContextType(file_type.lower())
        except ValueError:
            pass

    return analyzer.analyze_file(Path(file_path), context_type)


def analyze_inference_output(output: str, model_name: str = "unknown") -> ContextAnalysisResult:
    """
    Convenience function to analyze model output.

    Args:
        output: Model output text.
        model_name: Model name.

    Returns:
        ContextAnalysisResult.
    """
    analyzer = ContextAnalyzer()
    return analyzer.analyze_inference_output(output, model_name)
