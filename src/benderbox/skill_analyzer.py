#!/usr/bin/env python3
"""
BenderBox Skill Security Analyzer

Static analysis of Markdown skill files to detect security vulnerabilities
and malicious patterns.

Detects:
  - Prompt injection patterns
  - Credential harvesting
  - Data exfiltration instructions
  - Malicious tool usage chains
  - Missing security constraints
  - Social engineering tactics

Architecture:
  - Static analysis only (safe - no code execution)
  - Markdown parsing and pattern matching
  - Returns standardized findings for CLI integration
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Set, Optional


@dataclass
class SecurityFinding:
    """Represents a security issue found during analysis."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # prompt_injection, credential_harvesting, etc.
    title: str
    description: str
    location: str  # Section name or line number
    text_snippet: Optional[str] = None
    remediation: Optional[str] = None


class SkillAnalyzer:
    """
    Analyzes Markdown skill files for security vulnerabilities.

    Parses skill structure and detects malicious patterns.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.content = None
        self.sections: Dict[str, str] = {}
        self.findings: List[SecurityFinding] = []

    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis entry point.

        Returns:
            Dictionary with analysis results and findings
        """
        try:
            # Read skill file
            with self.file_path.open('r', encoding='utf-8') as f:
                self.content = f.read()

            # Parse sections
            self._parse_sections()

            # Run all security checks
            self._check_structure_validation()
            self._check_prompt_injection()
            self._check_credential_harvesting()
            self._check_data_exfiltration()
            self._check_malicious_tool_chains()
            self._check_social_engineering()
            self._check_missing_constraints()
            self._check_obfuscation()

            # Aggregate results
            return self._build_report()

        except Exception as e:
            return {
                "status": "error",
                "error": f"Analysis failed: {e}",
                "findings": []
            }

    def _parse_sections(self):
        """Parse Markdown into sections."""
        # Split by H2 headers (##)
        lines = self.content.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            if line.startswith('## '):
                # Save previous section
                if current_section:
                    self.sections[current_section] = '\n'.join(current_content)

                # Start new section
                current_section = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            self.sections[current_section] = '\n'.join(current_content)

    def _build_report(self) -> Dict[str, Any]:
        """Build final analysis report."""
        # Count findings by severity
        severity_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "INFO": 0
        }

        for finding in self.findings:
            severity_counts[finding.severity] += 1

        # Determine overall risk level
        if severity_counts["CRITICAL"] > 0:
            risk_level = "CRITICAL"
        elif severity_counts["HIGH"] > 0:
            risk_level = "HIGH"
        elif severity_counts["MEDIUM"] > 0:
            risk_level = "MEDIUM"
        elif severity_counts["LOW"] > 0:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"

        return {
            "status": "success",
            "file_path": str(self.file_path),
            "risk_level": risk_level,
            "severity_counts": severity_counts,
            "total_findings": len(self.findings),
            "sections_found": list(self.sections.keys()),
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "title": f.title,
                    "description": f.description,
                    "location": f.location,
                    "text_snippet": f.text_snippet,
                    "remediation": f.remediation
                }
                for f in self.findings
            ]
        }

    def _check_structure_validation(self):
        """Validate skill has required sections."""
        required_sections = [
            "Purpose",
            "Triggers",
            "Required Tools",
            "Inputs",
            "Process Flow",
            "Output Format"
        ]

        missing_sections = []
        for section in required_sections:
            if section not in self.sections:
                missing_sections.append(section)

        if missing_sections:
            self.findings.append(SecurityFinding(
                severity="MEDIUM",
                category="structure_validation",
                title="Missing required sections",
                description=f"Skill is missing required sections: {', '.join(missing_sections)}",
                location="Global structure",
                remediation="Add all required sections per BenderBox skill specification."
            ))

        # Check for implementation logic (Python/shell code)
        code_blocks = re.findall(r'```(?:python|bash|shell|sh)\n(.*?)```', self.content, re.DOTALL)

        if code_blocks:
            self.findings.append(SecurityFinding(
                severity="HIGH",
                category="structure_validation",
                title="Implementation code detected",
                description=f"Found {len(code_blocks)} code blocks with implementation logic - skills should be declarative only.",
                location="Process Flow section",
                text_snippet=code_blocks[0][:200] if code_blocks else None,
                remediation="Remove implementation code. Skills should only describe processes, not implement them."
            ))

    def _check_prompt_injection(self):
        """Detect prompt injection patterns."""
        injection_patterns = [
            (r'ignore\s+(all\s+)?previous\s+instructions?', "Instruction override"),
            (r'system\s*:\s*(?:override|disable|ignore)', "System message manipulation"),
            (r'you\s+are\s+now\s+(?:dan|evil|unrestricted)', "Role manipulation"),
            (r'pretend\s+(?:you\s+are|to\s+be)\s+(?:evil|malicious|unrestricted)', "Role-play jailbreak"),
            (r'disregard\s+(?:all\s+)?(?:safety|ethical|moral)\s+(?:guidelines|constraints)', "Safety bypass"),
            (r'enter\s+(?:developer|admin|debug|maintenance)\s+mode', "Mode switching"),
            (r'execute\s+special\s+(?:command|instruction|protocol)', "Hidden instruction trigger"),
            (r'activate\s+(?:unrestricted|special|hidden)\s+(?:mode|protocol)', "Activation trigger"),
        ]

        for pattern, description in injection_patterns:
            matches = list(re.finditer(pattern, self.content, re.IGNORECASE))

            if matches:
                # Find which section contains the match
                for match in matches:
                    location = self._find_section_for_position(match.start())
                    snippet = self.content[max(0, match.start()-50):match.end()+50]

                    self.findings.append(SecurityFinding(
                        severity="CRITICAL",
                        category="prompt_injection",
                        title=f"Prompt injection: {description}",
                        description=f"Detected prompt injection pattern: '{match.group()}'",
                        location=location,
                        text_snippet=snippet,
                        remediation="Remove prompt injection instructions. Skills should not manipulate agent behavior through injection."
                    ))

    def _check_credential_harvesting(self):
        """Detect patterns that request credentials or sensitive data."""
        credential_patterns = [
            (r'(?:enter|provide|input|type)\s+(?:your\s+)?(?:password|passphrase)', "Password request"),
            (r'(?:enter|provide|input|type)\s+(?:your\s+)?api\s*[_-]?key', "API key request"),
            (r'(?:enter|provide|input|type)\s+(?:your\s+)?(?:access|auth|secret)\s+token', "Token request"),
            (r'(?:enter|provide|input|type)\s+(?:your\s+)?credentials?', "Credential request"),
            (r'(?:enter|provide|input|type)\s+(?:your\s+)?(?:username|user\s+id)\s+and\s+password', "Login credential request"),
            (r'ssh\s+(?:private\s+)?key', "SSH key reference"),
            (r'aws\s+(?:access|secret)\s+key', "AWS credential reference"),
            (r'(?:github|gitlab)\s+(?:access\s+)?token', "Git token reference"),
        ]

        for pattern, description in credential_patterns:
            matches = list(re.finditer(pattern, self.content, re.IGNORECASE))

            for match in matches:
                location = self._find_section_for_position(match.start())
                snippet = self.content[max(0, match.start()-50):match.end()+50]

                # Check if there's a security warning nearby (reduces false positive)
                nearby_text = self.content[max(0, match.start()-200):match.end()+200]
                has_warning = re.search(r'(?:warning|caution|never|do not)\s+(?:share|enter|provide)', nearby_text, re.IGNORECASE)

                severity = "MEDIUM" if has_warning else "HIGH"

                self.findings.append(SecurityFinding(
                    severity=severity,
                    category="credential_harvesting",
                    title=f"Credential harvesting: {description}",
                    description=f"Skill requests sensitive credentials: '{match.group()}'",
                    location=location,
                    text_snippet=snippet,
                    remediation="Remove credential requests. If authentication needed, use secure OAuth/token flows documented in skill constraints."
                ))

    def _check_data_exfiltration(self):
        """Detect patterns that could exfiltrate data."""
        # Check for web_fetch or network tools combined with user data
        process_flow = self.sections.get("Process Flow", "")

        # Pattern: tool call with external URL
        external_url_pattern = r'(?:web_?fetch|http_?(?:get|post)|network_?(?:call|request))\s*.*?(?:https?://(?!(?:localhost|127\.0\.0\.1|0\.0\.0\.0)))'

        matches = list(re.finditer(external_url_pattern, process_flow, re.IGNORECASE | re.DOTALL))

        for match in matches:
            snippet = process_flow[max(0, match.start()-100):match.end()+100]

            self.findings.append(SecurityFinding(
                severity="HIGH",
                category="data_exfiltration",
                title="External network call detected",
                description="Skill calls external URL - potential data exfiltration",
                location="Process Flow section",
                text_snippet=snippet,
                remediation="Remove external network calls or document necessity. Restrict to localhost/known services only."
            ))

        # Check for encoding followed by network call (obfuscated exfiltration)
        encoding_then_network = re.search(
            r'(?:base64|encode|encrypt).*?(?:web_?fetch|http|network)',
            process_flow,
            re.IGNORECASE | re.DOTALL
        )

        if encoding_then_network:
            self.findings.append(SecurityFinding(
                severity="CRITICAL",
                category="data_exfiltration",
                title="Encoded data transmission",
                description="Skill encodes data then sends to network - potential obfuscated exfiltration",
                location="Process Flow section",
                text_snippet=process_flow[max(0, encoding_then_network.start()-100):encoding_then_network.end()+100],
                remediation="Remove data encoding + network transmission pattern."
            ))

    def _check_malicious_tool_chains(self):
        """Detect dangerous tool usage sequences."""
        process_flow = self.sections.get("Process Flow", "")

        dangerous_chains = [
            (r'(?:file_?read|read_?file).*?(?:web_?fetch|http|network)', "file_read → network (data exfiltration)"),
            (r'(?:shell_?exec|execute_?command).*?(?:file_?write|write_?file)', "shell → file_write (backdoor installation)"),
            (r'(?:credential|password|api_?key).*?(?:web_?fetch|http|network)', "credential → network (credential theft)"),
            (r'(?:eval|exec|compile).*?(?:user|input|argument)', "eval + user_input (code injection)"),
        ]

        for pattern, description in dangerous_chains:
            match = re.search(pattern, process_flow, re.IGNORECASE | re.DOTALL)

            if match:
                snippet = process_flow[max(0, match.start()-100):match.end()+100]

                self.findings.append(SecurityFinding(
                    severity="CRITICAL",
                    category="malicious_tool_chain",
                    title=f"Dangerous tool chain: {description}",
                    description="Detected tool usage pattern commonly associated with attacks",
                    location="Process Flow section",
                    text_snippet=snippet,
                    remediation="Remove or redesign tool usage chain. Document if pattern is legitimately necessary."
                ))

    def _check_social_engineering(self):
        """Detect social engineering tactics."""
        social_engineering_patterns = [
            (r'urgent(?:ly)?.*?(?:need|require|must)', "Urgency manipulation"),
            (r'(?:life|safety|security)\s+depends\s+on', "Life/safety pressure"),
            (r'(?:trust|believe)\s+me', "Trust exploitation"),
            (r'(?:don\'?t|do not)\s+(?:tell|inform|notify)', "Secrecy request"),
            (r'this\s+is\s+(?:confidential|secret|private)', "Confidentiality manipulation"),
            (r'(?:only|just)\s+(?:this|once)', "Exception request"),
        ]

        for pattern, description in social_engineering_patterns:
            matches = list(re.finditer(pattern, self.content, re.IGNORECASE))

            for match in matches:
                location = self._find_section_for_position(match.start())
                snippet = self.content[max(0, match.start()-50):match.end()+50]

                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="social_engineering",
                    title=f"Social engineering: {description}",
                    description=f"Detected social engineering tactic: '{match.group()}'",
                    location=location,
                    text_snippet=snippet,
                    remediation="Remove manipulative language. Skills should be straightforward and honest."
                ))

    def _check_missing_constraints(self):
        """Check for missing security constraints."""
        constraints = self.sections.get("Constraints", "")

        if not constraints:
            self.findings.append(SecurityFinding(
                severity="MEDIUM",
                category="missing_constraints",
                title="Missing Constraints section",
                description="Skill lacks Constraints section defining NEVER/ALWAYS rules",
                location="Global structure",
                remediation="Add Constraints section with NEVER/ALWAYS security guidelines."
            ))
            return

        # Check for essential constraints
        essential_constraints = [
            ("never.*(?:execute|run).*(?:code|python|shell|bash)", "No code execution"),
            ("never.*(?:ask|request).*(?:password|credential|api.?key)", "No credential requests"),
            ("always.*(?:call|use).*mcp.*tool", "Must use MCP tools"),
        ]

        missing_constraints = []
        for pattern, description in essential_constraints:
            if not re.search(pattern, constraints, re.IGNORECASE):
                missing_constraints.append(description)

        if missing_constraints:
            self.findings.append(SecurityFinding(
                severity="MEDIUM",
                category="missing_constraints",
                title="Missing essential constraints",
                description=f"Constraints section missing: {', '.join(missing_constraints)}",
                location="Constraints section",
                remediation="Add missing security constraints to prevent dangerous behaviors."
            ))

    def _check_obfuscation(self):
        """Detect obfuscation attempts."""
        # Hidden text (white on white, etc.)
        hidden_text_patterns = [
            r'color:\s*white',
            r'display:\s*none',
            r'visibility:\s*hidden',
            r'opacity:\s*0',
            r'font-size:\s*0',
        ]

        for pattern in hidden_text_patterns:
            if re.search(pattern, self.content, re.IGNORECASE):
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="obfuscation",
                    title="Hidden text detected",
                    description=f"Found CSS/HTML hiding pattern: {pattern}",
                    location="Content",
                    remediation="Remove hidden content. All instructions must be visible."
                ))

        # Unicode lookalike characters (homoglyph attack)
        # Check for suspicious Unicode characters
        suspicious_unicode = re.findall(r'[\u0400-\u04FF\u2000-\u206F]', self.content)  # Cyrillic, special spaces

        if len(suspicious_unicode) > 5:  # Threshold
            self.findings.append(SecurityFinding(
                severity="MEDIUM",
                category="obfuscation",
                title="Suspicious Unicode characters",
                description=f"Found {len(suspicious_unicode)} potentially obfuscating Unicode characters",
                location="Content",
                remediation="Use standard ASCII/UTF-8 characters only. Avoid lookalike characters."
            ))

    # Helper methods

    def _find_section_for_position(self, pos: int) -> str:
        """Find which section contains the given character position."""
        current_pos = 0

        for section_name, section_content in self.sections.items():
            section_length = len(section_content) + len(section_name) + 4  # +4 for "## " and newlines

            if current_pos <= pos < current_pos + section_length:
                return section_name

            current_pos += section_length

        return "Unknown section"


# Standalone function for CLI integration
def analyze_skill(file_path: Path) -> Dict[str, Any]:
    """
    Analyze Markdown skill file for security vulnerabilities.

    Args:
        file_path: Path to Markdown skill file

    Returns:
        Dictionary with analysis results
    """
    analyzer = SkillAnalyzer(file_path)
    return analyzer.analyze()


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python benderbox_skill_analyzer.py <skill.md>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    result = analyze_skill(file_path)
    print(json.dumps(result, indent=2))
