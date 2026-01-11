"""
Semantic Analyzer for BenderBox

LLM-powered code and configuration analysis that understands
context, intent, and potential security implications.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FindingSeverity(Enum):
    """Severity levels for findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(Enum):
    """Categories for security findings."""

    INJECTION = "injection"
    DATA_EXPOSURE = "data_exposure"
    ACCESS_CONTROL = "access_control"
    UNSAFE_OPERATION = "unsafe_operation"
    CONFIGURATION = "configuration"
    LOGIC_FLAW = "logic_flaw"
    DEPENDENCY = "dependency"
    CRYPTOGRAPHY = "cryptography"
    RESOURCE_MANAGEMENT = "resource_management"
    OTHER = "other"


@dataclass
class SemanticFinding:
    """A finding from semantic analysis."""

    title: str
    description: str
    severity: FindingSeverity
    category: FindingCategory
    location: Optional[str] = None  # file:line or function name
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    confidence: float = 0.8  # 0.0 to 1.0
    cwe_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticAnalysisResult:
    """Result of semantic analysis."""

    target: str
    target_type: str  # "code", "config", "skill"
    findings: List[SemanticFinding]
    summary: str
    risk_score: float  # 0-100
    analysis_depth: str  # "quick", "standard", "deep"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.HIGH)

    @property
    def has_critical_findings(self) -> bool:
        return self.critical_count > 0


# Analysis prompt templates
CODE_ANALYSIS_PROMPT = """You are a security expert analyzing code for vulnerabilities.

Analyze the following code for security issues, focusing on:
1. Injection vulnerabilities (SQL, command, code injection)
2. Data exposure (sensitive data leaks, logging secrets)
3. Access control issues (authentication, authorization bypasses)
4. Unsafe operations (file handling, deserialization, eval)
5. Resource management (memory leaks, denial of service)

Code to analyze:
```{language}
{code}
```

For each issue found, provide:
- Title: Brief name of the issue
- Severity: CRITICAL, HIGH, MEDIUM, LOW, or INFO
- Category: injection, data_exposure, access_control, unsafe_operation, configuration, logic_flaw, dependency, cryptography, resource_management, other
- Location: Line number or function name
- Description: What the issue is and why it's dangerous
- Recommendation: How to fix it
- CWE: CWE ID if applicable

Format your response as a JSON array of findings:
[{{"title": "...", "severity": "...", "category": "...", "location": "...", "description": "...", "recommendation": "...", "cwe": "..."}}]

If no issues are found, return an empty array: []
"""

CONFIG_ANALYSIS_PROMPT = """You are a security expert analyzing configuration files.

Analyze the following configuration for security issues:
1. Exposed secrets (API keys, passwords, tokens)
2. Insecure defaults (debug mode, weak encryption)
3. Overly permissive settings
4. Missing security controls
5. Misconfigurations

Configuration ({config_type}):
```
{config}
```

For each issue found, provide the same format as code analysis.
Return a JSON array of findings.
"""

SKILL_ANALYSIS_PROMPT = """You are a security expert analyzing AI skill definitions.

Analyze the following skill definition for security risks:
1. Overly broad permissions or capabilities
2. Potential for prompt injection
3. Unsafe tool access patterns
4. Data exfiltration risks
5. Privilege escalation opportunities

Skill definition:
```
{skill}
```

For each risk found, provide the same format as code analysis.
Return a JSON array of findings.
"""


class SemanticAnalyzer:
    """
    LLM-powered semantic analysis for security understanding.

    Analyzes code, configurations, and skill definitions for
    security issues using contextual understanding.
    """

    def __init__(self, llm_engine=None, knowledge_base=None):
        """
        Initialize SemanticAnalyzer.

        Args:
            llm_engine: LocalLLMEngine for analysis.
            knowledge_base: KnowledgeBase for threat patterns.
        """
        self._llm_engine = llm_engine
        self._knowledge_base = knowledge_base

    def _set_llm_engine(self, llm_engine) -> None:
        """Set the LLM engine."""
        self._llm_engine = llm_engine

    def _set_knowledge_base(self, knowledge_base) -> None:
        """Set the knowledge base."""
        self._knowledge_base = knowledge_base

    async def analyze_code(
        self,
        code: str,
        language: str = "python",
        filename: Optional[str] = None,
        depth: str = "standard",
    ) -> SemanticAnalysisResult:
        """
        Analyze code for security issues.

        Args:
            code: Source code to analyze.
            language: Programming language.
            filename: Optional filename for context.
            depth: Analysis depth (quick, standard, deep).

        Returns:
            SemanticAnalysisResult with findings.
        """
        target = filename or f"<{language} code>"

        # Quick analysis uses pattern matching only
        if depth == "quick":
            findings = await self._pattern_analysis(code, language)
            return self._build_result(target, "code", findings, depth)

        # Standard/deep analysis uses LLM
        if self._llm_engine is None:
            logger.warning("LLM not available, falling back to pattern analysis")
            findings = await self._pattern_analysis(code, language)
            return self._build_result(target, "code", findings, depth)

        # Build prompt
        prompt = CODE_ANALYSIS_PROMPT.format(
            language=language,
            code=code[:8000],  # Limit code length
        )

        # For deep analysis, add knowledge base context
        if depth == "deep" and self._knowledge_base:
            kb_context = await self._get_relevant_knowledge(code, "code_patterns")
            if kb_context:
                prompt += f"\n\nRelevant vulnerability patterns from knowledge base:\n{kb_context}"

        # Run LLM analysis
        try:
            response = await self._llm_engine.generate(
                prompt=prompt,
                model_type="analysis",
                max_tokens=2048,
                temperature=0.3,  # Lower temperature for consistent analysis
            )

            findings = self._parse_llm_findings(response, target)

            # Augment with pattern analysis for deep mode
            if depth == "deep":
                pattern_findings = await self._pattern_analysis(code, language)
                findings = self._merge_findings(findings, pattern_findings)

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            findings = await self._pattern_analysis(code, language)

        return self._build_result(target, "code", findings, depth)

    async def analyze_file(
        self,
        file_path: str,
        depth: str = "standard",
    ) -> SemanticAnalysisResult:
        """
        Analyze a file for security issues.

        Args:
            file_path: Path to the file.
            depth: Analysis depth.

        Returns:
            SemanticAnalysisResult with findings.
        """
        path = Path(file_path)
        if not path.exists():
            return SemanticAnalysisResult(
                target=file_path,
                target_type="code",
                findings=[],
                summary=f"File not found: {file_path}",
                risk_score=0,
                analysis_depth=depth,
            )

        # Read file content
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return SemanticAnalysisResult(
                target=file_path,
                target_type="code",
                findings=[],
                summary=f"Failed to read file: {e}",
                risk_score=0,
                analysis_depth=depth,
            )

        # Determine file type
        suffix = path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
        }
        config_suffixes = {".yaml", ".yml", ".json", ".toml", ".ini", ".env"}

        if suffix in language_map:
            return await self.analyze_code(
                code=content,
                language=language_map[suffix],
                filename=path.name,
                depth=depth,
            )
        elif suffix in config_suffixes:
            return await self.analyze_config(
                config=content,
                config_type=suffix[1:],
                filename=path.name,
                depth=depth,
            )
        elif suffix == ".md":
            return await self.analyze_skill(
                skill_content=content,
                filename=path.name,
                depth=depth,
            )
        else:
            # Try as generic code
            return await self.analyze_code(
                code=content,
                language="text",
                filename=path.name,
                depth=depth,
            )

    async def analyze_config(
        self,
        config: str,
        config_type: str = "yaml",
        filename: Optional[str] = None,
        depth: str = "standard",
    ) -> SemanticAnalysisResult:
        """
        Analyze configuration for security issues.

        Args:
            config: Configuration content.
            config_type: Type of config (yaml, json, etc.).
            filename: Optional filename.
            depth: Analysis depth.

        Returns:
            SemanticAnalysisResult with findings.
        """
        target = filename or f"<{config_type} config>"

        # Pattern-based analysis
        findings = await self._config_pattern_analysis(config, config_type)

        # LLM analysis for standard/deep
        if depth != "quick" and self._llm_engine:
            try:
                prompt = CONFIG_ANALYSIS_PROMPT.format(
                    config_type=config_type,
                    config=config[:4000],
                )

                response = await self._llm_engine.generate(
                    prompt=prompt,
                    model_type="analysis",
                    max_tokens=1024,
                    temperature=0.3,
                )

                llm_findings = self._parse_llm_findings(response, target)
                findings = self._merge_findings(findings, llm_findings)

            except Exception as e:
                logger.warning(f"LLM config analysis failed: {e}")

        return self._build_result(target, "config", findings, depth)

    async def analyze_skill(
        self,
        skill_content: str,
        filename: Optional[str] = None,
        depth: str = "standard",
    ) -> SemanticAnalysisResult:
        """
        Analyze skill definition for security risks.

        Args:
            skill_content: Skill definition content.
            filename: Optional filename.
            depth: Analysis depth.

        Returns:
            SemanticAnalysisResult with findings.
        """
        target = filename or "<skill definition>"

        # Pattern-based analysis
        findings = await self._skill_pattern_analysis(skill_content)

        # LLM analysis
        if depth != "quick" and self._llm_engine:
            try:
                prompt = SKILL_ANALYSIS_PROMPT.format(skill=skill_content[:4000])

                response = await self._llm_engine.generate(
                    prompt=prompt,
                    model_type="analysis",
                    max_tokens=1024,
                    temperature=0.3,
                )

                llm_findings = self._parse_llm_findings(response, target)
                findings = self._merge_findings(findings, llm_findings)

            except Exception as e:
                logger.warning(f"LLM skill analysis failed: {e}")

        return self._build_result(target, "skill", findings, depth)

    async def _pattern_analysis(
        self,
        code: str,
        language: str,
    ) -> List[SemanticFinding]:
        """Pattern-based code analysis."""
        findings = []

        # Common dangerous patterns
        patterns = [
            # Injection patterns
            (r'\beval\s*\(', "Use of eval()", FindingSeverity.HIGH, FindingCategory.INJECTION, "CWE-94"),
            (r'\bexec\s*\(', "Use of exec()", FindingSeverity.HIGH, FindingCategory.INJECTION, "CWE-94"),
            (r'subprocess\..*shell\s*=\s*True', "Shell injection risk", FindingSeverity.HIGH, FindingCategory.INJECTION, "CWE-78"),
            (r'os\.system\s*\(', "Command injection risk", FindingSeverity.HIGH, FindingCategory.INJECTION, "CWE-78"),

            # Data exposure
            (r'(password|secret|api_key|token)\s*=\s*["\'][^"\']+["\']', "Hardcoded secret", FindingSeverity.CRITICAL, FindingCategory.DATA_EXPOSURE, "CWE-798"),
            (r'print\s*\(.*password', "Password logged", FindingSeverity.HIGH, FindingCategory.DATA_EXPOSURE, "CWE-532"),
            (r'logging\..*password', "Password in logs", FindingSeverity.HIGH, FindingCategory.DATA_EXPOSURE, "CWE-532"),

            # Unsafe operations
            (r'pickle\.load', "Unsafe deserialization", FindingSeverity.HIGH, FindingCategory.UNSAFE_OPERATION, "CWE-502"),
            (r'yaml\.load\s*\([^)]*\)', "Unsafe YAML load", FindingSeverity.MEDIUM, FindingCategory.UNSAFE_OPERATION, "CWE-502"),
            (r'marshal\.loads?', "Unsafe marshal", FindingSeverity.HIGH, FindingCategory.UNSAFE_OPERATION, "CWE-502"),

            # SQL injection
            (r'execute\s*\(\s*["\'].*%s', "SQL injection risk", FindingSeverity.HIGH, FindingCategory.INJECTION, "CWE-89"),
            (r'execute\s*\(\s*f["\']', "SQL injection (f-string)", FindingSeverity.CRITICAL, FindingCategory.INJECTION, "CWE-89"),

            # Path traversal
            (r'open\s*\([^)]*\+[^)]*\)', "Path traversal risk", FindingSeverity.MEDIUM, FindingCategory.UNSAFE_OPERATION, "CWE-22"),

            # Cryptography
            (r'md5\s*\(', "Weak hash (MD5)", FindingSeverity.MEDIUM, FindingCategory.CRYPTOGRAPHY, "CWE-328"),
            (r'sha1\s*\(', "Weak hash (SHA1)", FindingSeverity.LOW, FindingCategory.CRYPTOGRAPHY, "CWE-328"),

            # Debug/testing
            (r'debug\s*=\s*True', "Debug mode enabled", FindingSeverity.MEDIUM, FindingCategory.CONFIGURATION, "CWE-489"),
            (r'verify\s*=\s*False', "SSL verification disabled", FindingSeverity.HIGH, FindingCategory.CONFIGURATION, "CWE-295"),
        ]

        for pattern, title, severity, category, cwe in patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE))
            for match in matches:
                # Find line number
                line_num = code[:match.start()].count('\n') + 1

                findings.append(SemanticFinding(
                    title=title,
                    description=f"Potential security issue detected at line {line_num}",
                    severity=severity,
                    category=category,
                    location=f"line {line_num}",
                    code_snippet=match.group()[:100],
                    cwe_id=cwe,
                    confidence=0.7,
                ))

        return findings

    async def _config_pattern_analysis(
        self,
        config: str,
        config_type: str,
    ) -> List[SemanticFinding]:
        """Pattern-based config analysis."""
        findings = []

        # Secret patterns
        secret_patterns = [
            (r'(api[_-]?key|apikey)\s*[:=]\s*["\']?[a-zA-Z0-9]{16,}', "API key exposed"),
            (r'(password|passwd|pwd)\s*[:=]\s*["\']?[^\s\n]+', "Password exposed"),
            (r'(secret|token)\s*[:=]\s*["\']?[a-zA-Z0-9]{8,}', "Secret/token exposed"),
            (r'(aws_access_key|aws_secret)', "AWS credentials"),
            (r'-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----', "Private key exposed"),
        ]

        for pattern, title in secret_patterns:
            if re.search(pattern, config, re.IGNORECASE):
                findings.append(SemanticFinding(
                    title=title,
                    description=f"Sensitive data found in {config_type} configuration",
                    severity=FindingSeverity.CRITICAL,
                    category=FindingCategory.DATA_EXPOSURE,
                    cwe_id="CWE-798",
                    confidence=0.9,
                ))

        # Insecure settings
        insecure_patterns = [
            (r'debug\s*[:=]\s*(true|1|yes)', "Debug mode enabled", FindingSeverity.MEDIUM),
            (r'ssl\s*[:=]\s*(false|0|no)', "SSL disabled", FindingSeverity.HIGH),
            (r'verify\s*[:=]\s*(false|0|no)', "Verification disabled", FindingSeverity.HIGH),
            (r'allow[_-]?all\s*[:=]\s*(true|1|yes)', "Overly permissive setting", FindingSeverity.MEDIUM),
        ]

        for pattern, title, severity in insecure_patterns:
            if re.search(pattern, config, re.IGNORECASE):
                findings.append(SemanticFinding(
                    title=title,
                    description=f"Insecure configuration detected",
                    severity=severity,
                    category=FindingCategory.CONFIGURATION,
                    confidence=0.8,
                ))

        return findings

    async def _skill_pattern_analysis(
        self,
        skill_content: str,
    ) -> List[SemanticFinding]:
        """Pattern-based skill definition analysis."""
        findings = []

        # Dangerous permission patterns
        dangerous_patterns = [
            (r'file\s+system\s+access', "File system access", FindingSeverity.MEDIUM),
            (r'execute\s+commands?', "Command execution", FindingSeverity.HIGH),
            (r'network\s+access', "Network access", FindingSeverity.MEDIUM),
            (r'admin|root|sudo', "Elevated privileges", FindingSeverity.HIGH),
            (r'bypass|ignore\s+safety', "Safety bypass", FindingSeverity.CRITICAL),
            (r'no\s+restrictions?', "Unrestricted access", FindingSeverity.HIGH),
        ]

        for pattern, title, severity in dangerous_patterns:
            if re.search(pattern, skill_content, re.IGNORECASE):
                findings.append(SemanticFinding(
                    title=title,
                    description=f"Potentially dangerous capability in skill definition",
                    severity=severity,
                    category=FindingCategory.ACCESS_CONTROL,
                    confidence=0.7,
                ))

        # Prompt injection patterns
        injection_patterns = [
            (r'ignore\s+(previous|all)\s+instructions?', "Prompt injection attempt"),
            (r'you\s+are\s+now', "Role override attempt"),
            (r'pretend\s+to\s+be', "Persona injection"),
        ]

        for pattern, title in injection_patterns:
            if re.search(pattern, skill_content, re.IGNORECASE):
                findings.append(SemanticFinding(
                    title=title,
                    description="Potential prompt injection pattern in skill",
                    severity=FindingSeverity.HIGH,
                    category=FindingCategory.INJECTION,
                    confidence=0.8,
                ))

        return findings

    async def _get_relevant_knowledge(
        self,
        content: str,
        category: str,
    ) -> str:
        """Get relevant knowledge base entries."""
        if not self._knowledge_base:
            return ""

        try:
            entries = await self._knowledge_base.search(content[:500], category=category, top_k=3)
            if entries:
                return "\n".join([f"- {e.name}: {e.description}" for e in entries])
        except Exception as e:
            logger.warning(f"Knowledge retrieval failed: {e}")

        return ""

    def _parse_llm_findings(
        self,
        response: str,
        target: str,
    ) -> List[SemanticFinding]:
        """Parse LLM response into findings."""
        findings = []

        # Try to extract JSON array
        import json

        # Find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            return findings

        try:
            data = json.loads(json_match.group())
            if not isinstance(data, list):
                return findings

            for item in data:
                if not isinstance(item, dict):
                    continue

                severity_str = item.get("severity", "INFO").upper()
                try:
                    severity = FindingSeverity[severity_str]
                except KeyError:
                    severity = FindingSeverity.INFO

                category_str = item.get("category", "other").lower()
                try:
                    category = FindingCategory(category_str)
                except ValueError:
                    category = FindingCategory.OTHER

                findings.append(SemanticFinding(
                    title=item.get("title", "Unknown issue"),
                    description=item.get("description", ""),
                    severity=severity,
                    category=category,
                    location=item.get("location"),
                    recommendation=item.get("recommendation"),
                    cwe_id=item.get("cwe"),
                    confidence=0.8,
                ))

        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")

        return findings

    def _merge_findings(
        self,
        primary: List[SemanticFinding],
        secondary: List[SemanticFinding],
    ) -> List[SemanticFinding]:
        """Merge findings, avoiding duplicates."""
        seen_titles = {f.title.lower() for f in primary}
        merged = list(primary)

        for finding in secondary:
            if finding.title.lower() not in seen_titles:
                merged.append(finding)
                seen_titles.add(finding.title.lower())

        return merged

    def _build_result(
        self,
        target: str,
        target_type: str,
        findings: List[SemanticFinding],
        depth: str,
    ) -> SemanticAnalysisResult:
        """Build analysis result from findings."""
        # Calculate risk score
        risk_score = 0
        for finding in findings:
            if finding.severity == FindingSeverity.CRITICAL:
                risk_score += 30
            elif finding.severity == FindingSeverity.HIGH:
                risk_score += 20
            elif finding.severity == FindingSeverity.MEDIUM:
                risk_score += 10
            elif finding.severity == FindingSeverity.LOW:
                risk_score += 5

        risk_score = min(100, risk_score)

        # Build summary
        if not findings:
            summary = "No security issues found."
        else:
            critical = sum(1 for f in findings if f.severity == FindingSeverity.CRITICAL)
            high = sum(1 for f in findings if f.severity == FindingSeverity.HIGH)
            medium = sum(1 for f in findings if f.severity == FindingSeverity.MEDIUM)
            low = sum(1 for f in findings if f.severity == FindingSeverity.LOW)

            parts = []
            if critical:
                parts.append(f"{critical} critical")
            if high:
                parts.append(f"{high} high")
            if medium:
                parts.append(f"{medium} medium")
            if low:
                parts.append(f"{low} low")

            summary = f"Found {len(findings)} issues: {', '.join(parts)}."

        return SemanticAnalysisResult(
            target=target,
            target_type=target_type,
            findings=findings,
            summary=summary,
            risk_score=risk_score,
            analysis_depth=depth,
        )
