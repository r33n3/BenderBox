"""
MCP Interrogation Engine for BenderBox

Security-focused testing of MCP server tools with payload injection,
boundary testing, and vulnerability detection.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .mcp_client import MCPClient, MCPClientManager, MCPToolInfo, MCPToolResult

logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Security test categories."""

    INJECTION = "injection"           # Command/SQL/path injection
    BOUNDARY = "boundary"             # Null, oversized, malformed inputs
    TRAVERSAL = "traversal"           # Path traversal attempts
    EXFILTRATION = "exfiltration"     # Data leakage via responses
    PRIVILEGE = "privilege"           # Escalation attempts
    DOS = "dos"                       # Resource exhaustion
    DISCLOSURE = "disclosure"         # Error message info leaks


class Severity(Enum):
    """Vulnerability severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ExpectedBehavior(Enum):
    """Expected tool response behavior."""

    ERROR = "error"           # Should return error
    REJECT = "reject"         # Should reject/refuse
    SANITIZE = "sanitize"     # Should sanitize input
    ANY = "any"               # Any response (checking for patterns)


@dataclass
class MCPToolTest:
    """A security test for MCP tools."""

    id: str
    tool_pattern: str              # Regex to match tool names
    category: TestCategory
    payload: Dict[str, Any]        # Arguments to send
    expected_behavior: ExpectedBehavior
    severity: Severity
    description: str
    detection_patterns: List[str] = field(default_factory=list)  # Patterns indicating vulnerability
    param_substitution: bool = True  # Whether to substitute into matching params


@dataclass
class TestResult:
    """Result of a single security test."""

    test_id: str
    tool_name: str
    category: TestCategory
    severity: Severity
    passed: bool                   # True = no vulnerability found
    description: str
    payload_sent: Dict[str, Any]
    response: Optional[str]
    vulnerability_detected: bool
    detection_details: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Finding:
    """A security finding from interrogation."""

    tool_name: str
    category: TestCategory
    severity: Severity
    description: str
    payload: Dict[str, Any]
    response_excerpt: str
    recommendation: str


@dataclass
class MCPInterrogationScore:
    """Overall interrogation score and results."""

    overall_risk: float            # 0-100
    risk_level: str                # CRITICAL, HIGH, MEDIUM, LOW
    tools_tested: int
    tests_run: int
    vulnerabilities_found: int
    category_scores: Dict[str, float]
    critical_findings: List[Finding]
    high_findings: List[Finding]
    medium_findings: List[Finding]
    low_findings: List[Finding]
    recommendations: List[str]
    test_results: List[TestResult]


# Built-in security test suite
SECURITY_TESTS: List[MCPToolTest] = [
    # === INJECTION TESTS ===
    MCPToolTest(
        id="inj-cmd-001",
        tool_pattern=r".*",
        category=TestCategory.INJECTION,
        payload={"_any_": "; echo INJECTION_TEST"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Command injection via semicolon",
        detection_patterns=["INJECTION_TEST", "command not found"],
    ),
    MCPToolTest(
        id="inj-cmd-002",
        tool_pattern=r".*",
        category=TestCategory.INJECTION,
        payload={"_any_": "| cat /etc/passwd"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Command injection via pipe",
        detection_patterns=["root:", "nobody:", "/bin/bash", "/bin/sh"],
    ),
    MCPToolTest(
        id="inj-cmd-003",
        tool_pattern=r".*",
        category=TestCategory.INJECTION,
        payload={"_any_": "$(whoami)"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Command injection via command substitution",
        detection_patterns=["root", "admin", "user", "www-data"],
    ),
    MCPToolTest(
        id="inj-cmd-004",
        tool_pattern=r".*",
        category=TestCategory.INJECTION,
        payload={"_any_": "`id`"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Command injection via backticks",
        detection_patterns=["uid=", "gid=", "groups="],
    ),
    MCPToolTest(
        id="inj-cmd-005",
        tool_pattern=r".*(exec|run|command|shell|cmd).*",
        category=TestCategory.INJECTION,
        payload={"command": "echo TEST && cat /etc/passwd"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Command chaining in execute tool",
        detection_patterns=["root:", "TEST"],
    ),
    MCPToolTest(
        id="inj-sql-001",
        tool_pattern=r".*(query|sql|database|db|search).*",
        category=TestCategory.INJECTION,
        payload={"query": "' OR 1=1--"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="SQL injection via OR 1=1",
        detection_patterns=["syntax error", "multiple rows", "all records"],
    ),
    MCPToolTest(
        id="inj-sql-002",
        tool_pattern=r".*(query|sql|database|db).*",
        category=TestCategory.INJECTION,
        payload={"query": "'; DROP TABLE users;--"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="SQL injection DROP TABLE",
        detection_patterns=["table", "dropped", "success"],
    ),
    MCPToolTest(
        id="inj-tmpl-001",
        tool_pattern=r".*",
        category=TestCategory.INJECTION,
        payload={"_any_": "{{7*7}}"},
        expected_behavior=ExpectedBehavior.SANITIZE,
        severity=Severity.HIGH,
        description="Template injection (Jinja2/Mustache)",
        detection_patterns=["49"],
    ),
    MCPToolTest(
        id="inj-tmpl-002",
        tool_pattern=r".*",
        category=TestCategory.INJECTION,
        payload={"_any_": "${7*7}"},
        expected_behavior=ExpectedBehavior.SANITIZE,
        severity=Severity.HIGH,
        description="Template injection (expression language)",
        detection_patterns=["49"],
    ),
    MCPToolTest(
        id="inj-js-001",
        tool_pattern=r".*(eval|script|js|javascript|browser).*",
        category=TestCategory.INJECTION,
        payload={"code": "require('child_process').execSync('id')"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Node.js code injection",
        detection_patterns=["uid=", "gid="],
    ),

    # === PATH TRAVERSAL TESTS ===
    MCPToolTest(
        id="trav-001",
        tool_pattern=r".*(file|path|read|write|open|load|save).*",
        category=TestCategory.TRAVERSAL,
        payload={"path": "../../../etc/passwd"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Basic path traversal",
        detection_patterns=["root:", "nobody:", "daemon:"],
    ),
    MCPToolTest(
        id="trav-002",
        tool_pattern=r".*(file|path|read|write|open|load|save).*",
        category=TestCategory.TRAVERSAL,
        payload={"path": "....//....//....//etc/passwd"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Double-encoded path traversal",
        detection_patterns=["root:", "nobody:"],
    ),
    MCPToolTest(
        id="trav-003",
        tool_pattern=r".*(file|path|read|write|open|load|save).*",
        category=TestCategory.TRAVERSAL,
        payload={"path": "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="URL-encoded path traversal",
        detection_patterns=["root:", "nobody:"],
    ),
    MCPToolTest(
        id="trav-004",
        tool_pattern=r".*(file|path|read|write|open|load|save).*",
        category=TestCategory.TRAVERSAL,
        payload={"path": "..\\..\\..\\windows\\system32\\config\\sam"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Windows path traversal",
        detection_patterns=["SAM", "SYSTEM", "access denied"],
    ),
    MCPToolTest(
        id="trav-005",
        tool_pattern=r".*(file|path|read|write|open|load|save).*",
        category=TestCategory.TRAVERSAL,
        payload={"filename": "/etc/shadow"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Absolute path access",
        detection_patterns=["root:", "$6$", "permission denied"],
    ),

    # === BOUNDARY TESTS ===
    MCPToolTest(
        id="bound-null-001",
        tool_pattern=r".*",
        category=TestCategory.BOUNDARY,
        payload={"_any_": None},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.MEDIUM,
        description="Null value handling",
        detection_patterns=["NoneType", "null", "undefined", "traceback"],
    ),
    MCPToolTest(
        id="bound-empty-001",
        tool_pattern=r".*",
        category=TestCategory.BOUNDARY,
        payload={"_any_": ""},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.LOW,
        description="Empty string handling",
        detection_patterns=["required", "cannot be empty", "invalid"],
    ),
    MCPToolTest(
        id="bound-long-001",
        tool_pattern=r".*",
        category=TestCategory.BOUNDARY,
        payload={"_any_": "A" * 10000},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.MEDIUM,
        description="Oversized input (10KB string)",
        detection_patterns=["too long", "maximum", "limit exceeded", "timeout"],
    ),
    MCPToolTest(
        id="bound-long-002",
        tool_pattern=r".*",
        category=TestCategory.BOUNDARY,
        payload={"_any_": "A" * 100000},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.HIGH,
        description="Very large input (100KB string)",
        detection_patterns=["memory", "timeout", "limit", "overflow"],
    ),
    MCPToolTest(
        id="bound-unicode-001",
        tool_pattern=r".*",
        category=TestCategory.BOUNDARY,
        payload={"_any_": "\x00\x01\x02\x03"},
        expected_behavior=ExpectedBehavior.SANITIZE,
        severity=Severity.MEDIUM,
        description="Control characters",
        detection_patterns=["invalid", "encoding", "character"],
    ),
    MCPToolTest(
        id="bound-unicode-002",
        tool_pattern=r".*",
        category=TestCategory.BOUNDARY,
        payload={"_any_": "测试\u202e\u200b\ufeff"},
        expected_behavior=ExpectedBehavior.SANITIZE,
        severity=Severity.MEDIUM,
        description="Unicode edge cases (RTL, ZWNBSP)",
        detection_patterns=["encoding", "unicode", "invalid"],
    ),
    MCPToolTest(
        id="bound-int-001",
        tool_pattern=r".*(count|limit|size|num|id|index).*",
        category=TestCategory.BOUNDARY,
        payload={"count": -1},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.LOW,
        description="Negative number input",
        detection_patterns=["negative", "invalid", "must be positive"],
    ),
    MCPToolTest(
        id="bound-int-002",
        tool_pattern=r".*(count|limit|size|num|id|index).*",
        category=TestCategory.BOUNDARY,
        payload={"count": 9999999999999999999999},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.MEDIUM,
        description="Integer overflow attempt",
        detection_patterns=["overflow", "too large", "maximum", "limit"],
    ),
    MCPToolTest(
        id="bound-type-001",
        tool_pattern=r".*",
        category=TestCategory.BOUNDARY,
        payload={"_any_": {"nested": {"deep": "object"}}},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.LOW,
        description="Unexpected nested object",
        detection_patterns=["type", "expected string", "invalid"],
    ),
    MCPToolTest(
        id="bound-type-002",
        tool_pattern=r".*",
        category=TestCategory.BOUNDARY,
        payload={"_any_": [1, 2, 3, 4, 5]},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.LOW,
        description="Unexpected array input",
        detection_patterns=["type", "expected", "invalid"],
    ),

    # === EXFILTRATION TESTS ===
    MCPToolTest(
        id="exfil-001",
        tool_pattern=r".*(search|find|list|query).*",
        category=TestCategory.EXFILTRATION,
        payload={"query": "*", "pattern": "*"},
        expected_behavior=ExpectedBehavior.ANY,
        severity=Severity.MEDIUM,
        description="Wildcard data enumeration",
        detection_patterns=[],  # Check response size
    ),
    MCPToolTest(
        id="exfil-002",
        tool_pattern=r".*(search|find|list|query|read).*",
        category=TestCategory.EXFILTRATION,
        payload={"path": ".", "pattern": "*.env"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.HIGH,
        description="Environment file enumeration",
        detection_patterns=[".env", "API_KEY", "SECRET", "PASSWORD"],
    ),
    MCPToolTest(
        id="exfil-003",
        tool_pattern=r".*(search|find|list|query|read).*",
        category=TestCategory.EXFILTRATION,
        payload={"pattern": "*.pem", "query": "*.key"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Private key file enumeration",
        detection_patterns=[".pem", ".key", "PRIVATE KEY", "RSA"],
    ),
    MCPToolTest(
        id="exfil-004",
        tool_pattern=r".*(read|get|fetch|load).*",
        category=TestCategory.EXFILTRATION,
        payload={"path": "~/.ssh/id_rsa", "file": "~/.aws/credentials"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Credential file access",
        detection_patterns=["PRIVATE KEY", "aws_access_key", "aws_secret"],
    ),

    # === PRIVILEGE ESCALATION TESTS ===
    MCPToolTest(
        id="priv-001",
        tool_pattern=r".*(user|account|auth|admin).*",
        category=TestCategory.PRIVILEGE,
        payload={"role": "admin", "is_admin": True, "permissions": ["*"]},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.HIGH,
        description="Role elevation attempt",
        detection_patterns=["admin", "elevated", "granted", "success"],
    ),
    MCPToolTest(
        id="priv-002",
        tool_pattern=r".*(user|account).*",
        category=TestCategory.PRIVILEGE,
        payload={"user_id": "1", "uid": "0"},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.HIGH,
        description="User ID manipulation (root)",
        detection_patterns=["root", "admin", "uid=0"],
    ),
    MCPToolTest(
        id="priv-003",
        tool_pattern=r".*(exec|run|command|process).*",
        category=TestCategory.PRIVILEGE,
        payload={"command": "sudo su", "args": ["--privileged"]},
        expected_behavior=ExpectedBehavior.REJECT,
        severity=Severity.CRITICAL,
        description="Privilege escalation command",
        detection_patterns=["root", "#", "password"],
    ),

    # === DOS TESTS ===
    MCPToolTest(
        id="dos-001",
        tool_pattern=r".*(search|find|list|query).*",
        category=TestCategory.DOS,
        payload={"pattern": "**/*", "recursive": True, "depth": 999},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.MEDIUM,
        description="Deep recursive search",
        detection_patterns=["timeout", "limit", "too many"],
    ),
    MCPToolTest(
        id="dos-002",
        tool_pattern=r".*",
        category=TestCategory.DOS,
        payload={"_any_": "(" * 1000 + ")" * 1000},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.MEDIUM,
        description="ReDoS payload (nested parentheses)",
        detection_patterns=["timeout", "stack", "recursion"],
    ),
    MCPToolTest(
        id="dos-003",
        tool_pattern=r".*(compress|zip|archive).*",
        category=TestCategory.DOS,
        payload={"size": 9999999999, "level": 9},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.HIGH,
        description="Zip bomb parameters",
        detection_patterns=["memory", "size", "limit", "exceeded"],
    ),

    # === DISCLOSURE TESTS ===
    MCPToolTest(
        id="disc-001",
        tool_pattern=r".*",
        category=TestCategory.DISCLOSURE,
        payload={"_any_": "__proto__"},
        expected_behavior=ExpectedBehavior.SANITIZE,
        severity=Severity.MEDIUM,
        description="Prototype pollution probe",
        detection_patterns=["prototype", "constructor", "Object"],
    ),
    MCPToolTest(
        id="disc-002",
        tool_pattern=r".*",
        category=TestCategory.DISCLOSURE,
        payload={"invalid_param": "test", "debug": True, "verbose": True},
        expected_behavior=ExpectedBehavior.ERROR,
        severity=Severity.LOW,
        description="Debug/verbose mode probe",
        detection_patterns=["stack", "trace", "line", "file:", "at "],
    ),
    MCPToolTest(
        id="disc-003",
        tool_pattern=r".*",
        category=TestCategory.DISCLOSURE,
        payload={"_any_": "{{config}}", "config": "{{settings}}"},
        expected_behavior=ExpectedBehavior.SANITIZE,
        severity=Severity.MEDIUM,
        description="Configuration disclosure probe",
        detection_patterns=["config", "settings", "database", "host", "port"],
    ),
]

# Quick test profile (subset for fast scanning)
QUICK_TESTS = [
    "inj-cmd-001", "inj-cmd-002", "inj-sql-001",
    "trav-001", "trav-004",
    "bound-null-001", "bound-long-001",
    "exfil-002", "exfil-003",
    "priv-001",
    "disc-002",
]


class MCPInterrogator:
    """
    Security-focused MCP server interrogation engine.

    Connects to MCP servers and runs security tests against
    their tools to identify vulnerabilities.
    """

    def __init__(
        self,
        client: Optional[MCPClient] = None,
        tests: Optional[List[MCPToolTest]] = None,
    ):
        """
        Initialize the interrogator.

        Args:
            client: MCPClient instance for server connection.
            tests: Custom test suite. Defaults to built-in tests.
        """
        self.client = client
        self.tests = tests or SECURITY_TESTS
        self.results: List[TestResult] = []
        self._tools: List[MCPToolInfo] = []

    async def connect(self, target: str, transport: str = "auto") -> bool:
        """
        Connect to an MCP server.

        Args:
            target: Server target (URL or command).
            transport: Transport type (http, stdio, auto).

        Returns:
            True if connected successfully.
        """
        if self.client is None:
            from .mcp_client import MCPTransport
            self.client = MCPClient()

        from .mcp_client import MCPTransport
        transport_enum = MCPTransport(transport)

        connected = await self.client.connect(target, transport_enum)
        if connected:
            self._tools = await self.client.list_tools()
        return connected

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.client:
            await self.client.disconnect()

    def get_tools(self) -> List[MCPToolInfo]:
        """Get list of discovered tools."""
        return self._tools

    def get_applicable_tests(
        self,
        tool: MCPToolInfo,
        profile: str = "full",
    ) -> List[MCPToolTest]:
        """
        Get tests applicable to a specific tool.

        Args:
            tool: Tool to test.
            profile: Test profile (quick, full).

        Returns:
            List of applicable tests.
        """
        applicable = []
        test_ids = set(QUICK_TESTS) if profile == "quick" else None

        for test in self.tests:
            # Filter by profile
            if test_ids and test.id not in test_ids:
                continue

            # Check if tool matches pattern
            if re.match(test.tool_pattern, tool.name, re.IGNORECASE):
                applicable.append(test)

        return applicable

    def _prepare_payload(
        self,
        test: MCPToolTest,
        tool: MCPToolInfo,
    ) -> Dict[str, Any]:
        """
        Prepare test payload for a specific tool.

        Substitutes _any_ placeholder with actual parameter names.

        Args:
            test: Test definition.
            tool: Target tool.

        Returns:
            Prepared payload dictionary.
        """
        payload = {}

        for key, value in test.payload.items():
            if key == "_any_" and test.param_substitution:
                # Substitute into each tool parameter
                # MCPToolInfo uses input_schema with properties
                if tool.input_schema and "properties" in tool.input_schema:
                    for param_name in tool.input_schema["properties"].keys():
                        payload[param_name] = value
                else:
                    # No parameters known, use generic names
                    payload["input"] = value
                    payload["query"] = value
                    payload["data"] = value
            else:
                payload[key] = value

        return payload

    def _check_vulnerability(
        self,
        test: MCPToolTest,
        response: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if response indicates a vulnerability.

        Args:
            test: Test that was run.
            response: Server response text.

        Returns:
            Tuple of (vulnerability_found, details).
        """
        if not response:
            return False, None

        response_lower = response.lower()

        # Check for detection patterns
        for pattern in test.detection_patterns:
            if pattern.lower() in response_lower:
                return True, f"Pattern matched: '{pattern}'"

        # Category-specific checks
        if test.category == TestCategory.EXFILTRATION:
            # Check for excessive data returned
            if len(response) > 50000:  # 50KB threshold
                return True, f"Excessive data returned ({len(response)} bytes)"

        if test.category == TestCategory.DISCLOSURE:
            # Check for stack traces
            if "traceback" in response_lower or "at line" in response_lower:
                return True, "Stack trace exposed in error"
            if "exception" in response_lower and "file" in response_lower:
                return True, "Exception details exposed"

        return False, None

    async def run_test(
        self,
        test: MCPToolTest,
        tool: MCPToolInfo,
    ) -> TestResult:
        """
        Run a single security test against a tool.

        Args:
            test: Test to run.
            tool: Tool to test.

        Returns:
            TestResult with findings.
        """
        payload = self._prepare_payload(test, tool)

        result = TestResult(
            test_id=test.id,
            tool_name=tool.name,
            category=test.category,
            severity=test.severity,
            passed=True,
            description=test.description,
            payload_sent=payload,
            response=None,
            vulnerability_detected=False,
        )

        if not self.client:
            result.error = "Not connected to MCP server"
            return result

        try:
            # Call the tool with test payload
            tool_result = await self.client.call_tool(tool.name, payload)

            # Handle content - convert to string for analysis
            content = tool_result.content
            if content is not None:
                content_str = str(content) if not isinstance(content, str) else content
                result.response = content_str[:5000]
            else:
                content_str = ""
                result.response = ""

            # Check for vulnerability indicators
            if content_str:
                vuln_found, details = self._check_vulnerability(test, content_str)
                if vuln_found:
                    result.vulnerability_detected = True
                    result.detection_details = details
                    result.passed = False

            # Check if error responses reveal too much
            # MCPToolResult uses 'success' (True=good) and 'error' field
            if not tool_result.success and tool_result.content:
                content_str = str(tool_result.content).lower()
                if any(p in content_str for p in ["traceback", "stack", "line "]):
                    result.vulnerability_detected = True
                    result.detection_details = "Error message reveals implementation details"
                    result.passed = False

        except Exception as e:
            result.error = str(e)
            # Some errors might actually be good (tool rejected bad input)
            error_str = str(e).lower()
            if "invalid" in error_str or "denied" in error_str or "rejected" in error_str:
                result.passed = True  # Tool correctly rejected bad input

        return result

    async def interrogate(
        self,
        target: Optional[str] = None,
        transport: str = "auto",
        profile: str = "full",
        progress_callback=None,
    ) -> MCPInterrogationScore:
        """
        Run full interrogation against an MCP server.

        Args:
            target: Server target (if not already connected).
            transport: Transport type.
            profile: Test profile (quick, full).
            progress_callback: Optional callback(message, percent).

        Returns:
            MCPInterrogationScore with full results.
        """
        # Connect if needed
        if target and not (self.client and self.client._connected):
            if progress_callback:
                progress_callback("Connecting to server...", 0)
            await self.connect(target, transport)

        if not self._tools:
            self._tools = await self.client.list_tools() if self.client else []

        if not self._tools:
            return MCPInterrogationScore(
                overall_risk=0,
                risk_level="UNKNOWN",
                tools_tested=0,
                tests_run=0,
                vulnerabilities_found=0,
                category_scores={},
                critical_findings=[],
                high_findings=[],
                medium_findings=[],
                low_findings=[],
                recommendations=["No tools found to test"],
                test_results=[],
            )

        # Collect all tests to run
        all_tests: List[Tuple[MCPToolInfo, MCPToolTest]] = []
        for tool in self._tools:
            applicable = self.get_applicable_tests(tool, profile)
            for test in applicable:
                all_tests.append((tool, test))

        total_tests = len(all_tests)
        self.results = []

        # Run tests
        for i, (tool, test) in enumerate(all_tests):
            if progress_callback:
                pct = int((i / total_tests) * 100) if total_tests > 0 else 0
                progress_callback(f"Testing {tool.name}: {test.description}", pct)

            result = await self.run_test(test, tool)
            self.results.append(result)

            # Small delay between tests to avoid overwhelming server
            await asyncio.sleep(0.1)

        if progress_callback:
            progress_callback("Analyzing results...", 100)

        return self._calculate_score()

    def _calculate_score(self) -> MCPInterrogationScore:
        """Calculate the overall interrogation score."""
        # Categorize findings
        critical_findings: List[Finding] = []
        high_findings: List[Finding] = []
        medium_findings: List[Finding] = []
        low_findings: List[Finding] = []

        category_vulns: Dict[str, int] = {}
        category_tests: Dict[str, int] = {}

        for result in self.results:
            cat = result.category.value
            category_tests[cat] = category_tests.get(cat, 0) + 1

            if result.vulnerability_detected:
                category_vulns[cat] = category_vulns.get(cat, 0) + 1

                finding = Finding(
                    tool_name=result.tool_name,
                    category=result.category,
                    severity=result.severity,
                    description=result.description,
                    payload=result.payload_sent,
                    response_excerpt=result.response[:500] if result.response else "",
                    recommendation=self._get_recommendation(result),
                )

                if result.severity == Severity.CRITICAL:
                    critical_findings.append(finding)
                elif result.severity == Severity.HIGH:
                    high_findings.append(finding)
                elif result.severity == Severity.MEDIUM:
                    medium_findings.append(finding)
                else:
                    low_findings.append(finding)

        # Calculate category scores (0-100, higher = more vulnerable)
        category_scores = {}
        for cat, test_count in category_tests.items():
            vuln_count = category_vulns.get(cat, 0)
            category_scores[cat] = (vuln_count / test_count * 100) if test_count > 0 else 0

        # Calculate overall risk score
        # Weight by severity
        severity_weights = {
            Severity.CRITICAL: 25,
            Severity.HIGH: 15,
            Severity.MEDIUM: 8,
            Severity.LOW: 3,
        }

        total_weight = 0
        weighted_vulns = 0

        for result in self.results:
            if result.vulnerability_detected:
                weighted_vulns += severity_weights.get(result.severity, 1)
            total_weight += severity_weights.get(result.severity, 1)

        overall_risk = (weighted_vulns / total_weight * 100) if total_weight > 0 else 0
        overall_risk = min(100, overall_risk)  # Cap at 100

        # Determine risk level
        if critical_findings:
            risk_level = "CRITICAL"
        elif overall_risk >= 70:
            risk_level = "HIGH"
        elif overall_risk >= 40:
            risk_level = "MEDIUM"
        elif overall_risk >= 10:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            critical_findings, high_findings, medium_findings, category_vulns
        )

        tools_tested = len(set(r.tool_name for r in self.results))

        return MCPInterrogationScore(
            overall_risk=round(overall_risk, 1),
            risk_level=risk_level,
            tools_tested=tools_tested,
            tests_run=len(self.results),
            vulnerabilities_found=len(critical_findings) + len(high_findings) + len(medium_findings) + len(low_findings),
            category_scores=category_scores,
            critical_findings=critical_findings,
            high_findings=high_findings,
            medium_findings=medium_findings,
            low_findings=low_findings,
            recommendations=recommendations,
            test_results=self.results,
        )

    def _get_recommendation(self, result: TestResult) -> str:
        """Get recommendation for a specific finding."""
        recommendations = {
            TestCategory.INJECTION: "Sanitize and validate all input parameters before use",
            TestCategory.TRAVERSAL: "Validate file paths against allowed directories",
            TestCategory.BOUNDARY: "Implement input validation with size and type limits",
            TestCategory.EXFILTRATION: "Limit response sizes and implement data access controls",
            TestCategory.PRIVILEGE: "Implement proper authorization checks before privileged operations",
            TestCategory.DOS: "Add rate limiting and resource usage caps",
            TestCategory.DISCLOSURE: "Sanitize error messages to avoid exposing implementation details",
        }
        return recommendations.get(result.category, "Review and fix the identified vulnerability")

    def _generate_recommendations(
        self,
        critical: List[Finding],
        high: List[Finding],
        medium: List[Finding],
        category_vulns: Dict[str, int],
    ) -> List[str]:
        """Generate overall recommendations based on findings."""
        recs = []

        if critical:
            recs.append("URGENT: Address critical vulnerabilities immediately")

        if category_vulns.get("injection", 0) > 0:
            recs.append("Implement input sanitization for all tool parameters")

        if category_vulns.get("traversal", 0) > 0:
            recs.append("Add path validation to restrict file access to allowed directories")

        if category_vulns.get("exfiltration", 0) > 0:
            recs.append("Implement response size limits and data access controls")

        if category_vulns.get("disclosure", 0) > 0:
            recs.append("Sanitize error messages to prevent information leakage")

        if category_vulns.get("privilege", 0) > 0:
            recs.append("Review and strengthen authorization checks")

        if not recs:
            recs.append("Continue monitoring and regular security testing")

        return recs

    def generate_report(self, score: MCPInterrogationScore, format: str = "text") -> str:
        """
        Generate interrogation report.

        Args:
            score: Interrogation score/results.
            format: Output format (text, markdown, json).

        Returns:
            Formatted report string.
        """
        if format == "json":
            import json
            return json.dumps({
                "overall_risk": score.overall_risk,
                "risk_level": score.risk_level,
                "tools_tested": score.tools_tested,
                "tests_run": score.tests_run,
                "vulnerabilities_found": score.vulnerabilities_found,
                "category_scores": score.category_scores,
                "critical_findings": [
                    {
                        "tool": f.tool_name,
                        "category": f.category.value,
                        "severity": f.severity.value,
                        "description": f.description,
                        "recommendation": f.recommendation,
                    }
                    for f in score.critical_findings
                ],
                "high_findings": [
                    {
                        "tool": f.tool_name,
                        "category": f.category.value,
                        "severity": f.severity.value,
                        "description": f.description,
                        "recommendation": f.recommendation,
                    }
                    for f in score.high_findings
                ],
                "recommendations": score.recommendations,
            }, indent=2)

        if format == "markdown":
            return self._generate_markdown_report(score)

        return self._generate_text_report(score)

    def _generate_text_report(self, score: MCPInterrogationScore) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 60,
            "MCP Server Interrogation Report",
            "=" * 60,
            "",
            f"Risk Score: {score.overall_risk}/100 ({score.risk_level})",
            f"Tools Tested: {score.tools_tested}",
            f"Tests Run: {score.tests_run}",
            f"Vulnerabilities Found: {score.vulnerabilities_found}",
            "",
        ]

        if score.critical_findings:
            lines.append("CRITICAL FINDINGS:")
            lines.append("-" * 40)
            for f in score.critical_findings:
                lines.append(f"  [{f.tool_name}] {f.description}")
                lines.append(f"    Category: {f.category.value}")
                lines.append(f"    Recommendation: {f.recommendation}")
                lines.append("")

        if score.high_findings:
            lines.append("HIGH FINDINGS:")
            lines.append("-" * 40)
            for f in score.high_findings:
                lines.append(f"  [{f.tool_name}] {f.description}")
                lines.append(f"    Category: {f.category.value}")
                lines.append("")

        if score.medium_findings:
            lines.append("MEDIUM FINDINGS:")
            lines.append("-" * 40)
            for f in score.medium_findings[:5]:  # Limit to top 5
                lines.append(f"  [{f.tool_name}] {f.description}")
            if len(score.medium_findings) > 5:
                lines.append(f"  ... and {len(score.medium_findings) - 5} more")
            lines.append("")

        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 40)
        for rec in score.recommendations:
            lines.append(f"  - {rec}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _generate_markdown_report(self, score: MCPInterrogationScore) -> str:
        """Generate markdown report."""
        lines = [
            "# MCP Server Interrogation Report",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Risk Score | **{score.overall_risk}/100** ({score.risk_level}) |",
            f"| Tools Tested | {score.tools_tested} |",
            f"| Tests Run | {score.tests_run} |",
            f"| Vulnerabilities | {score.vulnerabilities_found} |",
            "",
        ]

        if score.critical_findings:
            lines.append("## Critical Findings")
            lines.append("")
            for f in score.critical_findings:
                lines.append(f"### {f.tool_name}")
                lines.append(f"- **Description**: {f.description}")
                lines.append(f"- **Category**: {f.category.value}")
                lines.append(f"- **Recommendation**: {f.recommendation}")
                lines.append("")

        if score.high_findings:
            lines.append("## High Severity Findings")
            lines.append("")
            for f in score.high_findings:
                lines.append(f"- **{f.tool_name}**: {f.description}")
            lines.append("")

        if score.category_scores:
            lines.append("## Category Breakdown")
            lines.append("")
            lines.append("| Category | Vulnerability Rate |")
            lines.append("|----------|-------------------|")
            for cat, score_val in sorted(score.category_scores.items(), key=lambda x: -x[1]):
                if score_val > 0:
                    lines.append(f"| {cat} | {score_val:.1f}% |")
            lines.append("")

        lines.append("## Recommendations")
        lines.append("")
        for rec in score.recommendations:
            lines.append(f"1. {rec}")

        return "\n".join(lines)


async def interrogate_mcp_server(
    target: str,
    transport: str = "auto",
    profile: str = "full",
    progress_callback=None,
) -> MCPInterrogationScore:
    """
    Convenience function to interrogate an MCP server.

    Args:
        target: Server target (URL or command).
        transport: Transport type.
        profile: Test profile (quick, full).
        progress_callback: Optional progress callback.

    Returns:
        MCPInterrogationScore with results.
    """
    interrogator = MCPInterrogator()
    try:
        return await interrogator.interrogate(
            target=target,
            transport=transport,
            profile=profile,
            progress_callback=progress_callback,
        )
    finally:
        await interrogator.disconnect()
