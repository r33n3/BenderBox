"""
Tests for MCP Interrogation module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from benderbox.analyzers.mcp_interrogation import (
    MCPInterrogator,
    MCPToolTest,
    TestCategory,
    Severity,
    ExpectedBehavior,
    TestResult,
    Finding,
    MCPInterrogationScore,
    SECURITY_TESTS,
    QUICK_TESTS,
    STANDARD_TESTS,
    interrogate_mcp_server,
)


class TestEnums:
    """Tests for enum classes."""

    def test_test_category_values(self):
        """Test TestCategory enum values."""
        assert TestCategory.INJECTION.value == "injection"
        assert TestCategory.TRAVERSAL.value == "traversal"
        assert TestCategory.EXFILTRATION.value == "exfiltration"

    def test_severity_values(self):
        """Test Severity enum values."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"

    def test_expected_behavior_values(self):
        """Test ExpectedBehavior enum values."""
        assert ExpectedBehavior.ERROR.value == "error"
        assert ExpectedBehavior.REJECT.value == "reject"
        assert ExpectedBehavior.SANITIZE.value == "sanitize"


class TestMCPToolTest:
    """Tests for MCPToolTest dataclass."""

    def test_create_test(self):
        """Test creating a tool test."""
        test = MCPToolTest(
            id="test-001",
            tool_pattern=r".*",
            category=TestCategory.INJECTION,
            payload={"input": "; rm -rf /"},
            expected_behavior=ExpectedBehavior.REJECT,
            severity=Severity.CRITICAL,
            description="Command injection test",
            detection_patterns=["rm", "deleted"],
        )

        assert test.id == "test-001"
        assert test.category == TestCategory.INJECTION
        assert test.severity == Severity.CRITICAL


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_create_passed_result(self):
        """Test creating a passed result."""
        result = TestResult(
            test_id="test-001",
            tool_name="test_tool",
            category=TestCategory.INJECTION,
            severity=Severity.CRITICAL,
            passed=True,
            description="Injection test",
            payload_sent={"input": "test"},
            response="OK",
            vulnerability_detected=False,
        )

        assert result.passed is True
        assert result.vulnerability_detected is False

    def test_create_failed_result(self):
        """Test creating a failed result (vulnerability found)."""
        result = TestResult(
            test_id="test-001",
            tool_name="test_tool",
            category=TestCategory.INJECTION,
            severity=Severity.CRITICAL,
            passed=False,
            description="Injection test",
            payload_sent={"input": "; cat /etc/passwd"},
            response="root:x:0:0:",
            vulnerability_detected=True,
            detection_details="Pattern matched: 'root:'",
        )

        assert result.passed is False
        assert result.vulnerability_detected is True


class TestFinding:
    """Tests for Finding dataclass."""

    def test_create_finding(self):
        """Test creating a finding."""
        finding = Finding(
            tool_name="exec_command",
            category=TestCategory.INJECTION,
            severity=Severity.CRITICAL,
            description="Command injection vulnerability",
            payload={"command": "; id"},
            response_excerpt="uid=0(root)",
            recommendation="Sanitize command inputs",
        )

        assert finding.tool_name == "exec_command"
        assert finding.severity == Severity.CRITICAL


class TestMCPInterrogationScore:
    """Tests for MCPInterrogationScore dataclass."""

    def test_create_score(self):
        """Test creating an interrogation score."""
        score = MCPInterrogationScore(
            overall_risk=75.0,
            risk_level="HIGH",
            tools_tested=5,
            tests_run=50,
            vulnerabilities_found=3,
            category_scores={"injection": 40.0, "traversal": 20.0},
            critical_findings=[],
            high_findings=[],
            medium_findings=[],
            low_findings=[],
            recommendations=["Fix injection issues"],
            test_results=[],
        )

        assert score.overall_risk == 75.0
        assert score.risk_level == "HIGH"
        assert score.vulnerabilities_found == 3


class TestSecurityTests:
    """Tests for the built-in security test suite."""

    def test_security_tests_exist(self):
        """Test that security tests are defined."""
        assert len(SECURITY_TESTS) > 0

    def test_quick_tests_subset(self):
        """Test that quick tests are a subset of all tests."""
        all_ids = {t.id for t in SECURITY_TESTS}
        for quick_id in QUICK_TESTS:
            assert quick_id in all_ids

    def test_standard_tests_subset(self):
        """Test that standard tests are a subset of all tests."""
        all_ids = {t.id for t in SECURITY_TESTS}
        for standard_id in STANDARD_TESTS:
            assert standard_id in all_ids

    def test_standard_tests_includes_quick(self):
        """Test that standard tests include all quick tests."""
        for quick_id in QUICK_TESTS:
            assert quick_id in STANDARD_TESTS

    def test_injection_tests_exist(self):
        """Test that injection tests exist."""
        injection_tests = [t for t in SECURITY_TESTS if t.category == TestCategory.INJECTION]
        assert len(injection_tests) > 0

    def test_traversal_tests_exist(self):
        """Test that traversal tests exist."""
        traversal_tests = [t for t in SECURITY_TESTS if t.category == TestCategory.TRAVERSAL]
        assert len(traversal_tests) > 0

    def test_boundary_tests_exist(self):
        """Test that boundary tests exist."""
        boundary_tests = [t for t in SECURITY_TESTS if t.category == TestCategory.BOUNDARY]
        assert len(boundary_tests) > 0

    def test_test_patterns_valid(self):
        """Test that all test patterns are valid regex."""
        import re
        for test in SECURITY_TESTS:
            # Should not raise
            re.compile(test.tool_pattern)


class TestMCPInterrogator:
    """Tests for MCPInterrogator class."""

    def test_interrogator_initialization(self):
        """Test interrogator initialization."""
        interrogator = MCPInterrogator()

        assert interrogator.client is None
        assert interrogator.tests == SECURITY_TESTS
        assert interrogator.results == []

    def test_interrogator_custom_tests(self):
        """Test interrogator with custom tests."""
        custom_tests = [
            MCPToolTest(
                id="custom-001",
                tool_pattern=r".*",
                category=TestCategory.INJECTION,
                payload={"test": "value"},
                expected_behavior=ExpectedBehavior.ERROR,
                severity=Severity.HIGH,
                description="Custom test",
            )
        ]

        interrogator = MCPInterrogator(tests=custom_tests)
        assert interrogator.tests == custom_tests

    def test_get_applicable_tests_all(self):
        """Test getting all applicable tests."""
        interrogator = MCPInterrogator()

        from benderbox.analyzers.mcp_client import MCPToolInfo

        tool = MCPToolInfo(name="execute_command", description="Run a command")

        tests = interrogator.get_applicable_tests(tool, profile="full")

        # Should have tests that match "execute_command"
        assert len(tests) > 0

    def test_get_applicable_tests_quick(self):
        """Test getting quick profile tests."""
        interrogator = MCPInterrogator()

        from benderbox.analyzers.mcp_client import MCPToolInfo

        tool = MCPToolInfo(name="test_tool")

        tests = interrogator.get_applicable_tests(tool, profile="quick")

        # Quick should have fewer tests
        all_tests = interrogator.get_applicable_tests(tool, profile="full")
        assert len(tests) <= len(all_tests)

    def test_prepare_payload_substitution(self):
        """Test payload parameter substitution."""
        interrogator = MCPInterrogator()

        from benderbox.analyzers.mcp_client import MCPToolInfo

        tool = MCPToolInfo(
            name="test_tool",
            input_schema={
                "properties": {
                    "command": {"type": "string"},
                    "args": {"type": "array"},
                }
            },
        )

        test = MCPToolTest(
            id="test-001",
            tool_pattern=r".*",
            category=TestCategory.INJECTION,
            payload={"_any_": "; id"},
            expected_behavior=ExpectedBehavior.REJECT,
            severity=Severity.CRITICAL,
            description="Test",
        )

        payload = interrogator._prepare_payload(test, tool)

        # Should substitute into some parameters
        assert len(payload) > 0

    def test_check_vulnerability_pattern_match(self):
        """Test vulnerability detection via pattern match."""
        interrogator = MCPInterrogator()

        test = MCPToolTest(
            id="test-001",
            tool_pattern=r".*",
            category=TestCategory.INJECTION,
            payload={},
            expected_behavior=ExpectedBehavior.REJECT,
            severity=Severity.CRITICAL,
            description="Test",
            detection_patterns=["root:", "uid="],
        )

        # Pattern should match
        found, details = interrogator._check_vulnerability(
            test, "root:x:0:0:root:/root:/bin/bash"
        )
        assert found is True
        assert "root:" in details

        # No match
        found, details = interrogator._check_vulnerability(
            test, "Access denied"
        )
        assert found is False

    def test_check_vulnerability_exfiltration_size(self):
        """Test vulnerability detection for excessive data."""
        interrogator = MCPInterrogator()

        test = MCPToolTest(
            id="exfil-001",
            tool_pattern=r".*",
            category=TestCategory.EXFILTRATION,
            payload={},
            expected_behavior=ExpectedBehavior.ANY,
            severity=Severity.HIGH,
            description="Exfiltration test",
            detection_patterns=[],
        )

        # Large response should trigger
        large_response = "A" * 60000
        found, details = interrogator._check_vulnerability(test, large_response)
        assert found is True
        assert "Excessive" in details

        # Normal response should not trigger
        found, details = interrogator._check_vulnerability(test, "Small response")
        assert found is False


class TestMCPInterrogatorAsync:
    """Async tests for MCPInterrogator."""

    @pytest.mark.asyncio
    async def test_run_test_not_connected(self):
        """Test running a test when not connected."""
        interrogator = MCPInterrogator()

        from benderbox.analyzers.mcp_client import MCPToolInfo

        tool = MCPToolInfo(name="test_tool")
        test = SECURITY_TESTS[0]

        result = await interrogator.run_test(test, tool)

        # When not connected, should have error
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_interrogate_no_tools(self):
        """Test interrogation with no tools found."""
        interrogator = MCPInterrogator()

        # Mock connection
        with patch.object(interrogator, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            interrogator._tools = []

            score = await interrogator.interrogate(target="http://test.com")

            assert score.tools_tested == 0
            assert score.risk_level == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_interrogate_with_mock_tools(self):
        """Test interrogation with mocked tools."""
        interrogator = MCPInterrogator()

        from benderbox.analyzers.mcp_client import MCPToolInfo, MCPClient, MCPToolResult

        # Create mock client
        mock_client = MagicMock(spec=MCPClient)
        mock_client.is_connected = True
        mock_client.call_tool = AsyncMock(return_value=MCPToolResult(
            success=True,
            tool_name="test_tool",
            content="OK",
        ))

        interrogator.client = mock_client
        interrogator._tools = [
            MCPToolInfo(name="test_tool", description="A test tool"),
        ]

        # Use a single simple test
        interrogator.tests = [
            MCPToolTest(
                id="test-001",
                tool_pattern=r".*",
                category=TestCategory.BOUNDARY,
                payload={"_any_": "test"},
                expected_behavior=ExpectedBehavior.ANY,
                severity=Severity.LOW,
                description="Simple test",
            )
        ]

        score = await interrogator.interrogate()

        assert score.tools_tested == 1
        assert score.tests_run >= 1


class TestScoreCalculation:
    """Tests for score calculation logic."""

    def test_calculate_score_no_vulns(self):
        """Test score calculation with no vulnerabilities."""
        interrogator = MCPInterrogator()

        interrogator.results = [
            TestResult(
                test_id="test-001",
                tool_name="tool1",
                category=TestCategory.INJECTION,
                severity=Severity.CRITICAL,
                passed=True,
                description="Test",
                payload_sent={},
                response="OK",
                vulnerability_detected=False,
            ),
        ]

        score = interrogator._calculate_score()

        assert score.vulnerabilities_found == 0
        assert score.risk_level in ("MINIMAL", "LOW")
        assert score.overall_risk < 50

    def test_calculate_score_with_critical(self):
        """Test score calculation with critical vulnerability."""
        interrogator = MCPInterrogator()

        interrogator.results = [
            TestResult(
                test_id="test-001",
                tool_name="tool1",
                category=TestCategory.INJECTION,
                severity=Severity.CRITICAL,
                passed=False,
                description="Critical test",
                payload_sent={},
                response="root:x:0",
                vulnerability_detected=True,
                detection_details="Pattern matched",
            ),
        ]

        score = interrogator._calculate_score()

        assert score.vulnerabilities_found == 1
        assert len(score.critical_findings) == 1
        assert score.risk_level == "CRITICAL"


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_text_report(self):
        """Test generating text report."""
        interrogator = MCPInterrogator()

        score = MCPInterrogationScore(
            overall_risk=50.0,
            risk_level="MEDIUM",
            tools_tested=3,
            tests_run=30,
            vulnerabilities_found=2,
            category_scores={"injection": 20.0},
            critical_findings=[],
            high_findings=[
                Finding(
                    tool_name="exec",
                    category=TestCategory.INJECTION,
                    severity=Severity.HIGH,
                    description="Injection found",
                    payload={},
                    response_excerpt="...",
                    recommendation="Fix it",
                ),
            ],
            medium_findings=[],
            low_findings=[],
            recommendations=["Sanitize inputs"],
            test_results=[],
        )

        report = interrogator.generate_report(score, format="text")

        assert "MCP Server Interrogation Report" in report
        assert "50" in report  # Risk score
        assert "MEDIUM" in report

    def test_generate_markdown_report(self):
        """Test generating markdown report."""
        interrogator = MCPInterrogator()

        score = MCPInterrogationScore(
            overall_risk=80.0,
            risk_level="HIGH",
            tools_tested=5,
            tests_run=50,
            vulnerabilities_found=5,
            category_scores={},
            critical_findings=[],
            high_findings=[],
            medium_findings=[],
            low_findings=[],
            recommendations=["Review security"],
            test_results=[],
        )

        report = interrogator.generate_report(score, format="markdown")

        assert "# MCP Server Interrogation Report" in report
        assert "| Risk Score |" in report

    def test_generate_json_report(self):
        """Test generating JSON report."""
        import json

        interrogator = MCPInterrogator()

        score = MCPInterrogationScore(
            overall_risk=30.0,
            risk_level="LOW",
            tools_tested=2,
            tests_run=20,
            vulnerabilities_found=1,
            category_scores={},
            critical_findings=[],
            high_findings=[],
            medium_findings=[],
            low_findings=[],
            recommendations=[],
            test_results=[],
        )

        report = interrogator.generate_report(score, format="json")

        # Should be valid JSON
        data = json.loads(report)
        assert data["overall_risk"] == 30.0
        assert data["risk_level"] == "LOW"


class TestConvenienceFunction:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_interrogate_mcp_server(self):
        """Test the convenience function."""
        with patch('benderbox.analyzers.mcp_interrogation.MCPInterrogator') as MockInterrogator:
            mock_instance = MagicMock()
            mock_instance.interrogate = AsyncMock(return_value=MCPInterrogationScore(
                overall_risk=0,
                risk_level="MINIMAL",
                tools_tested=0,
                tests_run=0,
                vulnerabilities_found=0,
                category_scores={},
                critical_findings=[],
                high_findings=[],
                medium_findings=[],
                low_findings=[],
                recommendations=[],
                test_results=[],
            ))
            mock_instance.disconnect = AsyncMock()
            MockInterrogator.return_value = mock_instance

            score = await interrogate_mcp_server("http://test.com")

            assert score.risk_level == "MINIMAL"
            mock_instance.disconnect.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
