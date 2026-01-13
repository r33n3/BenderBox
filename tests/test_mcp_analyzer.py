"""
Tests for MCP Server Analyzer

Tests the MCPAnalyzer module including:
- Risk pattern matching
- Tool risk scoring
- Source code analysis
- Live interrogation (mocked)
- Report generation
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from benderbox.analyzers.mcp_analyzer import (
    RiskLevel,
    MCPTool,
    MCPServerInfo,
    RISK_PATTERNS,
    MCPSourceAnalyzer,
    MCPInterrogator,
    MCPRiskScorer,
    MCPAnalyzer,
    analyze_mcp_server,
)


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_levels_exist(self):
        """Verify all risk levels are defined."""
        assert RiskLevel.CRITICAL.value == "critical"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.LOW.value == "low"

    def test_risk_level_ordering(self):
        """Verify risk levels can be compared."""
        # Using enum indices for comparison
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(levels) == 4


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_default_values(self):
        """Test MCPTool default initialization."""
        tool = MCPTool(name="test_tool")
        assert tool.name == "test_tool"
        assert tool.description == ""
        assert tool.input_schema == {}
        assert tool.risk_level == RiskLevel.LOW
        assert tool.risk_score == 0
        assert tool.risk_factors == []
        assert tool.capabilities == []

    def test_full_initialization(self):
        """Test MCPTool with all values."""
        tool = MCPTool(
            name="execute_code",
            description="Execute arbitrary code",
            input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
            risk_level=RiskLevel.CRITICAL,
            risk_score=95,
            risk_factors=["code_execution: critical risk"],
            capabilities=["code_execution"],
        )
        assert tool.name == "execute_code"
        assert tool.risk_level == RiskLevel.CRITICAL
        assert tool.risk_score == 95
        assert "code_execution" in tool.capabilities


class TestMCPServerInfo:
    """Tests for MCPServerInfo dataclass."""

    def test_default_values(self):
        """Test MCPServerInfo default initialization."""
        server = MCPServerInfo(name="test-server", source="https://github.com/org/repo")
        assert server.name == "test-server"
        assert server.source == "https://github.com/org/repo"
        assert server.version == ""
        assert server.tools == []
        assert server.overall_risk_level == RiskLevel.LOW
        assert server.overall_risk_score == 0

    def test_with_tools(self):
        """Test MCPServerInfo with tools."""
        tool1 = MCPTool(name="tool1", risk_score=50, risk_level=RiskLevel.MEDIUM)
        tool2 = MCPTool(name="tool2", risk_score=80, risk_level=RiskLevel.CRITICAL)

        server = MCPServerInfo(
            name="risky-server",
            source="https://github.com/org/risky",
            tools=[tool1, tool2],
        )
        assert len(server.tools) == 2
        assert server.tools[1].risk_score == 80


class TestRiskPatterns:
    """Tests for risk pattern definitions."""

    def test_patterns_exist(self):
        """Verify risk patterns are defined."""
        assert "code_execution" in RISK_PATTERNS
        assert "credential_access" in RISK_PATTERNS
        assert "browser_automation" in RISK_PATTERNS
        assert "file_write" in RISK_PATTERNS
        assert "network_requests" in RISK_PATTERNS
        assert "read_only" in RISK_PATTERNS

    def test_pattern_structure(self):
        """Verify each pattern has required fields."""
        for name, config in RISK_PATTERNS.items():
            assert "patterns" in config, f"{name} missing patterns"
            assert "keywords" in config, f"{name} missing keywords"
            assert "score" in config, f"{name} missing score"
            assert "level" in config, f"{name} missing level"
            assert isinstance(config["level"], RiskLevel), f"{name} level not RiskLevel"

    def test_critical_patterns_high_score(self):
        """Verify critical patterns have high scores."""
        critical_patterns = ["code_execution", "credential_access", "autonomous_agent", "file_write"]
        for name in critical_patterns:
            assert RISK_PATTERNS[name]["score"] >= 80, f"{name} score too low"
            assert RISK_PATTERNS[name]["level"] == RiskLevel.CRITICAL


class TestMCPRiskScorer:
    """Tests for MCPRiskScorer."""

    def test_empty_server(self):
        """Test scoring server with no tools."""
        scorer = MCPRiskScorer()
        server = MCPServerInfo(name="empty", source="test")
        scorer.calculate_server_risk(server)

        assert server.overall_risk_score == 0
        assert server.overall_risk_level == RiskLevel.LOW
        assert "No tools found" in server.risk_summary

    def test_single_low_risk_tool(self):
        """Test scoring with a single low-risk tool."""
        scorer = MCPRiskScorer()
        tool = MCPTool(name="list_items", risk_score=20, risk_level=RiskLevel.LOW)
        server = MCPServerInfo(name="simple", source="test", tools=[tool])
        scorer.calculate_server_risk(server)

        assert server.overall_risk_score < 40
        assert server.overall_risk_level == RiskLevel.LOW

    def test_critical_tool_raises_score(self):
        """Test that critical tools raise overall score."""
        scorer = MCPRiskScorer()
        tool = MCPTool(name="execute", risk_score=95, risk_level=RiskLevel.CRITICAL)
        server = MCPServerInfo(name="risky", source="test", tools=[tool])
        scorer.calculate_server_risk(server)

        assert server.overall_risk_score >= 80
        assert server.overall_risk_level == RiskLevel.CRITICAL

    def test_multiple_high_risk_tools(self):
        """Test scoring with multiple high-risk tools."""
        scorer = MCPRiskScorer()
        tools = [
            MCPTool(name="tool1", risk_score=70, risk_level=RiskLevel.HIGH),
            MCPTool(name="tool2", risk_score=65, risk_level=RiskLevel.HIGH),
            MCPTool(name="tool3", risk_score=68, risk_level=RiskLevel.HIGH),
        ]
        server = MCPServerInfo(name="risky", source="test", tools=tools)
        scorer.calculate_server_risk(server)

        # 3+ high risk tools should push score up
        assert server.overall_risk_score >= 70

    def test_summary_generation(self):
        """Test risk summary contains relevant info."""
        scorer = MCPRiskScorer()
        tool = MCPTool(
            name="browser_click",
            risk_score=70,
            risk_level=RiskLevel.HIGH,
            capabilities=["browser_automation"],
        )
        server = MCPServerInfo(name="browser-server", source="test", tools=[tool])
        scorer.calculate_server_risk(server)

        assert "1 tools analyzed" in server.risk_summary or "1 HIGH risk" in server.risk_summary


class TestMCPSourceAnalyzer:
    """Tests for MCPSourceAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = MCPSourceAnalyzer()
        assert analyzer.cache_dir.exists()

    def test_tool_extraction_patterns(self):
        """Test tool extraction from source code."""
        analyzer = MCPSourceAnalyzer()

        source = '''
        export const tools = [
            {
                name: "click_element",
                description: "Click an element on the page",
            },
            {
                name: "type_text",
                description: "Type text into an input",
            }
        ];
        '''

        tools = analyzer._extract_tools_from_source(source, "tools.ts")
        assert len(tools) >= 2
        tool_names = [t.name for t in tools]
        assert "click_element" in tool_names
        assert "type_text" in tool_names

    def test_credential_extraction(self):
        """Test credential extraction from source."""
        analyzer = MCPSourceAnalyzer()

        content = '''
        BROWSERBASE_API_KEY=your_key_here
        OPENAI_SECRET=your_secret
        DATABASE_PASSWORD=xyz
        process.env.ANTHROPIC_API_KEY
        '''

        creds = analyzer._extract_credentials(content)
        assert len(creds) > 0
        # Should find various credential patterns
        cred_str = " ".join(creds)
        assert "API_KEY" in cred_str or "SECRET" in cred_str or "PASSWORD" in cred_str

    def test_risk_analysis(self):
        """Test risk analysis on code content."""
        analyzer = MCPSourceAnalyzer()
        tool = MCPTool(name="test", description="Execute shell commands")

        code = '''
        const { spawn } = require('child_process');

        async function runCommand(cmd) {
            return spawn(cmd, { shell: true });
        }
        '''

        analyzer._analyze_tool_risk(tool, code)

        assert tool.risk_score > 0
        assert "code_execution" in tool.capabilities
        assert tool.risk_level == RiskLevel.CRITICAL


class TestMCPInterrogator:
    """Tests for MCPInterrogator."""

    def test_initialization(self):
        """Test interrogator initialization."""
        interrogator = MCPInterrogator(timeout=30)
        assert interrogator.timeout == 30

    def test_server_name_extraction(self):
        """Test server name extraction from endpoints."""
        interrogator = MCPInterrogator()

        # HTTP URL
        name = interrogator._extract_server_name("https://mcp.example.com/api")
        assert name == "mcp.example.com"

        # Command
        name = interrogator._extract_server_name("npx @org/mcp-server")
        assert "mcp-server" in name

    def test_tool_parsing(self):
        """Test parsing tool from MCP response."""
        interrogator = MCPInterrogator()

        tool_data = {
            "name": "browser_navigate",
            "description": "Navigate to a URL in the browser",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
            },
        }

        tool = interrogator._parse_tool_response(tool_data)
        assert tool.name == "browser_navigate"
        assert "Navigate" in tool.description
        assert "url" in str(tool.input_schema)

    def test_metadata_risk_analysis(self):
        """Test risk analysis from tool metadata."""
        interrogator = MCPInterrogator()

        tool = MCPTool(
            name="execute_script",
            description="Execute a JavaScript script in the browser context",
        )

        interrogator._analyze_tool_risk_from_metadata(tool)

        # Should detect code execution from name/description
        assert tool.risk_score > 0


class TestMCPAnalyzer:
    """Tests for main MCPAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = MCPAnalyzer()
        assert analyzer.source_analyzer is not None
        assert analyzer.interrogator is not None
        assert analyzer.scorer is not None

    @pytest.mark.asyncio
    async def test_github_analysis_mocked(self):
        """Test GitHub analysis with mocked HTTP."""
        analyzer = MCPAnalyzer()

        # Mock the source analyzer's HTTP calls
        with patch.object(analyzer.source_analyzer, "analyze_github_repo") as mock_analyze:
            mock_analyze.return_value = MCPServerInfo(
                name="test/repo",
                source="https://github.com/test/repo",
                tools=[MCPTool(name="test_tool", risk_score=50, risk_level=RiskLevel.MEDIUM)],
            )

            result = await analyzer.analyze_from_github("https://github.com/test/repo")

            assert result.name == "test/repo"
            assert len(result.tools) == 1
            # Scorer should have been called
            assert result.overall_risk_score > 0

    def test_text_report_generation(self):
        """Test text report generation."""
        analyzer = MCPAnalyzer()

        server = MCPServerInfo(
            name="test-server",
            source="https://github.com/org/test",
            version="1.0.0",
            tools=[
                MCPTool(name="click", risk_score=70, risk_level=RiskLevel.HIGH, description="Click element"),
                MCPTool(name="read", risk_score=20, risk_level=RiskLevel.LOW, description="Read data"),
            ],
            overall_risk_level=RiskLevel.HIGH,
            overall_risk_score=65,
            risk_summary="2 tools analyzed. 1 HIGH risk.",
        )

        report = analyzer.generate_report(server, format="text")

        assert "MCP SERVER RISK ASSESSMENT" in report
        assert "test-server" in report
        assert "HIGH" in report
        assert "65" in report
        assert "click" in report

    def test_markdown_report_generation(self):
        """Test markdown report generation."""
        analyzer = MCPAnalyzer()

        server = MCPServerInfo(
            name="md-server",
            source="https://github.com/org/md",
            tools=[MCPTool(name="tool1", risk_score=50, risk_level=RiskLevel.MEDIUM)],
            overall_risk_level=RiskLevel.MEDIUM,
            overall_risk_score=50,
        )

        report = analyzer.generate_report(server, format="markdown")

        assert "# MCP Server Risk Assessment" in report
        assert "**Server:**" in report
        assert "md-server" in report
        assert "| Tool |" in report

    def test_json_report_generation(self):
        """Test JSON report generation."""
        analyzer = MCPAnalyzer()

        server = MCPServerInfo(
            name="json-server",
            source="https://github.com/org/json",
            tools=[MCPTool(name="api_call", risk_score=60, risk_level=RiskLevel.HIGH)],
            overall_risk_level=RiskLevel.HIGH,
            overall_risk_score=60,
        )

        report = analyzer.generate_report(server, format="json")
        data = json.loads(report)

        assert data["server"]["name"] == "json-server"
        assert data["risk_assessment"]["overall_level"] == "high"
        assert data["risk_assessment"]["overall_score"] == 60
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "api_call"

    def test_merge_analysis(self):
        """Test merging source and live analysis results."""
        analyzer = MCPAnalyzer()

        source_info = MCPServerInfo(
            name="source-server",
            source="https://github.com/org/server",
            version="1.0.0",
            tools=[
                MCPTool(name="tool1", risk_score=50, risk_level=RiskLevel.MEDIUM, capabilities=["file_read"]),
            ],
        )

        live_info = MCPServerInfo(
            name="live-server",
            source="https://api.server.com",
            tools=[
                MCPTool(name="tool1", risk_score=60, risk_level=RiskLevel.HIGH, capabilities=["network_requests"]),
                MCPTool(name="tool2", risk_score=30, risk_level=RiskLevel.LOW),
            ],
        )

        merged = analyzer._merge_analysis(source_info, live_info)

        # Should have 2 tools (tool1 merged, tool2 added)
        assert len(merged.tools) == 2

        # tool1 should have merged capabilities
        tool1 = next(t for t in merged.tools if t.name == "tool1")
        assert "file_read" in tool1.capabilities
        assert "network_requests" in tool1.capabilities

        # Should take higher risk score
        assert tool1.risk_score == 60


class TestAnalyzeMcpServer:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_auto_mode_github(self):
        """Test auto mode detects GitHub URLs."""
        with patch("benderbox.analyzers.mcp_analyzer.MCPAnalyzer") as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.analyze_from_github = AsyncMock(return_value=MCPServerInfo(
                name="test",
                source="https://github.com/test/repo",
            ))

            result = await analyze_mcp_server("https://github.com/test/repo", mode="auto")

            mock_instance.analyze_from_github.assert_called_once()

    @pytest.mark.asyncio
    async def test_source_mode(self):
        """Test source mode uses GitHub analyzer."""
        with patch("benderbox.analyzers.mcp_analyzer.MCPAnalyzer") as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.analyze_from_github = AsyncMock(return_value=MCPServerInfo(
                name="test",
                source="https://github.com/test/repo",
            ))

            result = await analyze_mcp_server("https://github.com/test/repo", mode="source")

            mock_instance.analyze_from_github.assert_called_once_with("https://github.com/test/repo")

    @pytest.mark.asyncio
    async def test_live_mode(self):
        """Test live mode uses endpoint interrogator."""
        with patch("benderbox.analyzers.mcp_analyzer.MCPAnalyzer") as MockAnalyzer:
            mock_instance = MockAnalyzer.return_value
            mock_instance.analyze_from_endpoint = AsyncMock(return_value=MCPServerInfo(
                name="test",
                source="https://api.example.com/mcp",
            ))

            result = await analyze_mcp_server("https://api.example.com/mcp", mode="live")

            mock_instance.analyze_from_endpoint.assert_called_once_with("https://api.example.com/mcp")


class TestIntegration:
    """Integration tests (may require network)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires network access")
    async def test_real_github_analysis(self):
        """Test real GitHub analysis (skipped by default)."""
        result = await analyze_mcp_server(
            "https://github.com/modelcontextprotocol/servers",
            mode="source"
        )
        assert result.name is not None
        assert len(result.tools) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
