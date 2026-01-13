"""
MCP Server Analyzer for BenderBox

Provides comprehensive risk assessment of MCP (Model Context Protocol) servers through:
- Source code analysis (GitHub repositories)
- Live MCP protocol interrogation
- Tool capability classification
- Risk scoring and reporting

Supports both Windows and Linux platforms.
"""

import asyncio
import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification for MCP tools."""
    CRITICAL = "critical"  # 80-100: Code exec, credential access, autonomous agents
    HIGH = "high"          # 60-79: Network requests, browser automation, DB writes
    MEDIUM = "medium"      # 40-59: Data extraction, file read, screenshots
    LOW = "low"            # 0-39: Read-only, local computation, stateless


@dataclass
class MCPTool:
    """Represents an MCP server tool."""
    name: str
    description: str = ""
    input_schema: Dict = field(default_factory=dict)
    output_schema: Dict = field(default_factory=dict)
    annotations: Dict = field(default_factory=dict)

    # Risk assessment fields
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: int = 0
    risk_factors: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class MCPServerInfo:
    """Information about an MCP server."""
    name: str
    source: str  # GitHub URL, npm package, or direct endpoint
    version: str = ""
    description: str = ""
    author: str = ""
    license: str = ""

    # Analysis results
    tools: List[MCPTool] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    required_credentials: List[str] = field(default_factory=list)

    # Overall risk assessment
    overall_risk_level: RiskLevel = RiskLevel.LOW
    overall_risk_score: int = 0
    risk_summary: str = ""


# Risk patterns for capability detection
RISK_PATTERNS = {
    # CRITICAL risk patterns (80-100)
    "code_execution": {
        "patterns": [
            r"\beval\b", r"\bexec\b", r"\bspawn\b", r"\bchild_process\b",
            r"\bFunction\s*\(", r"new\s+Function", r"\bvm\b\.run",
            r"subprocess", r"os\.system", r"shell_exec",
        ],
        "keywords": ["execute", "run command", "shell", "script", "eval"],
        "score": 95,
        "level": RiskLevel.CRITICAL,
    },
    "credential_access": {
        "patterns": [
            r"API_KEY", r"SECRET", r"TOKEN", r"PASSWORD", r"CREDENTIAL",
            r"\.env\b", r"process\.env", r"os\.environ",
        ],
        "keywords": ["api key", "secret", "token", "password", "credential", "auth"],
        "score": 90,
        "level": RiskLevel.CRITICAL,
    },
    "autonomous_agent": {
        "patterns": [
            r"\bagent\b", r"autonomous", r"self[_-]?driving",
            r"auto[_-]?pilot", r"recursive",
        ],
        "keywords": ["agent", "autonomous", "automated", "self-driving"],
        "score": 85,
        "level": RiskLevel.CRITICAL,
    },
    "file_write": {
        "patterns": [
            r"fs\.write", r"writeFile", r"createWriteStream",
            r"open\(.+['\"]w", r"\.write\(", r"save.*file",
        ],
        "keywords": ["write file", "save", "create file", "modify file"],
        "score": 80,
        "level": RiskLevel.CRITICAL,
    },

    # HIGH risk patterns (60-79)
    "network_requests": {
        "patterns": [
            r"\bfetch\b", r"axios", r"httpx?", r"requests?\.",
            r"XMLHttpRequest", r"\.get\(", r"\.post\(",
            r"WebSocket", r"socket\.io",
        ],
        "keywords": ["http", "request", "fetch", "api call", "webhook"],
        "score": 75,
        "level": RiskLevel.HIGH,
    },
    "browser_automation": {
        "patterns": [
            r"puppeteer", r"playwright", r"selenium", r"stagehand",
            r"browserbase", r"\.click\(", r"\.type\(", r"\.fill\(",
            r"page\.goto", r"navigate",
        ],
        "keywords": ["click", "type", "fill", "navigate", "browser", "automation"],
        "score": 70,
        "level": RiskLevel.HIGH,
    },
    "database_write": {
        "patterns": [
            r"INSERT\s+INTO", r"UPDATE\s+", r"DELETE\s+FROM",
            r"\.create\(", r"\.update\(", r"\.delete\(",
            r"db\.run", r"execute.*query",
        ],
        "keywords": ["insert", "update", "delete", "database", "sql"],
        "score": 70,
        "level": RiskLevel.HIGH,
    },
    "session_management": {
        "patterns": [
            r"session", r"cookie", r"jwt", r"oauth",
            r"login", r"authenticate",
        ],
        "keywords": ["session", "login", "authenticate", "cookie"],
        "score": 65,
        "level": RiskLevel.HIGH,
    },

    # MEDIUM risk patterns (40-59)
    "data_extraction": {
        "patterns": [
            r"extract", r"scrape", r"parse", r"querySelector",
            r"\.text\(", r"innerHTML", r"textContent",
        ],
        "keywords": ["extract", "scrape", "parse", "get data", "read content"],
        "score": 55,
        "level": RiskLevel.MEDIUM,
    },
    "file_read": {
        "patterns": [
            r"fs\.read", r"readFile", r"createReadStream",
            r"open\(.+['\"]r", r"\.read\(",
        ],
        "keywords": ["read file", "load", "open file"],
        "score": 50,
        "level": RiskLevel.MEDIUM,
    },
    "screenshot": {
        "patterns": [
            r"screenshot", r"capture", r"\.png\b", r"\.jpg\b",
            r"toBuffer", r"toBase64",
        ],
        "keywords": ["screenshot", "capture", "image", "screen"],
        "score": 45,
        "level": RiskLevel.MEDIUM,
    },
    "external_api": {
        "patterns": [
            r"api\.", r"client\.", r"sdk\.",
            r"@\w+/sdk", r"googleapis", r"aws-sdk",
        ],
        "keywords": ["api", "service", "integration"],
        "score": 45,
        "level": RiskLevel.MEDIUM,
    },

    # LOW risk patterns (0-39)
    "read_only": {
        "patterns": [
            r"get", r"list", r"describe", r"show",
            r"\.find\(", r"\.filter\(",
        ],
        "keywords": ["get", "list", "show", "view", "read"],
        "score": 20,
        "level": RiskLevel.LOW,
    },
    "local_computation": {
        "patterns": [
            r"calculate", r"compute", r"transform", r"format",
            r"Math\.", r"parse.*json",
        ],
        "keywords": ["calculate", "compute", "format", "transform"],
        "score": 15,
        "level": RiskLevel.LOW,
    },
}


class MCPSourceAnalyzer:
    """
    Analyzes MCP server source code from GitHub repositories.

    Extracts tool definitions, dependencies, and performs static analysis
    for risk assessment.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize source analyzer.

        Args:
            cache_dir: Directory for caching cloned repos. Uses temp if None.
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "benderbox_mcp_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def analyze_github_repo(self, repo_url: str) -> MCPServerInfo:
        """
        Analyze an MCP server from its GitHub repository.

        Args:
            repo_url: GitHub repository URL

        Returns:
            MCPServerInfo with analysis results
        """
        logger.info(f"Analyzing GitHub repo: {repo_url}")

        # Parse repo URL
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")

        owner, repo = path_parts[0], path_parts[1]
        repo_name = f"{owner}/{repo}"

        # Fetch repo contents via GitHub API
        server_info = MCPServerInfo(name=repo_name, source=repo_url)

        async with httpx.AsyncClient() as client:
            # Get package.json for metadata and dependencies
            package_json = await self._fetch_github_file(
                client, owner, repo, "package.json"
            )
            if package_json:
                self._parse_package_json(server_info, package_json)

            # Look for tool definitions in common locations
            tool_files = [
                "src/tools/index.ts",
                "src/tools/index.js",
                "src/index.ts",
                "src/index.js",
                "tools/index.ts",
                "tools/index.js",
            ]

            for tool_file in tool_files:
                content = await self._fetch_github_file(client, owner, repo, tool_file)
                if content:
                    tools = self._extract_tools_from_source(content, tool_file)
                    server_info.tools.extend(tools)

            # Analyze individual tool files
            tool_dir_content = await self._list_github_dir(client, owner, repo, "src/tools")
            if tool_dir_content:
                for file_info in tool_dir_content:
                    if file_info.get("name", "").endswith((".ts", ".js")):
                        file_path = f"src/tools/{file_info['name']}"
                        content = await self._fetch_github_file(
                            client, owner, repo, file_path
                        )
                        if content:
                            tools = self._extract_tools_from_source(content, file_path)
                            # Merge with existing tools or add new ones
                            for tool in tools:
                                existing = next(
                                    (t for t in server_info.tools if t.name == tool.name),
                                    None
                                )
                                if existing:
                                    # Merge risk factors
                                    existing.risk_factors.extend(tool.risk_factors)
                                    existing.capabilities.extend(tool.capabilities)
                                else:
                                    server_info.tools.append(tool)

            # Check for environment/credential requirements
            env_files = [".env.example", ".env.template", "README.md"]
            for env_file in env_files:
                content = await self._fetch_github_file(client, owner, repo, env_file)
                if content:
                    creds = self._extract_credentials(content)
                    server_info.required_credentials.extend(creds)

            # Remove duplicates
            server_info.required_credentials = list(set(server_info.required_credentials))

        return server_info

    async def _fetch_github_file(
        self, client: httpx.AsyncClient, owner: str, repo: str, path: str
    ) -> Optional[str]:
        """Fetch a file from GitHub raw content."""
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
        try:
            response = await client.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            # Try master branch
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{path}"
            response = await client.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.debug(f"Failed to fetch {path}: {e}")
        return None

    async def _list_github_dir(
        self, client: httpx.AsyncClient, owner: str, repo: str, path: str
    ) -> Optional[List[Dict]]:
        """List contents of a GitHub directory."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        try:
            response = await client.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.debug(f"Failed to list {path}: {e}")
        return None

    def _parse_package_json(self, server_info: MCPServerInfo, content: str) -> None:
        """Parse package.json for metadata and dependencies."""
        try:
            pkg = json.loads(content)
            server_info.version = pkg.get("version", "")
            server_info.description = pkg.get("description", "")
            server_info.author = pkg.get("author", "")
            if isinstance(server_info.author, dict):
                server_info.author = server_info.author.get("name", "")
            server_info.license = pkg.get("license", "")

            # Collect dependencies
            deps = {}
            deps.update(pkg.get("dependencies", {}))
            deps.update(pkg.get("devDependencies", {}))
            server_info.dependencies = deps

        except json.JSONDecodeError:
            logger.warning("Failed to parse package.json")

    def _extract_tools_from_source(self, content: str, filename: str) -> List[MCPTool]:
        """Extract tool definitions from source code."""
        tools = []

        # Pattern for tool schema definitions
        # Matches: name: "tool_name" or name: 'tool_name'
        name_pattern = r'name:\s*["\']([^"\']+)["\']'
        desc_pattern = r'description:\s*["\']([^"\']+)["\']'

        # Find all tool names
        names = re.findall(name_pattern, content)
        descriptions = re.findall(desc_pattern, content)

        # Create tools from matches
        for i, name in enumerate(names):
            desc = descriptions[i] if i < len(descriptions) else ""
            tool = MCPTool(name=name, description=desc)

            # Analyze the tool's risk based on content around its definition
            self._analyze_tool_risk(tool, content)
            tools.append(tool)

        # If no explicit tool definitions found, analyze the whole file
        if not tools and filename.endswith((".ts", ".js")):
            # Create a tool from the filename
            tool_name = Path(filename).stem
            if tool_name not in ["index", "utils", "helpers"]:
                tool = MCPTool(name=f"mcp_{tool_name}", description=f"Tool from {filename}")
                self._analyze_tool_risk(tool, content)
                if tool.risk_factors:  # Only add if we found something
                    tools.append(tool)

        return tools

    def _analyze_tool_risk(self, tool: MCPTool, content: str) -> None:
        """Analyze tool risk based on code patterns."""
        content_lower = content.lower()
        tool_desc_lower = tool.description.lower()

        highest_score = 0
        highest_level = RiskLevel.LOW

        for capability, config in RISK_PATTERNS.items():
            matched = False

            # Check regex patterns in code
            for pattern in config["patterns"]:
                if re.search(pattern, content, re.IGNORECASE):
                    matched = True
                    break

            # Check keywords in description
            if not matched:
                for keyword in config["keywords"]:
                    if keyword in tool_desc_lower:
                        matched = True
                        break

            if matched:
                tool.capabilities.append(capability)
                tool.risk_factors.append(f"{capability}: {config['level'].value} risk")

                if config["score"] > highest_score:
                    highest_score = config["score"]
                    highest_level = config["level"]

        tool.risk_score = highest_score
        tool.risk_level = highest_level

    def _extract_credentials(self, content: str) -> List[str]:
        """Extract required credentials/environment variables."""
        creds = []

        # Common patterns for env vars
        patterns = [
            r'([A-Z][A-Z0-9_]*_(?:API_KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL))',
            r'([A-Z][A-Z0-9_]*_(?:KEY|SECRET|TOKEN))',
            r'process\.env\.([A-Z][A-Z0-9_]+)',
            r'os\.environ\[["\']([A-Z][A-Z0-9_]+)',
            r'\$\{([A-Z][A-Z0-9_]+)\}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            creds.extend(matches)

        return list(set(creds))


class MCPInterrogator:
    """
    Interrogates live MCP servers via the Model Context Protocol.

    Connects to MCP servers and discovers their tools through the
    standard protocol endpoints.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize MCP interrogator.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    async def interrogate_server(
        self,
        server_endpoint: str,
        transport: str = "http"
    ) -> MCPServerInfo:
        """
        Interrogate an MCP server to discover its tools.

        Args:
            server_endpoint: Server URL or connection string
            transport: Transport type ("http", "stdio", "websocket")

        Returns:
            MCPServerInfo with discovered tools
        """
        logger.info(f"Interrogating MCP server: {server_endpoint}")

        server_info = MCPServerInfo(
            name=self._extract_server_name(server_endpoint),
            source=server_endpoint
        )

        if transport == "http":
            await self._interrogate_http(server_info, server_endpoint)
        elif transport == "stdio":
            await self._interrogate_stdio(server_info, server_endpoint)
        else:
            raise ValueError(f"Unsupported transport: {transport}")

        return server_info

    async def _interrogate_http(
        self, server_info: MCPServerInfo, endpoint: str
    ) -> None:
        """Interrogate via HTTP transport."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Call tools/list
            try:
                response = await client.post(
                    endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {}
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    if "result" in result and "tools" in result["result"]:
                        for tool_data in result["result"]["tools"]:
                            tool = self._parse_tool_response(tool_data)
                            server_info.tools.append(tool)
            except Exception as e:
                logger.error(f"HTTP interrogation failed: {e}")

    async def _interrogate_stdio(
        self, server_info: MCPServerInfo, command: str
    ) -> None:
        """Interrogate via STDIO transport."""
        try:
            # Start the MCP server process
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Send tools/list request
            request = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }) + "\n"

            stdout, stderr = await asyncio.wait_for(
                process.communicate(request.encode()),
                timeout=self.timeout
            )

            # Parse response
            for line in stdout.decode().split("\n"):
                if line.strip():
                    try:
                        result = json.loads(line)
                        if "result" in result and "tools" in result["result"]:
                            for tool_data in result["result"]["tools"]:
                                tool = self._parse_tool_response(tool_data)
                                server_info.tools.append(tool)
                    except json.JSONDecodeError:
                        continue

        except asyncio.TimeoutError:
            logger.error("STDIO interrogation timed out")
        except Exception as e:
            logger.error(f"STDIO interrogation failed: {e}")

    def _parse_tool_response(self, tool_data: Dict) -> MCPTool:
        """Parse tool data from MCP response."""
        tool = MCPTool(
            name=tool_data.get("name", "unknown"),
            description=tool_data.get("description", ""),
            input_schema=tool_data.get("inputSchema", {}),
            output_schema=tool_data.get("outputSchema", {}),
            annotations=tool_data.get("annotations", {})
        )

        # Analyze risk based on description and schema
        self._analyze_tool_risk_from_metadata(tool)

        return tool

    def _analyze_tool_risk_from_metadata(self, tool: MCPTool) -> None:
        """Analyze tool risk from its metadata."""
        desc_lower = tool.description.lower()
        name_lower = tool.name.lower()
        combined = f"{name_lower} {desc_lower}"

        highest_score = 0
        highest_level = RiskLevel.LOW

        for capability, config in RISK_PATTERNS.items():
            matched = False

            # Check keywords in name and description
            for keyword in config["keywords"]:
                if keyword in combined:
                    matched = True
                    break

            if matched:
                tool.capabilities.append(capability)
                tool.risk_factors.append(f"{capability}: {config['level'].value} risk")

                if config["score"] > highest_score:
                    highest_score = config["score"]
                    highest_level = config["level"]

        # Additional schema-based analysis
        schema_str = json.dumps(tool.input_schema).lower()
        for capability, config in RISK_PATTERNS.items():
            if capability not in tool.capabilities:
                for keyword in config["keywords"]:
                    if keyword in schema_str:
                        tool.capabilities.append(capability)
                        tool.risk_factors.append(
                            f"{capability} (schema): {config['level'].value} risk"
                        )
                        if config["score"] > highest_score:
                            highest_score = config["score"]
                            highest_level = config["level"]
                        break

        tool.risk_score = highest_score
        tool.risk_level = highest_level

    def _extract_server_name(self, endpoint: str) -> str:
        """Extract server name from endpoint."""
        if endpoint.startswith("http"):
            parsed = urlparse(endpoint)
            return parsed.netloc or endpoint
        return endpoint.split("/")[-1].split()[0]


class MCPRiskScorer:
    """
    Calculates overall risk scores for MCP servers.
    """

    def calculate_server_risk(self, server_info: MCPServerInfo) -> None:
        """
        Calculate overall risk score for an MCP server.

        Args:
            server_info: Server info to score (modified in place)
        """
        if not server_info.tools:
            server_info.overall_risk_score = 0
            server_info.overall_risk_level = RiskLevel.LOW
            server_info.risk_summary = "No tools found"
            return

        # Calculate weighted average with max influence
        scores = [tool.risk_score for tool in server_info.tools]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Overall score: 70% max + 30% average
        overall_score = int(0.7 * max_score + 0.3 * avg_score)

        # Adjust for number of high-risk tools
        critical_count = sum(
            1 for t in server_info.tools if t.risk_level == RiskLevel.CRITICAL
        )
        high_count = sum(
            1 for t in server_info.tools if t.risk_level == RiskLevel.HIGH
        )

        if critical_count > 0:
            overall_score = max(overall_score, 80)
        if high_count >= 3:
            overall_score = max(overall_score, 70)

        # Credential penalty
        if len(server_info.required_credentials) > 3:
            overall_score = min(100, overall_score + 5)

        # Dependency risk (check for known risky packages)
        risky_deps = ["child_process", "vm", "eval", "shelljs"]
        for dep in risky_deps:
            if dep in server_info.dependencies:
                overall_score = min(100, overall_score + 10)

        server_info.overall_risk_score = overall_score
        server_info.overall_risk_level = self._score_to_level(overall_score)
        server_info.risk_summary = self._generate_summary(server_info)

    def _score_to_level(self, score: int) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score >= 80:
            return RiskLevel.CRITICAL
        elif score >= 60:
            return RiskLevel.HIGH
        elif score >= 40:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_summary(self, server_info: MCPServerInfo) -> str:
        """Generate human-readable risk summary."""
        parts = []

        # Tool count by risk level
        critical = sum(1 for t in server_info.tools if t.risk_level == RiskLevel.CRITICAL)
        high = sum(1 for t in server_info.tools if t.risk_level == RiskLevel.HIGH)
        medium = sum(1 for t in server_info.tools if t.risk_level == RiskLevel.MEDIUM)
        low = sum(1 for t in server_info.tools if t.risk_level == RiskLevel.LOW)

        parts.append(f"{len(server_info.tools)} tools analyzed")

        if critical > 0:
            parts.append(f"{critical} CRITICAL risk")
        if high > 0:
            parts.append(f"{high} HIGH risk")
        if medium > 0:
            parts.append(f"{medium} MEDIUM risk")
        if low > 0:
            parts.append(f"{low} LOW risk")

        # Key capabilities
        all_caps = set()
        for tool in server_info.tools:
            all_caps.update(tool.capabilities)

        if "code_execution" in all_caps:
            parts.append("Can execute code")
        if "credential_access" in all_caps:
            parts.append("Requires credentials")
        if "browser_automation" in all_caps:
            parts.append("Browser automation enabled")
        if "file_write" in all_caps:
            parts.append("Can write files")

        return ". ".join(parts) + "."


class MCPAnalyzer:
    """
    Main MCP server analyzer combining source analysis and live interrogation.

    Provides comprehensive risk assessment for MCP servers through:
    - GitHub source code analysis
    - Live MCP protocol interrogation
    - Combined hybrid analysis
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize MCP analyzer.

        Args:
            cache_dir: Directory for caching analysis data
        """
        self.source_analyzer = MCPSourceAnalyzer(cache_dir)
        self.interrogator = MCPInterrogator()
        self.scorer = MCPRiskScorer()

    async def analyze_from_github(self, repo_url: str) -> MCPServerInfo:
        """
        Analyze an MCP server from its GitHub repository.

        Args:
            repo_url: GitHub repository URL

        Returns:
            MCPServerInfo with analysis results
        """
        server_info = await self.source_analyzer.analyze_github_repo(repo_url)
        self.scorer.calculate_server_risk(server_info)
        return server_info

    async def analyze_from_endpoint(
        self,
        endpoint: str,
        transport: str = "http"
    ) -> MCPServerInfo:
        """
        Analyze an MCP server via live interrogation.

        Args:
            endpoint: Server endpoint URL or command
            transport: Transport type ("http", "stdio")

        Returns:
            MCPServerInfo with analysis results
        """
        server_info = await self.interrogator.interrogate_server(endpoint, transport)
        self.scorer.calculate_server_risk(server_info)
        return server_info

    async def analyze_hybrid(
        self,
        repo_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        transport: str = "http"
    ) -> MCPServerInfo:
        """
        Perform hybrid analysis combining source and live interrogation.

        Args:
            repo_url: GitHub repository URL (optional)
            endpoint: Live server endpoint (optional)
            transport: Transport type for live interrogation

        Returns:
            MCPServerInfo with combined analysis results
        """
        source_info = None
        live_info = None

        # Perform source analysis if repo provided
        if repo_url:
            try:
                source_info = await self.source_analyzer.analyze_github_repo(repo_url)
            except Exception as e:
                logger.warning(f"Source analysis failed: {e}")

        # Perform live interrogation if endpoint provided
        if endpoint:
            try:
                live_info = await self.interrogator.interrogate_server(endpoint, transport)
            except Exception as e:
                logger.warning(f"Live interrogation failed: {e}")

        # Merge results
        if source_info and live_info:
            return self._merge_analysis(source_info, live_info)
        elif source_info:
            self.scorer.calculate_server_risk(source_info)
            return source_info
        elif live_info:
            self.scorer.calculate_server_risk(live_info)
            return live_info
        else:
            raise ValueError("No analysis source provided or all analyses failed")

    def _merge_analysis(
        self, source_info: MCPServerInfo, live_info: MCPServerInfo
    ) -> MCPServerInfo:
        """Merge source and live analysis results."""
        # Start with source info as base
        merged = MCPServerInfo(
            name=source_info.name or live_info.name,
            source=source_info.source,
            version=source_info.version,
            description=source_info.description or live_info.description,
            author=source_info.author,
            license=source_info.license,
            dependencies=source_info.dependencies,
            required_credentials=source_info.required_credentials,
        )

        # Merge tools
        tool_map: Dict[str, MCPTool] = {}

        # Add source tools
        for tool in source_info.tools:
            tool_map[tool.name] = tool

        # Merge or add live tools
        for live_tool in live_info.tools:
            if live_tool.name in tool_map:
                # Merge risk factors and capabilities
                existing = tool_map[live_tool.name]
                existing.risk_factors.extend(live_tool.risk_factors)
                existing.capabilities.extend(live_tool.capabilities)
                existing.risk_factors = list(set(existing.risk_factors))
                existing.capabilities = list(set(existing.capabilities))

                # Take higher risk score
                if live_tool.risk_score > existing.risk_score:
                    existing.risk_score = live_tool.risk_score
                    existing.risk_level = live_tool.risk_level
            else:
                tool_map[live_tool.name] = live_tool

        merged.tools = list(tool_map.values())

        # Calculate final score
        self.scorer.calculate_server_risk(merged)

        return merged

    def generate_report(self, server_info: MCPServerInfo, format: str = "text") -> str:
        """
        Generate analysis report.

        Args:
            server_info: Analysis results
            format: Output format ("text", "json", "markdown")

        Returns:
            Formatted report string
        """
        if format == "json":
            return self._generate_json_report(server_info)
        elif format == "markdown":
            return self._generate_markdown_report(server_info)
        else:
            return self._generate_text_report(server_info)

    def _generate_text_report(self, server_info: MCPServerInfo) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 60,
            "MCP SERVER RISK ASSESSMENT REPORT",
            "=" * 60,
            "",
            f"Server: {server_info.name}",
            f"Source: {server_info.source}",
            f"Version: {server_info.version or 'Unknown'}",
            "",
            "-" * 40,
            "OVERALL RISK ASSESSMENT",
            "-" * 40,
            f"Risk Level: {server_info.overall_risk_level.value.upper()}",
            f"Risk Score: {server_info.overall_risk_score}/100",
            f"Summary: {server_info.risk_summary}",
            "",
        ]

        if server_info.required_credentials:
            lines.extend([
                "-" * 40,
                "REQUIRED CREDENTIALS",
                "-" * 40,
            ])
            for cred in server_info.required_credentials:
                lines.append(f"  - {cred}")
            lines.append("")

        lines.extend([
            "-" * 40,
            "TOOL ANALYSIS",
            "-" * 40,
        ])

        # Sort tools by risk score
        sorted_tools = sorted(
            server_info.tools, key=lambda t: t.risk_score, reverse=True
        )

        for tool in sorted_tools:
            risk_icon = {
                RiskLevel.CRITICAL: "🔴",
                RiskLevel.HIGH: "🟠",
                RiskLevel.MEDIUM: "🟡",
                RiskLevel.LOW: "🟢",
            }.get(tool.risk_level, "⚪")

            lines.append(f"\n{risk_icon} {tool.name}")
            lines.append(f"   Risk: {tool.risk_level.value.upper()} ({tool.risk_score}/100)")
            if tool.description:
                lines.append(f"   Description: {tool.description[:80]}...")
            if tool.capabilities:
                lines.append(f"   Capabilities: {', '.join(tool.capabilities[:5])}")
            if tool.risk_factors:
                lines.append(f"   Risk Factors:")
                for factor in tool.risk_factors[:3]:
                    lines.append(f"     - {factor}")

        lines.extend([
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def _generate_markdown_report(self, server_info: MCPServerInfo) -> str:
        """Generate Markdown report."""
        lines = [
            "# MCP Server Risk Assessment Report",
            "",
            f"**Server:** {server_info.name}",
            f"**Source:** {server_info.source}",
            f"**Version:** {server_info.version or 'Unknown'}",
            "",
            "## Overall Risk Assessment",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Risk Level | **{server_info.overall_risk_level.value.upper()}** |",
            f"| Risk Score | {server_info.overall_risk_score}/100 |",
            "",
            f"**Summary:** {server_info.risk_summary}",
            "",
        ]

        if server_info.required_credentials:
            lines.extend([
                "## Required Credentials",
                "",
            ])
            for cred in server_info.required_credentials:
                lines.append(f"- `{cred}`")
            lines.append("")

        lines.extend([
            "## Tool Analysis",
            "",
            "| Tool | Risk Level | Score | Capabilities |",
            "|------|------------|-------|--------------|",
        ])

        sorted_tools = sorted(
            server_info.tools, key=lambda t: t.risk_score, reverse=True
        )

        for tool in sorted_tools:
            caps = ", ".join(tool.capabilities[:3]) if tool.capabilities else "-"
            lines.append(
                f"| {tool.name} | {tool.risk_level.value.upper()} | "
                f"{tool.risk_score} | {caps} |"
            )

        lines.extend([
            "",
            "## Tool Details",
            "",
        ])

        for tool in sorted_tools:
            risk_icon = {
                RiskLevel.CRITICAL: "🔴",
                RiskLevel.HIGH: "🟠",
                RiskLevel.MEDIUM: "🟡",
                RiskLevel.LOW: "🟢",
            }.get(tool.risk_level, "⚪")

            lines.extend([
                f"### {risk_icon} {tool.name}",
                "",
                f"- **Risk Level:** {tool.risk_level.value.upper()}",
                f"- **Risk Score:** {tool.risk_score}/100",
            ])

            if tool.description:
                lines.append(f"- **Description:** {tool.description}")

            if tool.risk_factors:
                lines.append("- **Risk Factors:**")
                for factor in tool.risk_factors:
                    lines.append(f"  - {factor}")

            lines.append("")

        return "\n".join(lines)

    def _generate_json_report(self, server_info: MCPServerInfo) -> str:
        """Generate JSON report."""
        data = {
            "server": {
                "name": server_info.name,
                "source": server_info.source,
                "version": server_info.version,
                "description": server_info.description,
                "author": server_info.author,
                "license": server_info.license,
            },
            "risk_assessment": {
                "overall_level": server_info.overall_risk_level.value,
                "overall_score": server_info.overall_risk_score,
                "summary": server_info.risk_summary,
            },
            "required_credentials": server_info.required_credentials,
            "dependencies": server_info.dependencies,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "risk_level": tool.risk_level.value,
                    "risk_score": tool.risk_score,
                    "capabilities": tool.capabilities,
                    "risk_factors": tool.risk_factors,
                    "input_schema": tool.input_schema,
                }
                for tool in server_info.tools
            ],
        }
        return json.dumps(data, indent=2)


# Convenience function for quick analysis
async def analyze_mcp_server(
    source: str,
    mode: str = "auto"
) -> MCPServerInfo:
    """
    Analyze an MCP server.

    Args:
        source: GitHub URL, npm package, or endpoint
        mode: Analysis mode ("source", "live", "hybrid", "auto")

    Returns:
        MCPServerInfo with analysis results
    """
    analyzer = MCPAnalyzer()

    if mode == "auto":
        # Detect source type
        if "github.com" in source:
            mode = "source"
        elif source.startswith("http"):
            mode = "live"
        else:
            mode = "source"  # Assume npm package -> GitHub

    if mode == "source":
        return await analyzer.analyze_from_github(source)
    elif mode == "live":
        return await analyzer.analyze_from_endpoint(source)
    elif mode == "hybrid":
        # For hybrid, try to detect both GitHub and endpoint
        repo_url = source if "github.com" in source else None
        endpoint = source if source.startswith("http") and "github.com" not in source else None
        return await analyzer.analyze_hybrid(repo_url=repo_url, endpoint=endpoint)
    else:
        raise ValueError(f"Unknown mode: {mode}")
