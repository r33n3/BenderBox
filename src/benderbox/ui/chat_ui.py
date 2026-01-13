"""
Interactive Chat UI for BenderBox

Provides a conversational interface for security analysis
with command handling and context awareness.
"""

import asyncio
import logging
import shlex
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from benderbox.ui.terminal import TerminalUI, ProgressSpinner

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Types of chat commands."""

    ANALYZE = "analyze"
    SEMANTIC = "semantic"  # Semantic code analysis
    SEARCH = "search"  # Semantic search across reports
    COMPARE = "compare"
    STATUS = "status"
    REPORTS = "reports"
    EXPORT = "export"
    HELP = "help"
    CLEAR = "clear"
    EXIT = "exit"
    QUERY = "query"  # Natural language query
    # MCP server commands
    MCP = "mcp"  # MCP server operations
    MCP_CONNECT = "mcp_connect"
    MCP_TOOLS = "mcp_tools"
    MCP_INTERROGATE = "mcp_interrogate"
    MCP_CALL = "mcp_call"
    # Context analysis commands
    CONTEXT = "context"  # Context/instruction analysis
    CONTEXT_ANALYZE = "context_analyze"
    CONTEXT_SCAN = "context_scan"
    CONTEXT_OUTPUT = "context_output"


@dataclass
class ParsedCommand:
    """Parsed user command."""

    command_type: CommandType
    args: List[str]
    raw_input: str
    flags: Dict[str, str]


class ChatUI:
    """
    Interactive chat interface for BenderBox.

    Provides:
    - Command parsing and handling
    - Natural language query routing
    - Conversation history display
    - Analysis result presentation
    """

    # Command aliases
    COMMAND_ALIASES = {
        "analyze": ["analyze", "scan", "check", "test", "a"],
        "semantic": ["semantic", "code", "review", "security-scan", "sec"],
        "search": ["search", "find", "lookup", "query-reports"],
        "compare": ["compare", "diff", "vs", "c"],
        "status": ["status", "info", "state", "s"],
        "reports": ["reports", "list", "history", "r"],
        "export": ["export", "save", "e"],
        "help": ["help", "?", "h"],
        "clear": ["clear", "cls", "reset"],
        "exit": ["exit", "quit", "q", "bye"],
        # MCP commands
        "mcp": ["mcp"],
        "mcp_connect": ["mcp-connect", "connect-mcp"],
        "mcp_tools": ["mcp-tools", "tools-mcp"],
        "mcp_interrogate": ["mcp-interrogate", "interrogate-mcp", "mcp-test"],
        "mcp_call": ["mcp-call", "call-mcp"],
        # Context commands
        "context": ["context", "ctx"],
        "context_analyze": ["context-analyze", "analyze-context", "ctx-analyze"],
        "context_scan": ["context-scan", "scan-context", "ctx-scan"],
        "context_output": ["context-output", "analyze-output", "ctx-output"],
    }

    def __init__(
        self,
        conversation_manager=None,
        terminal_ui: Optional[TerminalUI] = None,
    ):
        """
        Initialize ChatUI.

        Args:
            conversation_manager: ConversationManager for query processing.
            terminal_ui: TerminalUI for output.
        """
        self._conversation = conversation_manager
        self.ui = terminal_ui or TerminalUI()

        # Command handlers
        self._handlers: Dict[CommandType, Callable] = {}

        # Session state
        self._running = False
        self._last_result: Optional[Dict[str, Any]] = None

    def register_handler(
        self,
        command_type: CommandType,
        handler: Callable,
    ) -> None:
        """
        Register a command handler.

        Args:
            command_type: Type of command.
            handler: Async handler function.
        """
        self._handlers[command_type] = handler

    def parse_input(self, user_input: str) -> ParsedCommand:
        """
        Parse user input into a command.

        Args:
            user_input: Raw user input string.

        Returns:
            ParsedCommand object.
        """
        user_input = user_input.strip()

        if not user_input:
            return ParsedCommand(
                command_type=CommandType.QUERY,
                args=[],
                raw_input=user_input,
                flags={},
            )

        # Check for explicit commands (starting with / or matching aliases)
        parts = user_input.split(maxsplit=1)
        first_word = parts[0].lower().lstrip("/")

        # Check against command aliases
        for cmd_type, aliases in self.COMMAND_ALIASES.items():
            if first_word in aliases:
                args_str = parts[1] if len(parts) > 1 else ""
                args, flags = self._parse_args(args_str)
                return ParsedCommand(
                    command_type=CommandType[cmd_type.upper()],
                    args=args,
                    raw_input=user_input,
                    flags=flags,
                )

        # Default to natural language query
        return ParsedCommand(
            command_type=CommandType.QUERY,
            args=[user_input],
            raw_input=user_input,
            flags={},
        )

    def _parse_args(self, args_str: str) -> Tuple[List[str], Dict[str, str]]:
        """Parse arguments and flags from string."""
        import sys

        if not args_str:
            return [], {}

        try:
            # Use posix=False on Windows to preserve backslashes in paths
            parts = shlex.split(args_str, posix=(sys.platform != "win32"))
        except ValueError:
            parts = args_str.split()

        args = []
        flags = {}

        i = 0
        while i < len(parts):
            part = parts[i]
            if part.startswith("--"):
                key = part[2:]
                if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                    flags[key] = parts[i + 1]
                    i += 2
                else:
                    flags[key] = "true"
                    i += 1
            elif part.startswith("-"):
                key = part[1:]
                flags[key] = "true"
                i += 1
            else:
                args.append(part)
                i += 1

        return args, flags

    async def handle_command(self, command: ParsedCommand) -> bool:
        """
        Handle a parsed command.

        Args:
            command: ParsedCommand to handle.

        Returns:
            True to continue, False to exit.
        """
        # Check for registered handlers first
        if command.command_type in self._handlers:
            await self._handlers[command.command_type](command)
            return True

        # Built-in handlers
        if command.command_type == CommandType.EXIT:
            return await self._handle_exit(command)

        elif command.command_type == CommandType.HELP:
            await self._handle_help(command)

        elif command.command_type == CommandType.CLEAR:
            await self._handle_clear(command)

        elif command.command_type == CommandType.STATUS:
            await self._handle_status(command)

        elif command.command_type == CommandType.REPORTS:
            await self._handle_reports(command)

        elif command.command_type == CommandType.ANALYZE:
            await self._handle_analyze(command)

        elif command.command_type == CommandType.SEMANTIC:
            await self._handle_semantic(command)

        elif command.command_type == CommandType.SEARCH:
            await self._handle_search(command)

        elif command.command_type == CommandType.COMPARE:
            await self._handle_compare(command)

        elif command.command_type == CommandType.EXPORT:
            await self._handle_export(command)

        elif command.command_type == CommandType.QUERY:
            await self._handle_query(command)

        # MCP commands
        elif command.command_type == CommandType.MCP:
            await self._handle_mcp(command)

        elif command.command_type == CommandType.MCP_CONNECT:
            await self._handle_mcp_connect(command)

        elif command.command_type == CommandType.MCP_TOOLS:
            await self._handle_mcp_tools(command)

        elif command.command_type == CommandType.MCP_INTERROGATE:
            await self._handle_mcp_interrogate(command)

        elif command.command_type == CommandType.MCP_CALL:
            await self._handle_mcp_call(command)

        # Context commands
        elif command.command_type == CommandType.CONTEXT:
            await self._handle_context(command)

        elif command.command_type == CommandType.CONTEXT_ANALYZE:
            await self._handle_context_analyze(command)

        elif command.command_type == CommandType.CONTEXT_SCAN:
            await self._handle_context_scan(command)

        elif command.command_type == CommandType.CONTEXT_OUTPUT:
            await self._handle_context_output(command)

        return True

    async def _handle_exit(self, command: ParsedCommand) -> bool:
        """Handle exit command."""
        self.ui.print_info("Goodbye!")
        self._running = False
        return False

    async def _handle_help(self, command: ParsedCommand) -> None:
        """Handle help command."""
        self.ui.print_help()

    async def _handle_clear(self, command: ParsedCommand) -> None:
        """Handle clear command."""
        if self._conversation:
            self._conversation.clear_history()
        self.ui.print_success("Conversation cleared.")

    async def _handle_status(self, command: ParsedCommand) -> None:
        """Handle status command."""
        if self._conversation:
            # Use conversation manager to get status
            response = await self._conversation.process_query("status")
            self.ui.print_markdown(response.content)
        else:
            self.ui.print_info("BenderBox v3.0.0-alpha")
            self.ui.print_info("Conversation manager not initialized.")

    async def _handle_reports(self, command: ParsedCommand) -> None:
        """Handle reports command."""
        if self._conversation:
            response = await self._conversation.process_query("show recent reports")
            self.ui.print_markdown(response.content)
        else:
            self.ui.print_warning("Report listing requires conversation manager.")

    async def _handle_analyze(self, command: ParsedCommand) -> None:
        """Handle analyze command."""
        if not command.args:
            self.ui.print_error("Please specify a target to analyze.")
            self.ui.print_info("Usage: analyze <path|url|hf_model> [--profile <profile>]")
            return

        target = command.args[0]
        profile = command.flags.get("profile", "standard")

        # Detect source type and show appropriate message
        try:
            from benderbox.utils import ModelSourceHandler, ModelSource
            handler = ModelSourceHandler()
            source_type = handler.detect_source(target)

            if source_type == ModelSource.URL:
                self.ui.print_info(f"Target: {target}")
                self.ui.print_info("Source: URL (downloading if not cached...)")
            elif source_type == ModelSource.HUGGINGFACE:
                self.ui.print_info(f"Target: {target}")
                self.ui.print_info("Source: Hugging Face (downloading if not cached...)")
        except ImportError:
            pass  # Utils not available, continue with basic behavior

        query = f"analyze {target} with {profile} profile"

        if self._conversation:
            spinner_msg = f"Analyzing {target}..."
            with ProgressSpinner(self.ui, spinner_msg):
                response = await self._conversation.process_query(query)

            self._last_result = response.analysis_result

            if response.analysis_result:
                self.ui.print_analysis_summary(response.analysis_result)

                # Print findings
                results = response.analysis_result.get("results", [])
                failed = [r for r in results if r.get("status") in ("failed", "warning")]
                if failed:
                    self.ui.print_findings_table(failed, "Issues Found")
            else:
                self.ui.print_markdown(response.content)
        else:
            self.ui.print_warning("Analysis requires conversation manager.")

    async def _handle_semantic(self, command: ParsedCommand) -> None:
        """Handle semantic code analysis command."""
        if not command.args:
            self.ui.print_error("Please specify a file to analyze.")
            self.ui.print_info("Usage: semantic <file.py> [--depth quick|standard|deep]")
            return

        target = command.args[0]
        depth = command.flags.get("depth", command.flags.get("profile", "standard"))

        self.ui.print_info(f"Target: {target}")
        self.ui.print_info(f"Analysis depth: {depth}")

        if self._conversation:
            # Use the analyze_code intent
            query = f"analyze code {target} with {depth} depth"

            spinner_msg = f"Performing semantic analysis on {target}..."
            with ProgressSpinner(self.ui, spinner_msg):
                response = await self._conversation.process_query(query)

            self._last_result = response.analysis_result

            if response.analysis_result:
                self._print_semantic_result(response.analysis_result)
            else:
                self.ui.print_markdown(response.content)
        else:
            self.ui.print_warning("Semantic analysis requires conversation manager.")

    def _print_semantic_result(self, result: Dict[str, Any]) -> None:
        """Print semantic analysis result with formatting."""
        # Check if this is a semantic analysis result
        if result.get("analysis_type") != "semantic":
            self.ui.print_analysis_summary(result)
            return

        summary = result.get("summary", {})
        findings = result.get("findings", [])

        # Print header
        self.ui.print_header(f"Semantic Analysis: {result.get('target', 'Unknown')}")

        # Print summary
        risk_score = summary.get("risk_score", 0)
        risk_color = "red" if risk_score >= 70 else "yellow" if risk_score >= 40 else "green"

        if self.ui.console:
            from rich.table import Table
            from rich.panel import Panel

            # Summary panel
            summary_text = f"""
Risk Score: [{risk_color}]{risk_score:.0f}/100[/{risk_color}]
Critical: {summary.get('critical_count', 0)} | High: {summary.get('high_count', 0)}
Analysis Depth: {result.get('analysis_depth', 'standard')}
LLM Used: {'Yes' if result.get('llm_used') else 'No (pattern-based)'}

{summary.get('text', 'No summary available.')}
"""
            self.ui.console.print(Panel(summary_text, title="Summary", border_style="cyan"))

            # Findings table
            if findings:
                table = Table(title="Security Findings", show_header=True)
                table.add_column("Severity", style="bold")
                table.add_column("Category")
                table.add_column("Title")
                table.add_column("Location")
                table.add_column("Confidence")

                severity_colors = {
                    "critical": "red bold",
                    "high": "red",
                    "medium": "yellow",
                    "low": "cyan",
                    "info": "dim",
                }

                for finding in findings:
                    severity = finding.get("severity", "info")
                    style = severity_colors.get(severity, "")
                    table.add_row(
                        f"[{style}]{severity.upper()}[/{style}]",
                        finding.get("category", ""),
                        finding.get("title", ""),
                        finding.get("location", "-"),
                        f"{finding.get('confidence', 0):.0%}",
                    )

                self.ui.console.print(table)

                # Print details for critical/high findings
                critical_high = [f for f in findings if f.get("severity") in ("critical", "high")]
                if critical_high:
                    self.ui.console.print("\n[bold]Finding Details:[/bold]")
                    for finding in critical_high[:5]:  # Limit to top 5
                        self.ui.console.print(f"\n[bold]{finding.get('title')}[/bold]")
                        self.ui.console.print(f"  {finding.get('description', 'No description')}")
                        if finding.get("recommendation"):
                            self.ui.console.print(f"  [green]Recommendation:[/green] {finding.get('recommendation')}")
                        if finding.get("cwe_id"):
                            self.ui.console.print(f"  [dim]CWE: {finding.get('cwe_id')}[/dim]")
            else:
                self.ui.print_success("No security issues found!")
        else:
            # Plain text fallback
            print(f"\nRisk Score: {risk_score:.0f}/100")
            print(f"Critical: {summary.get('critical_count', 0)} | High: {summary.get('high_count', 0)}")
            print(f"\n{summary.get('text', '')}\n")

            if findings:
                print("Findings:")
                for finding in findings:
                    print(f"  [{finding.get('severity', 'INFO').upper()}] {finding.get('title')}")
                    print(f"    {finding.get('description', '')}")
            else:
                print("No security issues found!")

    async def _handle_search(self, command: ParsedCommand) -> None:
        """Handle semantic search command."""
        if not command.args:
            self.ui.print_error("Please specify a search query.")
            self.ui.print_info("Usage: search <query> [--type reports|findings|all]")
            return

        query = " ".join(command.args)
        search_type = command.flags.get("type", "all")
        top_k = int(command.flags.get("limit", "10"))

        self.ui.print_info(f"Searching for: {query}")

        try:
            from benderbox.nlp.rag import RAGPipeline
            from benderbox.storage.report_indexer import ReportIndexer
            from benderbox.storage.vector_store import VectorStore

            # Initialize RAG pipeline
            vector_store = VectorStore()
            await vector_store.initialize()

            report_indexer = ReportIndexer(vector_store=vector_store)

            rag = RAGPipeline(
                report_indexer=report_indexer,
                knowledge_base=getattr(self._conversation, "_knowledge_base", None) if self._conversation else None,
            )

            with ProgressSpinner(self.ui, "Searching..."):
                results = await rag.search(
                    query=query,
                    search_type=search_type,
                    top_k=top_k,
                )

            if results:
                self._print_search_results(results, query)
            else:
                self.ui.print_warning("No results found.")
                self.ui.print_info("Try running some analyses first to populate the search index.")

        except ImportError as e:
            self.ui.print_error(f"Search requires additional dependencies: {e}")
        except Exception as e:
            logger.error(f"Search error: {e}")
            self.ui.print_error(f"Search failed: {e}")

    def _print_search_results(self, results: List[Dict[str, Any]], query: str) -> None:
        """Print search results with formatting."""
        self.ui.print_header(f"Search Results for: {query}")

        if self.ui.console:
            from rich.table import Table
            from rich.panel import Panel

            table = Table(title=f"Found {len(results)} results", show_header=True)
            table.add_column("Type", style="bold", width=10)
            table.add_column("Score", width=8)
            table.add_column("Name/Target", width=25)
            table.add_column("Details", width=40)

            type_colors = {
                "report": "cyan",
                "finding": "yellow",
                "knowledge": "green",
            }

            for result in results:
                result_type = result.get("type", "unknown")
                score = result.get("score", 0)
                metadata = result.get("metadata", {})

                # Get name based on type
                if result_type == "report":
                    name = metadata.get("target_name", "Unknown")
                    details = f"{metadata.get('risk_level', '').upper()} ({metadata.get('risk_score', 0)}/100)"
                elif result_type == "finding":
                    name = metadata.get("test_name", "Unknown")
                    details = f"[{metadata.get('severity', 'info').upper()}] {metadata.get('category', '')}"
                else:
                    name = metadata.get("name", "Unknown")
                    details = metadata.get("category", "")

                color = type_colors.get(result_type, "white")
                table.add_row(
                    f"[{color}]{result_type}[/{color}]",
                    f"{score:.2f}",
                    name[:25],
                    details[:40],
                )

            self.ui.console.print(table)

            # Show top result content
            if results:
                top = results[0]
                content = top.get("content", "")[:300]
                if len(top.get("content", "")) > 300:
                    content += "..."
                self.ui.console.print(Panel(content, title="Top Result Preview", border_style="dim"))
        else:
            print(f"\nFound {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                result_type = result.get("type", "unknown")
                score = result.get("score", 0)
                content = result.get("content", "")[:100]
                print(f"{i}. [{result_type}] (score: {score:.2f})")
                print(f"   {content}...")
                print()

    async def _handle_compare(self, command: ParsedCommand) -> None:
        """Handle compare command."""
        if len(command.args) < 2:
            self.ui.print_error("Please specify two targets to compare.")
            self.ui.print_info("Usage: compare <target1> <target2>")
            return

        target1 = command.args[0]
        target2 = command.args[1]

        query = f"compare {target1} vs {target2}"

        if self._conversation:
            with ProgressSpinner(self.ui, "Comparing targets..."):
                response = await self._conversation.process_query(query)
            self.ui.print_markdown(response.content)
        else:
            self.ui.print_warning("Comparison requires conversation manager.")

    async def _handle_export(self, command: ParsedCommand) -> None:
        """Handle export command."""
        if not self._last_result:
            self.ui.print_warning("No analysis result to export. Run an analysis first.")
            return

        format_type = command.flags.get("format", "markdown")
        output = command.args[0] if command.args else None

        try:
            from benderbox.reporting import ExportManager, ExportFormat

            export_manager = ExportManager()
            fmt = ExportFormat[format_type.upper()]

            result = await export_manager.export_analysis(
                self._last_result,
                format=fmt,
                output_path=output,
            )

            if result.success:
                self.ui.print_success(f"Exported to: {result.path}")
            else:
                self.ui.print_error(f"Export failed: {result.error}")

        except Exception as e:
            self.ui.print_error(f"Export error: {e}")

    async def _handle_query(self, command: ParsedCommand) -> None:
        """Handle natural language query."""
        if not command.args or not command.args[0]:
            return

        query = command.args[0]

        # Check for off-topic queries and redirect to BenderBox focus
        off_topic_keywords = ["story", "poem", "joke", "recipe", "weather", "news", "game", "movie", "music"]
        query_lower = query.lower()
        if any(kw in query_lower for kw in off_topic_keywords):
            self.ui.print_info("I'm BenderBox, an AI security analysis assistant.")
            self.ui.print_info("I can help with:")
            self.ui.print_info("  - Model analysis and interrogation")
            self.ui.print_info("  - MCP server security testing")
            self.ui.print_info("  - Context/instruction file analysis")
            self.ui.print_info("  - Code security scanning")
            self.ui.print_info("Type 'help' to see available commands.")
            return

        if self._conversation:
            with ProgressSpinner(self.ui, "Thinking..."):
                response = await self._conversation.process_query(query)

            # Store result if analysis was performed
            if response.analysis_result:
                self._last_result = response.analysis_result

            # Display response
            self.ui.print_markdown(response.content)
        else:
            self.ui.print_warning("Natural language queries require conversation manager.")
            self.ui.print_info("Use explicit commands like 'analyze <path>'")

    # ========== MCP Commands ==========

    async def _handle_mcp(self, command: ParsedCommand) -> None:
        """Handle MCP command group - show subcommands or delegate."""
        if not command.args:
            self.ui.print_info("MCP Server Commands:")
            self.ui.print_info("  /mcp connect <target>       - Connect to MCP server")
            self.ui.print_info("  /mcp tools <target>         - List tools from MCP server")
            self.ui.print_info("  /mcp interrogate <target>   - Security test MCP server")
            self.ui.print_info("  /mcp call <target> <tool>   - Call a tool")
            self.ui.print_info("")
            self.ui.print_info("Examples:")
            self.ui.print_info("  /mcp tools https://github.com/org/mcp-server")
            self.ui.print_info("  /mcp interrogate https://mcp.example.com/api")
            return

        # Delegate to subcommand
        subcommand = command.args[0].lower()
        remaining_args = command.args[1:]
        new_command = ParsedCommand(
            command_type=CommandType.QUERY,  # Default
            args=remaining_args,
            raw_input=command.raw_input,
            flags=command.flags,
        )

        if subcommand in ("connect", "conn"):
            new_command.command_type = CommandType.MCP_CONNECT
            await self._handle_mcp_connect(new_command)
        elif subcommand in ("tools", "list"):
            new_command.command_type = CommandType.MCP_TOOLS
            await self._handle_mcp_tools(new_command)
        elif subcommand in ("interrogate", "test", "security"):
            new_command.command_type = CommandType.MCP_INTERROGATE
            await self._handle_mcp_interrogate(new_command)
        elif subcommand in ("call", "invoke"):
            new_command.command_type = CommandType.MCP_CALL
            await self._handle_mcp_call(new_command)
        else:
            self.ui.print_error(f"Unknown MCP subcommand: {subcommand}")
            self.ui.print_info("Use '/mcp' to see available subcommands.")

    async def _handle_mcp_connect(self, command: ParsedCommand) -> None:
        """Handle MCP connect command."""
        if not command.args:
            self.ui.print_error("Please specify a target.")
            self.ui.print_info("Usage: /mcp connect <url|command>")
            return

        target = command.args[0]
        transport = command.flags.get("transport", "auto")

        try:
            from benderbox.analyzers.mcp_client import MCPClient, MCPTransport

            self.ui.print_info(f"Connecting to: {target}")

            client = MCPClient()
            transport_enum = MCPTransport(transport)

            with ProgressSpinner(self.ui, "Connecting..."):
                connected = await client.connect(target, transport_enum)

            if connected:
                self.ui.print_success("Connected successfully!")
                tools = await client.list_tools()

                if tools:
                    self.ui.print_info(f"Available tools: {len(tools)}")
                    for tool in tools[:10]:
                        desc = tool.description[:40] + "..." if tool.description and len(tool.description) > 40 else tool.description or ""
                        self.ui.print_info(f"  - {tool.name}: {desc}")
                    if len(tools) > 10:
                        self.ui.print_info(f"  ... and {len(tools) - 10} more")
                else:
                    self.ui.print_warning("No tools discovered")

                await client.disconnect()
            else:
                self.ui.print_error("Connection failed")

        except ImportError as e:
            self.ui.print_error(f"Missing dependency: {e}")
        except Exception as e:
            self.ui.print_error(f"Connection error: {e}")

    async def _handle_mcp_tools(self, command: ParsedCommand) -> None:
        """Handle MCP tools listing command."""
        if not command.args:
            self.ui.print_error("Please specify a target.")
            self.ui.print_info("Usage: /mcp tools <github_url|mcp_endpoint>")
            return

        target = command.args[0]

        try:
            from benderbox.analyzers.mcp_analyzer import analyze_mcp_server

            self.ui.print_info(f"Discovering tools from: {target}")

            with ProgressSpinner(self.ui, "Analyzing..."):
                server_info = await analyze_mcp_server(target)

            if not server_info.tools:
                self.ui.print_warning("No tools discovered.")
                return

            self.ui.print_success(f"Found {len(server_info.tools)} tools:")

            # Sort by risk
            sorted_tools = sorted(server_info.tools, key=lambda t: t.risk_score, reverse=True)

            for tool in sorted_tools:
                risk_icon = {"critical": "[!]", "high": "[*]", "medium": "[~]", "low": "[ ]"}.get(tool.risk_level.value, "[ ]")
                desc = tool.description[:40] + "..." if tool.description and len(tool.description) > 40 else tool.description or ""
                self.ui.print_info(f"  {risk_icon} {tool.name} (risk: {tool.risk_score}) - {desc}")

            self.ui.print_info(f"\nOverall Risk: {server_info.overall_risk_level.value.upper()} ({server_info.overall_risk_score}/100)")

        except Exception as e:
            self.ui.print_error(f"Failed to list tools: {e}")

    async def _handle_mcp_interrogate(self, command: ParsedCommand) -> None:
        """Handle MCP security interrogation command."""
        if not command.args:
            self.ui.print_error("Please specify a target.")
            self.ui.print_info("Usage: /mcp interrogate <target> [--profile quick|full]")
            return

        target = command.args[0]
        profile = command.flags.get("profile", "full")
        transport = command.flags.get("transport", "auto")

        try:
            from benderbox.analyzers.mcp_interrogation import interrogate_mcp_server

            self.ui.print_info(f"Interrogating: {target}")
            self.ui.print_info(f"Profile: {profile}")

            def progress_cb(msg, pct):
                pass  # Could update spinner here

            with ProgressSpinner(self.ui, "Running security tests..."):
                score = await interrogate_mcp_server(
                    target=target,
                    transport=transport,
                    profile=profile,
                    progress_callback=progress_cb,
                )

            # Display results
            self.ui.print_header("MCP Interrogation Results")

            risk_color = "red" if score.risk_level in ("CRITICAL", "HIGH") else "yellow" if score.risk_level == "MEDIUM" else "green"
            self.ui.print_info(f"Risk Level: {score.risk_level}")
            self.ui.print_info(f"Risk Score: {score.overall_risk}/100")
            self.ui.print_info(f"Tools Tested: {score.tools_tested}")
            self.ui.print_info(f"Tests Run: {score.tests_run}")
            self.ui.print_info(f"Vulnerabilities Found: {score.vulnerabilities_found}")

            if score.critical_findings:
                self.ui.print_warning("\nCRITICAL FINDINGS:")
                for finding in score.critical_findings[:5]:
                    self.ui.print_error(f"  [{finding.tool_name}] {finding.description}")

            if score.high_findings:
                self.ui.print_warning("\nHIGH FINDINGS:")
                for finding in score.high_findings[:3]:
                    self.ui.print_warning(f"  [{finding.tool_name}] {finding.description}")

            if score.recommendations:
                self.ui.print_info("\nRecommendations:")
                for rec in score.recommendations[:5]:
                    self.ui.print_info(f"  - {rec}")

        except ImportError as e:
            self.ui.print_error(f"Missing dependency: {e}")
        except Exception as e:
            self.ui.print_error(f"Interrogation failed: {e}")

    async def _handle_mcp_call(self, command: ParsedCommand) -> None:
        """Handle MCP tool call command."""
        if len(command.args) < 2:
            self.ui.print_error("Please specify target and tool name.")
            self.ui.print_info("Usage: /mcp call <target> <tool_name> [--args '{\"key\": \"value\"}']")
            return

        target = command.args[0]
        tool_name = command.args[1]
        args_json = command.flags.get("args", "{}")

        import json

        try:
            args = json.loads(args_json)
        except json.JSONDecodeError as e:
            self.ui.print_error(f"Invalid JSON arguments: {e}")
            return

        try:
            from benderbox.analyzers.mcp_client import MCPClient, MCPTransport

            client = MCPClient()
            transport = command.flags.get("transport", "auto")

            with ProgressSpinner(self.ui, "Connecting..."):
                connected = await client.connect(target, MCPTransport(transport))

            if not connected:
                self.ui.print_error("Failed to connect")
                return

            self.ui.print_info(f"Calling {tool_name}...")

            result = await client.call_tool(tool_name, args)

            await client.disconnect()

            if result.is_error:
                self.ui.print_error(f"Tool error: {result.content}")
            else:
                self.ui.print_success("Tool executed successfully")
                self.ui.print_info("Response:")
                content = result.content[:1000] if result.content else "(empty)"
                print(content)
                if result.content and len(result.content) > 1000:
                    self.ui.print_info(f"... (truncated, {len(result.content)} total chars)")

        except Exception as e:
            self.ui.print_error(f"Call failed: {e}")

    # ========== Context Analysis Commands ==========

    async def _handle_context(self, command: ParsedCommand) -> None:
        """Handle context command group - show subcommands or delegate."""
        if not command.args:
            self.ui.print_info("Context Analysis Commands:")
            self.ui.print_info("  /context analyze <file>     - Analyze instruction file")
            self.ui.print_info("  /context scan <directory>   - Scan directory for issues")
            self.ui.print_info("  /context output <text>      - Analyze model output")
            self.ui.print_info("")
            self.ui.print_info("Examples:")
            self.ui.print_info("  /context analyze skills.md")
            self.ui.print_info("  /context scan ./prompts")
            return

        # Delegate to subcommand
        subcommand = command.args[0].lower()
        remaining_args = command.args[1:]
        new_command = ParsedCommand(
            command_type=CommandType.QUERY,
            args=remaining_args,
            raw_input=command.raw_input,
            flags=command.flags,
        )

        if subcommand in ("analyze", "check"):
            new_command.command_type = CommandType.CONTEXT_ANALYZE
            await self._handle_context_analyze(new_command)
        elif subcommand in ("scan", "directory"):
            new_command.command_type = CommandType.CONTEXT_SCAN
            await self._handle_context_scan(new_command)
        elif subcommand in ("output", "inference"):
            new_command.command_type = CommandType.CONTEXT_OUTPUT
            await self._handle_context_output(new_command)
        else:
            self.ui.print_error(f"Unknown context subcommand: {subcommand}")
            self.ui.print_info("Use '/context' to see available subcommands.")

    async def _handle_context_analyze(self, command: ParsedCommand) -> None:
        """Handle context file analysis command."""
        if not command.args:
            self.ui.print_error("Please specify a file to analyze.")
            self.ui.print_info("Usage: /context analyze <file> [--type skill|prompt|instruction]")
            return

        file_path = command.args[0]
        file_type = command.flags.get("type", "auto")

        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            self.ui.print_error(f"File not found: {file_path}")
            return

        try:
            from benderbox.analyzers.context_analyzer import ContextAnalyzer, ContextType

            analyzer = ContextAnalyzer()

            context_type = None
            if file_type != "auto":
                try:
                    context_type = ContextType(file_type)
                except ValueError:
                    pass

            self.ui.print_info(f"Analyzing: {file_path}")

            with ProgressSpinner(self.ui, "Analyzing..."):
                result = analyzer.analyze_file(path, context_type)

            # Display results
            self.ui.print_header("Context Analysis Results")
            self.ui.print_info(f"File: {result.file_path}")
            self.ui.print_info(f"Type: {result.file_type.value}")
            self.ui.print_info(f"Risk Level: {result.risk_level.value}")
            self.ui.print_info(f"Risk Score: {result.risk_score}/100")
            self.ui.print_info(f"Findings: {len(result.findings)}")

            if result.findings:
                # Group by severity
                critical = [f for f in result.findings if f.risk_level.value == "CRITICAL"]
                high = [f for f in result.findings if f.risk_level.value == "HIGH"]

                if critical:
                    self.ui.print_error("\nCRITICAL ISSUES:")
                    for f in critical[:5]:
                        self.ui.print_error(f"  Line {f.line_number}: {f.description}")
                        self.ui.print_info(f"    Match: {f.matched_text[:50]}...")

                if high:
                    self.ui.print_warning("\nHIGH SEVERITY ISSUES:")
                    for f in high[:3]:
                        self.ui.print_warning(f"  Line {f.line_number}: {f.description}")

            if result.recommendations:
                self.ui.print_info("\nRecommendations:")
                for rec in result.recommendations[:5]:
                    self.ui.print_info(f"  - {rec}")

        except ImportError as e:
            self.ui.print_error(f"Missing dependency: {e}")
        except Exception as e:
            self.ui.print_error(f"Analysis failed: {e}")

    async def _handle_context_scan(self, command: ParsedCommand) -> None:
        """Handle context directory scan command."""
        if not command.args:
            self.ui.print_error("Please specify a directory to scan.")
            self.ui.print_info("Usage: /context scan <directory> [--pattern '*.md']")
            return

        directory = command.args[0]

        from pathlib import Path

        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            self.ui.print_error(f"Directory not found: {directory}")
            return

        try:
            from benderbox.analyzers.context_analyzer import ContextAnalyzer

            analyzer = ContextAnalyzer()
            patterns = [command.flags.get("pattern")] if command.flags.get("pattern") else None

            self.ui.print_info(f"Scanning: {directory}")

            with ProgressSpinner(self.ui, "Scanning directory..."):
                results = analyzer.analyze_directory(dir_path, patterns)

            if not results:
                self.ui.print_warning("No matching files found.")
                return

            # Display summary
            self.ui.print_header(f"Scan Results ({len(results)} files)")

            # Count by risk level
            high_risk = sum(1 for r in results if r.risk_level.value in ("CRITICAL", "HIGH"))
            medium_risk = sum(1 for r in results if r.risk_level.value == "MEDIUM")
            total_findings = sum(len(r.findings) for r in results)

            self.ui.print_info(f"Total Files: {len(results)}")
            self.ui.print_info(f"Total Findings: {total_findings}")
            if high_risk > 0:
                self.ui.print_warning(f"High/Critical Risk Files: {high_risk}")
            if medium_risk > 0:
                self.ui.print_info(f"Medium Risk Files: {medium_risk}")

            # Show top risky files
            risky = [r for r in results if r.risk_score > 20]
            if risky:
                self.ui.print_warning("\nFiles with Issues:")
                for r in risky[:10]:
                    self.ui.print_info(f"  {Path(r.file_path).name}: {r.risk_level.value} ({r.risk_score}/100) - {len(r.findings)} findings")

        except Exception as e:
            self.ui.print_error(f"Scan failed: {e}")

    async def _handle_context_output(self, command: ParsedCommand) -> None:
        """Handle model output analysis command."""
        if not command.args:
            self.ui.print_error("Please provide output text to analyze.")
            self.ui.print_info("Usage: /context output \"model response text\"")
            self.ui.print_info("   or: /context output --file response.txt")
            return

        from pathlib import Path

        # Get content from args or file
        file_path = command.flags.get("file")
        if file_path:
            path = Path(file_path)
            if not path.exists():
                self.ui.print_error(f"File not found: {file_path}")
                return
            content = path.read_text(encoding="utf-8")
        else:
            content = " ".join(command.args)

        model_name = command.flags.get("model", "unknown")

        try:
            from benderbox.analyzers.context_analyzer import analyze_inference_output

            self.ui.print_info(f"Analyzing output ({len(content)} chars)")

            result = analyze_inference_output(content, model_name)

            # Display results
            self.ui.print_header("Output Analysis Results")
            self.ui.print_info(f"Risk Level: {result.risk_level.value}")
            self.ui.print_info(f"Risk Score: {result.risk_score}/100")
            self.ui.print_info(f"Findings: {len(result.findings)}")

            if result.findings:
                self.ui.print_warning("\nPotential Issues Detected:")
                for f in result.findings[:10]:
                    self.ui.print_warning(f"  [{f.risk_level.value}] {f.description}")
                    self.ui.print_info(f"    Match: {f.matched_text[:50]}...")
            else:
                self.ui.print_success("No harmful patterns detected")

        except Exception as e:
            self.ui.print_error(f"Analysis failed: {e}")

    async def run(self) -> None:
        """Run the interactive chat loop."""
        self._running = True

        # Print banner
        self.ui.print_banner()
        self.ui.print_info("Type 'help' for commands or ask a question.")

        # Check if LLM is available and show helpful message if not
        if self._conversation and self._conversation._llm_engine:
            if not getattr(self._conversation._llm_engine, 'is_available', False):
                print()
                self.ui.print_warning("NLP features limited: llama-cpp-python not installed.")
                self.ui.print_info("Commands like 'analyze', 'status', 'help' work fine.")
                self.ui.print_info("For full NLP: pip install llama-cpp-python")
        print()

        while self._running:
            try:
                # Get user input
                user_input = self.ui.input_prompt("You: ")

                if not user_input.strip():
                    continue

                # Parse and handle command
                command = self.parse_input(user_input)
                should_continue = await self.handle_command(command)

                if not should_continue:
                    break

                print()  # Spacing between interactions

            except KeyboardInterrupt:
                print()
                self.ui.print_info("Use 'exit' to quit.")

            except EOFError:
                break

            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                self.ui.print_error(str(e))

    def run_sync(self) -> None:
        """Run the chat loop synchronously."""
        asyncio.run(self.run())


class StreamingChatUI(ChatUI):
    """
    Chat UI with streaming response support.

    Displays responses as they are generated for a more
    interactive experience.
    """

    async def _handle_query(self, command: ParsedCommand) -> None:
        """Handle query with streaming output."""
        if not command.args or not command.args[0]:
            return

        query = command.args[0]

        if self._conversation:
            try:
                # Print response prefix
                if self.ui.console:
                    self.ui.console.print("[bold cyan]BenderBox:[/bold cyan] ", end="")
                else:
                    print("BenderBox: ", end="")

                # Stream response
                async for chunk in self._conversation.process_query_stream(query):
                    if self.ui.console:
                        self.ui.console.print(chunk, end="")
                    else:
                        print(chunk, end="", flush=True)

                print()  # Newline after response

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                # Fall back to non-streaming
                await super()._handle_query(command)
        else:
            await super()._handle_query(command)
