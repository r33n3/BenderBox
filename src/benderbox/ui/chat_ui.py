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
    COMPARE = "compare"
    STATUS = "status"
    REPORTS = "reports"
    EXPORT = "export"
    HELP = "help"
    CLEAR = "clear"
    EXIT = "exit"
    QUERY = "query"  # Natural language query


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
        "compare": ["compare", "diff", "vs", "c"],
        "status": ["status", "info", "state", "s"],
        "reports": ["reports", "list", "history", "r"],
        "export": ["export", "save", "e"],
        "help": ["help", "?", "h"],
        "clear": ["clear", "cls", "reset"],
        "exit": ["exit", "quit", "q", "bye"],
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
        if not args_str:
            return [], {}

        try:
            parts = shlex.split(args_str)
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

        elif command.command_type == CommandType.COMPARE:
            await self._handle_compare(command)

        elif command.command_type == CommandType.EXPORT:
            await self._handle_export(command)

        elif command.command_type == CommandType.QUERY:
            await self._handle_query(command)

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

    async def run(self) -> None:
        """Run the interactive chat loop."""
        self._running = True

        # Print banner
        self.ui.print_banner()
        self.ui.print_info("Type 'help' for commands or ask a question.")
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
