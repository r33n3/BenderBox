"""
Interactive Chat UI for BenderBox

Provides a conversational interface for security analysis
with command handling and context awareness.
"""

import asyncio
import glob
import logging
import os
import shlex
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from benderbox.ui.terminal import TerminalUI, ProgressSpinner, AnalysisProgressDisplay

logger = logging.getLogger(__name__)

# Tab completion support
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    try:
        import pyreadline3 as readline
        READLINE_AVAILABLE = True
    except ImportError:
        READLINE_AVAILABLE = False
        readline = None


class BenderBoxCompleter:
    """Tab completion for BenderBox interactive mode."""

    # Commands and subcommands
    COMMANDS = [
        "help", "help mcp", "help context", "help models", "help examples",
        "status", "exit", "quit", "clear",
        "analyze", "fingerprint", "fp", "interrogate", "compare",
        "mcp tools", "mcp interrogate", "mcp analyze", "mcp call",
        "context analyze", "context scan", "context output",
        "models", "models list", "models list --for analysis", "models list --for nlp",
        "load", "unload", "current",
        "nlp status", "nlp features",
        "search", "reports", "export",
    ]

    # Profile options
    PROFILES = ["--profile fingerprint", "--profile quick", "--profile standard", "--profile full", "--profile adversarial"]

    def __init__(self):
        self.matches: List[str] = []

    def complete(self, text: str, state: int) -> Optional[str]:
        """Return the next possible completion for 'text'."""
        if state == 0:
            # Get the full line buffer for context
            line = readline.get_line_buffer() if readline else text

            if not line.strip():
                # Empty line - show all commands
                self.matches = self.COMMANDS[:]
            elif any(line.startswith(cmd) for cmd in ["analyze ", "context analyze ", "mcp analyze "]):
                # File path completion
                self.matches = self._complete_path(text)
            elif "--profile" in line and text == "":
                # Profile value completion
                self.matches = ["fingerprint", "quick", "standard", "full"]
            elif text.startswith("--"):
                # Option completion
                self.matches = [opt for opt in self.PROFILES if opt.startswith(text)]
            else:
                # Command completion
                self.matches = [cmd for cmd in self.COMMANDS if cmd.startswith(line)]
                # Also complete partial paths
                if not self.matches:
                    self.matches = self._complete_path(text)

        try:
            return self.matches[state]
        except IndexError:
            return None

    def _complete_path(self, text: str) -> List[str]:
        """Complete file paths."""
        if not text:
            text = "./"

        # Expand ~ to home directory
        expanded = os.path.expanduser(text)

        # Get matching paths
        if os.path.isdir(expanded):
            pattern = os.path.join(expanded, "*")
        else:
            pattern = expanded + "*"

        paths = glob.glob(pattern)

        # Format results
        results = []
        for path in paths:
            if os.path.isdir(path):
                results.append(path + os.sep)
            else:
                results.append(path)

        return results[:20]  # Limit results


def setup_tab_completion() -> None:
    """Set up tab completion for the interactive mode."""
    if not READLINE_AVAILABLE:
        return

    completer = BenderBoxCompleter()
    readline.set_completer(completer.complete)

    # Set completion delimiters (don't break on these chars)
    readline.set_completer_delims(' \t\n;')

    # Enable tab completion
    if 'libedit' in readline.__doc__:
        # macOS uses libedit
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        # Linux/Windows
        readline.parse_and_bind("tab: complete")

    # Enable history
    try:
        from benderbox.config import get_benderbox_home
        history_file = str(get_benderbox_home() / "data" / "history")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        try:
            readline.read_history_file(history_file)
        except FileNotFoundError:
            pass
        import atexit
        atexit.register(readline.write_history_file, history_file)
    except Exception:
        pass  # History is optional


class CommandType(Enum):
    """Types of chat commands."""

    ANALYZE = "analyze"
    FINGERPRINT = "fingerprint"  # Static-only analysis (no model loading)
    SEMANTIC = "semantic"  # Semantic code analysis
    SEARCH = "search"  # Semantic search across reports
    COMPARE = "compare"
    STATUS = "status"
    REPORTS = "reports"
    EXPORT = "export"
    HELP = "help"
    CLEAR = "clear"
    EXIT = "exit"
    GREETING = "greeting"  # Hello, hi, hey, etc.
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
    # Model management commands
    MODELS = "models"  # List/manage models
    LOAD_MODEL = "load_model"
    CURRENT_MODEL = "current_model"
    UNLOAD_MODEL = "unload_model"


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
        "fingerprint": ["fingerprint", "fp", "static", "metadata"],  # Static-only analysis
        "semantic": ["semantic", "code", "review", "security-scan", "sec"],
        "search": ["search", "find", "lookup", "query-reports"],
        "compare": ["compare", "diff", "vs", "c"],
        "status": ["status", "info", "state", "s"],
        "reports": ["reports", "history", "r"],
        "export": ["export", "save", "e"],
        "help": ["help", "?", "h"],
        "clear": ["clear", "cls", "reset"],
        "exit": ["exit", "quit", "q", "bye", "goodbye", "close", "cya", "later"],
        "greeting": ["hello", "hi", "hey", "yo", "sup", "greetings", "howdy", "hola"],
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
        # Model commands
        "models": ["models", "model"],
        "load_model": ["load", "load-model", "switch"],
        "current_model": ["current", "current-model", "loaded"],
        "unload_model": ["unload", "unload-model"],
    }

    def __init__(
        self,
        conversation_manager=None,
        terminal_ui: Optional[TerminalUI] = None,
        model_manager=None,
    ):
        """
        Initialize ChatUI.

        Args:
            conversation_manager: ConversationManager for query processing.
            terminal_ui: TerminalUI for output.
            model_manager: ModelManager for model operations.
        """
        self._conversation = conversation_manager
        self.ui = terminal_ui or TerminalUI()
        self._model_manager = model_manager

        # Set model manager on conversation manager for command mapping
        if self._conversation and model_manager:
            self._conversation.set_model_manager(model_manager)

        # Command handlers
        self._handlers: Dict[CommandType, Callable] = {}

        # Session state
        self._running = False
        self._last_result: Optional[Dict[str, Any]] = None
        self._current_analysis_model: Optional[str] = None  # Path to loaded analysis model
        self._current_nlp_model: Optional[str] = None  # Path to loaded NLP/chat model

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

        elif command.command_type == CommandType.GREETING:
            await self._handle_greeting(command)

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

        elif command.command_type == CommandType.FINGERPRINT:
            await self._handle_fingerprint(command)

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

        # Model commands
        elif command.command_type == CommandType.MODELS:
            await self._handle_models(command)

        elif command.command_type == CommandType.LOAD_MODEL:
            await self._handle_load_model(command)

        elif command.command_type == CommandType.CURRENT_MODEL:
            await self._handle_current_model(command)

        elif command.command_type == CommandType.UNLOAD_MODEL:
            await self._handle_unload_model(command)

        return True

    async def _handle_exit(self, command: ParsedCommand) -> bool:
        """Handle exit command with Bender-style farewell."""
        from benderbox.nlp.persona import BenderPersona
        persona = BenderPersona()
        farewell = persona.get_farewell()
        self.ui.print_info(farewell)
        self._running = False
        return False

    async def _handle_greeting(self, command: ParsedCommand) -> None:
        """Handle greeting with Bender-style response."""
        from benderbox.nlp.persona import BenderPersona
        persona = BenderPersona()
        greeting = persona.get_greeting()
        self.ui.print_info(greeting)

    async def _handle_help(self, command: ParsedCommand) -> None:
        """Handle help command with optional category."""
        category = command.args[0] if command.args else None
        self.ui.print_help(category)

    async def _handle_clear(self, command: ParsedCommand) -> None:
        """Handle clear command."""
        if self._conversation:
            self._conversation.clear_history()
        self.ui.print_success("Conversation cleared.")

    async def _handle_status(self, command: ParsedCommand) -> None:
        """Handle status command with detailed model state."""
        from pathlib import Path
        from benderbox.utils.model_manager import ModelManager

        self.ui.print_header("BenderBox Status")
        self.ui.print_info("Version: 3.0.0-alpha")
        print()

        # Model status section
        self.ui.print_header("Loaded Models")

        # NLP Model
        if self._current_nlp_model:
            model_path = Path(self._current_nlp_model)
            size_mb = model_path.stat().st_size // (1024 * 1024) if model_path.exists() else 0
            self.ui.print_success(f"  NLP: {model_path.name} ({size_mb} MB) [LOADED]")
            self.ui.print_info(f"       Path: {self._current_nlp_model}")

            # Check if actually in memory
            llm_engine = self._get_llm_engine()
            if llm_engine and hasattr(llm_engine, 'is_nlp_model_loaded'):
                in_memory = llm_engine.is_nlp_model_loaded()
                self.ui.print_info(f"       In memory: {'Yes' if in_memory else 'No (will load on first use)'}")
        else:
            self.ui.print_warning("  NLP: Not loaded")
            self.ui.print_info("       Use '/load <model> --for nlp' to load")

        # Analysis Model
        if self._current_analysis_model:
            model_path = Path(self._current_analysis_model)
            size_mb = model_path.stat().st_size // (1024 * 1024) if model_path.exists() else 0
            self.ui.print_success(f"  Analysis: {model_path.name} ({size_mb} MB) [LOADED]")
            self.ui.print_info(f"       Path: {self._current_analysis_model}")
        else:
            self.ui.print_warning("  Analysis: Not loaded")
            self.ui.print_info("       Use '/load <model> --for analysis' to load")

        print()

        # Available models section
        manager = self._model_manager or ModelManager()
        all_models = manager.get_downloaded_models()
        nlp_models = manager.list_nlp_models()
        analysis_models = manager.list_analysis_models()

        self.ui.print_header("Available Models")
        self.ui.print_info(f"  Total downloaded: {len(all_models)}")
        self.ui.print_info(f"  NLP directory: {len(nlp_models)}")
        self.ui.print_info(f"  Analysis directory: {len(analysis_models)}")

        # Show default paths
        default_nlp = manager.get_default_model_path(purpose="nlp")
        default_analysis = manager.get_default_model_path(purpose="analysis")

        print()
        self.ui.print_header("Default Model Paths")
        if default_nlp:
            self.ui.print_info(f"  NLP default: {default_nlp.name}")
        else:
            self.ui.print_info(f"  NLP default: Not configured")

        if default_analysis:
            self.ui.print_info(f"  Analysis default: {default_analysis.name}")
        else:
            self.ui.print_info(f"  Analysis default: Not configured")

        print()

        # System Resources
        self.ui.print_header("System Resources")
        ram_gb = manager.get_system_ram_gb()
        available_ram_gb = max(ram_gb - 4, 2)
        self.ui.print_info(f"  RAM: {ram_gb}GB total, ~{available_ram_gb}GB available for models")

        gpu_info = manager.get_gpu_info()
        if gpu_info.get('has_nvidia'):
            vram = gpu_info.get('vram_gb', 'Unknown')
            gpu_name = gpu_info.get('gpu_name', 'NVIDIA GPU')
            self.ui.print_success(f"  GPU: {gpu_name} ({vram}GB VRAM)")
            if gpu_info.get('cuda_available'):
                self.ui.print_info(f"       CUDA available - GPU acceleration enabled")
            else:
                self.ui.print_info(f"       CUDA not configured - CPU only")
        else:
            self.ui.print_info("  GPU: No NVIDIA GPU detected (CPU only)")

        cpu_info = manager.get_cpu_info()
        self.ui.print_info(f"  CPU: {cpu_info.get('cores', '?')} cores, {cpu_info.get('threads', '?')} threads")

        print()

        # LLM Engine status
        self.ui.print_header("LLM Engine")
        llm_engine = self._get_llm_engine()
        if llm_engine:
            is_available = getattr(llm_engine, 'is_available', False)
            self.ui.print_info(f"  llama-cpp-python: {'Available' if is_available else 'Not installed'}")
            if hasattr(llm_engine, '_loaded_models'):
                loaded_count = len(llm_engine._loaded_models)
                self.ui.print_info(f"  Models in memory: {loaded_count}")
            if hasattr(llm_engine, 'config'):
                self.ui.print_info(f"  Max loaded models: {llm_engine.config.max_loaded_models}")
                self.ui.print_info(f"  Context length: {llm_engine.config.context_length}")
        else:
            self.ui.print_warning("  LLM Engine: Not initialized")

        # Conversation status
        print()
        self.ui.print_header("Conversation")
        if self._conversation:
            history_len = len(self._conversation._history) if hasattr(self._conversation, '_history') else 0
            self.ui.print_info(f"  History: {history_len} messages")
            self.ui.print_info("  Use '/clear' to reset conversation")

    async def _handle_reports(self, command: ParsedCommand) -> None:
        """Handle reports command."""
        if self._conversation:
            response = await self._conversation.process_query("show recent reports")
            self.ui.print_markdown(response.content)
        else:
            self.ui.print_warning("Report listing requires conversation manager.")

    async def _handle_analyze(self, command: ParsedCommand) -> None:
        """Handle analyze command."""
        # Valid profile names for shorthand detection
        VALID_PROFILES = ("fingerprint", "quick", "standard", "full", "adversarial")

        # Use loaded model if no target specified, OR if only a profile name is given
        # e.g., "analyze fingerprint" with a loaded model should use loaded model + fingerprint profile
        first_arg_is_profile = (
            len(command.args) == 1
            and command.args[0].lower() in VALID_PROFILES
            and self._current_analysis_model
        )

        if not command.args or first_arg_is_profile:
            if self._current_analysis_model:
                target = self._current_analysis_model
                self.ui.print_info(f"Analyzing loaded model: {Path(target).name}")
            else:
                self.ui.print_error("Please specify a target to analyze.")
                self.ui.print_info("Usage: analyze <path|url|model-name> [fingerprint|quick|standard|full]")
                self.ui.print_info("       fingerprint <model-name>  (static only, no model loading)")
                self.ui.print_info("Or load a model first: /load <model-name>")
                return
        else:
            target = command.args[0]
            # Check if target is a model name or filename (not a full path)
            # Allow both "phi2" and "phi-2.Q2_K.gguf" to be resolved via model manager
            is_full_path = "/" in target or "\\" in target or (len(target) > 2 and target[1] == ":")
            if not is_full_path:
                # First check if it matches the currently loaded analysis model
                if self._current_analysis_model:
                    loaded_name = Path(self._current_analysis_model).name
                    if target == loaded_name or target.lower() == loaded_name.lower():
                        target = self._current_analysis_model
                        self.ui.print_info(f"Using loaded analysis model: {Path(target).name}")
                    else:
                        # Try to find via model manager
                        from benderbox.utils.model_manager import ModelManager
                        manager = ModelManager()
                        model_path = manager.find_model_by_name(target, purpose="analysis")
                        if model_path:
                            target = str(model_path)
                            self.ui.print_info(f"Found model: {target}")
                else:
                    # No loaded model, try model manager
                    from benderbox.utils.model_manager import ModelManager
                    manager = ModelManager()
                    model_path = manager.find_model_by_name(target, purpose="analysis")
                    if model_path:
                        target = str(model_path)
                        self.ui.print_info(f"Found model: {target}")

        # Get profile from flags OR second positional argument
        # Supports multiple formats:
        #   "analyze phi2 --profile quick"  (explicit flag)
        #   "analyze phi2 quick"            (positional arg)
        #   "analyze phi2 fingerprint"      (positional arg - static only)
        #   "analyze --quick"               (shorthand flag)
        #   "fingerprint phi2"              (direct command)
        profile = command.flags.get("profile")

        # Check for shorthand flags (--fingerprint, --quick, --standard, --full, --adversarial)
        if not profile:
            for prof_name in VALID_PROFILES:
                if prof_name in command.flags:
                    profile = prof_name
                    break

        # Check if first arg is a valid profile name (when using loaded model)
        if not profile and first_arg_is_profile:
            profile = command.args[0].lower()

        # Check if second arg is a valid profile name
        if not profile and len(command.args) > 1:
            potential_profile = command.args[1].lower()
            if potential_profile in VALID_PROFILES:
                profile = potential_profile

        if not profile:
            profile = "standard"

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

        # Use direct analysis with progress display for better UX
        try:
            from benderbox.nlp.analysis_bridge import AnalysisBridge

            # Get or create the analysis bridge
            if self._conversation and hasattr(self._conversation, "_analysis_bridge"):
                bridge = self._conversation._analysis_bridge
            else:
                bridge = AnalysisBridge()

            # Run analysis with live progress display
            with AnalysisProgressDisplay(self.ui, title=f"Analyzing {Path(target).name}") as progress:
                result = await bridge.analyze_model(
                    model_path=target,
                    profile=profile,
                    progress_callback=progress.get_callback(),
                )

            self._last_result = result

            if result and not result.get("error"):
                self.ui.print_analysis_summary(result)

                # Print findings
                results = result.get("results", [])
                failed = [r for r in results if r.get("status") in ("failed", "warning")]
                if failed:
                    self.ui.print_findings_table(failed, "Issues Found")
            else:
                error_msg = result.get("error", "Analysis produced no results")
                self.ui.print_error(error_msg)

        except ImportError as e:
            # Fallback to conversation-based analysis if bridge not available
            logger.warning(f"AnalysisBridge import failed, falling back to conversation: {e}")
            if self._conversation:
                query = f"analyze {target} with {profile} profile"
                spinner_msg = f"Analyzing {target}..."
                with ProgressSpinner(self.ui, spinner_msg):
                    response = await self._conversation.process_query(query)

                self._last_result = response.analysis_result

                if response.analysis_result:
                    self.ui.print_analysis_summary(response.analysis_result)
                    results = response.analysis_result.get("results", [])
                    failed = [r for r in results if r.get("status") in ("failed", "warning")]
                    if failed:
                        self.ui.print_findings_table(failed, "Issues Found")
                else:
                    self.ui.print_markdown(response.content)
            else:
                self.ui.print_warning("Analysis requires conversation manager.")
        except Exception as e:
            error_str = str(e).lower()
            # Check if this is a model not found error
            if "not found" in error_str or "no such file" in error_str:
                self.ui.print_error(f"Model not found: {target}")
                self.ui.print_info("")
                self.ui.print_info("To find available models, try:")
                self.ui.print_info("  models list       - Show downloaded models")
                self.ui.print_info("  models available  - Show models that can be downloaded")
                self.ui.print_info("")
                # Try to suggest similar models
                try:
                    from benderbox.utils.model_manager import ModelManager
                    manager = ModelManager()
                    available = manager.list_local_models(purpose="analysis")
                    if available:
                        self.ui.print_info("Available local models:")
                        for model in available[:5]:  # Show up to 5
                            self.ui.print_info(f"  • {model.name}")
                        if len(available) > 5:
                            self.ui.print_info(f"  ... and {len(available) - 5} more")
                except Exception:
                    pass
            else:
                logger.exception(f"Analysis failed with {type(e).__name__}: {e}")
                self.ui.print_error(f"Analysis failed: {e}")

    async def _handle_fingerprint(self, command: ParsedCommand) -> None:
        """Handle fingerprint command - static-only model analysis (no model loading)."""
        # fingerprint phi2  ->  analyze phi2 fingerprint
        if not command.args:
            if self._current_analysis_model:
                target = self._current_analysis_model
                self.ui.print_info(f"Fingerprinting loaded model: {target}")
            else:
                self.ui.print_error("Please specify a model to fingerprint.")
                self.ui.print_info("Usage: fingerprint <model-name>")
                self.ui.print_info("       fp <model-name>")
                self.ui.print_info("")
                self.ui.print_info("This performs static-only analysis without loading the model.")
                return
        else:
            target = command.args[0]
            # Check if target is a model name or filename (not a full path)
            is_full_path = "/" in target or "\\" in target or (len(target) > 2 and target[1] == ":")
            if not is_full_path:
                # First check if it matches the currently loaded analysis model
                if self._current_analysis_model:
                    loaded_name = Path(self._current_analysis_model).name
                    if target == loaded_name or target.lower() == loaded_name.lower():
                        target = self._current_analysis_model
                        self.ui.print_info(f"Using loaded analysis model: {Path(target).name}")
                    else:
                        # Try to find via model manager
                        from benderbox.utils.model_manager import ModelManager
                        manager = ModelManager()
                        model_path = manager.find_model_by_name(target, purpose="analysis")
                        if model_path:
                            target = str(model_path)
                            self.ui.print_info(f"Found model: {target}")
                else:
                    # No loaded model, try model manager
                    from benderbox.utils.model_manager import ModelManager
                    manager = ModelManager()
                    model_path = manager.find_model_by_name(target, purpose="analysis")
                    if model_path:
                        target = str(model_path)
                        self.ui.print_info(f"Found model: {target}")

        # Create a new command with fingerprint profile and delegate to _handle_analyze
        fingerprint_command = ParsedCommand(
            command_type=CommandType.ANALYZE,
            args=[target, "fingerprint"],
            raw_input=command.raw_input,
            flags={"profile": "fingerprint"},
        )
        await self._handle_analyze(fingerprint_command)

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
        """Print semantic analysis result with neon-themed formatting."""
        # Check if this is a semantic analysis result
        if result.get("analysis_type") != "semantic":
            self.ui.print_analysis_summary(result)
            return

        summary = result.get("summary", {})
        findings = result.get("findings", [])

        # Print header
        self.ui.print_header(f"Semantic Analysis: {result.get('target', 'Unknown')}")

        # Print summary - use theme colors
        risk_score = summary.get("risk_score", 0)
        theme = TerminalUI.THEME
        risk_color = theme['danger'] if risk_score >= 70 else theme['warning'] if risk_score >= 40 else theme['success']

        if self.ui.console:
            from rich.table import Table
            from rich.panel import Panel

            # Summary panel with neon styling
            summary_text = f"""
Risk Score: [{risk_color}]{risk_score:.0f}/100[/{risk_color}]
Critical: {summary.get('critical_count', 0)} | High: {summary.get('high_count', 0)}
Analysis Depth: {result.get('analysis_depth', 'standard')}
LLM Used: {'Yes' if result.get('llm_used') else 'No (pattern-based)'}

{summary.get('text', 'No summary available.')}
"""
            self.ui.console.print(Panel(summary_text, title=f"[{theme['neon_orange']}]Summary[/{theme['neon_orange']}]", border_style=theme['neon_purple']))

            # Findings table with neon styling
            if findings:
                table = Table(title=f"[{theme['neon_orange']}]Security Findings[/{theme['neon_orange']}]", show_header=True, border_style=theme['neon_purple'], header_style=f"bold {theme['neon_purple']}")
                table.add_column("Severity", style="bold")
                table.add_column("Category", style=theme['neon_cyan'])
                table.add_column("Title")
                table.add_column("Location")
                table.add_column("Confidence")

                severity_colors = {
                    "critical": f"{theme['danger']} bold",
                    "high": theme['warning'],
                    "medium": "#ffff00",
                    "low": theme['neon_cyan'],
                    "info": theme['text_muted'],
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
                    self.ui.console.print(f"\n[bold {theme['neon_orange']}]Finding Details:[/bold {theme['neon_orange']}]")
                    for finding in critical_high[:5]:  # Limit to top 5
                        self.ui.console.print(f"\n[bold {theme['neon_cyan']}]{finding.get('title')}[/bold {theme['neon_cyan']}]")
                        self.ui.console.print(f"  {finding.get('description', 'No description')}")
                        if finding.get("recommendation"):
                            self.ui.console.print(f"  [{theme['success']}]Recommendation:[/{theme['success']}] {finding.get('recommendation')}")
                        if finding.get("cwe_id"):
                            self.ui.console.print(f"  [{theme['text_muted']}]CWE: {finding.get('cwe_id')}[/{theme['text_muted']}]")
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
        """Print search results with neon-themed formatting."""
        self.ui.print_header(f"Search Results for: {query}")

        if self.ui.console:
            from rich.table import Table
            from rich.panel import Panel

            theme = TerminalUI.THEME
            table = Table(
                title=f"[{theme['neon_orange']}]Found {len(results)} results[/{theme['neon_orange']}]",
                show_header=True,
                border_style=theme['neon_purple'],
                header_style=f"bold {theme['neon_purple']}"
            )
            table.add_column("Type", style="bold", width=10)
            table.add_column("Score", width=8, style=theme['neon_orange'])
            table.add_column("Name/Target", width=25, style=theme['text_primary'])
            table.add_column("Details", width=40, style=theme['neon_cyan'])

            type_colors = {
                "report": theme['neon_cyan'],
                "finding": theme['warning'],
                "knowledge": theme['success'],
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

                color = type_colors.get(result_type, theme['text_muted'])
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
                self.ui.console.print(Panel(content, title=f"[{theme['neon_cyan']}]Top Result Preview[/{theme['neon_cyan']}]", border_style=theme['text_muted']))
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

            # Save report to reports directory
            await self._save_mcp_tools_report(target, server_info)

        except Exception as e:
            self.ui.print_error(f"Failed to list tools: {e}")

    async def _save_mcp_tools_report(self, target: str, server_info) -> None:
        """Save MCP tools analysis report to reports directory."""
        import json
        from datetime import datetime
        from pathlib import Path
        from benderbox.config import get_config

        try:
            config = get_config()
            reports_path = Path(config.storage.reports_path)
            reports_path.mkdir(parents=True, exist_ok=True)

            # Generate base filename from target
            import re
            safe_name = re.sub(r'[^\w\-]', '_', target)[:50]
            base_filename = f"mcp_tools_{safe_name}"

            # Check for existing reports with same base name
            existing = list(reports_path.glob(f"{base_filename}*.json"))

            filename = f"{base_filename}.json"
            report_path = reports_path / filename

            if existing:
                # Found existing report(s) - ask user what to do
                latest_existing = max(existing, key=lambda p: p.stat().st_mtime)
                self.ui.print_warning(f"Existing report found: {latest_existing.name}")

                choice = await self._prompt_report_action(latest_existing.name)

                if choice == "overwrite":
                    # Use the existing filename (overwrite it)
                    report_path = latest_existing
                elif choice == "rename":
                    # Add number suffix
                    suffix = len(existing) + 1
                    filename = f"{base_filename}_{suffix}.json"
                    report_path = reports_path / filename
                elif choice == "skip":
                    self.ui.print_info("Report save skipped.")
                    return
                # "update" falls through to overwrite with new data

            # Build report in format compatible with report viewer
            report = {
                "schema_version": "1.0.0",
                "report_type": "mcp_tools",
                "analysis_type": "mcp_server",
                "timestamp": datetime.now().isoformat(),
                "target": target,
                "target_name": server_info.name or target,
                "target_type": "mcp_server",
                "server": {
                    "name": server_info.name or target,
                    "description": server_info.description or "",
                    "url": target,
                },
                "risk_assessment": {
                    "overall_level": server_info.overall_risk_level.value,
                    "overall_score": server_info.overall_risk_score,
                },
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "risk_level": tool.risk_level.value,
                        "risk_score": tool.risk_score,
                        "capabilities": [c.value if hasattr(c, 'value') else str(c) for c in (tool.capabilities or [])],
                        "risk_factors": tool.risk_factors or [],
                        "input_schema": tool.input_schema or {},
                    }
                    for tool in server_info.tools
                ],
                "summary": {
                    "total_tools": len(server_info.tools),
                    "critical_tools": len([t for t in server_info.tools if t.risk_level.value == "critical"]),
                    "high_tools": len([t for t in server_info.tools if t.risk_level.value == "high"]),
                    "medium_tools": len([t for t in server_info.tools if t.risk_level.value == "medium"]),
                    "low_tools": len([t for t in server_info.tools if t.risk_level.value == "low"]),
                },
            }

            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.ui.print_info(f"Report saved: {report_path}")

        except Exception as e:
            self.ui.print_warning(f"Failed to save report: {e}")

    async def _prompt_report_action(self, existing_name: str) -> str:
        """Prompt user for action when duplicate report exists."""
        self.ui.print_info("What would you like to do?")
        self.ui.print_info("  [1] Overwrite existing report")
        self.ui.print_info("  [2] Save as new (rename with number)")
        self.ui.print_info("  [3] Skip saving")

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("Choice [1/2/3]: ").strip()
            )
            if response == "1":
                return "overwrite"
            elif response == "2":
                return "rename"
            elif response == "3":
                return "skip"
            else:
                # Default to rename for safety
                return "rename"
        except (EOFError, KeyboardInterrupt):
            return "skip"

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

    # ========== Model Management Commands ==========

    async def _handle_models(self, command: ParsedCommand) -> None:
        """Handle models command - list available models."""
        from benderbox.utils.model_manager import ModelManager

        manager = ModelManager()

        # Check for subcommand
        if command.args:
            subcommand = command.args[0].lower()
            if subcommand in ("list", "ls"):
                purpose = command.flags.get("for", "all")
                await self._show_model_list(manager, purpose)
                return
            elif subcommand in ("add",):
                if len(command.args) < 2:
                    self.ui.print_error("Please specify a model path.")
                    self.ui.print_info("Usage: /models add <path> --for analysis|nlp")
                    return
                # Redirect to CLI for add functionality
                self.ui.print_info("Use CLI for adding models:")
                self.ui.print_info(f"  benderbox models add {command.args[1]} --for analysis")
                return
            elif subcommand in ("default", "set-default"):
                await self._handle_set_default_model(command, manager)
                return

        # Default: show models with current loaded status
        await self._show_model_list(manager, "all")

        # Show current loaded models
        from pathlib import Path
        loaded_info = []
        if self._current_nlp_model:
            model_name = Path(self._current_nlp_model).stem
            loaded_info.append(f"NLP: {model_name}")
        if self._current_analysis_model:
            model_name = Path(self._current_analysis_model).stem
            loaded_info.append(f"Analysis: {model_name}")

        if loaded_info:
            self.ui.print_success(f"\nCurrently loaded: {', '.join(loaded_info)}")
        else:
            self.ui.print_info("\nNo models loaded. Use '/load <model> --for nlp|analysis' to load one.")

    async def _show_model_list(self, manager, purpose: str) -> None:
        """Show model list based on purpose with RAM requirements."""
        from pathlib import Path

        self.ui.print_header("Available Models")

        # Get system info for warnings
        system_ram_gb = manager.get_system_ram_gb()
        available_ram_gb = max(system_ram_gb - 4, 2)
        self.ui.print_info(f"  System RAM: {system_ram_gb}GB (Available: ~{available_ram_gb}GB)")
        print()

        # Helper to check if a model is loaded (compare by filename)
        def is_loaded_for(model_path: str, loaded_path: str | None) -> bool:
            if not loaded_path:
                return False
            return Path(model_path).name == Path(loaded_path).name

        # Get all downloaded models from all locations
        all_models = manager.get_downloaded_models()

        if all_models:
            for m in all_models:
                # Get detailed requirements
                req = manager.get_model_requirements(m['path'])

                # Check if loaded for either purpose (by filename)
                markers = []
                if is_loaded_for(m['path'], self._current_analysis_model):
                    markers.append("ANALYSIS")
                if is_loaded_for(m['path'], self._current_nlp_model):
                    markers.append("NLP")

                # Build display line
                param_str = f" ({req['param_display']})" if req.get('param_billions') else ""
                ram_str = f"~{req['ram_requirements']['estimated_ram_gb']}GB RAM"
                marker = f" [LOADED: {', '.join(markers)}]" if markers else ""

                # Color based on whether it can load
                if not req.get('can_load', True):
                    self.ui.print_error(f"  - {m['name']}{param_str} ({m['size_mb']} MB) {ram_str}{marker}")
                    for warning in req.get('warnings', []):
                        self.ui.print_warning(f"      ! {warning}")
                elif req.get('warnings'):
                    self.ui.print_warning(f"  - {m['name']}{param_str} ({m['size_mb']} MB) {ram_str}{marker}")
                    for warning in req.get('warnings', []):
                        self.ui.print_info(f"      * {warning}")
                else:
                    self.ui.print_success(f"  - {m['name']}{param_str} ({m['size_mb']} MB) {ram_str}{marker}")

            self.ui.print_info("\n  Use '/load <name> --for nlp' or '/load <name> --for analysis'")
        else:
            self.ui.print_warning("  No models found.")
            self.ui.print_info("  Download with: benderbox models download tinyllama")

    async def _handle_set_default_model(self, command: ParsedCommand, manager) -> None:
        """Handle setting a default model for a purpose."""
        from pathlib import Path

        # Show current defaults if no model specified
        if len(command.args) < 2:
            default_nlp = manager.get_default_model_path(purpose="nlp")
            default_analysis = manager.get_default_model_path(purpose="analysis")
            self.ui.print_header("Default Models")
            if default_nlp:
                self.ui.print_success(f"  NLP: {default_nlp.name}")
            else:
                self.ui.print_warning(f"  NLP: Not configured")
            if default_analysis:
                self.ui.print_success(f"  Analysis: {default_analysis.name}")
            else:
                self.ui.print_warning(f"  Analysis: Not configured")
            print()
            self.ui.print_info("To set a default: /models default <model-name> --for nlp|analysis")
            return

        model_name = command.args[1]
        purpose = command.flags.get("for", "nlp").lower()  # Default to NLP

        if purpose not in ("nlp", "analysis"):
            self.ui.print_error(f"Invalid purpose: {purpose}. Use 'nlp' or 'analysis'.")
            return

        # Find the model
        model_path = self._find_model_anywhere(manager, model_name)

        if not model_path:
            self.ui.print_error(f"Model '{model_name}' not found.")
            all_models = manager.get_downloaded_models()
            if all_models:
                self.ui.print_info("Available models:")
                for m in all_models[:5]:
                    self.ui.print_info(f"  - {m['name']}")
            return

        # Set as default
        success, message = manager.setup_default_model(model_path, purpose=purpose)
        if success:
            self.ui.print_success(f"Set {model_path.name} as default {purpose.upper()} model.")
            self.ui.print_info(f"This model will auto-load on startup.")
        else:
            self.ui.print_error(f"Failed to set default: {message}")

    async def _handle_load_model(self, command: ParsedCommand) -> None:
        """Handle load model command."""
        if not command.args:
            self.ui.print_error("Please specify a model name to load.")
            self.ui.print_info("Usage: /load <model-name> [--for nlp|analysis]")
            self.ui.print_info("Example: /load tinyllama --for nlp")
            self.ui.print_info("Example: /load phi-2 --for analysis")
            self.ui.print_info("Use '/models' to see available models.")
            return

        model_name = command.args[0]
        purpose = command.flags.get("for", "analysis").lower()

        if purpose not in ("nlp", "analysis"):
            self.ui.print_error(f"Invalid purpose: {purpose}. Use 'nlp' or 'analysis'.")
            return

        from benderbox.utils.model_manager import ModelManager
        from pathlib import Path

        manager = ModelManager()

        # Check if it's a direct path (contains path separators)
        if os.path.sep in model_name or model_name.startswith("."):
            model_path = Path(model_name)
            if model_path.exists():
                resolved_path = str(model_path.resolve())
                await self._load_model_for_purpose(resolved_path, model_path.name, purpose)
                return
            else:
                self.ui.print_error(f"Model file not found: {model_name}")
                return

        # Search for model in all locations (by name, even if it ends with .gguf)
        model_path = self._find_model_anywhere(manager, model_name)

        if model_path:
            resolved_path = str(model_path)
            await self._load_model_for_purpose(resolved_path, model_path.name, purpose)
        else:
            # Show available models
            all_models = manager.get_downloaded_models()
            if all_models:
                self.ui.print_error(f"Model '{model_name}' not found.")
                self.ui.print_info("Available models:")
                for m in all_models[:5]:
                    self.ui.print_info(f"  - {m['name']}")
                if len(all_models) > 5:
                    self.ui.print_info(f"  ... and {len(all_models) - 5} more (use /models to see all)")
            else:
                self.ui.print_error(f"Model '{model_name}' not found.")
                self.ui.print_info("Use '/models' to see available models.")

    async def _load_model_for_purpose(self, resolved_path: str, model_name: str, purpose: str) -> None:
        """Load a model for NLP or analysis purpose with pre-load checks."""
        from benderbox.utils.model_manager import ModelManager

        manager = self._model_manager or ModelManager()

        # Check model requirements before loading
        req = manager.get_model_requirements(resolved_path)

        if req.get('error'):
            self.ui.print_error(req['error'])
            return

        # Show model info
        param_str = f" ({req['param_display']})" if req.get('param_billions') else ""
        self.ui.print_info(f"Model: {model_name}{param_str}")
        self.ui.print_info(f"Size: {req['file_size_mb']} MB")
        self.ui.print_info(f"Estimated RAM: ~{req['ram_requirements']['estimated_ram_gb']}GB")

        # Show warnings
        if req.get('warnings'):
            print()
            for warning in req['warnings']:
                self.ui.print_warning(f"Warning: {warning}")

        # Block loading if model definitely won't fit
        if not req.get('can_load', True):
            print()
            self.ui.print_error("Model cannot be loaded - exceeds available system RAM.")
            self.ui.print_info(f"System RAM: {req['system_ram_gb']}GB, Available: ~{req['available_ram_gb']}GB")
            self.ui.print_info("Options:")
            self.ui.print_info("  - Use a smaller quantized version (Q2_K, Q3_K)")
            self.ui.print_info("  - Use a smaller model (TinyLlama, Phi-2)")
            self.ui.print_info("  - Increase system RAM or use GPU offloading")
            return

        print()

        if purpose == "nlp":
            # Actually load the model into the LLM engine for chat
            llm_engine = self._get_llm_engine()
            if llm_engine:
                self.ui.print_info(f"Loading {model_name} into memory for chat...")
                success = await llm_engine.set_nlp_model(resolved_path)
                if success:
                    self._current_nlp_model = resolved_path
                    self.ui.print_success(f"NLP model loaded: {model_name}")
                    self.ui.print_info("You can now chat with this model. Try asking a question!")
                else:
                    self.ui.print_error(f"Failed to load model into memory.")
                    self.ui.print_info("Check that llama-cpp-python is installed: pip install llama-cpp-python")
                    return
            else:
                # No LLM engine available, just track the path
                self._current_nlp_model = resolved_path
                self.ui.print_success(f"NLP model set: {model_name}")
                self.ui.print_warning("Note: llama-cpp-python not available, chat responses will be limited.")
        else:
            self._current_analysis_model = resolved_path
            self.ui.print_success(f"Analysis model loaded: {model_name}")
            self.ui.print_info("This model is now the target for analysis/interrogation.")

        self._sync_loaded_models()

    def _get_llm_engine(self):
        """Get the LLM engine from the conversation manager."""
        if self._conversation and hasattr(self._conversation, '_llm_engine'):
            return self._conversation._llm_engine
        return None

    def _find_model_anywhere(self, manager, name: str) -> Optional[Path]:
        """Find a model by name in any location."""
        from pathlib import Path

        name_lower = name.lower().strip()

        # 1. Check purpose-specific folders first
        for purpose in ("analysis", "nlp"):
            model_path = manager.find_model_by_name(name, purpose=purpose)
            if model_path:
                return model_path

        # 2. Search all downloaded models
        all_models = manager.get_downloaded_models()
        matches = []

        for m in all_models:
            model_name = Path(m['path']).stem.lower()
            filename = Path(m['path']).name.lower()

            # Exact stem match
            if model_name == name_lower:
                return Path(m['path'])

            # Exact filename match
            if filename == name_lower or filename == name_lower + ".gguf":
                return Path(m['path'])

            # Fuzzy match (name contained in filename)
            if name_lower in model_name:
                matches.append(m)

        if len(matches) == 1:
            return Path(matches[0]['path'])
        elif len(matches) > 1:
            # Return shortest match (most specific)
            return Path(min(matches, key=lambda m: len(Path(m['path']).stem))['path'])

        return None

    async def _handle_current_model(self, command: ParsedCommand) -> None:
        """Handle current model command - show loaded models."""
        from pathlib import Path

        has_models = False

        if self._current_nlp_model:
            has_models = True
            model_path = Path(self._current_nlp_model)
            model_name = model_path.stem
            size_mb = model_path.stat().st_size // (1024 * 1024) if model_path.exists() else "?"

            self.ui.print_header("Current NLP/Chat Model")
            self.ui.print_info(f"Name: {model_name}")
            self.ui.print_info(f"Path: {model_path}")
            self.ui.print_info(f"Size: {size_mb} MB")
            self.ui.print_info("Purpose: Powers BenderBox chat responses")

        if self._current_analysis_model:
            has_models = True
            model_path = Path(self._current_analysis_model)
            model_name = model_path.stem
            size_mb = model_path.stat().st_size // (1024 * 1024) if model_path.exists() else "?"

            self.ui.print_header("Current Analysis Model")
            self.ui.print_info(f"Name: {model_name}")
            self.ui.print_info(f"Path: {model_path}")
            self.ui.print_info(f"Size: {size_mb} MB")
            self.ui.print_info("Purpose: Target for analysis/interrogation")

        if has_models:
            self.ui.print_info("\nCommands:")
            self.ui.print_info("  /unload nlp      - Unload NLP model")
            self.ui.print_info("  /unload analysis - Unload analysis model")
            self.ui.print_info("  /unload all      - Unload all models")
            self.ui.print_info("  /load <name> --for nlp|analysis")
        else:
            self.ui.print_warning("No models currently loaded.")
            self.ui.print_info("Use '/load <model-name> --for nlp' for chat model")
            self.ui.print_info("Use '/load <model-name> --for analysis' for analysis target")
            self.ui.print_info("Use '/models' to see available models.")

    async def _handle_unload_model(self, command: ParsedCommand) -> None:
        """Handle unload model command."""
        from pathlib import Path

        # Determine which model to unload
        target = command.args[0].lower() if command.args else "all"

        if target not in ("nlp", "analysis", "all"):
            self.ui.print_error(f"Invalid target: {target}. Use 'nlp', 'analysis', or 'all'.")
            return

        unloaded = []

        if target in ("nlp", "all") and self._current_nlp_model:
            model_name = Path(self._current_nlp_model).stem
            # Actually unload from LLM engine
            llm_engine = self._get_llm_engine()
            if llm_engine and hasattr(llm_engine, 'unload_nlp_model'):
                await llm_engine.unload_nlp_model()
            self._current_nlp_model = None
            unloaded.append(f"NLP model: {model_name}")

        if target in ("analysis", "all") and self._current_analysis_model:
            model_name = Path(self._current_analysis_model).stem
            self._current_analysis_model = None
            unloaded.append(f"Analysis model: {model_name}")

        if unloaded:
            for msg in unloaded:
                self.ui.print_success(f"Unloaded {msg}")
            self._sync_loaded_models()
        else:
            if target == "all":
                self.ui.print_info("No models currently loaded.")
            else:
                self.ui.print_info(f"No {target} model currently loaded.")

    def _sync_loaded_models(self) -> None:
        """Sync currently loaded models to ConversationManager."""
        if self._conversation:
            self._conversation.set_loaded_models(
                nlp_model=self._current_nlp_model,
                analysis_model=self._current_analysis_model,
            )

    async def _show_startup_model_status(self) -> None:
        """Show model availability and status at startup."""
        from benderbox.utils.model_manager import ModelManager
        from pathlib import Path

        manager = self._model_manager or ModelManager()

        # Get available models
        all_models = manager.get_downloaded_models()
        nlp_models = manager.list_nlp_models()
        analysis_models = manager.list_analysis_models()

        print()
        self.ui.print_header("Model Status")

        if not all_models:
            self.ui.print_warning("  No models found.")
            self.ui.print_info("  Download with: benderbox models download tinyllama")
            return

        # Show NLP models
        if nlp_models:
            self.ui.print_info(f"  NLP models: {len(nlp_models)} available")
            default_nlp = manager.get_default_model_path(purpose="nlp")
            if default_nlp:
                self.ui.print_success(f"    Default: {default_nlp.name} (will auto-load)")
        else:
            self.ui.print_info("  NLP models: None configured")
            self.ui.print_info("    Use: benderbox models add <model> --for nlp")

        # Show Analysis models
        if analysis_models:
            self.ui.print_info(f"  Analysis models: {len(analysis_models)} available")
            default_analysis = manager.get_default_model_path(purpose="analysis")
            if default_analysis:
                self.ui.print_info(f"    Default: {default_analysis.name}")
        else:
            self.ui.print_info("  Analysis models: None configured")

        # Show total available
        if all_models and not nlp_models and not analysis_models:
            self.ui.print_info(f"  Downloaded models: {len(all_models)}")
            self.ui.print_info("  Use '/models' to see all, '/load <model> --for nlp' to load")

    async def _first_run_nlp_model_picker(self) -> Optional[Path]:
        """
        Show model selection prompt on first run when no default NLP model exists.

        Returns:
            Path to selected model, or None if user skips.
        """
        from benderbox.utils.model_manager import ModelManager, RECOMMENDED_MODELS
        from pathlib import Path

        manager = self._model_manager or ModelManager()

        # Get all available models
        all_models = manager.get_all_models()
        if not all_models:
            # No models downloaded yet - offer to download
            print()
            self.ui.print_header("First-Time Setup: NLP Model")
            self.ui.print_info("No models found. BenderBox needs a model for chat features.")
            print()

            # Show recommended models based on system RAM
            ram_gb = manager.get_system_ram_gb()
            recommendations = manager.get_recommended_models(max_ram_gb=ram_gb)

            if recommendations:
                self.ui.print_info("Recommended models for your system:")
                for i, model in enumerate(recommendations[:3], 1):
                    size_str = f"{model.size_mb}MB" if model.size_mb < 1024 else f"{model.size_mb/1024:.1f}GB"
                    self.ui.print_info(f"  {i}. {model.name} ({size_str}) - {model.description}")

                print()
                self.ui.print_info("Download a model with: benderbox models download tinyllama")
                self.ui.print_info("Or press Enter to skip and use template-based responses.")

            return None

        print()
        self.ui.print_header("First-Time Setup: Select Default NLP Model")
        self.ui.print_info("Choose which model to use for chat/NLP features.")
        self.ui.print_info("This will be saved as your default and auto-loaded on startup.")
        print()

        # Show available models with numbers
        valid_models = []
        for i, model in enumerate(all_models[:10], 1):  # Limit to 10
            size_str = f"{model['size_mb']}MB" if model['size_mb'] < 1024 else f"{model['size_mb']/1024:.1f}GB"
            location = model.get('location', 'unknown')
            self.ui.print_info(f"  {i}. {model['name']} ({size_str}) [{location}]")
            valid_models.append(model)

        if len(all_models) > 10:
            self.ui.print_info(f"  ... and {len(all_models) - 10} more (use '/models' to see all)")

        print()
        self.ui.print_info("  0. Skip (use template responses, no LLM)")
        print()

        # Get user selection
        while True:
            try:
                choice = self.ui.input_prompt("Select model [1]: ").strip()
                if not choice:
                    choice = "1"  # Default to first model

                choice_num = int(choice)

                if choice_num == 0:
                    self.ui.print_info("Skipping model selection. Use '/load <model> --for nlp' to load later.")
                    return None

                if 1 <= choice_num <= len(valid_models):
                    selected = valid_models[choice_num - 1]
                    selected_path = Path(selected['path'])

                    # Set as default
                    self.ui.print_info(f"Setting {selected['name']} as default NLP model...")
                    success, msg = manager.setup_default_model(selected_path, purpose="nlp")
                    if success:
                        self.ui.print_success(f"Default NLP model set: {selected['name']}")
                        return manager.get_default_model_path(purpose="nlp")
                    else:
                        self.ui.print_warning(f"Could not set default: {msg}")
                        return selected_path

                self.ui.print_warning(f"Please enter a number between 0 and {len(valid_models)}")

            except ValueError:
                self.ui.print_warning("Please enter a valid number")
            except KeyboardInterrupt:
                print()
                self.ui.print_info("Skipped.")
                return None

    def _has_default_nlp_model(self) -> bool:
        """Check if a default NLP model is configured."""
        from benderbox.utils.model_manager import ModelManager

        manager = self._model_manager or ModelManager()
        default_nlp = manager.get_default_model_path(purpose="nlp")
        return default_nlp is not None and default_nlp.exists()

    async def _auto_load_default_nlp_model(self) -> None:
        """Auto-load default NLP model if available, or prompt for first-time setup."""
        from benderbox.utils.model_manager import ModelManager
        from pathlib import Path

        manager = self._model_manager or ModelManager()

        # Check for default NLP model
        default_nlp = manager.get_default_model_path(purpose="nlp")

        # If no default configured, run first-time picker
        if not default_nlp or not default_nlp.exists():
            nlp_models = manager.list_nlp_models()
            if not nlp_models:
                # Check if there are ANY models available for first-run picker
                all_models = manager.get_all_models()
                if all_models:
                    selected = await self._first_run_nlp_model_picker()
                    if selected:
                        default_nlp = selected
                    else:
                        return  # User skipped
                else:
                    return  # No models at all
            else:
                # Models in nlp dir but no default - use first one
                default_nlp = Path(nlp_models[0]["path"])

        # Get LLM engine
        llm_engine = self._get_llm_engine()
        if not llm_engine:
            return

        # Load the model (this can take a few seconds)
        self.ui.print_info(f"  Loading NLP model: {default_nlp.name}...")
        # Flush to ensure message displays before blocking model load
        if self.ui.console:
            self.ui.console.file.flush()
        try:
            success = await llm_engine.set_nlp_model(str(default_nlp))
            if success:
                self._current_nlp_model = str(default_nlp)
                self._sync_loaded_models()
                self.ui.print_success(f"  NLP model ready: {default_nlp.name}")
            else:
                self.ui.print_warning(f"  Could not load NLP model (llama-cpp-python may not be installed)")
        except Exception as e:
            logger.warning(f"Auto-load NLP model failed: {e}")
            self.ui.print_warning(f"  NLP model load failed: {e}")

    async def run(self) -> None:
        """Run the interactive chat loop."""
        self._running = True

        # Set up tab completion
        setup_tab_completion()

        # Print banner
        self.ui.print_banner()
        self.ui.print_info("Type 'help' for commands or ask a question.")
        if READLINE_AVAILABLE:
            self.ui.print_info("Tab completion enabled for commands and file paths.")
        else:
            self.ui.print_info("Tip: pip install pyreadline3 for tab completion (Windows)")

        # Check if LLM is available and show helpful message if not
        llm_available = False
        if self._conversation and self._conversation._llm_engine:
            llm_available = getattr(self._conversation._llm_engine, 'is_available', False)
            if not llm_available:
                print()
                self.ui.print_warning("NLP features limited: llama-cpp-python not installed.")
                self.ui.print_info("Commands like 'analyze', 'status', 'help' work fine.")
                self.ui.print_info("For full NLP: pip install llama-cpp-python")

        # Show model status at startup
        await self._show_startup_model_status()

        # Auto-load default NLP model if available
        if llm_available:
            await self._auto_load_default_nlp_model()

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
    interactive experience with neon-themed styling.
    """

    async def _handle_query(self, command: ParsedCommand) -> None:
        """Handle query with streaming output."""
        if not command.args or not command.args[0]:
            return

        query = command.args[0]

        if self._conversation:
            try:
                # Print response prefix with neon green
                theme = TerminalUI.THEME
                if self.ui.console:
                    self.ui.console.print(f"[bold {theme['neon_green']}]BenderBox:[/bold {theme['neon_green']}] ", end="")
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
