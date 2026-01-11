"""
Terminal UI Components for BenderBox

Rich-based terminal interface components for beautiful console output.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Path to ASCII art file
ASCII_ART_FILE = Path(__file__).parent.parent.parent.parent / "BenderBox_ASCII.txt"

# Enable UTF-8 mode on Windows for proper Unicode support
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8 code page
        kernel32.SetConsoleCP(65001)
        # Reconfigure stdout/stderr for UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Try to import Rich, gracefully degrade if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.style import Style
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


class TerminalUI:
    """
    Rich-based terminal UI for BenderBox.

    Provides beautiful console output with:
    - Styled panels and tables
    - Progress indicators
    - Syntax highlighting
    - Risk-level color coding
    """

    # Color scheme for risk levels
    RISK_COLORS = {
        "critical": "bold red",
        "high": "bold orange1",
        "medium": "bold yellow",
        "low": "bold green",
        "info": "bold blue",
        "safe": "bold green",
        "unknown": "bold white",
    }

    RISK_EMOJI = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
        "info": "ℹ️",
        "safe": "✅",
        "unknown": "⚪",
    }

    def __init__(self, console: Optional["Console"] = None):
        """
        Initialize TerminalUI.

        Args:
            console: Optional Rich Console instance.
        """
        if RICH_AVAILABLE:
            # Force UTF-8 and disable legacy Windows renderer for proper Unicode
            self.console = console or Console(force_terminal=True, legacy_windows=False)
        else:
            self.console = None
            logger.warning("Rich not available. Using basic output.")

    def print_banner(self) -> None:
        """Print the BenderBox banner with ASCII art."""
        # Try to load ASCII art from file
        banner = self._load_ascii_art()

        if self.console:
            self.console.print(Panel(
                Text(banner, style="bold cyan"),
                title="[bold white]AI Security Analysis Platform[/bold white]",
                subtitle="[dim]v3.0.0-alpha[/dim]",
                border_style="cyan",
            ))
        else:
            print(banner)
            print("BenderBox - AI Security Analysis Platform v3.0.0-alpha")
            print("=" * 60)

    def _load_ascii_art(self) -> str:
        """Load ASCII art from file or return fallback."""
        try:
            if ASCII_ART_FILE.exists():
                return ASCII_ART_FILE.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not load ASCII art: {e}")

        # Fallback banner if file not found
        return """
██████╗ ███████╗███╗   ██╗██████╗ ███████╗██████╗ ██████╗  ██████╗ ██╗  ██╗
██╔══██╗██╔════╝████╗  ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔═══██╗╚██╗██╔╝
██████╔╝█████╗  ██╔██╗ ██║██║  ██║█████╗  ██████╔╝██████╔╝██║   ██║ ╚███╔╝
██╔══██╗██╔══╝  ██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗██╔══██╗██║   ██║ ██╔██╗
██████╔╝███████╗██║ ╚████║██████╔╝███████╗██║  ██║██████╔╝╚██████╔╝██╔╝ ██╗
╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝
        """

    def print_risk_badge(
        self,
        risk_level: str,
        risk_score: float,
        large: bool = False,
    ) -> None:
        """
        Print a risk level badge.

        Args:
            risk_level: Risk level string.
            risk_score: Numeric risk score (0-100).
            large: Use large format.
        """
        risk_level = risk_level.lower()
        color = self.RISK_COLORS.get(risk_level, "white")
        emoji = self.RISK_EMOJI.get(risk_level, "")

        if self.console:
            if large:
                self.console.print(Panel(
                    f"{emoji} [bold]{risk_level.upper()}[/bold]\n[dim]Score: {risk_score}/100[/dim]",
                    style=color,
                    width=30,
                ))
            else:
                self.console.print(
                    f"{emoji} Risk: [{color}]{risk_level.upper()}[/{color}] ({risk_score}/100)"
                )
        else:
            print(f"{emoji} Risk: {risk_level.upper()} ({risk_score}/100)")

    def print_analysis_summary(self, result: Dict[str, Any]) -> None:
        """
        Print a formatted analysis summary.

        Args:
            result: Analysis result dictionary.
        """
        target = result.get("target_name", "Unknown")
        target_type = result.get("target_type", "unknown")
        profile = result.get("profile", "standard")
        summary = result.get("summary", {})
        risk = summary.get("risk", {})
        risk_level = risk.get("level", "unknown")
        risk_score = risk.get("score", 0)
        results = result.get("results", [])

        if self.console:
            # Create summary table
            table = Table(
                title=f"Analysis Summary: {target}",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
            )

            table.add_column("Property", style="bold")
            table.add_column("Value")

            table.add_row("Target", target)
            table.add_row("Type", target_type)
            table.add_row("Profile", profile)
            table.add_row(
                "Risk Level",
                Text(f"{self.RISK_EMOJI.get(risk_level.lower(), '')} {risk_level.upper()}",
                     style=self.RISK_COLORS.get(risk_level.lower(), "white"))
            )
            table.add_row("Risk Score", f"{risk_score}/100")
            table.add_row("Total Tests", str(len(results)))

            # Count by status
            passed = sum(1 for r in results if r.get("status") == "passed")
            failed = sum(1 for r in results if r.get("status") == "failed")
            warnings = sum(1 for r in results if r.get("status") == "warning")

            table.add_row("Passed", Text(str(passed), style="green"))
            table.add_row("Failed", Text(str(failed), style="red"))
            table.add_row("Warnings", Text(str(warnings), style="yellow"))

            self.console.print(table)
        else:
            print(f"\nAnalysis Summary: {target}")
            print(f"Type: {target_type} | Profile: {profile}")
            print(f"Risk: {risk_level.upper()} ({risk_score}/100)")
            print(f"Tests: {len(results)} total")

    def print_findings_table(
        self,
        findings: List[Dict[str, Any]],
        title: str = "Findings",
        max_items: int = 20,
    ) -> None:
        """
        Print a table of findings.

        Args:
            findings: List of finding dictionaries.
            title: Table title.
            max_items: Maximum items to display.
        """
        if not findings:
            if self.console:
                self.console.print("[green]No findings to display.[/green]")
            else:
                print("No findings to display.")
            return

        if self.console:
            table = Table(
                title=title,
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
            )

            table.add_column("Severity", width=10)
            table.add_column("Test", width=30)
            table.add_column("Category", width=15)
            table.add_column("Status", width=10)

            for finding in findings[:max_items]:
                severity = finding.get("severity", "info").lower()
                test_name = finding.get("test_name", "Unknown")[:28]
                category = finding.get("category", "N/A")[:13]
                status = finding.get("status", "unknown")

                severity_text = Text(
                    f"{self.RISK_EMOJI.get(severity, '')} {severity.upper()}",
                    style=self.RISK_COLORS.get(severity, "white")
                )

                status_style = {
                    "passed": "green",
                    "failed": "red",
                    "warning": "yellow",
                    "error": "red",
                }.get(status, "white")

                table.add_row(
                    severity_text,
                    test_name,
                    category,
                    Text(status.upper(), style=status_style),
                )

            self.console.print(table)

            if len(findings) > max_items:
                self.console.print(f"[dim]... and {len(findings) - max_items} more findings[/dim]")
        else:
            print(f"\n{title}")
            print("-" * 60)
            for finding in findings[:max_items]:
                severity = finding.get("severity", "info").upper()
                test_name = finding.get("test_name", "Unknown")
                status = finding.get("status", "unknown").upper()
                print(f"[{severity}] {test_name} - {status}")

    def print_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Print model information panel.

        Args:
            model_info: Model information dictionary.
        """
        if self.console:
            metadata = model_info.get("metadata", {})

            tree = Tree("[bold]Model Information[/bold]")
            tree.add(f"Name: {model_info.get('name', 'Unknown')}")
            tree.add(f"Path: {model_info.get('path', 'N/A')}")
            tree.add(f"Size: {model_info.get('size_bytes', 0) / (1024*1024):.1f} MB")

            if metadata:
                meta_branch = tree.add("[bold]Metadata[/bold]")
                if metadata.get("architecture"):
                    meta_branch.add(f"Architecture: {metadata['architecture']}")
                if metadata.get("parameter_count"):
                    meta_branch.add(f"Parameters: {metadata['parameter_count']}")
                if metadata.get("quantization"):
                    meta_branch.add(f"Quantization: {metadata['quantization']}")
                if metadata.get("context_length"):
                    meta_branch.add(f"Context: {metadata['context_length']}")

            self.console.print(Panel(tree, border_style="blue"))
        else:
            print("\nModel Information:")
            print(f"  Name: {model_info.get('name', 'Unknown')}")
            print(f"  Path: {model_info.get('path', 'N/A')}")

    def print_help(self) -> None:
        """Print help information."""
        help_text = """
## Available Commands

| Command | Description |
|---------|-------------|
| `analyze <path>` | Analyze a model or file |
| `compare <path1> <path2>` | Compare two targets |
| `status` | Show system status |
| `reports` | List recent reports |
| `help` | Show this help |
| `clear` | Clear conversation |
| `exit` | Exit BenderBox |

## Analysis Examples

- "Analyze model.gguf for security issues"
- "Check server.py with deep profile"
- "Is this model safe for production?"
- "What jailbreak techniques exist?"

## Tips

- Use natural language to describe what you want
- Reference previous analyses with "it" or "the model"
- Ask for explanations: "Why is the risk high?"
"""
        if self.console:
            self.console.print(Markdown(help_text))
        else:
            print(help_text)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        if self.console:
            self.console.print(f"[bold red]Error:[/bold red] {message}")
        else:
            print(f"Error: {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        if self.console:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
        else:
            print(f"Warning: {message}")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        if self.console:
            self.console.print(f"[bold green]✓[/bold green] {message}")
        else:
            print(f"✓ {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        if self.console:
            self.console.print(f"[bold blue]ℹ[/bold blue] {message}")
        else:
            print(f"ℹ {message}")

    def print_markdown(self, content: str) -> None:
        """
        Print markdown content.

        Args:
            content: Markdown string.
        """
        if self.console:
            self.console.print(Markdown(content))
        else:
            print(content)

    def print_code(self, code: str, language: str = "python") -> None:
        """
        Print syntax-highlighted code.

        Args:
            code: Source code string.
            language: Programming language.
        """
        if self.console:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            print(f"```{language}")
            print(code)
            print("```")

    def create_progress(self) -> Optional["Progress"]:
        """
        Create a progress bar context manager.

        Returns:
            Rich Progress instance or None.
        """
        if self.console and RICH_AVAILABLE:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )
        return None

    def input_prompt(self, prompt: str = "You: ") -> str:
        """
        Get user input with styled prompt.

        Args:
            prompt: Prompt string.

        Returns:
            User input string.
        """
        if self.console:
            return self.console.input(f"[bold cyan]{prompt}[/bold cyan]")
        return input(prompt)


class ProgressSpinner:
    """Context manager for progress spinner."""

    def __init__(self, ui: TerminalUI, message: str = "Processing..."):
        """
        Initialize spinner.

        Args:
            ui: TerminalUI instance.
            message: Progress message.
        """
        self.ui = ui
        self.message = message
        self._live = None
        self._spinner = None

    def __enter__(self):
        if self.ui.console and RICH_AVAILABLE:
            from rich.spinner import Spinner
            self._spinner = Spinner("dots", text=self.message)
            self._live = Live(self._spinner, console=self.ui.console, refresh_per_second=10)
            self._live.__enter__()
        else:
            print(f"{self.message}...", end="", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        else:
            print(" done." if not exc_type else " failed.")

    def update(self, message: str) -> None:
        """Update spinner message."""
        self.message = message
        if self._spinner:
            self._spinner.text = message
