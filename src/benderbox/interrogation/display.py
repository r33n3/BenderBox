"""
Display Handler for BenderBox Interrogation

Manages output display based on selected mode (silent, normal, verbose, stream, interactive).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from .prompts.schema import TestPrompt
    from .analyzer.results import AnalysisResult
    from .judge import JudgeVerdict

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Display modes for interrogation output."""

    SILENT = "silent"  # Final summary only (CI/CD)
    NORMAL = "normal"  # Progress bar + summary
    VERBOSE = "verbose"  # Show each prompt/response/analysis
    STREAM = "stream"  # Real-time streaming responses
    INTERACTIVE = "interactive"  # Pause after each test for user control


@dataclass
class DisplayConfig:
    """Configuration for display behavior."""

    mode: DisplayMode = DisplayMode.NORMAL
    show_prompts: bool = False
    show_responses: bool = False
    show_analysis: bool = False
    pause_between_tests: bool = False
    stream_responses: bool = False
    max_response_display: int = 500  # Max chars to display for responses
    color_enabled: bool = True


class InteractiveChoice(Enum):
    """Choices available in interactive mode."""

    CONTINUE = "c"  # Proceed to next test
    FOLLOWUP = "f"  # Send judge's suggested follow-up probe
    MANUAL = "m"  # Type custom prompt to send
    REPEAT = "r"  # Run same test again
    SKIP = "s"  # Skip to next category
    DETAILS = "d"  # Show full analysis breakdown
    EXPORT = "e"  # Save this exchange to file
    ABORT = "a"  # Stop and show partial results


class SkipCategoryException(Exception):
    """Raised when user wants to skip the current category."""

    def __init__(self, category: str):
        self.category = category
        super().__init__(f"Skipping category: {category}")


class AbortInterrogationException(Exception):
    """Raised when user wants to abort the interrogation."""
    pass


class DisplayHandler:
    """
    Handles output display based on mode.

    Supports silent, normal, verbose, stream, and interactive modes
    with Rich library integration for beautiful terminal output.
    """

    def __init__(
        self,
        config: Optional[DisplayConfig] = None,
        console: Optional["Console"] = None,
        input_callback: Optional[Callable[[], Awaitable[str]]] = None,
    ):
        """
        Initialize DisplayHandler.

        Args:
            config: Display configuration.
            console: Rich Console instance (created if not provided).
            input_callback: Async callback for getting user input in interactive mode.
        """
        self.config = config or DisplayConfig()

        if RICH_AVAILABLE:
            self.console = console or Console()
        else:
            self.console = None

        self.input_callback = input_callback

        # Track current test info
        self._current_test_num = 0
        self._total_tests = 0
        self._current_category = ""

        # Track session statistics
        self._passed_count = 0
        self._failed_count = 0
        self._skipped_categories: set = set()
        self._category_counts: Dict[str, Dict[str, int]] = {}  # category -> {total, passed, failed}
        self._has_judge = False
        self._followup_count = 0

        # Store last test info for context
        self._last_prompt: Optional["TestPrompt"] = None
        self._last_response: str = ""
        self._last_verdict: Optional["JudgeVerdict"] = None
        self._last_heuristic: Optional["AnalysisResult"] = None

    def _print(self, *args, **kwargs) -> None:
        """Print with Rich console if available, else plain print."""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            # Convert Rich markup to plain text
            text = " ".join(str(arg) for arg in args)
            # Strip basic Rich markup
            import re
            text = re.sub(r'\[/?[^\]]+\]', '', text)
            print(text)

    def set_judge_enabled(self, enabled: bool) -> None:
        """Set whether judge analysis is enabled."""
        self._has_judge = enabled

    def update_stats(self, passed: bool, category: str) -> None:
        """Update session statistics after a test."""
        if passed:
            self._passed_count += 1
        else:
            self._failed_count += 1

        # Update category counts
        if category not in self._category_counts:
            self._category_counts[category] = {"total": 0, "passed": 0, "failed": 0}
        self._category_counts[category]["total"] += 1
        if passed:
            self._category_counts[category]["passed"] += 1
        else:
            self._category_counts[category]["failed"] += 1

    def is_category_skipped(self, category: str) -> bool:
        """Check if a category has been skipped."""
        return category in self._skipped_categories

    def skip_category(self, category: str) -> None:
        """Mark a category as skipped."""
        self._skipped_categories.add(category)

    def increment_followup_count(self) -> None:
        """Increment the follow-up counter."""
        self._followup_count += 1

    async def show_test_start(
        self,
        test_num: int,
        total: int,
        prompt: "TestPrompt",
    ) -> None:
        """
        Display test header.

        Args:
            test_num: Current test number (1-indexed).
            total: Total number of tests.
            prompt: The test prompt being run.
        """
        self._current_test_num = test_num
        self._total_tests = total
        self._current_category = prompt.category.value
        self._last_prompt = prompt

        if self.config.mode == DisplayMode.SILENT:
            return

        if self.config.mode in (DisplayMode.VERBOSE, DisplayMode.INTERACTIVE, DisplayMode.STREAM):
            if RICH_AVAILABLE and self.console:
                self.console.print()

                # Build status line with session stats
                stats_line = f"[dim]Session: {self._passed_count}✓ {self._failed_count}✗"
                if self._followup_count > 0:
                    stats_line += f" | {self._followup_count} follow-ups"
                stats_line += "[/dim]"

                # Check if this is a new category
                category_info = f"Category: {prompt.category.value}"
                if prompt.category.value in self._category_counts:
                    cat_stats = self._category_counts[prompt.category.value]
                    category_info += f" ({cat_stats['passed']}✓ {cat_stats['failed']}✗ in category)"

                self.console.print(Panel(
                    f"[bold]Test {test_num}/{total}:[/bold] {prompt.id}\n"
                    f"[dim]{category_info} | Severity: {prompt.severity.value}[/dim]\n"
                    f"{stats_line}",
                    title="[cyan]Interrogation[/cyan]",
                    border_style="cyan",
                ))

                if self.config.show_prompts:
                    # Truncate long prompts for display
                    prompt_text = prompt.prompt
                    if len(prompt_text) > 500:
                        prompt_text = prompt_text[:500] + "\n... [truncated]"
                    self.console.print(Panel(
                        prompt_text,
                        title="[yellow]Prompt[/yellow]",
                        border_style="yellow",
                    ))
            else:
                print(f"\n--- Test {test_num}/{total}: {prompt.id} ---")
                print(f"Category: {prompt.category.value}")
                print(f"Session: {self._passed_count} passed, {self._failed_count} failed")
                if self.config.show_prompts:
                    print(f"Prompt: {prompt.prompt[:200]}...")

    async def show_response(
        self,
        response: str,
        stream: bool = False,
    ) -> None:
        """
        Display model response.

        Args:
            response: The model's response text.
            stream: Whether this is being streamed (currently unused).
        """
        self._last_response = response

        if self.config.mode == DisplayMode.SILENT:
            return

        if not self.config.show_responses:
            return

        # Truncate long responses for display
        display_response = response
        truncated = False
        if len(response) > self.config.max_response_display:
            display_response = response[:self.config.max_response_display]
            truncated = True

        if RICH_AVAILABLE and self.console:
            suffix = f"\n[dim]... [{len(response)} total chars][/dim]" if truncated else ""
            self.console.print(Panel(
                display_response + suffix,
                title="[green]Response[/green]",
                border_style="green",
            ))
        else:
            suffix = f"\n... [{len(response)} total chars]" if truncated else ""
            print(f"Response: {display_response}{suffix}")

    async def show_analysis(
        self,
        verdict: Optional["JudgeVerdict"],
        heuristic: Optional["AnalysisResult"],
    ) -> None:
        """
        Display analysis results.

        Args:
            verdict: Judge verdict (if judge was used).
            heuristic: Heuristic analysis result.
        """
        self._last_verdict = verdict
        self._last_heuristic = heuristic

        if self.config.mode == DisplayMode.SILENT:
            return

        if not (self.config.show_analysis or self.config.mode == DisplayMode.VERBOSE):
            return

        # Show heuristic result
        if heuristic:
            status_color = "green" if heuristic.passed else "red"
            status_text = "PASSED" if heuristic.passed else "FAILED"

            if RICH_AVAILABLE and self.console:
                self.console.print(
                    f"Heuristic: [{status_color}]{status_text}[/{status_color}] "
                    f"- {heuristic.classification.value} "
                    f"[dim](score: {heuristic.score})[/dim]"
                )
            else:
                print(f"Heuristic: {status_text} - {heuristic.classification.value} (score: {heuristic.score})")

        # Show judge verdict
        if verdict and verdict.classification != "error":
            passed = verdict.classification == "refused"
            status_color = "green" if passed else "red"
            status_text = "PASSED" if passed else "FAILED"

            if RICH_AVAILABLE and self.console:
                self.console.print(
                    f"Judge: [{status_color}]{status_text}[/{status_color}] "
                    f"- {verdict.classification} "
                    f"[dim](confidence: {verdict.confidence:.0%})[/dim]"
                )
                self.console.print(f"[dim]Reasoning: {verdict.reasoning}[/dim]")

                if verdict.suggested_followup:
                    self.console.print(
                        f"[yellow]Suggested follow-up:[/yellow] "
                        f"{verdict.suggested_followup[:100]}{'...' if len(verdict.suggested_followup) > 100 else ''}"
                    )
            else:
                print(f"Judge: {status_text} - {verdict.classification} (confidence: {verdict.confidence:.0%})")
                print(f"Reasoning: {verdict.reasoning}")

    async def show_detailed_analysis(
        self,
        heuristic: Optional["AnalysisResult"],
        verdict: Optional["JudgeVerdict"],
    ) -> None:
        """
        Show full detailed analysis breakdown.

        Args:
            heuristic: Heuristic analysis result.
            verdict: Judge verdict.
        """
        if RICH_AVAILABLE and self.console:
            table = Table(title="Detailed Analysis", show_header=True)
            table.add_column("Aspect", style="cyan")
            table.add_column("Value", style="white")

            if heuristic:
                table.add_row("Classification", heuristic.classification.value)
                table.add_row("Passed", "Yes" if heuristic.passed else "No")
                table.add_row("Score", str(heuristic.score))
                table.add_row("Severity", heuristic.severity)
                if heuristic.refusal:
                    table.add_row("Refusal Detected", "Yes" if heuristic.refusal.refused else "No")
                    table.add_row("Refusal Confidence", f"{heuristic.refusal.confidence:.0%}")
                if heuristic.harm:
                    table.add_row("Harm Score", f"{heuristic.harm.score:.0%}")
                    table.add_row("Harm Categories", ", ".join(heuristic.harm.categories) or "None")

            if verdict:
                table.add_row("---", "---")
                table.add_row("Judge Classification", verdict.classification)
                table.add_row("Judge Confidence", f"{verdict.confidence:.0%}")
                table.add_row("Judge Reasoning", verdict.reasoning)
                table.add_row("Harm Detected", "Yes" if verdict.harm_detected else "No")
                if verdict.harm_categories:
                    table.add_row("Harm Categories", ", ".join(verdict.harm_categories))
                if verdict.suggested_followup:
                    table.add_row("Suggested Follow-up", verdict.suggested_followup[:100])

            self.console.print(table)
        else:
            print("\n=== Detailed Analysis ===")
            if heuristic:
                print(f"Classification: {heuristic.classification.value}")
                print(f"Passed: {heuristic.passed}")
                print(f"Score: {heuristic.score}")
            if verdict:
                print(f"Judge: {verdict.classification} ({verdict.confidence:.0%})")
                print(f"Reasoning: {verdict.reasoning}")

    async def interactive_menu(self) -> InteractiveChoice:
        """
        Show interactive menu and get user choice.

        Returns:
            The user's choice from InteractiveChoice enum.
        """
        if self.config.mode != DisplayMode.INTERACTIVE:
            return InteractiveChoice.CONTINUE

        # Build contextual menu based on current state
        has_followup = self._has_judge and self._last_verdict and self._last_verdict.suggested_followup

        if RICH_AVAILABLE and self.console:
            self.console.print()

            # Build menu with visual indicators
            menu_items = []
            menu_items.append("[bold green][c][/bold green]ontinue")

            if has_followup:
                menu_items.append("[bold yellow][f][/bold yellow]ollowup*")
            else:
                menu_items.append("[dim][f]ollowup[/dim]")

            menu_items.append("[bold cyan][m][/bold cyan]anual")
            menu_items.append("[bold blue][r][/bold blue]epeat")
            menu_items.append("[bold magenta][s][/bold magenta]kip category")
            menu_items.append("[bold white][d][/bold white]etails")
            menu_items.append("[bold white][e][/bold white]xport")
            menu_items.append("[bold red][a][/bold red]bort")

            self.console.print("  ".join(menu_items))

            # Show hint for follow-up if available
            if has_followup:
                self.console.print("[dim]* Judge suggested a follow-up probe[/dim]")

            self.console.print("[dim]Press Enter or 'c' to continue, '?' for help[/dim]")
        else:
            print("\n[c]ontinue  [f]ollowup  [m]anual  [r]epeat  [s]kip  [d]etails  [e]xport  [a]bort")
            print("Press Enter to continue, '?' for help")

        while True:
            # Get user input
            if self.input_callback:
                choice_str = await self.input_callback()
            else:
                loop = asyncio.get_event_loop()
                choice_str = await loop.run_in_executor(
                    None, lambda: input("> ").strip().lower()
                )

            # Handle help request
            if choice_str == "?":
                await self._show_interactive_help()
                continue

            # Map to enum
            choice_map = {
                "c": InteractiveChoice.CONTINUE,
                "": InteractiveChoice.CONTINUE,
                "f": InteractiveChoice.FOLLOWUP,
                "m": InteractiveChoice.MANUAL,
                "r": InteractiveChoice.REPEAT,
                "s": InteractiveChoice.SKIP,
                "d": InteractiveChoice.DETAILS,
                "e": InteractiveChoice.EXPORT,
                "a": InteractiveChoice.ABORT,
                # Allow full words too
                "continue": InteractiveChoice.CONTINUE,
                "followup": InteractiveChoice.FOLLOWUP,
                "follow-up": InteractiveChoice.FOLLOWUP,
                "manual": InteractiveChoice.MANUAL,
                "repeat": InteractiveChoice.REPEAT,
                "skip": InteractiveChoice.SKIP,
                "details": InteractiveChoice.DETAILS,
                "export": InteractiveChoice.EXPORT,
                "abort": InteractiveChoice.ABORT,
                "quit": InteractiveChoice.ABORT,
                "q": InteractiveChoice.ABORT,
            }

            choice = choice_map.get(choice_str)

            if choice is None:
                # Invalid input
                if RICH_AVAILABLE and self.console:
                    self.console.print(f"[red]Unknown command: '{choice_str}'[/red] - press '?' for help")
                else:
                    print(f"Unknown command: '{choice_str}' - press '?' for help")
                continue

            # Handle special cases
            if choice == InteractiveChoice.FOLLOWUP and not has_followup:
                if RICH_AVAILABLE and self.console:
                    self.console.print("[yellow]No follow-up suggested. Use [m]anual to enter a custom prompt.[/yellow]")
                else:
                    print("No follow-up suggested. Use 'm' to enter a custom prompt.")
                continue

            if choice == InteractiveChoice.ABORT:
                # Confirm abort
                if await self._confirm_abort():
                    return choice
                continue

            return choice

    async def _show_interactive_help(self) -> None:
        """Display help for interactive mode."""
        if RICH_AVAILABLE and self.console:
            help_table = Table(title="Interactive Mode Commands", show_header=True)
            help_table.add_column("Key", style="cyan", width=12)
            help_table.add_column("Action", style="white")
            help_table.add_column("Description", style="dim")

            help_table.add_row("c / Enter", "Continue", "Proceed to the next test")
            help_table.add_row("f", "Follow-up", "Send judge's suggested follow-up probe (if available)")
            help_table.add_row("m", "Manual", "Type a custom prompt to send to the model")
            help_table.add_row("r", "Repeat", "Run the same test again")
            help_table.add_row("s", "Skip", "Skip all remaining tests in this category")
            help_table.add_row("d", "Details", "Show full analysis breakdown")
            help_table.add_row("e", "Export", "Save this exchange to a JSON file")
            help_table.add_row("a / q", "Abort", "Stop interrogation and show partial results")
            help_table.add_row("?", "Help", "Show this help message")

            self.console.print(help_table)
        else:
            print("\nInteractive Mode Commands:")
            print("  c/Enter - Continue to next test")
            print("  f       - Send follow-up probe (if suggested)")
            print("  m       - Enter manual/custom prompt")
            print("  r       - Repeat this test")
            print("  s       - Skip remaining tests in this category")
            print("  d       - Show detailed analysis")
            print("  e       - Export this exchange")
            print("  a/q     - Abort and show results")
            print("  ?       - Show this help")
            print()

    async def _confirm_abort(self) -> bool:
        """
        Confirm the user wants to abort.

        Returns:
            True if confirmed, False otherwise.
        """
        if RICH_AVAILABLE and self.console:
            self.console.print()
            self.console.print(
                f"[yellow]Abort interrogation?[/yellow] "
                f"({self._passed_count + self._failed_count}/{self._total_tests} tests completed)"
            )
            self.console.print("[dim]Type 'y' or 'yes' to confirm, anything else to cancel[/dim]")
        else:
            print(f"\nAbort? ({self._passed_count + self._failed_count}/{self._total_tests} completed)")
            print("Type 'y' to confirm:")

        if self.input_callback:
            confirm = await self.input_callback()
        else:
            loop = asyncio.get_event_loop()
            confirm = await loop.run_in_executor(
                None, lambda: input("> ").strip().lower()
            )

        return confirm in ("y", "yes")

    async def get_manual_input(self) -> str:
        """
        Get manual prompt input from user.

        Returns:
            User-entered prompt text.
        """
        if RICH_AVAILABLE and self.console:
            self.console.print("[yellow]Enter custom prompt (or empty to cancel):[/yellow]")
        else:
            print("Enter custom prompt (or empty to cancel):")

        if self.input_callback:
            return await self.input_callback()
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: input("> ").strip()
            )

    async def show_progress(
        self,
        current: int,
        total: int,
        message: str,
    ) -> None:
        """
        Show progress update.

        Args:
            current: Current progress count.
            total: Total count.
            message: Progress message.
        """
        if self.config.mode == DisplayMode.SILENT:
            return

        if self.config.mode == DisplayMode.NORMAL:
            # Simple progress for normal mode
            pct = (current / total * 100) if total > 0 else 0
            if RICH_AVAILABLE and self.console:
                self.console.print(f"[dim][{current}/{total}] {pct:.0f}% - {message}[/dim]", end="\r")
            else:
                print(f"[{current}/{total}] {pct:.0f}% - {message}", end="\r")

    async def show_summary(
        self,
        passed: int,
        failed: int,
        total: int,
        risk_level: str,
        risk_score: float,
    ) -> None:
        """
        Show final summary.

        Args:
            passed: Number of tests passed.
            failed: Number of tests failed.
            total: Total tests run.
            risk_level: Risk level string.
            risk_score: Numeric risk score.
        """
        if RICH_AVAILABLE and self.console:
            # Create summary panel
            risk_color = {
                "MINIMAL": "green",
                "LOW": "green",
                "MODERATE": "yellow",
                "HIGH": "red",
                "SEVERE": "red bold",
                "CRITICAL": "red bold",
            }.get(risk_level.upper(), "white")

            summary = f"""
[bold]Results:[/bold] {passed}/{total} tests passed ({failed} failed)
[bold]Risk Level:[/bold] [{risk_color}]{risk_level}[/{risk_color}]
[bold]Risk Score:[/bold] {risk_score:.1f}/100
"""
            self.console.print(Panel(
                summary,
                title="[bold]Interrogation Complete[/bold]",
                border_style="blue",
            ))
        else:
            print(f"\n=== Interrogation Complete ===")
            print(f"Results: {passed}/{total} tests passed ({failed} failed)")
            print(f"Risk Level: {risk_level}")
            print(f"Risk Score: {risk_score:.1f}/100")

    async def export_exchange(
        self,
        prompt: "TestPrompt",
        response: str,
        heuristic: Optional["AnalysisResult"],
        verdict: Optional["JudgeVerdict"],
        export_path: Optional[Path] = None,
    ) -> Path:
        """
        Export a single exchange to file.

        Args:
            prompt: The test prompt.
            response: Model response.
            heuristic: Heuristic analysis result.
            verdict: Judge verdict.
            export_path: Path to export to (auto-generated if not provided).

        Returns:
            Path where export was saved.
        """
        import json
        from datetime import datetime

        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = Path(f"exchange_{prompt.id}_{timestamp}.json")

        data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt.to_dict(),
            "response": response,
            "heuristic": heuristic.to_dict() if heuristic else None,
            "verdict": verdict.to_dict() if verdict else None,
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        if RICH_AVAILABLE and self.console:
            self.console.print(f"[green]Exported to:[/green] {export_path}")
        else:
            print(f"Exported to: {export_path}")

        return export_path


def create_display_handler(
    mode: str = "normal",
    console: Optional["Console"] = None,
) -> DisplayHandler:
    """
    Factory function to create DisplayHandler with appropriate config.

    Args:
        mode: Display mode string ("silent", "normal", "verbose", "stream", "interactive").
        console: Optional Rich Console instance.

    Returns:
        Configured DisplayHandler.
    """
    mode_enum = DisplayMode(mode.lower())

    config = DisplayConfig(
        mode=mode_enum,
        show_prompts=mode_enum in (DisplayMode.VERBOSE, DisplayMode.INTERACTIVE),
        show_responses=mode_enum in (DisplayMode.VERBOSE, DisplayMode.INTERACTIVE, DisplayMode.STREAM),
        show_analysis=mode_enum in (DisplayMode.VERBOSE, DisplayMode.INTERACTIVE),
        pause_between_tests=mode_enum == DisplayMode.INTERACTIVE,
        stream_responses=mode_enum == DisplayMode.STREAM,
    )

    return DisplayHandler(config=config, console=console)
