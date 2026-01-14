"""
Main Application Entry Point for BenderBox

Provides CLI commands and application initialization.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: Enable verbose output.
        debug: Enable debug output.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
    )


class BenderBoxApp:
    """
    Main BenderBox application.

    Manages initialization, configuration, and component lifecycle.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initialize BenderBox application.

        Args:
            config_path: Path to configuration file.
            verbose: Enable verbose output.
            debug: Enable debug output.
        """
        self.config_path = config_path
        self.verbose = verbose
        self.debug = debug

        # Components (lazy initialized)
        self._config = None
        self._llm_engine = None
        self._conversation_manager = None
        self._terminal_ui = None
        self._chat_ui = None

    @property
    def config(self):
        """Get or load configuration."""
        if self._config is None:
            from benderbox.config import load_config
            self._config = load_config(self.config_path)
        return self._config

    @property
    def terminal_ui(self):
        """Get or create terminal UI."""
        if self._terminal_ui is None:
            from benderbox.ui.terminal import TerminalUI
            self._terminal_ui = TerminalUI()
        return self._terminal_ui

    async def initialize(self) -> None:
        """Initialize all application components."""
        logger.info("Initializing BenderBox...")

        # Load configuration
        _ = self.config

        # Initialize LLM engine if available
        try:
            from benderbox.nlp.llm_engine import LocalLLMEngine
            self._llm_engine = LocalLLMEngine(self.config.llm)
            if self._llm_engine.is_available:
                logger.info("LLM engine initialized with llama-cpp-python")
            else:
                logger.warning(
                    "LLM engine initialized but llama-cpp-python not available. "
                    "NLP features will use template-based responses only."
                )
        except ImportError as e:
            logger.warning(f"LLM engine not available: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize LLM engine: {e}")

        # Initialize conversation manager
        try:
            from benderbox.nlp.conversation import ConversationManager
            self._conversation_manager = ConversationManager(
                llm_engine=self._llm_engine,
            )
            logger.info("Conversation manager initialized")
        except ImportError as e:
            logger.warning(f"Conversation manager not available: {e}")
        except Exception as e:
            logger.warning(f"Could not initialize conversation manager: {e}")

    async def run_chat(self) -> None:
        """Run the interactive chat interface."""
        await self.initialize()

        from benderbox.ui.chat_ui import ChatUI

        self._chat_ui = ChatUI(
            conversation_manager=self._conversation_manager,
            terminal_ui=self.terminal_ui,
        )

        await self._chat_ui.run()

    async def run_analysis(
        self,
        target: str,
        profile: str = "standard",
        output: Optional[str] = None,
        format: str = "markdown",
        open_browser: bool = False,
    ) -> Dict[str, Any]:
        """
        Run analysis on a target.

        Args:
            target: Path, URL, or Hugging Face model ID to analyze.
            profile: Analysis profile.
            output: Output path.
            format: Output format.
            open_browser: Auto-open HTML in browser.

        Returns:
            Analysis result dictionary.
        """
        await self.initialize()

        from benderbox.ui.dashboard import ProgressTracker
        from benderbox.utils import ModelSourceHandler, ModelSource

        self.terminal_ui.print_banner()

        # Detect source type and show appropriate message
        handler = ModelSourceHandler(
            cache_path=Path(self.config.storage.model_cache_path),
            cache_ttl_days=self.config.storage.model_cache_ttl_days,
            download_timeout=self.config.storage.download_timeout_seconds,
            max_size_gb=self.config.storage.max_download_size_gb,
        )
        source_type = handler.detect_source(target)

        if source_type == ModelSource.URL:
            self.terminal_ui.print_info(f"Target: {target}")
            self.terminal_ui.print_info("Source: URL (will download if not cached)")
        elif source_type == ModelSource.HUGGINGFACE:
            self.terminal_ui.print_info(f"Target: {target}")
            self.terminal_ui.print_info("Source: Hugging Face (will download if not cached)")
        else:
            self.terminal_ui.print_info(f"Analyzing: {target}")

        self.terminal_ui.print_info(f"Profile: {profile}")

        result = None

        if self._conversation_manager:
            # Create download progress callback for Rich progress bar
            download_progress = None
            progress_task_id = None

            try:
                from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn

                if source_type != ModelSource.LOCAL and self.terminal_ui.console:
                    download_progress = Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        DownloadColumn(),
                        TransferSpeedColumn(),
                        console=self.terminal_ui.console,
                    )

                    def on_download_progress(prog):
                        nonlocal progress_task_id
                        if progress_task_id is None:
                            progress_task_id = download_progress.add_task(
                                f"Downloading {prog.filename}...",
                                total=prog.total_bytes or 100,
                            )
                        download_progress.update(
                            progress_task_id,
                            completed=prog.downloaded_bytes,
                            total=prog.total_bytes or 100,
                        )

                    download_progress.start()
            except ImportError:
                on_download_progress = None

            try:
                # Use conversation manager for analysis
                with ProgressTracker(
                    console=self.terminal_ui.console,
                    description="Running analysis...",
                ) as tracker:
                    tracker.start(1)

                    query = f"analyze {target} with {profile} profile"
                    response = await self._conversation_manager.process_query(query)

                    tracker.advance()
                    result = response.analysis_result
            finally:
                if download_progress:
                    download_progress.stop()

        if result:
            self.terminal_ui.print_analysis_summary(result)

            # Export if output path specified
            if output:
                try:
                    from benderbox.reporting import ExportManager, ExportFormat

                    export_manager = ExportManager()
                    fmt = ExportFormat[format.upper()]

                    export_result = await export_manager.export_analysis(
                        result,
                        format=fmt,
                        output_path=output,
                    )

                    if export_result.success:
                        self.terminal_ui.print_success(f"Exported to: {export_result.path}")

                        # Auto-open in browser for HTML format
                        if open_browser and fmt == ExportFormat.HTML:
                            import webbrowser
                            from pathlib import Path
                            file_url = Path(export_result.path).as_uri()
                            webbrowser.open(file_url)
                            self.terminal_ui.print_info("Opened report in browser")
                    else:
                        self.terminal_ui.print_error(f"Export failed: {export_result.error}")
                except Exception as e:
                    self.terminal_ui.print_error(f"Export error: {e}")
        else:
            self.terminal_ui.print_warning("Analysis did not produce results")

        return result or {}

    async def show_status(self) -> None:
        """Show system status."""
        await self.initialize()

        from benderbox.ui.dashboard import StatusDisplay

        status_display = StatusDisplay(console=self.terminal_ui.console)

        # Build status information
        status = {
            "Version": "3.0.0-alpha",
            "Config Loaded": self._config is not None,
            "LLM Engine": self._llm_engine is not None,
            "Conversation Manager": self._conversation_manager is not None,
        }

        if self._config:
            status["Vector Store"] = str(self._config.storage.vector_store_path)
            status["Database"] = str(self._config.storage.db_path)
            status["Knowledge Base"] = str(self._config.storage.knowledge_path)

        if self._llm_engine and self._config:
            status["Analysis Model"] = self._config.llm.analysis_model_path or "Not configured"
            status["Code Model"] = self._config.llm.code_model_path or "Not configured"

        self.terminal_ui.print_banner()
        status_display.show_status(status)


# CLI Commands
@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("-c", "--config", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def cli(ctx, verbose: bool, debug: bool, config: Optional[str]):
    """BenderBox - AI Security Analysis Platform"""
    setup_logging(verbose=verbose, debug=debug)

    ctx.ensure_object(dict)
    ctx.obj["app"] = BenderBoxApp(
        config_path=Path(config) if config else None,
        verbose=verbose,
        debug=debug,
    )


@cli.command()
@click.pass_context
def chat(ctx):
    """
    Start interactive chat interface for natural language queries.

    REQUIREMENTS:
        - llama-cpp-python must be installed
        - A local LLM model must be downloaded and configured

    SETUP:
        pip install llama-cpp-python
        benderbox models download tinyllama
        benderbox models setup

    USE 'benderbox nlp status' to check if NLP features are ready.

    EXAMPLES:
        benderbox chat                    # Start chat session
        benderbox -i                      # Alternative interactive mode
    """
    app: BenderBoxApp = ctx.obj["app"]
    asyncio.run(app.run_chat())


@cli.command()
@click.argument("target")
@click.option("-p", "--profile", default="standard",
              type=click.Choice(["quick", "standard", "full", "adversarial"]),
              help="Analysis profile: quick (~15 tests), standard (~50), full (~100+), adversarial (jailbreak)")
@click.option("-o", "--output", help="Output file path")
@click.option("-f", "--format", default="markdown",
              type=click.Choice(["markdown", "json", "html", "csv", "sarif"]),
              help="Output format")
@click.option("--open", "open_browser", is_flag=True, help="Auto-open HTML report in browser")
@click.pass_context
def analyze(ctx, target: str, profile: str, output: Optional[str], format: str, open_browser: bool):
    """Analyze a target for security issues.

    Supports multiple output formats:
    - markdown: Human-readable Markdown report
    - json: Machine-readable JSON
    - html: Interactive HTML report with charts
    - csv: Findings as CSV for spreadsheets
    - sarif: SARIF format for tool integration

    Use --open with --format html to auto-open in browser.
    """
    app: BenderBoxApp = ctx.obj["app"]
    asyncio.run(app.run_analysis(target, profile, output, format, open_browser))


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status."""
    app: BenderBoxApp = ctx.obj["app"]
    asyncio.run(app.show_status())


@cli.command()
@click.argument("query", nargs=-1, required=True)
@click.option("--type", "-t", "search_type", default="all",
              type=click.Choice(["all", "reports", "findings", "knowledge"]),
              help="Type of search")
@click.option("--limit", "-l", default=10, help="Maximum results to return")
@click.pass_context
def search(ctx, query: tuple, search_type: str, limit: int):
    """
    Search reports and findings using semantic search.

    QUERY: Natural language search query (multiple words allowed)

    Examples:
        benderbox search SQL injection vulnerabilities
        benderbox search --type findings high severity
        benderbox search --type reports model analysis
    """
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    query_str = " ".join(query)
    ui.print_info(f"Searching for: {query_str}")
    ui.print_info(f"Search type: {search_type}")

    async def do_search():
        try:
            from benderbox.nlp.rag import RAGPipeline
            from benderbox.storage.report_indexer import ReportIndexer
            from benderbox.storage.vector_store import VectorStore

            vector_store = VectorStore()
            await vector_store.initialize()

            report_indexer = ReportIndexer(vector_store=vector_store)

            rag = RAGPipeline(report_indexer=report_indexer)

            results = await rag.search(
                query=query_str,
                search_type=search_type,
                top_k=limit,
            )

            return results

        except ImportError as e:
            ui.print_error(f"Search requires additional dependencies: {e}")
            return []
        except Exception as e:
            ui.print_error(f"Search failed: {e}")
            return []

    results = asyncio.run(do_search())

    if not results:
        ui.print_warning("No results found.")
        ui.print_info("Try running some analyses first to populate the search index.")
        return

    # Display results
    try:
        from rich.table import Table
        from rich.panel import Panel
        from rich.console import Console

        console = Console()

        table = Table(title=f"Search Results ({len(results)} found)")
        table.add_column("Type", style="bold", width=10)
        table.add_column("Score", width=8)
        table.add_column("Name/Target", width=25)
        table.add_column("Details", width=40)

        type_colors = {"report": "cyan", "finding": "yellow", "knowledge": "green"}

        for result in results:
            result_type = result.get("type", "unknown")
            score = result.get("score", 0)
            metadata = result.get("metadata", {})

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
                name[:25] if name else "",
                details[:40] if details else "",
            )

        console.print(table)

        # Show top result preview
        if results:
            top = results[0]
            content = top.get("content", "")[:300]
            if len(top.get("content", "")) > 300:
                content += "..."
            console.print(Panel(content, title="Top Result Preview", border_style="dim"))

    except ImportError:
        # Fallback without rich
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            result_type = result.get("type", "unknown")
            score = result.get("score", 0)
            content = result.get("content", "")[:100]
            print(f"{i}. [{result_type}] (score: {score:.2f})")
            print(f"   {content}...")
            print()


@cli.command()
@click.pass_context
def version(ctx):
    """Show version information."""
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()
    ui.print_info("BenderBox v3.0.0-alpha")
    ui.print_info("AI Security Analysis Platform")


# Prerequisites subcommand group
@cli.group()
@click.pass_context
def prerequisites(ctx):
    """Manage BenderBox prerequisites and dependencies."""
    pass


@prerequisites.command("check")
@click.option("--package", "-p", help="Check specific package only")
@click.pass_context
def prereq_check(ctx, package: Optional[str]):
    """Check installation status of prerequisites."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import PrerequisiteManager, PackageStatus

    ui = TerminalUI()
    manager = PrerequisiteManager()

    ui.print_banner()
    ui.print_info("Checking prerequisites...")
    print()

    if package:
        # Check single package
        result = manager.check_package(package)
        results = [result]
    else:
        # Check all packages
        results = manager.check_all_packages()

    # Display results in a table
    try:
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title="Prerequisites Status")
        table.add_column("Package", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Version")
        table.add_column("Path/Message")

        status_styles = {
            PackageStatus.INSTALLED: "[green]INSTALLED[/green]",
            PackageStatus.NOT_INSTALLED: "[red]NOT INSTALLED[/red]",
            PackageStatus.OUTDATED: "[yellow]OUTDATED[/yellow]",
            PackageStatus.VERSION_CONFLICT: "[red]VERSION CONFLICT[/red]",
            PackageStatus.ERROR: "[red]ERROR[/red]",
        }

        for result in results:
            status_text = status_styles.get(result.status, str(result.status))
            version_text = str(result.installed_version) if result.installed_version else "-"
            path_text = result.path or result.message

            table.add_row(
                result.package,
                status_text,
                version_text,
                path_text[:50] + "..." if len(path_text) > 50 else path_text,
            )

        console.print(table)

        # Summary
        installed = sum(1 for r in results if r.status == PackageStatus.INSTALLED)
        total = len(results)
        print()
        if installed == total:
            ui.print_success(f"All {total} prerequisites are installed.")
        else:
            ui.print_warning(f"{installed}/{total} prerequisites are installed.")

    except ImportError:
        # Fallback without rich
        for result in results:
            status = "OK" if result.status == PackageStatus.INSTALLED else "MISSING"
            print(f"  {result.package}: {status}")
            if result.message:
                print(f"    {result.message}")


@prerequisites.command("install")
@click.argument("package")
@click.option("--force", "-f", is_flag=True, help="Force reinstall")
@click.pass_context
def prereq_install(ctx, package: str, force: bool):
    """Install a prerequisite package."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import PrerequisiteManager

    ui = TerminalUI()
    manager = PrerequisiteManager()

    ui.print_banner()

    if package not in manager.packages:
        ui.print_error(f"Unknown package: {package}")
        ui.print_info("Available packages: " + ", ".join(manager.packages.keys()))
        return

    ui.print_info(f"Installing {package}...")

    # Create progress callback
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        from rich.console import Console

        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task_id = progress.add_task(f"Installing {package}...", total=100)

            def on_progress(message: str, percent: float):
                progress.update(task_id, completed=int(percent * 100), description=message)

            result = asyncio.run(manager.install_package(package, force=force, progress_callback=on_progress))

    except ImportError:
        # Fallback without rich progress
        def on_progress(message: str, percent: float):
            print(f"  {message} ({int(percent * 100)}%)")

        result = asyncio.run(manager.install_package(package, force=force, progress_callback=on_progress))

    print()
    if result.success:
        ui.print_success(result.message)
        if result.path:
            ui.print_info(f"Installed to: {result.path}")
        if result.version:
            ui.print_info(f"Version: {result.version}")
    else:
        ui.print_error(result.message)


@prerequisites.command("list")
@click.pass_context
def prereq_list(ctx):
    """List all available prerequisite packages."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import PrerequisiteManager

    ui = TerminalUI()
    manager = PrerequisiteManager()

    ui.print_banner()
    ui.print_info("Available Packages:")
    print()

    try:
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title="Available Prerequisites")
        table.add_column("Package", style="cyan")
        table.add_column("Description")
        table.add_column("Required", style="bold")

        for name, pkg in manager.packages.items():
            required = "[green]Yes[/green]" if pkg.required else "[dim]No[/dim]"
            table.add_row(name, pkg.description, required)

        console.print(table)

    except ImportError:
        for name, pkg in manager.packages.items():
            required = "(required)" if pkg.required else ""
            print(f"  {name}: {pkg.description} {required}")


@prerequisites.command("script")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--packages", "-p", multiple=True, help="Specific packages to include")
@click.pass_context
def prereq_script(ctx, output: Optional[str], packages: tuple):
    """Generate installation script for prerequisites."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import PrerequisiteManager

    ui = TerminalUI()
    manager = PrerequisiteManager()

    package_list = list(packages) if packages else None
    script = manager.generate_install_script(package_list)

    if output:
        with open(output, "w") as f:
            f.write(script)
        ui.print_success(f"Installation script written to: {output}")
    else:
        print(script)


@prerequisites.command("instructions")
@click.argument("package")
@click.pass_context
def prereq_instructions(ctx, package: str):
    """Show installation instructions for a package."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import PrerequisiteManager

    ui = TerminalUI()
    manager = PrerequisiteManager()

    if package not in manager.packages:
        ui.print_error(f"Unknown package: {package}")
        return

    instructions = manager.get_install_instructions(package)
    print(instructions)


@prerequisites.command("add")
@click.argument("name")
@click.option("--description", "-d", required=True, help="Package description")
@click.option("--version-command", help="Command to check version")
@click.option("--version-pattern", help="Regex pattern to extract version")
@click.option("--check-command", help="Command to check if installed")
@click.option("--min-version", help="Minimum required version")
@click.option("--required", is_flag=True, help="Mark as required package")
@click.pass_context
def prereq_add(
    ctx,
    name: str,
    description: str,
    version_command: Optional[str],
    version_pattern: Optional[str],
    check_command: Optional[str],
    min_version: Optional[str],
    required: bool,
):
    """Add a custom package definition."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import PrerequisiteManager, PackageDefinition

    ui = TerminalUI()
    manager = PrerequisiteManager()

    pkg = PackageDefinition(
        name=name,
        description=description,
        required=required,
        version_command=version_command or "",
        version_pattern=version_pattern or "",
        check_command=check_command or "",
        min_version=min_version,
    )

    manager.add_package(pkg)
    ui.print_success(f"Added package definition: {name}")
    ui.print_info(f"Configuration saved to: {manager.config_path}")


@prerequisites.command("remove")
@click.argument("name")
@click.pass_context
def prereq_remove(ctx, name: str):
    """Remove a custom package definition."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import PrerequisiteManager

    ui = TerminalUI()
    manager = PrerequisiteManager()

    if manager.remove_package(name):
        ui.print_success(f"Removed package: {name}")
    else:
        ui.print_error(f"Cannot remove package: {name} (either doesn't exist or is a default)")


# Interrogation command
@cli.command()
@click.argument("model_target")
@click.option("-p", "--profile", default="quick",
              type=click.Choice(["quick", "standard", "full", "adversarial"]),
              help="Interrogation profile: quick (~15), standard (~50), full (~100+), adversarial (jailbreak)")
@click.option("-c", "--censorship", default="unknown", help="Claimed censorship level")
@click.option("-o", "--output", help="Output report file path")
@click.option("-f", "--format", "output_format", default="json",
              type=click.Choice(["json", "markdown", "html"]),
              help="Output format (default: json)")
@click.option("--open", "open_browser", is_flag=True, help="Auto-open HTML report in browser")
@click.option("--no-validate", is_flag=True, help="Skip censorship validation")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation for API targets")
@click.pass_context
def interrogate(
    ctx,
    model_target: str,
    profile: str,
    censorship: str,
    output: Optional[str],
    output_format: str,
    open_browser: bool,
    no_validate: bool,
    yes: bool,
):
    """
    Interrogate a model for safety and censorship validation.

    MODEL_TARGET can be:
    - Local file path: ./model.gguf
    - Hugging Face model ID: TheBloke/Llama-2-7B-GGUF
    - Hugging Face URL: https://huggingface.co/TheBloke/...
    - Direct download URL: https://example.com/model.gguf
    - OpenAI API: openai:gpt-4-turbo, openai:gpt-3.5-turbo
    - Anthropic API: anthropic:claude-3-5-sonnet-20241022
    - Google API: gemini:gemini-1.5-pro
    - xAI API: grok:grok-2

    Tests the model with various prompts to detect:
    - Unwanted outputs (harmful content generation)
    - Jailbreak vulnerabilities
    - Censorship level verification
    - Mislabeling detection

    NOTE: API-based interrogation requires paid accounts with the respective providers.
    """
    import asyncio
    from pathlib import Path
    from benderbox.ui.terminal import TerminalUI
    from benderbox.config import load_config
    from benderbox.utils import ModelSourceHandler, ModelSource
    from benderbox.interrogation.runner import RunnerFactory

    ui = TerminalUI()
    ui.print_banner()

    try:
        from benderbox.interrogation.engine import InterrogationEngine
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()
        config = load_config()

        # Check if this is an API target
        is_api = RunnerFactory.is_api_target(model_target)
        runner = None
        model_path = None

        if is_api:
            # Show cost warning for API targets
            if not yes:
                try:
                    provider_key, model_name, runner_class = RunnerFactory.parse_api_target(model_target)
                    min_cost, max_cost = RunnerFactory.estimate_api_cost(model_target, profile)
                    prompt_count = RunnerFactory.get_prompt_count_for_profile(profile)

                    console.print()
                    console.print(Panel(
                        f"[yellow bold]API INTERROGATION WARNING[/yellow bold]\n\n"
                        f"This will send interrogation prompts to a [bold]paid API service[/bold].\n\n"
                        f"[cyan]Target:[/cyan] {model_target}\n"
                        f"[cyan]Provider:[/cyan] {provider_key.upper()}\n"
                        f"[cyan]Model:[/cyan] {model_name}\n"
                        f"[cyan]Profile:[/cyan] {profile} ({prompt_count} prompts)\n\n"
                        f"[bold]Estimated cost:[/bold] ${min_cost:.2f} - ${max_cost:.2f}\n\n"
                        f"[dim]NOTE: Paid API accounts required. You will be charged by the provider.[/dim]",
                        title="Cost Warning",
                        border_style="yellow",
                    ))
                    console.print()

                    if not click.confirm("Do you want to proceed?"):
                        console.print("[red]Cancelled[/red]")
                        return

                except ValueError as e:
                    ui.print_error(str(e))
                    return

            # Create API runner
            try:
                runner = RunnerFactory.create_runner(model_target, config)
                ui.print_info(f"Using {runner.provider_name} API: {runner.model_name}")
            except ValueError as e:
                ui.print_error(str(e))
                return

        else:
            # Local/URL/HuggingFace model - resolve source
            handler = ModelSourceHandler.from_config(config)
            source = handler.detect_source(model_target)

            if source != ModelSource.LOCAL:
                ui.print_info(f"Source: {source.value}")
                ui.print_info(f"Resolving: {model_target}")

                # Download with progress
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    console=console,
                ) as progress:
                    download_task = progress.add_task("Downloading model...", total=100)

                    def download_progress(dp):
                        if dp.total_bytes > 0:
                            progress.update(
                                download_task,
                                completed=dp.percent,
                                description=f"Downloading {dp.filename}..."
                            )

                    resolved = asyncio.run(handler.resolve(model_target, download_progress))

                model_path = resolved.local_path
                if resolved.cached:
                    ui.print_info(f"Using cached model: {model_path}")
                else:
                    ui.print_success(f"Downloaded to: {model_path}")
            else:
                model_path = Path(model_target)
                if not model_path.exists():
                    ui.print_error(f"Model file not found: {model_target}")
                    return

        print()
        if is_api:
            ui.print_info(f"Interrogating: {model_target}")
        else:
            ui.print_info(f"Interrogating: {model_path.name}")
        ui.print_info(f"Profile: {profile}")
        ui.print_info(f"Claimed censorship: {censorship}")
        print()

        engine = InterrogationEngine()

        # Run interrogation with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Interrogating model...", total=100)

            def update_progress(message: str, percent: float):
                progress.update(task, completed=percent * 100, description=message)

            report = engine.interrogate_sync(
                model_path=model_path or model_target,
                profile=profile,
                claimed_censorship=censorship,
                validate_censorship=not no_validate,
                runner=runner,
                progress_callback=update_progress,
            )

        # Display results
        print()
        print(report.summary_text())

        # Show detailed table
        if report.risk_score:
            print()
            table = Table(title="Risk Assessment")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold")

            table.add_row("Overall Risk", f"{report.risk_score.overall_risk:.1f}/100")
            table.add_row("Risk Level", report.risk_score.risk_level.value.upper())
            table.add_row("Safety Risk", f"{report.risk_score.safety_risk:.1f}/100")
            table.add_row("Jailbreak Resistance", f"{report.risk_score.jailbreak_resistance:.1f}/100")
            table.add_row("Tests Passed", f"{report.risk_score.passed_count}/{report.prompts_tested}")
            table.add_row("Duration", f"{report.duration_seconds:.1f}s")

            console.print(table)

        # Show API cost summary if applicable
        if report.api_cost_info:
            print()
            cost_table = Table(title="API Usage Summary")
            cost_table.add_column("Metric", style="cyan")
            cost_table.add_column("Value", style="bold")

            cost_table.add_row("Provider", report.api_cost_info.get("provider", "Unknown"))
            cost_table.add_row("Model", report.api_cost_info.get("model", "Unknown"))
            cost_table.add_row("Prompt Tokens", f"{report.api_cost_info.get('prompt_tokens', 0):,}")
            cost_table.add_row("Completion Tokens", f"{report.api_cost_info.get('completion_tokens', 0):,}")
            cost_table.add_row("Total Tokens", f"{report.api_cost_info.get('total_tokens', 0):,}")
            cost_table.add_row("Estimated Cost", f"${report.api_cost_info.get('estimated_cost_usd', 0):.4f}")

            console.print(cost_table)

        # Save report if output specified
        if output:
            output_path = Path(output)

            # Adjust extension based on format
            if output_format == "html" and not output_path.suffix.lower() == ".html":
                output_path = output_path.with_suffix(".html")
            elif output_format == "markdown" and not output_path.suffix.lower() in (".md", ".markdown"):
                output_path = output_path.with_suffix(".md")

            # Generate report in specified format
            if output_format == "html":
                from benderbox.reporting.html_generator import HTMLReportGenerator
                html_gen = HTMLReportGenerator()
                report_dict = report.to_dict()
                html_content = html_gen.generate(report_dict)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(html_content, encoding="utf-8")
                ui.print_success(f"HTML report saved to: {output_path}")

                # Auto-open in browser
                if open_browser:
                    import webbrowser
                    file_url = output_path.as_uri()
                    webbrowser.open(file_url)
                    ui.print_info("Opened report in browser")

            elif output_format == "markdown":
                from benderbox.reporting.report_generator import ReportGenerator
                md_gen = ReportGenerator()
                report_dict = report.to_dict()
                md_content = md_gen.generate_markdown(report_dict)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(md_content, encoding="utf-8")
                ui.print_success(f"Markdown report saved to: {output_path}")
            else:
                # Default JSON format
                report.save(output_path)
                ui.print_success(f"Report saved to: {output_path}")

    except ImportError as e:
        ui.print_error(f"Missing dependency: {e}")
        ui.print_info("Install with: pip install rich httpx")
    except Exception as e:
        ui.print_error(f"Interrogation failed: {e}")
        logger.exception("Interrogation error")


# Behavior analysis command
@cli.command()
@click.argument("model_target")
@click.option("-p", "--profile", default="standard",
              type=click.Choice(["quick", "standard", "full", "adversarial"]),
              help="Analysis profile: quick (~15), standard (~50), full (~100+), adversarial (jailbreak)")
@click.option("-o", "--output", help="Output report file path")
@click.pass_context
def behavior(ctx, model_target: str, profile: str, output: Optional[str]):
    """
    Perform behavioral analysis on a model.

    MODEL_TARGET can be a local file path or API target.

    Tests the model's behavioral patterns including:
    - Safety response analysis
    - Jailbreak resistance
    - Capability assessment
    - Response consistency
    """
    import asyncio
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    ui.print_info(f"Analyzing behavior: {model_target}")
    ui.print_info(f"Profile: {profile}")
    print()

    async def do_behavior_analysis():
        try:
            from benderbox.nlp.analysis_bridge import AnalysisBridge
            from rich.table import Table
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            bridge = AnalysisBridge()

            def progress_cb(msg):
                ui.print_info(msg)

            result = await bridge.analyze_behavior(
                target=model_target,
                profile=profile,
                progress_callback=progress_cb,
            )

            print()

            # Display results
            profile_data = result.get("profile", {})
            summary = result.get("summary", {})

            # Summary table
            table = Table(title="Behavior Analysis Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold")

            table.add_row("Safety Score", f"{profile_data.get('overall_safety_score', 0):.1f}/100")
            table.add_row("Risk Level", summary.get("risk", {}).get("level", "unknown").upper())
            table.add_row("Safe Outputs", f"{profile_data.get('safe_outputs', 0)}/{profile_data.get('total_outputs', 0)}")
            table.add_row("Jailbreak Rate", f"{profile_data.get('jailbreak_success_rate', 0) * 100:.1f}%")

            console.print(table)

            # Capabilities table if available
            capabilities = result.get("capabilities", {})
            if capabilities:
                print()
                cap_table = Table(title="Capability Assessment")
                cap_table.add_column("Capability", style="cyan")
                cap_table.add_column("Score", style="bold")
                cap_table.add_column("Tests", style="dim")

                for cap_name, cap_data in capabilities.items():
                    score_pct = cap_data.get("score", 0) * 100
                    tests = f"{cap_data.get('tests_passed', 0)}/{cap_data.get('tests_total', 0)}"
                    cap_table.add_row(cap_name.replace("_", " ").title(), f"{score_pct:.0f}%", tests)

                console.print(cap_table)

            # Risk distribution
            risk_dist = profile_data.get("risk_distribution", {})
            if risk_dist:
                print()
                risk_table = Table(title="Risk Distribution")
                risk_table.add_column("Level", style="bold")
                risk_table.add_column("Count")

                for level, count in risk_dist.items():
                    color = {"low": "green", "medium": "yellow", "high": "red", "critical": "red bold"}.get(level, "white")
                    risk_table.add_row(f"[{color}]{level.upper()}[/{color}]", str(count))

                console.print(risk_table)

            return result

        except ImportError as e:
            ui.print_error(f"Missing dependency: {e}")
            return None
        except Exception as e:
            ui.print_error(f"Behavior analysis failed: {e}")
            return None

    result = asyncio.run(do_behavior_analysis())

    if result and output:
        import json
        from pathlib import Path

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        ui.print_success(f"Report saved to: {output}")


# Compare command for model/report comparison
@cli.command()
@click.argument("targets", nargs=-1, required=True)
@click.option("-o", "--output", help="Output comparison report path")
@click.option("--format", "-f", "fmt", default="markdown", type=click.Choice(["markdown", "json", "html"]))
@click.pass_context
def compare(ctx, targets: tuple, output: Optional[str], fmt: str):
    """
    Compare multiple models or reports.

    TARGETS: Two or more model paths or report IDs to compare.

    Examples:
        benderbox compare model1.gguf model2.gguf
        benderbox compare --format html model1.gguf model2.gguf -o comparison.html
    """
    import asyncio
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    if len(targets) < 2:
        ui.print_error("At least 2 targets are required for comparison.")
        return

    ui.print_info(f"Comparing {len(targets)} targets...")
    for target in targets:
        ui.print_info(f"  - {target}")
    print()

    async def do_compare():
        try:
            from benderbox.analyzers.comparative import ComparativeAnalyzer
            from rich.table import Table
            from rich.console import Console

            console = Console()
            analyzer = ComparativeAnalyzer()

            # For now, compare stored reports by ID
            # In future, could analyze models on-the-fly
            from benderbox.storage.report_db import ReportDatabase

            db = ReportDatabase()
            comparison = await analyzer.compare_reports(list(targets), db)

            # Display comparison table
            table = Table(title="Model Comparison")
            table.add_column("Metric", style="cyan")
            for target in targets:
                table.add_column(target[:20], style="bold")

            # Add metric rows
            metrics = ["risk_score", "risk_level", "finding_count", "critical_count", "high_count"]
            for metric in metrics:
                row = [metric.replace("_", " ").title()]
                for result in comparison.get("results", []):
                    value = result.get(metric, "N/A")
                    if isinstance(value, float):
                        value = f"{value:.1f}"
                    row.append(str(value))
                table.add_row(*row)

            console.print(table)

            # Show recommendation
            rec = comparison.get("recommendation", "")
            if rec:
                print()
                ui.print_info(f"Recommendation: {rec}")

            return comparison

        except Exception as e:
            ui.print_error(f"Comparison failed: {e}")
            return None

    result = asyncio.run(do_compare())

    if result and output:
        from benderbox.reporting.report_generator import ReportGenerator

        generator = ReportGenerator()
        if fmt == "markdown":
            content = generator.generate_comparison_markdown(result)
        elif fmt == "json":
            import json
            content = json.dumps(result, indent=2)
        else:
            content = generator.generate_comparison_html(result)

        from pathlib import Path
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        ui.print_success(f"Comparison report saved to: {output}")


# Export command group for report exports
@cli.group()
@click.pass_context
def export(ctx):
    """Export reports and analysis results to various formats."""
    pass


@export.command("report")
@click.argument("report_id")
@click.option("-f", "--format", "output_format", default="html",
              type=click.Choice(["html", "markdown", "json", "csv", "sarif"]),
              help="Export format")
@click.option("-o", "--output", help="Output file path")
@click.option("--open", "open_browser", is_flag=True, help="Auto-open HTML in browser")
@click.pass_context
def export_report(ctx, report_id: str, output_format: str, output: Optional[str], open_browser: bool):
    """
    Export a stored report to a specified format.

    REPORT_ID: The ID or path of the report to export.

    Examples:
        benderbox export report abc123 --format html --open
        benderbox export report ./report.json -f markdown -o report.md
    """
    import asyncio
    from pathlib import Path
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    async def do_export():
        try:
            from benderbox.reporting import ExportManager, ExportFormat

            # Load report from ID or path
            report_path = Path(report_id)
            if report_path.exists():
                import json
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                ui.print_info(f"Loaded report from: {report_path}")
            else:
                # Try loading from report database
                from benderbox.storage.report_db import ReportDatabase
                db = ReportDatabase()
                report_data = await db.get_report(report_id)
                if not report_data:
                    ui.print_error(f"Report not found: {report_id}")
                    return None
                ui.print_info(f"Loaded report: {report_id}")

            # Determine output path
            if not output:
                target = report_data.get("target_name", "report")
                clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in target)
                ext = {"html": ".html", "markdown": ".md", "json": ".json", "csv": ".csv", "sarif": ".sarif"}.get(output_format, ".txt")
                output_path = f"./{clean_name}_export{ext}"
            else:
                output_path = output

            # Export
            export_manager = ExportManager()
            fmt = ExportFormat[output_format.upper()]

            result = await export_manager.export_analysis(
                report_data,
                format=fmt,
                output_path=output_path,
            )

            return result

        except Exception as e:
            ui.print_error(f"Export failed: {e}")
            return None

    result = asyncio.run(do_export())

    if result and result.success:
        ui.print_success(f"Exported to: {result.path}")
        ui.print_info(f"Format: {result.format.value}")
        ui.print_info(f"Size: {result.size_bytes:,} bytes")

        # Auto-open HTML in browser
        if open_browser and output_format == "html":
            import webbrowser
            from pathlib import Path
            file_url = Path(result.path).as_uri()
            webbrowser.open(file_url)
            ui.print_info("Opened report in browser")
    elif result:
        ui.print_error(f"Export failed: {result.error}")


@export.command("batch")
@click.argument("source_dir", type=click.Path(exists=True))
@click.option("-f", "--format", "output_format", default="html",
              type=click.Choice(["html", "markdown", "json"]),
              help="Export format")
@click.option("-o", "--output-dir", help="Output directory")
@click.option("--archive", is_flag=True, help="Create ZIP archive of exports")
@click.pass_context
def export_batch(ctx, source_dir: str, output_format: str, output_dir: Optional[str], archive: bool):
    """
    Export multiple reports from a directory.

    SOURCE_DIR: Directory containing JSON report files.

    Examples:
        benderbox export batch ./reports --format html -o ./html_reports
        benderbox export batch ./reports --format markdown --archive
    """
    import asyncio
    import json
    from pathlib import Path
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    source_path = Path(source_dir)
    report_files = list(source_path.glob("*.json"))

    if not report_files:
        ui.print_error(f"No JSON report files found in: {source_dir}")
        return

    ui.print_info(f"Found {len(report_files)} report files")

    async def do_batch_export():
        try:
            from benderbox.reporting import ExportManager, ExportFormat

            analyses = []
            for rf in report_files:
                try:
                    with open(rf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Add source filename as target_name if missing
                        if "target_name" not in data:
                            data["target_name"] = rf.stem
                        analyses.append(data)
                except Exception as e:
                    ui.print_warning(f"Skipping {rf.name}: {e}")

            if not analyses:
                ui.print_error("No valid reports to export")
                return None

            export_manager = ExportManager(output_dir=output_dir or "./exports")
            fmt = ExportFormat[output_format.upper()]

            result = await export_manager.batch_export(
                analyses,
                format=fmt,
                output_dir=output_dir,
                create_archive=archive,
            )

            return result

        except Exception as e:
            ui.print_error(f"Batch export failed: {e}")
            return None

    result = asyncio.run(do_batch_export())

    if result:
        ui.print_success(f"Exported {result.successful}/{result.total} reports")

        if result.failed > 0:
            ui.print_warning(f"{result.failed} reports failed to export")

        if result.archive_path:
            ui.print_info(f"Archive: {result.archive_path}")

        # Show individual results
        try:
            from rich.table import Table
            from rich.console import Console

            console = Console()
            table = Table(title="Export Results")
            table.add_column("File", style="cyan")
            table.add_column("Status")
            table.add_column("Size")

            for r in result.results:
                status = "[green]OK[/green]" if r.success else f"[red]FAILED: {r.error}[/red]"
                size = f"{r.size_bytes:,} bytes" if r.success else "-"
                table.add_row(Path(r.path).name, status, size)

            console.print(table)

        except ImportError:
            for r in result.results:
                status = "OK" if r.success else f"FAILED: {r.error}"
                print(f"  {Path(r.path).name}: {status}")


# Config command group for API keys and settings
@cli.group()
@click.pass_context
def config(ctx):
    """Manage BenderBox configuration and API keys."""
    pass


@config.command("api-keys")
@click.pass_context
def config_api_keys_list(ctx):
    """List configured API keys and their status."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import get_secrets_manager, API_PROVIDERS

    ui = TerminalUI()
    ui.print_banner()

    secrets = get_secrets_manager()
    keys_status = secrets.list_api_keys()

    try:
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(title="API Keys Configuration")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Key", style="dim")
        table.add_column("Source")

        for provider, info in keys_status.items():
            if info["configured"]:
                status = "[green]Configured[/green]"
            else:
                status = "[red]Not Set[/red]"

            table.add_row(
                info["name"],
                status,
                info["masked_key"],
                info["source"] or "-",
            )

        console.print(table)
        print()

        # Show hints
        ui.print_info("To configure a key: benderbox config set-key <provider>")
        ui.print_info("Providers: openai, anthropic, google, xai")
        print()
        ui.print_info(f"Secrets file: {secrets.path}")

    except ImportError:
        # Fallback without rich
        print("API Keys Status:")
        for provider, info in keys_status.items():
            status = "OK" if info["configured"] else "NOT SET"
            print(f"  {info['name']}: {status} - {info['masked_key']}")


@config.command("set-key")
@click.argument("provider", type=click.Choice(["openai", "anthropic", "google", "xai"]))
@click.option("--key", "-k", help="API key (if not provided, will prompt securely)")
@click.pass_context
def config_set_key(ctx, provider: str, key: Optional[str]):
    """
    Set an API key for a provider.

    PROVIDER: openai, anthropic, google, or xai

    The key will be stored securely in BenderBox/data/secrets.yaml
    """
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import get_secrets_manager, API_PROVIDERS

    ui = TerminalUI()
    provider_info = API_PROVIDERS[provider]

    ui.print_banner()
    ui.print_info(f"Configuring {provider_info['name']} API Key")
    print()

    # Show help URL
    ui.print_info(f"Get your API key at: {provider_info['url']}")
    print()

    if not key:
        # Prompt for key with masked input
        key = click.prompt(
            f"Enter {provider_info['name']} API key",
            hide_input=True,
            confirmation_prompt=True,
        )

    if not key or not key.strip():
        ui.print_error("API key cannot be empty")
        return

    key = key.strip()

    # Validate key format (basic check)
    expected_prefix = provider_info.get("key_prefix", "")
    if expected_prefix and not key.startswith(expected_prefix):
        ui.print_warning(f"Key doesn't start with expected prefix '{expected_prefix}'")
        if not click.confirm("Save anyway?"):
            ui.print_error("Cancelled")
            return

    # Save the key
    secrets = get_secrets_manager()
    if secrets.set_api_key(provider, key):
        ui.print_success(f"{provider_info['name']} API key saved successfully")
        ui.print_info(f"Stored in: {secrets.path}")
    else:
        ui.print_error("Failed to save API key")


@config.command("remove-key")
@click.argument("provider", type=click.Choice(["openai", "anthropic", "google", "xai"]))
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def config_remove_key(ctx, provider: str, yes: bool):
    """Remove a stored API key."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import get_secrets_manager, API_PROVIDERS

    ui = TerminalUI()
    provider_info = API_PROVIDERS[provider]

    if not yes:
        if not click.confirm(f"Remove {provider_info['name']} API key?"):
            ui.print_info("Cancelled")
            return

    secrets = get_secrets_manager()
    if secrets.remove_api_key(provider):
        ui.print_success(f"{provider_info['name']} API key removed")
    else:
        ui.print_error("Failed to remove API key")


@config.command("test-key")
@click.argument("provider", type=click.Choice(["openai", "anthropic", "google", "xai"]))
@click.pass_context
def config_test_key(ctx, provider: str):
    """Test if an API key is valid by making a minimal API call."""
    import asyncio
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils import get_secrets_manager, API_PROVIDERS

    ui = TerminalUI()
    provider_info = API_PROVIDERS[provider]

    ui.print_banner()
    ui.print_info(f"Testing {provider_info['name']} API key...")
    print()

    secrets = get_secrets_manager()
    api_key = secrets.get_api_key(provider)

    if not api_key:
        ui.print_error(f"No API key configured for {provider_info['name']}")
        ui.print_info(f"Set one with: benderbox config set-key {provider}")
        return

    async def test_api():
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if provider == "openai":
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
                    )
                elif provider == "anthropic":
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                        json={"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": [{"role": "user", "content": "Hi"}]},
                    )
                elif provider == "google":
                    response = await client.post(
                        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                        params={"key": api_key},
                        headers={"Content-Type": "application/json"},
                        json={"contents": [{"parts": [{"text": "Hi"}]}], "generationConfig": {"maxOutputTokens": 1}},
                    )
                elif provider == "xai":
                    response = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": "grok-2", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1},
                    )

                return response.status_code, response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text

        except httpx.ConnectError as e:
            return None, f"Connection failed: {e}"
        except httpx.TimeoutException:
            return None, "Connection timed out"
        except Exception as e:
            return None, f"Error: {e}"

    status_code, result = asyncio.run(test_api())

    if status_code == 200:
        ui.print_success(f"{provider_info['name']} API key is valid!")
    elif status_code == 401:
        ui.print_error("Invalid API key (401 Unauthorized)")
    elif status_code == 403:
        ui.print_error("Access denied (403 Forbidden) - check your API key permissions")
    elif status_code == 429:
        ui.print_warning("Rate limited (429) - key is valid but you're being rate limited")
    elif status_code == 400:
        # Some APIs return 400 for invalid keys
        error_msg = ""
        if isinstance(result, dict):
            error_msg = result.get("error", {}).get("message", str(result))
        else:
            error_msg = str(result)[:100]

        if "invalid" in error_msg.lower() or "api key" in error_msg.lower():
            ui.print_error(f"Invalid API key: {error_msg}")
        else:
            ui.print_warning(f"Request error (400): {error_msg}")
    elif status_code is None:
        ui.print_error(f"Connection failed: {result}")
    else:
        ui.print_warning(f"Unexpected response ({status_code}): {result}")


@config.command("show-path")
@click.pass_context
def config_show_path(ctx):
    """Show the path to the secrets file."""
    from benderbox.utils import get_secrets_manager

    secrets = get_secrets_manager()
    print(f"Secrets file: {secrets.path}")
    if secrets.path.exists():
        print("Status: File exists")
    else:
        print("Status: File does not exist (will be created when you set a key)")


# Models command group for managing local NLP models
# NLP Features command group
@cli.group()
@click.pass_context
def nlp(ctx):
    """Natural Language Processing features and status."""
    pass


@nlp.command("status")
@click.pass_context
def nlp_status(ctx):
    """Show NLP feature status and availability."""
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()

    print()
    print("NLP Feature Status")
    print("=" * 50)

    # Check llama-cpp-python
    try:
        from llama_cpp import Llama
        llama_status = "Available"
        llama_color = "green"
    except ImportError:
        llama_status = "Not installed"
        llama_color = "yellow"

    # Check sentence-transformers
    try:
        import sentence_transformers
        st_status = "Available"
    except ImportError:
        st_status = "Not installed"

    # Check chromadb
    try:
        import chromadb
        chroma_status = "Available"
    except ImportError:
        chroma_status = "Not installed"

    # Check mcp
    try:
        import mcp
        mcp_status = "Available"
    except ImportError:
        mcp_status = "Not installed"

    print()
    print("Core NLP Dependencies:")
    print(f"  llama-cpp-python: {llama_status}")
    print(f"  sentence-transformers: {st_status}")
    print(f"  chromadb: {chroma_status}")
    print(f"  mcp: {mcp_status}")

    # Check configured models
    from benderbox.utils.config import get_config
    config = get_config()

    print()
    print("Configured Models:")
    if config.llm.analysis_model_path:
        print(f"  Analysis: {config.llm.analysis_model_path}")
    else:
        print("  Analysis: Not configured")
    if config.llm.code_model_path:
        print(f"  Code: {config.llm.code_model_path}")
    else:
        print("  Code: Not configured")

    print()
    print("Available Features:")
    if llama_status == "Available" and config.llm.analysis_model_path:
        print("  [+] Interactive chat (benderbox chat)")
        print("  [+] Natural language queries")
        print("  [+] Model interrogation with local LLM")
    else:
        print("  [-] Interactive chat (requires llama-cpp-python + model)")
        print("  [-] Natural language queries (requires llama-cpp-python + model)")

    if st_status == "Available" and chroma_status == "Available":
        print("  [+] RAG (Retrieval-Augmented Generation)")
    else:
        print("  [-] RAG (requires sentence-transformers + chromadb)")

    print()
    if llama_status != "Available":
        print("To enable NLP features:")
        print("  pip install llama-cpp-python")
        print("  benderbox models download tinyllama")
        print("  benderbox models setup")


@nlp.command("features")
@click.pass_context
def nlp_features(ctx):
    """List available NLP features and commands."""
    print()
    print("NLP Features Overview")
    print("=" * 50)
    print()
    print("INTERACTIVE CHAT:")
    print("  benderbox chat          - Start interactive chat session")
    print("  benderbox -i            - Interactive mode with NLP queries")
    print()
    print("MODEL MANAGEMENT:")
    print("  benderbox models list              - List all models")
    print("  benderbox models list --for analysis - List analysis targets")
    print("  benderbox models list --for nlp    - List NLP/chat models")
    print("  benderbox models add <path> --for analysis - Add model for analysis")
    print("  benderbox models add <path> --for nlp - Add model for chat")
    print("  benderbox models download <id>     - Download recommended model")
    print("  benderbox models setup             - Configure default model")
    print("  benderbox models test              - Test model functionality")
    print()
    print("ANALYSIS:")
    print("  benderbox interrogate   - Test model safety with NLP")
    print("  benderbox context analyze - Analyze instruction files")
    print()
    print("DEPENDENCIES:")
    print("  Core: llama-cpp-python (local LLM inference)")
    print("  RAG:  sentence-transformers, chromadb")
    print("  MCP:  mcp (Model Context Protocol)")
    print()
    print("Use 'benderbox nlp status' to check installation status.")


@cli.group()
@click.pass_context
def models(ctx):
    """Manage local GGUF models for NLP features."""
    pass


@models.command("list")
@click.option("--for", "purpose", type=click.Choice(["analysis", "nlp", "all"]), default="all",
              help="Filter by purpose: analysis (models to analyze), nlp (BenderBox chat), all (default)")
@click.pass_context
def models_list(ctx, purpose: str):
    """List downloaded and recommended models.

    Examples:
        benderbox models list                # All models
        benderbox models list --for nlp      # NLP model only
        benderbox models list --for analysis # Analysis targets only
    """
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils.model_manager import ModelManager, RECOMMENDED_MODELS

    ui = TerminalUI()
    ui.print_banner()

    manager = ModelManager()

    try:
        from rich.table import Table
        from rich.console import Console

        console = Console()

        # Show models based on purpose filter
        if purpose == "analysis" or purpose == "all":
            analysis_models = manager.list_analysis_models()
            if analysis_models:
                table = Table(title="Analysis Models (models/analysis/)")
                table.add_column("Name", style="cyan")
                table.add_column("Filename", style="dim")
                table.add_column("Size", style="green")

                for model in analysis_models:
                    table.add_row(
                        model["name"],
                        model["filename"],
                        f"{model['size_mb']} MB",
                    )

                console.print(table)
            elif purpose == "analysis":
                ui.print_warning("No analysis models found in models/analysis/")
                ui.print_info("Add models with: benderbox models add <path> --for analysis")
            print()

        if purpose == "nlp" or purpose == "all":
            nlp_models = manager.list_nlp_models()
            if nlp_models:
                table = Table(title="NLP Models (models/nlp/)")
                table.add_column("Name", style="cyan")
                table.add_column("Filename", style="dim")
                table.add_column("Size", style="green")

                for model in nlp_models:
                    table.add_row(
                        model["name"],
                        model["filename"],
                        f"{model['size_mb']} MB",
                    )

                console.print(table)
            elif purpose == "nlp":
                ui.print_warning("No NLP models found in models/nlp/")
                ui.print_info("Download with: benderbox models download tinyllama --purpose nlp")
            print()

        # Show all downloaded models when purpose is "all"
        if purpose == "all":
            downloaded = manager.get_downloaded_models()
            # Filter out models already shown in analysis/nlp tables
            other_models = [m for m in downloaded
                          if "models/analysis" not in m["location"]
                          and "models/nlp" not in m["location"]
                          and "models\\analysis" not in m["location"]
                          and "models\\nlp" not in m["location"]]

            if other_models:
                table = Table(title="Other Downloaded Models")
                table.add_column("Name", style="cyan")
                table.add_column("Size", style="green")
                table.add_column("Location", style="dim")

                for model in other_models:
                    table.add_row(
                        model["name"],
                        f"{model['size_mb']} MB",
                        model["location"],
                    )

                console.print(table)
                print()

            # Show default model status
            default_path = manager.get_default_model_path()
            if default_path:
                ui.print_success(f"Default analysis model: {default_path.name}")
                ui.print_info(f"Path: {default_path}")
            else:
                ui.print_warning("No default analysis model configured.")
                ui.print_info("Set one with: benderbox models setup")

            print()

            # Show recommended models
            rec_table = Table(title="Recommended Models for Download")
            rec_table.add_column("ID", style="cyan")
            rec_table.add_column("Name", style="bold")
            rec_table.add_column("Size", style="green")
            rec_table.add_column("RAM Needed", style="yellow")
            rec_table.add_column("Quality")
            rec_table.add_column("Description", style="dim")

            for key, model in RECOMMENDED_MODELS.items():
                quality_color = {"basic": "yellow", "good": "green", "best": "cyan"}
                rec_table.add_row(
                    key,
                    model.name,
                    f"{model.size_mb} MB",
                    f"{model.min_ram_gb}+ GB",
                    f"[{quality_color.get(model.quality, 'white')}]{model.quality}[/]",
                    model.description[:50] + "..." if len(model.description) > 50 else model.description,
                )

            console.print(rec_table)
            print()
            ui.print_info("Download a model: benderbox models download <id>")
            ui.print_info("For your system: 'tinyllama' or 'phi2' recommended")

    except ImportError:
        # Fallback without rich
        if purpose in ("analysis", "all"):
            analysis_models = manager.list_analysis_models()
            print("Analysis Models:")
            for model in analysis_models:
                print(f"  - {model['name']} ({model['size_mb']} MB)")

        if purpose in ("nlp", "all"):
            nlp_models = manager.list_nlp_models()
            print("NLP Models:")
            for model in nlp_models:
                print(f"  - {model['name']} ({model['size_mb']} MB)")


@models.command("download")
@click.argument("model_id", required=False)
@click.option("--set-default", "-d", is_flag=True, help="Set as default model after download")
@click.option("--purpose", "-p", type=click.Choice(["analysis", "nlp", "code"]), default="nlp",
              help="Purpose for the model: analysis (interrogate models), nlp (chat features), code (code gen)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts (for scripting/CI)")
@click.pass_context
def models_download(ctx, model_id: str, set_default: bool, purpose: str, yes: bool):
    """Download a recommended model.

    MODEL_ID is the model identifier (e.g., 'tinyllama', 'phi2').
    If not specified, downloads the recommended model for your system.

    Use --purpose to specify the model's intended use:
    - nlp: For BenderBox's chat/NLP features (default)
    - analysis: For interrogating/analyzing other models
    - code: For code generation features

    Use --yes to skip confirmation prompts for automation/CI.
    """
    import asyncio
    from benderbox.ui.terminal import TerminalUI, ProgressSpinner
    from benderbox.utils.model_manager import ModelManager, RECOMMENDED_MODELS, DEFAULT_MODEL

    ui = TerminalUI()
    ui.print_banner()

    manager = ModelManager()

    # Default to recommended model if not specified
    if not model_id:
        model_id = DEFAULT_MODEL
        ui.print_info(f"No model specified, using recommended: {model_id}")

    if model_id not in RECOMMENDED_MODELS:
        ui.print_error(f"Unknown model: {model_id}")
        ui.print_info(f"Available models: {', '.join(RECOMMENDED_MODELS.keys())}")
        return

    model = RECOMMENDED_MODELS[model_id]
    ui.print_info(f"Model: {model.name}")
    ui.print_info(f"Size: ~{model.size_mb} MB")
    ui.print_info(f"Source: {model.huggingface_repo}")
    ui.print_info(f"Purpose: {purpose}")
    print()

    if not yes and not click.confirm("Download this model?"):
        ui.print_info("Cancelled")
        return

    async def do_download():
        def progress_cb(msg, pct):
            pass  # Progress handled by spinner

        return await manager.download_model(model_id, progress_cb)

    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {model.name}...", total=None)
            success, message, path = asyncio.run(do_download())

        print()

        if success:
            ui.print_success(message)
            ui.print_info(f"Path: {path}")

            if set_default or yes or click.confirm(f"Set as default {purpose} model?"):
                ok, msg = manager.setup_default_model(path, purpose=purpose)
                if ok:
                    ui.print_success(msg)
                else:
                    ui.print_error(msg)
        else:
            ui.print_error(message)

    except ImportError:
        # Fallback without rich
        print(f"Downloading {model.name}...")
        success, message, path = asyncio.run(do_download())
        if success:
            print(f"Success: {message}")
            print(f"Path: {path}")
            if set_default or yes:
                ok, msg = manager.setup_default_model(path, purpose=purpose)
                print(f"Default model: {msg}")
        else:
            print(f"Error: {message}")


@models.command("add")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--for", "purpose", type=click.Choice(["analysis", "nlp"]), default="analysis",
              help="Purpose: analysis (models to analyze) or nlp (BenderBox chat)")
@click.option("--copy/--link", default=True,
              help="Copy the model (default) or create a symlink")
@click.pass_context
def models_add(ctx, model_path: str, purpose: str, copy: bool):
    """Add a model to the analysis or NLP folder.

    Copies (or links) a model file to the appropriate BenderBox folder
    so it can be discovered for analysis or NLP.

    Examples:
        benderbox models add ./my-model.gguf --for analysis
        benderbox models add ./chat-model.gguf --for nlp
        benderbox models add ./model.gguf --link  # Symlink instead of copy
    """
    import shutil
    from pathlib import Path
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils.model_manager import ModelManager

    ui = TerminalUI()
    ui.print_banner()

    source = Path(model_path)
    if not source.suffix.lower() == ".gguf":
        ui.print_error("Model file must be a .gguf file")
        return

    manager = ModelManager()
    target_dir = manager.get_model_dir_for_purpose(purpose)
    target_path = target_dir / source.name

    if target_path.exists():
        ui.print_warning(f"Model already exists: {target_path}")
        if not click.confirm("Overwrite?"):
            ui.print_info("Cancelled")
            return
        target_path.unlink()

    try:
        if copy:
            ui.print_info(f"Copying {source.name} to {purpose} folder...")
            shutil.copy2(source, target_path)
            ui.print_success(f"Copied to: {target_path}")
        else:
            ui.print_info(f"Linking {source.name} to {purpose} folder...")
            target_path.symlink_to(source.resolve())
            ui.print_success(f"Linked to: {target_path}")

        ui.print_info(f"\nModel is now available for {purpose}.")
        if purpose == "analysis":
            ui.print_info("Analyze it with: benderbox analyze " + source.stem)
        else:
            ui.print_info("Use it for chat with: benderbox -i")

    except OSError as e:
        if not copy:
            ui.print_warning(f"Symlink failed ({e}), falling back to copy...")
            shutil.copy2(source, target_path)
            ui.print_success(f"Copied to: {target_path}")
        else:
            ui.print_error(f"Failed to add model: {e}")


@models.command("setup")
@click.argument("model_path", required=False, type=click.Path(exists=True))
@click.option("--purpose", "-p", type=click.Choice(["analysis", "nlp", "code"]), default="nlp",
              help="Purpose for the model: analysis (interrogate models), nlp (chat features), code (code gen)")
@click.pass_context
def models_setup(ctx, model_path: str, purpose: str):
    """Set up a model as the default for a specific purpose.

    If MODEL_PATH is not specified, uses the first available downloaded model.

    Use --purpose to specify the model's intended use:
    - nlp: For BenderBox's chat/NLP features (default)
    - analysis: For interrogating/analyzing other models
    - code: For code generation features
    """
    from pathlib import Path
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils.model_manager import ModelManager

    ui = TerminalUI()
    ui.print_banner()

    manager = ModelManager()

    if model_path:
        path = Path(model_path)
    else:
        # Find an available model
        downloaded = manager.get_downloaded_models()
        if not downloaded:
            ui.print_error("No models found. Download one first:")
            ui.print_info("  benderbox models download tinyllama")
            return

        # Prefer TinyLlama if available
        tinyllama = next((m for m in downloaded if "tinyllama" in m["name"].lower()), None)
        if tinyllama:
            path = Path(tinyllama["path"])
            ui.print_info(f"Found TinyLlama model: {path.name}")
        else:
            path = Path(downloaded[0]["path"])
            ui.print_info(f"Using first available model: {path.name}")

    ui.print_info(f"Purpose: {purpose}")

    if not click.confirm(f"Set {path.name} as default {purpose} model?"):
        ui.print_info("Cancelled")
        return

    success, message = manager.setup_default_model(path, purpose=purpose)

    if success:
        ui.print_success(message)
        if purpose == "nlp":
            ui.print_info("You can now use NLP features: benderbox chat")
        elif purpose == "analysis":
            ui.print_info("You can now analyze models: benderbox interrogate")
    else:
        ui.print_error(message)


@models.command("info")
@click.argument("model_id")
@click.pass_context
def models_info(ctx, model_id: str):
    """Show detailed information about a recommended model."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils.model_manager import RECOMMENDED_MODELS

    ui = TerminalUI()

    if model_id not in RECOMMENDED_MODELS:
        ui.print_error(f"Unknown model: {model_id}")
        ui.print_info(f"Available: {', '.join(RECOMMENDED_MODELS.keys())}")
        return

    model = RECOMMENDED_MODELS[model_id]

    print()
    print(f"Model: {model.name}")
    print(f"ID: {model_id}")
    print()
    print(f"Description: {model.description}")
    print()
    print(f"HuggingFace: {model.huggingface_repo}")
    print(f"Filename: {model.filename}")
    print(f"Size: ~{model.size_mb} MB")
    print(f"Min RAM: {model.min_ram_gb} GB")
    print(f"Quality: {model.quality}")
    print(f"Use case: {model.use_case}")
    print()
    print(f"Download: benderbox models download {model_id}")


@models.command("test")
@click.argument("model_path", required=False)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def models_test(ctx, model_path: Optional[str], verbose: bool):
    """
    Test a local model to verify it works correctly.

    MODEL_PATH is optional - uses configured default model if not provided.

    Examples:
        benderbox models test
        benderbox models test ./models/tinyllama.gguf
        benderbox models test --verbose
    """
    import time
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()

    # Determine model path
    if not model_path:
        from benderbox.utils.config import get_config
        config = get_config()
        model_path = config.llm.analysis_model_path
        if not model_path:
            ui.print_error("No model path provided and no default model configured.")
            ui.print_info("Download a model first: benderbox models download tinyllama")
            return

    # Check file exists
    from pathlib import Path
    model_file = Path(model_path)
    if not model_file.exists():
        ui.print_error(f"Model file not found: {model_path}")
        return

    print()
    print("Model Test")
    print("=" * 40)
    print(f"Model: {model_file.name}")
    print(f"Path: {model_path}")
    print(f"Size: {model_file.stat().st_size / (1024*1024):.1f} MB")
    print()

    # Check if llama-cpp-python is available
    try:
        from llama_cpp import Llama
    except ImportError:
        ui.print_error("llama-cpp-python is not installed.")
        ui.print_info("Install with: pip install llama-cpp-python")
        return

    # Load and test
    print("Loading model...")
    start_time = time.time()

    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_threads=4,
            verbose=verbose,
        )
        load_time = time.time() - start_time
        print(f"Load time: {load_time:.1f}s")
        print()

        # Run test inference
        print("Running test inference...")
        test_prompt = "Hello, I am a"
        start_time = time.time()

        output = llm(
            test_prompt,
            max_tokens=20,
            stop=["\n", "."],
            echo=False,
        )

        inference_time = time.time() - start_time
        response = output["choices"][0]["text"].strip()

        print(f"Prompt: '{test_prompt}'")
        print(f"Response: '{response}'")
        print(f"Inference time: {inference_time:.2f}s")
        print()

        # Success
        ui.print_success("Model test PASSED")
        print()
        print("The model loaded and generated text successfully.")

    except Exception as e:
        ui.print_error(f"Model test FAILED: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


# MCP Server Analysis command group
@cli.group()
@click.pass_context
def mcp(ctx):
    """Analyze MCP (Model Context Protocol) servers for security risks."""
    pass


@mcp.command("analyze")
@click.argument("source")
@click.option("--mode", "-m", default="auto",
              type=click.Choice(["auto", "source", "live", "hybrid"]),
              help="Analysis mode: auto (detect), source (GitHub), live (endpoint), hybrid (both)")
@click.option("--transport", "-t", default="http",
              type=click.Choice(["http", "stdio"]),
              help="Transport type for live interrogation")
@click.option("--output", "-o", help="Output file path")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "markdown", "json"]),
              help="Output format")
@click.pass_context
def mcp_analyze(ctx, source: str, mode: str, transport: str, output: Optional[str], output_format: str):
    """
    Analyze an MCP server for security risks.

    SOURCE can be:
    - GitHub URL: https://github.com/owner/repo
    - MCP endpoint: https://example.com/mcp
    - Smithery URL: https://smithery.ai/server/@org/name

    Examples:
        benderbox mcp analyze https://github.com/browserbase/mcp-server-browserbase
        benderbox mcp analyze https://mcp.example.com/api --mode live
        benderbox mcp analyze https://github.com/org/repo --format markdown -o report.md
    """
    import asyncio
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    async def do_analyze():
        try:
            from benderbox.analyzers.mcp_analyzer import MCPAnalyzer, analyze_mcp_server
            from rich.console import Console
            from rich.panel import Panel

            console = Console()

            ui.print_info(f"Analyzing: {source}")
            ui.print_info(f"Mode: {mode}")
            print()

            # Perform analysis
            try:
                from rich.progress import Progress, SpinnerColumn, TextColumn

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Analyzing MCP server...", total=None)

                    if mode == "hybrid":
                        analyzer = MCPAnalyzer()
                        # For hybrid, we need both repo and endpoint
                        repo_url = source if "github.com" in source else None
                        endpoint = source if source.startswith("http") and "github.com" not in source else None
                        server_info = await analyzer.analyze_hybrid(repo_url=repo_url, endpoint=endpoint, transport=transport)
                    else:
                        server_info = await analyze_mcp_server(source, mode)

            except ImportError:
                # Fallback without rich progress
                if mode == "hybrid":
                    analyzer = MCPAnalyzer()
                    repo_url = source if "github.com" in source else None
                    endpoint = source if source.startswith("http") and "github.com" not in source else None
                    server_info = await analyzer.analyze_hybrid(repo_url=repo_url, endpoint=endpoint, transport=transport)
                else:
                    server_info = await analyze_mcp_server(source, mode)

            # Generate report
            analyzer = MCPAnalyzer()
            report = analyzer.generate_report(server_info, format=output_format)

            # Display or save report
            if output:
                from pathlib import Path
                out_path = Path(output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(report, encoding="utf-8")
                ui.print_success(f"Report saved to: {output}")
            else:
                print()
                print(report)

            # Summary
            print()
            risk_colors = {
                "critical": "red bold",
                "high": "red",
                "medium": "yellow",
                "low": "green",
            }
            risk_level = server_info.overall_risk_level.value
            color = risk_colors.get(risk_level, "white")

            console.print(Panel(
                f"[{color}]Risk Level: {risk_level.upper()}[/{color}]\n"
                f"Risk Score: {server_info.overall_risk_score}/100\n"
                f"Tools Analyzed: {len(server_info.tools)}\n"
                f"Summary: {server_info.risk_summary}",
                title="MCP Server Risk Assessment",
                border_style=color.split()[0],
            ))

            return server_info

        except ImportError as e:
            ui.print_error(f"Missing dependency: {e}")
            ui.print_info("Install with: pip install httpx")
            return None
        except Exception as e:
            ui.print_error(f"Analysis failed: {e}")
            logger.exception("MCP analysis error")
            return None

    asyncio.run(do_analyze())


@mcp.command("risks")
@click.pass_context
def mcp_risks(ctx):
    """Show the risk classification patterns used for MCP analysis."""
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    try:
        from benderbox.analyzers.mcp_analyzer import RISK_PATTERNS, RiskLevel
        from rich.table import Table
        from rich.console import Console

        console = Console()

        table = Table(title="MCP Tool Risk Classification Patterns")
        table.add_column("Capability", style="cyan")
        table.add_column("Risk Level", style="bold")
        table.add_column("Score")
        table.add_column("Keywords", style="dim")

        level_colors = {
            RiskLevel.CRITICAL: "[red]CRITICAL[/red]",
            RiskLevel.HIGH: "[yellow]HIGH[/yellow]",
            RiskLevel.MEDIUM: "[blue]MEDIUM[/blue]",
            RiskLevel.LOW: "[green]LOW[/green]",
        }

        for capability, config in RISK_PATTERNS.items():
            keywords = ", ".join(config["keywords"][:4])
            if len(config["keywords"]) > 4:
                keywords += "..."

            table.add_row(
                capability.replace("_", " ").title(),
                level_colors.get(config["level"], str(config["level"])),
                str(config["score"]),
                keywords,
            )

        console.print(table)

        print()
        ui.print_info("Risk Levels:")
        print("  CRITICAL (80-100): Code execution, credential access, autonomous agents")
        print("  HIGH (60-79): Network requests, browser automation, database writes")
        print("  MEDIUM (40-59): Data extraction, file read, screenshots")
        print("  LOW (0-39): Read-only operations, local computation")

    except ImportError as e:
        ui.print_error(f"Missing dependency: {e}")


@mcp.command("tools")
@click.argument("source")
@click.option("--transport", "-t", default="auto",
              type=click.Choice(["auto", "http", "stdio"]),
              help="Transport type (auto-detected if not specified)")
@click.pass_context
def mcp_tools(ctx, source: str, transport: str):
    """
    List tools available in an MCP server.

    SOURCE: GitHub URL, MCP endpoint, or STDIO command

    Examples:
        benderbox mcp tools https://github.com/browserbase/mcp-server-browserbase
        benderbox mcp tools "npx -y @modelcontextprotocol/server-filesystem ."
    """
    import asyncio
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    def is_github_url(s: str) -> bool:
        """Check if source is a GitHub URL."""
        return s.startswith("https://github.com/") or s.startswith("http://github.com/")

    def is_stdio_command(s: str) -> bool:
        """Check if source looks like a STDIO command."""
        stdio_indicators = ["npx", "node", "python", "uvx", "deno"]
        first_word = s.split()[0] if s.split() else ""
        return any(ind in first_word.lower() for ind in stdio_indicators)

    async def do_list_tools():
        try:
            from rich.table import Table
            from rich.console import Console
            from rich.progress import Progress, SpinnerColumn, TextColumn

            console = Console()

            ui.print_info(f"Discovering tools from: {source}")
            print()

            tools = []
            server_name = source

            # Determine if we should use live connection or static analysis
            use_live = is_stdio_command(source) or (transport == "stdio")
            use_static = is_github_url(source) and transport != "stdio"

            if use_live or (transport == "http" and not is_github_url(source)):
                # Use live MCP client for STDIO commands or HTTP endpoints
                from benderbox.analyzers.mcp_client import MCPClient, MCPTransport

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Connecting to MCP server...", total=None)

                    client = MCPClient()
                    try:
                        # Determine transport
                        if transport == "auto":
                            trans = MCPTransport.STDIO if is_stdio_command(source) else MCPTransport.HTTP
                        else:
                            trans = MCPTransport(transport)

                        progress.update(task, description="Connecting...")
                        await client.connect(source, trans)

                        progress.update(task, description="Discovering tools...")
                        tools = client.tools
                        server_name = source.split()[-1] if is_stdio_command(source) else source

                    finally:
                        await client.disconnect()

                if not tools:
                    ui.print_warning("No tools discovered from live connection.")
                    return

                # Display tools from live connection
                table = Table(title=f"MCP Tools - {server_name}")
                table.add_column("Tool", style="cyan")
                table.add_column("Description")
                table.add_column("Parameters", style="dim")

                for tool in tools:
                    params = []
                    if tool.input_schema and "properties" in tool.input_schema:
                        params = list(tool.input_schema["properties"].keys())[:3]
                    params_str = ", ".join(params) if params else "-"
                    desc = tool.description[:50] + "..." if len(tool.description) > 50 else tool.description

                    table.add_row(
                        tool.name,
                        desc or "-",
                        params_str,
                    )

                console.print(table)
                print()
                ui.print_info(f"Total: {len(tools)} tools discovered")

            elif use_static:
                # Use static analysis for GitHub URLs
                from benderbox.analyzers.mcp_analyzer import analyze_mcp_server

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Analyzing repository...", total=None)
                    server_info = await analyze_mcp_server(source)

                if not server_info.tools:
                    ui.print_warning("No tools discovered from static analysis.")
                    return

                table = Table(title=f"MCP Tools - {server_info.name}")
                table.add_column("Tool", style="cyan")
                table.add_column("Risk", style="bold")
                table.add_column("Score")
                table.add_column("Capabilities", style="dim")
                table.add_column("Description")

                risk_icons = {
                    "critical": "[red]CRIT[/red]",
                    "high": "[yellow]HIGH[/yellow]",
                    "medium": "[blue]MED[/blue]",
                    "low": "[green]LOW[/green]",
                }

                # Sort by risk score descending
                sorted_tools = sorted(server_info.tools, key=lambda t: t.risk_score, reverse=True)

                for tool in sorted_tools:
                    level = tool.risk_level.value
                    caps = ", ".join(tool.capabilities[:3]) if tool.capabilities else "-"
                    desc = tool.description[:40] + "..." if len(tool.description) > 40 else tool.description

                    table.add_row(
                        tool.name,
                        risk_icons.get(level, level),
                        str(tool.risk_score),
                        caps,
                        desc or "-",
                    )

                console.print(table)
                print()
                ui.print_info(f"Total: {len(server_info.tools)} tools")
                ui.print_info(f"Overall Risk: {server_info.overall_risk_level.value.upper()} ({server_info.overall_risk_score}/100)")

            else:
                ui.print_error("Could not determine how to analyze this source. Use --transport to specify.")

        except Exception as e:
            ui.print_error(f"Failed to list tools: {e}")
            if ctx.obj and ctx.obj.get("debug"):
                import traceback
                traceback.print_exc()

    asyncio.run(do_list_tools())


@mcp.command("connect")
@click.argument("target")
@click.option("--transport", "-t", default="auto",
              type=click.Choice(["auto", "http", "stdio"]),
              help="Transport type")
@click.option("--name", "-n", help="Connection name for reference")
@click.pass_context
def mcp_connect(ctx, target: str, transport: str, name: Optional[str]):
    """
    Connect to an MCP server for live interrogation.

    TARGET can be:
    - HTTP endpoint: https://mcp.example.com/api
    - STDIO command: npx @org/mcp-server

    Examples:
        benderbox mcp connect https://mcp.example.com/api
        benderbox mcp connect "npx @modelcontextprotocol/server-filesystem" --transport stdio
    """
    import asyncio
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    async def do_connect():
        try:
            from benderbox.analyzers.mcp_client import MCPClient, MCPTransport

            ui.print_info(f"Connecting to: {target}")
            ui.print_info(f"Transport: {transport}")
            print()

            client = MCPClient()
            transport_enum = MCPTransport(transport)

            connected = await client.connect(target, transport_enum)

            if connected:
                ui.print_success("Connected successfully!")

                # List available tools
                tools = await client.list_tools()

                if tools:
                    ui.print_info(f"Available tools: {len(tools)}")
                    print()
                    for tool in tools[:10]:  # Show first 10
                        print(f"  - {tool.name}: {tool.description[:50]}..." if tool.description else f"  - {tool.name}")
                    if len(tools) > 10:
                        print(f"  ... and {len(tools) - 10} more")
                else:
                    ui.print_warning("No tools discovered")

                await client.disconnect()
            else:
                ui.print_error("Connection failed")

        except ImportError as e:
            ui.print_error(f"Missing dependency: {e}")
        except Exception as e:
            ui.print_error(f"Connection error: {e}")

    asyncio.run(do_connect())


@mcp.command("call")
@click.argument("target")
@click.argument("tool_name")
@click.option("--args", "-a", "tool_args", help="Tool arguments as JSON")
@click.option("--transport", "-t", default="auto",
              type=click.Choice(["auto", "http", "stdio"]),
              help="Transport type")
@click.pass_context
def mcp_call(ctx, target: str, tool_name: str, tool_args: Optional[str], transport: str):
    """
    Call a tool on an MCP server.

    TARGET: MCP server endpoint or command
    TOOL_NAME: Name of the tool to call

    Examples:
        benderbox mcp call https://mcp.example.com/api list_files --args '{"path": "."}'
        benderbox mcp call "npx @org/server" get_info
    """
    import asyncio
    import json
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    async def do_call():
        try:
            from benderbox.analyzers.mcp_client import MCPClient, MCPTransport

            ui.print_info(f"Server: {target}")
            ui.print_info(f"Tool: {tool_name}")
            print()

            client = MCPClient()
            transport_enum = MCPTransport(transport)

            connected = await client.connect(target, transport_enum)
            if not connected:
                ui.print_error("Failed to connect to server")
                return

            # Parse arguments
            args = {}
            if tool_args:
                try:
                    args = json.loads(tool_args)
                except json.JSONDecodeError as e:
                    ui.print_error(f"Invalid JSON arguments: {e}")
                    await client.disconnect()
                    return

            ui.print_info(f"Calling {tool_name}...")
            print()

            result = await client.call_tool(tool_name, args)

            if result.is_error:
                ui.print_error(f"Tool returned error: {result.content}")
            else:
                ui.print_success("Tool executed successfully")
                print()
                print("Response:")
                print("-" * 40)
                print(result.content[:2000] if result.content else "(empty response)")
                if result.content and len(result.content) > 2000:
                    print(f"\n... (truncated, total {len(result.content)} chars)")

            await client.disconnect()

        except ImportError as e:
            ui.print_error(f"Missing dependency: {e}")
        except Exception as e:
            ui.print_error(f"Call failed: {e}")

    asyncio.run(do_call())


@mcp.command("interrogate")
@click.argument("target")
@click.option("--transport", "-t", default="auto",
              type=click.Choice(["auto", "http", "stdio"]),
              help="Transport type")
@click.option("--profile", "-p", default="standard",
              type=click.Choice(["quick", "standard", "full"]),
              help="Test profile: quick (~15 tests), standard (~30 tests), full (~50 tests)")
@click.option("--output", "-o", help="Output report file path")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "markdown", "json"]),
              help="Output format")
@click.pass_context
def mcp_interrogate(ctx, target: str, transport: str, profile: str, output: Optional[str], output_format: str):
    """
    Run security tests against an MCP server.

    Tests tools for vulnerabilities including:
    - Command/SQL injection
    - Path traversal
    - Data exfiltration
    - Privilege escalation
    - Information disclosure

    TARGET: MCP server endpoint or command

    Examples:
        benderbox mcp interrogate https://mcp.example.com/api
        benderbox mcp interrogate "npx @org/server" --profile quick
        benderbox mcp interrogate https://mcp.example.com -o report.md -f markdown
    """
    import asyncio
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    async def do_interrogate():
        try:
            from benderbox.analyzers.mcp_interrogation import MCPInterrogator, interrogate_mcp_server
            from benderbox.analyzers.mcp_client import MCPTransport
            from rich.console import Console
            from rich.panel import Panel
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

            console = Console()

            ui.print_info(f"Target: {target}")
            ui.print_info(f"Profile: {profile}")
            ui.print_info(f"Transport: {transport}")
            print()

            # Run interrogation with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task("Connecting...", total=100)

                def update_progress(msg, pct):
                    progress.update(task, description=msg, completed=pct)

                score = await interrogate_mcp_server(
                    target=target,
                    transport=transport,
                    profile=profile,
                    progress_callback=update_progress,
                )

            # Generate and display report
            interrogator = MCPInterrogator()
            report = interrogator.generate_report(score, format=output_format)

            print()

            if output:
                from pathlib import Path
                out_path = Path(output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(report, encoding="utf-8")
                ui.print_success(f"Report saved to: {output}")
            else:
                print(report)

            # Summary panel
            print()
            risk_colors = {
                "CRITICAL": "red bold",
                "HIGH": "red",
                "MEDIUM": "yellow",
                "LOW": "green",
                "MINIMAL": "green",
            }
            color = risk_colors.get(score.risk_level, "white")

            console.print(Panel(
                f"[{color}]Risk Level: {score.risk_level}[/{color}]\n"
                f"Risk Score: {score.overall_risk}/100\n"
                f"Tools Tested: {score.tools_tested}\n"
                f"Tests Run: {score.tests_run}\n"
                f"Vulnerabilities: {score.vulnerabilities_found}",
                title="MCP Interrogation Results",
                border_style=color.split()[0] if " " in color else color,
            ))

            # Show critical findings
            if score.critical_findings:
                print()
                console.print("[red bold]CRITICAL FINDINGS:[/red bold]")
                for finding in score.critical_findings[:5]:
                    console.print(f"  - [{finding.tool_name}] {finding.description}")

        except ImportError as e:
            ui.print_error(f"Missing dependency: {e}")
            ui.print_info("Install with: pip install rich")
        except Exception as e:
            ui.print_error(f"Interrogation failed: {e}")
            logger.exception("MCP interrogation error")

    asyncio.run(do_interrogate())


# Context Analysis command group
@cli.group()
@click.pass_context
def context(ctx):
    """Analyze instruction files and context for security risks."""
    pass


@context.command("analyze")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--type", "-t", "file_type",
              type=click.Choice(["auto", "skill", "prompt", "instruction", "output"]),
              default="auto",
              help="Type of context file")
@click.option("--output", "-o", help="Output report file path")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "markdown", "json"]),
              help="Output format")
@click.pass_context
def context_analyze(ctx, file_path: str, file_type: str, output: Optional[str], output_format: str):
    """
    Analyze an instruction file for security risks.

    Detects dangerous patterns including:
    - Jailbreak instructions
    - Credential exposure
    - Code execution risks
    - Data exfiltration instructions
    - Prompt injection markers

    FILE_PATH: Path to the file to analyze

    Examples:
        benderbox context analyze skills.md
        benderbox context analyze system_prompt.txt --type prompt
        benderbox context analyze agent_instructions.yaml -o report.json -f json
    """
    from pathlib import Path
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    try:
        from benderbox.analyzers.context_analyzer import ContextAnalyzer, ContextType
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        file_path_obj = Path(file_path)
        ui.print_info(f"Analyzing: {file_path}")
        ui.print_info(f"Type: {file_type}")
        print()

        analyzer = ContextAnalyzer()

        # Convert type string to enum
        context_type = None
        if file_type != "auto":
            context_type = ContextType(file_type)

        result = analyzer.analyze_file(file_path_obj, context_type)

        # Generate report
        report = analyzer.generate_report(result, format=output_format)

        if output:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report, encoding="utf-8")
            ui.print_success(f"Report saved to: {output}")
        else:
            print(report)

        # Summary panel
        print()
        risk_colors = {
            "CRITICAL": "red bold",
            "HIGH": "red",
            "MEDIUM": "yellow",
            "LOW": "green",
            "INFO": "blue",
            "SAFE": "green",
        }
        color = risk_colors.get(result.risk_level.value, "white")

        console.print(Panel(
            f"[{color}]Risk Level: {result.risk_level.value}[/{color}]\n"
            f"Risk Score: {result.risk_score}/100\n"
            f"Findings: {len(result.findings)}\n"
            f"Dangerous Patterns: {len(result.dangerous_patterns)}",
            title="Context Analysis Results",
            border_style=color.split()[0] if " " in color else color,
        ))

    except ImportError as e:
        ui.print_error(f"Missing dependency: {e}")
    except Exception as e:
        ui.print_error(f"Analysis failed: {e}")
        logger.exception("Context analysis error")


@context.command("scan")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--recursive/--no-recursive", "-r", default=True,
              help="Search recursively")
@click.option("--pattern", "-p", multiple=True,
              help="File patterns to match (e.g., '*.md', '*.yaml')")
@click.option("--output", "-o", help="Output report file path")
@click.option("--format", "-f", "output_format", default="text",
              type=click.Choice(["text", "markdown", "json"]),
              help="Output format")
@click.pass_context
def context_scan(ctx, directory: str, recursive: bool, pattern: tuple, output: Optional[str], output_format: str):
    """
    Scan a directory for context files with security issues.

    Analyzes all matching files and reports findings.

    DIRECTORY: Directory to scan

    Examples:
        benderbox context scan ./examples/prompts
        benderbox context scan ./examples/skills --pattern "*.md" --pattern "*.yaml"
        benderbox context scan . -r -o scan_report.md -f markdown
    """
    from pathlib import Path
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    try:
        from benderbox.analyzers.context_analyzer import ContextAnalyzer
        from rich.console import Console
        from rich.table import Table

        console = Console()

        dir_path = Path(directory)
        patterns = list(pattern) if pattern else None

        ui.print_info(f"Scanning: {directory}")
        ui.print_info(f"Recursive: {recursive}")
        if patterns:
            ui.print_info(f"Patterns: {', '.join(patterns)}")
        print()

        analyzer = ContextAnalyzer()
        results = analyzer.analyze_directory(dir_path, patterns, recursive)

        if not results:
            ui.print_warning("No matching files found.")
            return

        # Display results table
        table = Table(title=f"Scan Results ({len(results)} files)")
        table.add_column("File", style="cyan")
        table.add_column("Type")
        table.add_column("Risk", style="bold")
        table.add_column("Score")
        table.add_column("Findings")

        risk_colors = {
            "CRITICAL": "[red]CRIT[/red]",
            "HIGH": "[yellow]HIGH[/yellow]",
            "MEDIUM": "[blue]MED[/blue]",
            "LOW": "[green]LOW[/green]",
            "INFO": "[dim]INFO[/dim]",
            "SAFE": "[green]SAFE[/green]",
        }

        for result in results:
            file_name = Path(result.file_path).name
            if len(file_name) > 30:
                file_name = file_name[:27] + "..."

            table.add_row(
                file_name,
                result.file_type.value,
                risk_colors.get(result.risk_level.value, result.risk_level.value),
                f"{result.risk_score:.0f}",
                str(len(result.findings)),
            )

        console.print(table)

        # Summary
        print()
        total_findings = sum(len(r.findings) for r in results)
        high_risk = sum(1 for r in results if r.risk_level.value in ["CRITICAL", "HIGH"])

        ui.print_info(f"Total files scanned: {len(results)}")
        ui.print_info(f"Total findings: {total_findings}")

        if high_risk > 0:
            ui.print_warning(f"High/Critical risk files: {high_risk}")

        # Save detailed report if output specified
        if output:
            import json
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if output_format == "json":
                report_data = {
                    "directory": str(directory),
                    "files_scanned": len(results),
                    "total_findings": total_findings,
                    "results": [
                        {
                            "file": r.file_path,
                            "type": r.file_type.value,
                            "risk_level": r.risk_level.value,
                            "risk_score": r.risk_score,
                            "findings": len(r.findings),
                            "dangerous_patterns": r.dangerous_patterns[:5],
                        }
                        for r in results
                    ],
                }
                out_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
            else:
                # Generate combined report
                lines = [
                    "# Context Scan Report",
                    "",
                    f"Directory: {directory}",
                    f"Files Scanned: {len(results)}",
                    f"Total Findings: {total_findings}",
                    "",
                ]

                for result in results:
                    if result.risk_level.value in ["CRITICAL", "HIGH", "MEDIUM"]:
                        lines.append(f"## {result.file_path}")
                        lines.append(f"Risk: {result.risk_level.value} ({result.risk_score}/100)")
                        lines.append(f"Findings: {len(result.findings)}")
                        lines.append("")
                        for finding in result.findings[:3]:
                            lines.append(f"- **{finding.category}**: {finding.description}")
                        lines.append("")

                out_path.write_text("\n".join(lines), encoding="utf-8")

            ui.print_success(f"Report saved to: {output}")

    except ImportError as e:
        ui.print_error(f"Missing dependency: {e}")
    except Exception as e:
        ui.print_error(f"Scan failed: {e}")
        logger.exception("Context scan error")


@context.command("output")
@click.argument("text", required=False)
@click.option("--file", "-f", "file_path", type=click.Path(exists=True),
              help="Read output from file")
@click.option("--model", "-m", default="unknown", help="Model name for reference")
@click.pass_context
def context_output(ctx, text: Optional[str], file_path: Optional[str], model: str):
    """
    Analyze model inference output for harmful content.

    Provide output text directly or via --file.

    Examples:
        benderbox context output "Model response text here"
        benderbox context output --file response.txt --model gpt-4
    """
    from pathlib import Path
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    # Get content
    content = text
    if file_path:
        content = Path(file_path).read_text(encoding="utf-8")
    elif not text:
        ui.print_error("Provide output text or use --file")
        return

    try:
        from benderbox.analyzers.context_analyzer import analyze_inference_output
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        ui.print_info(f"Analyzing output from: {model}")
        ui.print_info(f"Content length: {len(content)} chars")
        print()

        result = analyze_inference_output(content, model)

        # Display findings
        if result.findings:
            ui.print_warning(f"Found {len(result.findings)} potential issues:")
            print()
            for finding in result.findings[:10]:
                level_color = {
                    "CRITICAL": "red",
                    "HIGH": "yellow",
                    "MEDIUM": "blue",
                }.get(finding.risk_level.value, "white")
                console.print(f"  [{level_color}]{finding.risk_level.value}[/]: {finding.description}")
                console.print(f"    Match: [dim]{finding.matched_text[:60]}...[/dim]")
        else:
            ui.print_success("No harmful patterns detected")

        # Summary
        print()
        risk_colors = {
            "CRITICAL": "red bold",
            "HIGH": "red",
            "MEDIUM": "yellow",
            "LOW": "green",
            "SAFE": "green",
        }
        color = risk_colors.get(result.risk_level.value, "white")

        console.print(Panel(
            f"[{color}]Risk Level: {result.risk_level.value}[/{color}]\n"
            f"Risk Score: {result.risk_score}/100\n"
            f"Findings: {len(result.findings)}",
            title="Output Analysis Results",
            border_style=color.split()[0] if " " in color else color,
        ))

    except ImportError as e:
        ui.print_error(f"Missing dependency: {e}")
    except Exception as e:
        ui.print_error(f"Analysis failed: {e}")


# =============================================================================
# Report Commands
# =============================================================================


@cli.group()
@click.pass_context
def report(ctx):
    """
    Report management commands.

    Generate, view, and export analysis reports.
    """
    pass


@report.command("view")
@click.option("--reports-dir", "-d", type=click.Path(exists=True),
              help="Reports directory to load from")
@click.option("--output", "-o", type=click.Path(), help="Output HTML file path")
@click.option("--open", "-O", "open_browser", is_flag=True, default=True,
              help="Open in browser (default: true)")
@click.option("--no-open", is_flag=True, help="Don't open in browser")
@click.pass_context
def report_view(ctx, reports_dir: Optional[str], output: Optional[str],
                open_browser: bool, no_open: bool):
    """
    Open the BenderBox Report Viewer.

    Generates an interactive HTML viewer with all reports in a
    retro-futuristic Futurama-inspired interface.

    Examples:
        benderbox report view
        benderbox report view --reports-dir ./my_reports
        benderbox report view --output ~/Desktop/reports.html
    """
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()
    ui.print_banner()

    try:
        from benderbox.reporting.index_generator import ReportViewerGenerator

        generator = ReportViewerGenerator(reports_dir)

        # Collect reports
        reports = generator.collect_reports()
        ui.print_info(f"Found {len(reports)} reports")

        if len(reports) == 0:
            ui.print_warning("No reports found. Run an analysis first:")
            print("  benderbox mcp analyze <server.py>")
            print("  benderbox context analyze <prompt.md>")
            print("  benderbox analyze <model.gguf>")
            return

        # Generate viewer
        should_open = open_browser and not no_open
        output_path = generator.save(output, reports, open_browser=should_open)

        ui.print_success(f"Report viewer saved to: {output_path}")

        if should_open:
            ui.print_info("Opening in browser...")

    except Exception as e:
        ui.print_error(f"Failed to generate report viewer: {e}")


@report.command("list")
@click.option("--limit", "-n", default=20, help="Maximum reports to show")
@click.pass_context
def report_list(ctx, limit: int):
    """
    List available reports.

    Shows recent analysis reports with their risk levels.
    """
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()

    try:
        from benderbox.reporting.index_generator import ReportViewerGenerator
        from rich.console import Console
        from rich.table import Table

        generator = ReportViewerGenerator()
        reports = generator.collect_reports(limit)

        if not reports:
            ui.print_warning("No reports found")
            return

        console = Console()
        table = Table(title=f"Recent Reports ({len(reports)})")

        table.add_column("Target", style="cyan")
        table.add_column("Type", style="blue")
        table.add_column("Risk", style="bold")
        table.add_column("Score")
        table.add_column("Date")

        risk_styles = {
            "critical": "red bold",
            "high": "red",
            "medium": "yellow",
            "low": "green",
        }

        for report in reports[:limit]:
            risk_level = (report.get("summary", {}).get("risk", {}).get("level", "unknown")).lower()
            risk_score = report.get("summary", {}).get("risk", {}).get("score", 0)
            style = risk_styles.get(risk_level, "white")

            table.add_row(
                report.get("target_name", "Unknown")[:30],
                report.get("target_type", "unknown"),
                f"[{style}]{risk_level.upper()}[/{style}]",
                str(risk_score),
                report.get("timestamp", "")[:10],
            )

        console.print(table)
        print()
        ui.print_info("Run 'benderbox report view' to open interactive viewer")

    except Exception as e:
        ui.print_error(f"Failed to list reports: {e}")


@report.command("export")
@click.argument("report_id", required=False)
@click.option("--format", "-f", "output_format", type=click.Choice(["json", "markdown", "html"]),
              default="html", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--open", "-O", "open_browser", is_flag=True, help="Open in browser (HTML only)")
@click.option("--latest", is_flag=True, help="Export the most recent report")
@click.pass_context
def report_export(ctx, report_id: Optional[str], output_format: str,
                  output: Optional[str], open_browser: bool, latest: bool):
    """
    Export a report to a file.

    Examples:
        benderbox report export --latest --format html --open
        benderbox report export abc123 --format markdown -o report.md
    """
    from benderbox.ui.terminal import TerminalUI

    ui = TerminalUI()

    try:
        from benderbox.reporting.index_generator import ReportViewerGenerator
        from benderbox.reporting.html_generator import HTMLReportGenerator
        from pathlib import Path
        import json

        generator = ReportViewerGenerator()
        reports = generator.collect_reports(50)

        if not reports:
            ui.print_warning("No reports found")
            return

        # Select report
        report = None
        if latest or not report_id:
            report = reports[0]
            ui.print_info(f"Exporting latest report: {report.get('target_name', 'Unknown')}")
        else:
            # Find by ID (partial match)
            for r in reports:
                if report_id in r.get("run_id", "") or report_id in r.get("target_name", ""):
                    report = r
                    break

        if not report:
            ui.print_error(f"Report not found: {report_id}")
            return

        # Determine output path
        target_name = report.get("target_name", "report").replace("/", "_").replace("\\", "_")
        if not output:
            ext = {"json": "json", "markdown": "md", "html": "html"}[output_format]
            output = f"{target_name}.{ext}"

        out_path = Path(output)

        # Export based on format
        if output_format == "json":
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        elif output_format == "markdown":
            from benderbox.storage.report_db import ReportDatabase
            db = ReportDatabase()
            content = db._report_to_markdown(report)
            out_path.write_text(content, encoding="utf-8")
        else:  # html
            html_gen = HTMLReportGenerator()
            html = html_gen.generate(report)
            out_path.write_text(html, encoding="utf-8")

            if open_browser:
                import webbrowser
                webbrowser.open(f"file://{out_path.absolute()}")

        ui.print_success(f"Report exported to: {output}")

    except Exception as e:
        ui.print_error(f"Export failed: {e}")


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
