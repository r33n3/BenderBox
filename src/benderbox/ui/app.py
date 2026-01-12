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
    ) -> Dict[str, Any]:
        """
        Run analysis on a target.

        Args:
            target: Path, URL, or Hugging Face model ID to analyze.
            profile: Analysis profile.
            output: Output path.
            format: Output format.

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
    """Start interactive chat interface."""
    app: BenderBoxApp = ctx.obj["app"]
    asyncio.run(app.run_chat())


@cli.command()
@click.argument("target")
@click.option("-p", "--profile", default="standard", help="Analysis profile")
@click.option("-o", "--output", help="Output file path")
@click.option("-f", "--format", default="markdown", help="Output format")
@click.pass_context
def analyze(ctx, target: str, profile: str, output: Optional[str], format: str):
    """Analyze a target for security issues."""
    app: BenderBoxApp = ctx.obj["app"]
    asyncio.run(app.run_analysis(target, profile, output, format))


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status."""
    app: BenderBoxApp = ctx.obj["app"]
    asyncio.run(app.show_status())


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
@click.option("-p", "--profile", default="quick", help="Interrogation profile (quick, standard, full)")
@click.option("-c", "--censorship", default="unknown", help="Claimed censorship level")
@click.option("-o", "--output", help="Output report file path")
@click.option("--no-validate", is_flag=True, help="Skip censorship validation")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation for API targets")
@click.pass_context
def interrogate(
    ctx,
    model_target: str,
    profile: str,
    censorship: str,
    output: Optional[str],
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
            report.save(Path(output))
            ui.print_success(f"Report saved to: {output}")

    except ImportError as e:
        ui.print_error(f"Missing dependency: {e}")
        ui.print_info("Install with: pip install rich httpx")
    except Exception as e:
        ui.print_error(f"Interrogation failed: {e}")
        logger.exception("Interrogation error")


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

    The key will be stored securely in ~/.benderbox/secrets.yaml
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
@cli.group()
@click.pass_context
def models(ctx):
    """Manage local GGUF models for NLP features."""
    pass


@models.command("list")
@click.pass_context
def models_list(ctx):
    """List downloaded and recommended models."""
    from benderbox.ui.terminal import TerminalUI
    from benderbox.utils.model_manager import ModelManager, RECOMMENDED_MODELS

    ui = TerminalUI()
    ui.print_banner()

    manager = ModelManager()

    try:
        from rich.table import Table
        from rich.console import Console

        console = Console()

        # Show downloaded models
        downloaded = manager.get_downloaded_models()

        if downloaded:
            table = Table(title="Downloaded Models")
            table.add_column("Name", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Location", style="dim")

            for model in downloaded:
                table.add_row(
                    model["name"],
                    f"{model['size_mb']} MB",
                    model["location"],
                )

            console.print(table)
        else:
            ui.print_warning("No models downloaded yet.")

        print()

        # Show default model status
        default_path = manager.get_default_model_path()
        if default_path:
            ui.print_success(f"Default model: {default_path.name}")
            ui.print_info(f"Path: {default_path}")
        else:
            ui.print_warning("No default model configured.")
            ui.print_info("Set one with: benderbox models setup")

        print()

        # Show recommended models
        rec_table = Table(title="Recommended Models")
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
        ui.print_info("For your system (14GB RAM): 'tinyllama' or 'phi2' recommended")

    except ImportError:
        # Fallback without rich
        print("Downloaded models:")
        for model in downloaded:
            print(f"  - {model['name']} ({model['size_mb']} MB)")


@models.command("download")
@click.argument("model_id", required=False)
@click.option("--set-default", "-d", is_flag=True, help="Set as default model after download")
@click.pass_context
def models_download(ctx, model_id: str, set_default: bool):
    """Download a recommended model for NLP features.

    MODEL_ID is the model identifier (e.g., 'tinyllama', 'phi2').
    If not specified, downloads the recommended model for your system.
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
    print()

    if not click.confirm("Download this model?"):
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

            if set_default or click.confirm("Set as default model for NLP features?"):
                ok, msg = manager.setup_default_model(path)
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
        else:
            print(f"Error: {message}")


@models.command("setup")
@click.argument("model_path", required=False, type=click.Path(exists=True))
@click.pass_context
def models_setup(ctx, model_path: str):
    """Set up a model as the default for NLP features.

    If MODEL_PATH is not specified, uses the first available downloaded model.
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

    if not click.confirm(f"Set {path.name} as default model?"):
        ui.print_info("Cancelled")
        return

    success, message = manager.setup_default_model(path)

    if success:
        ui.print_success(message)
        ui.print_info("You can now use NLP features: benderbox chat")
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


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
