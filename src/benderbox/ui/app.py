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
            logger.info("LLM engine initialized")
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


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
