"""
Dashboard UI Components for BenderBox

Live dashboard displays for analysis progress and system status.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import Rich components
try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class TaskStatus(Enum):
    """Status of a dashboard task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DashboardTask:
    """A task displayed on the dashboard."""
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


@dataclass
class DashboardState:
    """State of the dashboard."""
    title: str = "BenderBox Analysis"
    tasks: List[DashboardTask] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    max_logs: int = 10

    def add_task(self, name: str, description: str) -> DashboardTask:
        """Add a new task to the dashboard."""
        task = DashboardTask(name=name, description=description)
        self.tasks.append(task)
        return task

    def add_log(self, message: str) -> None:
        """Add a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]

    def update_metric(self, key: str, value: Any) -> None:
        """Update a metric value."""
        self.metrics[key] = value


class AnalysisDashboard:
    """
    Live dashboard for analysis progress.

    Displays:
    - Overall progress
    - Individual task status
    - Real-time metrics
    - Activity log
    """

    RISK_COLORS = {
        "critical": "bold red",
        "high": "bold orange1",
        "medium": "bold yellow",
        "low": "bold green",
        "info": "bold blue",
        "safe": "bold green",
    }

    def __init__(
        self,
        console: Optional["Console"] = None,
        refresh_rate: int = 4,
    ):
        """
        Initialize AnalysisDashboard.

        Args:
            console: Rich Console instance.
            refresh_rate: Refresh rate per second.
        """
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.refresh_rate = refresh_rate
        self.state = DashboardState()
        self._live: Optional["Live"] = None
        self._progress: Optional["Progress"] = None
        self._task_ids: Dict[str, "TaskID"] = {}

    def _create_layout(self) -> "Layout":
        """Create the dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )

        layout["body"].split_row(
            Layout(name="progress", ratio=2),
            Layout(name="metrics", ratio=1),
        )

        return layout

    def _render_header(self) -> "Panel":
        """Render the header panel."""
        return Panel(
            Text(self.state.title, style="bold white", justify="center"),
            style="cyan",
            box=box.ROUNDED,
        )

    def _render_progress(self) -> "Panel":
        """Render the progress panel."""
        if not self.state.tasks:
            return Panel("No tasks", title="Progress", border_style="blue")

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Task", width=25)
        table.add_column("Status", width=12)
        table.add_column("Progress", width=15)
        table.add_column("Time", width=10)

        for task in self.state.tasks:
            # Status styling
            status_style = {
                TaskStatus.PENDING: "dim",
                TaskStatus.RUNNING: "bold cyan",
                TaskStatus.COMPLETED: "bold green",
                TaskStatus.FAILED: "bold red",
                TaskStatus.SKIPPED: "dim yellow",
            }.get(task.status, "white")

            status_text = Text(task.status.value.upper(), style=status_style)

            # Progress bar
            filled = int(task.progress * 10)
            bar = f"[{'=' * filled}{' ' * (10 - filled)}] {task.progress * 100:.0f}%"

            # Duration
            duration = ""
            if task.duration:
                duration = f"{task.duration:.1f}s"

            table.add_row(
                task.name[:23],
                status_text,
                bar,
                duration,
            )

        return Panel(table, title="Progress", border_style="blue")

    def _render_metrics(self) -> "Panel":
        """Render the metrics panel."""
        if not self.state.metrics:
            return Panel("No metrics", title="Metrics", border_style="green")

        lines = []
        for key, value in self.state.metrics.items():
            # Format value based on type
            if isinstance(value, float):
                formatted = f"{value:.2f}"
            elif isinstance(value, dict) and "level" in value:
                # Risk level
                level = value["level"].lower()
                color = self.RISK_COLORS.get(level, "white")
                formatted = f"[{color}]{level.upper()}[/{color}]"
            else:
                formatted = str(value)

            lines.append(f"[bold]{key}:[/bold] {formatted}")

        content = "\n".join(lines)
        return Panel(content, title="Metrics", border_style="green")

    def _render_logs(self) -> "Panel":
        """Render the activity log panel."""
        if not self.state.logs:
            return Panel("No activity", title="Activity Log", border_style="yellow")

        content = "\n".join(self.state.logs[-5:])
        return Panel(content, title="Activity Log", border_style="yellow")

    def _render(self) -> "Layout":
        """Render the complete dashboard."""
        layout = self._create_layout()
        layout["header"].update(self._render_header())
        layout["progress"].update(self._render_progress())
        layout["metrics"].update(self._render_metrics())
        layout["footer"].update(self._render_logs())
        return layout

    def start(self) -> None:
        """Start the live dashboard."""
        if not RICH_AVAILABLE or not self.console:
            logger.warning("Rich not available. Dashboard disabled.")
            return

        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=True,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live dashboard."""
        if self._live:
            self._live.stop()
            self._live = None

    def update(self) -> None:
        """Update the dashboard display."""
        if self._live:
            self._live.update(self._render())

    def add_task(self, name: str, description: str = "") -> DashboardTask:
        """Add a task to the dashboard."""
        task = self.state.add_task(name, description or name)
        self.update()
        return task

    def start_task(self, task: DashboardTask) -> None:
        """Mark a task as started."""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        self.state.add_log(f"Started: {task.name}")
        self.update()

    def update_task_progress(
        self,
        task: DashboardTask,
        progress: float,
    ) -> None:
        """Update task progress (0.0 to 1.0)."""
        task.progress = min(1.0, max(0.0, progress))
        self.update()

    def complete_task(
        self,
        task: DashboardTask,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a task as completed."""
        task.status = TaskStatus.COMPLETED
        task.progress = 1.0
        task.end_time = datetime.now()
        task.result = result
        self.state.add_log(f"Completed: {task.name}")
        self.update()

    def fail_task(self, task: DashboardTask, error: str) -> None:
        """Mark a task as failed."""
        task.status = TaskStatus.FAILED
        task.end_time = datetime.now()
        task.error = error
        self.state.add_log(f"Failed: {task.name} - {error}")
        self.update()

    def skip_task(self, task: DashboardTask, reason: str = "") -> None:
        """Mark a task as skipped."""
        task.status = TaskStatus.SKIPPED
        task.end_time = datetime.now()
        self.state.add_log(f"Skipped: {task.name}" + (f" - {reason}" if reason else ""))
        self.update()

    def set_metric(self, key: str, value: Any) -> None:
        """Set a metric value."""
        self.state.update_metric(key, value)
        self.update()

    def log(self, message: str) -> None:
        """Add a log message."""
        self.state.add_log(message)
        self.update()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class ProgressTracker:
    """
    Simple progress tracker for batch operations.

    Uses Rich Progress bar for visual feedback.
    """

    def __init__(
        self,
        console: Optional["Console"] = None,
        description: str = "Processing...",
    ):
        """
        Initialize ProgressTracker.

        Args:
            console: Rich Console instance.
            description: Progress description.
        """
        self.console = console or (Console() if RICH_AVAILABLE else None)
        self.description = description
        self._progress: Optional["Progress"] = None
        self._task_id: Optional["TaskID"] = None
        self._total: int = 0
        self._completed: int = 0

    def start(self, total: int) -> None:
        """Start tracking progress."""
        self._total = total
        self._completed = 0

        if RICH_AVAILABLE and self.console:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.description,
                total=total,
            )
        else:
            print(f"{self.description} (0/{total})")

    def advance(self, amount: int = 1) -> None:
        """Advance progress by amount."""
        self._completed += amount

        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=amount)
        else:
            print(f"Progress: {self._completed}/{self._total}")

    def update(self, description: str) -> None:
        """Update progress description."""
        self.description = description

        if self._progress and self._task_id is not None:
            self._progress.update(
                self._task_id,
                description=description,
            )

    def stop(self) -> None:
        """Stop tracking progress."""
        if self._progress:
            self._progress.stop()
            self._progress = None

    def __enter__(self):
        """Context manager entry (requires start() to be called separately)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class StatusDisplay:
    """
    Status display for system information.

    Shows system status, configuration, and resource usage.
    """

    def __init__(self, console: Optional["Console"] = None):
        """
        Initialize StatusDisplay.

        Args:
            console: Rich Console instance.
        """
        self.console = console or (Console() if RICH_AVAILABLE else None)

    def show_status(
        self,
        status: Dict[str, Any],
        title: str = "System Status",
    ) -> None:
        """Display system status."""
        if not RICH_AVAILABLE or not self.console:
            print(f"\n{title}")
            print("-" * 40)
            for key, value in status.items():
                print(f"  {key}: {value}")
            return

        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Property", style="bold")
        table.add_column("Value")
        table.add_column("Status", width=10)

        for key, value in status.items():
            # Determine status indicator
            if isinstance(value, dict):
                display_value = value.get("value", str(value))
                status_ok = value.get("ok", True)
            elif isinstance(value, bool):
                display_value = "Yes" if value else "No"
                status_ok = value
            else:
                display_value = str(value)
                status_ok = True

            status_text = Text(
                "OK" if status_ok else "WARN",
                style="green" if status_ok else "yellow",
            )

            table.add_row(key, str(display_value), status_text)

        self.console.print(table)

    def show_config(
        self,
        config: Dict[str, Any],
        title: str = "Configuration",
    ) -> None:
        """Display configuration settings."""
        if not RICH_AVAILABLE or not self.console:
            print(f"\n{title}")
            print("-" * 40)
            self._print_dict(config)
            return

        from rich.tree import Tree

        tree = Tree(f"[bold]{title}[/bold]")
        self._build_tree(tree, config)
        self.console.print(Panel(tree, border_style="blue"))

    def _print_dict(self, d: Dict, indent: int = 2) -> None:
        """Print dictionary recursively (plain text)."""
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}")

    def _build_tree(self, tree: "Tree", data: Dict) -> None:
        """Build Rich tree from dictionary."""
        for key, value in data.items():
            if isinstance(value, dict):
                branch = tree.add(f"[bold]{key}[/bold]")
                self._build_tree(branch, value)
            elif isinstance(value, list):
                branch = tree.add(f"[bold]{key}[/bold]")
                for item in value:
                    if isinstance(item, dict):
                        self._build_tree(branch, item)
                    else:
                        branch.add(str(item))
            else:
                tree.add(f"[bold]{key}:[/bold] {value}")


async def run_with_dashboard(
    tasks: List[Callable],
    task_names: List[str],
    title: str = "BenderBox Analysis",
    console: Optional["Console"] = None,
) -> List[Any]:
    """
    Run async tasks with a live dashboard.

    Args:
        tasks: List of async callables.
        task_names: Names for each task.
        title: Dashboard title.
        console: Rich Console instance.

    Returns:
        List of task results.
    """
    dashboard = AnalysisDashboard(console=console)
    dashboard.state.title = title

    # Create dashboard tasks
    dash_tasks = []
    for name in task_names:
        dash_tasks.append(dashboard.add_task(name))

    results = []

    with dashboard:
        for i, (task_callable, dash_task) in enumerate(zip(tasks, dash_tasks)):
            dashboard.start_task(dash_task)

            try:
                result = await task_callable()
                dashboard.complete_task(dash_task, result if isinstance(result, dict) else {"result": result})
                results.append(result)
            except Exception as e:
                dashboard.fail_task(dash_task, str(e))
                results.append(None)
                logger.error(f"Task {dash_task.name} failed: {e}")

            # Update overall progress metric
            progress = (i + 1) / len(tasks) * 100
            dashboard.set_metric("Overall Progress", f"{progress:.0f}%")

    return results
