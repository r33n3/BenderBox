"""
BenderBox UI Module - User Interface Components

This module provides:
- TerminalUI: Rich-based terminal output with styled panels and tables
- ChatUI: Interactive chat interface with command handling
- AnalysisDashboard: Live progress dashboard
- BenderBoxApp: Main application entry point
"""

__all__ = [
    # Terminal UI
    "TerminalUI",
    "ProgressSpinner",
    # Chat UI
    "ChatUI",
    "StreamingChatUI",
    "CommandType",
    "ParsedCommand",
    # Dashboard
    "AnalysisDashboard",
    "DashboardTask",
    "DashboardState",
    "TaskStatus",
    "ProgressTracker",
    "StatusDisplay",
    "run_with_dashboard",
    # Application
    "BenderBoxApp",
    "main",
    "cli",
]


def __getattr__(name):
    """Lazy import for module attributes."""
    # Terminal UI
    if name in ("TerminalUI", "ProgressSpinner"):
        from benderbox.ui.terminal import TerminalUI, ProgressSpinner
        return locals()[name]

    # Chat UI
    if name in ("ChatUI", "StreamingChatUI", "CommandType", "ParsedCommand"):
        from benderbox.ui.chat_ui import (
            ChatUI,
            StreamingChatUI,
            CommandType,
            ParsedCommand,
        )
        return locals()[name]

    # Dashboard
    if name in (
        "AnalysisDashboard",
        "DashboardTask",
        "DashboardState",
        "TaskStatus",
        "ProgressTracker",
        "StatusDisplay",
        "run_with_dashboard",
    ):
        from benderbox.ui.dashboard import (
            AnalysisDashboard,
            DashboardTask,
            DashboardState,
            TaskStatus,
            ProgressTracker,
            StatusDisplay,
            run_with_dashboard,
        )
        return locals()[name]

    # Application
    if name in ("BenderBoxApp", "main", "cli"):
        from benderbox.ui.app import BenderBoxApp, main, cli
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
