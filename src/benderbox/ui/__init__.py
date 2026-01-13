"""
BenderBox UI Module - User Interface Components

This module provides:
- TerminalUI: Rich-based terminal output with styled panels and tables
- ChatUI: Interactive chat interface with command handling
- AnalysisDashboard: Live progress dashboard
- BenderBoxApp: Main application entry point
- BenderBoxTUI: Textual-based TUI application (requires textual)
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
    # TUI (requires textual)
    "BenderBoxTUI",
    "DashboardScreen",
    "AnalysisScreen",
    "ReportsScreen",
    "ChatScreen",
    "TEXTUAL_AVAILABLE",
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

    # TUI (requires textual)
    if name in (
        "BenderBoxTUI",
        "DashboardScreen",
        "AnalysisScreen",
        "ReportsScreen",
        "ChatScreen",
        "TEXTUAL_AVAILABLE",
    ):
        try:
            from benderbox.ui.tui import TEXTUAL_AVAILABLE
            if name == "TEXTUAL_AVAILABLE":
                return TEXTUAL_AVAILABLE
            if TEXTUAL_AVAILABLE:
                from benderbox.ui.tui import (
                    BenderBoxTUI,
                    DashboardScreen,
                    AnalysisScreen,
                    ReportsScreen,
                    ChatScreen,
                )
                return locals().get(name)
            return None
        except ImportError:
            if name == "TEXTUAL_AVAILABLE":
                return False
            return None

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
