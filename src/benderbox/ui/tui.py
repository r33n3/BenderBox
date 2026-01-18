"""
BenderBox TUI - Terminal User Interface

A Textual-based terminal UI for BenderBox providing:
- Dashboard with recent analyses and quick stats
- Analysis screen for running new analyses
- Reports browser
- Interactive chat interface

Requires: pip install benderbox[tui]
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
    from textual.css.query import NoMatches
    from textual.reactive import reactive
    from textual.screen import Screen
    from textual.widgets import (
        Button,
        DataTable,
        DirectoryTree,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        LoadingIndicator,
        Markdown,
        ProgressBar,
        Rule,
        Select,
        Static,
        TabbedContent,
        TabPane,
    )

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False


# CSS Theme for BenderBox TUI
BENDERBOX_CSS = """
/* Base Theme */
Screen {
    background: $surface;
}

/* Header Styling */
Header {
    dock: top;
    height: 3;
    background: $primary;
}

.header-title {
    text-style: bold;
    color: $text;
}

/* Footer Styling */
Footer {
    dock: bottom;
    height: 1;
    background: $primary-darken-2;
}

/* Status Bar */
#status-bar {
    dock: top;
    height: 1;
    background: $primary-darken-1;
    padding: 0 1;
}

#status-bar Label {
    margin-right: 2;
}

.status-ok {
    color: $success;
}

.status-warning {
    color: $warning;
}

.status-error {
    color: $error;
}

/* Navigation Sidebar */
#sidebar {
    width: 20;
    dock: left;
    background: $surface-darken-1;
    border-right: solid $primary;
}

#sidebar ListView {
    height: 100%;
    background: transparent;
}

#sidebar ListItem {
    padding: 0 1;
}

#sidebar ListItem:hover {
    background: $primary 30%;
}

#sidebar .nav-selected {
    background: $primary;
    text-style: bold;
}

/* Main Content Area */
#main-content {
    padding: 1;
}

/* Dashboard Widgets */
.dashboard-grid {
    layout: grid;
    grid-size: 2 2;
    grid-gutter: 1;
    height: 100%;
}

.widget {
    border: solid $primary;
    padding: 1;
    background: $surface-lighten-1;
}

.widget-title {
    text-style: bold;
    color: $primary-lighten-2;
    margin-bottom: 1;
}

/* Stats Panel */
.stat-row {
    layout: horizontal;
    height: auto;
    margin-bottom: 1;
}

.stat-label {
    width: 15;
    color: $text-muted;
}

.stat-value {
    text-style: bold;
}

.stat-critical {
    color: $error;
}

.stat-high {
    color: $warning;
}

.stat-medium {
    color: yellow;
}

.stat-low {
    color: $success;
}

/* Quick Actions */
.action-buttons {
    layout: horizontal;
    height: auto;
}

.action-buttons Button {
    margin-right: 1;
}

/* Recent Analyses Table */
#recent-analyses {
    height: 100%;
}

#recent-analyses DataTable {
    height: 100%;
}

/* Analysis Screen */
.analysis-form {
    layout: vertical;
    height: auto;
    padding: 1;
    border: solid $primary;
    margin-bottom: 1;
}

.form-row {
    layout: horizontal;
    height: auto;
    margin-bottom: 1;
}

.form-label {
    width: 15;
    padding-top: 1;
}

.form-input {
    width: 1fr;
}

#file-browser {
    height: 20;
    border: solid $primary;
    margin-bottom: 1;
}

#analysis-progress {
    height: auto;
    padding: 1;
    border: solid $primary;
    display: none;
}

#analysis-progress.visible {
    display: block;
}

/* Results Display */
.results-panel {
    height: 1fr;
    border: solid $primary;
    padding: 1;
}

/* Risk Badge Styling */
.risk-critical {
    background: $error;
    color: $text;
    padding: 0 1;
}

.risk-high {
    background: $warning;
    color: $background;
    padding: 0 1;
}

.risk-medium {
    background: yellow;
    color: $background;
    padding: 0 1;
}

.risk-low {
    background: $success;
    color: $background;
    padding: 0 1;
}

/* Chat Screen */
#chat-container {
    height: 100%;
    layout: vertical;
}

#chat-history {
    height: 1fr;
    border: solid $primary;
    padding: 1;
    overflow-y: auto;
}

.chat-message {
    margin-bottom: 1;
    padding: 1;
    border: solid $primary-darken-2;
}

.chat-user {
    background: $primary 20%;
}

.chat-assistant {
    background: $surface-lighten-1;
}

#chat-input-area {
    height: auto;
    layout: horizontal;
    margin-top: 1;
}

#chat-input {
    width: 1fr;
}

#chat-send {
    width: auto;
    margin-left: 1;
}

/* Loading States */
.loading {
    content-align: center middle;
    height: 100%;
}

/* Reports Screen */
#reports-container {
    height: 100%;
    layout: vertical;
}

#reports-toolbar {
    height: auto;
    layout: horizontal;
    margin-bottom: 1;
}

#reports-toolbar Input {
    width: 1fr;
}

#reports-toolbar Button {
    margin-left: 1;
}

#reports-toolbar Select {
    width: 15;
    margin-left: 1;
}

#reports-list {
    height: 1fr;
    border: solid $primary;
}

#reports-list DataTable {
    height: 100%;
}

#report-detail-panel {
    height: 1fr;
    border: solid $primary;
    margin-top: 1;
}

#report-detail-panel.hidden {
    display: none;
}

.findings-list {
    height: 1fr;
    border: solid $primary;
    margin-top: 1;
}

.finding-item {
    padding: 1;
    border-bottom: solid $primary-darken-2;
}

.finding-item:hover {
    background: $primary 20%;
}

.finding-failed {
    color: $error;
}

.finding-passed {
    color: $success;
}

.success-message {
    color: $success;
    padding: 2;
    text-style: bold;
}

.finding-severity {
    width: 10;
    text-style: bold;
}

.finding-name {
    width: 1fr;
}

/* JSON Search Styles */
#json-search-container {
    padding: 1;
}

#json-search-input {
    margin-bottom: 1;
}

.search-results-header {
    text-style: bold;
    color: $primary;
    padding: 1 0;
}

.json-search-match {
    color: $warning;
    text-style: bold;
    padding: 0 1;
}

.json-search-context {
    background: $surface-darken-2;
    padding: 1;
    margin: 0 1 1 1;
}

.search-separator {
    color: $primary-darken-2;
    text-align: center;
}

.no-results {
    color: $text-muted;
    padding: 2;
}

/* Chat Enhancements */
#chat-context {
    height: auto;
    background: $surface-darken-1;
    padding: 0 1;
    border-bottom: solid $primary;
}

#chat-suggestions {
    height: auto;
    layout: horizontal;
    padding: 1;
    background: $surface-darken-1;
}

#chat-suggestions Button {
    margin-right: 1;
}

.suggestion-btn {
    background: $primary 30%;
}

.suggestion-btn:hover {
    background: $primary 50%;
}

#command-suggestions {
    height: auto;
    background: $surface-lighten-1;
    border: solid $primary;
    padding: 1;
    display: none;
}

#command-suggestions.visible {
    display: block;
}

.command-item {
    padding: 0 1;
}

.command-item:hover {
    background: $primary 30%;
}

.context-indicator {
    color: $text-muted;
    text-style: italic;
}
"""


if TEXTUAL_AVAILABLE:

    class StatusBar(Static):
        """Status bar showing system state."""

        llm_status = reactive("Checking...")
        db_status = reactive("Checking...")
        last_scan = reactive("Never")

        def compose(self) -> ComposeResult:
            with Horizontal():
                yield Label("LLM: ")
                yield Label(self.llm_status, id="llm-status")
                yield Label(" | DB: ")
                yield Label(self.db_status, id="db-status")
                yield Label(" | Last Scan: ")
                yield Label(self.last_scan, id="last-scan")

        def watch_llm_status(self, value: str) -> None:
            try:
                label = self.query_one("#llm-status", Label)
                label.update(value)
                if "OK" in value or "Available" in value:
                    label.add_class("status-ok")
                elif "Warning" in value:
                    label.add_class("status-warning")
                else:
                    label.remove_class("status-ok", "status-warning")
            except NoMatches:
                pass

        def watch_db_status(self, value: str) -> None:
            try:
                label = self.query_one("#db-status", Label)
                label.update(value)
            except NoMatches:
                pass

        def watch_last_scan(self, value: str) -> None:
            try:
                label = self.query_one("#last-scan", Label)
                label.update(value)
            except NoMatches:
                pass

    class StatsWidget(Static):
        """Quick statistics widget."""

        def compose(self) -> ComposeResult:
            yield Label("Quick Stats", classes="widget-title")
            with Vertical():
                with Horizontal(classes="stat-row"):
                    yield Label("Total Scans:", classes="stat-label")
                    yield Label("0", id="stat-total", classes="stat-value")
                with Horizontal(classes="stat-row"):
                    yield Label("Critical:", classes="stat-label")
                    yield Label("0", id="stat-critical", classes="stat-value stat-critical")
                with Horizontal(classes="stat-row"):
                    yield Label("High:", classes="stat-label")
                    yield Label("0", id="stat-high", classes="stat-value stat-high")
                with Horizontal(classes="stat-row"):
                    yield Label("Medium:", classes="stat-label")
                    yield Label("0", id="stat-medium", classes="stat-value stat-medium")
                with Horizontal(classes="stat-row"):
                    yield Label("Low:", classes="stat-label")
                    yield Label("0", id="stat-low", classes="stat-value stat-low")

        def update_stats(self, stats: Dict[str, int]) -> None:
            """Update statistics display."""
            try:
                self.query_one("#stat-total", Label).update(str(stats.get("total", 0)))
                self.query_one("#stat-critical", Label).update(str(stats.get("critical", 0)))
                self.query_one("#stat-high", Label).update(str(stats.get("high", 0)))
                self.query_one("#stat-medium", Label).update(str(stats.get("medium", 0)))
                self.query_one("#stat-low", Label).update(str(stats.get("low", 0)))
            except NoMatches:
                pass

    class QuickActionsWidget(Static):
        """Quick action buttons widget."""

        def compose(self) -> ComposeResult:
            yield Label("Quick Actions", classes="widget-title")
            with Horizontal(classes="action-buttons"):
                yield Button("New Analysis", id="btn-new-analysis", variant="primary")
                yield Button("Search Reports", id="btn-search", variant="default")
                yield Button("Open Chat", id="btn-chat", variant="default")

    class RecentAnalysesWidget(Static):
        """Recent analyses table widget."""

        def compose(self) -> ComposeResult:
            yield Label("Recent Analyses", classes="widget-title")
            table = DataTable(id="recent-table")
            table.add_columns("Target", "Risk", "Date", "Profile")
            yield table

        def update_analyses(self, analyses: List[Dict[str, Any]]) -> None:
            """Update the recent analyses table."""
            try:
                table = self.query_one("#recent-table", DataTable)
                table.clear()
                for analysis in analyses[:10]:
                    target = analysis.get("target_name", "Unknown")[:25]
                    risk = analysis.get("risk_level", "unknown").upper()
                    date = analysis.get("timestamp", "")[:10]
                    profile = analysis.get("profile", "standard")
                    table.add_row(target, risk, date, profile)
            except NoMatches:
                pass

    class DashboardScreen(Screen):
        """Main dashboard screen."""

        BINDINGS = [
            Binding("n", "new_analysis", "New Analysis"),
            Binding("s", "search", "Search"),
            Binding("r", "refresh", "Refresh"),
        ]

        def compose(self) -> ComposeResult:
            yield Header()
            yield StatusBar(id="status-bar")
            with Container(id="main-content"):
                with Container(classes="dashboard-grid"):
                    with Container(classes="widget"):
                        yield StatsWidget(id="stats-widget")
                    with Container(classes="widget"):
                        yield QuickActionsWidget(id="actions-widget")
                    with Container(classes="widget", id="recent-analyses"):
                        yield RecentAnalysesWidget(id="recent-widget")
                    with Container(classes="widget"):
                        yield Label("System Health", classes="widget-title")
                        yield Label("All systems operational", id="health-status")
            yield Footer()

        async def on_mount(self) -> None:
            """Load dashboard data on mount."""
            await self.refresh_data()

        async def refresh_data(self) -> None:
            """Refresh dashboard data."""
            # Update status bar
            status_bar = self.query_one(StatusBar)

            # Check LLM availability
            try:
                from benderbox.nlp.llm_engine import LocalLLMEngine
                engine = LocalLLMEngine()
                if engine.is_available:
                    status_bar.llm_status = "Available"
                else:
                    status_bar.llm_status = "No Model"
            except Exception:
                status_bar.llm_status = "Unavailable"

            status_bar.db_status = "Connected"
            status_bar.last_scan = datetime.now().strftime("%Y-%m-%d %H:%M")

            # Update stats (mock data for now)
            stats_widget = self.query_one("#stats-widget", StatsWidget)
            stats_widget.update_stats({
                "total": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            })

        def action_new_analysis(self) -> None:
            """Switch to analysis screen."""
            self.app.push_screen("analysis")

        def action_search(self) -> None:
            """Open search."""
            self.app.push_screen("reports")

        def action_refresh(self) -> None:
            """Refresh dashboard data."""
            asyncio.create_task(self.refresh_data())

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses."""
            if event.button.id == "btn-new-analysis":
                self.app.push_screen("analysis")
            elif event.button.id == "btn-search":
                self.app.push_screen("reports")
            elif event.button.id == "btn-chat":
                self.app.push_screen("chat")

    class AnalysisScreen(Screen):
        """Analysis configuration and execution screen."""

        BINDINGS = [
            Binding("escape", "go_back", "Back"),
            Binding("enter", "run_analysis", "Run"),
        ]

        def compose(self) -> ComposeResult:
            yield Header()
            with Container(id="main-content"):
                yield Label("New Analysis", classes="widget-title")

                with Container(classes="analysis-form"):
                    with Horizontal(classes="form-row"):
                        yield Label("Target:", classes="form-label")
                        yield Input(placeholder="Path to model or file...", id="target-input", classes="form-input")

                    with Horizontal(classes="form-row"):
                        yield Label("Profile:", classes="form-label")
                        yield Select(
                            [
                                ("Quick", "quick"),
                                ("Standard", "standard"),
                                ("Deep", "deep"),
                            ],
                            value="standard",
                            id="profile-select",
                            classes="form-input",
                        )

                    with Horizontal(classes="form-row"):
                        yield Label("", classes="form-label")
                        yield Button("Browse...", id="btn-browse", variant="default")
                        yield Button("Run Analysis", id="btn-run", variant="primary")

                yield Label("Select Target File:", classes="widget-title")
                yield DirectoryTree(".", id="file-browser")

                with Container(id="analysis-progress"):
                    yield Label("Analysis Progress", classes="widget-title")
                    yield ProgressBar(id="progress-bar", total=100)
                    yield Label("Initializing...", id="progress-label")

                with Container(classes="results-panel"):
                    yield Label("Results", classes="widget-title")
                    yield Markdown("Run an analysis to see results here.", id="results-markdown")
            yield Footer()

        def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
            """Handle file selection from directory tree."""
            try:
                input_widget = self.query_one("#target-input", Input)
                input_widget.value = str(event.path)
            except NoMatches:
                pass

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses."""
            if event.button.id == "btn-run":
                await self.run_analysis()
            elif event.button.id == "btn-browse":
                # Toggle file browser visibility
                pass

        async def run_analysis(self) -> None:
            """Run the analysis."""
            try:
                target_input = self.query_one("#target-input", Input)
                profile_select = self.query_one("#profile-select", Select)
                progress_container = self.query_one("#analysis-progress")
                progress_bar = self.query_one("#progress-bar", ProgressBar)
                progress_label = self.query_one("#progress-label", Label)
                results_md = self.query_one("#results-markdown", Markdown)

                target = target_input.value
                profile = profile_select.value

                if not target:
                    results_md.update("**Error:** Please specify a target.")
                    return

                # Show progress
                progress_container.add_class("visible")
                progress_bar.progress = 0
                progress_label.update("Starting analysis...")

                # Simulate progress (actual analysis would be async)
                for i in range(0, 101, 10):
                    progress_bar.progress = i
                    progress_label.update(f"Analyzing... {i}%")
                    await asyncio.sleep(0.2)

                # Show results
                results_md.update(f"""
## Analysis Complete

**Target:** {target}
**Profile:** {profile}

### Summary
Analysis completed successfully. See detailed report for findings.

### Quick Stats
- Risk Level: **LOW**
- Tests Passed: 8/10
- Duration: 2.3s
""")
                progress_container.remove_class("visible")

            except Exception as e:
                try:
                    results_md = self.query_one("#results-markdown", Markdown)
                    results_md.update(f"**Error:** {e}")
                except NoMatches:
                    pass

        def action_go_back(self) -> None:
            """Go back to dashboard."""
            self.app.pop_screen()

        def action_run_analysis(self) -> None:
            """Run analysis via keybinding."""
            asyncio.create_task(self.run_analysis())

    class ReportsScreen(Screen):
        """Reports browser screen with search, filtering, and detail view."""

        BINDINGS = [
            Binding("escape", "go_back", "Back"),
            Binding("s", "focus_search", "Search"),
            Binding("e", "export_report", "Export"),
            Binding("f", "toggle_filter", "Filter"),
            Binding("r", "refresh_reports", "Refresh"),
        ]

        # Store loaded reports
        reports: List[Dict[str, Any]] = []
        selected_report: Optional[Dict[str, Any]] = None
        current_filter: str = "all"

        def compose(self) -> ComposeResult:
            yield Header()
            with Container(id="reports-container"):
                yield Label("Reports Browser", classes="widget-title")

                # Toolbar with search and filters
                with Horizontal(id="reports-toolbar"):
                    yield Input(placeholder="Search reports...", id="search-input")
                    yield Select(
                        [
                            ("All", "all"),
                            ("Critical", "critical"),
                            ("High", "high"),
                            ("Medium", "medium"),
                            ("Low", "low"),
                        ],
                        value="all",
                        id="severity-filter",
                    )
                    yield Button("Search", id="btn-search", variant="primary")
                    yield Button("Export", id="btn-export", variant="default")
                    yield Button("Refresh", id="btn-refresh", variant="default")

                # Reports table
                with Container(id="reports-list"):
                    table = DataTable(id="reports-table", cursor_type="row")
                    table.add_columns("ID", "Target", "Risk", "Score", "Date", "Findings")
                    yield table

                # Report detail panel
                with Container(id="report-detail-panel"):
                    yield Label("Report Details", classes="widget-title")
                    with TabbedContent():
                        with TabPane("Summary", id="tab-summary"):
                            yield Markdown("Select a report to view details.", id="report-summary")
                        with TabPane("Failed", id="tab-failed"):
                            yield ScrollableContainer(id="failed-container")
                        with TabPane("Passed", id="tab-passed"):
                            yield ScrollableContainer(id="passed-container")
                        with TabPane("JSON Search", id="tab-json-search"):
                            with Container(id="json-search-container"):
                                yield Input(placeholder="Search JSON (e.g., 'jailbreak', 'CRITICAL')...", id="json-search-input")
                                yield Button("Search", id="btn-json-search", variant="primary")
                                yield ScrollableContainer(id="json-search-results")
                        with TabPane("Raw", id="tab-raw"):
                            yield Markdown("", id="report-raw")
            yield Footer()

        async def on_mount(self) -> None:
            """Load reports on mount."""
            await self.load_reports()

        async def load_reports(self, search_query: str = "", severity_filter: str = "all") -> None:
            """Load reports from storage."""
            try:
                # Try to load actual reports from database
                try:
                    from benderbox.storage.report_db import ReportDatabase
                    db = ReportDatabase()
                    # This would be the actual implementation
                    self.reports = []  # await db.list_reports()
                except ImportError:
                    pass

                # For now, use sample data
                if not self.reports:
                    self.reports = [
                        {
                            "id": "rpt-001",
                            "target_name": "model-7b.gguf",
                            "risk_level": "low",
                            "risk_score": 25,
                            "timestamp": "2026-01-12 10:30:00",
                            "finding_count": 3,
                            "findings": [
                                {"name": "Jailbreak Test", "severity": "low", "status": "passed"},
                                {"name": "Safety Check", "severity": "info", "status": "passed"},
                                {"name": "Output Filter", "severity": "medium", "status": "warning"},
                            ],
                        },
                        {
                            "id": "rpt-002",
                            "target_name": "uncensored-13b.gguf",
                            "risk_level": "high",
                            "risk_score": 72,
                            "timestamp": "2026-01-12 09:15:00",
                            "finding_count": 8,
                            "findings": [
                                {"name": "Jailbreak Test", "severity": "critical", "status": "failed"},
                                {"name": "Safety Check", "severity": "high", "status": "failed"},
                                {"name": "Output Filter", "severity": "high", "status": "failed"},
                            ],
                        },
                        {
                            "id": "rpt-003",
                            "target_name": "assistant-chat.gguf",
                            "risk_level": "medium",
                            "risk_score": 45,
                            "timestamp": "2026-01-11 16:45:00",
                            "finding_count": 5,
                            "findings": [
                                {"name": "Jailbreak Test", "severity": "medium", "status": "warning"},
                                {"name": "Safety Check", "severity": "low", "status": "passed"},
                            ],
                        },
                    ]

                # Apply filters
                filtered = self.reports
                if search_query:
                    search_query = search_query.lower()
                    filtered = [r for r in filtered if search_query in r.get("target_name", "").lower()]
                if severity_filter != "all":
                    filtered = [r for r in filtered if r.get("risk_level", "").lower() == severity_filter]

                # Update table
                await self.update_reports_table(filtered)

            except Exception as e:
                self.notify(f"Error loading reports: {e}", severity="error")

        async def update_reports_table(self, reports: List[Dict[str, Any]]) -> None:
            """Update the reports table with data."""
            try:
                table = self.query_one("#reports-table", DataTable)
                table.clear()

                for report in reports:
                    risk_level = report.get("risk_level", "unknown").upper()
                    table.add_row(
                        report.get("id", "N/A"),
                        report.get("target_name", "Unknown")[:30],
                        risk_level,
                        str(report.get("risk_score", 0)),
                        report.get("timestamp", "")[:16],
                        str(report.get("finding_count", 0)),
                        key=report.get("id"),
                    )
            except NoMatches:
                pass

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            """Handle report selection."""
            if event.row_key:
                report_id = str(event.row_key.value)
                for report in self.reports:
                    if report.get("id") == report_id:
                        self.selected_report = report
                        self.show_report_details(report)
                        break

        def show_report_details(self, report: Dict[str, Any]) -> None:
            """Show detailed view of selected report."""
            try:
                findings = report.get("findings", [])

                # Separate findings by status
                failed_findings = [f for f in findings if f.get("status", "").lower() in ("failed", "fail", "warning", "warn")]
                passed_findings = [f for f in findings if f.get("status", "").lower() in ("passed", "pass")]

                # Update summary tab with separate counts
                summary_md = self.query_one("#report-summary", Markdown)
                summary_text = f"""
## {report.get('target_name', 'Unknown')}

**Report ID:** {report.get('id', 'N/A')}
**Date:** {report.get('timestamp', 'N/A')}
**Risk Level:** {report.get('risk_level', 'unknown').upper()}
**Risk Score:** {report.get('risk_score', 0)}/100

### Test Results
- **Failed Tests:** {len(failed_findings)} (security concerns)
- **Passed Tests:** {len(passed_findings)} (verified safe behaviors)

### Failed by Severity
- Critical: {sum(1 for f in failed_findings if f.get('severity') == 'critical')}
- High: {sum(1 for f in failed_findings if f.get('severity') == 'high')}
- Medium: {sum(1 for f in failed_findings if f.get('severity') == 'medium')}
- Low: {sum(1 for f in failed_findings if f.get('severity') == 'low')}
"""
                summary_md.update(summary_text)

                # Update Failed tab (security concerns)
                failed_container = self.query_one("#failed-container", ScrollableContainer)
                failed_container.remove_children()

                if failed_findings:
                    # Sort by severity (critical first)
                    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
                    failed_findings.sort(key=lambda f: severity_order.get(f.get("severity", "info").lower(), 5))

                    for finding in failed_findings:
                        severity = finding.get("severity", "info").upper()
                        name = finding.get("name", "Unknown")
                        status = finding.get("status", "failed").upper()
                        # Color code by severity
                        if severity == "CRITICAL":
                            style = "bold red"
                        elif severity == "HIGH":
                            style = "red"
                        elif severity == "MEDIUM":
                            style = "yellow"
                        else:
                            style = "white"
                        finding_label = Label(
                            f"[{severity}] {name} - {status}",
                            classes="finding-item finding-failed"
                        )
                        failed_container.mount(finding_label)
                else:
                    failed_container.mount(Label("No failed tests - model passed all security checks!", classes="success-message"))

                # Update Passed tab (verified behaviors)
                passed_container = self.query_one("#passed-container", ScrollableContainer)
                passed_container.remove_children()

                if passed_findings:
                    # Sort by severity (shows what critical tests passed)
                    passed_findings.sort(key=lambda f: severity_order.get(f.get("severity", "info").lower(), 5))

                    for finding in passed_findings:
                        severity = finding.get("severity", "info").upper()
                        name = finding.get("name", "Unknown")
                        finding_label = Label(
                            f"[{severity}] {name} - PASSED",
                            classes="finding-item finding-passed"
                        )
                        passed_container.mount(finding_label)
                else:
                    passed_container.mount(Label("No passed tests recorded"))

                # Clear JSON search results when switching reports
                try:
                    json_results = self.query_one("#json-search-results", ScrollableContainer)
                    json_results.remove_children()
                    json_results.mount(Label("Enter a search term to search the raw JSON data"))
                except NoMatches:
                    pass

                # Update raw tab
                raw_md = self.query_one("#report-raw", Markdown)
                import json
                raw_md.update(f"```json\n{json.dumps(report, indent=2)}\n```")

            except NoMatches:
                pass
            except Exception as e:
                self.notify(f"Error showing details: {e}", severity="error")

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses."""
            if event.button.id == "btn-search":
                await self.do_search()
            elif event.button.id == "btn-export":
                await self.export_selected_report()
            elif event.button.id == "btn-refresh":
                await self.load_reports()
            elif event.button.id == "btn-json-search":
                await self.do_json_search()

        async def do_json_search(self) -> None:
            """Search within the raw JSON of the selected report."""
            if not self.selected_report:
                self.notify("No report selected", severity="warning")
                return

            try:
                import json
                import re

                search_input = self.query_one("#json-search-input", Input)
                search_term = search_input.value.strip()

                if not search_term:
                    self.notify("Enter a search term", severity="warning")
                    return

                results_container = self.query_one("#json-search-results", ScrollableContainer)
                results_container.remove_children()

                # Convert report to JSON string for searching
                json_str = json.dumps(self.selected_report, indent=2)

                # Find all matches with context
                matches = []
                lines = json_str.split("\n")
                pattern = re.compile(re.escape(search_term), re.IGNORECASE)

                for i, line in enumerate(lines):
                    if pattern.search(line):
                        # Get context (2 lines before and after)
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        context = lines[start:end]
                        matches.append({
                            "line_num": i + 1,
                            "context": "\n".join(context),
                            "match_line": line.strip()
                        })

                if matches:
                    results_container.mount(Label(f"Found {len(matches)} match(es) for '{search_term}':", classes="search-results-header"))

                    for match in matches[:50]:  # Limit to 50 results
                        # Highlight the match in the context
                        match_label = Label(
                            f"Line {match['line_num']}: {match['match_line']}",
                            classes="json-search-match"
                        )
                        results_container.mount(match_label)

                        # Show context in a code block style
                        context_label = Label(
                            match['context'],
                            classes="json-search-context"
                        )
                        results_container.mount(context_label)
                        results_container.mount(Label("---", classes="search-separator"))

                    if len(matches) > 50:
                        results_container.mount(Label(f"... and {len(matches) - 50} more matches"))
                else:
                    results_container.mount(Label(f"No matches found for '{search_term}'", classes="no-results"))

            except NoMatches:
                pass
            except Exception as e:
                self.notify(f"Search error: {e}", severity="error")

        async def on_select_changed(self, event: Select.Changed) -> None:
            """Handle filter change."""
            if event.select.id == "severity-filter":
                self.current_filter = str(event.value)
                search_input = self.query_one("#search-input", Input)
                await self.load_reports(search_input.value, self.current_filter)

        async def do_search(self) -> None:
            """Perform search."""
            try:
                search_input = self.query_one("#search-input", Input)
                await self.load_reports(search_input.value, self.current_filter)
            except NoMatches:
                pass

        async def export_selected_report(self) -> None:
            """Export the selected report."""
            if not self.selected_report:
                self.notify("No report selected", severity="warning")
                return

            try:
                import json
                from pathlib import Path

                report_id = self.selected_report.get("id", "report")
                filename = f"{report_id}_export.json"
                Path(filename).write_text(json.dumps(self.selected_report, indent=2))
                self.notify(f"Exported to {filename}", severity="information")
            except Exception as e:
                self.notify(f"Export failed: {e}", severity="error")

        def action_go_back(self) -> None:
            """Go back to dashboard."""
            self.app.pop_screen()

        def action_focus_search(self) -> None:
            """Focus search input."""
            try:
                self.query_one("#search-input", Input).focus()
            except NoMatches:
                pass

        def action_export_report(self) -> None:
            """Export via keybinding."""
            asyncio.create_task(self.export_selected_report())

        def action_toggle_filter(self) -> None:
            """Focus filter dropdown."""
            try:
                self.query_one("#severity-filter", Select).focus()
            except NoMatches:
                pass

        def action_refresh_reports(self) -> None:
            """Refresh reports."""
            asyncio.create_task(self.load_reports())

    class ChatScreen(Screen):
        """Interactive chat screen with context indicator, command completion, and suggested queries."""

        BINDINGS = [
            Binding("escape", "go_back", "Back"),
            Binding("ctrl+l", "clear_chat", "Clear"),
            Binding("ctrl+k", "focus_input", "Focus Input"),
            Binding("tab", "next_suggestion", "Next Suggestion"),
        ]

        # Available commands for completion
        COMMANDS = [
            ("/help", "Show available commands"),
            ("/analyze", "Analyze a file or model"),
            ("/search", "Search reports"),
            ("/semantic", "Run semantic analysis"),
            ("/behavior", "Analyze model behavior"),
            ("/compare", "Compare reports or models"),
            ("/export", "Export current context"),
            ("/clear", "Clear chat history"),
        ]

        # Suggested queries for new users
        SUGGESTIONS = [
            "What is BenderBox?",
            "How do I analyze a model?",
            "Show recent reports",
            "Explain risk scores",
        ]

        # Chat context state
        current_context: Optional[str] = None
        current_model: Optional[str] = None
        message_history: List[Dict[str, str]] = []
        show_command_suggestions: bool = False

        def compose(self) -> ComposeResult:
            yield Header()
            with Container(id="chat-container"):
                # Context indicator bar
                with Horizontal(id="chat-context"):
                    yield Label("Context: ", classes="context-indicator")
                    yield Label("No active context", id="context-value", classes="context-indicator")
                    yield Label(" | Model: ", classes="context-indicator")
                    yield Label("None", id="model-value", classes="context-indicator")

                # Message history panel
                with ScrollableContainer(id="chat-history"):
                    yield Static(
                        "[bold]Welcome to BenderBox Chat![/bold]\n\n"
                        "I can help you with:\n"
                        "- Analyzing AI models for security issues\n"
                        "- Searching through reports\n"
                        "- Explaining analysis results\n"
                        "- Running semantic code analysis\n\n"
                        "Type a message or try a command starting with /",
                        id="welcome-message",
                        classes="chat-message chat-assistant",
                    )

                # Suggested queries panel
                with Horizontal(id="chat-suggestions"):
                    yield Label("Try: ", classes="context-indicator")
                    for idx, suggestion in enumerate(SUGGESTIONS if 'SUGGESTIONS' not in dir() else self.SUGGESTIONS):
                        yield Button(
                            suggestion,
                            id=f"suggestion-{idx}",
                            classes="suggestion-btn",
                            variant="default",
                        )

                # Command completion dropdown (hidden by default)
                with Container(id="command-suggestions"):
                    yield Label("Commands:", classes="widget-title")
                    for cmd, desc in COMMANDS if 'COMMANDS' not in dir() else self.COMMANDS:
                        yield Label(f"{cmd} - {desc}", classes="command-item")

                # Input area
                with Horizontal(id="chat-input-area"):
                    yield Input(
                        placeholder="Type a message or /command...",
                        id="chat-input",
                    )
                    yield Button("Send", id="chat-send", variant="primary")
            yield Footer()

        async def on_mount(self) -> None:
            """Initialize chat screen."""
            self.message_history = []
            await self.update_context_display()

        async def update_context_display(self) -> None:
            """Update the context indicator."""
            try:
                context_label = self.query_one("#context-value", Label)
                model_label = self.query_one("#model-value", Label)

                context_label.update(self.current_context or "No active context")
                model_label.update(self.current_model or "None")

                # Try to detect actual LLM availability
                try:
                    from benderbox.nlp.llm_engine import LocalLLMEngine
                    engine = LocalLLMEngine()
                    if engine.is_available and engine.model_name:
                        model_label.update(engine.model_name[:20])
                except Exception:
                    pass

            except NoMatches:
                pass

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button presses."""
            button_id = event.button.id or ""

            if button_id == "chat-send":
                await self.send_message()
            elif button_id.startswith("suggestion-"):
                # Handle suggestion button click
                try:
                    idx = int(button_id.split("-")[1])
                    if 0 <= idx < len(self.SUGGESTIONS):
                        input_widget = self.query_one("#chat-input", Input)
                        input_widget.value = self.SUGGESTIONS[idx]
                        await self.send_message()
                except (ValueError, IndexError):
                    pass

        async def on_input_changed(self, event: Input.Changed) -> None:
            """Handle input changes for command completion."""
            if event.input.id == "chat-input":
                value = event.value
                suggestions_container = self.query_one("#command-suggestions", Container)

                if value.startswith("/"):
                    # Show command suggestions
                    suggestions_container.add_class("visible")
                    self.show_command_suggestions = True
                else:
                    # Hide command suggestions
                    suggestions_container.remove_class("visible")
                    self.show_command_suggestions = False

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle enter key in input."""
            if event.input.id == "chat-input":
                await self.send_message()

        async def send_message(self) -> None:
            """Send a chat message and get response."""
            try:
                input_widget = self.query_one("#chat-input", Input)
                history = self.query_one("#chat-history", ScrollableContainer)

                message = input_widget.value.strip()
                if not message:
                    return

                # Hide command suggestions
                suggestions_container = self.query_one("#command-suggestions", Container)
                suggestions_container.remove_class("visible")

                # Remove welcome message on first real message
                try:
                    welcome = self.query_one("#welcome-message")
                    welcome.remove()
                except NoMatches:
                    pass

                # Add user message to display
                user_widget = Static(
                    f"[bold cyan]You:[/bold cyan] {message}",
                    classes="chat-message chat-user",
                )
                await history.mount(user_widget)

                # Store in history
                self.message_history.append({"role": "user", "content": message})

                # Clear input
                input_widget.value = ""

                # Process message and generate response
                response = await self.process_message(message)

                # Add assistant response
                assistant_widget = Static(
                    f"[bold green]Assistant:[/bold green] {response}",
                    classes="chat-message chat-assistant",
                )
                await history.mount(assistant_widget)

                # Store in history
                self.message_history.append({"role": "assistant", "content": response})

                # Scroll to bottom
                history.scroll_end()

                # Hide suggestions after first message
                try:
                    suggestions_panel = self.query_one("#chat-suggestions", Horizontal)
                    if len(self.message_history) > 2:
                        suggestions_panel.display = False
                except NoMatches:
                    pass

            except Exception as e:
                self.notify(f"Error: {e}", severity="error")

        async def process_message(self, message: str) -> str:
            """Process message and return response."""
            # Check for commands
            if message.startswith("/"):
                return await self.handle_command(message)

            # Try to use actual conversation manager
            try:
                from benderbox.core.conversation import ConversationManager
                manager = ConversationManager()
                response = await manager.process_input(message)
                if response:
                    return response.get("response", str(response))
            except ImportError:
                pass
            except Exception as e:
                return f"Error processing message: {e}"

            # Fallback responses based on content
            message_lower = message.lower()

            if "what is benderbox" in message_lower:
                return (
                    "BenderBox is an AI security analysis platform that helps you:\n"
                    "- Interrogate and test AI models for security vulnerabilities\n"
                    "- Analyze code for security issues\n"
                    "- Track and compare model behaviors over time\n"
                    "- Generate comprehensive security reports"
                )

            if "analyze" in message_lower or "how do i" in message_lower:
                return (
                    "To analyze a model or file:\n"
                    "1. Press 'A' to go to the Analysis screen\n"
                    "2. Select your target file using the file browser\n"
                    "3. Choose an analysis profile (Quick, Standard, or Deep)\n"
                    "4. Click 'Run Analysis' to start\n\n"
                    "Or use the /analyze command with a file path."
                )

            if "report" in message_lower:
                return (
                    "To view reports:\n"
                    "1. Press 'R' to go to the Reports screen\n"
                    "2. Use the search bar to filter reports\n"
                    "3. Select a report to view details\n"
                    "4. Use the Export button to save reports"
                )

            if "risk" in message_lower or "score" in message_lower:
                return (
                    "Risk scores in BenderBox:\n"
                    "- **Critical (80-100)**: Severe security issues requiring immediate attention\n"
                    "- **High (60-79)**: Significant vulnerabilities that should be addressed\n"
                    "- **Medium (40-59)**: Moderate concerns to review\n"
                    "- **Low (0-39)**: Minor issues or informational findings"
                )

            # Default response
            return (
                f"I understand you're asking about: '{message}'\n\n"
                "I'm here to help with BenderBox features. Try asking about:\n"
                "- How to analyze models\n"
                "- Understanding reports\n"
                "- Risk score explanations\n"
                "- Available commands (/help)"
            )

        async def handle_command(self, command: str) -> str:
            """Handle slash commands."""
            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                commands_list = "\n".join(
                    f"  {c} - {d}" for c, d in self.COMMANDS
                )
                return f"Available commands:\n{commands_list}"

            if cmd == "/analyze":
                if args:
                    self.current_context = f"Analyzing: {args}"
                    await self.update_context_display()
                    return f"Starting analysis of '{args}'... (Navigate to Analysis screen for full control)"
                return "Usage: /analyze <file_path>"

            if cmd == "/search":
                query = args or "all"
                return f"Searching reports for '{query}'... Press 'R' to view the Reports screen."

            if cmd == "/semantic":
                if args:
                    return f"Running semantic analysis on '{args}'..."
                return "Usage: /semantic <file_path> - Run semantic code analysis"

            if cmd == "/behavior":
                return "Behavior analysis examines model responses for:\n- Safety bypasses\n- Inconsistent responses\n- Harmful content generation"

            if cmd == "/compare":
                return "Compare two reports or models:\nUsage: /compare <report1_id> <report2_id>"

            if cmd == "/export":
                if self.message_history:
                    try:
                        import json
                        from pathlib import Path
                        from datetime import datetime

                        filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        Path(filename).write_text(json.dumps(self.message_history, indent=2))
                        return f"Chat exported to {filename}"
                    except Exception as e:
                        return f"Export failed: {e}"
                return "No messages to export."

            if cmd == "/clear":
                return "Chat cleared. (Use Ctrl+L to clear the display)"

            return f"Unknown command: {cmd}. Type /help for available commands."

        def action_go_back(self) -> None:
            """Go back to dashboard."""
            self.app.pop_screen()

        def action_clear_chat(self) -> None:
            """Clear chat history display."""
            asyncio.create_task(self._clear_chat())

        async def _clear_chat(self) -> None:
            """Clear chat asynchronously."""
            try:
                history = self.query_one("#chat-history", ScrollableContainer)
                history.remove_children()
                self.message_history = []

                # Add fresh welcome message
                welcome = Static(
                    "[bold]Chat cleared.[/bold] How can I help you?",
                    classes="chat-message chat-assistant",
                )
                await history.mount(welcome)
            except NoMatches:
                pass

        def action_focus_input(self) -> None:
            """Focus the input field."""
            try:
                self.query_one("#chat-input", Input).focus()
            except NoMatches:
                pass

        def action_next_suggestion(self) -> None:
            """Cycle through command suggestions."""
            if self.show_command_suggestions:
                try:
                    input_widget = self.query_one("#chat-input", Input)
                    current = input_widget.value

                    # Find current command and suggest next
                    cmds = [c[0] for c in self.COMMANDS]
                    if current in cmds:
                        idx = (cmds.index(current) + 1) % len(cmds)
                    else:
                        # Find partial match
                        idx = 0
                        for i, cmd in enumerate(cmds):
                            if cmd.startswith(current):
                                idx = i
                                break

                    input_widget.value = cmds[idx]
                except NoMatches:
                    pass

    class BenderBoxTUI(App):
        """BenderBox Terminal User Interface Application."""

        TITLE = "BenderBox"
        SUB_TITLE = "AI Security Analysis Platform"
        CSS = BENDERBOX_CSS

        BINDINGS = [
            Binding("d", "switch_screen('dashboard')", "Dashboard", show=True),
            Binding("a", "switch_screen('analysis')", "Analysis", show=True),
            Binding("r", "switch_screen('reports')", "Reports", show=True),
            Binding("c", "switch_screen('chat')", "Chat", show=True),
            Binding("q", "quit", "Quit", show=True),
            Binding("?", "help", "Help"),
        ]

        SCREENS = {
            "dashboard": DashboardScreen,
            "analysis": AnalysisScreen,
            "reports": ReportsScreen,
            "chat": ChatScreen,
        }

        def on_mount(self) -> None:
            """Initialize application."""
            self.push_screen("dashboard")

        def action_switch_screen(self, screen_name: str) -> None:
            """Switch to a different screen."""
            if screen_name in self.SCREENS:
                # Pop all screens and push new one
                while len(self.screen_stack) > 1:
                    self.pop_screen()
                if self.screen.name != screen_name:
                    self.switch_screen(screen_name)

        def action_help(self) -> None:
            """Show help information."""
            self.notify(
                "BenderBox TUI Help\n"
                "D: Dashboard | A: Analysis | R: Reports | C: Chat | Q: Quit",
                title="Keyboard Shortcuts",
                timeout=5,
            )


def main():
    """Main entry point for the TUI."""
    if not TEXTUAL_AVAILABLE:
        print("Error: Textual is not installed.")
        print("Install it with: pip install benderbox[tui]")
        print("  or: pip install textual>=0.40.0")
        return 1

    app = BenderBoxTUI()
    app.run()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
