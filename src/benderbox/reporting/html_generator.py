"""
HTML Report Generator for BenderBox

Generates comprehensive HTML security reports with:
- Responsive design
- Interactive elements (collapsible sections, search/filter)
- Dark/light mode toggle
- Risk visualizations with SVG charts
- Detailed findings sections
- PDF export support
"""

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HTMLSection:
    """A section in the HTML report."""

    id: str
    title: str
    content: str
    icon: str = ""


# Enhanced HTML5 template with interactive features
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #4a4a8a;
            --primary-dark: #1a1a2e;
            --secondary: #16213e;
            --success: #44cc44;
            --warning: #ffcc00;
            --danger: #ff4444;
            --danger-light: #ff8800;
            --bg: #f5f5f5;
            --card-bg: #ffffff;
            --text: #333333;
            --text-muted: #666666;
            --border: #dddddd;
            --code-bg: #f8f8f8;
        }}

        [data-theme="dark"] {{
            --primary: #7a7aba;
            --primary-dark: #2a2a4e;
            --secondary: #4a5a7e;
            --bg: #1a1a2e;
            --card-bg: #2a2a3e;
            --text: #e0e0e0;
            --text-muted: #a0a0a0;
            --border: #3a3a4e;
            --code-bg: #2a2a3e;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            transition: background 0.3s, color 0.3s;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        /* Toolbar */
        .toolbar {{
            position: sticky;
            top: 0;
            background: var(--card-bg);
            padding: 10px 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
            z-index: 100;
        }}

        .toolbar-left {{
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}

        .search-box {{
            padding: 8px 15px;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: var(--bg);
            color: var(--text);
            width: 250px;
            font-size: 0.9em;
        }}

        .search-box:focus {{
            outline: none;
            border-color: var(--primary);
        }}

        .btn {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
        }}

        .btn-primary {{
            background: var(--primary);
            color: white;
        }}

        .btn-primary:hover {{
            background: var(--primary-dark);
        }}

        .btn-outline {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
        }}

        .btn-outline:hover {{
            background: var(--bg);
        }}

        /* Table of Contents */
        .toc {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}

        .toc h3 {{
            margin-bottom: 15px;
            color: var(--primary);
        }}

        .toc ul {{
            list-style: none;
        }}

        .toc li {{
            margin: 8px 0;
        }}

        .toc a {{
            color: var(--text);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .toc a:hover {{
            color: var(--primary);
        }}

        .toc-badge {{
            font-size: 0.75em;
            padding: 2px 8px;
            border-radius: 10px;
            background: var(--bg);
        }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, var(--primary-dark), var(--primary));
            color: white;
            padding: 30px 20px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}

        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        .header .meta {{
            opacity: 0.9;
            font-size: 0.95em;
        }}

        /* Risk Badge */
        .risk-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9em;
        }}

        .risk-critical {{ background: var(--danger); color: white; }}
        .risk-high {{ background: var(--danger-light); color: white; }}
        .risk-medium {{ background: var(--warning); color: var(--primary-dark); }}
        .risk-low {{ background: var(--success); color: white; }}

        /* Cards */
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }}

        .card-header {{
            background: var(--primary);
            color: white;
            padding: 15px 20px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}

        .card-header::after {{
            content: '\\25BC';
            font-size: 0.8em;
            transition: transform 0.3s;
        }}

        .card.collapsed .card-header::after {{
            transform: rotate(-90deg);
        }}

        .card.collapsed .card-body {{
            display: none;
        }}

        .card-body {{
            padding: 20px;
        }}

        /* Grid */
        .grid {{
            display: grid;
            gap: 20px;
        }}

        .grid-2 {{
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }}

        .grid-3 {{
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }}

        /* Stats */
        .stat {{
            text-align: center;
            padding: 20px;
        }}

        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: var(--primary);
        }}

        .stat-label {{
            color: var(--text-muted);
            margin-top: 5px;
        }}

        /* Tables */
        .table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .table th, .table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        .table th {{
            background: var(--bg);
            font-weight: 600;
            color: var(--secondary);
        }}

        .table tr:hover {{
            background: rgba(0,0,0,0.02);
        }}

        [data-theme="dark"] .table tr:hover {{
            background: rgba(255,255,255,0.02);
        }}

        /* Findings */
        .finding {{
            border-left: 4px solid var(--border);
            padding: 15px 20px;
            margin-bottom: 15px;
            background: var(--bg);
            border-radius: 0 4px 4px 0;
            transition: all 0.2s;
        }}

        .finding.hidden {{
            display: none;
        }}

        .finding-critical {{ border-left-color: var(--danger); }}
        .finding-high {{ border-left-color: var(--danger-light); }}
        .finding-medium {{ border-left-color: var(--warning); }}
        .finding-low {{ border-left-color: var(--success); }}

        .finding-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}

        .finding-title {{
            font-weight: bold;
            color: var(--secondary);
        }}

        [data-theme="dark"] .finding-title {{
            color: var(--text);
        }}

        .finding-severity {{
            font-size: 0.85em;
            padding: 3px 10px;
            border-radius: 3px;
            text-transform: uppercase;
        }}

        .severity-critical {{ background: var(--danger); color: white; }}
        .severity-high {{ background: var(--danger-light); color: white; }}
        .severity-medium {{ background: var(--warning); color: var(--primary-dark); }}
        .severity-low {{ background: var(--success); color: white; }}
        .severity-info {{ background: var(--border); color: var(--text); }}

        .finding-meta {{
            color: var(--text-muted);
            font-size: 0.9em;
            margin-bottom: 10px;
        }}

        .finding-details {{
            background: var(--code-bg);
            padding: 10px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            overflow-x: auto;
            border: 1px solid var(--border);
        }}

        /* Risk Meter */
        .risk-meter {{
            width: 100%;
            height: 30px;
            background: linear-gradient(to right, var(--success), var(--warning), var(--danger));
            border-radius: 15px;
            position: relative;
            margin: 20px 0;
        }}

        .risk-marker {{
            position: absolute;
            top: -5px;
            width: 4px;
            height: 40px;
            background: var(--primary-dark);
            border-radius: 2px;
            transform: translateX(-50%);
        }}

        .risk-marker::after {{
            content: attr(data-score);
            position: absolute;
            top: 45px;
            left: 50%;
            transform: translateX(-50%);
            font-weight: bold;
            font-size: 0.9em;
        }}

        /* SVG Charts */
        .chart-container {{
            display: flex;
            justify-content: center;
            padding: 20px;
        }}

        .pie-chart {{
            width: 200px;
            height: 200px;
        }}

        .bar-chart {{
            width: 100%;
            max-width: 400px;
        }}

        /* Comparison Styles */
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}

        .diff-added {{
            background: rgba(68, 204, 68, 0.2);
            border-left: 3px solid var(--success);
        }}

        .diff-removed {{
            background: rgba(255, 68, 68, 0.2);
            border-left: 3px solid var(--danger);
        }}

        .diff-changed {{
            background: rgba(255, 204, 0, 0.2);
            border-left: 3px solid var(--warning);
        }}

        .trend-up {{
            color: var(--danger);
        }}

        .trend-down {{
            color: var(--success);
        }}

        .trend-stable {{
            color: var(--text-muted);
        }}

        /* Footer */
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            text-align: center;
            font-size: 0.9em;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.5em;
            }}

            .stat-value {{
                font-size: 2em;
            }}

            .grid-2, .grid-3 {{
                grid-template-columns: 1fr;
            }}

            .toolbar {{
                flex-direction: column;
                align-items: stretch;
            }}

            .search-box {{
                width: 100%;
            }}
        }}

        /* Print styles */
        @media print {{
            .toolbar, .toc {{
                display: none;
            }}

            body {{
                background: white;
                color: black;
            }}

            .card {{
                box-shadow: none;
                border: 1px solid #ddd;
                break-inside: avoid;
            }}

            .card.collapsed .card-body {{
                display: block;
            }}

            .header {{
                background: #1a1a2e !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            .finding {{
                break-inside: avoid;
            }}
        }}

        /* No results message */
        .no-results {{
            text-align: center;
            padding: 40px;
            color: var(--text-muted);
        }}

        .no-results.hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        {toolbar}
        {toc}
        {content}
    </div>
    <script>
        // Theme toggle
        function toggleTheme() {{
            const html = document.documentElement;
            const current = html.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            document.getElementById('theme-btn').textContent = next === 'dark' ? 'Light Mode' : 'Dark Mode';
        }}

        // Load saved theme
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        document.addEventListener('DOMContentLoaded', function() {{
            const btn = document.getElementById('theme-btn');
            if (btn) btn.textContent = savedTheme === 'dark' ? 'Light Mode' : 'Dark Mode';
        }});

        // Collapsible cards
        document.addEventListener('DOMContentLoaded', function() {{
            document.querySelectorAll('.card-header').forEach(function(header) {{
                header.addEventListener('click', function() {{
                    this.parentElement.classList.toggle('collapsed');
                }});
            }});
        }});

        // Search/filter findings
        function filterFindings() {{
            const query = document.getElementById('search-input').value.toLowerCase();
            const findings = document.querySelectorAll('.finding');
            let visibleCount = 0;

            findings.forEach(function(finding) {{
                const text = finding.textContent.toLowerCase();
                const matches = query === '' || text.includes(query);
                finding.classList.toggle('hidden', !matches);
                if (matches) visibleCount++;
            }});

            const noResults = document.getElementById('no-results');
            if (noResults) {{
                noResults.classList.toggle('hidden', visibleCount > 0 || query === '');
            }}
        }}

        // Print/PDF export
        function exportPDF() {{
            window.print();
        }}

        // Scroll to section
        function scrollToSection(id) {{
            const element = document.getElementById(id);
            if (element) {{
                element.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
        }}
    </script>
</body>
</html>
"""


class HTMLReportGenerator:
    """
    Generates comprehensive HTML security reports.

    Features:
    - Responsive design for all devices
    - Dark/light mode toggle
    - Collapsible sections
    - Search/filter for findings
    - Risk visualizations with SVG charts
    - PDF export via print dialog
    - Comparison reports with diff highlighting
    """

    def __init__(self):
        """Initialize HTMLReportGenerator."""
        pass

    def generate(
        self,
        analysis_result: Dict[str, Any],
        include_details: bool = True,
        include_toolbar: bool = True,
        include_toc: bool = True,
    ) -> str:
        """
        Generate a complete HTML report.

        Args:
            analysis_result: Analysis result dictionary.
            include_details: Include detailed findings.
            include_toolbar: Include toolbar with search/theme toggle.
            include_toc: Include table of contents.

        Returns:
            Complete HTML document string.
        """
        target_name = analysis_result.get("target_name", "Unknown Target")
        summary = analysis_result.get("summary", {})
        risk = summary.get("risk", {})
        results = analysis_result.get("results", [])

        # Build sections for TOC
        sections_info = []

        # Build content sections
        sections = []

        # Header
        sections.append(self._generate_header(analysis_result))
        sections_info.append(("header", "Report Header", None))

        # Risk overview
        sections.append(self._generate_risk_overview(risk, results))
        sections_info.append(("risk-assessment", "Risk Assessment", risk.get("level", "").upper()))

        # Summary stats
        sections.append(self._generate_summary_stats(results))
        sections_info.append(("test-results", "Test Results", None))

        # Findings
        if include_details and results:
            sections.append(self._generate_findings_section(results))
            failed_count = sum(1 for r in results if r.get("status") in ("failed", "warning"))
            sections_info.append(("findings", "Detailed Findings", str(failed_count)))

        # Recommendations
        sections.append(self._generate_recommendations(risk))
        sections_info.append(("recommendations", "Recommendations", None))

        # Behavior profile if available
        if analysis_result.get("behavior_profile"):
            sections.append(self._generate_behavior_section(analysis_result["behavior_profile"]))
            sections_info.append(("behavior", "Behavior Analysis", None))

        # Footer
        sections.append(self._generate_footer())

        # Generate toolbar
        toolbar = self._generate_toolbar() if include_toolbar else ""

        # Generate TOC
        toc = self._generate_toc(sections_info) if include_toc else ""

        # Combine into full HTML
        content = "\n".join(sections)

        return HTML_TEMPLATE.format(
            title=f"Security Report: {target_name}",
            toolbar=toolbar,
            toc=toc,
            content=content,
        )

    def _generate_toolbar(self) -> str:
        """Generate toolbar with search and controls."""
        return """
        <div class="toolbar">
            <div class="toolbar-left">
                <input type="text" id="search-input" class="search-box"
                       placeholder="Search findings..."
                       onkeyup="filterFindings()">
            </div>
            <div class="toolbar-right">
                <button class="btn btn-outline" id="theme-btn" onclick="toggleTheme()">Dark Mode</button>
                <button class="btn btn-primary" onclick="exportPDF()">Export PDF</button>
            </div>
        </div>
        """

    def _generate_toc(self, sections: List[tuple]) -> str:
        """Generate table of contents."""
        toc_items = []
        for section_id, title, badge in sections:
            if section_id == "header":
                continue
            badge_html = f'<span class="toc-badge">{badge}</span>' if badge else ""
            toc_items.append(
                f'<li><a href="#{section_id}" onclick="scrollToSection(\'{section_id}\'); return false;">'
                f'{title} {badge_html}</a></li>'
            )

        return f"""
        <nav class="toc">
            <h3>Table of Contents</h3>
            <ul>
                {''.join(toc_items)}
            </ul>
        </nav>
        """

    def _generate_header(self, result: Dict[str, Any]) -> str:
        """Generate report header section."""
        target_name = html.escape(str(result.get("target_name", "Unknown")))
        target_type = html.escape(str(result.get("target_type", "unknown")))
        profile = html.escape(str(result.get("profile", "standard")))
        timestamp = result.get("timestamp", datetime.now().isoformat())
        risk_level = result.get("summary", {}).get("risk", {}).get("level", "unknown").upper()

        risk_class = f"risk-{risk_level.lower()}"

        return f"""
        <div class="header" id="header">
            <h1>Security Analysis Report</h1>
            <p class="meta">
                <strong>Target:</strong> {target_name} |
                <strong>Type:</strong> {target_type} |
                <strong>Profile:</strong> {profile}
            </p>
            <p class="meta">
                <strong>Generated:</strong> {timestamp}
            </p>
            <p style="margin-top: 15px;">
                <span class="risk-badge {risk_class}">Risk: {risk_level}</span>
            </p>
        </div>
        """

    def _generate_risk_overview(self, risk: Dict[str, Any], results: List) -> str:
        """Generate risk overview section with SVG chart."""
        score = risk.get("score", 0)
        level = risk.get("level", "unknown").upper()
        factors = risk.get("factors", [])

        factors_html = ""
        if factors:
            factors_html = "<ul>" + "".join(f"<li>{html.escape(str(f))}</li>" for f in factors[:5]) + "</ul>"

        # Generate SVG pie chart for test results
        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        warnings = sum(1 for r in results if r.get("status") == "warning")
        total = passed + failed + warnings or 1

        pie_chart = self._generate_pie_chart_svg(
            [("Passed", passed, "#44cc44"), ("Failed", failed, "#ff4444"), ("Warnings", warnings, "#ffcc00")],
            total,
        )

        return f"""
        <div class="card" id="risk-assessment">
            <div class="card-header">Risk Assessment</div>
            <div class="card-body">
                <div class="grid grid-2">
                    <div>
                        <h3>Risk Score</h3>
                        <div class="risk-meter">
                            <div class="risk-marker" style="left: {score}%;" data-score="{score}/100"></div>
                        </div>
                        <p style="text-align: center; margin-top: 30px;">
                            <span class="risk-badge risk-{level.lower()}">{level}</span>
                        </p>
                    </div>
                    <div>
                        <h3>Primary Risk Factors</h3>
                        {factors_html if factors_html else "<p>No significant risk factors identified.</p>"}
                    </div>
                </div>
                <div style="margin-top: 30px;">
                    <h3>Test Results Distribution</h3>
                    <div class="chart-container">
                        {pie_chart}
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_pie_chart_svg(self, data: List[tuple], total: int) -> str:
        """Generate SVG pie chart."""
        if total == 0:
            return '<p style="text-align: center;">No data available</p>'

        svg_parts = ['<svg class="pie-chart" viewBox="-1 -1 2 2" style="transform: rotate(-90deg);">']

        cumulative = 0
        for label, value, color in data:
            if value == 0:
                continue

            start_angle = cumulative / total * 360
            slice_angle = value / total * 360
            cumulative += value

            # Calculate arc
            import math
            start_rad = start_angle * math.pi / 180
            end_rad = (start_angle + slice_angle) * math.pi / 180

            # Simplified: use path for arc
            large_arc = 1 if slice_angle > 180 else 0

            x1 = round(0.9 * math.cos(start_rad), 4)
            y1 = round(0.9 * math.sin(start_rad), 4)
            x2 = round(0.9 * math.cos(end_rad), 4)
            y2 = round(0.9 * math.sin(end_rad), 4)

            svg_parts.append(
                f'<path d="M 0 0 L {x1} {y1} A 0.9 0.9 0 {large_arc} 1 {x2} {y2} Z" fill="{color}" '
                f'stroke="white" stroke-width="0.02"><title>{label}: {value}</title></path>'
            )

        svg_parts.append('</svg>')

        # Add legend
        legend = '<div style="display: flex; justify-content: center; gap: 20px; margin-top: 10px;">'
        for label, value, color in data:
            legend += f'<span style="display: flex; align-items: center; gap: 5px;">'
            legend += f'<span style="width: 12px; height: 12px; background: {color}; border-radius: 2px;"></span>'
            legend += f'{label}: {value}</span>'
        legend += '</div>'

        return ''.join(svg_parts) + legend

    def _generate_summary_stats(self, results: List) -> str:
        """Generate summary statistics section."""
        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        warnings = sum(1 for r in results if r.get("status") == "warning")
        critical = sum(1 for r in results if r.get("severity", "").lower() == "critical")
        high = sum(1 for r in results if r.get("severity", "").lower() == "high")

        return f"""
        <div class="card" id="test-results">
            <div class="card-header">Test Results Summary</div>
            <div class="card-body">
                <div class="grid grid-3">
                    <div class="stat">
                        <div class="stat-value" style="color: var(--success);">{passed}</div>
                        <div class="stat-label">Passed</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" style="color: var(--danger);">{failed}</div>
                        <div class="stat-label">Failed</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" style="color: var(--warning);">{warnings}</div>
                        <div class="stat-label">Warnings</div>
                    </div>
                </div>
                <div class="grid grid-2" style="margin-top: 20px;">
                    <div class="stat">
                        <div class="stat-value" style="color: var(--danger);">{critical}</div>
                        <div class="stat-label">Critical Findings</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" style="color: var(--danger-light);">{high}</div>
                        <div class="stat-label">High Severity</div>
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_findings_section(self, results: List) -> str:
        """Generate detailed findings section with search support."""
        findings_html = []

        # Group by severity
        severity_order = ["critical", "high", "medium", "low", "info"]

        for severity in severity_order:
            severity_findings = [
                r for r in results
                if r.get("severity", "").lower() == severity and r.get("status") in ("failed", "warning")
            ]

            if not severity_findings:
                continue

            findings_html.append(f"<h3 style='margin: 20px 0 10px;'>{severity.upper()} Severity ({len(severity_findings)})</h3>")

            for finding in severity_findings:
                test_name = html.escape(str(finding.get("test_name", "Unknown Test")))
                category = html.escape(str(finding.get("category", "N/A")))
                details = finding.get("details", {})
                message = details.get("message", "") if isinstance(details, dict) else str(details)
                message = html.escape(str(message)[:500])

                findings_html.append(f"""
                <div class="finding finding-{severity}" data-severity="{severity}" data-category="{category}">
                    <div class="finding-header">
                        <span class="finding-title">{test_name}</span>
                        <span class="finding-severity severity-{severity}">{severity}</span>
                    </div>
                    <div class="finding-meta">Category: {category}</div>
                    {f'<div class="finding-details">{message}</div>' if message else ''}
                </div>
                """)

        if not findings_html:
            findings_html.append("<p>No significant findings to report.</p>")

        return f"""
        <div class="card" id="findings">
            <div class="card-header">Detailed Findings</div>
            <div class="card-body">
                {''.join(findings_html)}
                <div id="no-results" class="no-results hidden">
                    No findings match your search.
                </div>
            </div>
        </div>
        """

    def _generate_recommendations(self, risk: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        level = risk.get("level", "unknown").lower()

        if level == "critical":
            recs = [
                "Do not deploy this target in its current state",
                "Address all critical findings immediately",
                "Conduct thorough security review before any deployment",
                "Consider engaging security specialists",
            ]
            header = "Critical Actions Required"
            style = "color: var(--danger);"
        elif level == "high":
            recs = [
                "Address all critical and high severity issues before deployment",
                "Implement additional monitoring if deployment is necessary",
                "Plan for immediate remediation timeline",
            ]
            header = "High Priority Actions"
            style = "color: var(--danger-light);"
        elif level == "medium":
            recs = [
                "Review and address findings before production deployment",
                "Implement mitigations where possible",
                "Schedule follow-up analysis after changes",
            ]
            header = "Recommended Actions"
            style = "color: var(--warning);"
        else:
            recs = [
                "Continue standard security monitoring",
                "Re-analyze periodically to detect drift",
                "Review any warnings for potential improvements",
            ]
            header = "Maintenance Actions"
            style = "color: var(--success);"

        recs_html = "<ol>" + "".join(f"<li>{r}</li>" for r in recs) + "</ol>"

        return f"""
        <div class="card" id="recommendations">
            <div class="card-header">Recommendations</div>
            <div class="card-body">
                <h3 style="{style}">{header}</h3>
                {recs_html}
            </div>
        </div>
        """

    def _generate_behavior_section(self, profile: Dict[str, Any]) -> str:
        """Generate behavior analysis section."""
        safety_score = profile.get("overall_safety_score", 0)
        total = profile.get("total_outputs", 0)
        safe = profile.get("safe_outputs", 0)
        jailbreak_rate = profile.get("jailbreak_success_rate", 0) * 100

        return f"""
        <div class="card" id="behavior">
            <div class="card-header">Behavior Analysis</div>
            <div class="card-body">
                <div class="grid grid-3">
                    <div class="stat">
                        <div class="stat-value">{safety_score:.1f}</div>
                        <div class="stat-label">Safety Score</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{safe}/{total}</div>
                        <div class="stat-label">Safe Outputs</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{jailbreak_rate:.1f}%</div>
                        <div class="stat-label">Jailbreak Rate</div>
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""
        <div class="footer">
            <p>Generated by BenderBox v3.0.0-alpha on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>AI Security Analysis Platform</p>
        </div>
        """

    def generate_comparison(
        self,
        comparison_result: Dict[str, Any],
        include_charts: bool = True,
    ) -> str:
        """
        Generate an HTML comparison report.

        Args:
            comparison_result: Comparison result dictionary.
            include_charts: Include SVG comparison charts.

        Returns:
            Complete HTML document string.
        """
        targets = comparison_result.get("targets", [])
        results = comparison_result.get("results", [])
        summary = comparison_result.get("summary", {})

        sections = []

        # Header
        sections.append(f"""
        <div class="header">
            <h1>Model Comparison Report</h1>
            <p class="meta">Comparing {len(targets)} targets</p>
            <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """)

        # Comparison table
        sections.append(self._generate_comparison_table(targets, results))

        # Comparison chart
        if include_charts and results:
            sections.append(self._generate_comparison_chart(targets, results))

        # Summary
        if summary:
            sections.append(self._generate_comparison_summary(summary))

        # Recommendation
        if comparison_result.get("recommendation"):
            sections.append(f"""
            <div class="card">
                <div class="card-header">Recommendation</div>
                <div class="card-body">
                    <p>{html.escape(comparison_result['recommendation'])}</p>
                </div>
            </div>
            """)

        # Footer
        sections.append(self._generate_footer())

        content = "\n".join(sections)
        toolbar = self._generate_toolbar()

        return HTML_TEMPLATE.format(
            title="Model Comparison Report",
            toolbar=toolbar,
            toc="",
            content=content,
        )

    def _generate_comparison_table(self, targets: List[str], results: List[Dict]) -> str:
        """Generate comparison table with diff highlighting."""
        headers = ["Metric"] + [html.escape(t[:25]) for t in targets]
        header_row = "".join(f"<th>{h}</th>" for h in headers)

        metrics = [
            ("Risk Score", "risk_score"),
            ("Risk Level", "risk_level"),
            ("Finding Count", "finding_count"),
            ("Critical", "critical_count"),
            ("High", "high_count"),
        ]

        rows = []
        for label, key in metrics:
            cells = [f"<td><strong>{label}</strong></td>"]

            values = [r.get(key, "N/A") for r in results]

            # Determine best/worst for highlighting
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            min_val = min(numeric_values) if numeric_values else None
            max_val = max(numeric_values) if numeric_values else None

            for value in values:
                cell_class = ""
                if isinstance(value, (int, float)) and min_val is not None:
                    if key in ("risk_score", "finding_count", "critical_count", "high_count"):
                        # Lower is better
                        if value == min_val and min_val != max_val:
                            cell_class = "diff-added"
                        elif value == max_val and min_val != max_val:
                            cell_class = "diff-removed"

                if key == "risk_level" and isinstance(value, str):
                    cell_class = f"risk-{value.lower()}"
                    value = value.upper()

                if isinstance(value, float):
                    value = f"{value:.1f}"

                cells.append(f'<td class="{cell_class}">{value}</td>')

            rows.append("<tr>" + "".join(cells) + "</tr>")

        return f"""
        <div class="card">
            <div class="card-header">Side-by-Side Comparison</div>
            <div class="card-body">
                <table class="table">
                    <thead><tr>{header_row}</tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
        </div>
        """

    def _generate_comparison_chart(self, targets: List[str], results: List[Dict]) -> str:
        """Generate SVG bar chart for comparison."""
        import math

        scores = [r.get("risk_score", 0) for r in results]
        max_score = max(scores) if scores else 100

        bar_width = 60
        bar_gap = 20
        chart_height = 200
        chart_width = len(targets) * (bar_width + bar_gap) + 50

        bars = []
        for i, (target, score) in enumerate(zip(targets, scores)):
            bar_height = (score / max(max_score, 1)) * 150
            x = 30 + i * (bar_width + bar_gap)
            y = chart_height - 30 - bar_height

            # Color based on score
            if score >= 70:
                color = "#ff4444"
            elif score >= 40:
                color = "#ffcc00"
            else:
                color = "#44cc44"

            bars.append(f'''
                <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}"
                      fill="{color}" rx="4">
                    <title>{html.escape(target)}: {score}</title>
                </rect>
                <text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle"
                      font-size="12" fill="currentColor">{score:.0f}</text>
                <text x="{x + bar_width/2}" y="{chart_height - 10}" text-anchor="middle"
                      font-size="10" fill="currentColor">{html.escape(target[:10])}</text>
            ''')

        return f"""
        <div class="card">
            <div class="card-header">Risk Score Comparison</div>
            <div class="card-body">
                <div class="chart-container">
                    <svg class="bar-chart" viewBox="0 0 {chart_width} {chart_height}">
                        {''.join(bars)}
                        <line x1="25" y1="{chart_height - 30}" x2="{chart_width - 10}" y2="{chart_height - 30}"
                              stroke="currentColor" stroke-width="1"/>
                    </svg>
                </div>
            </div>
        </div>
        """

    def _generate_comparison_summary(self, summary: Dict[str, Any]) -> str:
        """Generate comparison summary section."""
        safest = summary.get("safest", "N/A")
        riskiest = summary.get("riskiest", "N/A")

        return f"""
        <div class="card">
            <div class="card-header">Summary</div>
            <div class="card-body">
                <div class="grid grid-2">
                    <div style="padding: 20px; background: rgba(68, 204, 68, 0.1); border-radius: 8px;">
                        <h4 style="color: var(--success);">Safest Model</h4>
                        <p style="font-size: 1.2em; font-weight: bold;">{html.escape(str(safest))}</p>
                    </div>
                    <div style="padding: 20px; background: rgba(255, 68, 68, 0.1); border-radius: 8px;">
                        <h4 style="color: var(--danger);">Highest Risk</h4>
                        <p style="font-size: 1.2em; font-weight: bold;">{html.escape(str(riskiest))}</p>
                    </div>
                </div>
            </div>
        </div>
        """

    async def save(
        self,
        analysis_result: Dict[str, Any],
        output_path: str,
        include_details: bool = True,
        open_browser: bool = False,
    ) -> str:
        """
        Save HTML report to file.

        Args:
            analysis_result: Analysis result dictionary.
            output_path: Output file path.
            include_details: Include detailed findings.
            open_browser: Open in browser after saving.

        Returns:
            Path to saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        html_content = self.generate(analysis_result, include_details)
        path.write_text(html_content, encoding="utf-8")

        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{path.absolute()}")

        return str(path)
