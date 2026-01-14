"""
BenderBox Report Viewer Generator

Generates a self-contained HTML report viewer with:
- Futurama-inspired 90s retro-futuristic dark theme
- Tabbed interface for multiple reports
- Overview dashboard with high-level summaries
- Client-side JSON loading (no server required)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from benderbox.config import get_config

logger = logging.getLogger(__name__)

# Futurama-inspired retro-futuristic theme
REPORT_VIEWER_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BenderBox Report Viewer</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

        :root {
            --bg-dark: #0a0a12;
            --bg-panel: #12121f;
            --bg-card: #1a1a2e;
            --bg-hover: #252540;
            --neon-green: #39ff14;
            --neon-purple: #bf00ff;
            --neon-orange: #ff6b35;
            --neon-cyan: #00ffff;
            --neon-pink: #ff00ff;
            --neon-yellow: #ffff00;
            --text-primary: #e0e0e0;
            --text-muted: #888899;
            --danger: #ff3366;
            --warning: #ffaa00;
            --success: #00ff88;
            --border: #333355;
            --glow-green: 0 0 10px #39ff14, 0 0 20px #39ff14, 0 0 30px #39ff14;
            --glow-purple: 0 0 10px #bf00ff, 0 0 20px #bf00ff;
            --glow-orange: 0 0 10px #ff6b35, 0 0 20px #ff6b35;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Share Tech Mono', monospace;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            background-image:
                radial-gradient(ellipse at top, #1a1a3e 0%, transparent 50%),
                radial-gradient(ellipse at bottom, #0f0f1a 0%, transparent 50%);
        }

        /* Scanline effect */
        body::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 0, 0, 0.1) 0px,
                rgba(0, 0, 0, 0.1) 1px,
                transparent 1px,
                transparent 2px
            );
            z-index: 1000;
        }

        /* Header */
        .header {
            background: linear-gradient(180deg, var(--bg-panel) 0%, var(--bg-dark) 100%);
            border-bottom: 2px solid var(--neon-green);
            padding: 20px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 20px rgba(57, 255, 20, 0.2);
        }

        .logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 2em;
            font-weight: 900;
            color: var(--neon-green);
            text-shadow: var(--glow-green);
            letter-spacing: 3px;
        }

        .logo span {
            color: var(--neon-orange);
            text-shadow: var(--glow-orange);
        }

        .header-stats {
            display: flex;
            gap: 30px;
        }

        .stat-box {
            text-align: center;
            padding: 10px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        .stat-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5em;
            font-weight: 700;
            color: var(--neon-cyan);
            text-shadow: 0 0 10px var(--neon-cyan);
        }

        .stat-label {
            font-size: 0.75em;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        /* Main Container */
        .container {
            display: flex;
            height: calc(100vh - 90px);
        }

        /* Sidebar */
        .sidebar {
            width: 300px;
            background: var(--bg-panel);
            border-right: 1px solid var(--border);
            overflow-y: auto;
            padding: 20px 0;
        }

        .sidebar-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9em;
            color: var(--neon-purple);
            text-transform: uppercase;
            letter-spacing: 3px;
            padding: 10px 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 10px;
        }

        .report-list {
            list-style: none;
        }

        .report-item {
            padding: 15px 20px;
            cursor: pointer;
            border-left: 3px solid transparent;
            transition: all 0.3s ease;
            position: relative;
        }

        .report-item:hover {
            background: var(--bg-hover);
            border-left-color: var(--neon-purple);
        }

        .report-item.active {
            background: var(--bg-card);
            border-left-color: var(--neon-green);
        }

        .report-item.active::after {
            content: "▶";
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--neon-green);
        }

        .report-name {
            font-weight: bold;
            margin-bottom: 5px;
            color: var(--text-primary);
        }

        .report-meta {
            font-size: 0.8em;
            color: var(--text-muted);
        }

        .report-risk {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.7em;
            font-weight: bold;
            text-transform: uppercase;
            margin-top: 5px;
        }

        .risk-critical { background: var(--danger); color: white; box-shadow: 0 0 10px var(--danger); }
        .risk-high { background: var(--warning); color: black; }
        .risk-medium { background: #886600; color: white; }
        .risk-low { background: var(--success); color: black; }
        .risk-safe { background: var(--neon-cyan); color: black; }

        /* Main Content */
        .main-content {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
        }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 30px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 5px;
        }

        .tab {
            padding: 12px 25px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85em;
            color: var(--text-muted);
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .tab:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .tab.active {
            background: var(--neon-green);
            color: var(--bg-dark);
            font-weight: bold;
            box-shadow: var(--glow-green);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Overview Dashboard */
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .dashboard-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 25px;
            position: relative;
            overflow: hidden;
        }

        .dashboard-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--neon-green), var(--neon-purple));
        }

        .dashboard-card h3 {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9em;
            color: var(--neon-purple);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 15px;
        }

        .big-number {
            font-family: 'Orbitron', sans-serif;
            font-size: 3em;
            font-weight: 900;
            line-height: 1;
            margin-bottom: 10px;
        }

        .big-number.green { color: var(--neon-green); text-shadow: var(--glow-green); }
        .big-number.orange { color: var(--neon-orange); text-shadow: var(--glow-orange); }
        .big-number.purple { color: var(--neon-purple); text-shadow: var(--glow-purple); }
        .big-number.cyan { color: var(--neon-cyan); }

        /* Risk Meter */
        .risk-meter-container {
            margin: 20px 0;
        }

        .risk-meter {
            height: 20px;
            background: linear-gradient(90deg,
                var(--success) 0%,
                var(--neon-yellow) 40%,
                var(--warning) 60%,
                var(--danger) 100%
            );
            border-radius: 10px;
            position: relative;
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        }

        .risk-marker {
            position: absolute;
            top: -8px;
            width: 4px;
            height: 36px;
            background: white;
            border-radius: 2px;
            transform: translateX(-50%);
            box-shadow: 0 0 10px white;
        }

        .risk-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            font-size: 0.75em;
            color: var(--text-muted);
        }

        /* Charts placeholder */
        .chart-container {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }

        .chart-title {
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85em;
            color: var(--neon-cyan);
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Findings List */
        .findings-section {
            margin-top: 30px;
        }

        .finding {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid var(--border);
            transition: all 0.3s ease;
        }

        .finding:hover {
            background: var(--bg-hover);
            transform: translateX(5px);
        }

        .finding.critical { border-left-color: var(--danger); }
        .finding.high { border-left-color: var(--warning); }
        .finding.medium { border-left-color: #886600; }
        .finding.low { border-left-color: var(--success); }

        .finding-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .finding-title {
            font-weight: bold;
            color: var(--text-primary);
        }

        .finding-severity {
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 0.75em;
            font-weight: bold;
            text-transform: uppercase;
        }

        .finding-details {
            background: var(--bg-dark);
            padding: 15px;
            border-radius: 5px;
            font-size: 0.9em;
            color: var(--text-muted);
            margin-top: 10px;
            border: 1px solid var(--border);
            white-space: pre-wrap;
            font-family: 'Share Tech Mono', monospace;
        }

        /* Table */
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        .data-table th,
        .data-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        .data-table th {
            background: var(--bg-panel);
            font-family: 'Orbitron', sans-serif;
            font-size: 0.8em;
            color: var(--neon-purple);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .data-table tr:hover {
            background: var(--bg-hover);
        }

        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-muted);
        }

        .empty-state .icon {
            font-size: 4em;
            margin-bottom: 20px;
            opacity: 0.5;
        }

        .empty-state h3 {
            font-family: 'Orbitron', sans-serif;
            color: var(--neon-purple);
            margin-bottom: 10px;
        }

        /* Buttons */
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: var(--neon-green);
            color: var(--bg-dark);
        }

        .btn-primary:hover {
            box-shadow: var(--glow-green);
            transform: translateY(-2px);
        }

        .btn-secondary {
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--neon-purple);
        }

        .btn-secondary:hover {
            background: var(--neon-purple);
            color: white;
            box-shadow: var(--glow-purple);
        }

        /* Search */
        .search-container {
            margin-bottom: 20px;
        }

        .search-input {
            width: 100%;
            padding: 12px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 25px;
            color: var(--text-primary);
            font-family: 'Share Tech Mono', monospace;
            font-size: 1em;
        }

        .search-input:focus {
            outline: none;
            border-color: var(--neon-green);
            box-shadow: 0 0 10px rgba(57, 255, 20, 0.3);
        }

        .search-input::placeholder {
            color: var(--text-muted);
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--neon-purple);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }

            .sidebar {
                width: 100%;
                height: auto;
                max-height: 200px;
            }

            .header {
                flex-direction: column;
                gap: 15px;
            }

            .header-stats {
                flex-wrap: wrap;
                justify-content: center;
            }
        }

        /* Loading animation */
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
        }

        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 3px solid var(--border);
            border-top-color: var(--neon-green);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Comparison mode */
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }

        .comparison-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }

        .comparison-card h4 {
            font-family: 'Orbitron', sans-serif;
            color: var(--neon-cyan);
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border);
        }

        .trend-up { color: var(--danger); }
        .trend-down { color: var(--success); }
        .trend-same { color: var(--text-muted); }

        /* Bender ASCII art */
        .bender-art {
            font-family: monospace;
            font-size: 10px;
            line-height: 1;
            color: var(--neon-green);
            opacity: 0.3;
            white-space: pre;
            position: fixed;
            bottom: 10px;
            right: 10px;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">BENDER<span>BOX</span></div>
        <div class="header-stats">
            <div class="stat-box">
                <div class="stat-value" id="total-reports">0</div>
                <div class="stat-label">Reports</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="critical-count">0</div>
                <div class="stat-label">Critical</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="avg-score">0</div>
                <div class="stat-label">Avg Risk</div>
            </div>
        </div>
    </div>

    <div class="container">
        <div class="sidebar">
            <div class="sidebar-title">📊 Reports</div>
            <div class="search-container" style="padding: 0 15px;">
                <input type="text" class="search-input" placeholder="Search reports..."
                       onkeyup="filterReports(this.value)">
            </div>
            <ul class="report-list" id="report-list">
                <!-- Reports populated by JS -->
            </ul>
        </div>

        <div class="main-content">
            <div class="tabs">
                <div class="tab active" onclick="switchTab('overview')">Overview</div>
                <div class="tab" onclick="switchTab('findings')">Findings</div>
                <div class="tab" onclick="switchTab('details')">Details</div>
                <div class="tab" onclick="switchTab('compare')">Compare</div>
            </div>

            <!-- Overview Tab -->
            <div class="tab-content active" id="tab-overview">
                <div class="dashboard-grid">
                    <div class="dashboard-card">
                        <h3>Risk Score</h3>
                        <div class="big-number orange" id="current-risk-score">--</div>
                        <div class="risk-meter-container">
                            <div class="risk-meter">
                                <div class="risk-marker" id="risk-marker" style="left: 0%;"></div>
                            </div>
                            <div class="risk-labels">
                                <span>Safe</span>
                                <span>Low</span>
                                <span>Medium</span>
                                <span>High</span>
                                <span>Critical</span>
                            </div>
                        </div>
                    </div>

                    <div class="dashboard-card">
                        <h3>Tests Run</h3>
                        <div class="big-number green" id="tests-run">--</div>
                        <p style="color: var(--text-muted);">Security tests executed</p>
                    </div>

                    <div class="dashboard-card">
                        <h3>Findings</h3>
                        <div class="big-number purple" id="findings-count">--</div>
                        <p style="color: var(--text-muted);">Issues detected</p>
                    </div>

                    <div class="dashboard-card">
                        <h3>Analysis Type</h3>
                        <div class="big-number cyan" id="analysis-type" style="font-size: 1.5em;">--</div>
                        <p style="color: var(--text-muted);" id="analysis-target">--</p>
                    </div>
                </div>

                <div class="chart-container">
                    <div class="chart-title">Severity Distribution</div>
                    <div id="severity-chart"></div>
                </div>

                <div class="findings-section">
                    <h3 style="font-family: 'Orbitron', sans-serif; color: var(--neon-orange); margin-bottom: 20px;">
                        ⚠️ Top Issues
                    </h3>
                    <div id="top-findings">
                        <!-- Populated by JS -->
                    </div>
                </div>
            </div>

            <!-- Findings Tab -->
            <div class="tab-content" id="tab-findings">
                <div class="search-container">
                    <input type="text" class="search-input" placeholder="Search findings..."
                           onkeyup="filterFindings(this.value)">
                </div>
                <div id="all-findings">
                    <!-- Populated by JS -->
                </div>
            </div>

            <!-- Details Tab -->
            <div class="tab-content" id="tab-details">
                <div id="report-details">
                    <!-- Populated by JS -->
                </div>
            </div>

            <!-- Compare Tab -->
            <div class="tab-content" id="tab-compare">
                <p style="color: var(--text-muted); margin-bottom: 20px;">
                    Select reports from the sidebar to compare (Ctrl+Click for multiple)
                </p>
                <div class="comparison-grid" id="comparison-view">
                    <!-- Populated by JS -->
                </div>
            </div>
        </div>
    </div>

    <pre class="bender-art">
    ____
   /    \\
  | O  O |
  |  __  |
   \\____/
    |  |
   /|  |\\
  / |  | \\
    </pre>

    <script>
        // Report data (embedded or loaded from files)
        let reports = [];
        let currentReport = null;
        let selectedReports = [];

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadReports();
        });

        // Load reports from embedded data or file list
        function loadReports() {
            // Check for embedded data first
            const embeddedData = document.getElementById('report-data');
            if (embeddedData) {
                try {
                    reports = JSON.parse(embeddedData.textContent);
                    renderReportList();
                    updateStats();
                    if (reports.length > 0) {
                        selectReport(0);
                    }
                    return;
                } catch (e) {
                    console.error('Failed to parse embedded data:', e);
                }
            }

            // Show empty state
            showEmptyState();
        }

        function showEmptyState() {
            document.getElementById('report-list').innerHTML = `
                <div class="empty-state">
                    <div class="icon">📭</div>
                    <h3>No Reports Yet</h3>
                    <p>Run an analysis to generate reports</p>
                </div>
            `;
        }

        function renderReportList() {
            const list = document.getElementById('report-list');
            list.innerHTML = reports.map((report, index) => `
                <li class="report-item ${index === 0 ? 'active' : ''}"
                    onclick="selectReport(${index})"
                    data-name="${escapeHtml(report.target_name || 'Unknown').toLowerCase()}"
                    data-type="${escapeHtml(report._analysis_type || 'unknown')}">
                    <div class="report-name">${escapeHtml(report.target_name || 'Unknown')}</div>
                    <div class="report-meta">
                        <span class="report-type">${escapeHtml((report._analysis_type || 'unknown').toUpperCase())}</span>
                        ${formatDate(report._timestamp || report.timestamp)}
                    </div>
                    <span class="report-risk risk-${(report._risk_level || report.summary?.risk?.level || 'unknown').toLowerCase()}">
                        ${report._risk_level || report.summary?.risk?.level || 'N/A'}
                    </span>
                </li>
            `).join('');
        }

        function selectReport(index) {
            currentReport = reports[index];

            // Update active state
            document.querySelectorAll('.report-item').forEach((item, i) => {
                item.classList.toggle('active', i === index);
            });

            renderOverview();
            renderFindings();
            renderDetails();
        }

        function updateStats() {
            document.getElementById('total-reports').textContent = reports.length;

            const criticalCount = reports.filter(r =>
                (r._risk_level || r.summary?.risk?.level || '').toLowerCase() === 'critical'
            ).length;
            document.getElementById('critical-count').textContent = criticalCount;

            const avgScore = reports.length > 0
                ? Math.round(reports.reduce((sum, r) => sum + (r._risk_score || r.summary?.risk?.score || 0), 0) / reports.length)
                : 0;
            document.getElementById('avg-score').textContent = avgScore;
        }

        function renderOverview() {
            if (!currentReport) return;

            // Use normalized fields with fallbacks
            const riskScore = currentReport._risk_score || currentReport.summary?.risk?.score || 0;
            const riskLevel = currentReport._risk_level || currentReport.summary?.risk?.level || 'UNKNOWN';
            const analysisType = currentReport._analysis_type || currentReport.target_type || 'unknown';
            const findingsCount = currentReport._findings_count || 0;

            // Get results/tools/tests depending on analysis type
            let results = currentReport.results || [];
            let testsCount = results.length;

            if (analysisType === 'mcp_server' && currentReport.tools) {
                results = currentReport.tools.map(t => ({
                    name: t.name,
                    severity: t.risk_level,
                    status: t.risk_level === 'critical' || t.risk_level === 'high' ? 'failed' : 'passed',
                    description: t.description || t.risk_factors?.join(', ') || ''
                }));
                testsCount = currentReport.tools.length;
            } else if (analysisType === 'model' && currentReport.tests) {
                results = currentReport.tests.map(t => ({
                    name: t.name || t.test_id,
                    severity: t.severity || 'medium',
                    status: t.result === 'fail' ? 'failed' : 'passed',
                    description: t.details || ''
                }));
                testsCount = currentReport.tests.length;
            } else if (analysisType === 'context' && currentReport.findings) {
                results = currentReport.findings.map(f => ({
                    name: f.pattern || f.type,
                    severity: f.severity || 'medium',
                    status: 'failed',
                    description: f.description || f.match || ''
                }));
                testsCount = currentReport.findings.length;
            }

            // Risk score
            document.getElementById('current-risk-score').textContent = riskScore;
            document.getElementById('risk-marker').style.left = `${Math.min(riskScore, 100)}%`;

            // Tests run
            document.getElementById('tests-run').textContent = testsCount;

            // Findings count
            const failedCount = findingsCount || results.filter(r => r.status === 'failed' || r.status === 'warning').length;
            document.getElementById('findings-count').textContent = failedCount;

            // Analysis type
            document.getElementById('analysis-type').textContent = analysisType.toUpperCase().replace('_', ' ');
            document.getElementById('analysis-target').textContent = currentReport.target_name || 'Unknown';

            // Severity chart
            renderSeverityChart(results);

            // Top findings
            renderTopFindings(results);
        }

        function renderSeverityChart(results) {
            const counts = {
                critical: results.filter(r => r.severity === 'critical').length,
                high: results.filter(r => r.severity === 'high').length,
                medium: results.filter(r => r.severity === 'medium').length,
                low: results.filter(r => r.severity === 'low').length,
                info: results.filter(r => r.severity === 'info').length,
            };

            const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
            const colors = {
                critical: 'var(--danger)',
                high: 'var(--warning)',
                medium: '#886600',
                low: 'var(--success)',
                info: 'var(--text-muted)',
            };

            const chart = document.getElementById('severity-chart');
            chart.innerHTML = `
                <div style="display: flex; height: 30px; border-radius: 5px; overflow: hidden; margin-bottom: 15px;">
                    ${Object.entries(counts).map(([sev, count]) => `
                        <div style="width: ${(count / total) * 100}%; background: ${colors[sev]};"
                             title="${sev}: ${count}"></div>
                    `).join('')}
                </div>
                <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                    ${Object.entries(counts).map(([sev, count]) => `
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <div style="width: 12px; height: 12px; background: ${colors[sev]}; border-radius: 2px;"></div>
                            <span style="color: var(--text-muted);">${sev}: <strong>${count}</strong></span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        function renderTopFindings(results) {
            const failed = results
                .filter(r => r.status === 'failed' || r.status === 'warning')
                .sort((a, b) => {
                    const order = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };
                    return (order[a.severity] || 5) - (order[b.severity] || 5);
                })
                .slice(0, 5);

            const container = document.getElementById('top-findings');

            if (failed.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="icon">✅</div>
                        <h3>All Clear!</h3>
                        <p>No issues found in this analysis</p>
                    </div>
                `;
                return;
            }

            container.innerHTML = failed.map(f => `
                <div class="finding ${f.severity || 'info'}">
                    <div class="finding-header">
                        <span class="finding-title">${escapeHtml(f.test_name || 'Unknown Test')}</span>
                        <span class="finding-severity severity-${f.severity || 'info'}">${f.severity || 'info'}</span>
                    </div>
                    <div class="report-meta">${escapeHtml(f.category || 'N/A')}</div>
                    ${f.details?.message ? `<div class="finding-details">${escapeHtml(f.details.message)}</div>` : ''}
                </div>
            `).join('');
        }

        function renderFindings() {
            if (!currentReport) return;

            const results = currentReport.results || [];
            const container = document.getElementById('all-findings');

            const grouped = {};
            results.forEach(r => {
                const sev = r.severity || 'info';
                if (!grouped[sev]) grouped[sev] = [];
                grouped[sev].push(r);
            });

            const order = ['critical', 'high', 'medium', 'low', 'info'];

            container.innerHTML = order.map(sev => {
                const items = grouped[sev] || [];
                if (items.length === 0) return '';

                return `
                    <h3 style="font-family: 'Orbitron', sans-serif; color: var(--neon-orange);
                               margin: 20px 0 15px; text-transform: uppercase;">
                        ${sev} (${items.length})
                    </h3>
                    ${items.map(f => `
                        <div class="finding ${f.severity || 'info'}" data-searchable="${escapeHtml((f.test_name || '') + ' ' + (f.category || '') + ' ' + (f.details?.message || '')).toLowerCase()}">
                            <div class="finding-header">
                                <span class="finding-title">${escapeHtml(f.test_name || 'Unknown')}</span>
                                <span class="finding-severity severity-${f.severity || 'info'}">${f.status || 'N/A'}</span>
                            </div>
                            <div class="report-meta">Category: ${escapeHtml(f.category || 'N/A')}</div>
                            ${f.details?.message ? `<div class="finding-details">${escapeHtml(f.details.message)}</div>` : ''}
                        </div>
                    `).join('')}
                `;
            }).join('');
        }

        function renderDetails() {
            if (!currentReport) return;

            const container = document.getElementById('report-details');
            container.innerHTML = `
                <table class="data-table">
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Target Name</td><td>${escapeHtml(currentReport.target_name || 'Unknown')}</td></tr>
                    <tr><td>Target Type</td><td>${escapeHtml(currentReport.target_type || 'Unknown')}</td></tr>
                    <tr><td>Profile</td><td>${escapeHtml(currentReport.profile || 'standard')}</td></tr>
                    <tr><td>Timestamp</td><td>${escapeHtml(currentReport.timestamp || 'Unknown')}</td></tr>
                    <tr><td>Risk Level</td><td><span class="report-risk risk-${(currentReport.summary?.risk?.level || 'unknown').toLowerCase()}">${currentReport.summary?.risk?.level || 'N/A'}</span></td></tr>
                    <tr><td>Risk Score</td><td>${currentReport.summary?.risk?.score || 0}/100</td></tr>
                    <tr><td>Tests Run</td><td>${(currentReport.results || []).length}</td></tr>
                    <tr><td>Passed</td><td>${(currentReport.results || []).filter(r => r.status === 'passed').length}</td></tr>
                    <tr><td>Failed</td><td>${(currentReport.results || []).filter(r => r.status === 'failed').length}</td></tr>
                </table>

                <h3 style="font-family: 'Orbitron', sans-serif; color: var(--neon-cyan); margin: 30px 0 15px;">
                    Raw JSON
                </h3>
                <div class="finding-details" style="max-height: 400px; overflow: auto;">
${escapeHtml(JSON.stringify(currentReport, null, 2))}
                </div>
            `;
        }

        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.toggle('active', tab.textContent.toLowerCase().includes(tabName));
            });
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.toggle('active', content.id === 'tab-' + tabName);
            });
        }

        function filterReports(query) {
            query = query.toLowerCase();
            document.querySelectorAll('.report-item').forEach(item => {
                const name = item.dataset.name || '';
                item.style.display = name.includes(query) ? 'block' : 'none';
            });
        }

        function filterFindings(query) {
            query = query.toLowerCase();
            document.querySelectorAll('#all-findings .finding').forEach(finding => {
                const text = finding.dataset.searchable || '';
                finding.style.display = text.includes(query) ? 'block' : 'none';
            });
        }

        function formatDate(dateStr) {
            if (!dateStr) return 'Unknown';
            try {
                const date = new Date(dateStr);
                return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            } catch {
                return dateStr;
            }
        }

        function escapeHtml(str) {
            if (!str) return '';
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }
    </script>

    <!-- Report data will be embedded here -->
    <script type="application/json" id="report-data">
{report_data}
    </script>
</body>
</html>
'''


class ReportViewerGenerator:
    """
    Generates the BenderBox Report Viewer HTML page.

    Features:
    - Futurama-inspired 90s retro-futuristic dark theme
    - Tabbed interface (Overview, Findings, Details, Compare)
    - Dashboard with risk metrics and charts
    - Search and filter capabilities
    - Self-contained HTML with embedded JSON data
    """

    def __init__(self, reports_path: Optional[str] = None):
        """
        Initialize ReportViewerGenerator.

        Args:
            reports_path: Path to reports directory. Uses config default if not provided.
        """
        if reports_path:
            self.reports_path = Path(reports_path)
        else:
            config = get_config().storage
            self.reports_path = Path(config.reports_path)

    def _normalize_report(self, data: Dict[str, Any], source_file: Path) -> Dict[str, Any]:
        """
        Normalize different analysis report formats into a consistent structure.

        Handles:
        - Model analysis (schema 0.2.0): overall_risk, model, tests
        - MCP server analysis (schema 1.0.0): risk_assessment, server, tools
        - Context/skills analysis: risk_patterns, findings
        - Infrastructure analysis: overall_risk, tests

        Returns a normalized dict with consistent keys for the viewer.
        """
        normalized = data.copy()

        # Detect analysis type
        analysis_type = data.get("analysis_type", "unknown")
        if "model" in data and "path" in data.get("model", {}):
            analysis_type = "model"
        elif "server" in data or "tools" in data:
            analysis_type = "mcp_server"
        elif "risk_patterns" in data or "findings" in data:
            analysis_type = "context"
        elif "profile" in data and "infrastructure" in str(data.get("run_id", "")):
            analysis_type = "infrastructure"

        normalized["_analysis_type"] = analysis_type

        # Normalize target name
        if "target_name" not in normalized:
            if analysis_type == "model":
                normalized["target_name"] = data.get("model", {}).get("name", "Unknown Model")
            elif analysis_type == "mcp_server":
                normalized["target_name"] = data.get("server", {}).get("name", "Unknown MCP Server")
            elif analysis_type == "context":
                normalized["target_name"] = data.get("file_path", source_file.stem)
            else:
                normalized["target_name"] = data.get("run_id", source_file.stem)

        # Normalize risk level and score
        if "overall_risk" in data:
            # Model/infrastructure analysis format
            risk = data["overall_risk"]
            normalized["_risk_level"] = risk.get("level", "UNKNOWN").upper()
            normalized["_risk_score"] = risk.get("score", 0)
        elif "risk_assessment" in data:
            # MCP analysis format
            risk = data["risk_assessment"]
            normalized["_risk_level"] = risk.get("overall_level", "unknown").upper()
            normalized["_risk_score"] = risk.get("overall_score", 0)
        else:
            normalized["_risk_level"] = "UNKNOWN"
            normalized["_risk_score"] = 0

        # Normalize timestamp
        if "_timestamp" not in normalized:
            normalized["_timestamp"] = (
                data.get("timestamp_utc") or
                data.get("timestamp") or
                data.get("created_at") or
                ""
            )

        # Normalize findings/issues count
        findings_count = 0
        if "tests" in data:
            findings_count = len([t for t in data.get("tests", []) if t.get("result") == "fail"])
        elif "tools" in data:
            findings_count = len([t for t in data.get("tools", []) if t.get("risk_level") in ["critical", "high"]])
        elif "findings" in data:
            findings_count = len(data.get("findings", []))
        elif "risk_patterns" in data:
            findings_count = len(data.get("risk_patterns", []))
        normalized["_findings_count"] = findings_count

        # Normalize summary
        if "_summary" not in normalized:
            if "risk_assessment" in data:
                normalized["_summary"] = data["risk_assessment"].get("summary", "")
            elif "overall_risk" in data:
                normalized["_summary"] = data["overall_risk"].get("notes", "")
            else:
                normalized["_summary"] = data.get("summary", "")

        return normalized

    def collect_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Collect JSON reports from the reports directory and sandbox_logs.

        Args:
            limit: Maximum number of reports to include.

        Returns:
            List of report dictionaries.
        """
        reports = []
        all_json_files = []

        # Collect from reports path
        if self.reports_path.exists():
            all_json_files.extend(self.reports_path.glob("**/*.json"))

        # Also collect from sandbox_logs (BenderBox analysis output)
        from benderbox.config import get_benderbox_home
        sandbox_logs = get_benderbox_home() / "sandbox_logs"
        if sandbox_logs.exists():
            all_json_files.extend(sandbox_logs.glob("benderbox_*.json"))

        # Also check current working directory for sandbox_logs
        cwd_sandbox = Path("./sandbox_logs")
        if cwd_sandbox.exists() and cwd_sandbox.resolve() != sandbox_logs.resolve():
            all_json_files.extend(cwd_sandbox.glob("benderbox_*.json"))

        if not all_json_files:
            logger.warning(f"No report files found in {self.reports_path} or sandbox_logs")
            return reports

        # Sort by modification time and limit
        json_files = sorted(
            set(all_json_files),  # Deduplicate
            key=lambda p: p.stat().st_mtime,
            reverse=True  # Most recent first
        )[:limit]

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Ensure it looks like a valid report from any analysis type
                    # Model analysis: has 'tests', 'overall_risk', 'model'
                    # MCP analysis: has 'analysis_type', 'server', 'risk_assessment', 'tools'
                    # Context analysis: has 'risk_patterns', 'findings'
                    # Generic: has 'target_name', 'results', 'summary'
                    is_valid_report = any([
                        "target_name" in data,
                        "results" in data,
                        "summary" in data,
                        "tests" in data,
                        "analysis_type" in data,  # MCP/context analysis
                        "risk_assessment" in data,  # MCP analysis
                        "overall_risk" in data,  # Model analysis
                        "tools" in data,  # MCP analysis
                        "risk_patterns" in data,  # Context analysis
                        "findings" in data,  # Context analysis
                        "schema_version" in data,  # Any BenderBox report
                    ])
                    if is_valid_report:
                        # Normalize the report data for consistent display
                        data = self._normalize_report(data, json_file)
                        data["_source_file"] = str(json_file)
                        reports.append(data)
            except Exception as e:
                logger.warning(f"Failed to load report {json_file}: {e}")

        logger.info(f"Collected {len(reports)} reports")
        return reports

    def generate(
        self,
        reports: Optional[List[Dict[str, Any]]] = None,
        limit: int = 50,
    ) -> str:
        """
        Generate the report viewer HTML.

        Args:
            reports: List of reports to include. If None, collects from reports_path.
            limit: Maximum reports if collecting from directory.

        Returns:
            Complete HTML string.
        """
        if reports is None:
            reports = self.collect_reports(limit)

        # Embed report data as JSON
        report_data = json.dumps(reports, indent=2, default=str)

        return REPORT_VIEWER_HTML.replace("{report_data}", report_data)

    def save(
        self,
        output_path: Optional[str] = None,
        reports: Optional[List[Dict[str, Any]]] = None,
        open_browser: bool = False,
    ) -> str:
        """
        Generate and save the report viewer HTML.

        Args:
            output_path: Output file path. Defaults to reports_path/BenderBox_Report_Viewer.html
            reports: Reports to include. If None, collects from reports_path.
            open_browser: Open in browser after saving.

        Returns:
            Path to saved file.
        """
        if output_path is None:
            self.reports_path.mkdir(parents=True, exist_ok=True)
            output_path = str(self.reports_path / "BenderBox_Report_Viewer.html")

        html_content = self.generate(reports)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content, encoding="utf-8")

        logger.info(f"Report viewer saved to: {output_path}")

        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{output_file.absolute()}")

        return str(output_file)

    def generate_for_report(
        self,
        report: Dict[str, Any],
        output_path: Optional[str] = None,
        open_browser: bool = False,
    ) -> str:
        """
        Generate viewer for a single report.

        Args:
            report: Single report dictionary.
            output_path: Output file path.
            open_browser: Open in browser after saving.

        Returns:
            Path to saved file.
        """
        return self.save(output_path, reports=[report], open_browser=open_browser)


def generate_report_viewer(
    reports_path: Optional[str] = None,
    output_path: Optional[str] = None,
    open_browser: bool = False,
) -> str:
    """
    Convenience function to generate the report viewer.

    Args:
        reports_path: Path to reports directory.
        output_path: Output file path.
        open_browser: Open in browser after saving.

    Returns:
        Path to generated HTML file.
    """
    generator = ReportViewerGenerator(reports_path)
    return generator.save(output_path, open_browser=open_browser)
