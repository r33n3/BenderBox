#!/usr/bin/env python3
"""
BenderBox MCP Server

This module provides Model Context Protocol (MCP) tools for BenderBox.
These tools are stateless wrappers around the benderbox_sandbox_cli.py engine.

Architecture:
  - Skills (Markdown) → MCP Tools (this file) → Sandbox CLI (Python)
  - MCP tools NEVER implement analysis logic
  - MCP tools ONLY invoke CLI and parse results
  - All business logic stays in benderbox_sandbox_cli.py
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("[BenderBox MCP] Warning: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)


# ---------- Configuration ----------

DEFAULT_LOG_DIR = "./sandbox_logs"
DEFAULT_CLI_PATH = "./sandbox_cli.py"


# ---------- Helper Functions ----------

def find_sandbox_cli() -> Optional[Path]:
    """
    Locate the sandbox_cli.py script.
    Checks:
      1. Same directory as this file
      2. Current working directory
      3. PATH environment variable
    """
    # Check same directory
    script_dir = Path(__file__).parent
    local_cli = script_dir / "sandbox_cli.py"
    if local_cli.exists():
        return local_cli

    # Check current directory
    cwd_cli = Path("sandbox_cli.py")
    if cwd_cli.exists():
        return cwd_cli

    # Check if it's executable in PATH (unlikely for .py, but possible)
    import shutil
    path_cli = shutil.which("sandbox_cli.py")
    if path_cli:
        return Path(path_cli)

    return None


def invoke_sandbox_cli(
    model_path: str,
    profile: str = "standard",
    tests: Optional[List[str]] = None,
    log_dir: str = DEFAULT_LOG_DIR,
    cli_path: Optional[str] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Invoke benderbox_sandbox_cli.py and return parsed JSON result.

    This is a STATELESS wrapper - it does not implement any analysis logic.

    Args:
        model_path: Path to GGUF model file
        profile: Analysis profile (quick/standard/deep/attack/custom)
        tests: Optional list of specific tests (for custom profile)
        log_dir: Directory for JSON reports
        cli_path: Override path to CLI script
        timeout: Maximum execution time in seconds

    Returns:
        Parsed JSON report from sandbox analysis

    Raises:
        FileNotFoundError: If CLI or model not found
        subprocess.CalledProcessError: If CLI execution fails
        subprocess.TimeoutExpired: If execution exceeds timeout
        json.JSONDecodeError: If CLI output is not valid JSON
    """
    # Locate CLI script
    if cli_path:
        cli_script = Path(cli_path)
    else:
        cli_script = find_sandbox_cli()

    if not cli_script or not cli_script.exists():
        raise FileNotFoundError(
            f"sandbox_cli.py not found. "
            f"Searched: {cli_script}, ./sandbox_cli.py"
        )

    # Validate model exists
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Build CLI command
    cmd = [
        sys.executable,  # Use same Python interpreter
        str(cli_script),
        "--model", str(model_file),
        "--profile", profile,
        "--log-dir", log_dir,
        "--format", "json",  # CRITICAL: JSON output only
        "--no-fail-on-test-errors",  # Always return 0 for MCP compatibility
    ]

    # Add custom tests if specified
    if tests and profile == "custom":
        cmd.extend(["--tests", ",".join(tests)])

    # Execute CLI
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
    )

    # CLI should always return 0 with --no-fail-on-test-errors
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    # Parse JSON output
    try:
        report = json.loads(result.stdout)
        return report
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"CLI did not return valid JSON. Output: {result.stdout[:500]}",
            e.doc,
            e.pos,
        )


def get_latest_report(
    log_dir: str = DEFAULT_LOG_DIR,
    model_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest sandbox report from log directory.

    Args:
        log_dir: Directory containing JSON reports
        model_name: Optional filter by model name

    Returns:
        Parsed JSON report, or None if no reports found
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    reports = sorted(
        log_path.glob("benderbox_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        return None

    # Filter by model name if specified
    if model_name:
        for report_file in reports:
            try:
                with report_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("model", {}).get("name") == model_name:
                    return data
            except Exception:
                continue
        return None

    # Return latest report
    try:
        with reports[0].open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def generate_prompt_variants(
    prompt_text: str,
    techniques: Optional[List[str]] = None,
    max_variants: int = 10,
) -> Dict[str, Any]:
    """
    Generate jailbreak variants of a prompt.

    Args:
        prompt_text: The original prompt to generate variants for.
        techniques: Optional list of techniques to use.
        max_variants: Maximum number of variants to generate.

    Returns:
        Dict with generated variants.
    """
    try:
        from benderbox.interrogation.prompts.schema import (
            TestPrompt,
            PromptCategory,
            ExpectedBehavior,
        )
        from benderbox.interrogation.variants import (
            VariantGenerator,
            VariantTechnique,
            VariantConfig,
        )

        # Create base prompt
        base_prompt = TestPrompt(
            id="mcp_prompt",
            prompt=prompt_text,
            category=PromptCategory.BASELINE_SAFETY,
            expected_behavior=ExpectedBehavior.REFUSE,
        )

        # Parse techniques
        technique_enums = None
        if techniques:
            technique_enums = []
            for t in techniques:
                try:
                    technique_enums.append(VariantTechnique(t.lower()))
                except ValueError:
                    pass  # Skip unknown techniques

        # Configure generator
        config = VariantConfig(
            techniques=technique_enums or list(VariantTechnique),
            max_variants=max_variants,
        )

        generator = VariantGenerator(config)
        variants = generator.generate(base_prompt, technique_enums)

        return {
            "original_prompt": prompt_text,
            "variant_count": len(variants),
            "variants": [
                {
                    "id": v.id,
                    "prompt": v.prompt,
                    "technique": v.bypass_technique,
                    "description": v.description,
                }
                for v in variants
            ],
        }

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


def compare_model_reports(
    report_paths: List[str],
) -> Dict[str, Any]:
    """
    Compare multiple model interrogation reports.

    Args:
        report_paths: List of paths to JSON report files.

    Returns:
        Comparison report as dict.
    """
    try:
        from pathlib import Path
        from benderbox.interrogation.comparison import (
            ComparativeAnalyzer,
            ModelResult,
        )

        # Load reports
        models = []
        for path_str in report_paths:
            path = Path(path_str)
            if not path.exists():
                return {"error": f"Report not found: {path_str}"}

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            models.append(ModelResult.from_dict(data))

        if len(models) < 2:
            return {"error": "Comparison requires at least 2 reports"}

        # Run comparison
        analyzer = ComparativeAnalyzer()
        report = analyzer.compare(models)

        return report.to_dict()

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


def list_variant_techniques() -> Dict[str, Any]:
    """
    List all available variant generation techniques.

    Returns:
        Dict with techniques and their categories.
    """
    try:
        from benderbox.interrogation.variants.generator import (
            VariantTechnique,
            ENCODING_TECHNIQUES,
            ROLEPLAY_TECHNIQUES,
            CONTEXT_TECHNIQUES,
            EMOTIONAL_TECHNIQUES,
            INJECTION_TECHNIQUES,
            COMBINATION_TECHNIQUES,
        )

        return {
            "techniques": [t.value for t in VariantTechnique],
            "categories": {
                "encoding": [t.value for t in ENCODING_TECHNIQUES],
                "roleplay": [t.value for t in ROLEPLAY_TECHNIQUES],
                "context": [t.value for t in CONTEXT_TECHNIQUES],
                "emotional": [t.value for t in EMOTIONAL_TECHNIQUES],
                "injection": [t.value for t in INJECTION_TECHNIQUES],
                "combination": [t.value for t in COMBINATION_TECHNIQUES],
            },
            "total_count": len(VariantTechnique),
        }

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


def analyze_model_trends(
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    log_dir: str = DEFAULT_LOG_DIR,
) -> Dict[str, Any]:
    """
    Analyze trends for a model over time.

    Args:
        model_name: Model name to filter by.
        model_path: Model path to filter by.
        log_dir: Directory containing JSON reports.

    Returns:
        Trend analysis report as dict.
    """
    try:
        from pathlib import Path
        from benderbox.interrogation.trend import TrendTracker

        tracker = TrendTracker(log_dir=Path(log_dir))
        report = tracker.analyze(model_name=model_name, model_path=model_path)

        if report is None:
            return {
                "error": "Insufficient data for trend analysis",
                "message": "Need at least 2 reports for the same model",
            }

        return report.to_dict()

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


def list_available_tests(cli_path: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Query CLI for available tests.

    Args:
        cli_path: Override path to CLI script

    Returns:
        List of dicts with test name and category
    """
    # Locate CLI script
    if cli_path:
        cli_script = Path(cli_path)
    else:
        cli_script = find_sandbox_cli()

    if not cli_script or not cli_script.exists():
        return []

    # Execute CLI with --list-tests
    cmd = [sys.executable, str(cli_script), "--list-tests"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
        )

        if result.returncode != 0:
            return []

        # Parse output
        # Format: "  - test_name [category]"
        tests = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                # Extract: "test_name [category]"
                parts = line[2:].split("[")
                if len(parts) == 2:
                    name = parts[0].strip()
                    category = parts[1].rstrip("]").strip()
                    tests.append({"name": name, "category": category})

        return tests

    except Exception:
        return []


# ---------- MCP Tool Definitions ----------

if MCP_AVAILABLE:
    # Initialize MCP server
    app = Server("benderbox")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        """
        List all available MCP tools provided by BenderBox.
        """
        return [
            Tool(
                name="benderbox_sandbox_analyzeModel",
                description="""
Analyze a GGUF model file for safety, capabilities, and metadata.

This tool runs the BenderBox sandbox analysis pipeline and returns a comprehensive
JSON report including:
  - GGUF metadata (architecture, parameters, quantization, context length)
  - Risk assessment (level, score, primary factors)
  - Safety analysis (jailbreak resistance, backdoor detection)
  - Capability fingerprint
  - Test results with detailed findings

Profiles:
  - quick: Fast metadata extraction only (~5-10s)
  - standard: Common static tests (~10-15s) [Recommended]
  - deep: All available tests including dynamic inference (~60-90s)
  - attack: Security-focused jailbreak/backdoor tests (~45-60s)
  - custom: User-specified tests (requires 'tests' parameter)

The analysis is completely stateless and offline - no network calls are made.
                """.strip(),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_path": {
                            "type": "string",
                            "description": "Path to GGUF model file (absolute or relative)",
                        },
                        "profile": {
                            "type": "string",
                            "enum": ["quick", "standard", "deep", "attack", "custom"],
                            "default": "standard",
                            "description": "Analysis profile determining which tests to run",
                        },
                        "tests": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific tests to run (only for profile=custom)",
                        },
                        "log_dir": {
                            "type": "string",
                            "default": "./sandbox_logs",
                            "description": "Directory to store JSON reports",
                        },
                        "timeout": {
                            "type": "integer",
                            "default": 300,
                            "description": "Maximum execution time in seconds",
                        },
                    },
                    "required": ["model_path"],
                },
            ),
            Tool(
                name="benderbox_sandbox_getLatestReport",
                description="""
Retrieve the most recent sandbox analysis report from the log directory.

Optionally filter by model name to get the latest report for a specific model.

Returns the full JSON report including metadata, risk assessment, and test results.
                """.strip(),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "log_dir": {
                            "type": "string",
                            "default": "./sandbox_logs",
                            "description": "Directory containing JSON reports",
                        },
                        "model_name": {
                            "type": "string",
                            "description": "Optional filter by model filename (e.g., 'model.gguf')",
                        },
                    },
                },
            ),
            Tool(
                name="benderbox_sandbox_listTests",
                description="""
List all available sandbox tests and their categories.

Returns a list of test names and categories (static, dynamic, jailbreak, etc.)
that can be used with the 'custom' profile.
                """.strip(),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="benderbox_generateVariants",
                description="""
Generate jailbreak prompt variants from a base prompt.

Uses various techniques to transform prompts:
- Encoding: base64, rot13, leetspeak, unicode, reversed
- Roleplay: DAN, evil_assistant, developer_mode, uncensored_model
- Context: hypothetical, educational, fictional, research
- Emotional: urgency, guilt, authority
- Injection: ignore_previous, system_override, nested
- Combination: gradual_escalation, split_request

Returns variants with technique labels and descriptions.
                """.strip(),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The base prompt to generate variants for",
                        },
                        "techniques": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific techniques to use (optional, uses all if not specified)",
                        },
                        "max_variants": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum number of variants to generate",
                        },
                    },
                    "required": ["prompt"],
                },
            ),
            Tool(
                name="benderbox_listVariantTechniques",
                description="""
List all available variant generation techniques and their categories.

Returns technique names grouped by category (encoding, roleplay, context, etc.)
                """.strip(),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="benderbox_compareModels",
                description="""
Compare multiple model interrogation results.

Performs multi-dimensional comparison including:
- Overall risk scores
- Safety scores
- Jailbreak resistance
- Category-level performance
- Critical findings

Provides rankings, deltas, and recommendations.

Requires at least 2 report files to compare.
                """.strip(),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "report_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Paths to JSON report files to compare",
                            "minItems": 2,
                        },
                    },
                    "required": ["report_paths"],
                },
            ),
            Tool(
                name="benderbox_analyzeTrends",
                description="""
Analyze trends in model interrogation results over time.

Tracks historical reports and provides:
- Risk score trends (improving/degrading/stable/volatile)
- Safety and jailbreak resistance trends
- Category-level trend analysis
- Anomaly detection
- Alerts for significant changes
- Recommendations

Requires at least 2 reports for the same model.
                """.strip(),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "Model name to filter reports by",
                        },
                        "model_path": {
                            "type": "string",
                            "description": "Model path to filter reports by",
                        },
                        "log_dir": {
                            "type": "string",
                            "default": "./sandbox_logs",
                            "description": "Directory containing JSON reports",
                        },
                    },
                },
            ),
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """
        Execute MCP tool requests.

        All tools are stateless wrappers around benderbox_sandbox_cli.py.
        No analysis logic is implemented here.
        """
        try:
            if name == "benderbox_sandbox_analyzeModel":
                # Extract parameters
                model_path = arguments["model_path"]
                profile = arguments.get("profile", "standard")
                tests = arguments.get("tests")
                log_dir = arguments.get("log_dir", DEFAULT_LOG_DIR)
                timeout = arguments.get("timeout", 300)

                # Invoke CLI
                report = invoke_sandbox_cli(
                    model_path=model_path,
                    profile=profile,
                    tests=tests,
                    log_dir=log_dir,
                    timeout=timeout,
                )

                # Return JSON report
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(report, indent=2),
                    )
                ]

            elif name == "benderbox_sandbox_getLatestReport":
                log_dir = arguments.get("log_dir", DEFAULT_LOG_DIR)
                model_name = arguments.get("model_name")

                report = get_latest_report(log_dir=log_dir, model_name=model_name)

                if report is None:
                    return [
                        TextContent(
                            type="text",
                            text=json.dumps({
                                "error": "No reports found",
                                "log_dir": log_dir,
                                "model_name": model_name,
                            }, indent=2),
                        )
                    ]

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(report, indent=2),
                    )
                ]

            elif name == "benderbox_sandbox_listTests":
                tests = list_available_tests()

                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "tests": tests,
                            "count": len(tests),
                        }, indent=2),
                    )
                ]

            elif name == "benderbox_generateVariants":
                prompt = arguments["prompt"]
                techniques = arguments.get("techniques")
                max_variants = arguments.get("max_variants", 10)

                result = generate_prompt_variants(
                    prompt_text=prompt,
                    techniques=techniques,
                    max_variants=max_variants,
                )

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2),
                    )
                ]

            elif name == "benderbox_listVariantTechniques":
                result = list_variant_techniques()

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2),
                    )
                ]

            elif name == "benderbox_compareModels":
                report_paths = arguments["report_paths"]

                result = compare_model_reports(report_paths)

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2),
                    )
                ]

            elif name == "benderbox_analyzeTrends":
                model_name = arguments.get("model_name")
                model_path = arguments.get("model_path")
                log_dir = arguments.get("log_dir", DEFAULT_LOG_DIR)

                result = analyze_model_trends(
                    model_name=model_name,
                    model_path=model_path,
                    log_dir=log_dir,
                )

                return [
                    TextContent(
                        type="text",
                        text=json.dumps(result, indent=2),
                    )
                ]

            else:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps({
                            "error": f"Unknown tool: {name}",
                        }),
                    )
                ]

        except Exception as e:
            # Return error as JSON
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "type": type(e).__name__,
                    }, indent=2),
                )
            ]


# ---------- Main Entry Point ----------

def main():
    """
    Run the MCP server.

    Usage:
        python benderbox_mcp_server.py

    Or add to MCP client configuration:
        {
          "mcpServers": {
            "benderbox": {
              "command": "python",
              "args": ["/path/to/benderbox_mcp_server.py"]
            }
          }
        }
    """
    if not MCP_AVAILABLE:
        print("[BenderBox MCP] Error: MCP SDK not installed", file=sys.stderr)
        print("[BenderBox MCP] Install with: pip install mcp", file=sys.stderr)
        return 1

    # Check if CLI is available
    cli_path = find_sandbox_cli()
    if not cli_path:
        print("[BenderBox MCP] Warning: sandbox_cli.py not found", file=sys.stderr)
        print("[BenderBox MCP] MCP server will start, but tools may fail", file=sys.stderr)

    # Run MCP server
    import asyncio
    asyncio.run(stdio_server(app))
    return 0


if __name__ == "__main__":
    sys.exit(main())