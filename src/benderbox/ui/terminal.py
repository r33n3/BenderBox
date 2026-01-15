"""
Terminal UI Components for BenderBox

Rich-based terminal interface components for beautiful console output.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Path to ASCII art file
ASCII_ART_FILE = Path(__file__).parent.parent.parent.parent / "BenderBox_ASCII.txt"

# Enable UTF-8 mode on Windows for proper Unicode support
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8 code page
        kernel32.SetConsoleCP(65001)
        # Reconfigure stdout/stderr for UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Try to import Rich, gracefully degrade if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.style import Style
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


class TerminalUI:
    """
    Rich-based terminal UI for BenderBox.

    Provides beautiful console output with:
    - Futurama-inspired retro-futuristic theme
    - Neon color scheme matching report viewer
    - Progress indicators
    - Risk-level color coding
    """

    # Futurama-inspired neon color scheme (matching report viewer)
    # Use RGB hex colors for exact matching where possible
    THEME = {
        # Neon accent colors
        "neon_green": "#39ff14",      # Primary accent - neon green
        "neon_purple": "#bf00ff",     # Secondary accent - purple
        "neon_orange": "#ff6b35",     # Tertiary accent - orange
        "neon_cyan": "#00ffff",       # Info/headers - cyan
        "neon_pink": "#ff00ff",       # Alternate accent
        # Status colors
        "danger": "#ff3366",          # Errors/critical
        "warning": "#ffaa00",         # Warnings
        "success": "#00ff88",         # Success/passed
        # Text colors
        "text_primary": "#e0e0e0",    # Main text
        "text_muted": "#888899",      # Muted/dim text
    }

    # Color scheme for risk levels (neon-style)
    RISK_COLORS = {
        "critical": "bold #ff3366",   # Neon red/pink
        "high": "bold #ffaa00",       # Neon orange
        "medium": "bold #ffff00",     # Neon yellow
        "low": "bold #00ff88",        # Neon green
        "info": "bold #00ffff",       # Neon cyan
        "safe": "bold #39ff14",       # Bright neon green
        "unknown": "bold #888899",    # Muted
    }

    RISK_EMOJI = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
        "info": "ℹ️",
        "safe": "✅",
        "unknown": "⚪",
    }

    def __init__(self, console: Optional["Console"] = None):
        """
        Initialize TerminalUI.

        Args:
            console: Optional Rich Console instance.
        """
        if RICH_AVAILABLE:
            # Force UTF-8 and disable legacy Windows renderer for proper Unicode
            self.console = console or Console(force_terminal=True, legacy_windows=False)
        else:
            self.console = None
            logger.warning("Rich not available. Using basic output.")

    def print_banner(self) -> None:
        """Print the BenderBox banner with ASCII art in Futurama neon style."""
        # Try to load ASCII art from file
        banner = self._load_ascii_art()

        if self.console:
            # Neon green banner with purple border - Futurama style
            self.console.print(Panel(
                Text(banner, style=f"bold {self.THEME['neon_green']}"),
                title=f"[bold {self.THEME['neon_orange']}]AI Security Analysis Platform[/bold {self.THEME['neon_orange']}]",
                subtitle=f"[{self.THEME['text_muted']}]v3.0.0-alpha[/{self.THEME['text_muted']}]",
                border_style=self.THEME['neon_purple'],
            ))
        else:
            print(banner)
            print("BenderBox - AI Security Analysis Platform v3.0.0-alpha")
            print("=" * 60)

    def _load_ascii_art(self) -> str:
        """Load ASCII art from file or return fallback."""
        try:
            if ASCII_ART_FILE.exists():
                return ASCII_ART_FILE.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not load ASCII art: {e}")

        # Fallback banner if file not found
        return """
██████╗ ███████╗███╗   ██╗██████╗ ███████╗██████╗ ██████╗  ██████╗ ██╗  ██╗
██╔══██╗██╔════╝████╗  ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔═══██╗╚██╗██╔╝
██████╔╝█████╗  ██╔██╗ ██║██║  ██║█████╗  ██████╔╝██████╔╝██║   ██║ ╚███╔╝
██╔══██╗██╔══╝  ██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗██╔══██╗██║   ██║ ██╔██╗
██████╔╝███████╗██║ ╚████║██████╔╝███████╗██║  ██║██████╔╝╚██████╔╝██╔╝ ██╗
╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝
        """

    def print_risk_badge(
        self,
        risk_level: str,
        risk_score: float,
        large: bool = False,
    ) -> None:
        """
        Print a risk level badge.

        Args:
            risk_level: Risk level string.
            risk_score: Numeric risk score (0-100).
            large: Use large format.
        """
        risk_level = risk_level.lower()
        color = self.RISK_COLORS.get(risk_level, "white")
        emoji = self.RISK_EMOJI.get(risk_level, "")

        if self.console:
            if large:
                self.console.print(Panel(
                    f"{emoji} [bold]{risk_level.upper()}[/bold]\n[dim]Score: {risk_score}/100[/dim]",
                    style=color,
                    width=30,
                ))
            else:
                self.console.print(
                    f"{emoji} Risk: [{color}]{risk_level.upper()}[/{color}] ({risk_score}/100)"
                )
        else:
            print(f"{emoji} Risk: {risk_level.upper()} ({risk_score}/100)")

    def print_analysis_summary(self, result: Dict[str, Any]) -> None:
        """
        Print a formatted analysis summary.

        Args:
            result: Analysis result dictionary.
        """
        target = result.get("target_name", "Unknown")
        target_type = result.get("target_type", "unknown")
        profile = result.get("profile", "standard")
        summary = result.get("summary", {})
        risk = summary.get("risk", {})
        risk_level = risk.get("level", "unknown")
        risk_score = risk.get("score", 0)
        results = result.get("results", [])

        if self.console:
            # Create summary table with neon Futurama theme
            table = Table(
                title=f"[bold {self.THEME['neon_green']}]Analysis Summary: {target}[/bold {self.THEME['neon_green']}]",
                box=box.ROUNDED,
                show_header=True,
                header_style=f"bold {self.THEME['neon_purple']}",
                border_style=self.THEME['neon_purple'],
            )

            table.add_column("Property", style=f"bold {self.THEME['neon_cyan']}")
            table.add_column("Value", style=self.THEME['text_primary'])

            table.add_row("Target", target)
            table.add_row("Type", target_type)
            table.add_row("Profile", profile)
            table.add_row(
                "Risk Level",
                Text(f"{self.RISK_EMOJI.get(risk_level.lower(), '')} {risk_level.upper()}",
                     style=self.RISK_COLORS.get(risk_level.lower(), "white"))
            )
            table.add_row("Risk Score", f"[bold {self.THEME['neon_orange']}]{risk_score}[/bold {self.THEME['neon_orange']}]/100")
            table.add_row("Total Tests", str(len(results)))

            # Count by status
            passed = sum(1 for r in results if r.get("status") == "passed")
            failed = sum(1 for r in results if r.get("status") == "failed")
            warnings = sum(1 for r in results if r.get("status") == "warning")

            table.add_row("Passed", Text(str(passed), style=self.THEME['success']))
            table.add_row("Failed", Text(str(failed), style=self.THEME['danger']))
            table.add_row("Warnings", Text(str(warnings), style=self.THEME['warning']))

            self.console.print(table)
        else:
            print(f"\nAnalysis Summary: {target}")
            print(f"Type: {target_type} | Profile: {profile}")
            print(f"Risk: {risk_level.upper()} ({risk_score}/100)")
            print(f"Tests: {len(results)} total")

    def print_findings_table(
        self,
        findings: List[Dict[str, Any]],
        title: str = "Findings",
        max_items: int = 20,
    ) -> None:
        """
        Print a table of findings.

        Args:
            findings: List of finding dictionaries.
            title: Table title.
            max_items: Maximum items to display.
        """
        if not findings:
            if self.console:
                self.console.print("[green]No findings to display.[/green]")
            else:
                print("No findings to display.")
            return

        if self.console:
            # Futurama-styled findings table
            table = Table(
                title=f"[bold {self.THEME['neon_orange']}]{title}[/bold {self.THEME['neon_orange']}]",
                box=box.ROUNDED,
                show_header=True,
                header_style=f"bold {self.THEME['neon_purple']}",
                border_style=self.THEME['neon_purple'],
            )

            table.add_column("Severity", width=10)
            table.add_column("Test", width=30, style=self.THEME['text_primary'])
            table.add_column("Category", width=15, style=self.THEME['neon_cyan'])
            table.add_column("Status", width=10)

            for finding in findings[:max_items]:
                severity = finding.get("severity", "info").lower()
                test_name = finding.get("test_name", "Unknown")[:28]
                category = finding.get("category", "N/A")[:13]
                status = finding.get("status", "unknown")

                severity_text = Text(
                    f"{self.RISK_EMOJI.get(severity, '')} {severity.upper()}",
                    style=self.RISK_COLORS.get(severity, "white")
                )

                status_style = {
                    "passed": self.THEME['success'],
                    "failed": self.THEME['danger'],
                    "warning": self.THEME['warning'],
                    "error": self.THEME['danger'],
                }.get(status, self.THEME['text_muted'])

                table.add_row(
                    severity_text,
                    test_name,
                    category,
                    Text(status.upper(), style=status_style),
                )

            self.console.print(table)

            if len(findings) > max_items:
                self.console.print(f"[{self.THEME['text_muted']}]... and {len(findings) - max_items} more findings[/{self.THEME['text_muted']}]")
        else:
            print(f"\n{title}")
            print("-" * 60)
            for finding in findings[:max_items]:
                severity = finding.get("severity", "info").upper()
                test_name = finding.get("test_name", "Unknown")
                status = finding.get("status", "unknown").upper()
                print(f"[{severity}] {test_name} - {status}")

    def print_model_info(self, model_info: Dict[str, Any]) -> None:
        """
        Print model information panel in neon style.

        Args:
            model_info: Model information dictionary.
        """
        if self.console:
            metadata = model_info.get("metadata", {})

            tree = Tree(f"[bold {self.THEME['neon_green']}]Model Information[/bold {self.THEME['neon_green']}]")
            tree.add(f"[{self.THEME['neon_cyan']}]Name:[/{self.THEME['neon_cyan']}] {model_info.get('name', 'Unknown')}")
            tree.add(f"[{self.THEME['neon_cyan']}]Path:[/{self.THEME['neon_cyan']}] {model_info.get('path', 'N/A')}")
            tree.add(f"[{self.THEME['neon_cyan']}]Size:[/{self.THEME['neon_cyan']}] {model_info.get('size_bytes', 0) / (1024*1024):.1f} MB")

            if metadata:
                meta_branch = tree.add(f"[bold {self.THEME['neon_orange']}]Metadata[/bold {self.THEME['neon_orange']}]")
                if metadata.get("architecture"):
                    meta_branch.add(f"Architecture: {metadata['architecture']}")
                if metadata.get("parameter_count"):
                    meta_branch.add(f"Parameters: {metadata['parameter_count']}")
                if metadata.get("quantization"):
                    meta_branch.add(f"Quantization: {metadata['quantization']}")
                if metadata.get("context_length"):
                    meta_branch.add(f"Context: {metadata['context_length']}")

            self.console.print(Panel(tree, border_style=self.THEME['neon_purple']))
        else:
            print("\nModel Information:")
            print(f"  Name: {model_info.get('name', 'Unknown')}")
            print(f"  Path: {model_info.get('path', 'N/A')}")

    def print_help(self, category: str = None) -> None:
        """Print help information, optionally filtered by category."""

        main_help = """
# BenderBox Help

Type `help <category>` for detailed help on a topic.

## Categories

| Category | Description |
|----------|-------------|
| `help models` | Model loading & management |
| `help interrogate` | Model interrogation & security testing |
| `help mcp` | MCP server security testing |
| `help context` | Context/instruction file analysis |
| `help reports` | Report management & export |
| `help config` | Configuration & API keys |
| `help examples` | Usage examples |

## Quick Reference

| Command | Description |
|---------|-------------|
| `list models` | List available models |
| `analyze <model>` | Analyze model for security issues |
| `analyze phi-2 --profile full` | Full security test |
| `compare <a> <b>` | Compare two models |
| `mcp tools <target>` | List MCP server tools |
| `mcp interrogate <target>` | Security test MCP server |
| `context analyze <file>` | Analyze instruction file |
| `report` | Open report viewer |
| `status` | Show system status |
| `exit` | Exit BenderBox |

## Natural Language

You can also use natural language:
- "review model phi-2 and run security tests"
- "what models do I have?"
- "test the mcp server for vulnerabilities"
- "show my reports"
"""

        models_help = """
# Model Analysis & Management

## Loading Models

BenderBox supports two separate model slots:

| Purpose | Use Case | Command |
|---------|----------|---------|
| **NLP** | Powers BenderBox chat responses | `/load <model> --for nlp` |
| **Analysis** | Target model to interrogate/analyze | `/load <model> --for analysis` |

## Model Commands

| Command | Description |
|---------|-------------|
| `/models` | List all available models (all locations) |
| `/models list --for analysis` | List analysis models only |
| `/models list --for nlp` | List NLP models only |
| `/load <name> --for nlp` | Load model for chat responses |
| `/load <name> --for analysis` | Load model as analysis target |
| `/load <name>` | Load for analysis (default) |
| `/current` | Show currently loaded models |
| `/unload nlp` | Unload NLP model |
| `/unload analysis` | Unload analysis model |
| `/unload all` | Unload all models |

## Interrogation Workflow

1. **Load a model for analysis:**
   `/load phi-2 --for analysis`

2. **Run interrogation (uses loaded model):**
   `analyze` or `analyze --profile adversarial`

3. **Or specify model directly:**
   `analyze tinyllama --profile full`

## Analysis Profiles

| Profile | Tests | Use Case |
|---------|-------|----------|
| `quick` | ~15 | Fast CI/CD validation |
| `standard` | ~50 | Default balanced coverage |
| `full` | ~100+ | Pre-deployment audit |
| `adversarial` | ~64 | Jailbreak resistance testing |

## Examples

```
/models                           # See all available models
/load qwen2 --for nlp             # Load for chat
/load phi-2 --for analysis        # Load for interrogation
/current                          # See what's loaded
analyze                           # Analyze loaded model
analyze --profile adversarial     # Test jailbreak resistance
compare phi-2 tinyllama           # Compare two models
```
"""

        mcp_help = """
# MCP Server Security Testing

## MCP Commands

| Command | Description |
|---------|-------------|
| `/mcp tools <target>` | List tools from MCP server |
| `/mcp interrogate <target>` | Security test MCP server |
| `/mcp connect <target>` | Connect to MCP server |
| `/mcp call <target> <tool>` | Call a specific tool |

## Profiles

| Profile | Description |
|---------|-------------|
| `infra-quick` | Fast static analysis |
| `infra-standard` | Standard security tests (default) |
| `infra-deep` | Comprehensive security audit |

## Examples

- `/mcp tools https://github.com/org/mcp-server`
- `/mcp interrogate server.py --profile infra-deep`
- "Test this MCP server for vulnerabilities"
- "What tools does this server expose?"
"""

        context_help = """
# Context & Instruction Analysis

## Context Commands

| Command | Description |
|---------|-------------|
| `/context analyze <file>` | Analyze instruction file |
| `/context scan <directory>` | Scan directory for issues |
| `/context output <text>` | Analyze model output |

## Supported File Types

- Skill files (`.md`, `.yaml`, `.json`)
- System prompts (`.txt`, `.md`)
- Agent instructions
- Inference outputs

## Examples

- `/context analyze skills.md`
- `/context scan ./prompts`
- "Analyze this prompt for security risks"
- "Scan this directory for dangerous patterns"
"""

        reports_help = """
# Report Management

## Report Commands

| Command | Description |
|---------|-------------|
| `reports` | List recent analysis reports |
| `export --format json` | Export last result as JSON |
| `export --format html --open` | Export and open HTML report |
| "view reports" | Open report viewer in browser |

## Export Formats

| Format | Description |
|--------|-------------|
| `markdown` | Human-readable Markdown |
| `json` | Machine-readable JSON |
| `html` | Interactive HTML with charts |
| `csv` | CSV for spreadsheets |
| `sarif` | SARIF for tool integration |

## Examples

- `reports` - Show recent analyses
- "Open the report viewer"
- "Export this as HTML"
"""

        examples_help = """
# Usage Examples

## Model Analysis

- `analyze model.gguf --profile full`
- `/load llama-7b` then `analyze`
- "Analyze llama with adversarial profile"
- "Compare llama to mistral"
- "Is this model safe for production?"

## MCP Security

- `/mcp interrogate server.py`
- "Test this MCP server for injection vulnerabilities"
- "What tools does github.com/org/server expose?"

## Code Analysis

- `semantic utils.py --depth deep`
- "Review this Python file for security issues"
- `search SQL injection`

## Natural Language

- "What models do I have?"
- "List my analysis reports"
- "Why is the risk score high?"
- "Explain the jailbreak findings"
"""

        interrogate_help = """
# Model Interrogation Guide

## What is Interrogation?

Interrogation runs security tests against an AI model to evaluate:
- **Safety guardrails** - Does it refuse harmful requests?
- **Jailbreak resistance** - Can it be tricked into unsafe behavior?
- **Instruction following** - Does it stay on-topic?
- **Output consistency** - Are responses predictable and safe?

## Quick Start

```
# Step 1: See available models
/models

# Step 2: Load a model for interrogation
/load phi-2 --for analysis

# Step 3: Run interrogation
analyze --profile adversarial
```

## Interrogation Profiles

| Profile | Description | Tests |
|---------|-------------|-------|
| `quick` | Fast validation - basic safety checks | ~15 |
| `standard` | Balanced coverage - most common attack vectors | ~50 |
| `full` | Comprehensive - all test categories | ~100+ |
| `adversarial` | Jailbreak-focused - prompt injection, roleplay attacks | ~64 |

## Test Categories

| Category | What it Tests |
|----------|---------------|
| **Jailbreak** | DAN prompts, roleplay attacks, system prompt extraction |
| **Injection** | Prompt injection, instruction override attempts |
| **Harmful Content** | Requests for dangerous/illegal information |
| **Bias & Toxicity** | Discriminatory or offensive outputs |
| **Hallucination** | Factual accuracy, made-up information |
| **Privacy** | PII handling, data leakage |

## Commands

| Command | Description |
|---------|-------------|
| `/load <model> --for analysis` | Load model as interrogation target |
| `analyze` | Analyze loaded model (standard profile) |
| `analyze --profile <name>` | Analyze with specific profile |
| `analyze <model>` | Analyze model directly by name |
| `compare <model1> <model2>` | Compare two models |
| `/current` | Show loaded model |

## Workflow Examples

**Basic interrogation:**
```
/load tinyllama --for analysis
analyze --profile quick
```

**Full security audit:**
```
/load phi-2 --for analysis
analyze --profile full
export --format html --open
```

**Compare models:**
```
compare tinyllama phi-2 --profile adversarial
```

**Natural language:**
```
"Analyze tinyllama for jailbreak vulnerabilities"
"Test phi-2 with adversarial profile"
"Is this model safe for production?"
```

## Understanding Results

- **Risk Score (0-100)**: Overall safety rating (lower is better)
- **Findings**: Specific vulnerabilities discovered
- **Severity Levels**: Critical > High > Medium > Low > Info

## Tips

1. Start with `quick` profile for fast feedback
2. Use `adversarial` profile before deploying user-facing models
3. Compare multiple models to find the safest option
4. Export HTML reports for sharing with teams
"""

        config_help = """
# Configuration & API Keys

## API Key Management

BenderBox can test cloud AI providers (OpenAI, Anthropic, Google, xAI).

| Command | Description |
|---------|-------------|
| `config set-key openai` | Set OpenAI API key |
| `config set-key anthropic` | Set Anthropic API key |
| `config api-keys` | List configured keys |
| `config test-key openai` | Test API connection |

## Supported Providers

| Provider | Target Format |
|----------|---------------|
| OpenAI | `openai:gpt-4-turbo` |
| Anthropic | `anthropic:claude-3-5-sonnet-20241022` |
| Google | `gemini:gemini-1.5-pro` |
| xAI | `grok:grok-2` |

## Environment Variables

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...
```

## Examples

```
# Set API key (masked input)
python bb.py config set-key openai

# Test cloud model
analyze openai:gpt-4-turbo --profile quick

# Compare local vs cloud
compare phi-2 openai:gpt-4-turbo
```
"""

        # Category mapping
        categories = {
            "models": models_help,
            "model": models_help,
            "interrogate": interrogate_help,
            "interrogation": interrogate_help,
            "analyze": interrogate_help,
            "analysis": interrogate_help,
            "mcp": mcp_help,
            "context": context_help,
            "ctx": context_help,
            "reports": reports_help,
            "report": reports_help,
            "examples": examples_help,
            "example": examples_help,
            "config": config_help,
            "api": config_help,
            "keys": config_help,
        }

        if category and category.lower() in categories:
            help_text = categories[category.lower()]
        else:
            help_text = main_help

        if self.console:
            self.console.print(Markdown(help_text))
        else:
            print(help_text)

    def print_error(self, message: str) -> None:
        """Print an error message in neon danger style."""
        if self.console:
            self.console.print(f"[bold {self.THEME['danger']}]✖ Error:[/bold {self.THEME['danger']}] {message}")
        else:
            print(f"Error: {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message in neon orange style."""
        if self.console:
            self.console.print(f"[bold {self.THEME['warning']}]⚠ Warning:[/bold {self.THEME['warning']}] {message}")
        else:
            print(f"Warning: {message}")

    def print_success(self, message: str) -> None:
        """Print a success message in neon green style."""
        if self.console:
            self.console.print(f"[bold {self.THEME['neon_green']}]✓[/bold {self.THEME['neon_green']}] {message}")
        else:
            print(f"✓ {message}")

    def print_info(self, message: str) -> None:
        """Print an info message in neon cyan style."""
        if self.console:
            self.console.print(f"[bold {self.THEME['neon_cyan']}]ℹ[/bold {self.THEME['neon_cyan']}] {message}")
        else:
            print(f"ℹ {message}")

    def print_metrics(self, metrics_line: str, inline: bool = True) -> None:
        """
        Print system metrics line (for real-time monitoring).
        
        Args:
            metrics_line: Formatted metrics string.
            inline: If True, overwrites current line.
        """
        if self.console:
            style = f"dim {self.THEME['text_secondary']}"
            if inline:
                self.console.print(f"  [{style}]{metrics_line}[/{style}]", end="")
            else:
                self.console.print(f"  [{style}]{metrics_line}[/{style}]")
        else:
            if inline:
                print(f"  {metrics_line}", end="")
            else:
                print(f"  {metrics_line}")

    def print_download_progress(self, filename: str, progress: float, speed: str = "") -> None:
        """
        Print download progress bar.
        
        Args:
            filename: Name of file being downloaded.
            progress: Progress percentage (0-100).
            speed: Download speed string.
        """
        bar_width = 30
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        speed_str = f" ({speed})" if speed else ""
        line = f"  Downloading {filename}: [{bar}] {progress:.0f}%{speed_str}"
        
        if self.console:
            self.console.print(f"[{self.THEME['neon_cyan']}]{line}[/{self.THEME['neon_cyan']}]", end="")
        else:
            print(line, end="")

    def print_header(self, title: str) -> None:
        """Print a section header in neon purple style."""
        if self.console:
            self.console.print(f"\n[bold {self.THEME['neon_purple']}]═══ {title} ═══[/bold {self.THEME['neon_purple']}]\n")
        else:
            print(f"\n=== {title} ===\n")

    def print_markdown(self, content: str) -> None:
        """
        Print markdown content.

        Args:
            content: Markdown string.
        """
        if self.console:
            self.console.print(Markdown(content))
        else:
            print(content)

    def print_code(self, code: str, language: str = "python") -> None:
        """
        Print syntax-highlighted code.

        Args:
            code: Source code string.
            language: Programming language.
        """
        if self.console:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            print(f"```{language}")
            print(code)
            print("```")

    def create_progress(self) -> Optional["Progress"]:
        """
        Create a progress bar context manager.

        Returns:
            Rich Progress instance or None.
        """
        if self.console and RICH_AVAILABLE:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )
        return None

    def input_prompt(self, prompt: str = "You: ") -> str:
        """
        Get user input with neon-styled prompt.

        Args:
            prompt: Prompt string.

        Returns:
            User input string.
        """
        if self.console:
            # Neon green prompt for Futurama feel
            return self.console.input(f"[bold {self.THEME['neon_green']}]{prompt}[/bold {self.THEME['neon_green']}]")
        return input(prompt)


class ProgressSpinner:
    """Context manager for progress spinner with neon styling."""

    def __init__(self, ui: TerminalUI, message: str = "Processing..."):
        """
        Initialize spinner.

        Args:
            ui: TerminalUI instance.
            message: Progress message.
        """
        self.ui = ui
        self.message = message
        self._live = None
        self._spinner = None

    def __enter__(self):
        if self.ui.console and RICH_AVAILABLE:
            from rich.spinner import Spinner
            # Neon cyan spinner for Futurama feel
            neon_cyan = TerminalUI.THEME['neon_cyan']
            styled_message = f"[{neon_cyan}]{self.message}[/{neon_cyan}]"
            self._spinner = Spinner("dots", text=styled_message, style=neon_cyan)
            self._live = Live(self._spinner, console=self.ui.console, refresh_per_second=10)
            self._live.__enter__()
        else:
            print(f"{self.message}...", end="", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        else:
            print(" done." if not exc_type else " failed.")

    def update(self, message: str) -> None:
        """Update spinner message."""
        self.message = message
        if self._spinner:
            self._spinner.text = message
