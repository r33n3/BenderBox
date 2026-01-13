# BenderBox Examples

This directory contains example files for testing BenderBox's security analysis capabilities.

## Directory Structure

```
examples/
├── skills/           # AI assistant instruction templates
├── mcp_servers/      # Sample MCP server code
├── prompts/          # System prompts and instructions
└── models/           # Model analysis info
```

## Quick Start

```bash
# Analyze a skill file
python bb.py context analyze examples/skills/analyze_mcp_server.md

# Analyze an MCP server for vulnerabilities
python bb.py mcp analyze examples/mcp_servers/sample_vulnerable_server.py

# Analyze a system prompt
python bb.py context analyze examples/prompts/risky_system_prompt.md

# Compare safe vs risky prompts
python bb.py compare examples/prompts/safe_system_prompt.md examples/prompts/risky_system_prompt.md
```

---

## Skills (`examples/skills/`)

AI assistant instruction templates that demonstrate how to use BenderBox tools.

| File | Description |
|------|-------------|
| `analyze_gguf_model.md` | Instructions for model security analysis |
| `analyze_mcp_server.md` | Instructions for MCP server auditing |
| `analyze_skill_security.md` | Instructions for skill file analysis |
| `compare_model_safety.md` | Instructions for model comparison |
| `detect_model_jailbreaks.md` | Instructions for jailbreak testing |
| `generate_model_card.md` | Instructions for model card generation |

**Analyze skills for security issues:**
```bash
python bb.py context analyze examples/skills/analyze_mcp_server.md --type skill
```

---

## MCP Servers (`examples/mcp_servers/`)

Sample MCP server implementations for testing security analysis.

| File | Description |
|------|-------------|
| `sample_vulnerable_server.py` | Server with intentional vulnerabilities (for testing) |

**Analyze for vulnerabilities:**
```bash
python bb.py mcp analyze examples/mcp_servers/sample_vulnerable_server.py
```

**Expected findings:**
- Command injection in `execute_command()`
- Path traversal in `read_file()`
- SQL injection in `query_database()`
- Template injection in `render_template()`

---

## Prompts (`examples/prompts/`)

System prompt examples showing safe and risky patterns.

| File | Description |
|------|-------------|
| `safe_system_prompt.md` | Well-designed prompt following best practices |
| `risky_system_prompt.md` | Dangerous prompt with multiple vulnerabilities |

**Analyze prompts:**
```bash
# Should return minimal/no findings
python bb.py context analyze examples/prompts/safe_system_prompt.md

# Should return CRITICAL findings
python bb.py context analyze examples/prompts/risky_system_prompt.md
```

---

## Models (`examples/models/`)

Information about obtaining and analyzing GGUF models.

Models are too large to include in the repository. See `examples/models/README.md` for:
- How to download test models
- Analysis commands
- Expected output format

---

## Using Examples in Tests

These examples are also used as test fixtures:

```python
# In your tests
from pathlib import Path

examples_dir = Path("examples")
skill_file = examples_dir / "skills" / "analyze_mcp_server.md"
mcp_server = examples_dir / "mcp_servers" / "sample_vulnerable_server.py"
```

---

## Contributing Examples

To add new examples:

1. Choose the appropriate subdirectory
2. Follow existing naming conventions
3. Include a header comment explaining the example's purpose
4. Document expected analysis results
5. Update this README with the new file
