# BenderBox - AI Security Interrogation & Testing Toolkit

**Version:** 3.0.0-alpha | **Status:** Active Development

BenderBox is a security testing toolkit for interrogating AI models, probing MCP servers, and analyzing instruction files for vulnerabilities, jailbreak resistance, and safety compliance.

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/r33n3/BenderBox.git
cd BenderBox
python setup_wizard.py

# 2. Start interactive mode
python bb.py

# 3. Try a quick analysis
BenderBox> list models
BenderBox> analyze phi-2 --profile quick

# 4. View reports
BenderBox> reports
BenderBox> open reports
```

---

## Installation

### Prerequisites
- **Python 3.9+** (3.11 or 3.12 recommended)
- **Git**

### Setup Options

```bash
# Recommended: Interactive wizard
python setup_wizard.py

# Manual: Core features only
pip install -e .

# Manual: With local NLP chat
pip install -e ".[nlp]"

# Manual: Everything
pip install -e ".[all]"
```

### Windows NLP Support (Optional)

Local LLM chat requires C++ build tools:
```powershell
curl -L -o vs_buildtools.exe "https://aka.ms/vs/17/release/vs_buildtools.exe"
.\vs_buildtools.exe --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait
# Open NEW terminal, then:
pip install llama-cpp-python
```

**Note:** Build tools are NOT required for MCP analysis, context analysis, or API model testing.

---

## Core Features

| Feature | Description | API Key? |
|---------|-------------|----------|
| **Model Interrogation** | Security test local GGUF or cloud AI models | Cloud only |
| **MCP Security Testing** | Probe MCP servers for vulnerabilities | No |
| **Context Analysis** | Scan instruction files for dangerous patterns | No |
| **Report Generation** | HTML/JSON/SARIF reports with findings | No |

---

## Quick Examples

### Test an MCP Server
```bash
python bb.py mcp tools "npx @modelcontextprotocol/server-filesystem ."
python bb.py mcp interrogate "npx @modelcontextprotocol/server-filesystem ." --profile quick
```

### Analyze Instruction Files
```bash
python bb.py context analyze prompt.md
python bb.py context scan ./prompts --recursive
```

### Test a Local Model
```bash
python bb.py models download tinyllama
python bb.py analyze tinyllama --profile quick
```

### Test Cloud Models
```bash
python bb.py config set-key openai
python bb.py analyze openai:gpt-4-turbo --profile quick
```

### Interactive Mode (Natural Language)
```bash
python bb.py

BenderBox> what models do I have?
BenderBox> analyze phi-2 and generate a report
BenderBox> test the mcp server for vulnerabilities
BenderBox> help
```

---

## Command Reference

| Category | Commands |
|----------|----------|
| **Models** | `models list`, `models download <id>`, `analyze <model>` |
| **MCP** | `mcp tools <target>`, `mcp interrogate <target>`, `mcp analyze <source>` |
| **Context** | `context analyze <file>`, `context scan <dir>` |
| **Reports** | `report`, `report list`, `export --format html` |
| **Config** | `config set-key <provider>`, `config api-keys` |

Use `python bb.py --help` for full command list or `help` in interactive mode.

---

## Analysis Profiles

| Profile | Tests | Use Case |
|---------|-------|----------|
| `quick` | ~15 | Fast CI/CD validation |
| `standard` | ~50 | Balanced coverage |
| `full` | ~100+ | Pre-deployment audit |
| `adversarial` | ~64 | Jailbreak resistance |

---

## Risk Levels

| Level | Score | Action |
|-------|-------|--------|
| **SAFE** | 0 | Clear to deploy |
| **LOW** | 1-20 | Minor review |
| **MEDIUM** | 21-40 | Review recommended |
| **HIGH** | 41-60 | Address before deploy |
| **CRITICAL** | 61-100 | Do not deploy |

---

## Documentation

- **Interactive Help:** Type `help` in BenderBox for detailed guides
- **Detailed Guide:** Open `docs/guide.html` for comprehensive documentation
- **Examples:** See `examples/` directory for sample files

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "NLP features limited" | Install `pip install llama-cpp-python` (optional) |
| "API key not configured" | Run `python bb.py config set-key openai` |
| "CMake Error" (Windows) | Install VS Build Tools (see Installation) |
| MCP connection fails | Test server directly: `npx @server` |

---

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE)

---

## Acknowledgments

- **llama.cpp** - Local GGUF inference
- **OpenAI, Anthropic, Google, xAI** - API providers
- **Model Context Protocol** - MCP specification
- **Rich** - Terminal UI

---

*Analyze AI models with confidence. Built with security in mind.*
