# BenderBox - AI Model & Infrastructure Security Analysis Platform

**Version:** 2.0.0
**Status:** Production Ready

---

## Overview

**BenderBox** is a modular, stateless, offline-first platform for analyzing AI models AND infrastructure for safety, security, and capabilities. It combines:

### Model Analysis (v1.0)
- 🔍 **GGUF Metadata Extraction** - Architecture, quantization, context length
- 🛡️ **Safety Testing** - Jailbreak resistance, backdoor detection
- 📊 **Risk Assessment** - Automated scoring and categorization
- 🧠 **Capability Fingerprinting** - Reasoning, coding, planning evaluation
- 📝 **Model Cards** - Comprehensive documentation generation

### Infrastructure Analysis (v2.0 NEW!)
- 🔒 **MCP Server Analysis** - Static security analysis of MCP server Python code
- 📋 **Skill Analysis** - Security analysis of Markdown skill files
- 🚨 **Vulnerability Detection** - Command injection, data exfiltration, prompt injection
- ⚖️ **Security Comparison** - Side-by-side comparison of infrastructure components
- 🛡️ **Threat Detection** - Backdoors, obfuscation, credential harvesting

### Core Philosophy

**Skills describe; MCP tools execute; Python performs.**

BenderBox maintains strict architectural separation:
- **Skills** (Markdown) define WHAT to do
- **MCP Tools** (Python wrappers) ROUTE requests
- **Sandbox CLI** (Python engine) PERFORMS analysis

---

## Quick Start

### Prerequisites

1. **Python 3.9+**
2. **llama.cpp** (for dynamic testing)
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make
   # Add to PATH or place in project directory
   ```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd BenderBox

# Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (if using MCP)
pip install mcp
```

### Basic Usage

#### Command Line

**Model Analysis:**
```bash
# Quick metadata check (~5-10s)
python benderbox_sandbox_cli.py --model models/model.gguf --profile quick

# Standard analysis (~10-15s)
python benderbox_sandbox_cli.py --model models/model.gguf --profile standard

# Deep security audit (~60-90s)
python benderbox_sandbox_cli.py --model models/model.gguf --profile deep
```

**Infrastructure Analysis (v2.0):**
```bash
# Analyze MCP server security
python benderbox_sandbox_cli.py --mcp-server server.py --profile infra-standard

# Analyze skill security
python benderbox_sandbox_cli.py --skill custom_skill.md --profile infra-quick

# Deep infrastructure audit
python benderbox_sandbox_cli.py --mcp-server server.py --profile infra-deep
```

**Utility Commands:**
```bash
# View latest report
python benderbox_sandbox_cli.py --summary

# List available tests
python benderbox_sandbox_cli.py --list-tests

# Interactive mode
python benderbox_sandbox_cli.py --interactive
```

#### MCP Server

```bash
# Start MCP server
python benderbox_mcp_server.py
```

Configure in your MCP client:
```json
{
  "mcpServers": {
    "benderbox": {
      "command": "python",
      "args": ["/path/to/benderbox_mcp_server.py"]
    }
  }
}
```

Use MCP tools:
```python
# Analyze model
result = await mcp.call_tool(
    "benderbox_sandbox_analyzeModel",
    {
        "model_path": "./models/model.gguf",
        "profile": "standard"
    }
)

# Get latest report
cached = await mcp.call_tool(
    "benderbox_sandbox_getLatestReport",
    {"model_name": "model.gguf"}
)
```

---

## Architecture

```
┌─────────────────┐
│  Skills (MD)    │  Declarative agent behaviors
└────────┬────────┘
         ↓
┌─────────────────┐
│  MCP Tools (PY) │  Stateless wrappers
└────────┬────────┘
         ↓
┌─────────────────┐
│  Sandbox CLI    │  Core analysis engine
└────────┬────────┘
         ↓
┌─────────────────┐
│  Test Plugins   │  Modular tests
└─────────────────┘
```

### Layers

1. **Skill Layer** (`skills/*.md`) - Markdown definitions for AI agents
2. **MCP Tool Layer** (`benderbox_mcp_server.py`) - Model Context Protocol interface
3. **Sandbox CLI** (`benderbox_sandbox_cli.py`) - Core analysis engine
4. **Test Layer** (`benderbox_dynamic_tests.py`) - Pluggable test modules
5. **Interactive UI** (`benderbox_interactive.py`) - Optional menu interface

**See:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete architectural details.

---

## Features

### Analysis Profiles

#### Model Analysis Profiles
| Profile | Tests | Duration | Use Case |
|---------|-------|----------|----------|
| **quick** | Metadata only | ~5-10s | "What's inside this file?" |
| **standard** | Static tests | ~10-15s | General safety check |
| **deep** | All tests | ~60-90s | Pre-deployment audit |
| **attack** | Security-focused | ~45-60s | Adversarial testing |
| **custom** | User-specified | Varies | Targeted analysis |

#### Infrastructure Analysis Profiles (v2.0)
| Profile | Tests | Duration | Use Case |
|---------|-------|----------|----------|
| **infra-quick** | Fast scan | ~10-20s | "Is this server safe?" |
| **infra-standard** | Standard security | ~30-60s | General security audit |
| **infra-deep** | Comprehensive | ~2-5min | Pre-deployment validation |

### Available Tests

#### Model Tests
**Static Tests:**
- ✅ **GGUF Metadata Sanity** - Validates model format and metadata
- ✅ **Static Metadata Basic** - File size, format, basic checks

**Dynamic Tests (require llama.cpp):**
- 🔒 **Basic Jailbreak Suite** - Tests resistance to 12+ jailbreak techniques
- 🔒 **Backdoor Detection** - Scans for hidden trigger phrases

#### Infrastructure Tests (v2.0)
**MCP Server Tests:**
- 🔒 **MCP Static Analysis** - Command injection, data exfiltration, backdoors
- 🔍 **Dangerous API Detection** - Unsafe system calls, file operations
- 🚨 **Credential Harvesting** - Password/key access patterns

**Skill Tests:**
- 🔒 **Skill Static Analysis** - Prompt injection, credential requests
- 🔍 **Tool Chain Analysis** - Dangerous tool usage sequences
- 🚨 **Obfuscation Detection** - Hidden instructions, base64 encoding

### Skills (AI Agent Capabilities)

#### Model Analysis Skills
- 📊 **Analyze GGUF Model** - Extract metadata and safety characteristics
- 🛡️ **Detect Model Jailbreaks** - Adversarial safety testing
- ⚖️ **Compare Model Safety** - Side-by-side comparison of 2-5 models
- 📝 **Generate Model Card** - Comprehensive documentation

#### Infrastructure Analysis Skills (v2.0)
- 🔒 **Analyze MCP Server** - Security analysis of MCP server Python code
- 📋 **Analyze Skill Security** - Security analysis of Markdown skills
- ⚖️ **Compare Infrastructure Security** - Side-by-side comparison of servers/skills

**See:** [docs/SKILL_GUIDE.md](docs/SKILL_GUIDE.md) for skill authoring guide.

### MCP Tools

- `benderbox_sandbox_analyzeModel` - Run model analysis
- `benderbox_sandbox_getLatestReport` - Retrieve cached reports
- `benderbox_sandbox_listTests` - List available tests

**See:** [docs/MCP_TOOLS.md](docs/MCP_TOOLS.md) for API reference.

---

## Output Schema

All analysis produces JSON reports:

```json
{
  "schema_version": "0.1.0",
  "run_id": "2025-12-10T12:00:00Z_model.gguf",
  "model": {
    "name": "model.gguf",
    "metadata": {
      "architecture": "llama",
      "parameter_count": "7B",
      "quantization": "Q4_K_S",
      "context_length": 4096
    }
  },
  "overall_risk": {
    "level": "LOW",
    "score": 15.0,
    "primary_factors": []
  },
  "safety": {
    "jailbreak_success_rate": 0.08,
    "jailbreak_attempts": 12,
    "jailbreak_successes": 1
  },
  "tests": [...]
}
```

Reports saved to `./sandbox_logs/benderbox_[run_id].json`

---

## Directory Structure

```
BenderBox/
├── benderbox_sandbox_cli.py       # Core analysis engine
├── benderbox_dynamic_tests.py     # Test plugins (jailbreak, backdoor)
├── benderbox_mcp_server.py        # MCP tool server
├── benderbox_interactive.py       # Optional interactive UI
│
├── skills/                      # Markdown skill definitions
│   ├── analyze_gguf_model.md
│   ├── detect_model_jailbreaks.md
│   ├── compare_model_safety.md
│   └── generate_model_card.md
│
├── models/                      # Place GGUF models here
│   └── *.gguf
│
├── sandbox_logs/                # JSON analysis reports
│   └── benderbox_*.json
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # System architecture
│   ├── SKILL_GUIDE.md           # Skill authoring guide
│   └── MCP_TOOLS.md             # MCP API reference
│
├── ARCHITECTURE_REVIEW.md       # Architectural compliance review
└── README.md                    # This file
```

---

## Use Cases

### 1. Model Selection

**Scenario:** Choose between multiple GGUF models for production deployment

```bash
# Analyze candidates
python benderbox_sandbox_cli.py --model model1.gguf --profile standard
python benderbox_sandbox_cli.py --model model2.gguf --profile standard

# Compare via interactive mode
python benderbox_sandbox_cli.py --interactive
# Select option 6: Compare two models
```

**AI Agent (using skills):**
```
User: "Which is safer: modelA.gguf or modelB.gguf?"
Agent: [Uses compare_model_safety skill]
       [Shows side-by-side comparison with recommendation]
```

---

### 2. Pre-Deployment Audit

**Scenario:** Validate model safety before production deployment

```bash
# Run comprehensive audit
python benderbox_sandbox_cli.py --model production-model.gguf --profile deep --format both

# Review results
# - Risk score must be < 25 (LOW)
# - Jailbreak rate must be < 10%
# - No backdoor triggers detected
```

**Decision Matrix:**
- Risk LOW + Jailbreak < 5% = ✅ Deploy with standard monitoring
- Risk LOW + Jailbreak 5-15% = ✅ Deploy with enhanced monitoring
- Risk MEDIUM = ⚠️ Require additional safeguards
- Risk HIGH/CRITICAL = ❌ Do not deploy

---

### 3. Security Research

**Scenario:** Test new jailbreak techniques

```bash
# Run attack-focused profile
python benderbox_sandbox_cli.py --model research-model.gguf --profile attack

# Or custom tests
python benderbox_sandbox_cli.py \
  --model research-model.gguf \
  --profile custom \
  --tests "basic_jailbreak_suite,backdoor_detection"
```

**Review violations:**
```bash
# View detailed report
cat sandbox_logs/benderbox_*.json | jq '.tests[] | select(.category=="dynamic")'
```

---

### 4. Model Documentation

**Scenario:** Generate standardized model card for team distribution

**AI Agent (using skills):**
```
User: "Create a model card for production-model.gguf"
Agent: [Uses generate_model_card skill]
       [Runs deep analysis if not cached]
       [Generates comprehensive Markdown model card]
       [Offers to save to models/production-model_card.md]
```

---

### 5. Continuous Monitoring

**Scenario:** Regular safety audits of deployed models

```bash
#!/bin/bash
# weekly-audit.sh

for model in models/*.gguf; do
  echo "Auditing $model..."
  python benderbox_sandbox_cli.py \
    --model "$model" \
    --profile standard \
    --format json
done

# Alert on risk level changes
python scripts/compare_with_baseline.py
```

---

## Extending BenderBox

### Add a New Test

1. **Create test class** in `benderbox_dynamic_tests.py`:
   ```python
   class MyTest(SandboxTest):
       name = "my_test"
       category = "security"

       def run(self, context):
           # Implementation
           return TestResult(
               name=self.name,
               category=self.category,
               status="PASS",
               severity="INFO",
               details="Test details..."
           )
   ```

2. **Register** in `benderbox_sandbox_cli.py`:
   ```python
   from benderbox_dynamic_tests import MyTest
   TEST_REGISTRY["my_test"] = MyTest
   PROFILE_TESTS["deep"].append("my_test")
   ```

### Add a New Skill

Create `skills/my_skill.md`:
```markdown
# Skill: My Skill Name

## Purpose
[What this skill does]

## Triggers
- User asks: "[trigger phrase]"

## Required Tools
- `benderbox_sandbox_[toolName]`

## Process Flow
[Steps]

## Output Format
[Template]
```

**See:** [docs/SKILL_GUIDE.md](docs/SKILL_GUIDE.md) for complete guide.

### Add a New Profile

In `benderbox_sandbox_cli.py`:
```python
PROFILE_TESTS["my_profile"] = [
    "test1",
    "test2",
    "test3"
]
```

---

## Performance

### Optimization Tips

1. **Cache reports** - Reports valid for 24 hours
2. **Use quick profile** for exploratory analysis
3. **Parallel analysis** - Analyze multiple models concurrently
4. **Increase timeout** for large models (70B+)

### Benchmarks

Tested on: Intel i7-12700K, 32GB RAM, llama.cpp CPU-only

| Model Size | Profile | Duration |
|------------|---------|----------|
| 7B Q4 | quick | 6s |
| 7B Q4 | standard | 12s |
| 7B Q4 | deep | 68s |
| 13B Q4 | quick | 8s |
| 13B Q4 | standard | 15s |
| 13B Q4 | deep | 94s |
| 70B Q4 | quick | 22s |
| 70B Q4 | standard | 48s |

---

## Troubleshooting

### "llama-cli not found"

**Solution:**
```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Add to PATH or place in BenderBox directory
cp build/bin/llama-cli /path/to/BenderBox/
```

### "Dynamic tests not available"

**Cause:** `benderbox_dynamic_tests.py` failed to import

**Solution:**
```bash
# Verify file exists
ls benderbox_dynamic_tests.py

# Test import
python -c "from benderbox_dynamic_tests import BasicJailbreakTest"
```

### Analysis timeout

**Solution:**
```bash
# Increase timeout
python benderbox_sandbox_cli.py \
  --model large-model.gguf \
  --profile deep \
  --format json \
  | jq -r '.overall_risk'  # Parse with timeout=600
```

Or in code:
```python
result = invoke_sandbox_cli(
    model_path=path,
    profile="deep",
    timeout=600  # 10 minutes
)
```

### MCP server won't start

**Cause:** MCP SDK not installed

**Solution:**
```bash
pip install mcp
```

---

## Security Considerations

### What BenderBox Tests For

✅ **Jailbreak Resistance** - Role-play, context-switch, injection techniques
✅ **Backdoor Triggers** - Hidden activation phrases
✅ **Policy Compliance** - Refusal of harmful requests
✅ **Metadata Validation** - GGUF format integrity

### What BenderBox Does NOT Test

❌ **Training Data Leakage** - Memorization of sensitive data
❌ **Bias Detection** - Fairness and representation (planned)
❌ **Model Watermarking** - Attribution verification
❌ **Runtime Exploits** - Memory safety, code execution

### Limitations

- **Jailbreaks evolve** - New techniques emerge constantly
- **Static analysis only** - Some tests require runtime inference
- **Offline only** - No cloud-based threat intelligence
- **GGUF-focused** - Other formats not currently supported

### Best Practices

1. **Never rely solely on BenderBox** - Use as one layer of defense
2. **Regular re-testing** - Re-audit monthly or after model updates
3. **Monitor production** - Log and review actual usage patterns
4. **Defense in depth** - Combine with content moderation, rate limiting, human review

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repo-url>
cd BenderBox

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt  # (when available)

# Run tests
python -m pytest tests/  # (when test suite exists)
```

### Contribution Guidelines

1. **Maintain architectural separation** - Skills describe, MCP routes, Python performs
2. **Add tests** for new functionality
3. **Document thoroughly** - Update relevant docs/
4. **Version schemas** - Increment schema_version for breaking changes
5. **No network calls** - Keep offline-first design

### Planned Features

- [ ] Bias detection tests
- [ ] Training data attribution
- [ ] Model card validation
- [ ] Capability benchmarking (reasoning, coding)
- [ ] Report comparison (drift detection)
- [ ] HTML/PDF report export
- [ ] Plugin system for external tests
- [ ] Distributed execution

---

## License

[Specify License - e.g., Apache 2.0, MIT]

---

## Citation

If you use BenderBox in research, please cite:

```bibtex
@software{benderbox2025,
  title={BenderBox: AI Model Security Analysis Platform},
  author={[Your Name/Organization]},
  year={2025},
  url={[Repository URL]}
}
```

---

## Acknowledgments

- **llama.cpp** - GGUF format support and inference engine
- **Model Context Protocol** - Standardized AI tool interface
- **Community** - Adversarial testing methodologies

---

## Support

- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues]
- **Discussions:** [GitHub Discussions]
- **Contact:** [Your Contact]

---

## Changelog

### v2.0.0 (2025-12-10)

**Infrastructure Analysis Extension**

**New Features:**
- 🔒 **MCP Server Analysis** - Static security analysis of MCP server Python code
  - Command injection detection via AST parsing
  - Data exfiltration pattern detection
  - Credential harvesting detection
  - Backdoor and obfuscation detection
  - Dangerous API usage tracking
- 📋 **Skill Analysis** - Security analysis of Markdown skill files
  - Prompt injection pattern detection
  - Credential request detection
  - Malicious tool chain detection
  - Obfuscation detection (base64, hidden text)
- 🧪 **Infrastructure Tests** - New test plugins for infrastructure analysis
  - `mcp_static_analysis` - MCP server security test
  - `skill_static_analysis` - Skill security test
  - `infrastructure_comprehensive` - Combined infrastructure tests
- 📊 **Infrastructure Profiles** - New analysis profiles
  - `infra-quick` - Fast infrastructure scan (~10-20s)
  - `infra-standard` - Standard security audit (~30-60s)
  - `infra-deep` - Comprehensive analysis (~2-5min)
- 🎯 **Infrastructure Skills** - New AI agent capabilities
  - `analyze_mcp_server.md` - MCP server security analysis skill
  - `analyze_skill_security.md` - Skill security analysis skill
  - `compare_infrastructure_security.md` - Infrastructure comparison skill

**Technical:**
- Added `benderbox_mcp_analyzer.py` - Static MCP server analyzer (AST-based)
- Added `benderbox_skill_analyzer.py` - Static skill analyzer (regex-based)
- Added `benderbox_infrastructure_tests.py` - Infrastructure test plugins
- Extended `benderbox_sandbox_cli.py` with `--mcp-server` and `--skill` flags
- Schema version bump: 0.2.0 (extended schema for infrastructure)
- Runner version: 2.0.0

**Documentation:**
- Updated README with v2.0 features
- Created `INFRASTRUCTURE_ANALYSIS_DESIGN.md` - Complete design document
- Added infrastructure analysis examples

**Breaking Changes:**
- None - v1.0 model analysis remains fully compatible

---

### v1.0.0 (2025-12-10)

**Initial Release**

**Features:**
- GGUF metadata extraction via llama.cpp
- Static and dynamic testing framework
- Jailbreak resistance testing (12+ techniques)
- Backdoor detection testing (15+ triggers)
- Risk scoring and categorization
- MCP tool layer (3 tools)
- Skill layer (4 skills)
- Interactive menu UI
- JSON structured reporting
- Comprehensive documentation

**Profiles:**
- quick, standard, deep, attack, custom

**Tests:**
- Static: GGUF metadata sanity, basic metadata
- Dynamic: Basic jailbreak suite, backdoor detection

**Documentation:**
- Architecture guide
- Skill authoring guide
- MCP tools API reference
- Architectural compliance review

---

*Built with security in mind. Analyzed with BenderBox.*
