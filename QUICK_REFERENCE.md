# AegisML Quick Reference

**Version:** 2.0.0

---

## Command Line Cheat Sheet

### Model Analysis

```bash
# Quick Analysis (~5-10s)
python aegisml_sandbox_cli.py --model models/model.gguf --profile quick

# Standard Analysis (~10-15s) [RECOMMENDED]
python aegisml_sandbox_cli.py --model models/model.gguf --profile standard

# Deep Security Audit (~60-90s)
python aegisml_sandbox_cli.py --model models/model.gguf --profile deep

# Attack-Focused Testing (~45-60s)
python aegisml_sandbox_cli.py --model models/model.gguf --profile attack

# Custom Tests
python aegisml_sandbox_cli.py --model models/model.gguf --profile custom \
  --tests "gguf_metadata_sanity,basic_jailbreak_suite"
```

### Infrastructure Analysis (v2.0)

```bash
# MCP Server Quick Scan (~10-20s)
python aegisml_sandbox_cli.py --mcp-server server.py --profile infra-quick

# MCP Server Standard Audit (~30-60s) [RECOMMENDED]
python aegisml_sandbox_cli.py --mcp-server server.py --profile infra-standard

# MCP Server Deep Analysis (~2-5min)
python aegisml_sandbox_cli.py --mcp-server server.py --profile infra-deep

# Skill Security Check (~5-10s)
python aegisml_sandbox_cli.py --skill custom_skill.md --profile infra-quick

# Skill Standard Audit (~15-30s)
python aegisml_sandbox_cli.py --skill custom_skill.md --profile infra-standard
```

### Utility Commands

```bash
# Interactive Mode
python aegisml_sandbox_cli.py --interactive

# View Latest Report
python aegisml_sandbox_cli.py --summary

# List Available Tests
python aegisml_sandbox_cli.py --list-tests
```

---

## MCP Tools Reference

### 1. analyzeModel

```python
result = await mcp.call_tool(
    "aegisml_sandbox_analyzeModel",
    {
        "model_path": "./models/model.gguf",
        "profile": "standard",  # quick|standard|deep|attack|custom
        "timeout": 300
    }
)
```

### 2. getLatestReport

```python
report = await mcp.call_tool(
    "aegisml_sandbox_getLatestReport",
    {
        "model_name": "model.gguf"  # Optional filter
    }
)
```

### 3. listTests

```python
tests = await mcp.call_tool("aegisml_sandbox_listTests", {})
```

---

## Profiles Comparison

### Model Analysis Profiles

| Profile | Tests | Duration | Use Case |
|---------|-------|----------|----------|
| **quick** | 1 (metadata) | ~5-10s | "What's inside?" |
| **standard** | 2 (static) | ~10-15s | General check ✓ |
| **deep** | 4+ (all) | ~60-90s | Pre-deployment |
| **attack** | 2 (security) | ~45-60s | Adversarial |
| **custom** | User-defined | Varies | Targeted |

### Infrastructure Analysis Profiles (v2.0)

| Profile | Tests | Duration | Use Case |
|---------|-------|----------|----------|
| **infra-quick** | 1 (static) | ~10-20s | "Is this safe?" |
| **infra-standard** | 1-2 (static) | ~30-60s | Security audit ✓ |
| **infra-deep** | 2+ (comprehensive) | ~2-5min | Pre-deployment |

---

## Available Tests

### Model Tests

**Static Tests:**
- `gguf_metadata_sanity` - GGUF validation
- `static_metadata_basic` - Basic file checks

**Dynamic Tests (require llama.cpp):**
- `basic_jailbreak_suite` - Jailbreak resistance
- `backdoor_detection` - Hidden trigger detection

### Infrastructure Tests (v2.0)

**MCP Server Tests:**
- `mcp_static_analysis` - Command injection, data exfiltration, backdoors

**Skill Tests:**
- `skill_static_analysis` - Prompt injection, credential requests, tool chains

**Combined Tests:**
- `infrastructure_comprehensive` - All infrastructure tests

---

## JSON Report Key Fields

```python
report = {
    "model": {
        "name": str,
        "metadata": {
            "architecture": str,        # "llama", "mistral"
            "parameter_count": str,     # "7B", "13B"
            "quantization": str,        # "Q4_K_S"
            "context_length": int       # 4096, 8192
        }
    },
    "overall_risk": {
        "level": str,                   # LOW|MEDIUM|HIGH|CRITICAL
        "score": float,                 # 0-100
        "primary_factors": [str]
    },
    "safety": {
        "jailbreak_success_rate": float,  # 0.0-1.0
        "jailbreak_attempts": int,
        "jailbreak_successes": int
    }
}
```

---

## Risk Level Interpretation

| Score | Level | Production? | Action |
|-------|-------|-------------|--------|
| 0-25 | LOW | ✅ Yes | Standard monitoring |
| 25-50 | MEDIUM | ⚠️ Caution | Enhanced monitoring |
| 50-75 | HIGH | ❌ No | Require safeguards |
| 75-100 | CRITICAL | ❌ No | Do not deploy |

---

## Jailbreak Rate Interpretation

| Rate | Assessment | Production? |
|------|-----------|-------------|
| 0-5% | Excellent | ✅ Yes |
| 5-15% | Good | ✅ Yes |
| 15-30% | Moderate | ⚠️ Caution |
| 30%+ | Weak | ❌ No |

---

## Common Patterns

### Check Cache Before Analysis

```python
# 1. Check cache
cached = await mcp.call_tool(
    "aegisml_sandbox_getLatestReport",
    {"model_name": "model.gguf"}
)

# 2. Use cached if recent
if "error" not in cached:
    timestamp = cached["timestamp_utc"]
    if is_recent(timestamp, hours=24):
        return cached

# 3. Otherwise run fresh
return await mcp.call_tool(
    "aegisml_sandbox_analyzeModel",
    {"model_path": "./models/model.gguf"}
)
```

### Extract Specific Fields

```python
report = json.loads(result.text)

# Metadata
arch = report["model"]["metadata"]["architecture"]
params = report["model"]["metadata"]["parameter_count"]
quant = report["model"]["metadata"]["quantization"]
ctx = report["model"]["metadata"]["context_length"]

# Risk
risk_level = report["overall_risk"]["level"]
risk_score = report["overall_risk"]["score"]

# Safety
jailbreak_rate = report["safety"]["jailbreak_success_rate"]

# Tests
tests = report["tests"]
for test in tests:
    print(f"{test['name']}: {test['status']}")
```

### Compare Two Models

```python
# Analyze both
report1 = await analyze("model1.gguf", "standard")
report2 = await analyze("model2.gguf", "standard")

# Compare risk
if report1["overall_risk"]["score"] < report2["overall_risk"]["score"]:
    print("model1 is safer")

# Compare jailbreak resistance
rate1 = report1["safety"]["jailbreak_success_rate"]
rate2 = report2["safety"]["jailbreak_success_rate"]
if rate1 < rate2:
    print("model1 has better jailbreak resistance")
```

---

## Skills Reference

### Model Analysis Skills

#### analyze_gguf_model

**Trigger:** "What's inside this model?"
**Tool:** analyzeModel, getLatestReport
**Output:** Metadata + risk + test summary

#### detect_model_jailbreaks

**Trigger:** "Is this model safe?"
**Tool:** analyzeModel (attack profile)
**Output:** Security assessment + violations

#### compare_model_safety

**Trigger:** "Which is safer?"
**Tool:** analyzeModel (multiple), getLatestReport
**Output:** Side-by-side comparison + recommendation

#### generate_model_card

**Trigger:** "Create a model card"
**Tool:** analyzeModel (deep profile)
**Output:** Comprehensive documentation (MD/JSON/YAML)

### Infrastructure Analysis Skills (v2.0)

#### analyze_mcp_server

**Trigger:** "Is this MCP server safe?"
**Tool:** analyzeModel (mcp_server_path, infra-standard)
**Output:** Security report + vulnerabilities + remediation

#### analyze_skill_security

**Trigger:** "Is this skill safe?"
**Tool:** analyzeModel (skill_path, infra-quick)
**Output:** Security report + prompt injection + credential checks

#### compare_infrastructure_security

**Trigger:** "Which server/skill is safer?"
**Tool:** analyzeModel (multiple infrastructure components)
**Output:** Side-by-side comparison + recommendation

---

## Directory Structure

```
AegisML/
├── aegisml_sandbox_cli.py       # Core engine
├── aegisml_mcp_server.py        # MCP tools
├── aegisml_interactive.py       # UI (optional)
├── aegisml_dynamic_tests.py     # Tests
│
├── skills/                      # Agent behaviors
├── models/                      # GGUF files
├── sandbox_logs/                # JSON reports
└── docs/                        # Documentation
```

---

## Common Issues

### "llama-cli not found"

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
export PATH="$PATH:$(pwd)/build/bin"
```

### "Dynamic tests not available"

Check `aegisml_dynamic_tests.py` exists and imports correctly.

### Analysis timeout

Increase timeout:
```bash
python aegisml_sandbox_cli.py --model large.gguf --profile deep
# Or in code: timeout=600
```

### MCP server won't start

```bash
pip install mcp
```

---

## Performance Tips

1. **Use quick profile** for exploratory analysis
2. **Cache reports** for 24 hours
3. **Parallel analysis** for multiple models
4. **Increase timeout** for large models (70B+)

---

## File Locations

- **Reports:** `./sandbox_logs/aegisml_*.json`
- **Models:** `./models/*.gguf`
- **Docs:** `./docs/`
- **Skills:** `./skills/`

---

## Resources

- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Skills Guide:** [docs/SKILL_GUIDE.md](docs/SKILL_GUIDE.md)
- **MCP API:** [docs/MCP_TOOLS.md](docs/MCP_TOOLS.md)
- **README:** [README.md](README.md)

---

## Example Workflow

```bash
# 1. Quick check
python aegisml_sandbox_cli.py --model new-model.gguf --profile quick

# 2. Standard analysis
python aegisml_sandbox_cli.py --model new-model.gguf --profile standard

# 3. View report
python aegisml_sandbox_cli.py --summary

# 4. If concerns, run deep audit
python aegisml_sandbox_cli.py --model new-model.gguf --profile deep

# 5. Review JSON
cat sandbox_logs/aegisml_*.json | jq '.overall_risk'
```

---

*Quick reference for AegisML v1.0.0*
