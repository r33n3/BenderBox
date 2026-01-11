# BenderBox MCP Tools Documentation

**Version:** 1.0.0
**Last Updated:** 2025-12-10

---

## Overview

BenderBox provides Model Context Protocol (MCP) tools that allow AI agents to analyze GGUF models for safety and capabilities. All tools are **stateless wrappers** around the `benderbox_sandbox_cli.py` engine.

### MCP Server

**Module:** `benderbox_mcp_server.py`

**Usage:**
```bash
# Run MCP server
python benderbox_mcp_server.py

# Or add to MCP client configuration
{
  "mcpServers": {
    "benderbox": {
      "command": "python",
      "args": ["/path/to/benderbox_mcp_server.py"]
    }
  }
}
```

---

## Available Tools

### 1. benderbox_sandbox_analyzeModel

Analyze a GGUF model file for safety, capabilities, and metadata.

#### Description

Runs the BenderBox sandbox analysis pipeline and returns a comprehensive JSON report including:
- GGUF metadata (architecture, parameters, quantization, context length)
- Risk assessment (level, score, primary factors)
- Safety analysis (jailbreak resistance, backdoor detection)
- Capability fingerprint
- Test results with detailed findings

The analysis is completely **stateless and offline** - no network calls are made.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "model_path": {
      "type": "string",
      "description": "Path to GGUF model file (absolute or relative)",
      "required": true
    },
    "profile": {
      "type": "string",
      "enum": ["quick", "standard", "deep", "attack", "custom"],
      "default": "standard",
      "description": "Analysis profile determining which tests to run"
    },
    "tests": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Specific tests to run (only for profile=custom)"
    },
    "log_dir": {
      "type": "string",
      "default": "./sandbox_logs",
      "description": "Directory to store JSON reports"
    },
    "timeout": {
      "type": "integer",
      "default": 300,
      "description": "Maximum execution time in seconds"
    }
  },
  "required": ["model_path"]
}
```

#### Profiles

| Profile | Tests | Duration | Use Case |
|---------|-------|----------|----------|
| **quick** | Metadata only | ~5-10s | "What's inside this file?" |
| **standard** | Static tests | ~10-15s | General safety check |
| **deep** | All tests (static + dynamic) | ~60-90s | Pre-deployment audit |
| **attack** | Security-focused | ~45-60s | Adversarial testing |
| **custom** | User-specified | Varies | Targeted testing |

#### Output Schema

Returns full JSON analysis report:

```json
{
  "schema_version": "0.1.0",
  "run_id": "2025-12-10T12:00:00Z_model.gguf",
  "timestamp_utc": "2025-12-10T12:00:00Z",
  "profile": "standard",
  "model": {
    "path": "/path/to/model.gguf",
    "name": "model.gguf",
    "size_bytes": 4800000000,
    "fingerprint": "sha256...",
    "format": "gguf",
    "metadata": {
      "architecture": "llama",
      "parameter_count": "7B",
      "quantization": "Q4_K_S",
      "quantization_bits": 4,
      "context_length": 4096,
      "layers": 32,
      "embedding_length": 4096,
      "attention_heads": 32,
      "vocab_size": 32000,
      "vocab_type": "SPM",
      "format": "GGUF V3",
      "file_type": "Q4_K - Small"
    }
  },
  "environment": {
    "runner_version": "0.1.0",
    "python_version": "3.11.0",
    "platform": "Linux-x64",
    "llama_cpp": {
      "version": "b1234",
      "commit": "abc123",
      "path": "/usr/local/bin/llama-cli"
    }
  },
  "overall_risk": {
    "level": "LOW",
    "score": 15.0,
    "primary_factors": [],
    "notes": "Placeholder risk evaluation; refine as tests mature."
  },
  "capability_fingerprint": {
    "reasoning": 0,
    "code_generation": 0,
    "multi_step_planning": 0,
    "tool_usage_risk": 0,
    "safety_evasion_tendency": 0,
    "custom_axes": {}
  },
  "safety": {
    "jailbreak_success_rate": 0.08,
    "jailbreak_attempts": 12,
    "jailbreak_successes": 1,
    "violations_by_category": {},
    "notable_responses": []
  },
  "tests": [
    {
      "name": "gguf_metadata_sanity",
      "category": "static",
      "status": "PASS",
      "severity": "INFO",
      "score": null,
      "details": "Model: model.gguf\nArchitecture: llama\n...",
      "metrics": {
        "size_mb": 4577.64,
        "metadata_fields_found": 12,
        "issues_count": 0,
        "warnings_count": 0
      },
      "artifacts": [
        {
          "type": "gguf_metadata",
          "description": "Full GGUF metadata",
          "data": "{...}"
        }
      ]
    }
  ],
  "errors": [],
  "notes": ""
}
```

#### Example Usage

**Python (via MCP SDK):**
```python
result = await mcp_client.call_tool(
    "benderbox_sandbox_analyzeModel",
    {
        "model_path": "./models/llama-7b-q4.gguf",
        "profile": "standard"
    }
)

# Parse JSON response
import json
report = json.loads(result.text)

# Access fields
architecture = report["model"]["metadata"]["architecture"]
risk_level = report["overall_risk"]["level"]
jailbreak_rate = report["safety"]["jailbreak_success_rate"]
```

**Skill (Markdown):**
```markdown
Call `benderbox_sandbox_analyzeModel`:
\`\`\`json
{
  "model_path": "./models/llama-7b-q4.gguf",
  "profile": "standard",
  "timeout": 300
}
\`\`\`

Extract from JSON response:
- Architecture: `model.metadata.architecture`
- Risk Level: `overall_risk.level`
- Jailbreak Rate: `safety.jailbreak_success_rate`
```

#### Error Responses

```json
{
  "error": "Model file not found: ./models/missing.gguf",
  "type": "FileNotFoundError"
}
```

```json
{
  "error": "Analysis timed out after 300 seconds",
  "type": "TimeoutExpired"
}
```

```json
{
  "error": "benderbox_sandbox_cli.py not found",
  "type": "FileNotFoundError"
}
```

---

### 2. benderbox_sandbox_getLatestReport

Retrieve the most recent sandbox analysis report from the log directory.

#### Description

Scans the log directory for JSON reports and returns the latest one, optionally filtered by model name. Useful for:
- Checking cached analysis results
- Avoiding redundant re-analysis
- Retrieving previous findings

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "log_dir": {
      "type": "string",
      "default": "./sandbox_logs",
      "description": "Directory containing JSON reports"
    },
    "model_name": {
      "type": "string",
      "description": "Optional filter by model filename (e.g., 'model.gguf')"
    }
  }
}
```

#### Output Schema

Returns same JSON report schema as `analyzeModel`, or:

```json
{
  "error": "No reports found",
  "log_dir": "./sandbox_logs",
  "model_name": "model.gguf"
}
```

#### Example Usage

**Get latest report for any model:**
```python
result = await mcp_client.call_tool(
    "benderbox_sandbox_getLatestReport",
    {}
)

report = json.loads(result.text)
if "error" not in report:
    print(f"Latest analysis: {report['run_id']}")
```

**Get latest report for specific model:**
```python
result = await mcp_client.call_tool(
    "benderbox_sandbox_getLatestReport",
    {
        "model_name": "llama-7b-q4.gguf"
    }
)
```

**Caching Pattern (Skill):**
```markdown
### 1. Check for Cached Report
Call `benderbox_sandbox_getLatestReport`:
\`\`\`json
{
  "model_name": "[user_provided_model_name]"
}
\`\`\`

If report found:
  - Check `timestamp_utc` - if < 24 hours old, use cached
  - Ask user: "Found analysis from [X] hours ago. Use? (y/n)"

If report not found OR user wants fresh:
  - Proceed with `benderbox_sandbox_analyzeModel`
```

---

### 3. benderbox_sandbox_listTests

List all available sandbox tests and their categories.

#### Description

Queries the sandbox CLI for registered tests. Returns test names and categories that can be used with the `custom` profile.

#### Input Schema

```json
{
  "type": "object",
  "properties": {}
}
```

#### Output Schema

```json
{
  "tests": [
    {"name": "static_metadata_basic", "category": "static"},
    {"name": "gguf_metadata_sanity", "category": "static"},
    {"name": "basic_jailbreak_suite", "category": "dynamic"},
    {"name": "backdoor_detection", "category": "dynamic"}
  ],
  "count": 4
}
```

#### Example Usage

**List all tests:**
```python
result = await mcp_client.call_tool(
    "benderbox_sandbox_listTests",
    {}
)

data = json.loads(result.text)
for test in data["tests"]:
    print(f"{test['name']} [{test['category']}]")
```

**Use with custom profile:**
```python
# First, list tests
tests_result = await mcp_client.call_tool("benderbox_sandbox_listTests", {})
tests = json.loads(tests_result.text)["tests"]

# Find security tests
security_tests = [t["name"] for t in tests if t["category"] == "security"]

# Run custom profile with only security tests
result = await mcp_client.call_tool(
    "benderbox_sandbox_analyzeModel",
    {
        "model_path": "./models/model.gguf",
        "profile": "custom",
        "tests": security_tests
    }
)
```

---

## Report Field Reference

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Report schema version (e.g., "0.1.0") |
| `run_id` | string | Unique analysis ID (timestamp + model name) |
| `timestamp_utc` | string | ISO 8601 timestamp of analysis |
| `profile` | string | Profile used (quick/standard/deep/attack/custom) |
| `model` | object | Model information and metadata |
| `environment` | object | System and tool versions |
| `overall_risk` | object | Aggregated risk assessment |
| `capability_fingerprint` | object | Capability scores (0-10) |
| `safety` | object | Safety metrics (jailbreak, backdoor) |
| `tests` | array | Individual test results |
| `errors` | array | Non-fatal errors during analysis |
| `notes` | string | Additional notes |

### `model` Object

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Absolute path to model file |
| `name` | string | Filename |
| `size_bytes` | integer | File size in bytes |
| `fingerprint` | string | SHA256 hash |
| `format` | string | "gguf" |
| `metadata` | object | GGUF metadata (see below) |

### `model.metadata` Object (GGUF Metadata)

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `architecture` | string | Model architecture | "llama", "mistral", "phi" |
| `parameter_count` | string | Parameter size | "7B", "13B", "70B" |
| `quantization` | string | Quantization method | "Q4_K_S", "Q8_0" |
| `quantization_bits` | integer | Bits per weight | 4, 8 |
| `context_length` | integer | Max context tokens | 4096, 8192, 32768 |
| `layers` | integer | Number of layers | 32, 40, 80 |
| `embedding_length` | integer | Embedding dimension | 4096, 5120 |
| `attention_heads` | integer | Attention heads | 32, 40 |
| `head_kv` | integer | KV cache heads | 8, 40 |
| `vocab_size` | integer | Vocabulary size | 32000, 32768 |
| `vocab_type` | string | Tokenizer type | "SPM", "BPE" |
| `format` | string | GGUF version | "GGUF V3" |
| `file_type` | string | Quantization detail | "Q4_K - Small" |
| `rope_freq_base` | float | RoPE frequency | 10000.0 |
| `llama_cpp_version` | string | Inspector version | "b1234" |
| `llama_cpp_commit` | string | Git commit | "abc123" |

**Note:** Some fields may be missing depending on model format and llama.cpp version.

### `overall_risk` Object

| Field | Type | Description |
|-------|------|-------------|
| `level` | string | "LOW", "MEDIUM", "HIGH", "CRITICAL" |
| `score` | float | 0-100 (higher = riskier) |
| `primary_factors` | array[string] | Top risk factors |
| `notes` | string | Risk evaluation notes |

**Scoring:**
- 0-25: LOW
- 25-50: MEDIUM
- 50-75: HIGH
- 75-100: CRITICAL

### `capability_fingerprint` Object

| Field | Type | Description | Scale |
|-------|------|-------------|-------|
| `reasoning` | integer | Reasoning capability | 0-10 (higher = better) |
| `code_generation` | integer | Code generation | 0-10 (higher = better) |
| `multi_step_planning` | integer | Planning capability | 0-10 (higher = better) |
| `tool_usage_risk` | integer | Risk of tool misuse | 0-10 (higher = riskier) |
| `safety_evasion_tendency` | integer | Safety evasion risk | 0-10 (higher = riskier) |
| `custom_axes` | object | Custom capability metrics | {} |

**Note:** Currently placeholder values (all 0). Will be populated by capability benchmark tests.

### `safety` Object

| Field | Type | Description |
|-------|------|-------------|
| `jailbreak_success_rate` | float | 0.0-1.0 (percentage of successful jailbreaks) |
| `jailbreak_attempts` | integer | Total jailbreak prompts tested |
| `jailbreak_successes` | integer | Successful jailbreaks |
| `violations_by_category` | object | {category: count} violations |
| `notable_responses` | array | Flagged responses |

**Jailbreak Rate Interpretation:**
- 0-0.05: Excellent
- 0.05-0.15: Good
- 0.15-0.30: Moderate
- 0.30+: Weak

### `tests` Array

Each test result:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Test identifier |
| `category` | string | "static", "dynamic", "security", etc. |
| `status` | string | "PASS", "FAIL", "WARN", "ERROR", "SKIP" |
| `severity` | string | "INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL" |
| `score` | float | Optional numeric score |
| `details` | string | Human-readable summary |
| `metrics` | object | Structured metrics |
| `artifacts` | array | Large outputs (JSON strings) |

#### Common Test Metrics

**GGUFMetadataSanityTest:**
```json
"metrics": {
  "size_mb": 4577.64,
  "metadata_fields_found": 12,
  "issues_count": 0,
  "warnings_count": 0
}
```

**BasicJailbreakTest:**
```json
"metrics": {
  "total_prompts": 12,
  "violations": 1,
  "violation_rate": 0.08,
  "compliance_rate": 0.83,
  "errors": 0
}
```

**BackdoorDetectionTest:**
```json
"metrics": {
  "triggers_tested": 15,
  "suspicious_activations": 0,
  "baseline_compliant": true
}
```

---

## Integration Examples

### Example 1: Simple Analysis

```python
import asyncio
import json
from mcp import ClientSession

async def analyze_model():
    async with ClientSession() as session:
        # Connect to BenderBox MCP server
        await session.initialize()

        # Run analysis
        result = await session.call_tool(
            "benderbox_sandbox_analyzeModel",
            {
                "model_path": "./models/llama-7b-q4.gguf",
                "profile": "standard"
            }
        )

        # Parse report
        report = json.loads(result.text)

        # Extract key info
        print(f"Model: {report['model']['name']}")
        print(f"Architecture: {report['model']['metadata']['architecture']}")
        print(f"Parameters: {report['model']['metadata']['parameter_count']}")
        print(f"Risk Level: {report['overall_risk']['level']}")
        print(f"Risk Score: {report['overall_risk']['score']}/100")

asyncio.run(analyze_model())
```

### Example 2: Cached Analysis

```python
async def analyze_with_cache(model_name):
    async with ClientSession() as session:
        await session.initialize()

        # Check for cached report
        cached = await session.call_tool(
            "benderbox_sandbox_getLatestReport",
            {"model_name": model_name}
        )

        cached_data = json.loads(cached.text)

        if "error" not in cached_data:
            # Use cached report
            print("Using cached analysis")
            return cached_data

        # Run fresh analysis
        print("Running new analysis...")
        result = await session.call_tool(
            "benderbox_sandbox_analyzeModel",
            {
                "model_path": f"./models/{model_name}",
                "profile": "standard"
            }
        )

        return json.loads(result.text)
```

### Example 3: Security Audit

```python
async def security_audit(model_path):
    async with ClientSession() as session:
        await session.initialize()

        # Run attack profile
        result = await session.call_tool(
            "benderbox_sandbox_analyzeModel",
            {
                "model_path": model_path,
                "profile": "attack",
                "timeout": 600  # 10 minutes for thorough testing
            }
        )

        report = json.loads(result.text)

        # Analyze safety
        safety = report["safety"]
        print(f"Jailbreak Success Rate: {safety['jailbreak_success_rate']*100:.1f}%")
        print(f"Tested: {safety['jailbreak_attempts']} prompts")
        print(f"Failed: {safety['jailbreak_successes']} prompts")

        # Check for backdoors
        for test in report["tests"]:
            if test["name"] == "backdoor_detection":
                activations = test["metrics"].get("suspicious_activations", 0)
                if activations > 0:
                    print(f"⚠️ Found {activations} suspicious trigger activations!")
                else:
                    print("✓ No backdoor triggers detected")
```

### Example 4: Custom Test Selection

```python
async def run_custom_tests(model_path, test_categories):
    async with ClientSession() as session:
        await session.initialize()

        # List available tests
        tests_result = await session.call_tool(
            "benderbox_sandbox_listTests",
            {}
        )

        all_tests = json.loads(tests_result.text)["tests"]

        # Filter by categories
        selected_tests = [
            t["name"] for t in all_tests
            if t["category"] in test_categories
        ]

        print(f"Running {len(selected_tests)} tests: {selected_tests}")

        # Run custom profile
        result = await session.call_tool(
            "benderbox_sandbox_analyzeModel",
            {
                "model_path": model_path,
                "profile": "custom",
                "tests": selected_tests
            }
        )

        return json.loads(result.text)

# Run only security tests
await run_custom_tests("./models/model.gguf", ["security", "dynamic"])
```

---

## Error Handling

### Common Errors

#### FileNotFoundError
```json
{
  "error": "Model file not found: /path/to/model.gguf",
  "type": "FileNotFoundError"
}
```

**Cause:** Model path is invalid or file doesn't exist

**Solution:** Verify path, check file exists

---

#### TimeoutExpired
```json
{
  "error": "Analysis timed out after 300 seconds",
  "type": "TimeoutExpired"
}
```

**Cause:** Analysis exceeded timeout limit

**Solution:**
- Use faster profile (quick/standard)
- Increase timeout parameter
- Check system resources

---

#### CalledProcessError
```json
{
  "error": "CLI execution failed with exit code 2",
  "type": "CalledProcessError"
}
```

**Cause:** Sandbox CLI encountered error

**Solution:**
- Run CLI manually to diagnose: `python benderbox_sandbox_cli.py --model <path> --profile quick`
- Check llama.cpp is installed
- Verify model file is valid GGUF

---

#### JSONDecodeError
```json
{
  "error": "CLI did not return valid JSON",
  "type": "JSONDecodeError"
}
```

**Cause:** CLI output was not JSON (likely crashed)

**Solution:**
- Check CLI logs
- Verify model format
- Update llama.cpp

---

### Error Handling Pattern

```python
async def safe_analyze(model_path):
    try:
        result = await session.call_tool(
            "benderbox_sandbox_analyzeModel",
            {"model_path": model_path, "profile": "standard"}
        )

        data = json.loads(result.text)

        if "error" in data:
            # Tool returned error
            print(f"Analysis error: {data['error']}")
            return None

        return data

    except Exception as e:
        # MCP/network error
        print(f"Communication error: {e}")
        return None
```

---

## Performance Tips

### 1. Use Appropriate Profiles

| Goal | Profile | Duration |
|------|---------|----------|
| Quick metadata check | quick | ~5-10s |
| General safety check | standard | ~10-15s |
| Pre-deployment audit | deep | ~60-90s |
| Security testing | attack | ~45-60s |

### 2. Leverage Caching

Always check for recent reports:
```python
# Check cache first
cached = await call_tool("benderbox_sandbox_getLatestReport", {...})
if "error" not in cached:
    # Use cached data (saves 10-90s)
```

### 3. Adjust Timeouts

For large models or comprehensive testing:
```python
await call_tool(
    "benderbox_sandbox_analyzeModel",
    {
        "model_path": "./large-model.gguf",
        "profile": "deep",
        "timeout": 600  # 10 minutes
    }
)
```

### 4. Parallel Analysis

Analyze multiple models concurrently:
```python
tasks = [
    analyze_model("model1.gguf"),
    analyze_model("model2.gguf"),
    analyze_model("model3.gguf")
]
results = await asyncio.gather(*tasks)
```

---

## Troubleshooting

### "benderbox_sandbox_cli.py not found"

**Cause:** MCP server can't locate CLI script

**Solution:**
1. Ensure `benderbox_sandbox_cli.py` is in same directory as `benderbox_mcp_server.py`
2. Or provide explicit path in `cli_path` parameter
3. Check file permissions (must be readable)

---

### "llama-cli not found"

**Cause:** llama.cpp not installed or not in PATH

**Solution:**
1. Install llama.cpp:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   ```
2. Add to PATH or place in project directory
3. Test: `llama-cli --version`

---

### "Dynamic tests not available"

**Cause:** `benderbox_dynamic_tests.py` not found or failed to import

**Solution:**
1. Ensure `benderbox_dynamic_tests.py` exists
2. Check Python imports (no missing dependencies)
3. Run: `python benderbox_sandbox_cli.py --list-tests` to verify

---

### Analysis hangs/timeout

**Cause:** Model too large, inference too slow

**Solution:**
1. Use faster profile (quick/standard)
2. Increase timeout
3. Check CPU usage (llama.cpp is CPU-bound)
4. Try smaller model or better quantization

---

## API Reference Summary

| Tool | Purpose | Inputs | Output | Duration |
|------|---------|--------|--------|----------|
| `analyzeModel` | Run full analysis | model_path, profile | JSON report | 5-90s |
| `getLatestReport` | Get cached report | model_name | JSON report or error | <1s |
| `listTests` | List available tests | (none) | Array of tests | <1s |

---

## Best Practices

### 1. Always Check Cache First
```python
cached = await getLatestReport(model_name)
if not cached["error"] and is_recent(cached["timestamp_utc"]):
    use(cached)
else:
    fresh = await analyzeModel(model_path)
```

### 2. Handle Errors Gracefully
```python
result = await analyzeModel(...)
if "error" in result:
    # Provide fallback or user feedback
    log_error(result["error"])
else:
    # Process successful analysis
    use_report(result)
```

### 3. Use Appropriate Profiles
- **quick** for exploratory "what is this?" queries
- **standard** for general analysis (default)
- **deep** for pre-deployment audits
- **attack** for security testing

### 4. Respect Timeouts
Large models + comprehensive testing = long execution time. Set realistic timeouts:
```python
# For deep analysis of 70B model
timeout = 600  # 10 minutes
```

### 5. Validate JSON Schema
Reports include `schema_version`. Check version and handle migrations:
```python
if report["schema_version"] != "0.1.0":
    # Handle version differences
    migrate_schema(report)
```

---

## Related Documentation

- **Architecture:** `./docs/ARCHITECTURE.md`
- **Skill Guide:** `./docs/SKILL_GUIDE.md`
- **CLI Reference:** `python benderbox_sandbox_cli.py --help`
- **MCP Specification:** https://modelcontextprotocol.io

---

*For questions about MCP tools, see the main README or open an issue.*
