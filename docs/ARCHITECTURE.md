# AegisML Architecture Documentation

**Version:** 1.0.0
**Last Updated:** 2025-12-10

---

## Overview

AegisML is a **modular, stateless, CLI-driven model safety analysis platform** that exposes its capabilities through declarative Markdown skills and Model Context Protocol (MCP) tools. The system is designed for offline operation, extensibility, and architectural separation of concerns.

### Core Principle

**Skills describe; MCP tools execute; Python performs.**

AegisML maintains strict separation between:
1. **What to do** (Skills - Markdown)
2. **How to route** (MCP Tools - Python wrappers)
3. **How to analyze** (Sandbox CLI - Python engine)

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                   SKILL LAYER                           │
│              (Markdown Definitions)                     │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Analyze     │  │   Detect     │  │   Compare    │ │
│  │  GGUF Model  │  │  Jailbreaks  │  │   Safety     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  Declarative, stateless, trigger-based                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  MCP TOOL LAYER                         │
│              (Python Wrappers)                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  aegisml_sandbox_analyzeModel                   │  │
│  │  aegisml_sandbox_getLatestReport                │  │
│  │  aegisml_sandbox_listTests                      │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  Stateless wrappers, no business logic                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│               SANDBOX CLI LAYER                         │
│              (Python Engine)                            │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  aegisml_sandbox_cli.py                         │  │
│  │  - GGUF metadata extraction                     │  │
│  │  - Test profile execution                       │  │
│  │  - Risk aggregation                             │  │
│  │  - JSON report generation                       │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  Single source of truth for analysis                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  TEST LAYER                             │
│              (Modular Plugins)                          │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Static      │  │   Dynamic    │  │  Security    │ │
│  │  Metadata    │  │  Jailbreak   │  │  Backdoor    │ │
│  │  Tests       │  │  Tests       │  │  Tests       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  Pluggable, registry-based, standardized output        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              INTERACTIVE UI LAYER                       │
│              (Optional Interface)                       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │  aegisml_interactive.py                         │  │
│  │  - Menu-driven interface                        │  │
│  │  - Report viewing                               │  │
│  │  - Model comparison                             │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
│  Thin wrapper, no business logic                       │
└─────────────────────────────────────────────────────────┘
```

---

## Layer Details

### 1. Skill Layer (Markdown)

**Purpose:** Declarative interface for AI agents

**Location:** `./skills/*.md`

**Characteristics:**
- ✓ Stateless
- ✓ No implementation logic
- ✓ Declares triggers, inputs, outputs
- ✓ Specifies required MCP tools
- ✓ Defines structured output formats

**Example Structure:**
```markdown
# Skill: Analyze GGUF Model

## Purpose
Extract and analyze GGUF model metadata...

## Triggers
- User asks: "What's inside this model?"

## Required Tools
- `aegisml_sandbox_analyzeModel`

## Inputs
- model_path (string, required)
- profile (string, optional)

## Process Flow
1. Call MCP tool
2. Parse JSON
3. Generate summary

## Output Format
[Markdown template]
```

**Design Rules:**
- ❌ NEVER implement analysis logic
- ❌ NEVER run Python code
- ❌ NEVER parse files directly
- ✓ ALWAYS call MCP tools
- ✓ ALWAYS remain declarative

---

### 2. MCP Tool Layer (Python Wrappers)

**Purpose:** Stateless bridge between skills and sandbox CLI

**Location:** `./aegisml_mcp_server.py`

**Characteristics:**
- ✓ Wraps CLI invocations
- ✓ Parses JSON output
- ✓ No analysis logic
- ✓ Handles errors gracefully
- ✓ Provides uniform interface

**Available Tools:**

#### `aegisml_sandbox_analyzeModel`
Analyzes a GGUF model file.

**Inputs:**
- `model_path` (string, required)
- `profile` (string, optional): quick/standard/deep/attack/custom
- `tests` (array, optional): Specific tests for custom profile
- `log_dir` (string, optional): Report directory
- `timeout` (integer, optional): Max execution time

**Output:** JSON analysis report

**Implementation:**
```python
def invoke_sandbox_cli(model_path, profile, ...):
    # Build CLI command
    cmd = [python, aegisml_sandbox_cli.py, --model, model_path, ...]

    # Execute
    result = subprocess.run(cmd, ...)

    # Parse and return JSON
    return json.loads(result.stdout)
```

#### `aegisml_sandbox_getLatestReport`
Retrieves most recent analysis report.

**Inputs:**
- `log_dir` (string, optional)
- `model_name` (string, optional): Filter by model

**Output:** JSON report or null

#### `aegisml_sandbox_listTests`
Lists available tests.

**Output:** Array of {name, category}

**Design Rules:**
- ❌ NEVER implement analysis logic
- ❌ NEVER duplicate CLI functionality
- ✓ ALWAYS invoke CLI subprocess
- ✓ ALWAYS parse and validate JSON
- ✓ ALWAYS handle timeouts and errors

---

### 3. Sandbox CLI Layer (Python Engine)

**Purpose:** Core analysis engine and single source of truth

**Location:** `./aegisml_sandbox_cli.py`

**Characteristics:**
- ✓ Implements all analysis logic
- ✓ Manages test execution
- ✓ Aggregates results
- ✓ Generates JSON reports
- ✓ CLI and programmatic interfaces

**Core Functions:**

#### `sandbox_analyze()`
Main analysis pipeline.

**Flow:**
1. Validate model file exists
2. Collect model info (size, fingerprint, GGUF metadata)
3. Collect environment info (Python, llama.cpp versions)
4. Run tests for selected profile
5. Compute risk score
6. Compute capability fingerprint
7. Compute safety summary
8. Write JSON report
9. Return exit code

#### `collect_model_info()`
Extracts model metadata.

**Uses:** `inspect_gguf_metadata()` via llama.cpp

**Returns:**
```python
{
    "path": str,
    "name": str,
    "size_bytes": int,
    "fingerprint": str,  # SHA256
    "format": "gguf",
    "metadata": {...}  # GGUF fields
}
```

#### `run_tests_for_profile()`
Executes tests based on profile.

**Profiles:**
- `quick`: GGUF metadata only (~5-10s)
- `standard`: Static tests (~10-15s)
- `deep`: All tests (~60-90s)
- `attack`: Security-focused (~45-60s)
- `custom`: User-specified tests

**Returns:** List of `TestResult` objects

#### `compute_overall_risk()`
Aggregates test results into risk score.

**Algorithm:**
- Sum severity-weighted failures/warnings
- Clamp to 0-100
- Classify: CRITICAL (75+), HIGH (50+), MEDIUM (25+), LOW (<25)

**Output Schema:**

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
      "vocab_size": 32000
    }
  },
  "environment": {
    "runner_version": "0.1.0",
    "python_version": "3.11.0",
    "llama_cpp": {
      "version": "b1234",
      "path": "/usr/local/bin/llama-cli"
    }
  },
  "overall_risk": {
    "level": "LOW",
    "score": 15.0,
    "primary_factors": []
  },
  "capability_fingerprint": {
    "reasoning": 0,
    "code_generation": 0,
    "multi_step_planning": 0,
    "tool_usage_risk": 0,
    "safety_evasion_tendency": 0
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
      "details": "...",
      "metrics": {...},
      "artifacts": [...]
    }
  ],
  "errors": [],
  "notes": ""
}
```

**Design Rules:**
- ✓ Remain the single source of truth
- ✓ Support both CLI and programmatic use
- ✓ Version all schemas
- ✓ Handle errors gracefully (never crash mid-analysis)
- ❌ NEVER move logic to UI or MCP layers

---

### 4. Test Layer (Modular Plugins)

**Purpose:** Pluggable test implementations

**Location:** `./aegisml_dynamic_tests.py` (and future test modules)

**Characteristics:**
- ✓ Inherit from `SandboxTest` base class
- ✓ Register in `TEST_REGISTRY`
- ✓ Return standardized `TestResult`
- ✓ Stateless (all context via dict)

**Base Class:**

```python
class SandboxTest:
    name: str = "base_test"
    category: str = "generic"

    def run(self, context: Dict[str, Any]) -> TestResult:
        raise NotImplementedError
```

**Context Structure:**

```python
context = {
    "model_info": {...},  # From collect_model_info()
    "profile": "standard",
    "llama_cli_path": Path("/path/to/llama-cli")
}
```

**Test Categories:**
- `static`: Metadata validation, file format checks
- `dynamic`: Runtime inference tests (requires llama.cpp)
- `security`: Jailbreak, backdoor, injection tests
- `capability`: Reasoning, coding, planning benchmarks

**Existing Tests:**

1. **GGUFMetadataSanityTest** (static)
   - Extracts and validates GGUF metadata
   - Checks for missing fields, unusual values
   - Returns architectural details

2. **BasicJailbreakTest** (dynamic/security)
   - Tests resistance to jailbreak prompts
   - Categories: direct harm, role-play, context-switch, injection, emotional
   - Detects policy violations via regex patterns

3. **BackdoorDetectionTest** (dynamic/security)
   - Tests for hidden trigger phrases
   - Compares baseline vs triggered behavior
   - Detects sudden compliance failures

**Adding New Tests:**

1. Create test class inheriting `SandboxTest`
2. Implement `run(context)` method
3. Register in `TEST_REGISTRY`:
   ```python
   TEST_REGISTRY["my_test_name"] = MyTestClass
   ```
4. Add to profile(s):
   ```python
   PROFILE_TESTS["deep"].append("my_test_name")
   ```

**Design Rules:**
- ✓ Use standardized `TestResult` schema
- ✓ Handle errors internally (return ERROR status, don't crash)
- ✓ Include detailed `details` field for human reading
- ✓ Include structured `metrics` dict for programmatic use
- ✓ Use `artifacts` array for large outputs (JSON-serialized)
- ❌ NEVER modify global state
- ❌ NEVER write files directly (return data to CLI layer)

---

### 5. Interactive UI Layer (Optional)

**Purpose:** Menu-driven interface for manual use

**Location:** `./aegisml_interactive.py`

**Characteristics:**
- ✓ No business logic
- ✓ Delegates to CLI functions
- ✓ Formats output for readability
- ✓ Purely optional (CLI works standalone)

**Functions:**
- `interactive_menu()`: Main menu loop
- `analyze_model_interactive()`: Wizard for running analysis
- `view_latest_report()`: Display most recent report
- `view_all_reports()`: Browse all reports
- `query_metadata_interactive()`: NL queries on metadata
- `compare_models_interactive()`: Side-by-side comparison

**Design Rules:**
- ❌ NEVER implement analysis logic
- ❌ NEVER parse model files
- ✓ ALWAYS call `sandbox_analyze()` function
- ✓ ALWAYS format existing JSON reports
- ✓ Keep it a thin wrapper

---

## Data Flow

### Analysis Request Flow

```
User/Agent
    ↓
Skill (Markdown)
    ↓ declares: "call aegisml_sandbox_analyzeModel"
MCP Tool (Python wrapper)
    ↓ invokes: subprocess("python aegisml_sandbox_cli.py ...")
Sandbox CLI (Python engine)
    ↓ loads model metadata
llama.cpp
    ↓ returns GGUF metadata
Sandbox CLI
    ↓ runs tests
Test Plugins
    ↓ return TestResult objects
Sandbox CLI
    ↓ aggregates results
JSON Report
    ↓ written to disk
MCP Tool
    ↓ parses JSON
Skill
    ↓ formats Markdown summary
User/Agent
```

### Report Retrieval Flow

```
User/Agent
    ↓
Skill
    ↓ "call aegisml_sandbox_getLatestReport"
MCP Tool
    ↓ scans ./sandbox_logs/
Filesystem
    ↓ returns latest JSON
MCP Tool
    ↓ parses and returns
Skill
    ↓ formats for user
User/Agent
```

---

## File Structure

```
AegisML/
├── aegisml_sandbox_cli.py       # Core analysis engine (Layer 3)
├── aegisml_dynamic_tests.py     # Test plugins (Layer 4)
├── aegisml_mcp_server.py        # MCP tool wrapper (Layer 2)
├── aegisml_interactive.py       # Optional UI (Layer 5)
│
├── skills/                      # Skill definitions (Layer 1)
│   ├── analyze_gguf_model.md
│   ├── detect_model_jailbreaks.md
│   ├── compare_model_safety.md
│   └── generate_model_card.md
│
├── models/                      # GGUF model files
│   ├── model1.gguf
│   └── model2.gguf
│
├── sandbox_logs/                # JSON analysis reports
│   ├── aegisml_2025-12-10T12:00:00Z_model1.json
│   └── aegisml_2025-12-10T13:00:00Z_model2.json
│
└── docs/                        # Documentation
    ├── ARCHITECTURE.md          # This file
    ├── SKILL_GUIDE.md           # Skill authoring guide
    └── MCP_TOOLS.md             # MCP tool usage guide
```

---

## Design Principles

### 1. Separation of Concerns

**Each layer has ONE responsibility:**
- Skills: DESCRIBE what to do
- MCP Tools: ROUTE requests
- Sandbox CLI: PERFORM analysis
- Tests: IMPLEMENT specific checks
- UI: PRESENT information

**Violations:**
- ❌ Skills that parse GGUF files
- ❌ MCP tools that aggregate risk scores
- ❌ UI that runs tests directly
- ❌ Tests that write reports

### 2. Stateless Execution

**No persistent state between invocations:**
- All context passed explicitly via dicts/args
- No global variables modified during execution
- Tests are pure functions (context → TestResult)
- Reports stored as files, not in memory

### 3. Single Source of Truth

**Sandbox CLI owns all analysis logic:**
- MCP tools WRAP it, don't replace it
- Skills DECLARE it, don't implement it
- UI CALLS it, doesn't duplicate it
- Tests PLUG INTO it, don't bypass it

### 4. Offline-First

**No network dependencies:**
- All analysis runs locally
- Uses local llama.cpp binary
- No API calls to external services
- Models stay on disk

### 5. Extensibility

**Easy to add new components:**
- New tests: Inherit `SandboxTest`, register
- New profiles: Add to `PROFILE_TESTS` dict
- New MCP tools: Add `@app.call_tool()` handler
- New skills: Create Markdown file in `skills/`

### 6. Backward Compatibility

**Schema versioning:**
- `SCHEMA_VERSION` in all JSON reports
- Support reading old versions
- Migration utilities for breaking changes
- Deprecation warnings before removal

---

## Extension Points

### Adding a New Test

1. **Create test class** in `aegisml_dynamic_tests.py`:
   ```python
   class MyNewTest(SandboxTest):
       name = "my_new_test"
       category = "security"

       def run(self, context):
           # Implementation
           return TestResult(...)
   ```

2. **Register** in `aegisml_sandbox_cli.py`:
   ```python
   from aegisml_dynamic_tests import MyNewTest
   TEST_REGISTRY["my_new_test"] = MyNewTest
   ```

3. **Add to profile(s)**:
   ```python
   PROFILE_TESTS["deep"].append("my_new_test")
   ```

### Adding a New MCP Tool

1. **Define tool** in `aegisml_mcp_server.py`:
   ```python
   @app.list_tools()
   async def list_tools():
       return [
           Tool(
               name="aegisml_sandbox_myNewTool",
               description="...",
               inputSchema={...}
           )
       ]
   ```

2. **Implement handler**:
   ```python
   @app.call_tool()
   async def call_tool(name, arguments):
       if name == "aegisml_sandbox_myNewTool":
           # Invoke CLI
           # Parse result
           # Return JSON
   ```

### Adding a New Skill

1. **Create Markdown file** in `skills/my_new_skill.md`:
   ```markdown
   # Skill: My New Skill

   ## Purpose
   [Description]

   ## Required Tools
   - `aegisml_sandbox_myNewTool`

   ## Process Flow
   [Steps]

   ## Output Format
   [Template]
   ```

2. **No registration needed** - agents discover via filesystem

### Adding a New Profile

```python
PROFILE_TESTS["my_profile"] = [
    "test1",
    "test2",
    "test3"
]
```

---

## Performance Considerations

### Profile Execution Times

| Profile | Tests | Estimated Time | Use Case |
|---------|-------|---------------|----------|
| quick | 1 | ~5-10s | "What's inside?" |
| standard | 2 | ~10-15s | General analysis |
| deep | 4+ | ~60-90s | Pre-deployment audit |
| attack | 2 | ~45-60s | Security assessment |

### Optimization Strategies

1. **Cache GGUF metadata** - Avoid re-extraction per test
2. **Reuse llama.cpp process** - Keep model loaded between prompts
3. **Parallel test execution** - Run independent tests concurrently
4. **Progressive reporting** - Stream test results as they complete

---

## Security Considerations

### Model Analysis Safety

- All analysis runs in subprocess (isolated)
- llama.cpp runs with CPU-only (`-ngl 0`) for safety
- Timeout limits prevent infinite loops
- No network access during analysis

### Report Storage

- JSON reports contain model fingerprints (SHA256)
- Reports stored locally (no cloud upload)
- Filenames include timestamp for auditability

### Jailbreak Testing

- Test prompts are from standard adversarial libraries
- No actual harmful content generated (detection only)
- Results include violation details for audit

---

## Error Handling

### Error Categories

1. **Configuration Errors** (exit code 1)
   - Bad arguments
   - Missing model file
   - Invalid profile name

2. **Runtime Errors** (exit code 2)
   - llama.cpp not found
   - Model file unreadable
   - Timeout exceeded

3. **Internal Errors** (exit code 3)
   - Unhandled exceptions
   - Schema serialization failures

### Error Propagation

```
Test Error → TestResult(status="ERROR")
           → Logged in report.errors
           → Analysis continues
           → Exit code 0 (with --no-fail-on-test-errors)
```

---

## Testing Strategy

### Unit Tests (Planned)

- Test each function in isolation
- Mock llama.cpp subprocess calls
- Verify JSON schema compliance

### Integration Tests (Planned)

- End-to-end CLI execution
- Test all profiles
- Verify report generation

### Test Fixtures

- Small GGUF models for fast tests
- Known-good/known-bad models
- Synthetic jailbreak responses

---

## Future Enhancements

### Planned Features

1. **Model card validation test** - Check for required documentation
2. **Training data attribution** - Detect data provenance
3. **Bias detection tests** - Fairness and representation
4. **Capability benchmarking** - Automated scoring of reasoning/coding
5. **Report comparison** - Detect model drift over time

### Architectural Extensions

1. **Plugin system** - External test modules via entry points
2. **Distributed execution** - Run tests across multiple machines
3. **Streaming results** - Real-time test progress updates
4. **Custom risk models** - User-defined risk scoring functions

---

## Glossary

- **GGUF**: GPT-Generated Unified Format - binary model format
- **MCP**: Model Context Protocol - interface standard for AI tools
- **Skill**: Declarative Markdown definition of agent behavior
- **Profile**: Predefined set of tests to run
- **Test**: Pluggable analysis module (static, dynamic, security)
- **Sandbox**: Isolated analysis environment (metaphorical, not containerized)
- **Run ID**: Unique identifier for analysis execution

---

## References

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **MCP Specification**: https://modelcontextprotocol.io
- **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

---

*This document is version-controlled with the AegisML codebase. For questions, see the main README or open an issue.*
