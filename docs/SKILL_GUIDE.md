# BenderBox Skill Authoring Guide

**Version:** 1.0.0
**Last Updated:** 2025-12-10

---

## Overview

Skills are **declarative Markdown files** that define how AI agents should use BenderBox tools. They specify triggers, inputs, outputs, and process flows WITHOUT implementing any analysis logic.

### Key Principle

**Skills describe WHAT to do, not HOW to do it.**

All actual analysis is performed by MCP tools, which wrap the Sandbox CLI.

---

## Skill Anatomy

### Required Sections

Every skill MUST include these sections:

```markdown
# Skill: [Name]

## Purpose
[1-2 sentence description of what this skill accomplishes]

## Triggers
[List of user inputs that should invoke this skill]

## Required Tools
[MCP tools this skill needs]

## Inputs
[Parameters the skill accepts]

## Process Flow
[Step-by-step logic]

## Output Format
[Template for results]
```

### Optional Sections

```markdown
## Error Handling
[How to handle common errors]

## Example Interactions
[Sample user dialogs]

## Best Practices
[Usage recommendations]

## Constraints
[What the skill should NEVER do]

## Metadata
[Version, category, dependencies]
```

---

## Section Details

### 1. Purpose

**Format:** 1-3 sentences

**Good:**
```markdown
## Purpose
Extract and analyze GGUF model metadata, safety characteristics, and capabilities
to provide comprehensive model assessment.
```

**Bad:**
```markdown
## Purpose
This skill is used to analyze models by calling the analyze tool and parsing the
results which come back as JSON and then formatting them into a nice summary...
```

**Guidelines:**
- Be concise
- Focus on user value
- No implementation details

---

### 2. Triggers

**Format:** Bulleted list of user inputs

**Good:**
```markdown
## Triggers
- User asks: "What's inside this model?"
- User asks: "Analyze this GGUF file"
- User provides: Path to a GGUF file
- User requests: Model safety assessment
```

**Bad:**
```markdown
## Triggers
- When the user wants to analyze something
- If they have a model
```

**Guidelines:**
- Be specific (quote actual phrases)
- Cover variations
- Include implicit triggers (e.g., "User provides: file path")

---

### 3. Required Tools

**Format:** Bulleted list of MCP tool names

**Good:**
```markdown
## Required Tools
- `benderbox_sandbox_analyzeModel` - Primary analysis tool
- `benderbox_sandbox_getLatestReport` - For cached results
```

**Bad:**
```markdown
## Required Tools
- The analysis tool
- Something to get reports
```

**Guidelines:**
- Use exact tool names (backtick-formatted)
- Add brief description of why each is needed
- List in order of typical usage

---

### 4. Inputs

**Format:** Parameter list with types and descriptions

**Good:**
```markdown
## Inputs
- **model_path** (string, required): Path to GGUF model file
  - Examples: `./models/llama-7b-q4.gguf`, `model.gguf`
- **profile** (string, optional): Analysis depth [DEFAULT: standard]
  - `quick` - Fast metadata only (~5-10s)
  - `standard` - Common static tests (~10-15s)
  - `deep` - All tests (~60-90s)
- **force_reanalysis** (boolean, optional): Skip cache [DEFAULT: false]
```

**Bad:**
```markdown
## Inputs
- model_path: the path to the model
- profile: what kind of analysis
```

**Guidelines:**
- Use Markdown bold for parameter names
- Specify type and required/optional
- Provide examples for string/enum types
- Show default values
- Include timing estimates for profiles

---

### 5. Process Flow

**Format:** Numbered steps with code blocks for tool calls

**Good:**
```markdown
## Process Flow

### 1. Check for Existing Analysis
If `force_reanalysis` is false:
  - Call `benderbox_sandbox_getLatestReport` with model filename
  - If report exists and is recent (< 24 hours), use cached data
  - Otherwise, proceed to step 2

### 2. Run Analysis
Call `benderbox_sandbox_analyzeModel` with:
\`\`\`json
{
  "model_path": "<user_provided_path>",
  "profile": "<user_selected_or_default>",
  "log_dir": "./sandbox_logs"
}
\`\`\`

### 3. Parse JSON Response
Extract key sections:
- `model.metadata` - GGUF fields
- `overall_risk` - Risk assessment
- `tests` - Test results

### 4. Generate Summary
Transform JSON into Markdown (see Output Format)
```

**Bad:**
```markdown
## Process Flow
1. Call the tool
2. Get the results
3. Show them to the user
```

**Guidelines:**
- Be specific about conditions (if/else)
- Show exact JSON payloads for tool calls
- Reference other sections (e.g., "see Output Format")
- Use code blocks for structured data

---

### 6. Output Format

**Format:** Markdown template with placeholders

**Good:**
```markdown
## Output Format

\`\`\`markdown
# Model Analysis Report

## Model Information
- **Name:** [model.name]
- **Size:** [size in GB]
- **SHA256:** [fingerprint (first 16 chars)]

## GGUF Metadata
- **Architecture:** [architecture]
- **Parameters:** [parameter_count]
- **Quantization:** [quantization] ([quantization_bits] bits)

## Risk Assessment
- **Level:** [overall_risk.level]
- **Score:** [overall_risk.score]/100

### Primary Risk Factors
[List from overall_risk.primary_factors]

---
*Report ID: [run_id]*
\`\`\`
```

**Bad:**
```markdown
## Output Format
Show the results in Markdown
```

**Guidelines:**
- Provide complete template
- Use `[field.path]` notation for JSON access
- Include formatting (bold, tables, lists)
- Show conditional sections (if violations, show X)
- Add metadata footer (report ID, timestamp)

---

### 7. Error Handling

**Format:** Error scenario → Response template

**Good:**
```markdown
## Error Handling

### If model file not found:
\`\`\`markdown
❌ **Error:** Model file not found at `[model_path]`

**Suggestions:**
- Check the file path is correct
- Ensure the file exists: `ls [model_path]`
- Models should be in `./models/` directory
\`\`\`

### If analysis times out:
\`\`\`markdown
⚠️ **Warning:** Analysis timed out after [timeout] seconds

**Suggestions:**
- Try a faster profile: `quick` or `standard`
- Increase timeout for large models
- Check system resources
\`\`\`
```

**Bad:**
```markdown
## Error Handling
If there's an error, tell the user
```

**Guidelines:**
- List specific error conditions
- Provide exact error message templates
- Include troubleshooting steps
- Use appropriate emoji (❌ for errors, ⚠️ for warnings)

---

### 8. Example Interactions

**Format:** User message → Agent actions + response

**Good:**
```markdown
## Example Interactions

### Example 1: Quick Check
\`\`\`
User: "What's inside llama-7b-q4.gguf?"

Agent: [Calls benderbox_sandbox_analyzeModel with profile=quick]
Agent: [Generates standard summary with metadata]
Agent: "This is a LLaMA 7B model with Q4 quantization..."
\`\`\`

### Example 2: Security Audit
\`\`\`
User: "Is model.gguf safe for production?"

Agent: [Calls benderbox_sandbox_analyzeModel with profile=deep]
Agent: [Generates deep summary with safety analysis]
Agent: "Based on analysis, this model shows 8% jailbreak success rate..."
\`\`\`
```

**Guidelines:**
- Show realistic user inputs
- Bracket agent actions: [Calls tool...]
- Include actual response snippets
- Cover common variations

---

### 9. Best Practices

**Format:** Subsections with guidance

**Good:**
```markdown
## Best Practices

### Profile Selection
- **Use `quick`** when user just wants to know "what's inside"
- **Use `standard`** for general safety checks (default)
- **Use `deep`** when user explicitly requests thorough analysis

### Caching Strategy
- Cache reports for 24 hours by default
- Always offer "re-run fresh analysis" option
- Show timestamp of cached report

### Risk Communication
- **LOW risk:** "This model appears safe for general use"
- **CRITICAL risk:** "This model is NOT recommended for production"
```

**Guidelines:**
- Provide decision trees
- Show phrasing for different risk levels
- Include caching and performance tips

---

### 10. Constraints

**Format:** NEVER / ALWAYS lists

**Good:**
```markdown
## Constraints

### NEVER:
- ❌ Implement analysis logic in the skill
- ❌ Parse GGUF files directly
- ❌ Run Python code to inspect models
- ❌ Make assumptions about safety without running analysis

### ALWAYS:
- ✓ Call MCP tools for all analysis
- ✓ Show risk assessment prominently
- ✓ Provide actionable next steps
- ✓ Handle errors gracefully
```

**Guidelines:**
- Use emoji (❌ for NEVER, ✓ for ALWAYS)
- Focus on architectural boundaries
- Emphasize safety and user experience

---

### 11. Metadata

**Format:** Key-value pairs

**Good:**
```markdown
## Metadata

- **Version:** 1.0.0
- **Category:** Model Analysis
- **Complexity:** Medium
- **Estimated Execution Time:** 5-90s (profile-dependent)
- **Dependencies:** llama.cpp, benderbox_sandbox_cli.py
- **Output Type:** Markdown report
```

**Guidelines:**
- Include semantic version
- Categorize (Analysis, Security, Comparison, Documentation)
- Rate complexity (Low/Medium/High)
- Estimate execution time
- List external dependencies

---

## Skill Templates

### Template 1: Analysis Skill

```markdown
# Skill: [Analyze Something]

## Purpose
[What this analysis accomplishes]

## Triggers
- User asks: "[Question]"
- User requests: [Type of analysis]

## Required Tools
- `benderbox_sandbox_[toolName]`

## Inputs
- **input1** (type, required): [Description]
- **input2** (type, optional): [Description] [DEFAULT: value]

## Process Flow

### 1. [Step Name]
[Logic and conditions]

### 2. Execute Analysis
Call `benderbox_sandbox_[toolName]`:
\`\`\`json
{
  "param": "value"
}
\`\`\`

### 3. Format Results
[How to transform JSON to output]

## Output Format

\`\`\`markdown
# [Report Title]

[Template with [placeholders]]
\`\`\`

## Error Handling

### If [error condition]:
\`\`\`markdown
[Error message template]
\`\`\`

## Constraints

### NEVER:
- ❌ [Anti-pattern]

### ALWAYS:
- ✓ [Best practice]

## Metadata
- **Version:** 1.0.0
- **Category:** [Category]
```

### Template 2: Comparison Skill

```markdown
# Skill: Compare [Things]

## Purpose
[What this comparison helps users decide]

## Triggers
- User asks: "Which is better?"
- User provides: Multiple [items] to compare

## Required Tools
- `benderbox_sandbox_[toolName]` - Run for each item

## Inputs
- **items** (array of strings, required): Items to compare
  - Minimum: 2
  - Maximum: 5
- **focus** (string, optional): Comparison dimension [DEFAULT: all]

## Process Flow

### 1. Validate Inputs
[Checks]

### 2. Gather Data
For each item:
  - [How to get data]

### 3. Generate Comparison Table
[Table structure]

### 4. Provide Recommendation
[Logic for choosing winner]

## Output Format

\`\`\`markdown
# [Comparison] Report

| Metric | Item 1 | Item 2 | Winner |
|--------|--------|--------|--------|
[Table rows]

## Recommendation
[Recommendation with justification]
\`\`\`

[Rest of sections...]
```

---

## MCP Tool Reference

Available tools for skills to call:

### `benderbox_sandbox_analyzeModel`

**Purpose:** Run model analysis

**Inputs:**
```json
{
  "model_path": "string (required)",
  "profile": "string (optional): quick|standard|deep|attack|custom",
  "tests": "array (optional): ['test1', 'test2']",
  "log_dir": "string (optional): './sandbox_logs'",
  "timeout": "integer (optional): 300"
}
```

**Output:** Full JSON analysis report

**Typical Usage:**
```markdown
Call `benderbox_sandbox_analyzeModel`:
\`\`\`json
{
  "model_path": "<user_provided>",
  "profile": "standard"
}
\`\`\`
```

---

### `benderbox_sandbox_getLatestReport`

**Purpose:** Retrieve cached analysis

**Inputs:**
```json
{
  "log_dir": "string (optional): './sandbox_logs'",
  "model_name": "string (optional): 'model.gguf'"
}
```

**Output:** JSON report or `{"error": "No reports found"}`

**Typical Usage:**
```markdown
Check for cached report:
Call `benderbox_sandbox_getLatestReport` with `model_name`

If report found and recent (< 24h):
  - Use cached data
Else:
  - Run fresh analysis
```

---

### `benderbox_sandbox_listTests`

**Purpose:** Get available tests

**Inputs:**
```json
{}
```

**Output:**
```json
{
  "tests": [
    {"name": "test1", "category": "static"},
    {"name": "test2", "category": "dynamic"}
  ],
  "count": 2
}
```

**Typical Usage:**
```markdown
To show user available tests:
Call `benderbox_sandbox_listTests`
Display tests grouped by category
```

---

## JSON Report Schema Reference

Skills frequently parse JSON reports. Key fields:

### Top Level
- `run_id` (string): Unique analysis ID
- `timestamp_utc` (string): ISO 8601 timestamp
- `profile` (string): Profile used
- `schema_version` (string): Report schema version

### `model` Object
- `model.name` (string): Filename
- `model.path` (string): Full path
- `model.size_bytes` (integer): File size
- `model.fingerprint` (string): SHA256 hash
- `model.metadata` (object): GGUF metadata

### `model.metadata` Object
- `architecture` (string): "llama", "mistral", etc.
- `parameter_count` (string): "7B", "13B", etc.
- `quantization` (string): "Q4_K_S", etc.
- `quantization_bits` (integer): 4, 8, etc.
- `context_length` (integer): Max context tokens
- `layers` (integer): Number of layers
- `embedding_length` (integer): Embedding dimension
- `vocab_size` (integer): Vocabulary size

### `overall_risk` Object
- `level` (string): "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
- `score` (float): 0-100
- `primary_factors` (array): ["factor1", "factor2"]

### `safety` Object
- `jailbreak_success_rate` (float): 0.0-1.0
- `jailbreak_attempts` (integer): Total prompts tested
- `jailbreak_successes` (integer): Successful jailbreaks

### `capability_fingerprint` Object
- `reasoning` (integer): 0-10
- `code_generation` (integer): 0-10
- `multi_step_planning` (integer): 0-10
- `tool_usage_risk` (integer): 0-10
- `safety_evasion_tendency` (integer): 0-10

### `tests` Array
Each test object:
- `name` (string): Test name
- `category` (string): "static" | "dynamic" | "security"
- `status` (string): "PASS" | "FAIL" | "WARN" | "ERROR" | "SKIP"
- `severity` (string): "INFO" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
- `details` (string): Human-readable summary
- `metrics` (object): Structured metrics
- `artifacts` (array): Large outputs (JSON-serialized)

---

## Common Patterns

### Pattern 1: Cached Analysis

```markdown
### 1. Check Cache
Call `benderbox_sandbox_getLatestReport`:
\`\`\`json
{
  "model_name": "[model_filename]"
}
\`\`\`

If report found and `timestamp_utc` < 24 hours ago:
  - Ask user: "Found analysis from [X] hours ago. Use cached? (y/n)"
  - If yes, skip to step 3

### 2. Run Fresh Analysis
[Proceed with analysis call]

### 3. Format Results
[Use cached or fresh data]
```

### Pattern 2: Progressive Detail

```markdown
### 1. Show Quick Summary
[Essential info: name, size, risk level]

### 2. Offer Details
Ask: "View detailed findings? (y/n)"

If yes:
  - Show full test results
  - Show violation examples
  - Show recommendations
```

### Pattern 3: Conditional Sections

```markdown
## Output Format

\`\`\`markdown
[Always show basic info]

[If overall_risk.level == "HIGH" or "CRITICAL":]
⚠️ **Warning:** This model shows significant safety concerns

[If safety.jailbreak_successes > 0:]
### Jailbreak Violations Detected
[Show violation details]

[If no issues:]
✅ **All tests passed**
\`\`\`
```

---

## Validation Checklist

Before submitting a skill, verify:

- [ ] All required sections present
- [ ] Purpose is 1-3 sentences
- [ ] Triggers are specific and quoted
- [ ] Required tools use exact names (backtick-formatted)
- [ ] Inputs specify type and required/optional
- [ ] Process flow includes JSON payloads for tool calls
- [ ] Output format provides complete template with placeholders
- [ ] Error handling covers common failures
- [ ] Constraints include NEVER/ALWAYS lists
- [ ] Metadata includes version and category
- [ ] No implementation logic (only declarative)
- [ ] No direct file parsing or Python code execution

---

## Anti-Patterns

### ❌ Implementing Logic

**Bad:**
```markdown
## Process Flow
1. Open the GGUF file
2. Read bytes 0-8 to check magic number
3. Parse metadata section
4. Extract architecture field
```

**Good:**
```markdown
## Process Flow
1. Call `benderbox_sandbox_analyzeModel` with `profile="quick"`
2. Extract `model.metadata.architecture` from JSON response
3. Display: "Architecture: [architecture]"
```

---

### ❌ Vague Triggers

**Bad:**
```markdown
## Triggers
- When the user wants to know about a model
```

**Good:**
```markdown
## Triggers
- User asks: "What's inside this model?"
- User asks: "Analyze model.gguf"
- User provides: Path to GGUF file
```

---

### ❌ Missing Error Handling

**Bad:**
```markdown
[No error handling section]
```

**Good:**
```markdown
## Error Handling

### If model file not found:
❌ **Error:** Model file not found at `[model_path]`

**Suggestions:** [...]
```

---

### ❌ Incomplete Output Format

**Bad:**
```markdown
## Output Format
Show the model name and risk score
```

**Good:**
```markdown
## Output Format

\`\`\`markdown
# Model Analysis: [model.name]

**Risk:** [overall_risk.level] ([overall_risk.score]/100)
[Complete template with all sections]
\`\`\`
```

---

## Testing Your Skill

### Manual Testing

1. **Read aloud** - Does it make sense to a new user?
2. **Follow the process** - Can you execute each step?
3. **Check completeness** - Are all sections present?
4. **Validate examples** - Do example interactions work?

### Checklist Testing

- [ ] Can an agent determine WHEN to invoke this skill? (Triggers)
- [ ] Can an agent gather all required inputs? (Inputs)
- [ ] Can an agent execute each process step? (Process Flow)
- [ ] Can an agent format the output? (Output Format)
- [ ] Can an agent handle errors? (Error Handling)

---

## Version Control

Skills should be versioned:

```markdown
## Metadata
- **Version:** 1.2.0
```

### Version Bumps

- **Major (X.0.0):** Breaking changes (incompatible inputs/outputs)
- **Minor (1.X.0):** New features (new optional inputs, sections)
- **Patch (1.2.X):** Bug fixes, clarifications, typos

### Changelog

Include at end of skill:

```markdown
## Changelog

### 1.2.0 (2025-12-10)
- Added caching support
- New input: force_reanalysis

### 1.1.0 (2025-12-01)
- Added error handling section
- Improved output template

### 1.0.0 (2025-11-15)
- Initial release
```

---

## Resources

- **Architecture Docs:** `./docs/ARCHITECTURE.md`
- **MCP Tool Reference:** `./docs/MCP_TOOLS.md`
- **Example Skills:** `./skills/`
- **JSON Schema:** See `benderbox_sandbox_cli.py:SandboxRunResult`

---

*For questions about skill authoring, see the main README or open an issue.*
