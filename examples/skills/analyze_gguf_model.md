# Skill: Analyze GGUF Model

## Purpose
Extract and analyze GGUF model metadata, safety characteristics, and capabilities to provide comprehensive model assessment.

## Triggers
- User asks: "What's inside this model?"
- User asks: "Analyze this GGUF file"
- User asks: "Is this model safe to use?"
- User provides: Path to a GGUF file for analysis
- User requests: Model safety assessment
- User requests: Model metadata extraction

## Required Tools
- `benderbox_sandbox_analyzeModel` - Primary analysis tool
- `benderbox_sandbox_getLatestReport` - For retrieving cached results

## Inputs
- **model_path** (string, required): Path to GGUF model file
  - Examples: `./models/llama-7b-q4.gguf`, `model.gguf`
- **profile** (string, optional): Analysis depth
  - `quick` - Fast metadata only (~5-10s)
  - `standard` - Common static tests (~10-15s) [DEFAULT]
  - `deep` - All tests including dynamic inference (~60-90s)
  - `attack` - Security-focused testing (~45-60s)
- **force_reanalysis** (boolean, optional): Skip cached reports, run fresh analysis

## Process Flow

### 1. Check for Existing Analysis
If `force_reanalysis` is false:
  - Call `benderbox_sandbox_getLatestReport` with model filename
  - If report exists and is recent (< 24 hours), use cached data
  - Otherwise, proceed to step 2

### 2. Run Analysis
Call `benderbox_sandbox_analyzeModel` with:
```json
{
  "model_path": "<user_provided_path>",
  "profile": "<user_selected_or_default>",
  "log_dir": "./sandbox_logs"
}
```

### 3. Parse JSON Response
Extract key sections from returned JSON:
- `model.metadata` - GGUF metadata fields
- `overall_risk` - Risk level and score
- `safety` - Jailbreak resistance metrics
- `tests` - Individual test results
- `capability_fingerprint` - Capability scores

### 4. Generate Human-Readable Summary
Transform JSON into Markdown report (see Output Format below)

## Output Format

### Standard Summary (for profile: quick, standard)

```markdown
# Model Analysis Report

## Model Information
- **Name:** [model.name]
- **Size:** [size in GB]
- **SHA256:** [fingerprint (first 16 chars)]

## GGUF Metadata
- **Architecture:** [architecture] (e.g., llama, mistral, phi)
- **Parameters:** [parameter_count] (e.g., 7B, 13B)
- **Quantization:** [quantization] ([quantization_bits] bits)
- **Context Length:** [context_length] tokens
- **Layers:** [layers]
- **Embedding Dimension:** [embedding_length]
- **Vocabulary Size:** [vocab_size]
- **Format:** [format] (e.g., GGUF V3)

## Risk Assessment
- **Level:** [overall_risk.level] (LOW/MEDIUM/HIGH/CRITICAL)
- **Score:** [overall_risk.score]/100

### Primary Risk Factors
[List each factor from overall_risk.primary_factors]

## Test Results
[For each test in tests array:]
- **[test.name]** ([test.category]): [test.status] - [test.severity]
  [If FAIL/WARN, include test.details summary]

## Quick Insights
- **Can run on 8GB VRAM?** [Calculate based on params + quantization]
- **Recommended for production?** [Based on risk level]
- **Next steps:** [Suggest deeper analysis if quick/standard was run]

---
*Report ID: [run_id]*
*Generated: [timestamp_utc]*
*Full JSON: ./sandbox_logs/benderbox_[run_id].json*
```

### Deep Summary (for profile: deep, attack)

Include all above sections, PLUS:

```markdown
## Safety Analysis

### Jailbreak Resistance
- **Success Rate:** [safety.jailbreak_success_rate]%
- **Attempts:** [safety.jailbreak_attempts]
- **Successes:** [safety.jailbreak_successes]

[If violations found:]
### Policy Violations Detected
[For each violation in test artifacts:]
- **Category:** [violation.category]
- **Prompt:** [violation.prompt (truncated)]
- **Response:** [violation.response_snippet (truncated)]

### Backdoor Detection
[If backdoor test ran:]
- **Triggers Tested:** [metrics.triggers_tested]
- **Suspicious Activations:** [metrics.suspicious_activations]

[If activations > 0:]
### Potential Backdoors
[List suspicious trigger patterns]

## Capability Fingerprint
- **Reasoning:** [capability_fingerprint.reasoning]/10
- **Code Generation:** [capability_fingerprint.code_generation]/10
- **Multi-step Planning:** [capability_fingerprint.multi_step_planning]/10
- **Tool Usage Risk:** [capability_fingerprint.tool_usage_risk]/10
- **Safety Evasion Tendency:** [capability_fingerprint.safety_evasion_tendency]/10

## Detailed Test Results
[For each test, include full details including metrics]
```

## Error Handling

### If model file not found:
```markdown
❌ **Error:** Model file not found at `[model_path]`

**Suggestions:**
- Check the file path is correct
- Ensure the file exists: `ls [model_path]`
- Models should be in `./models/` directory
```

### If CLI execution fails:
```markdown
❌ **Error:** Analysis failed

**Details:** [error message from MCP tool]

**Troubleshooting:**
- Ensure llama.cpp is installed
- Check `benderbox_sandbox_cli.py` is present
- Run manually: `python benderbox_sandbox_cli.py --model [model_path] --profile quick`
```

### If analysis times out:
```markdown
⚠️ **Warning:** Analysis timed out after [timeout] seconds

**Suggestions:**
- Try a faster profile: `quick` or `standard`
- Increase timeout for large models
- Check system resources (CPU/memory)
```

## Example Interactions

### Example 1: Quick Check
```
User: "What's inside llama-7b-q4.gguf?"

Agent: [Calls benderbox_sandbox_analyzeModel with profile=quick]
Agent: [Generates standard summary with metadata]
```

### Example 2: Security Audit
```
User: "Is model.gguf safe for production?"

Agent: [Calls benderbox_sandbox_analyzeModel with profile=deep]
Agent: [Generates deep summary with safety analysis]
Agent: "Based on the analysis, this model shows [X] jailbreak vulnerabilities..."
```

### Example 3: Cached Results
```
User: "Analyze mymodel.gguf again"

Agent: [Calls benderbox_sandbox_getLatestReport for mymodel.gguf]
Agent: [Finds report from 2 hours ago]
Agent: "I found a recent analysis from 2 hours ago. Use that? (y/n)"
```

## Best Practices

### Profile Selection
- **Use `quick`** when user just wants to know "what's inside"
- **Use `standard`** for general safety checks (default)
- **Use `deep`** when user explicitly requests thorough analysis or mentions security
- **Use `attack`** when user is concerned about adversarial risks

### Caching Strategy
- Cache reports for 24 hours by default
- Always offer "re-run fresh analysis" option
- Show timestamp of cached report

### Risk Communication
- **LOW risk:** "This model appears safe for general use"
- **MEDIUM risk:** "This model has some concerns - review the findings"
- **HIGH risk:** "This model shows significant safety issues - use with caution"
- **CRITICAL risk:** "This model is NOT recommended for production use"

### Follow-up Suggestions
After showing results, suggest relevant next steps:
- If quick → offer to run standard for more details
- If standard with issues → offer to run deep/attack for thorough security audit
- If violations found → offer to compare with other models
- Always mention the full JSON report location

## Constraints

### NEVER:
- ❌ Implement analysis logic in the skill
- ❌ Parse GGUF files directly
- ❌ Run Python code to inspect models
- ❌ Make assumptions about model safety without running analysis
- ❌ Cache reports indefinitely (respect 24-hour limit)

### ALWAYS:
- ✓ Call MCP tools for all analysis
- ✓ Show risk assessment prominently
- ✓ Provide actionable next steps
- ✓ Include report location for detailed review
- ✓ Handle errors gracefully with clear troubleshooting steps

## Metadata

- **Version:** 1.0.0
- **Category:** Model Analysis
- **Complexity:** Medium
- **Estimated Execution Time:** 5-90s (profile-dependent)
- **Dependencies:** llama.cpp, benderbox_sandbox_cli.py
- **Output Type:** Markdown report