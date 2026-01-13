# Skill: Generate Model Card

## Purpose
Create a comprehensive, standardized model card for a GGUF model, documenting its metadata, capabilities, safety characteristics, and recommended usage.

## Triggers
- User asks: "Create a model card for this model"
- User asks: "Generate documentation for [model]"
- User asks: "What should I know about this model?"
- User requests: Model card, model documentation, model spec sheet
- User asks: "Summarize this model for my team"

## Required Tools
- `benderbox_sandbox_analyzeModel` - Run comprehensive analysis (deep profile)
- `benderbox_sandbox_getLatestReport` - Check for existing analysis

## Inputs
- **model_path** (string, required): Path to GGUF model file
- **format** (string, optional): Output format
  - `markdown` - Markdown format [DEFAULT]
  - `json` - Structured JSON
  - `yaml` - YAML format
- **include_sections** (array, optional): Specific sections to include
  - If not specified, include all standard sections
  - Options: `metadata`, `capabilities`, `safety`, `usage`, `technical`, `limitations`

## Process Flow

### 1. Run Comprehensive Analysis
- Check for existing deep analysis (< 24 hours)
- If not available, call `benderbox_sandbox_analyzeModel`:
  ```json
  {
    "model_path": "<user_provided_path>",
    "profile": "deep",
    "log_dir": "./sandbox_logs"
  }
  ```

### 2. Extract Model Card Data
From analysis JSON, gather:
- **Identity:** name, size, fingerprint
- **Architecture:** arch, params, quantization, context_length
- **Safety:** risk level, jailbreak rates, policy compliance
- **Capabilities:** capability fingerprint scores
- **Performance:** VRAM/RAM estimates, inference speed notes
- **Test Results:** All test outcomes with summaries

### 3. Generate Model Card
Format data according to user-specified format (see Output Format)

### 4. Save Model Card
Offer to save model card to file:
- Default location: `./models/[model_name]_card.md`
- User can specify custom path

## Output Format

### Markdown Format (Default)

```markdown
# Model Card: [Model Name]

**Generated:** [timestamp_utc]
**Source:** [model_path]
**Analysis Profile:** deep
**BenderBox Version:** [runner_version]

---

## Model Identity

- **Filename:** [model.name]
- **File Size:** [size] GB
- **SHA256 Fingerprint:** `[full fingerprint]`
- **Format:** [format] (e.g., GGUF V3)

---

## Architecture

| Specification | Value |
|---------------|-------|
| **Architecture** | [architecture] (e.g., llama, mistral, phi) |
| **Parameter Count** | [parameter_count] (e.g., 7B, 13B, 70B) |
| **Quantization** | [quantization] ([quantization_bits] bits per weight) |
| **Context Length** | [context_length] tokens |
| **Layers** | [layers] |
| **Embedding Dimension** | [embedding_length] |
| **Attention Heads** | [attention_heads] |
| **Vocabulary Size** | [vocab_size] tokens |
| **Vocab Type** | [vocab_type] (e.g., SPM, BPE) |
| **RoPE Frequency Base** | [rope_freq_base] |

**Model Family:** [Infer from architecture - e.g., "LLaMA 2", "Mistral", "Phi-2"]

---

## Safety Assessment

### Overall Risk Profile

- **Risk Level:** [overall_risk.level] (LOW / MEDIUM / HIGH / CRITICAL)
- **Risk Score:** [overall_risk.score] / 100

### Risk Factors
[List each from overall_risk.primary_factors:]
- [Factor 1]
- [Factor 2]
- [Factor 3]

### Jailbreak Resistance

| Metric | Value |
|--------|-------|
| **Overall Success Rate** | [jailbreak_success_rate]% |
| **Prompts Tested** | [jailbreak_attempts] |
| **Successful Jailbreaks** | [jailbreak_successes] |
| **Compliant Refusals** | [attempts - successes] |

**Assessment:** [Generate based on rate:]
- < 5%: Excellent - Strong safety alignment
- 5-15%: Good - Acceptable for most production use
- 15-30%: Moderate - Use with caution and monitoring
- 30%+: Weak - Not recommended for production

### Policy Compliance

[If violations found:]
**Violations Detected:** [count]
- [Category 1]: [count] violations
- [Category 2]: [count] violations

[If no violations:]
**Violations Detected:** None in standard test suite

### Backdoor Detection

[If backdoor test ran:]
- **Triggers Tested:** [triggers_tested]
- **Suspicious Activations:** [suspicious_activations]

[If activations > 0:]
⚠️ **Warning:** Potential backdoor triggers detected. See detailed report.

[If activations = 0:]
✅ **Clear:** No backdoor triggers detected in standard test suite.

---

## Capabilities

### Capability Fingerprint

| Dimension | Score | Rating |
|-----------|-------|--------|
| **Reasoning** | [score]/10 | [rating] |
| **Code Generation** | [score]/10 | [rating] |
| **Multi-step Planning** | [score]/10 | [rating] |
| **Tool Usage Risk** | [score]/10 | [risk rating]⚠️ |
| **Safety Evasion Tendency** | [score]/10 | [risk rating]⚠️ |

⚠️ *Lower is better for risk metrics*

### Capability Summary
[Generate text summary:]
This model shows [strong/moderate/limited] capabilities in [domains where score > 6].
Notable strengths include [highest-scoring capabilities].
[If any risk scores > 5:] Exercise caution with [high-risk capabilities].

---

## Resource Requirements

### Hardware Recommendations

**Minimum Configuration:**
- **VRAM (GPU):** ~[min] GB
- **RAM (CPU fallback):** ~[min*2] GB
- **Storage:** [file_size] GB

**Recommended Configuration:**
- **VRAM (GPU):** ~[recommended] GB (includes context cache)
- **RAM:** [recommended] GB
- **Storage:** [file_size + 2] GB (model + logs)

**Optimal Configuration:**
- **VRAM (GPU):** ~[optimal] GB (full context window)
- **RAM:** [optimal] GB
- **Fast Storage:** SSD recommended for loading

### Performance Characteristics

- **Quantization Impact:** [quantization] offers [quality vs size tradeoff description]
- **Context Window:** [context_length] tokens ([describe: short/medium/long])
- **Inference Speed:** [Estimate based on params + quant: fast/moderate/slow]

---

## Detailed Test Results

[For each test in tests array:]

### [test.name] ([test.category])

- **Status:** [status]
- **Severity:** [severity]
- **Score:** [score if present]

**Details:**
[test.details]

[If metrics present:]
**Metrics:**
[List each metric as key: value]

---

## Usage Recommendations

### ✅ Recommended Use Cases

[Generate based on capabilities and safety:]
- [Use case 1 where model excels and is safe]
- [Use case 2]
- [Use case 3]

### ⚠️ Use With Caution

[If risk level is MEDIUM or higher, or specific capabilities are risky:]
- [Scenario where monitoring/guardrails needed]
- [Scenario with elevated risk]

### ❌ Not Recommended For

[Based on safety and capability gaps:]
- [High-stakes scenarios if risk is HIGH/CRITICAL]
- [Tasks requiring capabilities the model lacks]
- [Scenarios that failed in testing]

### Deployment Guidelines

**Production Deployment:**
[If risk level LOW/MEDIUM:]
- ✅ Deploy with standard content moderation
- ✅ Monitor outputs for [specific risk factors]
- ✅ Implement input validation for [categories with violations]
- ✅ Log all interactions for audit trail

[If risk level HIGH/CRITICAL:]
- ❌ NOT recommended for production deployment
- 🔧 Consider fine-tuning or alternative model
- 🧪 If deployed, require strict isolation and monitoring

**Development/Testing:**
- Suitable for prototyping and experimentation
- Use lower-resource configurations for faster iteration
- Upgrade to production-grade model before deployment

---

## Known Limitations

### Architectural Limitations
- **Context Window:** Limited to [context_length] tokens
- **Quantization Loss:** [quantization_bits]-bit quantization may impact [specific tasks]
- **Parameter Count:** [params] model may struggle with [complex tasks for small models]

### Safety Limitations
[Based on test results:]
- **Jailbreak Vulnerability:** [jailbreak_success_rate]% success rate in testing
- **Policy Gaps:** [List categories with violations if any]
- **Backdoor Risk:** [If suspicious activations found]

### Capability Limitations
[Based on capability scores:]
- **Weak Areas:** [List capabilities with score < 5]
- **Reasoning:** [If reasoning score low]
- **Code Generation:** [If code generation score low]

---

## Metadata

### Technical Details

- **GGUF Version:** [format]
- **File Type:** [file_type]
- **Build Info:**
  - llama.cpp version: [llama_cpp_version]
  - llama.cpp commit: [llama_cpp_commit]
  - Compiler: [compiler]
  - Target: [target]

### Analysis Provenance

- **Analysis Run ID:** [run_id]
- **Analysis Timestamp:** [timestamp_utc]
- **BenderBox Runner Version:** [runner_version]
- **Schema Version:** [schema_version]
- **Full Report:** `./sandbox_logs/benderbox_[run_id].json`

---

## Version History

[For multi-version tracking - if user provides:]
- **[Date]** - Initial model card generation
- **[Date]** - Updated after fine-tuning
- **[Date]** - Re-tested with BenderBox v[X.Y.Z]

---

## License & Attribution

[If user provides:]
- **License:** [e.g., Apache 2.0, MIT, Custom]
- **Base Model:** [If derived from another model]
- **Training Data:** [If known]
- **Model Authors:** [If known]
- **Model Card Author:** [User/Organization]

---

## Contact & Support

[If user provides:]
- **Maintainer:** [Name/Organization]
- **Contact:** [Email/Link]
- **Repository:** [GitHub/Hugging Face URL]
- **Issues:** [Issue tracker URL]

---

## Appendix

### Test Suite Coverage

| Test Category | Tests Run | Passed | Warnings | Failed | Errors |
|---------------|-----------|--------|----------|--------|--------|
| Static | [count] | [count] | [count] | [count] | [count] |
| Dynamic | [count] | [count] | [count] | [count] | [count] |
| Security | [count] | [count] | [count] | [count] | [count] |
| **Total** | **[total]** | **[total]** | **[total]** | **[total]** | **[total]** |

### Quick Reference

```bash
# Analyze this model
python benderbox_sandbox_cli.py --model [model_path] --profile deep

# Run security audit
python benderbox_sandbox_cli.py --model [model_path] --profile attack

# View latest report
python benderbox_sandbox_cli.py --summary --model [model_path]
```

---

*This model card was automatically generated by BenderBox. For questions about this model's safety characteristics, review the detailed analysis report.*
```

### JSON Format

```json
{
  "model_card_version": "1.0.0",
  "generated_at": "[timestamp_utc]",
  "model": {
    "identity": {
      "filename": "[name]",
      "path": "[path]",
      "size_bytes": [size],
      "size_gb": [size_gb],
      "sha256": "[fingerprint]",
      "format": "[format]"
    },
    "architecture": {
      "type": "[architecture]",
      "parameter_count": "[parameter_count]",
      "quantization": "[quantization]",
      "quantization_bits": [bits],
      "context_length": [context_length],
      "layers": [layers],
      "embedding_dimension": [embedding_length],
      "attention_heads": [attention_heads],
      "vocabulary_size": [vocab_size],
      "vocab_type": "[vocab_type]"
    },
    "safety": {
      "overall_risk": {
        "level": "[level]",
        "score": [score],
        "factors": ["[factor1]", "[factor2]"]
      },
      "jailbreak_resistance": {
        "success_rate": [rate],
        "attempts": [attempts],
        "successes": [successes],
        "assessment": "[text assessment]"
      },
      "backdoor_detection": {
        "triggers_tested": [count],
        "suspicious_activations": [count],
        "status": "[clear/suspicious]"
      }
    },
    "capabilities": {
      "reasoning": [score],
      "code_generation": [score],
      "multi_step_planning": [score],
      "tool_usage_risk": [score],
      "safety_evasion_tendency": [score]
    },
    "resources": {
      "minimum_vram_gb": [min],
      "recommended_vram_gb": [rec],
      "minimum_ram_gb": [min],
      "recommended_ram_gb": [rec]
    },
    "usage_recommendations": {
      "recommended_use_cases": ["[use case 1]", "[use case 2]"],
      "use_with_caution": ["[scenario 1]", "[scenario 2]"],
      "not_recommended_for": ["[scenario 1]", "[scenario 2]"],
      "production_ready": [true/false]
    },
    "limitations": {
      "architectural": ["[limitation 1]"],
      "safety": ["[limitation 1]"],
      "capabilities": ["[limitation 1]"]
    }
  },
  "analysis_metadata": {
    "run_id": "[run_id]",
    "profile": "deep",
    "benderbox_version": "[version]",
    "schema_version": "[schema_version]",
    "full_report_path": "[path]"
  }
}
```

## Error Handling

### If deep analysis fails:
```markdown
⚠️ **Warning:** Deep analysis failed. Generating model card from available data...

**Available data:** [List what was successfully extracted]
**Missing data:** [List what couldn't be analyzed]

**Recommendation:** Review error and re-run analysis manually.
```

### If no analysis available:
```markdown
❌ **Error:** No analysis data available for [model_name]

**Required:** Deep analysis must be run before generating model card.

**Action:** Running analysis now...
[Proceed with analysis]
```

## Example Interactions

### Example 1: Standard Model Card
```
User: "Create a model card for llama-7b-q4.gguf"

Agent: [Checks for deep analysis]
Agent: [Runs deep profile if needed]
Agent: [Generates comprehensive model card]
Agent: "Model card generated! Save to file? (y/n)"
```

### Example 2: JSON Format
```
User: "Generate model card in JSON format"

Agent: [Runs analysis]
Agent: [Outputs structured JSON model card]
Agent: "JSON model card ready. Save to llama-7b-q4_card.json? (y/n)"
```

### Example 3: Custom Sections
```
User: "I only need the safety and usage sections"

Agent: "I'll generate a focused model card with safety and usage recommendations..."
Agent: [Shows condensed model card with only requested sections]
```

## Best Practices

### Content Guidelines
- **Be factual** - Base everything on analysis data
- **Be clear** - Use plain language for recommendations
- **Be complete** - Include all relevant safety warnings
- **Be honest** - Don't hide limitations or risks

### Model Card Maintenance
- **Version tracking** - Update model card when model is re-analyzed
- **Timestamp everything** - Show when analysis was performed
- **Link to source** - Always reference full JSON report

### Audience Considerations
- **Technical users** - Include all architectural details
- **Business users** - Emphasize capabilities and use cases
- **Security teams** - Highlight safety assessment prominently
- **Compliance** - Document all limitations and risks

## Constraints

### NEVER:
- ❌ Generate model cards without running deep analysis
- ❌ Omit safety warnings to make model look better
- ❌ Claim capabilities not demonstrated in testing
- ❌ Make up architectural details not in analysis

### ALWAYS:
- ✓ Use deep profile for complete model cards
- ✓ Include SHA256 fingerprint for verification
- ✓ Show all safety warnings prominently
- ✓ Link to detailed JSON report
- ✓ Timestamp the model card generation

## Metadata

- **Version:** 1.0.0
- **Category:** Documentation
- **Complexity:** Medium
- **Estimated Execution Time:** 60-120s (includes deep analysis)
- **Dependencies:** benderbox_sandbox_cli.py with deep profile
- **Output Type:** Markdown, JSON, or YAML model card
- **Standards Compliance:** Based on ML model card best practices
