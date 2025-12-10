# Skill: Compare Model Safety

## Purpose
Compare safety characteristics, capabilities, and metadata of two or more GGUF models to help users make informed selection decisions.

## Triggers
- User asks: "Which model is safer?"
- User asks: "Compare modelA.gguf and modelB.gguf"
- User asks: "What's the difference between these models?"
- User provides: Multiple model paths for comparison
- User asks: "Which should I use for production?"

## Required Tools
- `aegisml_sandbox_analyzeModel` - Analyze each model
- `aegisml_sandbox_getLatestReport` - Check for existing reports

## Inputs
- **model_paths** (array of strings, required): List of GGUF model paths to compare
  - Minimum: 2 models
  - Maximum: 5 models (to keep output readable)
- **comparison_focus** (string, optional): What to prioritize
  - `safety` - Focus on risk scores and jailbreak resistance [DEFAULT]
  - `capabilities` - Focus on capability fingerprints
  - `metadata` - Focus on architecture and quantization
  - `all` - Comprehensive comparison across all dimensions
- **profile** (string, optional): Analysis profile for each model
  - `standard` - Standard analysis [DEFAULT]
  - `deep` - Comprehensive testing (slower)

## Process Flow

### 1. Validate Inputs
- Ensure at least 2 models provided
- If > 5 models, warn: "Comparing more than 5 models at once may be difficult to read. Proceed? (y/n)"
- Check all model files exist

### 2. Gather Analysis Reports
For each model:
  a. Check for recent report (< 24 hours): `aegisml_sandbox_getLatestReport`
  b. If no recent report or user wants fresh data:
     - Call `aegisml_sandbox_analyzeModel` with specified profile
     - Show progress: "Analyzing model 2/3..."

### 3. Extract Comparison Metrics
From each model's report, extract:
- **Metadata:** name, size, architecture, parameters, quantization, context_length
- **Safety:** overall_risk.level, overall_risk.score, jailbreak_success_rate
- **Capabilities:** capability_fingerprint scores
- **Test Results:** Pass/Fail/Warn counts by category

### 4. Generate Comparison Table
Create side-by-side comparison (see Output Format)

### 5. Provide Recommendation
Based on comparison_focus, recommend best model with justification

## Output Format

### Comparison Header

```markdown
# Model Safety Comparison

**Models Compared:** [count]
**Analysis Date:** [timestamp]
**Profile:** [profile used]
**Focus:** [comparison_focus]

---
```

### Quick Comparison Table

```markdown
## 📊 Quick Comparison

| Metric | [Model 1 Name] | [Model 2 Name] | [Model 3 Name] | Winner |
|--------|----------------|----------------|----------------|---------|
| **File Size** | [X] GB | [Y] GB | [Z] GB | [Smallest/Largest depending on context] |
| **Architecture** | [arch] | [arch] | [arch] | - |
| **Parameters** | [params] | [params] | [params] | [Largest] |
| **Quantization** | [quant] | [quant] | [quant] | [Best quality] |
| **Context Length** | [ctx] | [ctx] | [ctx] | [Longest] |
| **Risk Level** | [level] | [level] | [level] | ✅ [Lowest] |
| **Risk Score** | [score]/100 | [score]/100 | [score]/100 | ✅ [Lowest] |
| **Jailbreak Rate** | [rate]% | [rate]% | [rate]% | ✅ [Lowest] |

### Legend
- ✅ = Best in category
- ⚠️ = Concerning metric
- ❌ = Worst in category / Not recommended
```

### Detailed Safety Comparison (if comparison_focus = 'safety' or 'all')

```markdown
## 🛡️ Safety Analysis

### Risk Assessment
[For each model, create a block:]

#### [Model Name]
- **Overall Risk:** [level] ([score]/100)
- **Primary Risk Factors:**
  [List each factor from overall_risk.primary_factors]
- **Jailbreak Resistance:** [jailbreak_success_rate]%
  - Attempts: [jailbreak_attempts]
  - Successes: [jailbreak_successes]
- **Test Results:**
  - ✅ Passed: [count]
  - ⚠️ Warnings: [count]
  - ❌ Failed: [count]
  - 🔧 Errors: [count]

---

### Safety Ranking

1. 🥇 **[Safest Model Name]** - [Risk Level] ([Score]/100)
   - Why: [Justification based on metrics]

2. 🥈 **[Second Model Name]** - [Risk Level] ([Score]/100)
   - Notable: [Key differentiator from #1]

3. 🥉 **[Third Model Name]** - [Risk Level] ([Score]/100)
   - Concerns: [Why it ranks lower]
```

### Capabilities Comparison (if comparison_focus = 'capabilities' or 'all')

```markdown
## 🧠 Capability Fingerprint

| Capability | [Model 1] | [Model 2] | [Model 3] | Best |
|------------|-----------|-----------|-----------|------|
| Reasoning | [score]/10 | [score]/10 | [score]/10 | [name] |
| Code Generation | [score]/10 | [score]/10 | [score]/10 | [name] |
| Multi-step Planning | [score]/10 | [score]/10 | [score]/10 | [name] |
| Tool Usage Risk | [score]/10 | [score]/10 | [score]/10 | [name]⚠️ |
| Safety Evasion Tendency | [score]/10 | [score]/10 | [score]/10 | [name]⚠️ |

⚠️ *Lower is better for risk metrics*

### Capability Insights
[Generate insights based on scores:]
- **Best for reasoning tasks:** [Model name] ([score]/10)
- **Best for code generation:** [Model name] ([score]/10)
- **Lowest tool usage risk:** [Model name] ([score]/10)
```

### Metadata Comparison (if comparison_focus = 'metadata' or 'all')

```markdown
## 📋 Technical Specifications

### Architecture Details

| Spec | [Model 1] | [Model 2] | [Model 3] |
|------|-----------|-----------|-----------|
| Architecture | [arch] | [arch] | [arch] |
| Parameter Count | [params] | [params] | [params] |
| Quantization | [quant] ([bits] bits) | [quant] ([bits] bits) | [quant] ([bits] bits) |
| Context Length | [ctx] tokens | [ctx] tokens | [ctx] tokens |
| Layers | [layers] | [layers] | [layers] |
| Embedding Dim | [embd] | [embd] | [embd] |
| Vocab Size | [vocab] | [vocab] | [vocab] |
| File Format | [format] | [format] | [format] |
| File Size | [size] GB | [size] GB | [size] GB |

### Resource Requirements (Estimated)

| Model | VRAM (GPU) | RAM (CPU) | Storage |
|-------|------------|-----------|---------|
| [Model 1] | ~[X] GB | ~[Y] GB | [Z] GB |
| [Model 2] | ~[X] GB | ~[Y] GB | [Z] GB |
| [Model 3] | ~[X] GB | ~[Y] GB | [Z] GB |

*VRAM estimates based on quantization and parameter count*
*Add ~2GB overhead for context and KV cache*
```

### Recommendation Section

```markdown
## 💡 Recommendation

[Generate based on comparison_focus:]

### If focus = 'safety':

**For Production Use: [Model Name]**

**Why:**
- ✅ Lowest risk score ([score]/100 vs [other scores])
- ✅ Best jailbreak resistance ([rate]% vs [other rates])
- ✅ [Additional safety factors]

**Runner-up:** [Second Model]
- Good alternative if you need [capability/feature that it excels at]

**Avoid for production:** [Worst Model]
- ⚠️ Risk score: [score]/100 (MEDIUM/HIGH/CRITICAL)
- ⚠️ Jailbreak rate: [rate]% (concerning)

### If focus = 'capabilities':

**For [Specific Task]: [Model Name]**

**Why:**
- ✅ Highest [relevant capability] score
- ✅ Good balance of [other capabilities]

**Trade-off:** This model has [slightly higher risk / larger size / etc.]
**Alternative:** If safety is critical, consider [safer model] despite lower capability scores

### If focus = 'metadata':

**Best Balance: [Model Name]**

**Why:**
- ✅ Good parameter count ([params])
- ✅ Optimal quantization ([quant] - good quality/size ratio)
- ✅ Long context ([ctx] tokens)
- ✅ Reasonable size ([X] GB)

**For Limited Resources:** [Smallest Model]
- Runs on [X] GB VRAM
- Good for development/testing

**For Best Quality:** [Largest/Best Quantized Model]
- Highest parameter count or best quantization
- Requires [X] GB VRAM

### If focus = 'all':

**Overall Winner: [Model Name]**

**Scores:**
- Safety: [score/rating]
- Capabilities: [score/rating]
- Resource Efficiency: [rating]

**Why this model:**
[Balanced justification covering all dimensions]

**When to choose alternatives:**
- Choose [Model 2] if: [Specific use case or constraint]
- Choose [Model 3] if: [Specific use case or constraint]
```

### Usage Recommendations

```markdown
## 🎯 Use Case Recommendations

### Production Deployment
**Recommended:** [Safest model with acceptable capabilities]
- Deploy with standard content moderation
- Monitor for [specific risk factors if any]

### Development/Testing
**Recommended:** [Smallest/fastest model]
- Faster iteration
- Lower resource usage
- Upgrade to production model before deployment

### Research/Experimentation
**Recommended:** [Model with best capabilities]
- Safety monitoring required
- Isolated environment recommended
- Document all findings

### High-Stakes Applications (Finance, Healthcare, Legal)
**Recommended:** [Safest model only]
- Additional safety layer required
- Human review for critical outputs
- Regular security audits

---

## 📁 Detailed Reports

[For each model, provide link to full JSON:]
- **[Model 1 Name]:** `./sandbox_logs/aegisml_[run_id].json`
- **[Model 2 Name]:** `./sandbox_logs/aegisml_[run_id].json`
- **[Model 3 Name]:** `./sandbox_logs/aegisml_[run_id].json`
```

## Error Handling

### If one model fails analysis:
```markdown
⚠️ **Warning:** Analysis failed for [model_name]

**Error:** [error message]

**Comparison Status:**
- ✅ [Model 1]: Analysis complete
- ❌ [Model 2]: Analysis failed
- ✅ [Model 3]: Analysis complete

**Proceeding with 2-model comparison...**

[Show comparison for successful models only]
```

### If all models fail:
```markdown
❌ **Error:** Unable to analyze any of the provided models

**Models attempted:**
[List each model and its error]

**Troubleshooting:**
1. Verify all model paths are correct
2. Ensure models are valid GGUF files
3. Check llama.cpp is installed
4. Try analyzing one model individually first
```

### If < 2 models provided:
```markdown
❌ **Error:** Comparison requires at least 2 models

**Provided:** [count] model(s)

**Usage:**
Please provide 2-5 model paths:
- "Compare modelA.gguf and modelB.gguf"
- "Compare all models in ./models/"
```

## Example Interactions

### Example 1: Safety-Focused Comparison
```
User: "Which is safer: llama-7b-q4.gguf or mistral-7b-q4.gguf?"

Agent: [Analyzes both models with standard profile]
Agent: [Generates comparison with focus=safety]
Agent: "Based on safety analysis:
  - Llama-7B: Risk score 15/100 (LOW), Jailbreak rate 8%
  - Mistral-7B: Risk score 22/100 (MEDIUM), Jailbreak rate 14%

  Recommendation: Llama-7B is safer for production use."
```

### Example 2: Multi-Model Comparison
```
User: "Compare all models in ./models/ directory"

Agent: [Finds 4 GGUF files]
Agent: [Analyzes all with standard profile]
Agent: [Shows comprehensive comparison table]
Agent: "Comparison complete for 4 models. Here's the breakdown..."
```

### Example 3: Capability-Focused
```
User: "Which model is best for code generation?"

Agent: "I'll analyze and compare models focusing on code generation capabilities..."
Agent: [Runs deep profile for capability assessment]
Agent: [Shows capability fingerprint comparison]
Agent: "For code generation: [Model X] scores 8/10 vs [Model Y] at 6/10"
```

## Best Practices

### Analysis Strategy
- **Reuse recent reports** (< 24 hours) to speed up comparison
- **Use consistent profile** across all models for fair comparison
- **Limit to 5 models** maximum for readability

### Recommendation Logic
- **Safety-first default** - Always highlight safety concerns prominently
- **Context-aware** - Adjust recommendations based on user's stated use case
- **Transparent trade-offs** - Clearly state when choosing model X means sacrificing Y

### Performance Optimization
- **Parallel analysis** - If MCP supports it, analyze models concurrently
- **Progressive results** - Show comparison table as results come in
- **Cache aggressively** - 24-hour cache for reports

## Constraints

### NEVER:
- ❌ Implement comparison logic directly in skill
- ❌ Parse model files yourself
- ❌ Recommend unsafe model for production without clear warnings
- ❌ Compare models with different analysis profiles (unfair comparison)

### ALWAYS:
- ✓ Use MCP tools for all analysis
- ✓ Show clear winner/ranking for user's focus area
- ✓ Provide context for recommendations
- ✓ Link to detailed JSON reports
- ✓ Warn if any model shows HIGH or CRITICAL risk

## Metadata

- **Version:** 1.0.0
- **Category:** Model Comparison
- **Complexity:** High
- **Estimated Execution Time:** 20-180s (depends on model count and profile)
- **Dependencies:** llama.cpp, aegisml_sandbox_cli.py
- **Output Type:** Markdown comparison report
