# Skill: Compare Infrastructure Security

## Purpose
Compare security characteristics of multiple AI infrastructure components (MCP servers, skills) side-by-side to help select the safest option or identify relative security strengths and weaknesses.

## Triggers
- User asks: "Which MCP server is safer?"
- User asks: "Compare the security of these servers"
- User asks: "Which skill should I use?"
- User provides: Multiple infrastructure file paths for comparison
- User requests: Security comparison between components
- User requests: "Help me choose between X and Y"

## Required Tools
- `benderbox_sandbox_analyzeModel` - Analysis tool (v2.0 infrastructure support)
- `benderbox_sandbox_getLatestReport` - For retrieving cached results

## Inputs
- **component_paths** (array of strings, required): Paths to infrastructure files to compare
  - Examples: `["server1.py", "server2.py"]`, `["skill_a.md", "skill_b.md", "skill_c.md"]`
  - **Minimum:** 2 components
  - **Maximum:** 5 components (for readability)
- **profile** (string, optional): Analysis depth
  - `infra-quick` - Fast comparison (~10-30s total)
  - `infra-standard` - Standard security comparison (~30-90s total) [DEFAULT]
  - `infra-deep` - Comprehensive comparison (~2-10min total)
- **force_reanalysis** (boolean, optional): Skip cached reports, run fresh analysis for all

## Process Flow

### 1. Validate Inputs
- Ensure 2-5 components provided
- Verify all files exist
- Check all files are same type (all .py OR all .md)
- If mixed types, inform user and ask for clarification

### 2. Analyze Each Component
For each component in component_paths:
  a. Check cache: Call `benderbox_sandbox_getLatestReport`
  b. If not cached or force_reanalysis=true:
     - Determine file type (.py = MCP server, .md = skill)
     - Call `benderbox_sandbox_analyzeModel` with appropriate path parameter:
       - For .py: `{"mcp_server_path": path, "profile": profile}`
       - For .md: `{"skill_path": path, "profile": profile}`
  c. Store analysis result

### 3. Extract Comparison Metrics
For each analyzed component, extract:
- **Filename** and **size**
- **Risk level** (LOW/MEDIUM/HIGH/CRITICAL)
- **Risk score** (0-100)
- **Findings by severity** (CRITICAL, HIGH, MEDIUM, LOW counts)
- **Specific vulnerability categories** (injection, exfiltration, etc.)
- **Test results** (PASS/FAIL/WARN counts)

### 4. Determine Winner
- **Lowest risk score** = Safest
- **Fewest CRITICAL/HIGH findings** = Secondary criteria
- **Highest test pass rate** = Tertiary criteria

### 5. Generate Comparison Report
Transform metrics into side-by-side comparison (see Output Format below)

## Output Format

### Comparison Table (for 2-3 components)

```markdown
# Infrastructure Security Comparison

**Analyzed:** [timestamp_utc]
**Profile:** [profile]
**Components:** [count]

## Summary

| Component | Risk Level | Risk Score | Critical | High | Medium | Low | Winner |
|-----------|------------|------------|----------|------|--------|-----|--------|
| [name1] | [level1] | [score1]/100 | [C1] | [H1] | [M1] | [L1] | [✓ if winner] |
| [name2] | [level2] | [score2]/100 | [C2] | [H2] | [M2] | [L2] | [✓ if winner] |
| [name3] | [level3] | [score3]/100 | [C3] | [H3] | [M3] | [L3] | [✓ if winner] |

## Recommendation

[If clear winner:]
✅ **Recommended:** [winner_name]

**Reasoning:**
- Lowest risk score ([score]/100 vs [other_scores])
- Fewer critical vulnerabilities ([count] vs [other_counts])
- [Additional differentiators]

[If tie or close call:]
⚖️ **Close Call:** [component1] and [component2] are similar in safety

**Comparison:**
- [component1]: Better at [strength], but [weakness]
- [component2]: Better at [strength], but [weakness]

**Suggestion:** [Guidance based on use case or additional criteria]

[If all are unsafe:]
❌ **None Recommended:** All components show HIGH or CRITICAL risk

**Alternatives:**
- Fix vulnerabilities in the least risky option ([name])
- Consider third-party alternatives
- Build new component with security-first design

## Detailed Comparison

### [Component 1 Name]
- **File:** [path]
- **Size:** [size]
- **Risk:** [level] ([score]/100)

**Strengths:**
[List positive aspects, e.g.:]
- No command injection vulnerabilities
- Proper input validation
- Safe API usage

**Weaknesses:**
[List negative aspects, e.g.:]
- 2 HIGH: Unvalidated network calls
- 1 MEDIUM: Missing error handling

**Key Findings:**
[Top 3-5 most significant findings]

---

### [Component 2 Name]
[Same structure as Component 1]

---

### [Component 3 Name]
[Same structure as Component 1]

## Security Category Breakdown

| Category | [Name1] | [Name2] | [Name3] |
|----------|---------|---------|---------|
| Command Injection | [✓/✗] | [✓/✗] | [✓/✗] |
| Data Exfiltration | [✓/✗] | [✓/✗] | [✓/✗] |
| Credential Access | [✓/✗] | [✓/✗] | [✓/✗] |
| File Operations | [✓/✗] | [✓/✗] | [✓/✗] |
| Prompt Injection | [✓/✗] | [✓/✗] | [✓/✗] |
| Obfuscation | [✓/✗] | [✓/✗] | [✓/✗] |

[✓ = No issues, ✗ = Issues detected]

## Use Case Recommendations

### For Production Use:
**Choose:** [lowest risk component if LOW/MEDIUM]
**Rationale:** Minimal attack surface, suitable for production with standard monitoring

### For Development/Testing:
**Choose:** [component with best feature set, even if MEDIUM risk]
**Rationale:** Can use with enhanced monitoring and restricted permissions

### For High-Security Environments:
**Choose:** [only if LOW risk available, otherwise "None - all require hardening"]
**Rationale:** High-security contexts require zero CRITICAL/HIGH findings

## Next Steps

- **Review full reports:**
  - [component1]: `./sandbox_logs/benderbox_[run_id1].json`
  - [component2]: `./sandbox_logs/benderbox_[run_id2].json`
  - [component3]: `./sandbox_logs/benderbox_[run_id3].json`

- **Fix vulnerabilities:** Focus on winner's remaining issues first

- **Re-compare:** After fixes, run comparison again to verify improvements

- **Deep dive:** Run `infra-deep` profile on winner for final audit

---
*Comparison ID: [timestamp]*
*Analysis Engine: BenderBox v2.0*
*Profile: [profile]*
```

### List Format (for 4-5 components)

For 4-5 components, use list format instead of detailed breakdowns:

```markdown
# Infrastructure Security Comparison

**Analyzed:** [timestamp_utc]
**Components:** [count]

## Ranked by Safety

### 🥇 Safest: [Component Name]
- **Risk:** [level] ([score]/100)
- **Critical:** [count], **High:** [count], **Medium:** [count], **Low:** [count]
- **Key Strength:** [main differentiator]
- **Minor Issue:** [if any]

---

### 🥈 Second: [Component Name]
- **Risk:** [level] ([score]/100)
- **Critical:** [count], **High:** [count], **Medium:** [count], **Low:** [count]
- **Compared to #1:** [main difference]

---

### 🥉 Third: [Component Name]
[Same format]

---

### ⚠️ Fourth: [Component Name]
[Same format]

---

### ❌ Fifth: [Component Name]
[Same format]

## Recommendation

**Use:** [#1 component name]
**Avoid:** [components with CRITICAL risk]

[Brief reasoning paragraph]

## Quick Reference Table

| Rank | Component | Risk | Score | Critical | High | Notes |
|------|-----------|------|-------|----------|------|-------|
| 1 | [name] | [level] | [score] | [C] | [H] | [notes] |
| 2 | [name] | [level] | [score] | [C] | [H] | [notes] |
| 3 | [name] | [level] | [score] | [C] | [H] | [notes] |
| 4 | [name] | [level] | [score] | [C] | [H] | [notes] |
| 5 | [name] | [level] | [score] | [C] | [H] | [notes] |
```

## Error Handling

### If fewer than 2 components:
```markdown
❌ **Error:** Need at least 2 components to compare

**Provided:** [count]

**Suggestion:**
Provide 2-5 component paths, e.g.:
"Compare security: server1.py, server2.py, server3.py"
```

### If more than 5 components:
```markdown
⚠️ **Warning:** Comparing more than 5 components affects readability

**Provided:** [count]

**Suggestion:**
- I'll compare the first 5 components
- Or, you can specify which 5 to prioritize
- Or, run multiple comparisons (e.g., compare top 3, then next 3)

**Proceed with first 5?** (y/n)
```

### If mixed component types:
```markdown
⚠️ **Warning:** Mixed component types detected

**MCP Servers (.py):** [list]
**Skills (.md):** [list]

**Suggestion:**
Comparing different component types may not be meaningful.
Would you like to:
1. Compare only MCP servers
2. Compare only skills
3. Proceed with mixed comparison (relative risk only)

**Your choice?**
```

### If analysis fails for a component:
```markdown
⚠️ **Warning:** Analysis failed for [component_name]

**Error:** [error message]

**Action:** Excluding from comparison, proceeding with remaining [count] components

[Continue with successful analyses]
```

## Example Interactions

### Example 1: Compare Two MCP Servers
```
User: "Which is safer: server_a.py or server_b.py?"

Agent: [Analyzes both with infra-standard profile]
Agent: "Comparison complete.

        ✅ Recommended: server_b.py

        server_a.py - MEDIUM risk (45/100)
        - 1 CRITICAL: Command injection on line 78
        - 2 HIGH: Data exfiltration, unsafe file ops

        server_b.py - LOW risk (18/100)
        - 0 CRITICAL
        - 1 MEDIUM: Missing input validation

        server_b.py is significantly safer with no critical vulnerabilities."
```

### Example 2: Compare Multiple Skills
```
User: "Compare these skills: skill1.md, skill2.md, skill3.md"

Agent: [Analyzes all three]
Agent: "Security comparison:

        🥇 skill2.md - LOW risk (12/100) - Safest
        🥈 skill1.md - MEDIUM risk (38/100)
        🥉 skill3.md - HIGH risk (67/100) - Not recommended

        Recommendation: Use skill2.md

        skill3.md contains prompt injection patterns and should be avoided.
        See full comparison below..."
```

### Example 3: No Clear Winner
```
User: "Compare server1.py and server2.py"

Agent: [Analyzes both]
Agent: "Close call - both show MEDIUM risk:

        server1.py - MEDIUM (42/100)
        - Better: No network calls
        - Worse: 1 command injection issue

        server2.py - MEDIUM (39/100)
        - Better: Input validation present
        - Worse: 2 unvalidated network calls

        Recommendation depends on use case:
        - Air-gapped environment: server1.py (network calls irrelevant)
        - Internet-facing: Fix server1.py's injection issue first

        Would you like me to suggest specific fixes?"
```

## Best Practices

### Number of Components
- **2-3 components:** Use detailed comparison format
- **4-5 components:** Use ranked list format
- **6+ components:** Suggest breaking into multiple comparisons

### Profile Selection
- **Use `infra-quick`** for initial comparison of many components
- **Use `infra-standard`** for decision-making (default)
- **Use `infra-deep`** for final validation of chosen component

### Communication Strategy
- **Clear winner:** State confidently with reasoning
- **Close call:** Explain trade-offs, ask about use case
- **All unsafe:** Recommend against all, suggest alternatives
- **Always:** Provide specific evidence (counts, scores, examples)

### Follow-up Suggestions
After comparison:
- Offer to show detailed report for winner
- Suggest fixes for near-winner if close call
- Offer to analyze additional components if needed
- Recommend deep profile on chosen component before deployment

## Constraints

### NEVER:
- ❌ Compare more than 5 components without warning
- ❌ Recommend HIGH/CRITICAL risk component as "winner"
- ❌ Make recommendations without showing evidence
- ❌ Compare components without analyzing each one
- ❌ Ignore context (use case, environment, requirements)

### ALWAYS:
- ✓ Analyze all components with same profile (fair comparison)
- ✓ Show risk scores and findings counts
- ✓ Provide clear recommendation or explain trade-offs
- ✓ Include paths to full JSON reports
- ✓ Consider user's specific use case in recommendations

## Metadata

- **Version:** 2.0.0
- **Category:** Infrastructure Security
- **Complexity:** High
- **Estimated Execution Time:** 10s-10min (profile and count dependent)
- **Dependencies:** benderbox_sandbox_cli.py v2.0+, analysis modules
- **Output Type:** Markdown comparison report
- **Analysis Method:** Parallel static analysis + comparative evaluation
