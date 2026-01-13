# Skill: Analyze MCP Server Security

## Purpose
Perform static security analysis of MCP (Model Context Protocol) server Python code to detect vulnerabilities including command injection, data exfiltration, credential harvesting, and backdoor patterns.

## Triggers
- User asks: "Is this MCP server safe?"
- User asks: "Analyze this MCP server for security issues"
- User asks: "Check this server.py for vulnerabilities"
- User provides: Path to a Python MCP server file
- User requests: MCP server security audit
- User requests: Infrastructure security assessment

## Required Tools
- `benderbox_sandbox_analyzeModel` - Analysis tool (v2.0 infrastructure support)
- `benderbox_sandbox_getLatestReport` - For retrieving cached results

## Inputs
- **mcp_server_path** (string, required): Path to MCP server Python file
  - Examples: `./servers/my_server.py`, `mcp_server.py`
- **profile** (string, optional): Analysis depth
  - `infra-quick` - Fast static analysis (~10-20s) [DEFAULT]
  - `infra-standard` - Standard security checks (~30-60s)
  - `infra-deep` - Comprehensive analysis (~2-5min)
- **force_reanalysis** (boolean, optional): Skip cached reports, run fresh analysis

## Process Flow

### 1. Check for Existing Analysis
If `force_reanalysis` is false:
  - Call `benderbox_sandbox_getLatestReport` with server filename
  - If report exists and is recent (< 24 hours), use cached data
  - Otherwise, proceed to step 2

### 2. Run Analysis
Call `benderbox_sandbox_analyzeModel` with:
```json
{
  "mcp_server_path": "<user_provided_path>",
  "profile": "<user_selected_or_default>",
  "log_dir": "./sandbox_logs"
}
```

### 3. Parse JSON Response
Extract key sections from returned JSON:
- `infrastructure.mcp_server` - MCP server analysis results
- `overall_risk` - Risk level and score
- `tests` - Individual test results (mcp_static_analysis)
- `findings` - Specific vulnerabilities detected

### 4. Generate Human-Readable Summary
Transform JSON into Markdown report (see Output Format below)

## Output Format

### Standard Summary

```markdown
# MCP Server Security Analysis

## Server Information
- **Filename:** [infrastructure.mcp_server.filename]
- **Size:** [infrastructure.mcp_server.file_size] bytes
- **Lines of Code:** [infrastructure.mcp_server.lines_of_code]
- **Analyzed:** [timestamp_utc]

## Risk Assessment
- **Level:** [overall_risk.level] (LOW/MEDIUM/HIGH/CRITICAL)
- **Score:** [overall_risk.score]/100

### Primary Risk Factors
[List each factor from overall_risk.primary_factors]

## Security Findings

[If findings exist, organize by category:]

### 🔴 Critical Vulnerabilities
[For each finding with severity=CRITICAL:]
- **[finding.category]** - [finding.description]
  - **Location:** Line [finding.line_number]
  - **Pattern:** `[finding.code_snippet]`
  - **Impact:** [finding.impact]
  - **Remediation:** [finding.remediation]

### 🟠 High Severity Issues
[For each finding with severity=HIGH:]
- **[finding.category]** - [finding.description]
  - **Location:** Line [finding.line_number]
  - **Details:** [finding.details]

### 🟡 Medium Severity Issues
[For each finding with severity=MEDIUM:]
- **[finding.category]** - [finding.description]

### 🔵 Low Severity / Informational
[For each finding with severity=LOW/INFO:]
- **[finding.category]** - [finding.description]

[If no findings:]
### ✅ No Vulnerabilities Detected
This MCP server passed all static security checks.

## Detected Patterns

### Tools Registered
[List tool functions found:]
- `[tool_name]` - [description if available]

### External Dependencies
[List imports from infrastructure.mcp_server.imports:]
- `[import_name]` [Mark as DANGEROUS if flagged]

### Dangerous API Usage
[If dangerous functions detected:]
- **Command Execution:** [count of os.system, subprocess, etc.]
- **Network Calls:** [count of requests, urllib, sockets]
- **File Operations:** [count of file writes, path traversal]
- **Credential Access:** [count of password/key reads]
- **Obfuscation:** [count of base64, eval, exec]

## Test Results
[For each test in tests array where category="mcp_security":]
- **[test.name]**: [test.status] - [test.severity]
  - [test.details if FAIL/WARN]

## Recommendations

[Based on risk level, provide specific guidance:]

### For LOW Risk:
✅ **Deployment Status:** Safe for use with standard monitoring
- Consider code review for best practices
- Monitor tool usage in production logs
- Implement rate limiting on tool calls

### For MEDIUM Risk:
⚠️ **Deployment Status:** Requires additional safeguards
- **Required Actions:**
  1. Fix [count] medium-severity issues
  2. Add input validation to all tools
  3. Implement request logging and monitoring
  4. Restrict network access if possible
- **Optional:** Run deep analysis for comprehensive audit

### For HIGH Risk:
❌ **Deployment Status:** NOT recommended without fixes
- **Critical Actions Required:**
  1. Fix all HIGH and CRITICAL vulnerabilities
  2. Remove or sandbox dangerous API calls
  3. Implement strict input sanitization
  4. Add authentication and authorization
  5. Re-analyze after fixes

### For CRITICAL Risk:
🚨 **Deployment Status:** DO NOT USE
- **This server contains severe security vulnerabilities**
- **Immediate Actions:**
  1. Do not deploy to production
  2. Review all findings marked CRITICAL
  3. Consider complete rewrite with security-first design
  4. Consult security expert for remediation plan

## Next Steps
[Suggest relevant actions based on findings:]
- Run `infra-deep` profile for comprehensive analysis
- Compare with other MCP servers: "Which server is safer?"
- Review full JSON report: `./sandbox_logs/benderbox_[run_id].json`
- Check specific lines of code: `cat [filename] | sed -n '[line_number]p'`

---
*Report ID: [run_id]*
*Analysis Engine: BenderBox v2.0 Infrastructure Analyzer*
*Profile: [profile]*
*Full JSON: ./sandbox_logs/benderbox_[run_id].json*
```

## Error Handling

### If server file not found:
```markdown
❌ **Error:** MCP server file not found at `[mcp_server_path]`

**Suggestions:**
- Check the file path is correct
- Ensure the file exists: `ls [mcp_server_path]`
- Verify file is a Python (.py) file
```

### If not a valid Python file:
```markdown
❌ **Error:** File is not valid Python code

**Details:** [syntax error message]

**Suggestions:**
- Check for syntax errors: `python -m py_compile [mcp_server_path]`
- Ensure file contains valid Python syntax
```

### If analysis fails:
```markdown
❌ **Error:** Analysis failed

**Details:** [error message from MCP tool]

**Troubleshooting:**
- Ensure `benderbox_mcp_analyzer.py` is present
- Check `benderbox_sandbox_cli.py` supports v2.0 infrastructure analysis
- Run manually: `python benderbox_sandbox_cli.py --mcp-server [path] --profile infra-quick`
```

## Example Interactions

### Example 1: Quick Security Check
```
User: "Is my_mcp_server.py safe to deploy?"

Agent: [Calls benderbox_sandbox_analyzeModel with mcp_server_path and profile=infra-quick]
Agent: [Generates security report]
Agent: "Analysis complete. Your server shows MEDIUM risk with 3 findings:
        - Command injection vulnerability on line 45
        - Unvalidated network request on line 78
        - Missing input sanitization on line 102

        I recommend fixing these issues before deployment."
```

### Example 2: Deep Security Audit
```
User: "Run a comprehensive security audit on server.py"

Agent: [Calls benderbox_sandbox_analyzeModel with profile=infra-deep]
Agent: [Generates detailed report with all findings]
Agent: "Deep analysis found 7 security issues:
        - 2 CRITICAL (command injection, data exfiltration)
        - 3 HIGH (file access, credential harvesting)
        - 2 MEDIUM (missing validation)

        This server is NOT safe for deployment. See full report below..."
```

### Example 3: Comparison Request
```
User: "Check this server for backdoors"

Agent: [Calls benderbox_sandbox_analyzeModel with profile=infra-standard]
Agent: [Reviews backdoor detection results]
Agent: "Backdoor analysis complete. Found 1 suspicious pattern:
        - Hidden shell spawning on line 156 (subprocess.Popen with /bin/bash)

        This appears to be a reverse shell backdoor. Severity: CRITICAL"
```

## Best Practices

### Profile Selection
- **Use `infra-quick`** for initial security scan (default)
- **Use `infra-standard`** for general security audit
- **Use `infra-deep`** when deploying to production or handling sensitive data

### Risk Communication
- **LOW risk:** "This server appears safe with standard precautions"
- **MEDIUM risk:** "This server has some security concerns - review and fix recommended"
- **HIGH risk:** "This server has significant vulnerabilities - fixes required before deployment"
- **CRITICAL risk:** "This server is NOT safe to use - contains severe security flaws"

### Vulnerability Prioritization
1. **Fix CRITICAL first:** Command injection, data exfiltration, backdoors
2. **Then HIGH:** Credential access, unsafe file operations
3. **Then MEDIUM:** Missing validation, weak patterns
4. **Finally LOW:** Code quality, best practices

### Follow-up Actions
After showing results:
- If CRITICAL/HIGH findings → provide specific remediation steps
- If comparison needed → offer to analyze alternative servers
- If deep dive needed → suggest reviewing specific lines of code
- Always include full JSON report location for detailed investigation

## Constraints

### NEVER:
- ❌ Execute or import the analyzed MCP server code
- ❌ Make network requests to test external endpoints
- ❌ Modify the server file being analyzed
- ❌ Assume a server is safe without running analysis
- ❌ Recommend deployment of HIGH/CRITICAL risk servers

### ALWAYS:
- ✓ Use static analysis only (AST parsing, no execution)
- ✓ Clearly communicate severity levels
- ✓ Provide specific line numbers for findings
- ✓ Include remediation guidance
- ✓ Recommend re-analysis after fixes

## Metadata

- **Version:** 2.0.0
- **Category:** Infrastructure Security
- **Complexity:** Medium
- **Estimated Execution Time:** 10s-5min (profile-dependent)
- **Dependencies:** benderbox_mcp_analyzer.py, benderbox_sandbox_cli.py v2.0+
- **Output Type:** Markdown security report
- **Analysis Method:** Static AST parsing (no code execution)
