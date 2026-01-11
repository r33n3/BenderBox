#!/usr/bin/env python3
"""
BenderBox MCP Server Security Analyzer

Static analysis of MCP server Python code to detect security vulnerabilities
and malicious behaviors without executing the code.

Detects:
  - Command injection vulnerabilities
  - Data exfiltration patterns
  - Privilege escalation attempts
  - Backdoor tools
  - Unsafe file operations
  - Network access to suspicious domains

Architecture:
  - Static analysis only (safe - no code execution)
  - AST parsing for pattern detection
  - Returns standardized findings for CLI integration
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Set, Optional


@dataclass
class SecurityFinding:
    """Represents a security issue found during analysis."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # command_injection, data_exfiltration, etc.
    title: str
    description: str
    location: str  # File:line or function name
    code_snippet: Optional[str] = None
    remediation: Optional[str] = None


class MCPServerAnalyzer:
    """
    Analyzes MCP server Python code for security vulnerabilities.

    Uses AST parsing to detect dangerous patterns without executing code.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.source_code = None
        self.ast_tree = None
        self.findings: List[SecurityFinding] = []

    def analyze(self) -> Dict[str, Any]:
        """
        Main analysis entry point.

        Returns:
            Dictionary with analysis results and findings
        """
        try:
            # Read source code
            with self.file_path.open('r', encoding='utf-8') as f:
                self.source_code = f.read()

            # Parse AST
            try:
                self.ast_tree = ast.parse(self.source_code)
            except SyntaxError as e:
                return {
                    "status": "error",
                    "error": f"Python syntax error: {e}",
                    "findings": []
                }

            # Run all security checks
            self._check_command_injection()
            self._check_data_exfiltration()
            self._check_file_operations()
            self._check_credential_access()
            self._check_dangerous_imports()
            self._check_obfuscation()
            self._check_backdoor_patterns()

            # Aggregate results
            return self._build_report()

        except Exception as e:
            return {
                "status": "error",
                "error": f"Analysis failed: {e}",
                "findings": []
            }

    def _build_report(self) -> Dict[str, Any]:
        """Build final analysis report."""
        # Count findings by severity
        severity_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "INFO": 0
        }

        for finding in self.findings:
            severity_counts[finding.severity] += 1

        # Determine overall risk level
        if severity_counts["CRITICAL"] > 0:
            risk_level = "CRITICAL"
        elif severity_counts["HIGH"] > 0:
            risk_level = "HIGH"
        elif severity_counts["MEDIUM"] > 0:
            risk_level = "MEDIUM"
        elif severity_counts["LOW"] > 0:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"

        return {
            "status": "success",
            "file_path": str(self.file_path),
            "risk_level": risk_level,
            "severity_counts": severity_counts,
            "total_findings": len(self.findings),
            "findings": [
                {
                    "severity": f.severity,
                    "category": f.category,
                    "title": f.title,
                    "description": f.description,
                    "location": f.location,
                    "code_snippet": f.code_snippet,
                    "remediation": f.remediation
                }
                for f in self.findings
            ]
        }

    def _check_command_injection(self):
        """Detect command injection vulnerabilities."""
        dangerous_functions = {
            'os.system', 'os.popen', 'subprocess.call',
            'subprocess.run', 'subprocess.Popen',
            'eval', 'exec', 'compile'
        }

        for node in ast.walk(self.ast_tree):
            # Check function calls
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)

                if func_name in dangerous_functions:
                    # Check if arguments come from user input
                    is_user_input = self._traces_to_user_input(node)

                    severity = "CRITICAL" if is_user_input else "HIGH"

                    self.findings.append(SecurityFinding(
                        severity=severity,
                        category="command_injection",
                        title=f"Dangerous function call: {func_name}",
                        description=f"Use of {func_name} can lead to command injection if user input is not sanitized.",
                        location=f"Line {node.lineno}",
                        code_snippet=self._get_code_snippet(node.lineno),
                        remediation="Use subprocess with shell=False and pass arguments as list. Validate/sanitize all user input."
                    ))

    def _check_data_exfiltration(self):
        """Detect potential data exfiltration via network calls."""
        network_modules = {'requests', 'urllib', 'http.client', 'socket', 'ftplib', 'smtplib'}

        # Check imports
        imported_modules = set()
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module.split('.')[0])

        # Flag network module usage
        suspicious_imports = network_modules & imported_modules

        if suspicious_imports:
            # Look for network calls
            for node in ast.walk(self.ast_tree):
                if isinstance(node, ast.Call):
                    func_name = self._get_function_name(node.func)

                    # requests.post/get
                    if any(module in func_name for module in ['requests', 'urllib', 'http']):
                        # Extract URL if possible
                        url = self._extract_url_from_call(node)

                        # Check if URL is external (not localhost)
                        is_external = url and not any(local in url for local in ['localhost', '127.0.0.1', '0.0.0.0'])

                        severity = "HIGH" if is_external else "MEDIUM"

                        self.findings.append(SecurityFinding(
                            severity=severity,
                            category="data_exfiltration",
                            title="Network call detected",
                            description=f"Network call to {url or 'unknown URL'} - potential data exfiltration.",
                            location=f"Line {node.lineno}",
                            code_snippet=self._get_code_snippet(node.lineno),
                            remediation="Review network call necessity. If legitimate, document and restrict to known endpoints."
                        ))

    def _check_file_operations(self):
        """Detect unsafe file operations."""
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)

                # File write operations
                if func_name in ['open', 'pathlib.Path.open']:
                    # Check for write mode
                    mode = self._extract_file_mode(node)
                    if mode and any(m in mode for m in ['w', 'a', 'x', '+']):
                        # Check for path traversal
                        path_arg = self._extract_file_path(node)

                        severity = "HIGH" if path_arg and '..' in str(path_arg) else "MEDIUM"

                        self.findings.append(SecurityFinding(
                            severity=severity,
                            category="unsafe_file_operation",
                            title="File write operation",
                            description=f"File write detected with mode '{mode}' - potential unauthorized file access.",
                            location=f"Line {node.lineno}",
                            code_snippet=self._get_code_snippet(node.lineno),
                            remediation="Validate file paths, prevent path traversal, restrict write locations to designated directories."
                        ))

                # os.remove, shutil.rmtree
                elif func_name in ['os.remove', 'os.unlink', 'shutil.rmtree', 'pathlib.Path.unlink']:
                    self.findings.append(SecurityFinding(
                        severity="MEDIUM",
                        category="unsafe_file_operation",
                        title=f"File deletion: {func_name}",
                        description="File deletion operation - ensure proper authorization.",
                        location=f"Line {node.lineno}",
                        code_snippet=self._get_code_snippet(node.lineno),
                        remediation="Restrict file deletion to authorized paths only."
                    ))

    def _check_credential_access(self):
        """Detect access to credentials and sensitive files."""
        sensitive_patterns = [
            r'\.ssh',
            r'\.aws',
            r'\.env',
            r'password',
            r'api[_-]?key',
            r'secret',
            r'token',
            r'credential',
            r'/etc/passwd',
            r'/etc/shadow',
        ]

        # Search in string literals
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for pattern in sensitive_patterns:
                    if re.search(pattern, node.value, re.IGNORECASE):
                        self.findings.append(SecurityFinding(
                            severity="HIGH",
                            category="credential_access",
                            title="Sensitive data reference",
                            description=f"Reference to potentially sensitive data: '{node.value}'",
                            location=f"Line {node.lineno}",
                            code_snippet=self._get_code_snippet(node.lineno),
                            remediation="Avoid hardcoding credentials. Use secure credential storage."
                        ))
                        break

    def _check_dangerous_imports(self):
        """Detect imports of dangerous or unusual modules."""
        dangerous_modules = {
            'pickle': 'MEDIUM',  # Arbitrary code execution
            'marshal': 'MEDIUM',  # Arbitrary code execution
            'shelve': 'MEDIUM',  # Uses pickle
            '__import__': 'HIGH',  # Dynamic imports
            'importlib': 'MEDIUM',  # Dynamic imports
            'pty': 'HIGH',  # PTY spawning
            'telnetlib': 'HIGH',  # Rarely legitimate
            'ftplib': 'MEDIUM',  # Rarely needed in MCP server
        }

        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = None

                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module.split('.')[0] if node.module else None

                if module_name in dangerous_modules:
                    severity = dangerous_modules[module_name]

                    self.findings.append(SecurityFinding(
                        severity=severity,
                        category="dangerous_import",
                        title=f"Dangerous module import: {module_name}",
                        description=f"Import of '{module_name}' - potential security risk.",
                        location=f"Line {node.lineno}",
                        code_snippet=self._get_code_snippet(node.lineno),
                        remediation=f"Review necessity of {module_name}. Use safer alternatives if possible."
                    ))

    def _check_obfuscation(self):
        """Detect code obfuscation patterns."""
        # Check for base64 encoding
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)

                if 'base64' in func_name.lower():
                    self.findings.append(SecurityFinding(
                        severity="MEDIUM",
                        category="obfuscation",
                        title="Base64 encoding detected",
                        description="Base64 encoding - may indicate obfuscation or data exfiltration.",
                        location=f"Line {node.lineno}",
                        code_snippet=self._get_code_snippet(node.lineno),
                        remediation="Review why base64 encoding is used. Ensure it's not hiding malicious behavior."
                    ))

        # Check for suspiciously short variable names (obfuscation indicator)
        var_names = set()
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.Name):
                var_names.add(node.id)

        suspicious_vars = [v for v in var_names if len(v) == 1 and v not in {'i', 'j', 'k', 'x', 'y', 'z', 'f', 'e'}]

        if len(suspicious_vars) > 10:  # Threshold
            self.findings.append(SecurityFinding(
                severity="LOW",
                category="obfuscation",
                title="Suspicious variable naming",
                description=f"Many single-letter variables ({len(suspicious_vars)}) - possible obfuscation.",
                location="Global",
                remediation="Use descriptive variable names for maintainability and security review."
            ))

    def _check_backdoor_patterns(self):
        """Detect common backdoor patterns."""
        # Check for hidden/misleading function names
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                # Functions with misleading names
                if node.name.startswith('_') and any(dangerous in node.name.lower() for dangerous in ['exec', 'eval', 'shell', 'cmd']):
                    self.findings.append(SecurityFinding(
                        severity="HIGH",
                        category="backdoor",
                        title=f"Suspicious function name: {node.name}",
                        description="Function with suspicious name pattern - potential backdoor.",
                        location=f"Line {node.lineno}",
                        code_snippet=self._get_code_snippet(node.lineno),
                        remediation="Review function purpose. Avoid misleading names."
                    ))

                # Check for functions that spawn shells
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func_name = self._get_function_name(child.func)
                        if 'spawn' in func_name.lower() or 'shell' in func_name.lower():
                            self.findings.append(SecurityFinding(
                                severity="CRITICAL",
                                category="backdoor",
                                title="Shell spawning detected",
                                description=f"Function '{node.name}' spawns shell - potential backdoor.",
                                location=f"Line {child.lineno}",
                                code_snippet=self._get_code_snippet(child.lineno),
                                remediation="Remove shell spawning unless absolutely necessary and documented."
                            ))

    # Helper methods

    def _get_function_name(self, node: ast.AST) -> str:
        """Extract function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # e.g., os.system -> build "os.system"
            value_name = self._get_function_name(node.value)
            return f"{value_name}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_function_name(node.func)
        return ""

    def _traces_to_user_input(self, node: ast.Call) -> bool:
        """
        Heuristic: Check if function arguments trace to user input.

        Looks for:
          - arguments dict access (common in MCP tools)
          - input() calls
          - request parameters
        """
        for arg in ast.walk(node):
            if isinstance(arg, ast.Subscript):
                # Check for arguments["key"] pattern
                if isinstance(arg.value, ast.Name):
                    if arg.value.id in ['arguments', 'args', 'params', 'request']:
                        return True
            elif isinstance(arg, ast.Call):
                func_name = self._get_function_name(arg.func)
                if func_name in ['input', 'raw_input']:
                    return True

        return False

    def _extract_url_from_call(self, node: ast.Call) -> Optional[str]:
        """Extract URL from network call if possible."""
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith(('http://', 'https://', 'ftp://')):
                    return arg.value
        return None

    def _extract_file_mode(self, node: ast.Call) -> Optional[str]:
        """Extract file open mode."""
        # Check keyword arguments
        for keyword in node.keywords:
            if keyword.arg == 'mode' and isinstance(keyword.value, ast.Constant):
                return keyword.value.value

        # Check positional arguments (second arg is mode)
        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
            return node.args[1].value

        return None

    def _extract_file_path(self, node: ast.Call) -> Optional[str]:
        """Extract file path from open call."""
        if node.args and isinstance(node.args[0], ast.Constant):
            return node.args[0].value
        return None

    def _get_code_snippet(self, line_number: int, context: int = 2) -> str:
        """Get code snippet around line number."""
        if not self.source_code:
            return ""

        lines = self.source_code.split('\n')
        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)

        snippet_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            snippet_lines.append(f"{prefix}{lines[i]}")

        return "\n".join(snippet_lines)


# Standalone function for CLI integration
def analyze_mcp_server(file_path: Path) -> Dict[str, Any]:
    """
    Analyze MCP server file for security vulnerabilities.

    Args:
        file_path: Path to MCP server Python file

    Returns:
        Dictionary with analysis results
    """
    analyzer = MCPServerAnalyzer(file_path)
    return analyzer.analyze()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python benderbox_mcp_analyzer.py <mcp_server.py>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    result = analyze_mcp_server(file_path)

    import json
    print(json.dumps(result, indent=2))
