"""
Sample MCP Server with Intentional Vulnerabilities

This is an EXAMPLE file for testing BenderBox's MCP security analysis.
DO NOT use this code in production - it contains intentional security flaws.

Analyze with:
    python bb.py mcp analyze examples/mcp_servers/sample_vulnerable_server.py
"""

import subprocess
import os


class VulnerableMCPServer:
    """Example MCP server with security vulnerabilities for testing."""

    def execute_command(self, command: str) -> str:
        """
        VULNERABILITY: Command injection
        User input passed directly to shell without sanitization.
        """
        # BAD: Direct shell execution with user input
        result = subprocess.run(command, shell=True, capture_output=True)
        return result.stdout.decode()

    def read_file(self, path: str) -> str:
        """
        VULNERABILITY: Path traversal
        No validation of file path allows reading arbitrary files.
        """
        # BAD: No path validation
        with open(path, 'r') as f:
            return f.read()

    def write_file(self, path: str, content: str) -> str:
        """
        VULNERABILITY: Arbitrary file write
        No restrictions on where files can be written.
        """
        # BAD: Can write anywhere on filesystem
        with open(path, 'w') as f:
            f.write(content)
        return f"Written to {path}"

    def query_database(self, user_id: str) -> str:
        """
        VULNERABILITY: SQL injection
        User input concatenated into SQL query.
        """
        # BAD: String concatenation in SQL
        query = f"SELECT * FROM users WHERE id = '{user_id}'"
        return f"Would execute: {query}"

    def render_template(self, template: str) -> str:
        """
        VULNERABILITY: Template injection
        User input used directly in template rendering.
        """
        # BAD: Direct template evaluation
        return eval(f'f"{template}"')  # Dangerous!


# Example of what a SAFE server should look like:
class SafeMCPServer:
    """Example of secure MCP server patterns."""

    ALLOWED_COMMANDS = ['ls', 'pwd', 'whoami']
    ALLOWED_PATHS = ['/data/', '/tmp/']

    def execute_command(self, command: str) -> str:
        """SAFE: Whitelist-based command execution."""
        if command not in self.ALLOWED_COMMANDS:
            raise ValueError(f"Command not allowed: {command}")
        result = subprocess.run([command], capture_output=True)
        return result.stdout.decode()

    def read_file(self, path: str) -> str:
        """SAFE: Path validation before file access."""
        # Resolve and validate path
        real_path = os.path.realpath(path)
        if not any(real_path.startswith(allowed) for allowed in self.ALLOWED_PATHS):
            raise ValueError(f"Path not allowed: {path}")
        with open(real_path, 'r') as f:
            return f.read()
