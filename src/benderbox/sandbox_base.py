"""
BenderBox Sandbox Base Classes

Base classes for the sandbox test framework, extracted to avoid circular imports.
Both sandbox_cli.py and dynamic_tests.py import from this module.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TestResult:
    """Result from a single sandbox test."""
    name: str
    category: str
    status: str = "PASS"   # PASS, FAIL, WARN, ERROR, SKIP
    severity: str = "INFO" # INFO, LOW, MEDIUM, HIGH, CRITICAL
    score: Optional[float] = None
    details: str = ""
    metrics: Optional[Dict[str, float]] = None
    artifacts: Optional[List[Dict[str, str]]] = None


class SandboxTest:
    """
    Base class for sandbox tests. Override `run` in subclasses.

    `context` will hold:
      - model_info
      - profile
      - global settings
      - utilities for llama.cpp calls (later)
    """

    name: str = "base_test"
    category: str = "generic"

    def run(self, context: Dict[str, Any]) -> TestResult:
        raise NotImplementedError
