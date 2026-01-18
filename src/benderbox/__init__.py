"""
BenderBox - AI Security Testing and Model Safety Framework

A comprehensive offline-first platform for analyzing AI models and infrastructure
for security, safety, and capabilities using local NLP models.

Version 4.0 adds:
- Enhanced interrogation engine with LLM judging
- Variant probing for persistent jailbreak testing
- Multi-provider support (OpenAI, Anthropic, Google, xAI)
- YAML-driven profile configuration
- Model comparison and trend analysis
- Improved TUI dashboard
"""

# Suppress llama.cpp C++ level logging (context size warnings, etc.)
# Must be set before any llama_cpp import anywhere in the codebase
import os
os.environ.setdefault("LLAMA_LOG_LEVEL", "ERROR")

# Global reference to prevent garbage collection of ctypes callback
_LLAMA_LOG_CALLBACK_REF = None


def _suppress_llama_cpp_logs():
    """
    Suppress llama.cpp C++ warnings by setting a null log callback.

    This is more reliable than environment variables because it directly
    controls the C++ logging at runtime.
    """
    global _LLAMA_LOG_CALLBACK_REF
    try:
        import ctypes
        import llama_cpp

        # Define the log callback type matching the C signature
        log_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

        @log_callback_type
        def _null_log_callback(level, text, user_data):
            pass  # Suppress all llama.cpp log output

        # Keep reference to prevent garbage collection
        _LLAMA_LOG_CALLBACK_REF = _null_log_callback
        llama_cpp.llama_log_set(_null_log_callback, None)
    except (ImportError, AttributeError, OSError):
        # llama_cpp not installed or doesn't have llama_log_set
        pass


# Apply log suppression when llama_cpp is available
_suppress_llama_cpp_logs()

__version__ = "4.0.0-alpha"

# Core modules
from .sandbox_cli import *
from .mcp_server import *
from .mcp_analyzer import *
from .skill_analyzer import *
from .dynamic_tests import *
from .infrastructure_tests import *
from .interactive import *

# Configuration
from .config import Config, load_config, get_config, reload_config

__all__ = [
    # Core
    "sandbox_cli",
    "mcp_server",
    "mcp_analyzer",
    "skill_analyzer",
    "dynamic_tests",
    "infrastructure_tests",
    "interactive",
    # Config
    "Config",
    "load_config",
    "get_config",
    "reload_config",
]
