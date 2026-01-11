"""
BenderBox - AI Security Testing and Model Safety Framework

A comprehensive offline-first platform for analyzing AI models and infrastructure
for security, safety, and capabilities using local NLP models.

Version 3.0 adds:
- Local LLM-powered analysis (llama.cpp)
- Semantic code understanding
- Conversational interface
- Vector-based knowledge search
- Persistent report database
"""

__version__ = "3.0.0-alpha"

# Core modules (v1-v2)
from .sandbox_cli import *
from .mcp_server import *
from .mcp_analyzer import *
from .skill_analyzer import *
from .dynamic_tests import *
from .infrastructure_tests import *
from .interactive import *

# v3.0 Configuration
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
    # v3.0 Config
    "Config",
    "load_config",
    "get_config",
    "reload_config",
]
