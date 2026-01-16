"""
Profile configuration system for BenderBox interrogation.

Provides YAML-based profile configuration for customizable testing.
"""

from .loader import ProfileLoader, ProfileConfig

__all__ = ["ProfileLoader", "ProfileConfig"]
