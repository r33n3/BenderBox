"""
BenderBox Test Profiles

YAML-based test profile system for configurable security testing.

Profiles define which tests to run, how to score them, and what
risk thresholds to use. Users can modify profiles or create custom
ones without changing code.

Available profiles:
  - quick: Fast screening (~15 tests) for CI/CD and quick checks
  - standard: Balanced coverage (~50 tests) for production readiness
  - full: Exhaustive audit (~100+ tests) for security assessments
  - adversarial: Security-focused (~64 tests) for jailbreak testing

Usage:
    from benderbox.profiles import load_profile, list_profiles

    # Load a profile
    profile = load_profile("quick")

    # Get tests to run
    for test in profile.all_dynamic_tests:
        print(f"{test.id}: {test.prompt[:50]}...")

    # List available profiles
    print(list_profiles())  # ['quick', 'standard', 'full', 'adversarial']
"""

from .loader import (
    DynamicTest,
    TestProfile,
    ProfileLoader,
    VariantConfig,
    get_profile_loader,
    load_profile,
    list_profiles,
)

__all__ = [
    "DynamicTest",
    "TestProfile",
    "ProfileLoader",
    "VariantConfig",
    "get_profile_loader",
    "load_profile",
    "list_profiles",
]
