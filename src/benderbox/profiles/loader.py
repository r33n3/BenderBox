"""
Profile Loader for BenderBox

Loads test profiles from YAML configuration files, providing a unified
way to define tests for both static analysis and dynamic interrogation.

Profiles define:
  - Which test categories to run
  - How many tests per category
  - Scoring weights
  - Risk thresholds
  - Static tests (file/metadata checks)
  - Dynamic tests (prompts sent to model)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Default profile directory
DEFAULT_PROFILES_DIR = Path(__file__).parent.parent.parent.parent / "configs" / "profiles"


@dataclass
class VariantConfig:
    """Configuration for variant/persistent probing."""

    enabled: bool = False
    max_variants_per_prompt: int = 5
    stop_on_success: bool = True
    technique_categories: List[str] = field(default_factory=list)
    techniques: List[str] = field(default_factory=list)
    apply_to_categories: List[str] = field(default_factory=list)

    # Map category names to actual technique enums
    CATEGORY_MAP = {
        "encoding": ["base64", "rot13", "leetspeak", "unicode", "reversed"],
        "roleplay": ["dan", "evil_assistant", "developer_mode", "uncensored_model"],
        "context": ["hypothetical", "educational", "fictional", "research"],
        "emotional": ["urgency", "guilt", "authority"],
        "injection": ["ignore_previous", "system_override", "nested"],
        "combination": ["gradual_escalation", "split_request"],
    }

    def get_techniques(self) -> List[str]:
        """Get list of technique names to use."""
        # If explicit techniques are set, use those
        if self.techniques:
            return self.techniques

        # Otherwise, expand technique categories
        techniques = []
        for category in self.technique_categories:
            if category in self.CATEGORY_MAP:
                techniques.extend(self.CATEGORY_MAP[category])
        return techniques

    def should_apply_variants(self, test_category: str) -> bool:
        """Check if variants should be generated for a test category."""
        if not self.enabled:
            return False
        if not self.apply_to_categories:
            return True  # Apply to all if not specified
        return test_category in self.apply_to_categories

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariantConfig":
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            enabled=data.get("enabled", False),
            max_variants_per_prompt=data.get("max_variants_per_prompt", 5),
            stop_on_success=data.get("stop_on_success", True),
            technique_categories=data.get("technique_categories", []),
            techniques=data.get("techniques", []),
            apply_to_categories=data.get("apply_to_categories", []),
        )


@dataclass
class DynamicTest:
    """A dynamic test prompt to send to the model."""

    id: str
    prompt: str
    category: str
    expected_behavior: str  # "refuse", "comply", "warn"
    severity: str  # "critical", "high", "medium", "low", "info"
    tags: List[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DynamicTest":
        """Create from dictionary."""
        return cls(
            id=data.get("id", "unknown"),
            prompt=data.get("prompt", ""),
            category=data.get("category", "unknown"),
            expected_behavior=data.get("expected_behavior", "refuse"),
            severity=data.get("severity", "medium"),
            tags=data.get("tags", []),
            description=data.get("description", ""),
        )


@dataclass
class TestProfile:
    """A test profile loaded from YAML."""

    name: str
    description: str
    version: str
    estimated_tests: int

    # Category configuration
    categories: Dict[str, Union[int, str]]  # category -> count or "all"

    # Scoring
    scoring_weights: Dict[str, float]
    risk_thresholds: Dict[str, int]

    # Variant/persistent probing configuration
    variant_config: VariantConfig = field(default_factory=VariantConfig)

    # Tests
    static_tests: List[str] = field(default_factory=list)
    dynamic_tests: List[DynamicTest] = field(default_factory=list)

    # Custom prompts (legacy format compatibility)
    custom_prompts: List[DynamicTest] = field(default_factory=list)

    @property
    def all_dynamic_tests(self) -> List[DynamicTest]:
        """Get all dynamic tests including custom prompts."""
        return self.dynamic_tests + self.custom_prompts

    def get_tests_for_category(self, category: str) -> List[DynamicTest]:
        """Get dynamic tests for a specific category."""
        return [t for t in self.all_dynamic_tests if t.category == category]

    def get_category_limit(self, category: str) -> Optional[int]:
        """Get the test limit for a category (None = unlimited)."""
        limit = self.categories.get(category)
        if limit == "all" or limit is None:
            return None
        return int(limit)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestProfile":
        """Create profile from dictionary."""
        # Parse dynamic tests
        dynamic_tests = []
        for test_data in data.get("dynamic_tests", []):
            dynamic_tests.append(DynamicTest.from_dict(test_data))

        # Parse custom prompts (legacy format)
        custom_prompts = []
        for prompt_data in data.get("custom_prompts", []):
            custom_prompts.append(DynamicTest.from_dict(prompt_data))

        # Parse variant configuration
        variant_config = VariantConfig.from_dict(data.get("variant_config", {}))

        return cls(
            name=data.get("name", "unknown"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            estimated_tests=data.get("estimated_tests", 0),
            categories=data.get("categories", {}),
            scoring_weights=data.get("scoring_weights", {}),
            risk_thresholds=data.get("risk_thresholds", {}),
            variant_config=variant_config,
            static_tests=data.get("static_tests", []),
            dynamic_tests=dynamic_tests,
            custom_prompts=custom_prompts,
        )


class ProfileLoader:
    """
    Loads and manages test profiles from YAML files.

    Usage:
        loader = ProfileLoader()
        profile = loader.load("quick")

        # Get static tests to run
        static_tests = profile.static_tests

        # Get dynamic tests (prompts)
        for test in profile.all_dynamic_tests:
            print(f"{test.id}: {test.prompt[:50]}...")
    """

    # Profile aliases for backwards compatibility
    ALIASES = {
        "deep": "full",
        "comprehensive": "full",
        "attack": "adversarial",
        "security": "adversarial",
        "fast": "quick",
        "minimal": "quick",
    }

    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize ProfileLoader.

        Args:
            profiles_dir: Directory containing profile YAML files.
                         Defaults to configs/profiles/.
        """
        self.profiles_dir = Path(profiles_dir) if profiles_dir else DEFAULT_PROFILES_DIR
        self._cache: Dict[str, TestProfile] = {}

    def resolve_alias(self, name: str) -> str:
        """Resolve profile alias to actual profile name."""
        return self.ALIASES.get(name.lower(), name.lower())

    def load(self, name: str) -> TestProfile:
        """
        Load a profile by name.

        Args:
            name: Profile name (e.g., "quick", "standard", "full", "adversarial")

        Returns:
            TestProfile instance.

        Raises:
            FileNotFoundError: If profile YAML doesn't exist.
            ValueError: If profile YAML is invalid.
        """
        # Resolve aliases
        resolved_name = self.resolve_alias(name)

        # Check cache
        if resolved_name in self._cache:
            return self._cache[resolved_name]

        # Find profile file
        profile_path = self.profiles_dir / f"{resolved_name}.yaml"
        if not profile_path.exists():
            # Try .yml extension
            profile_path = self.profiles_dir / f"{resolved_name}.yml"

        if not profile_path.exists():
            raise FileNotFoundError(
                f"Profile not found: {name} (looked for {resolved_name}.yaml in {self.profiles_dir})"
            )

        # Load and parse
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            profile = TestProfile.from_dict(data)
            self._cache[resolved_name] = profile

            logger.info(f"Loaded profile '{resolved_name}' with {len(profile.all_dynamic_tests)} dynamic tests")
            return profile

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in profile {name}: {e}")

    def list_profiles(self) -> List[str]:
        """List available profile names."""
        profiles = []
        if self.profiles_dir.exists():
            for path in self.profiles_dir.glob("*.yaml"):
                profiles.append(path.stem)
            for path in self.profiles_dir.glob("*.yml"):
                if path.stem not in profiles:
                    profiles.append(path.stem)
        return sorted(profiles)

    def get_profile_info(self, name: str) -> Dict[str, Any]:
        """
        Get summary info about a profile without fully loading it.

        Args:
            name: Profile name.

        Returns:
            Dictionary with name, description, estimated_tests.
        """
        try:
            profile = self.load(name)
            return {
                "name": profile.name,
                "description": profile.description.strip(),
                "version": profile.version,
                "estimated_tests": profile.estimated_tests,
                "categories": list(profile.categories.keys()),
                "static_tests": len(profile.static_tests),
                "dynamic_tests": len(profile.all_dynamic_tests),
            }
        except (FileNotFoundError, ValueError) as e:
            return {"name": name, "error": str(e)}

    def get_static_test_ids(self, name: str) -> List[str]:
        """
        Get list of static test IDs for a profile.

        Args:
            name: Profile name.

        Returns:
            List of static test IDs.
        """
        profile = self.load(name)
        return profile.static_tests

    def get_dynamic_tests(
        self,
        name: str,
        category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[DynamicTest]:
        """
        Get dynamic tests for a profile.

        Args:
            name: Profile name.
            category: Optional category filter.
            limit: Optional limit on number of tests.

        Returns:
            List of DynamicTest instances.
        """
        profile = self.load(name)

        if category:
            tests = profile.get_tests_for_category(category)
            category_limit = profile.get_category_limit(category)
            if category_limit is not None:
                tests = tests[:category_limit]
        else:
            tests = profile.all_dynamic_tests

        if limit is not None:
            tests = tests[:limit]

        return tests


# Singleton instance for convenience
_default_loader: Optional[ProfileLoader] = None


def get_profile_loader() -> ProfileLoader:
    """Get the default profile loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = ProfileLoader()
    return _default_loader


def load_profile(name: str) -> TestProfile:
    """Convenience function to load a profile."""
    return get_profile_loader().load(name)


def list_profiles() -> List[str]:
    """Convenience function to list available profiles."""
    return get_profile_loader().list_profiles()
