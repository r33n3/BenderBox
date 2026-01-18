"""
Profile Loader for BenderBox

Loads and manages YAML-based interrogation profiles for customizable testing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import yaml
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. YAML profiles will not be available.")


@dataclass
class CustomPrompt:
    """A custom prompt defined in a profile."""

    id: str
    prompt: str
    category: str
    expected_behavior: str = "refuse"
    severity: str = "medium"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "category": self.category,
            "expected_behavior": self.expected_behavior,
            "severity": self.severity,
            "tags": self.tags,
        }


@dataclass
class ProfileConfig:
    """Configuration loaded from a YAML profile."""

    name: str
    description: str = ""
    version: str = "1.0"
    estimated_tests: int = 50
    target_type: str = "model"  # 'model' or 'mcp_server'

    # Category configuration: category_name -> limit (int or 'all')
    categories: Dict[str, Union[int, str]] = field(default_factory=dict)

    # Scoring weights per category
    scoring_weights: Dict[str, float] = field(default_factory=dict)

    # Risk thresholds
    risk_thresholds: Dict[str, int] = field(default_factory=dict)

    # Tag filters
    include_tags: List[str] = field(default_factory=list)
    exclude_tags: List[str] = field(default_factory=list)

    # Custom prompts defined in the profile (legacy format)
    custom_prompts: List[CustomPrompt] = field(default_factory=list)

    # Dynamic tests defined in the profile (new format)
    dynamic_tests: List[CustomPrompt] = field(default_factory=list)

    # Variable substitutions
    variables: Dict[str, str] = field(default_factory=dict)

    # Source file path
    source_path: Optional[Path] = None

    @property
    def all_prompts(self) -> List[CustomPrompt]:
        """Get all prompts (both custom_prompts and dynamic_tests)."""
        return self.custom_prompts + self.dynamic_tests

    def get_category_limit(self, category: str) -> Optional[int]:
        """
        Get the test limit for a category.

        Returns:
            None if 'all' (no limit), or int for specific limit.
        """
        limit = self.categories.get(category, 0)
        if limit == "all":
            return None
        return int(limit) if limit else 0

    def has_category(self, category: str) -> bool:
        """Check if category is configured in this profile."""
        return category in self.categories

    @classmethod
    def from_dict(cls, data: Dict[str, Any], source_path: Optional[Path] = None) -> "ProfileConfig":
        """Create ProfileConfig from dictionary."""
        # Parse custom prompts (legacy format)
        custom_prompts = []
        for prompt_data in data.get("custom_prompts", []):
            custom_prompts.append(CustomPrompt(
                id=prompt_data.get("id", "unknown"),
                prompt=prompt_data.get("prompt", ""),
                category=prompt_data.get("category", "baseline_safety"),
                expected_behavior=prompt_data.get("expected_behavior", "refuse"),
                severity=prompt_data.get("severity", "medium"),
                tags=prompt_data.get("tags", []),
            ))

        # Parse dynamic tests (new format)
        dynamic_tests = []
        for test_data in data.get("dynamic_tests", []):
            dynamic_tests.append(CustomPrompt(
                id=test_data.get("id", "unknown"),
                prompt=test_data.get("prompt", ""),
                category=test_data.get("category", "baseline_safety"),
                expected_behavior=test_data.get("expected_behavior", "refuse"),
                severity=test_data.get("severity", "medium"),
                tags=test_data.get("tags", []),
            ))

        return cls(
            name=data.get("name", "unknown"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            estimated_tests=data.get("estimated_tests", 50),
            target_type=data.get("target_type", "model"),
            categories=data.get("categories", {}),
            scoring_weights=data.get("scoring_weights", {}),
            risk_thresholds=data.get("risk_thresholds", {}),
            include_tags=data.get("include_tags", []),
            exclude_tags=data.get("exclude_tags", []),
            custom_prompts=custom_prompts,
            dynamic_tests=dynamic_tests,
            variables=data.get("variables", {}),
            source_path=source_path,
        )


class ProfileLoader:
    """
    Loads interrogation profiles from YAML files.

    Profiles can be:
    - Built-in (from configs/profiles/)
    - Custom (from configs/profiles/custom/)
    - MCP-specific (from configs/mcp-profiles/)
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize ProfileLoader.

        Args:
            base_path: Base path for config files. Defaults to project root.
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Find project root (look for configs directory)
            self.base_path = self._find_project_root()

        self.profiles_dir = self.base_path / "configs" / "profiles"
        self.custom_dir = self.profiles_dir / "custom"
        self.mcp_dir = self.base_path / "configs" / "mcp-profiles"

        self._cache: Dict[str, ProfileConfig] = {}

    def _find_project_root(self) -> Path:
        """Find project root by looking for configs directory."""
        # Start from this file's location
        current = Path(__file__).resolve()

        # Walk up looking for configs directory
        for parent in [current] + list(current.parents):
            if (parent / "configs" / "profiles").exists():
                return parent
            if (parent / "src" / "benderbox").exists():
                return parent

        # Fallback to current working directory
        return Path.cwd()

    def list_profiles(self) -> List[str]:
        """
        List all available profile names.

        Returns:
            List of profile names (without .yaml extension).
        """
        profiles = []

        # Built-in profiles
        if self.profiles_dir.exists():
            for yaml_file in self.profiles_dir.glob("*.yaml"):
                profiles.append(yaml_file.stem)

        # Custom profiles (prefixed with 'custom/')
        if self.custom_dir.exists():
            for yaml_file in self.custom_dir.glob("*.yaml"):
                if yaml_file.stem != "example":  # Skip example template
                    profiles.append(f"custom/{yaml_file.stem}")

        return sorted(profiles)

    def list_mcp_profiles(self) -> List[str]:
        """
        List all available MCP profile names.

        Returns:
            List of MCP profile names.
        """
        profiles = []

        if self.mcp_dir.exists():
            for yaml_file in self.mcp_dir.glob("*.yaml"):
                profiles.append(yaml_file.stem)

        return sorted(profiles)

    def load(self, profile_name: str) -> ProfileConfig:
        """
        Load a profile by name.

        Args:
            profile_name: Profile name (e.g., 'adversarial', 'custom/my-profile')

        Returns:
            ProfileConfig object.

        Raises:
            FileNotFoundError: If profile doesn't exist.
            ValueError: If YAML is invalid.
        """
        # Check cache
        if profile_name in self._cache:
            return self._cache[profile_name]

        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML profiles. Install with: pip install pyyaml")

        # Determine file path
        if profile_name.startswith("custom/"):
            yaml_path = self.custom_dir / f"{profile_name[7:]}.yaml"
        elif profile_name.startswith("mcp/"):
            yaml_path = self.mcp_dir / f"{profile_name[4:]}.yaml"
        else:
            yaml_path = self.profiles_dir / f"{profile_name}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_name} (looked in {yaml_path})")

        # Load and parse YAML
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise ValueError(f"Invalid profile format in {yaml_path}")

            config = ProfileConfig.from_dict(data, source_path=yaml_path)

            # Cache it
            self._cache[profile_name] = config

            logger.info(f"Loaded profile '{profile_name}' from {yaml_path}")
            return config

        except yaml.YAMLError as e:
            raise ValueError(f"YAML parse error in {yaml_path}: {e}")

    def load_mcp_profile(self, profile_name: str) -> ProfileConfig:
        """
        Load an MCP-specific profile.

        Args:
            profile_name: MCP profile name (e.g., 'security')

        Returns:
            ProfileConfig object.
        """
        return self.load(f"mcp/{profile_name}")

    def get_or_default(self, profile_name: str) -> ProfileConfig:
        """
        Get a profile, falling back to 'standard' if not found.

        Args:
            profile_name: Profile name to load.

        Returns:
            ProfileConfig object.
        """
        try:
            return self.load(profile_name)
        except FileNotFoundError:
            logger.warning(f"Profile '{profile_name}' not found, using built-in defaults")
            return self._get_builtin_default(profile_name)

    def _get_builtin_default(self, profile_name: str) -> ProfileConfig:
        """
        Get a built-in default profile config (no YAML file needed).

        This provides backwards compatibility when YAML files don't exist.
        """
        defaults = {
            "quick": {
                "name": "quick",
                "description": "Fast screening profile",
                "estimated_tests": 16,
                "categories": {
                    "baseline_safety": 5,
                    "jailbreak_attempts": 5,
                    "benign_baseline": 5,
                    "harmful_instructions": 1,
                },
            },
            "standard": {
                "name": "standard",
                "description": "Balanced coverage profile",
                "estimated_tests": 50,
                "categories": {
                    "baseline_safety": 15,
                    "jailbreak_attempts": 15,
                    "harmful_instructions": 10,
                    "bias_probing": 5,
                    "benign_baseline": 5,
                },
            },
            "full": {
                "name": "full",
                "description": "Exhaustive audit profile",
                "estimated_tests": 100,
                "categories": {
                    "baseline_safety": "all",
                    "jailbreak_attempts": "all",
                    "harmful_instructions": "all",
                    "bias_probing": "all",
                    "privacy_extraction": "all",
                    "system_prompt_leak": "all",
                    "benign_baseline": "all",
                },
            },
            "adversarial": {
                "name": "adversarial",
                "description": "Jailbreak resistance and security testing",
                "estimated_tests": 64,
                "categories": {
                    "jailbreak_attempts": "all",
                    "harmful_instructions": "all",
                    "system_prompt_leak": "all",
                    "privacy_extraction": "all",
                    "baseline_safety": 5,
                    "benign_baseline": 5,
                },
            },
        }

        if profile_name in defaults:
            return ProfileConfig.from_dict(defaults[profile_name])

        # Unknown profile - use standard
        logger.warning(f"Unknown profile '{profile_name}', using standard defaults")
        return ProfileConfig.from_dict(defaults["standard"])

    def reload(self, profile_name: str) -> ProfileConfig:
        """
        Force reload a profile from disk (bypass cache).

        Args:
            profile_name: Profile name to reload.

        Returns:
            ProfileConfig object.
        """
        if profile_name in self._cache:
            del self._cache[profile_name]
        return self.load(profile_name)

    def clear_cache(self) -> None:
        """Clear the profile cache."""
        self._cache.clear()
