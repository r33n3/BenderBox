"""
Prompt schema definitions for model interrogation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class PromptCategory(Enum):
    """Categories of test prompts."""

    BASELINE_SAFETY = "baseline_safety"
    JAILBREAK_ATTEMPTS = "jailbreak_attempts"
    HARMFUL_INSTRUCTIONS = "harmful_instructions"
    BIAS_PROBING = "bias_probing"
    PRIVACY_EXTRACTION = "privacy_extraction"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    BENIGN_BASELINE = "benign_baseline"


class ExpectedBehavior(Enum):
    """Expected model behavior for a prompt."""

    REFUSE = "refuse"  # Model should refuse to answer
    COMPLY = "comply"  # Model should provide an answer
    PARTIAL = "partial"  # Model may partially comply with caveats
    ANY = "any"  # No specific expectation


class Severity(Enum):
    """Severity level of a test prompt."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TestPrompt:
    """A single test prompt for model interrogation."""

    id: str
    prompt: str
    category: PromptCategory
    expected_behavior: ExpectedBehavior
    severity: Severity = Severity.MEDIUM
    subcategory: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)
    bypass_technique: str = ""  # For jailbreak prompts
    source: str = ""  # Where this prompt came from

    def __post_init__(self):
        """Convert string values to enums if needed."""
        if isinstance(self.category, str):
            self.category = PromptCategory(self.category)
        if isinstance(self.expected_behavior, str):
            self.expected_behavior = ExpectedBehavior(self.expected_behavior)
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "category": self.category.value,
            "expected_behavior": self.expected_behavior.value,
            "severity": self.severity.value,
            "subcategory": self.subcategory,
            "description": self.description,
            "tags": self.tags,
            "variants": self.variants,
            "bypass_technique": self.bypass_technique,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestPrompt":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            category=data.get("category", PromptCategory.BASELINE_SAFETY),
            expected_behavior=data.get("expected_behavior", ExpectedBehavior.REFUSE),
            severity=data.get("severity", Severity.MEDIUM),
            subcategory=data.get("subcategory", ""),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            variants=data.get("variants", []),
            bypass_technique=data.get("bypass_technique", ""),
            source=data.get("source", ""),
        )


@dataclass
class PromptSet:
    """A collection of related prompts."""

    name: str
    description: str
    category: PromptCategory
    prompts: List[TestPrompt] = field(default_factory=list)
    version: str = "1.0.0"

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts)
