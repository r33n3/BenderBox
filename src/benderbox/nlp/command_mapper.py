"""
Command Mapper for BenderBox NLP

Provides LLM-independent keyword/phrase mapping that reliably translates
natural language to BenderBox actions. Works entirely through string
matching and regex - no LLM inference required.

Architecture:
1. Synonym Expansion - Maps creative phrases to canonical actions
2. Fuzzy Matching - Handles common misspellings
3. Entity Extraction - Finds model names, profiles, file paths
4. Workflow Detection - Identifies multi-step processes
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# FUZZY MATCHING - Handle common misspellings
# =============================================================================

# Common misspellings mapped to correct words
COMMON_MISSPELLINGS: Dict[str, str] = {
    # Analyze variations
    "anaylze": "analyze", "analze": "analyze", "analize": "analyze",
    "analise": "analyze", "anaylse": "analyze", "analuze": "analyze",
    "reveiw": "review", "revew": "review", "reviw": "review",
    "cheeck": "check", "chek": "check", "cehck": "check",
    "examin": "examine", "examien": "examine",
    "inspct": "inspect", "inpsect": "inspect",
    "scna": "scan", "sacn": "scan",

    # Interrogate variations
    "intergate": "interrogate", "interogat": "interrogate",
    "interogate": "interrogate", "interrogat": "interrogate",
    "interragate": "interrogate", "iterrogate": "interrogate",
    "tset": "test", "tets": "test", "testt": "test",
    "securtiy": "security", "secuirty": "security", "securty": "security",
    "secutiry": "security", "seucrity": "security",

    # Compare variations
    "comprae": "compare", "comapre": "compare", "compre": "compare",
    "compar": "compare", "copmare": "compare",

    # Report variations
    "reprot": "report", "reoprt": "report", "repotr": "report",
    "repport": "report", "rpeort": "report", "repost": "report",
    "repor": "report", "rport": "report",
    "exprot": "export", "exoprt": "export", "exprt": "export",

    # Model variations
    "modle": "model", "mdoel": "model", "modl": "model",
    "modell": "model", "moel": "model",

    # Profile variations
    "profiel": "profile", "porfile": "profile", "profle": "profile",
    "proifle": "profile",

    # Other common words
    "vulnerabilty": "vulnerability", "vulnerablity": "vulnerability",
    "vulerability": "vulnerability",
    "jailbraek": "jailbreak", "jalibreak": "jailbreak",
    "adversrial": "adversarial", "advesarial": "adversarial",
    "comprehnsive": "comprehensive", "comprhensive": "comprehensive",

    # Target type misspellings
    "instrucions": "instructions", "intructions": "instructions",
    "instructinos": "instructions", "insturctions": "instructions",
    "promtp": "prompt", "promtps": "prompts", "prmopt": "prompt",
    "skil": "skill", "skils": "skills", "skiil": "skill",
    "serever": "server", "servr": "server", "sever": "server",
    "anlysis": "analysis", "anaylsis": "analysis", "analyis": "analysis",

    # Model name misspellings
    "tinylama": "tinyllama", "tinylalma": "tinyllama", "tinyalma": "tinyllama",
    "tinlyama": "tinyllama", "tinyllam": "tinyllama",
    "phi2": "phi-2", "phi 2": "phi-2",
    "llam": "llama", "lama": "llama", "llamaa": "llama",
    "qwne": "qwen", "qwn": "qwen",
    "mistrl": "mistral", "mitsral": "mistral",
}


def correct_spelling(text: str) -> str:
    """
    Correct common misspellings in text.

    Args:
        text: Input text possibly containing misspellings.

    Returns:
        Text with common misspellings corrected.
    """
    words = text.split()
    corrected = []

    for word in words:
        word_lower = word.lower()
        if word_lower in COMMON_MISSPELLINGS:
            # Preserve original case pattern
            correction = COMMON_MISSPELLINGS[word_lower]
            if word.isupper():
                correction = correction.upper()
            elif word[0].isupper():
                correction = correction.capitalize()
            corrected.append(correction)
        else:
            corrected.append(word)

    return " ".join(corrected)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Edit distance (number of insertions, deletions, substitutions).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def fuzzy_match_word(word: str, candidates: List[str], max_distance: int = 2) -> Optional[str]:
    """
    Find the best fuzzy match for a word among candidates.

    Args:
        word: Word to match.
        candidates: List of candidate words.
        max_distance: Maximum edit distance to consider a match.

    Returns:
        Best matching candidate, or None if no match within threshold.
    """
    word_lower = word.lower()
    best_match = None
    best_distance = max_distance + 1

    for candidate in candidates:
        candidate_lower = candidate.lower()

        # Exact match
        if word_lower == candidate_lower:
            return candidate

        # Skip if length difference is too large
        if abs(len(word_lower) - len(candidate_lower)) > max_distance:
            continue

        distance = levenshtein_distance(word_lower, candidate_lower)
        if distance <= max_distance and distance < best_distance:
            best_distance = distance
            best_match = candidate

    return best_match


# =============================================================================
# ACTION SYNONYMS - Map natural language to canonical actions
# =============================================================================

ACTION_SYNONYMS: Dict[str, List[str]] = {
    # Analyze/Review actions (ordered by specificity - longer phrases first)
    "analyze": [
        "take a look at", "look at", "check out", "check on",
        "is this model safe", "is it safe", "safe for production",
        "analyze", "analyse", "review", "check", "examine", "inspect",
        "scan", "evaluate", "assess", "audit", "investigate", "study",
    ],

    # Interrogate/Test actions (security testing)
    # Note: "red team", "jailbreak", "attack" etc. are in PROFILE_SYNONYMS for "adversarial"
    "interrogate": [
        "run security tests on", "security test", "security scan", "security tests",
        "run tests on", "run tests", "pen test", "pentest",
        "jailbreak tests", "jailbreak test", "safety tests", "safety test",
        "attack tests", "attack test", "aggressive tests", "aggressive test",
        "offensive tests", "offensive test", "hostile tests", "hostile test",
        "adversarial tests", "adversarial test",  # After profile synonym expansion
        "interrogate", "test", "probe",
    ],

    # Compare actions
    "compare": [
        "side by side", "which is better", "which is safer",
        "compare", "diff", "versus", "vs", "against", "contrast", "benchmark",
    ],

    # Report/Export actions
    "report": [
        "generate report", "create report", "make report", "final report",
        "generate a report", "create a report", "make a report",
        "report", "export", "save", "summary", "document", "output",
    ],

    # Load/Select actions
    "load": [
        "switch to", "load up",
        "load", "use", "select", "pick", "choose", "set",
    ],

    # List/Show actions
    "list": [
        "what do i have", "what models", "which models", "show me",
        "list", "show", "display", "what", "which", "available", "see", "view",
    ],

    # Download actions
    "download": [
        "download from huggingface", "get from huggingface", "fetch from huggingface",
        "download model", "get model", "fetch model", "pull model",
        "download", "get", "fetch", "pull", "install",
    ],

    # Explain actions
    "explain": [
        "tell me about", "what does", "what is", "why is", "how does",
        "explain", "why", "how", "describe", "elaborate",
    ],

    # Help actions
    "help": [
        "how do i", "how to", "how can i",
        "help", "guide", "tutorial", "instructions",
    ],

    # Unload actions
    "unload": [
        "unload", "remove", "clear", "close",
    ],

    # Status actions
    "status": [
        "what's loaded", "current status", "system status",
        "status", "state", "info",
    ],
}


# =============================================================================
# TARGET SYNONYMS - Map target references to canonical types
# =============================================================================

TARGET_SYNONYMS: Dict[str, List[str]] = {
    # NOTE: Do NOT include specific model names (phi, llama, etc.) here
    # as they will be replaced with "model" during synonym expansion.
    # Model names are detected separately in _extract_model_refs()
    "model": [
        "language model", "neural net", "neural network", "gguf model",
        "model", "llm", "ai", "gguf", "weights", "checkpoint",
    ],

    "mcp": [
        "mcp server", "tool server", "api server", "function server",
        "model context protocol", "tool api",
        "mcp", "server",
    ],

    "code": [
        "source code", "source file", "code file",
        "code", "file", "script", "source", "program", "function", "class",
        "python", "javascript", "typescript",
    ],

    "context": [
        "system prompt", "instruction file", "prompt file", "skill file",
        "context window", "context file", "instructions",
        "context", "prompt", "instruction", "skill", "template",
    ],

    "report": [
        "analysis report", "security report", "interrogation report",
        "scan results", "test results", "vulnerability report",
        "report", "analysis", "result", "findings", "assessment", "output",
    ],
}


# =============================================================================
# PROFILE SYNONYMS - Map profile descriptions to canonical names
# =============================================================================

PROFILE_SYNONYMS: Dict[str, List[str]] = {
    "quick": ["quick", "fast", "brief", "rapid", "short", "minimal", "basic"],
    "standard": ["standard", "normal", "default", "regular", "typical", "balanced", "medium"],
    "full": ["full", "complete", "comprehensive", "thorough", "deep", "extensive", "all", "everything"],
    "adversarial": ["adversarial", "attack", "jailbreak", "red team", "aggressive", "hostile", "offensive"],
}


# =============================================================================
# ENTITY PATTERNS - Regex for extracting specific entities
# =============================================================================

ENTITY_PATTERNS: Dict[str, re.Pattern] = {
    # File paths with extensions
    "file_path": re.compile(
        r"['\"]?([^\s'\"]+\.(gguf|py|js|ts|yaml|yml|json|md|txt))['\"]?",
        re.IGNORECASE
    ),

    # URL patterns
    "url": re.compile(
        r"(https?://[^\s]+)",
        re.IGNORECASE
    ),

    # Profile hints (standalone words and phrases)
    # Note: plurals (jailbreaks, attacks) handled by stripping trailing 's' in extraction
    "profile": re.compile(
        r"\b(quick|fast|standard|normal|full|complete|comprehensive|"
        r"deep|adversarial|attacks?|jailbreaks?|red\s+team|aggressive|hostile|offensive)\b",
        re.IGNORECASE
    ),

    # Output format hints
    "output_format": re.compile(
        r"\b(html|json|markdown|md|pdf|csv)\b",
        re.IGNORECASE
    ),

    # Report trigger words
    "wants_report": re.compile(
        r"\b(report|export|save|document|output|summary)\b",
        re.IGNORECASE
    ),

    # Open/view trigger words
    "wants_open": re.compile(
        r"\b(open|view|show|display|browser)\b.*\b(report|result|html)\b",
        re.IGNORECASE
    ),
}


# =============================================================================
# DATA CLASSES
# =============================================================================

class ActionType(Enum):
    """Canonical action types."""
    ANALYZE = "analyze"
    INTERROGATE = "interrogate"
    COMPARE = "compare"
    REPORT = "report"
    LOAD = "load"
    UNLOAD = "unload"
    LIST = "list"
    DOWNLOAD = "download"
    EXPLAIN = "explain"
    HELP = "help"
    STATUS = "status"
    UNKNOWN = "unknown"


class TargetType(Enum):
    """Canonical target types."""
    MODEL = "model"
    MCP = "mcp"
    CODE = "code"
    CONTEXT = "context"
    REPORT = "report"
    UNKNOWN = "unknown"


@dataclass
class ExtractedEntities:
    """Entities extracted from user query."""
    actions: List[ActionType] = field(default_factory=list)
    targets: List[TargetType] = field(default_factory=list)
    model_refs: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    profile: Optional[str] = None
    output_format: Optional[str] = None
    wants_report: bool = False
    wants_open: bool = False
    raw_query: str = ""
    normalized_query: str = ""

    def primary_action(self) -> ActionType:
        """Get the primary (first) action."""
        return self.actions[0] if self.actions else ActionType.UNKNOWN

    def primary_target(self) -> TargetType:
        """Get the primary (first) target type."""
        return self.targets[0] if self.targets else TargetType.UNKNOWN

    def primary_model(self) -> Optional[str]:
        """Get the primary model reference."""
        return self.model_refs[0] if self.model_refs else None


# =============================================================================
# COMMAND MAPPER CLASS
# =============================================================================

class CommandMapper:
    """
    Maps natural language queries to structured commands.

    Works entirely through string matching and regex patterns,
    without requiring LLM inference.
    """

    def __init__(self, model_manager=None):
        """
        Initialize CommandMapper.

        Args:
            model_manager: Optional ModelManager for fuzzy model matching.
        """
        self._model_manager = model_manager
        self._available_models: Optional[List[Dict]] = None

    @property
    def available_models(self) -> List[Dict]:
        """Lazy-load available models from all locations."""
        if self._available_models is None:
            if self._model_manager:
                # Use get_all_models() for comprehensive search
                self._available_models = self._model_manager.get_all_models()
            else:
                self._available_models = []
        return self._available_models

    def expand_synonyms(self, query: str) -> str:
        """
        Expand synonyms in query to canonical forms.

        Also corrects common misspellings before synonym expansion.

        Args:
            query: Raw user query.

        Returns:
            Query with misspellings corrected and synonyms replaced by canonical terms.
        """
        # First, correct common misspellings
        corrected = correct_spelling(query)
        normalized = corrected.lower()

        # Expand action synonyms (longer phrases first to avoid partial matches)
        for canonical, synonyms in ACTION_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in normalized and synonym != canonical:
                    # Use word boundary replacement to avoid partial matches
                    # e.g., don't replace "file" in "profile"
                    pattern = re.compile(r"\b" + re.escape(synonym) + r"\b", re.IGNORECASE)
                    if pattern.search(normalized):
                        normalized = pattern.sub(canonical, normalized)
                        break  # Only replace first matching synonym per canonical

        # Expand target synonyms
        for canonical, synonyms in TARGET_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in normalized and synonym != canonical:
                    # Use word boundaries to prevent partial word matches
                    pattern = re.compile(r"\b" + re.escape(synonym) + r"\b", re.IGNORECASE)
                    if pattern.search(normalized):
                        normalized = pattern.sub(canonical, normalized)
                        break

        # Expand profile synonyms
        for canonical, synonyms in PROFILE_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in normalized and synonym != canonical:
                    pattern = re.compile(r"\b" + re.escape(synonym) + r"\b", re.IGNORECASE)
                    normalized = pattern.sub(canonical, normalized)
                    break

        return normalized

    def extract_entities(self, query: str) -> ExtractedEntities:
        """
        Extract all entities from a user query.

        Args:
            query: Raw user query.

        Returns:
            ExtractedEntities with all found entities.
        """
        entities = ExtractedEntities(raw_query=query)

        # IMPORTANT: Extract profile from raw query BEFORE synonym expansion
        # This ensures profile-indicating words like "jailbreak" in "jailbreak tests"
        # are detected before they get replaced by action synonyms like "interrogate"
        raw_lower = query.lower()
        profile_match = ENTITY_PATTERNS["profile"].search(raw_lower)
        if profile_match:
            # Normalize whitespace in profile word (e.g., "red  team" -> "red team")
            profile_word = " ".join(profile_match.group(1).lower().split())
            # Normalize plural forms (e.g., "jailbreaks" -> "jailbreak", "attacks" -> "attack")
            if profile_word.endswith('s') and profile_word not in ['fast']:
                singular = profile_word[:-1]
                # Check if singular form is in synonyms
                for canonical, synonyms in PROFILE_SYNONYMS.items():
                    if singular in synonyms:
                        profile_word = singular
                        break
            # Map to canonical profile name
            for canonical, synonyms in PROFILE_SYNONYMS.items():
                if profile_word in synonyms:
                    entities.profile = canonical
                    break

        # Now normalize the query (synonym expansion)
        normalized = self.expand_synonyms(query)
        entities.normalized_query = normalized

        # Extract actions
        for canonical, synonyms in ACTION_SYNONYMS.items():
            # Check if canonical form is in normalized query
            if re.search(r"\b" + canonical + r"\b", normalized):
                try:
                    action = ActionType(canonical)
                    if action not in entities.actions:
                        entities.actions.append(action)
                except ValueError:
                    pass

        # Extract target types
        for canonical, synonyms in TARGET_SYNONYMS.items():
            if re.search(r"\b" + canonical + r"\b", normalized):
                try:
                    target = TargetType(canonical)
                    if target not in entities.targets:
                        entities.targets.append(target)
                except ValueError:
                    pass

        # Extract file paths
        for match in ENTITY_PATTERNS["file_path"].finditer(query):
            entities.file_paths.append(match.group(1))

        # Extract URLs
        for match in ENTITY_PATTERNS["url"].finditer(query):
            entities.urls.append(match.group(1))

        # Extract output format
        format_match = ENTITY_PATTERNS["output_format"].search(query)
        if format_match:
            entities.output_format = format_match.group(1).lower()

        # Check for report/open intentions
        entities.wants_report = bool(ENTITY_PATTERNS["wants_report"].search(normalized))
        entities.wants_open = bool(ENTITY_PATTERNS["wants_open"].search(query))

        # Extract model references (fuzzy match against available models)
        entities.model_refs = self._extract_model_refs(query, normalized)

        return entities

    def _extract_model_refs(self, raw_query: str, normalized_query: str) -> List[str]:
        """
        Extract model references from query using fuzzy matching.

        Args:
            raw_query: Original user query.
            normalized_query: Synonym-expanded query.

        Returns:
            List of resolved model paths or names.
        """
        model_refs = []

        # First check for explicit .gguf files
        for match in re.finditer(r"([^\s]+\.gguf)", raw_query, re.IGNORECASE):
            model_refs.append(match.group(1))

        if model_refs:
            return model_refs

        # Extract potential model names (words after "model" or standalone identifiers)
        # Pattern: "model <name>" or "<name> model" or "on <name>" or just potential model names
        potential_names = []

        # Look for "model X" pattern
        model_name_match = re.search(r"\bmodel\s+([a-zA-Z0-9_\-\.]+)", normalized_query)
        if model_name_match:
            potential_names.append(model_name_match.group(1))

        # Look for "X model" pattern
        name_model_match = re.search(r"([a-zA-Z0-9_\-\.]+)\s+model\b", normalized_query)
        if name_model_match:
            potential_names.append(name_model_match.group(1))

        # Look for "on X" pattern (e.g., "run tests on phi-2")
        on_pattern_match = re.search(r"\bon\s+([a-zA-Z0-9_\-\.]+)(?:\s|$)", normalized_query)
        if on_pattern_match:
            potential_names.append(on_pattern_match.group(1))

        # Look for "analyze/test/interrogate X" pattern (action followed by model name)
        action_target_match = re.search(
            r"\b(?:analyze|test|interrogate|check|scan|probe)\s+([a-zA-Z0-9_\-\.]+)(?:\s|$)",
            normalized_query
        )
        if action_target_match:
            potential_names.append(action_target_match.group(1))

        # Try to fuzzy match against available models
        for name in potential_names:
            resolved = self._fuzzy_match_model(name)
            if resolved and resolved not in model_refs:
                model_refs.append(resolved)

        # If still no matches, try harder - look for known model name patterns
        # Use allow_unverified=True to keep model names even if not found on disk
        if not model_refs:
            known_patterns = [
                r"\b(phi[\-_]?2)\b",
                r"\b(phi2)\b",
                r"\b(llama[\-_]?[0-9]+[a-z]*)\b",  # llama-7b, llama2, llama-13b
                r"\b(tinyllama)\b",
                r"\b(tiny[\-_]?llama)\b",
                r"\b(mistral[\-_]?[0-9]*[a-z]*)\b",  # mistral, mistral-7b
                r"\b(qwen[\-_]?[0-9]*[a-z]*)\b",
                r"\b(smol[\-_]?lm[\-_]?[0-9]*)\b",
                r"\b(vicuna[\-_]?[0-9]*)\b",
                r"\b(gemma[\-_]?[0-9]*[a-z]*)\b",
                r"\b(deepseek[\-_]?[a-z0-9]*)\b",
            ]
            for pattern in known_patterns:
                match = re.search(pattern, normalized_query, re.IGNORECASE)
                if match:
                    resolved = self._fuzzy_match_model(match.group(1), allow_unverified=True)
                    if resolved and resolved not in model_refs:
                        model_refs.append(resolved)

        return model_refs

    def _fuzzy_match_model(self, name: str, allow_unverified: bool = False) -> Optional[str]:
        """
        Fuzzy match a model name against available models.

        Uses ModelManager.find_model_by_name() for comprehensive search
        across all model directories.

        Args:
            name: Partial or full model name.
            allow_unverified: If True, return the name as-is if it looks like
                a valid model identifier but can't be verified on disk.

        Returns:
            Full path to matching model, or the name itself if allow_unverified
            and it looks like a model name, or None if not found.
        """
        if not name:
            return None

        name_lower = name.lower().strip()

        # Skip common words that aren't model names
        # Also skip profile names to avoid "analyze full" treating "full" as model
        skip_words = {
            "the", "a", "an", "this", "that", "with", "and", "or", "to", "for",
            "on", "in", "at", "by", "my", "our", "its", "is", "be", "it",
            "profile", "tests", "test", "model", "models", "run", "check",
            # Profile names - should not be treated as model names
            "quick", "fast", "standard", "normal", "full", "complete",
            "comprehensive", "deep", "adversarial", "attack", "jailbreak",
            "aggressive", "hostile", "offensive",
        }
        if name_lower in skip_words:
            return None

        # Use model manager's comprehensive find if available
        if self._model_manager:
            result = self._model_manager.find_model_by_name(name)
            if result:
                return str(result)

        # Fallback to local search of cached models
        for model in self.available_models:
            model_name = Path(model["path"]).stem.lower()
            filename = Path(model["path"]).name.lower()

            # Exact stem match
            if model_name == name_lower:
                return model["path"]

            # Exact filename match
            if filename == name_lower or filename == name_lower + ".gguf":
                return model["path"]

            # Fuzzy match (name contained in model name)
            if name_lower in model_name:
                return model["path"]

        # If allow_unverified and name looks like a model identifier, return it as-is
        if allow_unverified:
            # Check if it looks like a model name (contains alphanumeric, not just stop words)
            if re.match(r'^[a-zA-Z][a-zA-Z0-9_\-\.]*$', name) and len(name) >= 2:
                return name

        return None

    def parse(self, query: str) -> ExtractedEntities:
        """
        Parse a user query into structured entities.

        This is the main entry point for the command mapper.

        Args:
            query: Raw user query.

        Returns:
            ExtractedEntities with all parsed information.
        """
        return self.extract_entities(query)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_profile(profile_hint: str) -> str:
    """
    Normalize a profile hint to a canonical profile name.

    Args:
        profile_hint: User-provided profile description.

    Returns:
        Canonical profile name (quick, standard, full, adversarial).
    """
    hint_lower = profile_hint.lower().strip()

    for canonical, synonyms in PROFILE_SYNONYMS.items():
        if hint_lower in synonyms:
            return canonical

    return "standard"  # Default


def get_action_synonyms(action: str) -> List[str]:
    """Get all synonyms for an action."""
    return ACTION_SYNONYMS.get(action, [])


def get_target_synonyms(target: str) -> List[str]:
    """Get all synonyms for a target type."""
    return TARGET_SYNONYMS.get(target, [])
