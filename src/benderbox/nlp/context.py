"""
Context Management for BenderBox

Tracks conversation state, resolves references, and maintains
analysis history for multi-turn conversations.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Stored analysis result with metadata."""

    result: Dict[str, Any]
    target_name: str
    target_type: str  # "model", "mcp_server", "skill"
    timestamp: datetime
    profile: str
    risk_level: str
    risk_score: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """Create from analysis result dictionary."""
        summary = data.get("summary", {})
        risk = summary.get("risk", {})

        return cls(
            result=data,
            target_name=data.get("target_name", "unknown"),
            target_type=data.get("target_type", "unknown"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            profile=data.get("profile", "standard"),
            risk_level=risk.get("level", "unknown"),
            risk_score=risk.get("score", 0.0),
        )


@dataclass
class ConversationContext:
    """Current conversation context."""

    # Current target being discussed
    current_target: Optional[str] = None
    current_target_type: Optional[str] = None

    # Most recent analysis
    last_analysis: Optional[AnalysisResult] = None

    # Analysis history (most recent first)
    analysis_history: List[AnalysisResult] = field(default_factory=list)

    # Referenced findings from current conversation
    referenced_findings: List[Dict[str, Any]] = field(default_factory=list)

    # Conversation metadata
    started_at: datetime = field(default_factory=datetime.now)
    turn_count: int = 0


# Reference resolution patterns
REFERENCE_PATTERNS = {
    # Direct references
    "it": re.compile(r"\bit\b", re.I),
    "this": re.compile(r"\bthis\b", re.I),
    "that": re.compile(r"\bthat\b", re.I),
    "the_model": re.compile(r"\bthe\s+model\b", re.I),
    "the_server": re.compile(r"\bthe\s+(server|mcp)\b", re.I),
    "the_skill": re.compile(r"\bthe\s+skill\b", re.I),

    # Previous reference
    "previous": re.compile(r"\b(previous|last|prior)\s+(analysis|scan|result|report)\b", re.I),

    # Comparison references
    "first": re.compile(r"\b(first|former)\s+(one|model|result)\b", re.I),
    "second": re.compile(r"\b(second|latter)\s+(one|model|result)\b", re.I),
}


class ContextManager:
    """
    Manages conversation context and reference resolution.

    Tracks:
    - Current target being discussed
    - Analysis history
    - Referenced findings
    - Conversation state
    """

    MAX_HISTORY = 10  # Maximum analysis results to keep

    def __init__(self):
        """Initialize ContextManager."""
        self._context = ConversationContext()

    @property
    def context(self) -> ConversationContext:
        """Get current context."""
        return self._context

    def get_current_target(self) -> Optional[str]:
        """Get the current target being discussed."""
        return self._context.current_target

    def get_current_target_type(self) -> Optional[str]:
        """Get the type of current target."""
        return self._context.current_target_type

    def set_current_target(self, target: str, target_type: str) -> None:
        """
        Set the current target.

        Args:
            target: Target path or name.
            target_type: Type of target (model, mcp_server, skill).
        """
        self._context.current_target = target
        self._context.current_target_type = target_type
        logger.debug(f"Set current target: {target} ({target_type})")

    def add_analysis_result(self, result: Dict[str, Any]) -> None:
        """
        Add an analysis result to history.

        Args:
            result: Analysis result dictionary.
        """
        analysis = AnalysisResult.from_dict(result)

        # Update current target
        self._context.current_target = analysis.target_name
        self._context.current_target_type = analysis.target_type

        # Set as last analysis
        self._context.last_analysis = analysis

        # Add to history (most recent first)
        self._context.analysis_history.insert(0, analysis)

        # Trim history if needed
        if len(self._context.analysis_history) > self.MAX_HISTORY:
            self._context.analysis_history = self._context.analysis_history[:self.MAX_HISTORY]

        logger.debug(f"Added analysis result: {analysis.target_name}")

    def get_last_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis result."""
        if self._context.last_analysis:
            return self._context.last_analysis.result
        return None

    def get_recent_analyses(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent analysis results.

        Args:
            limit: Maximum number of results.

        Returns:
            List of analysis result dictionaries.
        """
        return [a.result for a in self._context.analysis_history[:limit]]

    def get_analyses_by_type(self, target_type: str) -> List[Dict[str, Any]]:
        """
        Get analyses by target type.

        Args:
            target_type: Type of target.

        Returns:
            List of matching analysis results.
        """
        return [
            a.result for a in self._context.analysis_history
            if a.target_type == target_type
        ]

    def resolve_reference(self, text: str) -> Optional[str]:
        """
        Resolve a reference in text to an actual target.

        Args:
            text: Text containing potential reference.

        Returns:
            Resolved target name, or None if no resolution.
        """
        text_lower = text.lower()

        # Check for direct references to current target
        if REFERENCE_PATTERNS["it"].search(text_lower):
            return self._context.current_target
        if REFERENCE_PATTERNS["this"].search(text_lower):
            return self._context.current_target
        if REFERENCE_PATTERNS["that"].search(text_lower):
            return self._context.current_target

        # Check for type-specific references
        if REFERENCE_PATTERNS["the_model"].search(text_lower):
            # Find most recent model analysis
            for a in self._context.analysis_history:
                if a.target_type == "model":
                    return a.target_name
            return self._context.current_target

        if REFERENCE_PATTERNS["the_server"].search(text_lower):
            for a in self._context.analysis_history:
                if a.target_type in ("mcp_server", "infrastructure"):
                    return a.target_name
            return self._context.current_target

        if REFERENCE_PATTERNS["the_skill"].search(text_lower):
            for a in self._context.analysis_history:
                if a.target_type == "skill":
                    return a.target_name
            return self._context.current_target

        # Check for "previous analysis" references
        if REFERENCE_PATTERNS["previous"].search(text_lower):
            if self._context.last_analysis:
                return self._context.last_analysis.target_name

        # Check for ordinal references (first, second)
        if REFERENCE_PATTERNS["first"].search(text_lower):
            if len(self._context.analysis_history) >= 1:
                # First in recent order is most recent
                return self._context.analysis_history[0].target_name

        if REFERENCE_PATTERNS["second"].search(text_lower):
            if len(self._context.analysis_history) >= 2:
                return self._context.analysis_history[1].target_name

        return None

    def resolve_query_references(self, query: str) -> str:
        """
        Resolve all references in a query, replacing with actual values.

        Args:
            query: User query with potential references.

        Returns:
            Query with references resolved.
        """
        resolved = query

        # Try to resolve and replace common references
        target = self.resolve_reference(query)
        if target and self._context.current_target:
            # Replace generic references with specific target
            patterns_to_replace = [
                (r"\bit\b", target),
                (r"\bthe model\b", target),
                (r"\bthe server\b", target),
                (r"\bthat\b", target),
            ]
            for pattern, replacement in patterns_to_replace:
                if re.search(pattern, query, re.I):
                    # Only replace if context supports it
                    resolved = re.sub(pattern, replacement, resolved, flags=re.I)
                    break

        return resolved

    def add_referenced_finding(self, finding: Dict[str, Any]) -> None:
        """
        Add a finding to the referenced findings list.

        Args:
            finding: Finding dictionary.
        """
        self._context.referenced_findings.append(finding)
        # Keep only recent findings
        if len(self._context.referenced_findings) > 20:
            self._context.referenced_findings = self._context.referenced_findings[-20:]

    def get_referenced_findings(self) -> List[Dict[str, Any]]:
        """Get all referenced findings."""
        return self._context.referenced_findings

    def increment_turn(self) -> int:
        """Increment and return turn count."""
        self._context.turn_count += 1
        return self._context.turn_count

    def clear(self) -> None:
        """Clear all context."""
        self._context = ConversationContext()
        logger.debug("Context cleared")

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize context to dictionary.

        Returns:
            Serialized context.
        """
        return {
            "current_target": self._context.current_target,
            "current_target_type": self._context.current_target_type,
            "last_analysis": self._context.last_analysis.result if self._context.last_analysis else None,
            "analysis_history": [a.result for a in self._context.analysis_history],
            "referenced_findings": self._context.referenced_findings,
            "started_at": self._context.started_at.isoformat(),
            "turn_count": self._context.turn_count,
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """
        Restore context from dictionary.

        Args:
            data: Serialized context data.
        """
        self._context.current_target = data.get("current_target")
        self._context.current_target_type = data.get("current_target_type")

        if data.get("last_analysis"):
            self._context.last_analysis = AnalysisResult.from_dict(data["last_analysis"])

        self._context.analysis_history = [
            AnalysisResult.from_dict(r) for r in data.get("analysis_history", [])
        ]

        self._context.referenced_findings = data.get("referenced_findings", [])
        self._context.turn_count = data.get("turn_count", 0)

        if data.get("started_at"):
            self._context.started_at = datetime.fromisoformat(data["started_at"])

        logger.debug("Context restored from serialized data")

    def get_summary(self) -> str:
        """
        Get a summary of current context.

        Returns:
            Human-readable context summary.
        """
        lines = []

        if self._context.current_target:
            lines.append(f"Current target: {self._context.current_target} ({self._context.current_target_type})")

        if self._context.last_analysis:
            lines.append(f"Last analysis: {self._context.last_analysis.target_name} "
                        f"({self._context.last_analysis.risk_level})")

        lines.append(f"Analysis history: {len(self._context.analysis_history)} results")
        lines.append(f"Conversation turns: {self._context.turn_count}")

        return "\n".join(lines) if lines else "No context available"
