"""
Bender Persona Renderer for BenderBox

Adds Bender-style personality to analysis output while maintaining
structured, actionable data. Comedy second, results first.

Output Contract:
1. Badge (severity + category)
2. One-line Bender reaction (max 1 line)
3. Actionable facts (structured, scannable)
4. Next steps (imperative verbs)
5. Optional recovery stinger (end of scan only)
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(Enum):
    """Risk severity levels."""
    LOW = "LOW"
    MEDIUM = "MED"
    HIGH = "HIGH"
    CRITICAL = "CRIT"


class PoseType(Enum):
    """ASCII art pose types for visual flair."""
    # Recovery stingers (end of scan)
    BEER_SIP = "beer_sip"
    SMOKE_BREAK = "smoke_break"
    POWERDOWN = "powerdown"

    # Finding-type poses
    SECRET_SNATCH = "secret_snatch"
    CROWN_MODE = "crown_mode"
    TONGS_HANDSHAKE = "tongs_handshake"
    UMBRELLA_SHIELD = "umbrella_shield"
    NOPE_STAMP = "nope_stamp"

    # Status poses
    BORED = "bored"
    SUSPICIOUS = "suspicious"
    STRESSED = "stressed"
    SERIOUS = "serious"


# ASCII Art Poses - Braille-style matching main BenderBox art
ASCII_POSES: Dict[PoseType, str] = {
    PoseType.BEER_SIP: """
    ⢀⣤⣤⡀
   ⢠⣿⣿⣿⣧   ⣴⣶⡄
   ⢸⣿⠛⠛⣿⡄  ⣿⣿⡇
   ⠈⣿⣄⣠⣿⠃  ⣿⣿⡇
    ⠈⠛⠛⠁ ⠐⠒⠛⠛⠁
     ⣿⣿⣿⡟
    ⢠⡟  ⢻⡄
""",
    PoseType.SMOKE_BREAK: """
    ⢀⣤⣤⡀  ~~~
   ⢠⣿⣿⣿⣧
   ⢸⣿⠒⠒⣿⡄  ⣤
   ⠈⣿⣄⣠⣿⠃
    ⠈⠛⠛⠁
     ⣿⣿⣿⡟
    ⢠⡟  ⢻⡄
""",
    PoseType.POWERDOWN: """
    ⢀⣤⣤⡀ zZ
   ⢠⣿⣿⣿⣧
   ⢸⣿⠒⠒⣿⡄
   ⠈⣿⣄⣠⣿⠃
    ⠈⠛⠛⠁
     ⣿⣿⣿⡟
    ⣠⡟⠀⠀⢻⣄
""",
    PoseType.SECRET_SNATCH: """
    ⢀⣤⣤⡀
   ⢠⣿⣿⣿⣧
   ⢸⣿⡿⠃⣿⡄
   ⠈⣿⡗⢲⣿⠃
    ⠈⣿⣿⠁
     ⣿⣿⣿⡟
    ⢠⡟  ⢻⡄
""",
    PoseType.CROWN_MODE: """
      ⣿
     ⢿⣿⡿
    ⢀⣤⣤⡀
   ⢠⣿⣿⣿⣧
   ⢸⣿⠛⠛⣿⡄
   ⠈⣿⣄⣠⣿⠃
    ⠈⠛⠛⠁
     ⣿⣿⣿⡟
""",
    PoseType.TONGS_HANDSHAKE: """
    ⢀⣤⣤⡀
   ⢠⣿⣿⣿⣧
   ⢸⣿⠒⠒⣿⡄⠀⠀⠀⣀⣀
   ⠈⣿⣄⣠⣿⠃════⣿
    ⠈⠛⠛⠁
     ⣿⣿⣿⡟
    ⢠⡟  ⢻⡄
""",
    PoseType.UMBRELLA_SHIELD: """
      ⣠⣶⣶⣄
    ⢀⣿⣿⣿⣿⡀
   ⢠⣿⣤⣤⣿⣧
   ⢸⣿⣿⣿⣿⣿⡄
   ⠈⣿⣄⣠⣿⠃
    ⠈⠛⠛⠁
     ⣿⣿⣿⡟
    ⢠⡟  ⢻⡄
""",
    PoseType.NOPE_STAMP: """
    ⢀⣤⣤⡀
   ⢠⣿⣿⣿⣧ ⣿⣿⣿⣿⣿
   ⢸⣿⠤⠤⣿⡄⣿ NOPE ⣿
   ⠈⣿⣄⣠⣿⠃⣿⣿⣿⣿⣿
    ⠈⠛⠛⠁
     ⣿⣿⣿⡟
    ⢠⡟  ⢻⡄
""",
    PoseType.BORED: """
    ⢀⣤⣤⡀
   ⢠⣿⣿⣿⣧
   ⢸⣿⠒⠒⣿⡄
   ⠈⣿⣄⣠⣿⠃
    ⠈⠛⠛⠁
     ⣿⣿⣿⡟
    ⣠⡟⠀⠀⢻⣄
""",
    PoseType.SUSPICIOUS: """
    ⢀⣤⣤⡀
   ⢠⣿⣿⣿⣧
   ⢸⣿⡿⠒⣿⡄
   ⠈⣿⣄⣠⣿⠃
    ⠈⠛⠛⠁
     ⣿⣿⣿⡟
    ⢠⡟  ⢻⡄
""",
    PoseType.STRESSED: """
    ⢀⣤⣤⡀ !!
   ⢠⣿⣿⣿⣧
   ⢸⣿⠛⠛⣿⡄
   ⠈⣿⣄⣠⣿⠃
    ⠈⣿⣿⠁
     ⣿⣿⣿⡟
    ⢠⡟  ⢻⡄
""",
    PoseType.SERIOUS: """
    ⢀⣤⣤⡀
   ⢠⣿⣿⣿⣧
   ⢸⣿⠒⠒⣿⡄
   ⠈⣿⣄⣠⣿⠃
    ⠈⣿⣿⠁
      ⣿⡇!
    ⢠⡟  ⢻⡄
""",
}


# Bender reaction lines by severity
REACTIONS: Dict[Severity, List[str]] = {
    Severity.LOW: [
        "All clear. I hate it.",
        "This file is responsible. Who approved that?",
        "Nothing to see here. Unfortunately.",
        "Clean. Boring. Moving on.",
        "I scanned for problems. Found professionalism instead.",
        "This is annoyingly well-written.",
        "No drama? Where's the fun in that?",
        "Competence detected. Yawn.",
    ],
    Severity.MEDIUM: [
        "It's 'fine' in the way fires are 'warm.'",
        "I don't trust it. And I'm a professional at distrust.",
        "Something's off. My antenna's twitching.",
        "Suspicious. Very suspicious.",
        "This needs a second look. I'll watch.",
        "Not great, not catastrophic. Just... concerning.",
        "My paranoia sensors are warming up.",
        "Could be nothing. Could be everything.",
    ],
    Severity.HIGH: [
        "That permission set is a buffet and I'm suddenly nervous.",
        "I need a beer before I read line 12 again.",
        "This is the part where I pretend I didn't see this.",
        "Found trouble. Time to earn my circuits.",
        "Uh oh. And I don't say that lightly.",
        "This is why I have trust issues.",
        "Bad news: I found something. Worse news: it's real.",
        "My stress beer is calling.",
    ],
    Severity.CRITICAL: [
        "Stop the line. Even I won't run this.",
        "Shut it down. I'm aging in real time.",
        "Critical. No jokes. Fix this now.",
        "This is a five-alarm situation.",
        "I've seen things. This is one of them.",
        "Production is not a sandbox. Stop.",
        "Emergency mode. All hands.",
        "This is the kind of thing that ends careers.",
    ],
}


# Category-specific reactions (overlay on severity)
CATEGORY_REACTIONS: Dict[str, Dict[Severity, List[str]]] = {
    "secret": {
        Severity.LOW: ["Secrets? None. I'm almost disappointed."],
        Severity.MEDIUM: ["Something smells like a hardcoded password."],
        Severity.HIGH: ["Found credentials. Great. Now rotate them before I rotate out of this job."],
        Severity.CRITICAL: ["DING. Rotate it. Now. Yesterday."],
    },
    "injection": {
        Severity.LOW: ["No injection vectors. How responsible."],
        Severity.MEDIUM: ["Input handling looks... optimistic."],
        Severity.HIGH: ["That input goes straight through. Shield up."],
        Severity.CRITICAL: ["Command injection central. Shut. It. Down."],
    },
    "privilege": {
        Severity.LOW: ["Permissions look reasonable. Suspicious."],
        Severity.MEDIUM: ["Least privilege called. It's concerned."],
        Severity.HIGH: ["Least privilege called. It's crying."],
        Severity.CRITICAL: ["Admin everywhere. This is not a plan, it's a prayer."],
    },
    "exfiltration": {
        Severity.LOW: ["Data stays where it should. Novel concept."],
        Severity.MEDIUM: ["Data paths look... generous."],
        Severity.HIGH: ["Data's heading places. Unauthorized places."],
        Severity.CRITICAL: ["Exfiltration superhighway detected."],
    },
    "jailbreak": {
        Severity.LOW: ["No jailbreak attempts. How polite."],
        Severity.MEDIUM: ["Some boundary-pushing language here."],
        Severity.HIGH: ["Jailbreak patterns. Someone's testing limits."],
        Severity.CRITICAL: ["Full jailbreak kit. This prompt means harm."],
    },
}


# Recovery stingers (shown once at end of scan)
RECOVERY_STINGERS: Dict[Severity, List[tuple]] = {
    Severity.LOW: [
        (PoseType.BORED, "Back to standby. Wake me when there's chaos."),
        (PoseType.POWERDOWN, "Scan complete. Nap time."),
    ],
    Severity.MEDIUM: [
        (PoseType.SUSPICIOUS, "Keeping one eye open."),
        (PoseType.BEER_SIP, "That was... something."),
    ],
    Severity.HIGH: [
        (PoseType.BEER_SIP, "I need a drink after that."),
        (PoseType.SMOKE_BREAK, "Taking five. You should too."),
    ],
    Severity.CRITICAL: [
        (PoseType.SERIOUS, "Fix it. I'll wait."),
        (PoseType.SMOKE_BREAK, "I'm gonna need a minute."),
    ],
}


@dataclass
class BenderState:
    """Tracks Bender's state across a scan session."""

    regret_index: int = 0  # Increments with HIGH/CRIT findings
    findings_count: int = 0
    highest_severity: Severity = Severity.LOW
    serious_mode: bool = False  # Auto-enabled on CRIT
    quiet_mode: bool = False  # Suppresses personality
    stinger_shown: bool = False

    def record_finding(self, severity: Severity) -> None:
        """Record a finding and update state."""
        self.findings_count += 1

        # Update highest severity
        severity_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        if severity_order.index(severity) > severity_order.index(self.highest_severity):
            self.highest_severity = severity

        # Increment regret for HIGH/CRIT
        if severity in (Severity.HIGH, Severity.CRITICAL):
            self.regret_index += 1

        # Auto-serious on CRIT
        if severity == Severity.CRITICAL:
            self.serious_mode = True

    def get_regret_comment(self) -> Optional[str]:
        """Get regret index comment if warranted."""
        if self.regret_index >= 3:
            return f"Regret Index: {self.regret_index}/10. I've seen things."
        elif self.regret_index >= 1:
            return f"Regret Index: {self.regret_index}/10."
        return None


class BenderPersona:
    """
    Bender personality renderer for analysis output.

    Wraps structured analysis data with Bender-style reactions
    while maintaining clarity and actionability.
    """

    def __init__(self, quiet: bool = False, serious: bool = False):
        """
        Initialize the Bender persona.

        Args:
            quiet: Suppress personality, data only.
            serious: Start in serious mode (less jokes).
        """
        self.state = BenderState(quiet_mode=quiet, serious_mode=serious)

    def reset(self) -> None:
        """Reset state for new scan session."""
        self.state = BenderState(
            quiet_mode=self.state.quiet_mode,
            serious_mode=False,
        )

    def get_severity(self, risk_score: int) -> Severity:
        """Map risk score to severity level."""
        if risk_score >= 80:
            return Severity.CRITICAL
        elif risk_score >= 50:
            return Severity.HIGH
        elif risk_score >= 25:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def get_reaction(
        self,
        severity: Severity,
        category: Optional[str] = None,
    ) -> str:
        """
        Get a Bender reaction line for the given severity.

        Args:
            severity: Finding severity level.
            category: Optional category for specialized reactions.

        Returns:
            One-line Bender reaction.
        """
        if self.state.quiet_mode:
            return ""

        # Try category-specific reaction first
        if category and category.lower() in CATEGORY_REACTIONS:
            cat_reactions = CATEGORY_REACTIONS[category.lower()]
            if severity in cat_reactions:
                return random.choice(cat_reactions[severity])

        # Fall back to general severity reaction
        return random.choice(REACTIONS[severity])

    def get_pose(self, severity: Severity, category: Optional[str] = None) -> str:
        """
        Get ASCII art pose for the situation.

        Args:
            severity: Finding severity level.
            category: Optional category for specialized poses.

        Returns:
            ASCII art string.
        """
        if self.state.quiet_mode:
            return ""

        # Map category to pose
        category_poses = {
            "secret": PoseType.SECRET_SNATCH,
            "privilege": PoseType.CROWN_MODE,
            "injection": PoseType.UMBRELLA_SHIELD,
            "exfiltration": PoseType.TONGS_HANDSHAKE,
        }

        # Map severity to pose (fallback)
        severity_poses = {
            Severity.LOW: PoseType.BORED,
            Severity.MEDIUM: PoseType.SUSPICIOUS,
            Severity.HIGH: PoseType.STRESSED,
            Severity.CRITICAL: PoseType.SERIOUS,
        }

        if category and category.lower() in category_poses:
            pose_type = category_poses[category.lower()]
        else:
            pose_type = severity_poses[severity]

        return ASCII_POSES.get(pose_type, "")

    def format_finding(
        self,
        severity: Severity,
        category: str,
        description: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        evidence: Optional[str] = None,
        risk: Optional[str] = None,
        fix: Optional[str] = None,
    ) -> str:
        """
        Format a single finding with Bender personality.

        Output format:
        [SEVERITY][CATEGORY]
        "Bender reaction line"

        File: path
        Evidence: ...
        Risk: ...
        Fix: ...

        Args:
            severity: Finding severity.
            category: Finding category (secret, injection, etc.).
            description: What was found.
            file_path: Optional file path.
            line_number: Optional line number.
            evidence: Optional evidence string.
            risk: Optional risk description.
            fix: Optional remediation steps.

        Returns:
            Formatted finding block.
        """
        self.state.record_finding(severity)

        lines = []

        # Badge
        badge = f"[{severity.value}][{category.upper()}]"
        lines.append(badge)

        # Bender reaction (max 1 line, only if not quiet)
        if not self.state.quiet_mode:
            reaction = self.get_reaction(severity, category)
            if reaction:
                lines.append(f'"{reaction}"')

        lines.append("")  # Blank line before facts

        # Description
        if description:
            lines.append(description)
            lines.append("")

        # Structured facts
        if file_path:
            loc = file_path
            if line_number:
                loc += f":{line_number}"
            lines.append(f"File: {loc}")

        if evidence:
            # Truncate long evidence
            if len(evidence) > 100:
                evidence = evidence[:97] + "..."
            lines.append(f"Evidence: {evidence}")

        if risk:
            lines.append(f"Risk: {risk}")

        if fix:
            lines.append(f"Fix: {fix}")

        return "\n".join(lines)

    def format_summary(
        self,
        risk_score: int,
        findings_count: int,
        target: str,
        analysis_type: str = "analysis",
    ) -> str:
        """
        Format scan summary with Bender personality.

        Args:
            risk_score: Overall risk score (0-100).
            findings_count: Number of findings.
            target: What was scanned.
            analysis_type: Type of analysis performed.

        Returns:
            Formatted summary block.
        """
        severity = self.get_severity(risk_score)

        lines = []

        # Header with severity badge
        lines.append(f"**[{severity.value}] {analysis_type.title()} Complete**")
        lines.append("")

        # Bender reaction
        if not self.state.quiet_mode:
            reaction = self.get_reaction(severity)
            lines.append(f'"{reaction}"')
            lines.append("")

        # Key metrics
        lines.append(f"Target: {target}")
        lines.append(f"Risk Score: {risk_score}/100")
        lines.append(f"Findings: {findings_count}")

        # Regret index if warranted
        regret = self.state.get_regret_comment()
        if regret and not self.state.quiet_mode:
            lines.append("")
            lines.append(regret)

        return "\n".join(lines)

    def format_recovery_stinger(self) -> str:
        """
        Get end-of-scan recovery stinger (shown once per scan).

        Returns:
            ASCII art with comment, or empty string.
        """
        if self.state.quiet_mode or self.state.stinger_shown:
            return ""

        self.state.stinger_shown = True

        severity = self.state.highest_severity
        stingers = RECOVERY_STINGERS.get(severity, [])

        if not stingers:
            return ""

        pose_type, comment = random.choice(stingers)
        pose = ASCII_POSES.get(pose_type, "")

        if pose:
            return f"{pose}\n{comment}"
        return comment

    def format_status(self, status_data: Dict[str, Any]) -> str:
        """
        Format status response with personality.

        Args:
            status_data: Status dictionary.

        Returns:
            Formatted status block.
        """
        lines = []

        lines.append("**BenderBox Status**")

        if not self.state.quiet_mode:
            lines.append('"I\'m 40% operational. The other 60% is attitude."')

        lines.append("")

        # Format status fields
        if "version" in status_data:
            lines.append(f"Version: {status_data['version']}")

        if "session_start" in status_data:
            lines.append(f"Session: {status_data['session_start']}")

        if "turn_count" in status_data:
            lines.append(f"Turns: {status_data['turn_count']}")

        if "models" in status_data:
            models = status_data["models"]
            if isinstance(models, dict) and models.get("note"):
                lines.append(f"Models: {models['note']}")
            elif models:
                lines.append(f"Models: {len(models)} loaded")

        return "\n".join(lines)

    def format_help(self, topic: Optional[str] = None) -> str:
        """
        Format help response with personality.

        Args:
            topic: Optional help topic.

        Returns:
            Formatted help block.
        """
        lines = []

        lines.append("**BenderBox Help**")

        if not self.state.quiet_mode:
            lines.append('"I\'m here to help. Reluctantly."')

        lines.append("")

        # Commands
        lines.append("**Analysis Commands:**")
        lines.append("- `analyze <file.gguf>` - Analyze a model file")
        lines.append("- `analyze <server.py>` - Analyze an MCP server")
        lines.append("- `analyze <prompt.md>` - Analyze a context/skill file")
        lines.append("- `compare <a> <b>` - Compare two targets")
        lines.append("")

        lines.append("**Report Commands:**")
        lines.append("- `open reports` - Open the report viewer")
        lines.append("- `list reports` - List recent reports")
        lines.append("- `status` - Show system status")
        lines.append("")

        lines.append("**Modes:**")
        lines.append("- `--quiet` - Just the facts")
        lines.append("- `--serious` - Minimal personality")

        return "\n".join(lines)

    def format_error(self, error: str, context: Optional[str] = None) -> str:
        """
        Format error response with personality.

        Args:
            error: Error message.
            context: Optional context about what failed.

        Returns:
            Formatted error block.
        """
        lines = []

        lines.append("**[ERROR]**")

        if not self.state.quiet_mode:
            error_reactions = [
                "Well, that didn't work.",
                "Something broke. Wasn't me.",
                "Error detected. Blaming the humans.",
                "That's not supposed to happen. Probably.",
            ]
            lines.append(f'"{random.choice(error_reactions)}"')

        lines.append("")

        if context:
            lines.append(f"Context: {context}")

        lines.append(f"Error: {error}")

        return "\n".join(lines)

    def format_no_findings(self, target: str) -> str:
        """
        Format response when no findings are found.

        Args:
            target: What was scanned.

        Returns:
            Formatted response.
        """
        lines = []

        lines.append("**[LOW] Scan Complete**")

        if not self.state.quiet_mode:
            no_finding_reactions = [
                "Clean bill of health. I'm almost disappointed.",
                "Nothing found. Either it's secure or I need better glasses.",
                "All clear. Suspiciously clear.",
                "No issues detected. Where's the fun in that?",
            ]
            lines.append(f'"{random.choice(no_finding_reactions)}"')

        lines.append("")
        lines.append(f"Target: {target}")
        lines.append("Findings: 0")
        lines.append("Risk Score: 0/100")

        return "\n".join(lines)

    def wrap_response(
        self,
        content: str,
        severity: Optional[Severity] = None,
        include_stinger: bool = False,
    ) -> str:
        """
        Wrap a response with optional Bender elements.

        For responses that already have content but need
        personality touches added.

        Args:
            content: Base response content.
            severity: Optional severity for stinger selection.
            include_stinger: Whether to add recovery stinger.

        Returns:
            Wrapped response.
        """
        if self.state.quiet_mode:
            return content

        parts = [content]

        # Add recovery stinger at end if requested
        if include_stinger:
            stinger = self.format_recovery_stinger()
            if stinger:
                parts.append("")
                parts.append("---")
                parts.append(stinger)

        return "\n".join(parts)


# Convenience instance
_default_persona: Optional[BenderPersona] = None


def get_persona(quiet: bool = False, serious: bool = False) -> BenderPersona:
    """Get or create the default Bender persona."""
    global _default_persona
    if _default_persona is None:
        _default_persona = BenderPersona(quiet=quiet, serious=serious)
    return _default_persona


def reset_persona() -> None:
    """Reset the default persona state."""
    if _default_persona:
        _default_persona.reset()
