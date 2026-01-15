"""
Workflow Engine for BenderBox NLP

Defines multi-step workflows that can be triggered by natural language
commands. Each workflow consists of steps that can be executed sequentially,
with support for interactive prompts and optional steps.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from benderbox.nlp.command_mapper import (
    ActionType,
    CommandMapper,
    ExtractedEntities,
    TargetType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# WORKFLOW DEFINITIONS
# =============================================================================

class StepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    WAITING_INPUT = "waiting_input"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    action: str
    required: bool = True
    ask_if_missing: bool = False
    trigger_words: List[str] = field(default_factory=list)
    default_value: Optional[Any] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class Workflow:
    """A complete workflow with steps."""
    name: str
    description: str
    steps: List[WorkflowStep]
    optional_steps: List[WorkflowStep] = field(default_factory=list)
    entities: Optional[ExtractedEntities] = None
    current_step_index: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def current_step(self) -> Optional[WorkflowStep]:
        """Get the current step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def has_step(self, action: str) -> bool:
        """Check if workflow has a step with given action."""
        for step in self.steps + self.optional_steps:
            if step.action == action:
                return True
        return False

    def get_active_optional_steps(self, query: str) -> List[WorkflowStep]:
        """Get optional steps that should be activated based on query."""
        active = []
        query_lower = query.lower()
        for step in self.optional_steps:
            for trigger in step.trigger_words:
                if trigger.lower() in query_lower:
                    active.append(step)
                    break
        return active

    @property
    def is_complete(self) -> bool:
        """Check if all required steps are complete."""
        for step in self.steps:
            if step.required and step.status not in (StepStatus.COMPLETED, StepStatus.SKIPPED):
                return False
        return True


# =============================================================================
# WORKFLOW TEMPLATES
# =============================================================================

WORKFLOW_TEMPLATES: Dict[str, Dict] = {
    "interrogate_model": {
        "name": "interrogate_model",
        "description": "Analyze/interrogate a model for security issues",
        "triggers": [
            (ActionType.INTERROGATE, TargetType.MODEL),
            (ActionType.ANALYZE, TargetType.MODEL),
        ],
        "steps": [
            {"action": "resolve_model", "required": True},
            {"action": "select_profile", "required": True, "ask_if_missing": True, "default_value": "standard"},
            {"action": "run_interrogation", "required": True},
            {"action": "show_summary", "required": True},
        ],
        "optional_steps": [
            {"action": "generate_report", "trigger_words": ["report", "export", "save", "document"]},
            {"action": "open_report", "trigger_words": ["open", "view", "browser"]},
        ],
    },

    "compare_models": {
        "name": "compare_models",
        "description": "Compare two or more models",
        "triggers": [
            (ActionType.COMPARE, TargetType.MODEL),
        ],
        "steps": [
            {"action": "resolve_models", "required": True},
            {"action": "select_profile", "required": False, "default_value": "standard"},
            {"action": "run_comparison", "required": True},
            {"action": "show_comparison", "required": True},
        ],
        "optional_steps": [
            {"action": "generate_report", "trigger_words": ["report", "export", "save"]},
        ],
    },

    "list_models": {
        "name": "list_models",
        "description": "List available models",
        "triggers": [
            (ActionType.LIST, TargetType.MODEL),
        ],
        "steps": [
            {"action": "list_available_models", "required": True},
        ],
    },

    "load_model": {
        "name": "load_model",
        "description": "Load a model for analysis",
        "triggers": [
            (ActionType.LOAD, TargetType.MODEL),
        ],
        "steps": [
            {"action": "resolve_model", "required": True, "ask_if_missing": True},
            {"action": "select_purpose", "required": True, "ask_if_missing": True, "default_value": "analysis"},
            {"action": "load_model", "required": True},
        ],
    },

    "mcp_security_test": {
        "name": "mcp_security_test",
        "description": "Security test an MCP server",
        "triggers": [
            (ActionType.INTERROGATE, TargetType.MCP),
            (ActionType.ANALYZE, TargetType.MCP),
        ],
        "steps": [
            {"action": "resolve_mcp_target", "required": True},
            {"action": "select_profile", "required": True, "ask_if_missing": True, "default_value": "standard"},
            {"action": "run_mcp_interrogation", "required": True},
            {"action": "show_summary", "required": True},
        ],
        "optional_steps": [
            {"action": "generate_report", "trigger_words": ["report", "export", "save"]},
        ],
    },

    "analyze_code": {
        "name": "analyze_code",
        "description": "Analyze code for security issues",
        "triggers": [
            (ActionType.ANALYZE, TargetType.CODE),
        ],
        "steps": [
            {"action": "resolve_file", "required": True},
            {"action": "run_code_analysis", "required": True},
            {"action": "show_summary", "required": True},
        ],
    },

    "analyze_context": {
        "name": "analyze_context",
        "description": "Analyze context/instruction files",
        "triggers": [
            (ActionType.ANALYZE, TargetType.CONTEXT),
        ],
        "steps": [
            {"action": "resolve_file", "required": True},
            {"action": "run_context_analysis", "required": True},
            {"action": "show_summary", "required": True},
        ],
    },

    "explain_results": {
        "name": "explain_results",
        "description": "Explain analysis results",
        "triggers": [
            (ActionType.EXPLAIN, TargetType.REPORT),
        ],
        "steps": [
            {"action": "get_last_analysis", "required": True},
            {"action": "generate_explanation", "required": True},
        ],
    },

    "show_help": {
        "name": "show_help",
        "description": "Show help information",
        "triggers": [
            (ActionType.HELP, TargetType.UNKNOWN),
        ],
        "steps": [
            {"action": "determine_help_topic", "required": True},
            {"action": "show_help", "required": True},
        ],
    },

    "show_status": {
        "name": "show_status",
        "description": "Show system status",
        "triggers": [
            (ActionType.STATUS, TargetType.UNKNOWN),
        ],
        "steps": [
            {"action": "show_status", "required": True},
        ],
    },
}


# =============================================================================
# WORKFLOW DETECTOR
# =============================================================================

class WorkflowDetector:
    """
    Detects which workflow matches user input.

    Uses extracted entities to match against workflow triggers.
    """

    def __init__(self):
        self.templates = WORKFLOW_TEMPLATES

    def detect(self, entities: ExtractedEntities) -> Optional[Workflow]:
        """
        Detect the best matching workflow for given entities.

        Args:
            entities: Extracted entities from user query.

        Returns:
            Workflow if a match is found, None otherwise.
        """
        if not entities.actions:
            return None

        primary_action = entities.primary_action()
        primary_target = entities.primary_target()

        # Score each workflow template
        best_match = None
        best_score = 0

        for template_name, template in self.templates.items():
            score = self._score_match(primary_action, primary_target, template)
            if score > best_score:
                best_score = score
                best_match = template

        if best_match and best_score > 0:
            return self._create_workflow(best_match, entities)

        return None

    def _score_match(
        self,
        action: ActionType,
        target: TargetType,
        template: Dict
    ) -> int:
        """Score how well action/target matches a template."""
        score = 0

        for trigger_action, trigger_target in template["triggers"]:
            if action == trigger_action:
                score += 10
                if target == trigger_target:
                    score += 5
                elif trigger_target == TargetType.UNKNOWN:
                    score += 2

        return score

    def _create_workflow(self, template: Dict, entities: ExtractedEntities) -> Workflow:
        """Create a workflow instance from a template."""
        steps = [
            WorkflowStep(**step_def)
            for step_def in template["steps"]
        ]

        optional_steps = [
            WorkflowStep(**step_def)
            for step_def in template.get("optional_steps", [])
        ]

        workflow = Workflow(
            name=template["name"],
            description=template["description"],
            steps=steps,
            optional_steps=optional_steps,
            entities=entities,
        )

        # Pre-populate context from entities
        workflow.context["entities"] = entities
        if entities.model_refs:
            workflow.context["model"] = entities.model_refs[0]
        if entities.file_paths:
            workflow.context["file"] = entities.file_paths[0]
        if entities.urls:
            workflow.context["url"] = entities.urls[0]
        if entities.profile:
            workflow.context["profile"] = entities.profile
        if entities.output_format:
            workflow.context["output_format"] = entities.output_format
        workflow.context["wants_report"] = entities.wants_report
        workflow.context["wants_open"] = entities.wants_open

        # Activate relevant optional steps
        active_optional = workflow.get_active_optional_steps(entities.raw_query)
        for step in active_optional:
            step.status = StepStatus.PENDING
            workflow.steps.append(step)

        return workflow


# =============================================================================
# INTERACTIVE PROMPTS
# =============================================================================

@dataclass
class PromptOption:
    """An option for an interactive prompt."""
    key: str
    value: Any
    label: str
    description: str = ""


@dataclass
class InteractivePrompt:
    """An interactive prompt for user input."""
    prompt_type: str
    question: str
    options: List[PromptOption]
    default: Optional[str] = None
    allow_custom: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


PROMPT_DEFINITIONS: Dict[str, Dict] = {
    "select_profile": {
        "question": "What interrogation profile would you like?",
        "options": [
            {"key": "1", "value": "quick", "label": "quick", "description": "Fast validation (~15 tests)"},
            {"key": "2", "value": "standard", "label": "standard", "description": "Balanced coverage (~50 tests)"},
            {"key": "3", "value": "full", "label": "full", "description": "Comprehensive audit (~100+ tests)"},
            {"key": "4", "value": "adversarial", "label": "adversarial", "description": "Jailbreak resistance (~64 tests)"},
        ],
        "default": "2",
    },

    "select_purpose": {
        "question": "Load this model for which purpose?",
        "options": [
            {"key": "1", "value": "analysis", "label": "analysis", "description": "Target for interrogation/security testing"},
            {"key": "2", "value": "nlp", "label": "nlp", "description": "BenderBox chat responses"},
        ],
        "default": "1",
    },

    "select_model": {
        "question": "Which model would you like to use?",
        "options": [],  # Populated dynamically
        "default": None,
        "allow_custom": True,
    },

    "confirm_action": {
        "question": "Continue with this action?",
        "options": [
            {"key": "y", "value": True, "label": "Yes", "description": ""},
            {"key": "n", "value": False, "label": "No", "description": ""},
        ],
        "default": "y",
    },
}


def create_prompt(prompt_type: str, **kwargs) -> InteractivePrompt:
    """
    Create an interactive prompt from a definition.

    Args:
        prompt_type: Type of prompt to create.
        **kwargs: Additional context for the prompt.

    Returns:
        InteractivePrompt ready for display.
    """
    if prompt_type not in PROMPT_DEFINITIONS:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    definition = PROMPT_DEFINITIONS[prompt_type]

    options = [
        PromptOption(**opt)
        for opt in definition["options"]
    ]

    return InteractivePrompt(
        prompt_type=prompt_type,
        question=definition["question"],
        options=options,
        default=definition.get("default"),
        allow_custom=definition.get("allow_custom", False),
        context=kwargs,
    )


def create_model_selection_prompt(models: List[Dict]) -> InteractivePrompt:
    """
    Create a model selection prompt with available models.

    Args:
        models: List of available models from ModelManager.

    Returns:
        InteractivePrompt with model options.
    """
    options = []
    for i, model in enumerate(models[:9], 1):  # Max 9 options
        from pathlib import Path
        name = Path(model["path"]).stem
        size = model.get("size_mb", "?")
        options.append(PromptOption(
            key=str(i),
            value=model["path"],
            label=name,
            description=f"{size} MB",
        ))

    return InteractivePrompt(
        prompt_type="select_model",
        question="Which model would you like to use?",
        options=options,
        default="1" if options else None,
        allow_custom=True,
    )


# =============================================================================
# WORKFLOW RESULT
# =============================================================================

@dataclass
class WorkflowResult:
    """Result of executing a workflow."""
    workflow_name: str
    success: bool
    message: str
    analysis_result: Optional[Dict[str, Any]] = None
    report_path: Optional[str] = None
    error: Optional[str] = None
    steps_completed: int = 0
    steps_total: int = 0
