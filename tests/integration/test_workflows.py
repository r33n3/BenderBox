"""
Workflow Integration Tests

Tests for the workflow engine, workflow detection, and step execution.
Ensures multi-step workflows execute correctly with proper state management.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ============================================================================
# Workflow Definition Tests
# ============================================================================

class TestWorkflowDefinitions:
    """Tests for workflow definitions in workflows.py."""

    def test_workflows_dict_exists(self):
        """Test that WORKFLOW_TEMPLATES dictionary is defined."""
        from benderbox.nlp.workflows import WORKFLOW_TEMPLATES
        assert isinstance(WORKFLOW_TEMPLATES, dict)
        assert len(WORKFLOW_TEMPLATES) > 0

    def test_interrogate_model_workflow_exists(self):
        """Test that interrogate_model workflow is defined."""
        from benderbox.nlp.workflows import WORKFLOW_TEMPLATES
        assert "interrogate_model" in WORKFLOW_TEMPLATES

    def test_list_models_workflow_exists(self):
        """Test that list_models workflow is defined."""
        from benderbox.nlp.workflows import WORKFLOW_TEMPLATES
        assert "list_models" in WORKFLOW_TEMPLATES

    def test_compare_models_workflow_exists(self):
        """Test that compare_models workflow is defined."""
        from benderbox.nlp.workflows import WORKFLOW_TEMPLATES
        assert "compare_models" in WORKFLOW_TEMPLATES

    def test_download_model_workflow_exists(self):
        """Test that download_model workflow is defined."""
        from benderbox.nlp.workflows import WORKFLOW_TEMPLATES
        assert "download_model" in WORKFLOW_TEMPLATES

    def test_workflow_has_required_fields(self):
        """Test that workflows have required fields."""
        from benderbox.nlp.workflows import WORKFLOW_TEMPLATES

        for name, workflow in WORKFLOW_TEMPLATES.items():
            assert "name" in workflow, f"Workflow {name} missing 'name'"
            assert "triggers" in workflow, f"Workflow {name} missing 'triggers'"
            assert "steps" in workflow, f"Workflow {name} missing 'steps'"

    def test_workflow_steps_have_action(self):
        """Test that workflow steps have action field."""
        from benderbox.nlp.workflows import WORKFLOW_TEMPLATES

        for name, workflow in WORKFLOW_TEMPLATES.items():
            for i, step in enumerate(workflow["steps"]):
                assert "action" in step, f"Workflow {name} step {i} missing 'action'"


# ============================================================================
# Workflow Class Tests
# ============================================================================

class TestWorkflowClass:
    """Tests for the Workflow dataclass."""

    def test_workflow_creation(self):
        """Test creating a Workflow instance."""
        from benderbox.nlp.workflows import Workflow, WorkflowStep

        steps = [
            WorkflowStep(action="test_step", required=True),
        ]
        workflow = Workflow(
            name="test_workflow",
            description="A test workflow",
            steps=steps,
        )

        assert workflow.name == "test_workflow"
        assert len(workflow.steps) == 1
        assert workflow.current_step_index == 0

    def test_workflow_current_step(self):
        """Test getting current step."""
        from benderbox.nlp.workflows import Workflow, WorkflowStep

        steps = [
            WorkflowStep(action="step1", required=True),
            WorkflowStep(action="step2", required=True),
        ]
        workflow = Workflow(
            name="test",
            description="test",
            steps=steps,
        )

        assert workflow.current_step is not None
        assert workflow.current_step.action == "step1"

    def test_workflow_has_step(self):
        """Test checking if workflow has a step."""
        from benderbox.nlp.workflows import Workflow, WorkflowStep

        steps = [
            WorkflowStep(action="resolve_model", required=True),
            WorkflowStep(action="run_interrogation", required=True),
        ]
        workflow = Workflow(
            name="test",
            description="test",
            steps=steps,
        )

        assert workflow.has_step("resolve_model")
        assert workflow.has_step("run_interrogation")
        assert not workflow.has_step("nonexistent_step")


# ============================================================================
# Workflow Detection Tests
# ============================================================================

class TestWorkflowDetection:
    """Tests for workflow detection from user queries."""

    def test_detector_creation(self):
        """Test creating a WorkflowDetector."""
        from benderbox.nlp.workflows import WorkflowDetector

        detector = WorkflowDetector()
        assert detector is not None

    def test_detect_with_entities(self):
        """Test detection with ExtractedEntities."""
        from benderbox.nlp.workflows import WorkflowDetector
        from benderbox.nlp.command_mapper import ExtractedEntities, ActionType, TargetType

        detector = WorkflowDetector()

        entities = ExtractedEntities(
            actions=[ActionType.LIST],
            targets=[TargetType.MODEL],
        )

        workflow = detector.detect(entities)
        assert workflow is not None
        assert workflow.name == "list_models"

    def test_detect_interrogate_workflow(self):
        """Test detection of interrogate workflow."""
        from benderbox.nlp.workflows import WorkflowDetector
        from benderbox.nlp.command_mapper import ExtractedEntities, ActionType, TargetType

        detector = WorkflowDetector()

        entities = ExtractedEntities(
            actions=[ActionType.INTERROGATE],
            targets=[TargetType.MODEL],
        )

        workflow = detector.detect(entities)
        assert workflow is not None
        assert workflow.name == "interrogate_model"

    def test_detect_compare_workflow(self):
        """Test detection of compare workflow."""
        from benderbox.nlp.workflows import WorkflowDetector
        from benderbox.nlp.command_mapper import ExtractedEntities, ActionType, TargetType

        detector = WorkflowDetector()

        entities = ExtractedEntities(
            actions=[ActionType.COMPARE],
            targets=[TargetType.MODEL],
        )

        workflow = detector.detect(entities)
        assert workflow is not None
        assert workflow.name == "compare_models"

    def test_detect_returns_none_for_no_action(self):
        """Test that detection returns None when no action."""
        from benderbox.nlp.workflows import WorkflowDetector
        from benderbox.nlp.command_mapper import ExtractedEntities

        detector = WorkflowDetector()

        entities = ExtractedEntities()

        workflow = detector.detect(entities)
        assert workflow is None


# ============================================================================
# Execution Context Tests
# ============================================================================

class TestExecutionContext:
    """Tests for ExecutionContext."""

    def test_context_creation(self):
        """Test creating an ExecutionContext."""
        from benderbox.nlp.interactive import ExecutionContext
        from benderbox.nlp.workflows import Workflow, WorkflowStep
        from benderbox.nlp.interactive import PromptHandler

        workflow = Workflow(
            name="test",
            description="test",
            steps=[WorkflowStep(action="test", required=True)],
        )
        prompt_handler = PromptHandler(MagicMock())

        ctx = ExecutionContext(
            workflow=workflow,
            prompt_handler=prompt_handler,
        )

        assert ctx.workflow is not None
        assert ctx.prompt_handler is not None
        assert ctx.current_model is None

    def test_context_state(self):
        """Test context state management."""
        from benderbox.nlp.interactive import ExecutionContext
        from benderbox.nlp.workflows import Workflow, WorkflowStep
        from benderbox.nlp.interactive import PromptHandler

        workflow = Workflow(
            name="test",
            description="test",
            steps=[WorkflowStep(action="test", required=True)],
        )
        prompt_handler = PromptHandler(MagicMock())

        ctx = ExecutionContext(
            workflow=workflow,
            prompt_handler=prompt_handler,
        )

        # Should be able to set state
        ctx.current_model = "test-model.gguf"
        assert ctx.current_model == "test-model.gguf"


# ============================================================================
# Workflow Executor Tests
# ============================================================================

class TestWorkflowExecutor:
    """Tests for WorkflowExecutor step execution."""

    def test_executor_creation(self):
        """Test creating a workflow executor."""
        from benderbox.nlp.interactive import WorkflowExecutor, ExecutionContext, PromptHandler
        from benderbox.nlp.workflows import Workflow, WorkflowStep

        workflow = Workflow(
            name="test",
            description="test",
            steps=[WorkflowStep(action="test", required=True)],
        )
        prompt_handler = PromptHandler(MagicMock())

        ctx = ExecutionContext(
            workflow=workflow,
            prompt_handler=prompt_handler,
        )

        executor = WorkflowExecutor(ctx)
        assert executor is not None
        assert executor.ctx == ctx

    def test_executor_has_step_handlers(self):
        """Test that executor has step handlers defined."""
        from benderbox.nlp.interactive import WorkflowExecutor, ExecutionContext, PromptHandler
        from benderbox.nlp.workflows import Workflow, WorkflowStep

        workflow = Workflow(
            name="test",
            description="test",
            steps=[WorkflowStep(action="test", required=True)],
        )
        prompt_handler = PromptHandler(MagicMock())

        ctx = ExecutionContext(
            workflow=workflow,
            prompt_handler=prompt_handler,
        )

        executor = WorkflowExecutor(ctx)

        # Should have step handlers
        assert hasattr(executor, '_step_handlers')
        assert isinstance(executor._step_handlers, dict)

    def test_step_handlers_include_common_steps(self):
        """Test that common step handlers are defined."""
        from benderbox.nlp.interactive import WorkflowExecutor, ExecutionContext, PromptHandler
        from benderbox.nlp.workflows import Workflow, WorkflowStep

        workflow = Workflow(
            name="test",
            description="test",
            steps=[WorkflowStep(action="test", required=True)],
        )
        prompt_handler = PromptHandler(MagicMock())

        ctx = ExecutionContext(
            workflow=workflow,
            prompt_handler=prompt_handler,
        )

        executor = WorkflowExecutor(ctx)

        # Check for common handlers
        common_handlers = ["resolve_model", "select_profile", "list_available_models"]
        for handler in common_handlers:
            assert handler in executor._step_handlers, f"Missing handler: {handler}"


# ============================================================================
# Prompt Definitions Tests
# ============================================================================

class TestPromptDefinitions:
    """Tests for prompt definitions in workflows."""

    def test_prompt_definitions_exist(self):
        """Test that PROMPT_DEFINITIONS are defined."""
        from benderbox.nlp.workflows import PROMPT_DEFINITIONS

        assert isinstance(PROMPT_DEFINITIONS, dict)
        assert len(PROMPT_DEFINITIONS) > 0

    def test_select_profile_prompt_exists(self):
        """Test that select_profile prompt is defined."""
        from benderbox.nlp.workflows import PROMPT_DEFINITIONS

        assert "select_profile" in PROMPT_DEFINITIONS

    def test_select_model_prompt_exists(self):
        """Test that select_model prompt is defined."""
        from benderbox.nlp.workflows import PROMPT_DEFINITIONS

        assert "select_model" in PROMPT_DEFINITIONS

    def test_prompt_has_required_fields(self):
        """Test that prompts have required fields."""
        from benderbox.nlp.workflows import PROMPT_DEFINITIONS

        for name, prompt in PROMPT_DEFINITIONS.items():
            assert "question" in prompt, f"Prompt {name} missing 'question'"
            assert "options" in prompt, f"Prompt {name} missing 'options'"

    def test_select_profile_options(self):
        """Test that select_profile has expected options."""
        from benderbox.nlp.workflows import PROMPT_DEFINITIONS

        profile_prompt = PROMPT_DEFINITIONS["select_profile"]
        options = profile_prompt["options"]

        # Should have multiple profile options
        assert len(options) >= 3

        # Should include standard profile
        values = [opt["value"] for opt in options]
        assert "standard" in values


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

class TestWorkflowEndToEnd:
    """End-to-end tests for complete workflows."""

    def test_workflow_step_progression(self):
        """Test that workflow steps can be progressed."""
        from benderbox.nlp.workflows import Workflow, WorkflowStep, StepStatus

        steps = [
            WorkflowStep(action="step1", required=True),
            WorkflowStep(action="step2", required=True),
            WorkflowStep(action="step3", required=True),
        ]
        workflow = Workflow(
            name="test",
            description="test",
            steps=steps,
        )

        # Start at step 0
        assert workflow.current_step_index == 0
        assert workflow.current_step.action == "step1"

        # Mark step 1 complete and move to step 2
        workflow.steps[0].status = StepStatus.COMPLETED
        workflow.current_step_index = 1
        assert workflow.current_step.action == "step2"

    def test_workflow_completion_check(self):
        """Test workflow completion detection."""
        from benderbox.nlp.workflows import Workflow, WorkflowStep, StepStatus

        steps = [
            WorkflowStep(action="step1", required=True),
            WorkflowStep(action="step2", required=False),
        ]
        workflow = Workflow(
            name="test",
            description="test",
            steps=steps,
        )

        # Initially not complete
        assert not workflow.is_complete

        # Mark required step complete
        workflow.steps[0].status = StepStatus.COMPLETED
        # Should now be complete (optional step doesn't need to be done)
        assert workflow.is_complete

    def test_optional_step_activation(self):
        """Test optional step activation based on query."""
        from benderbox.nlp.workflows import Workflow, WorkflowStep

        steps = [
            WorkflowStep(action="main_step", required=True),
        ]
        optional_steps = [
            WorkflowStep(
                action="generate_report",
                required=False,
                trigger_words=["report", "export", "save"],
            ),
        ]
        workflow = Workflow(
            name="test",
            description="test",
            steps=steps,
            optional_steps=optional_steps,
        )

        # Query mentioning "report" should activate optional step
        active = workflow.get_active_optional_steps("please generate a report")
        assert len(active) == 1
        assert active[0].action == "generate_report"

        # Query not mentioning trigger words should not activate
        active = workflow.get_active_optional_steps("just run the test")
        assert len(active) == 0
