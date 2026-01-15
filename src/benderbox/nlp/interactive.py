"""
Interactive Prompt Handler for BenderBox NLP

Handles interactive user prompts during workflow execution,
including profile selection, model selection, and confirmations.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from benderbox.nlp.workflows import (
    InteractivePrompt,
    PromptOption,
    Workflow,
    WorkflowStep,
    StepStatus,
    create_prompt,
    create_model_selection_prompt,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT HANDLER
# =============================================================================

class PromptHandler:
    """
    Handles interactive prompts and user input during workflows.

    This class manages the flow of asking questions and processing
    responses. It can be used with different UI backends.
    """

    def __init__(self, ui_callback: Optional[Callable] = None):
        """
        Initialize PromptHandler.

        Args:
            ui_callback: Async callback for displaying prompts.
                        Signature: async def callback(prompt: InteractivePrompt) -> str
        """
        self._ui_callback = ui_callback
        self._pending_prompt: Optional[InteractivePrompt] = None
        self._response_future: Optional[asyncio.Future] = None

    def set_ui_callback(self, callback: Callable) -> None:
        """Set the UI callback for displaying prompts."""
        self._ui_callback = callback

    async def ask(self, prompt: InteractivePrompt) -> Any:
        """
        Ask the user a question and wait for response.

        Args:
            prompt: The prompt to display.

        Returns:
            The selected value from the prompt options.
        """
        if self._ui_callback:
            response = await self._ui_callback(prompt)
            return self._parse_response(prompt, response)
        else:
            # No UI callback - return default or first option
            if prompt.default:
                for opt in prompt.options:
                    if opt.key == prompt.default:
                        return opt.value
            return prompt.options[0].value if prompt.options else None

    def _parse_response(self, prompt: InteractivePrompt, response: str) -> Any:
        """
        Parse user response to get the actual value.

        Args:
            prompt: The original prompt.
            response: User's text response.

        Returns:
            The corresponding value for the response.
        """
        response = response.strip().lower()

        # Check for option key match
        for opt in prompt.options:
            if response == opt.key.lower():
                return opt.value
            if response == opt.label.lower():
                return opt.value
            if response == opt.value if isinstance(opt.value, str) else str(opt.value):
                return opt.value

        # Check for partial matches
        for opt in prompt.options:
            if opt.label.lower().startswith(response):
                return opt.value

        # If allow_custom, return the raw response
        if prompt.allow_custom:
            return response

        # Default to first option or default
        if prompt.default:
            for opt in prompt.options:
                if opt.key == prompt.default:
                    return opt.value

        return prompt.options[0].value if prompt.options else response

    async def ask_profile(self) -> str:
        """Ask user to select an interrogation profile."""
        prompt = create_prompt("select_profile")
        return await self.ask(prompt)

    async def ask_purpose(self) -> str:
        """Ask user to select model purpose (nlp or analysis)."""
        prompt = create_prompt("select_purpose")
        return await self.ask(prompt)

    async def ask_model(self, models: List[Dict]) -> str:
        """Ask user to select a model."""
        prompt = create_model_selection_prompt(models)
        return await self.ask(prompt)

    async def confirm(self, message: str) -> bool:
        """Ask user for confirmation."""
        prompt = create_prompt("confirm_action")
        prompt.question = message
        result = await self.ask(prompt)
        return result is True or result in ("y", "yes", "true", "1")


# =============================================================================
# WORKFLOW EXECUTOR
# =============================================================================

@dataclass
class ExecutionContext:
    """Context for workflow execution."""
    workflow: Workflow
    prompt_handler: PromptHandler
    analysis_bridge: Any = None  # AnalysisBridge instance
    model_manager: Any = None  # ModelManager instance
    ui: Any = None  # TerminalUI instance
    current_model: Optional[str] = None
    last_result: Optional[Dict] = None


class WorkflowExecutor:
    """
    Executes workflows step by step.

    Handles interactive prompts, step execution, and result collection.
    """

    def __init__(self, context: ExecutionContext):
        """
        Initialize WorkflowExecutor.

        Args:
            context: Execution context with dependencies.
        """
        self.ctx = context
        self._step_handlers: Dict[str, Callable] = {
            "resolve_model": self._resolve_model,
            "resolve_models": self._resolve_models,
            "resolve_file": self._resolve_file,
            "resolve_mcp_target": self._resolve_mcp_target,
            "select_profile": self._select_profile,
            "select_purpose": self._select_purpose,
            "run_interrogation": self._run_interrogation,
            "run_comparison": self._run_comparison,
            "run_mcp_interrogation": self._run_mcp_interrogation,
            "run_code_analysis": self._run_code_analysis,
            "run_context_analysis": self._run_context_analysis,
            "show_summary": self._show_summary,
            "show_comparison": self._show_comparison,
            "list_available_models": self._list_available_models,
            "load_model": self._load_model,
            "generate_report": self._generate_report,
            "open_report": self._open_report,
            "get_last_analysis": self._get_last_analysis,
            "generate_explanation": self._generate_explanation,
            "determine_help_topic": self._determine_help_topic,
            "show_help": self._show_help,
            "show_status": self._show_status,
        }

    async def execute(self) -> Dict[str, Any]:
        """
        Execute the workflow.

        Returns:
            Result dictionary with success status and any outputs.
        """
        workflow = self.ctx.workflow

        result = {
            "success": False,
            "workflow": workflow.name,
            "message": "",
            "analysis_result": None,
            "report_path": None,
        }

        try:
            for i, step in enumerate(workflow.steps):
                workflow.current_step_index = i
                step.status = StepStatus.IN_PROGRESS

                handler = self._step_handlers.get(step.action)
                if handler:
                    try:
                        step_result = await handler(step)
                        step.result = step_result
                        step.status = StepStatus.COMPLETED

                        # Store results in context for later steps
                        if step_result:
                            workflow.context[step.action] = step_result

                    except Exception as e:
                        step.status = StepStatus.FAILED
                        step.error = str(e)
                        if step.required:
                            result["message"] = f"Step '{step.action}' failed: {e}"
                            return result
                else:
                    logger.warning(f"No handler for step: {step.action}")
                    step.status = StepStatus.SKIPPED

            result["success"] = True
            result["message"] = "Workflow completed successfully"
            result["analysis_result"] = workflow.context.get("analysis_result")
            result["report_path"] = workflow.context.get("report_path")

        except Exception as e:
            result["message"] = f"Workflow failed: {e}"
            logger.exception(f"Workflow execution failed: {e}")

        return result

    # =========================================================================
    # STEP HANDLERS
    # =========================================================================

    async def _resolve_model(self, step: WorkflowStep) -> Optional[str]:
        """Resolve model from context or ask user."""
        workflow = self.ctx.workflow

        # Check if model already in context
        model = workflow.context.get("model")
        if model:
            return model

        # Check entities
        entities = workflow.context.get("entities")
        if entities and entities.model_refs:
            return entities.model_refs[0]

        # Check current loaded model
        if self.ctx.current_model:
            return self.ctx.current_model

        # Need to ask user
        if step.ask_if_missing and self.ctx.model_manager:
            models = self.ctx.model_manager.get_downloaded_models()
            if models:
                model = await self.ctx.prompt_handler.ask_model(models)
                workflow.context["model"] = model
                return model

        raise ValueError("No model specified. Use /models to see available models.")

    async def _resolve_models(self, step: WorkflowStep) -> List[str]:
        """Resolve multiple models for comparison."""
        workflow = self.ctx.workflow
        models = []

        entities = workflow.context.get("entities")
        if entities and len(entities.model_refs) >= 2:
            return entities.model_refs[:2]

        raise ValueError("Comparison requires two models. Example: 'compare phi-2 to llama'")

    async def _resolve_file(self, step: WorkflowStep) -> Optional[str]:
        """Resolve file path from context."""
        workflow = self.ctx.workflow

        file_path = workflow.context.get("file")
        if file_path:
            return file_path

        entities = workflow.context.get("entities")
        if entities and entities.file_paths:
            return entities.file_paths[0]

        raise ValueError("No file specified.")

    async def _resolve_mcp_target(self, step: WorkflowStep) -> Optional[str]:
        """Resolve MCP target from context."""
        workflow = self.ctx.workflow

        url = workflow.context.get("url")
        if url:
            return url

        file_path = workflow.context.get("file")
        if file_path:
            return file_path

        entities = workflow.context.get("entities")
        if entities:
            if entities.urls:
                return entities.urls[0]
            if entities.file_paths:
                return entities.file_paths[0]

        raise ValueError("No MCP target specified. Provide a URL or file path.")

    async def _select_profile(self, step: WorkflowStep) -> str:
        """Select interrogation profile."""
        workflow = self.ctx.workflow

        # Check if already set
        profile = workflow.context.get("profile")
        if profile:
            return profile

        # Check entities
        entities = workflow.context.get("entities")
        if entities and entities.profile:
            workflow.context["profile"] = entities.profile
            return entities.profile

        # Use default if not required to ask
        if not step.ask_if_missing and step.default_value:
            workflow.context["profile"] = step.default_value
            return step.default_value

        # Ask user
        profile = await self.ctx.prompt_handler.ask_profile()
        workflow.context["profile"] = profile
        return profile

    async def _select_purpose(self, step: WorkflowStep) -> str:
        """Select model purpose (nlp or analysis)."""
        workflow = self.ctx.workflow

        # Use default if available
        if step.default_value and not step.ask_if_missing:
            return step.default_value

        # Ask user
        purpose = await self.ctx.prompt_handler.ask_purpose()
        workflow.context["purpose"] = purpose
        return purpose

    async def _run_interrogation(self, step: WorkflowStep) -> Dict[str, Any]:
        """Run model interrogation."""
        workflow = self.ctx.workflow
        model = workflow.context.get("resolve_model") or workflow.context.get("model")
        profile = workflow.context.get("select_profile") or workflow.context.get("profile", "standard")

        if not model:
            raise ValueError("No model to interrogate")

        if self.ctx.analysis_bridge:
            result = await self.ctx.analysis_bridge.analyze_model(model, profile)
            workflow.context["analysis_result"] = result
            return result

        raise ValueError("Analysis bridge not available")

    async def _run_comparison(self, step: WorkflowStep) -> Dict[str, Any]:
        """Run model comparison."""
        workflow = self.ctx.workflow
        models = workflow.context.get("resolve_models", [])
        profile = workflow.context.get("profile", "standard")

        if len(models) < 2:
            raise ValueError("Need two models for comparison")

        if self.ctx.analysis_bridge:
            result = await self.ctx.analysis_bridge.compare_models(models[0], models[1], profile)
            workflow.context["analysis_result"] = result
            return result

        raise ValueError("Analysis bridge not available")

    async def _run_mcp_interrogation(self, step: WorkflowStep) -> Dict[str, Any]:
        """Run MCP server interrogation."""
        workflow = self.ctx.workflow
        target = workflow.context.get("resolve_mcp_target")
        profile = workflow.context.get("profile", "standard")

        if not target:
            raise ValueError("No MCP target specified")

        if self.ctx.analysis_bridge:
            result = await self.ctx.analysis_bridge.analyze_mcp_server(target, profile)
            workflow.context["analysis_result"] = result
            return result

        raise ValueError("Analysis bridge not available")

    async def _run_code_analysis(self, step: WorkflowStep) -> Dict[str, Any]:
        """Run code analysis."""
        workflow = self.ctx.workflow
        file_path = workflow.context.get("resolve_file")

        if not file_path:
            raise ValueError("No file specified")

        if self.ctx.analysis_bridge:
            result = await self.ctx.analysis_bridge.analyze_code(file_path)
            workflow.context["analysis_result"] = result
            return result

        raise ValueError("Analysis bridge not available")

    async def _run_context_analysis(self, step: WorkflowStep) -> Dict[str, Any]:
        """Run context/instruction file analysis."""
        workflow = self.ctx.workflow
        file_path = workflow.context.get("resolve_file")

        if not file_path:
            raise ValueError("No file specified")

        if self.ctx.analysis_bridge:
            result = await self.ctx.analysis_bridge.analyze_skill(file_path)
            workflow.context["analysis_result"] = result
            return result

        raise ValueError("Analysis bridge not available")

    async def _show_summary(self, step: WorkflowStep) -> None:
        """Show analysis summary."""
        workflow = self.ctx.workflow
        result = workflow.context.get("analysis_result")

        if result and self.ctx.ui:
            self.ctx.ui.print_analysis_summary(result)

    async def _show_comparison(self, step: WorkflowStep) -> None:
        """Show comparison results."""
        # Same as summary for now
        await self._show_summary(step)

    async def _list_available_models(self, step: WorkflowStep) -> List[Dict]:
        """List available models."""
        if self.ctx.model_manager:
            models = self.ctx.model_manager.get_downloaded_models()
            return models
        return []

    async def _load_model(self, step: WorkflowStep) -> str:
        """Load a model."""
        workflow = self.ctx.workflow
        model = workflow.context.get("resolve_model") or workflow.context.get("model")
        purpose = workflow.context.get("select_purpose") or workflow.context.get("purpose", "analysis")

        if model:
            workflow.context["loaded_model"] = model
            workflow.context["loaded_purpose"] = purpose
            return model

        raise ValueError("No model to load")

    async def _generate_report(self, step: WorkflowStep) -> Optional[str]:
        """Generate HTML report."""
        workflow = self.ctx.workflow
        result = workflow.context.get("analysis_result")

        if not result:
            return None

        try:
            from benderbox.reporting.html_generator import HTMLReportGenerator
            from pathlib import Path
            from datetime import datetime

            generator = HTMLReportGenerator()
            html = generator.generate(result)

            # Generate filename
            target = result.get("target", "analysis")
            if isinstance(target, str):
                target = Path(target).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{target}_{timestamp}.html"

            # Save to reports directory
            from benderbox.config import get_config
            config = get_config()
            reports_dir = config.reports_dir
            reports_dir.mkdir(parents=True, exist_ok=True)

            report_path = reports_dir / filename
            report_path.write_text(html, encoding="utf-8")

            workflow.context["report_path"] = str(report_path)
            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return None

    async def _open_report(self, step: WorkflowStep) -> None:
        """Open report in browser."""
        workflow = self.ctx.workflow
        report_path = workflow.context.get("report_path")

        if report_path:
            import webbrowser
            webbrowser.open(f"file://{report_path}")

    async def _get_last_analysis(self, step: WorkflowStep) -> Optional[Dict]:
        """Get the last analysis result."""
        return self.ctx.last_result

    async def _generate_explanation(self, step: WorkflowStep) -> str:
        """Generate explanation of results."""
        # This would use LLM if available
        return "Explanation not implemented yet."

    async def _determine_help_topic(self, step: WorkflowStep) -> str:
        """Determine help topic from query."""
        workflow = self.ctx.workflow
        entities = workflow.context.get("entities")

        if entities:
            query = entities.normalized_query.lower()
            if "model" in query or "interrogat" in query:
                return "interrogate"
            if "mcp" in query:
                return "mcp"
            if "context" in query or "prompt" in query:
                return "context"
            if "report" in query:
                return "reports"

        return None  # General help

    async def _show_help(self, step: WorkflowStep) -> None:
        """Show help."""
        workflow = self.ctx.workflow
        topic = workflow.context.get("determine_help_topic")

        if self.ctx.ui:
            self.ctx.ui.print_help(topic)

    async def _show_status(self, step: WorkflowStep) -> None:
        """Show system status."""
        if self.ctx.ui:
            self.ctx.ui.print_info("System status: OK")
