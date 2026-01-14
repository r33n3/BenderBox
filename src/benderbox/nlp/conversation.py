"""
Conversation Manager for BenderBox

Orchestrates multi-turn conversations by coordinating:
- Intent classification
- Context management
- Analysis execution
- Response generation
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from benderbox.nlp.intent import Intent, IntentRouter, IntentType
from benderbox.nlp.context import ContextManager
from benderbox.nlp.response import Message, ResponseContext, ResponseGenerator
from benderbox.nlp.analysis_bridge import AnalysisBridge

logger = logging.getLogger(__name__)


@dataclass
class Response:
    """Response from conversation processing."""

    content: str
    intent: Intent
    analysis_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConversationManager:
    """
    Manages multi-turn conversations with context awareness.

    Coordinates:
    - Intent classification
    - Analysis execution
    - Knowledge retrieval
    - Response generation
    - Context tracking
    """

    MAX_HISTORY = 20  # Maximum messages to keep in history

    def __init__(
        self,
        llm_engine=None,
        knowledge_base=None,
        report_db=None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize ConversationManager.

        Args:
            llm_engine: LocalLLMEngine for NLP tasks.
            knowledge_base: KnowledgeBase for threat knowledge.
            report_db: ReportDatabase for stored reports.
            log_dir: Directory for analysis logs.
        """
        self._llm_engine = llm_engine
        self._knowledge_base = knowledge_base
        self._report_db = report_db

        # Initialize components
        self._intent_router = IntentRouter(llm_engine)
        self._context_manager = ContextManager()
        self._response_generator = ResponseGenerator(llm_engine, knowledge_base)
        self._analysis_bridge = AnalysisBridge(
            log_dir=log_dir,
            llm_engine=llm_engine,
            knowledge_base=knowledge_base,
        )

        # Conversation history
        self._history: List[Message] = []

        # Session metadata
        self._session_start = datetime.now()

    def _set_llm_engine(self, llm_engine) -> None:
        """Set the LLM engine (for lazy initialization)."""
        self._llm_engine = llm_engine
        self._intent_router._set_llm_engine(llm_engine)
        self._response_generator._set_llm_engine(llm_engine)
        self._analysis_bridge._set_llm_engine(llm_engine)

    def _set_knowledge_base(self, knowledge_base) -> None:
        """Set the knowledge base (for lazy initialization)."""
        self._knowledge_base = knowledge_base
        self._response_generator._set_knowledge_base(knowledge_base)
        self._analysis_bridge._set_knowledge_base(knowledge_base)

    async def process_query(self, user_input: str) -> Response:
        """
        Process a user query and generate a response.

        Args:
            user_input: User's natural language input.

        Returns:
            Response object with generated content.
        """
        user_input = user_input.strip()

        if not user_input:
            return Response(
                content="Please enter a question or command.",
                intent=Intent(intent_type=IntentType.UNKNOWN, confidence=0.0),
            )

        # Increment turn count
        self._context_manager.increment_turn()

        # Add user message to history
        self._add_to_history("user", user_input)

        try:
            # Resolve any references in the query
            resolved_query = self._context_manager.resolve_query_references(user_input)

            # Classify intent
            intent = await self._intent_router.classify(resolved_query)
            logger.debug(f"Classified intent: {intent.intent_type.value} ({intent.confidence:.2f})")

            # Execute based on intent
            analysis_result = None
            knowledge = []
            error = None

            if intent.requires_analysis:
                try:
                    analysis_result = await self._execute_analysis(intent, resolved_query)
                    if analysis_result:
                        # Only add to context if it's a dict (some results are dataclasses)
                        if isinstance(analysis_result, dict):
                            self._context_manager.add_analysis_result(analysis_result)
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    error = str(e)

            # Retrieve relevant knowledge if needed
            if intent.intent_type == IntentType.QUERY_KNOWLEDGE and self._knowledge_base:
                try:
                    knowledge = await self._retrieve_knowledge(resolved_query)
                except Exception as e:
                    logger.warning(f"Knowledge retrieval failed: {e}")

            # Handle special intents
            if intent.intent_type == IntentType.LIST_REPORTS:
                try:
                    reports = await self._analysis_bridge.get_report_list(limit=10)
                    analysis_result = reports
                except Exception as e:
                    error = str(e)

            if intent.intent_type == IntentType.GET_STATUS:
                analysis_result = self._get_status()

            if intent.intent_type == IntentType.VIEW_REPORTS:
                try:
                    from benderbox.reporting.index_generator import ReportViewerGenerator
                    generator = ReportViewerGenerator()
                    reports = generator.collect_reports()
                    if reports:
                        output_path = generator.save(open_browser=True)
                        analysis_result = {
                            "action": "opened_report_viewer",
                            "report_count": len(reports),
                            "output_path": output_path,
                        }
                    else:
                        analysis_result = {
                            "action": "no_reports",
                            "message": "No reports found. Run an analysis first.",
                        }
                except Exception as e:
                    error = str(e)

            # Build response context
            response_context = ResponseContext(
                intent=intent,
                user_query=user_input,
                analysis_result=analysis_result,
                knowledge=knowledge,
                history=self._history[-10:],  # Recent history
                error=error,
            )

            # Generate response
            response_content = await self._response_generator.generate(response_context)

            # Add assistant message to history
            self._add_to_history("assistant", response_content)

            return Response(
                content=response_content,
                intent=intent,
                analysis_result=analysis_result,
                metadata={
                    "turn": self._context_manager.context.turn_count,
                    "resolved_query": resolved_query if resolved_query != user_input else None,
                },
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = f"Sorry, I encountered an error: {e}"
            self._add_to_history("assistant", error_response)
            return Response(
                content=error_response,
                intent=Intent(intent_type=IntentType.UNKNOWN, confidence=0.0),
                metadata={"error": str(e)},
            )

    async def process_query_stream(self, user_input: str) -> AsyncIterator[str]:
        """
        Process query with streaming response.

        Args:
            user_input: User's natural language input.

        Yields:
            Response content chunks.
        """
        # For streaming, we need LLM
        if self._llm_engine is None:
            response = await self.process_query(user_input)
            yield response.content
            return

        user_input = user_input.strip()
        self._context_manager.increment_turn()
        self._add_to_history("user", user_input)

        try:
            resolved_query = self._context_manager.resolve_query_references(user_input)
            intent = await self._intent_router.classify(resolved_query)

            # Non-LLM intents don't stream
            if not intent.requires_llm:
                response = await self.process_query(user_input)
                yield response.content
                return

            # Build context and stream response
            analysis_result = None
            if intent.requires_analysis:
                analysis_result = await self._execute_analysis(intent, resolved_query)
                if analysis_result:
                    self._context_manager.add_analysis_result(analysis_result)

            knowledge = []
            if intent.intent_type == IntentType.QUERY_KNOWLEDGE and self._knowledge_base:
                knowledge = await self._retrieve_knowledge(resolved_query)

            response_context = ResponseContext(
                intent=intent,
                user_query=user_input,
                analysis_result=analysis_result,
                knowledge=knowledge,
                history=self._history[-10:],
            )

            # Collect streamed content for history
            full_response = ""
            async for chunk in self._response_generator.generate_stream(response_context):
                full_response += chunk
                yield chunk

            self._add_to_history("assistant", full_response)

        except Exception as e:
            error_msg = f"Error: {e}"
            self._add_to_history("assistant", error_msg)
            yield error_msg

    async def _execute_analysis(
        self,
        intent: Intent,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """Execute analysis based on intent."""
        target = intent.parameters.get("target")
        targets = intent.parameters.get("targets", [])
        profile = intent.parameters.get("profile", "standard")

        # Handle comparison
        if intent.intent_type == IntentType.COMPARE and targets:
            result = await self._analysis_bridge.compare_targets(targets)
            # Return as dict to match other analysis methods
            return {
                "targets": result.targets,
                "comparison_type": result.comparison_type,
                "results": result.results,
                "summary": result.summary,
            }

        # Determine target from context if not specified
        if not target:
            target = self._context_manager.get_current_target()

        if not target:
            # Try to extract from query
            import re
            # Match various file types: models, code, configs
            path_match = re.search(
                r'["\']?([^\s"\']+\.(gguf|py|js|ts|c|cpp|h|go|rs|rb|java|php|md|yaml|yml|json))["\']?',
                query
            )
            if path_match:
                target = path_match.group(1)

        if not target:
            return None

        # Determine analysis type and execute
        # IMPORTANT: Check explicit intents FIRST before file extension fallbacks

        # Context analysis (instruction files, prompts, skills for security)
        if intent.intent_type == IntentType.CONTEXT_ANALYZE:
            from benderbox.analyzers.context_analyzer import analyze_context_file
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analyze_context_file(target)
            )
            return result

        # Semantic code analysis (explicit intent takes priority)
        if intent.intent_type == IntentType.ANALYZE_CODE:
            depth = profile if profile in ("quick", "standard", "deep") else "standard"
            language = intent.parameters.get("language", "python")

            # Check if target is a file path or inline code
            from pathlib import Path
            if target and Path(target).exists():
                return await self._analysis_bridge.analyze_file_semantic(target, depth=depth)
            elif target:
                # Target might be code content passed directly
                return await self._analysis_bridge.analyze_code(
                    code=target,
                    language=language,
                    depth=depth,
                )
            else:
                return None

        # Model analysis (explicit intent or .gguf extension)
        elif intent.intent_type == IntentType.ANALYZE_MODEL or target.endswith(".gguf"):
            if profile.startswith("infra"):
                profile = "standard"
            return await self._analysis_bridge.analyze_model(target, profile)

        # Infrastructure analysis (explicit intent or .py for MCP servers)
        elif intent.intent_type == IntentType.ANALYZE_INFRASTRUCTURE or target.endswith(".py"):
            if not profile.startswith("infra"):
                profile = "infra-standard"
            return await self._analysis_bridge.analyze_mcp_server(target, profile)

        # Skill analysis (explicit intent or .md/.yaml)
        elif intent.intent_type == IntentType.ANALYZE_SKILL or target.endswith((".md", ".yaml")):
            if not profile.startswith("infra"):
                profile = "infra-quick"
            return await self._analysis_bridge.analyze_skill(target, profile)

        # Behavior analysis
        elif intent.intent_type == IntentType.ANALYZE_BEHAVIOR:
            return await self._analysis_bridge.analyze_behavior(target, profile)

        return None

    async def _retrieve_knowledge(self, query: str) -> List[Any]:
        """Retrieve relevant knowledge for query."""
        if self._knowledge_base is None:
            return []

        try:
            return await self._knowledge_base.search(query, top_k=5)
        except Exception as e:
            logger.warning(f"Knowledge retrieval failed: {e}")
            return []

    def _get_status(self) -> Dict[str, Any]:
        """Get system status."""
        status = {
            "version": "3.0.0-alpha",
            "session_start": self._session_start.isoformat(),
            "turn_count": self._context_manager.context.turn_count,
            "history_length": len(self._history),
        }

        # Add analysis bridge status
        try:
            status.update(self._analysis_bridge.get_status())
        except Exception:
            pass

        # Add LLM status
        if self._llm_engine:
            try:
                status["models"] = self._llm_engine.get_model_info().get("models", {})
            except Exception:
                status["models"] = {}
        else:
            status["models"] = {"note": "LLM engine not configured"}

        # Add knowledge base status
        if self._knowledge_base:
            try:
                status["knowledge_count"] = self._knowledge_base.get_stats().get("total_entries", 0)
            except Exception:
                status["knowledge_count"] = 0
        else:
            status["knowledge_count"] = 0

        return status

    def _add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self._history.append(Message(
            role=role,
            content=content,
            metadata={"timestamp": datetime.now().isoformat()},
        ))

        # Trim history if needed
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        self._context_manager.clear()
        logger.debug("Conversation history cleared")

    def get_history(self) -> List[Message]:
        """Get conversation history."""
        return self._history.copy()

    async def summarize_conversation(self) -> str:
        """Generate summary of conversation."""
        if not self._history:
            return "No conversation history."

        if self._llm_engine:
            # Use LLM to summarize
            history_text = "\n".join([
                f"{m.role}: {m.content[:200]}..."
                for m in self._history[-10:]
            ])

            prompt = f"""Summarize this conversation briefly:

{history_text}

Summary:"""

            return await self._llm_engine.generate(
                prompt=prompt,
                model_type="analysis",
                max_tokens=200,
                temperature=0.5,
            )
        else:
            # Simple summary without LLM
            return f"Conversation with {len(self._history)} messages, {self._context_manager.context.turn_count} turns."

    def get_context_summary(self) -> str:
        """Get summary of current context."""
        return self._context_manager.get_summary()


def _load_banner() -> str:
    """Load the BenderBox ASCII banner."""
    from pathlib import Path

    # Try to find the ASCII art file
    ascii_file = Path(__file__).parent.parent.parent.parent / "BenderBox_ASCII.txt"

    if ascii_file.exists():
        try:
            return ascii_file.read_text(encoding="utf-8")
        except Exception:
            pass

    # Fallback banner
    return """
    ⢀⣤⣤⡀
   ⢠⣿⣿⣿⣧
   ⢸⣿⠛⠛⣿⡄  BenderBox v3.0
   ⠈⣿⣄⣠⣿⠃  AI Security Analysis
    ⠈⠛⠛⠁
     ⣿⣿⣿⡟
    ⢠⡟  ⢻⡄
"""


async def chat_loop(conversation: ConversationManager) -> None:
    """
    Interactive chat loop.

    Args:
        conversation: ConversationManager instance.
    """
    # Display banner
    banner = _load_banner()
    print(banner)
    print("Type 'help' for commands, 'exit' to quit")
    print('"I\'m 40% security analysis!"')
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                conversation.clear_history()
                print("History cleared.\n")
                continue

            if user_input.lower() == "context":
                print(conversation.get_context_summary())
                print()
                continue

            if user_input.lower() == "history":
                for msg in conversation.get_history()[-10:]:
                    print(f"{msg.role}: {msg.content[:100]}...")
                print()
                continue

            # Process query
            response = await conversation.process_query(user_input)
            print(f"\nBenderBox: {response.content}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


def main() -> int:
    """Main entry point for chat CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="BenderBox Chat - AI Security Assistant")
    parser.add_argument("--log-dir", default="./sandbox_logs", help="Analysis log directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Initialize conversation manager
    # Note: LLM and knowledge base loaded lazily if available
    conversation = ConversationManager(log_dir=args.log_dir)

    # Try to load optional components
    try:
        from benderbox.nlp.llm_engine import LocalLLMEngine
        from benderbox.config import get_config

        config = get_config()
        llm = LocalLLMEngine(config.llm)
        conversation._set_llm_engine(llm)
        print("LLM engine configured.")
    except Exception as e:
        print(f"Note: LLM not available ({e}). Using template responses.")

    try:
        from benderbox.storage.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        conversation._set_knowledge_base(kb)
        print("Knowledge base configured.")
    except Exception as e:
        print(f"Note: Knowledge base not available ({e}).")

    # Run chat loop
    try:
        asyncio.run(chat_loop(conversation))
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
