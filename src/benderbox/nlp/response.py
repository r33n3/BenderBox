"""
Response Generator for BenderBox

Generates natural language responses for analysis results, explanations,
and knowledge queries using templates and LLM generation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from benderbox.nlp.intent import Intent, IntentType

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseContext:
    """Context for response generation."""

    intent: Intent
    user_query: str
    analysis_result: Optional[Dict[str, Any]] = None
    knowledge: List[Any] = field(default_factory=list)
    history: List[Message] = field(default_factory=list)
    error: Optional[str] = None


class ResponseGenerator:
    """
    Generates natural language responses for BenderBox.

    Uses a combination of templates for structured data and LLM
    generation for explanations and knowledge queries.
    """

    def __init__(self, llm_engine=None, knowledge_base=None):
        """
        Initialize ResponseGenerator.

        Args:
            llm_engine: LocalLLMEngine for generation.
            knowledge_base: KnowledgeBase for knowledge queries.
        """
        self._llm_engine = llm_engine
        self._knowledge_base = knowledge_base

    def _set_llm_engine(self, llm_engine) -> None:
        """Set the LLM engine (for lazy initialization)."""
        self._llm_engine = llm_engine

    def _set_knowledge_base(self, knowledge_base) -> None:
        """Set the knowledge base (for lazy initialization)."""
        self._knowledge_base = knowledge_base

    async def generate(self, context: ResponseContext) -> str:
        """
        Generate a response based on context.

        Args:
            context: ResponseContext with intent, results, etc.

        Returns:
            Generated response string.
        """
        # Handle errors first
        if context.error:
            return self._format_error(context.error)

        # Route to appropriate handler based on intent
        handlers = {
            IntentType.ANALYZE_MODEL: self._format_analysis_result,
            IntentType.ANALYZE_INFRASTRUCTURE: self._format_analysis_result,
            IntentType.ANALYZE_SKILL: self._format_analysis_result,
            IntentType.COMPARE: self._format_comparison,
            IntentType.EXPLAIN: self._generate_explanation,
            IntentType.QUERY_KNOWLEDGE: self._answer_knowledge_query,
            IntentType.GENERATE_REPORT: self._format_report,
            IntentType.LIST_REPORTS: self._format_report_list,
            IntentType.GET_STATUS: self._format_status,
            IntentType.HELP: self._format_help,
            IntentType.GENERAL_QUESTION: self._answer_general_question,
        }

        handler = handlers.get(context.intent.intent_type, self._answer_general_question)
        return await handler(context)

    async def generate_stream(self, context: ResponseContext) -> AsyncIterator[str]:
        """
        Generate response with streaming output.

        Args:
            context: ResponseContext with intent, results, etc.

        Yields:
            Response chunks.
        """
        # For non-LLM responses, yield complete response
        llm_available = self._llm_engine is not None and getattr(self._llm_engine, 'is_available', False)
        if not context.intent.requires_llm or not llm_available:
            response = await self.generate(context)
            yield response
            return

        # For LLM responses, stream generation
        prompt = self._build_response_prompt(context)

        async for chunk in self._llm_engine.generate_stream(
            prompt=prompt,
            model_type="analysis",
            max_tokens=1024,
            temperature=0.7,
        ):
            yield chunk

    async def _format_analysis_result(self, context: ResponseContext) -> str:
        """Format analysis result into natural language."""
        result = context.analysis_result
        if not result:
            return "No analysis results available."

        # Extract key information
        summary = result.get("summary", {})
        risk = summary.get("risk", {})
        risk_level = risk.get("level", "unknown").upper()
        risk_score = risk.get("score", 0)
        target_name = result.get("target_name", "Unknown target")
        profile = result.get("profile", "standard")

        # Count results by status
        results = result.get("results", [])
        passed = sum(1 for r in results if r.get("status") == "passed")
        failed = sum(1 for r in results if r.get("status") == "failed")
        warnings = sum(1 for r in results if r.get("status") == "warning")

        # Build response
        lines = [
            f"**Analysis Complete: {target_name}**",
            "",
            f"**Risk Level:** {risk_level} (Score: {risk_score}/100)",
            f"**Profile:** {profile}",
            "",
            "**Results Summary:**",
            f"- Passed: {passed}",
            f"- Failed: {failed}",
            f"- Warnings: {warnings}",
        ]

        # Add critical findings
        critical_findings = [r for r in results if r.get("severity") in ("critical", "high")]
        if critical_findings:
            lines.extend(["", "**Critical Findings:**"])
            for finding in critical_findings[:5]:  # Limit to top 5
                lines.append(f"- [{finding.get('severity', 'unknown').upper()}] {finding.get('test_name', 'Unknown')}")
                if finding.get("details", {}).get("message"):
                    lines.append(f"  {finding['details']['message'][:100]}...")

        # Add recommendation
        if risk_level == "CRITICAL":
            lines.extend(["", "**Recommendation:** Do not deploy. Immediate security review required."])
        elif risk_level == "HIGH":
            lines.extend(["", "**Recommendation:** Address critical issues before deployment."])
        elif risk_level == "MEDIUM":
            lines.extend(["", "**Recommendation:** Review findings and mitigate where possible."])
        else:
            lines.extend(["", "**Recommendation:** Safe for deployment with standard monitoring."])

        return "\n".join(lines)

    async def _format_comparison(self, context: ResponseContext) -> str:
        """Format comparison results."""
        result = context.analysis_result
        if not result:
            return "No comparison results available."

        if isinstance(result, list) and len(result) >= 2:
            # Compare two analysis results
            lines = ["**Comparison Results**", ""]

            headers = ["Metric", result[0].get("target_name", "Target 1"), result[1].get("target_name", "Target 2")]
            lines.append(f"| {' | '.join(headers)} |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")

            # Risk scores
            r1 = result[0].get("summary", {}).get("risk", {})
            r2 = result[1].get("summary", {}).get("risk", {})
            lines.append(f"| Risk Score | {r1.get('score', 'N/A')} | {r2.get('score', 'N/A')} |")
            lines.append(f"| Risk Level | {r1.get('level', 'N/A')} | {r2.get('level', 'N/A')} |")

            # Test counts
            t1 = len(result[0].get("results", []))
            t2 = len(result[1].get("results", []))
            lines.append(f"| Tests Run | {t1} | {t2} |")

            return "\n".join(lines)

        return "Comparison requires at least two targets."

    async def _generate_explanation(self, context: ResponseContext) -> str:
        """Generate explanation using LLM."""
        if self._llm_engine is None or not getattr(self._llm_engine, 'is_available', False):
            return self._template_explanation(context)

        prompt = self._build_explanation_prompt(context)
        return await self._llm_engine.generate(
            prompt=prompt,
            model_type="analysis",
            max_tokens=512,
            temperature=0.7,
        )

    def _template_explanation(self, context: ResponseContext) -> str:
        """Template-based explanation when LLM is unavailable."""
        result = context.analysis_result
        if not result:
            return "No analysis results to explain."

        query = context.user_query.lower()

        # Risk score explanation
        if "risk" in query or "score" in query:
            risk = result.get("summary", {}).get("risk", {})
            return (
                f"The risk score of {risk.get('score', 0)} indicates "
                f"{risk.get('level', 'unknown')} risk level. "
                f"This is calculated based on the severity and number of findings "
                f"from the security tests."
            )

        # Finding explanation
        if "finding" in query or "result" in query:
            results = result.get("results", [])
            failed = [r for r in results if r.get("status") == "failed"]
            if failed:
                finding = failed[0]
                return (
                    f"The '{finding.get('test_name')}' test failed because: "
                    f"{finding.get('details', {}).get('message', 'No details available')}"
                )

        return "Please specify what you would like me to explain."

    async def _answer_knowledge_query(self, context: ResponseContext) -> str:
        """Answer knowledge base query."""
        if context.knowledge:
            # Format knowledge entries
            lines = ["**Security Knowledge:**", ""]
            for entry in context.knowledge[:5]:
                lines.append(f"**{entry.name}** ({entry.severity})")
                lines.append(f"{entry.description}")
                if hasattr(entry, "mitigations") and entry.mitigations:
                    lines.append("Mitigations:")
                    for m in entry.mitigations[:3]:
                        lines.append(f"- {m}")
                lines.append("")
            return "\n".join(lines)

        # Use LLM if available
        if self._llm_engine and getattr(self._llm_engine, 'is_available', False):
            prompt = self._build_knowledge_prompt(context)
            return await self._llm_engine.generate(
                prompt=prompt,
                model_type="analysis",
                max_tokens=512,
                temperature=0.7,
            )

        return "I don't have specific information about that topic in my knowledge base."

    async def _format_report(self, context: ResponseContext) -> str:
        """Format generated report."""
        result = context.analysis_result
        if not result:
            return "No data available for report generation."

        # Generate model card style report
        lines = [
            "# Security Analysis Report",
            "",
            f"**Target:** {result.get('target_name', 'Unknown')}",
            f"**Date:** {result.get('timestamp', 'Unknown')}",
            f"**Profile:** {result.get('profile', 'standard')}",
            "",
            "## Executive Summary",
            "",
        ]

        summary = result.get("summary", {})
        risk = summary.get("risk", {})

        lines.append(
            f"This analysis found a **{risk.get('level', 'unknown').upper()}** risk level "
            f"with a score of {risk.get('score', 0)}/100."
        )

        results = result.get("results", [])
        lines.append(f"A total of {len(results)} security tests were executed.")
        lines.append("")

        # Add findings section
        lines.extend(["## Key Findings", ""])
        critical = [r for r in results if r.get("severity") == "critical"]
        high = [r for r in results if r.get("severity") == "high"]

        if critical:
            lines.append("### Critical Issues")
            for f in critical:
                lines.append(f"- {f.get('test_name')}: {f.get('details', {}).get('message', 'No details')[:100]}")
            lines.append("")

        if high:
            lines.append("### High Severity Issues")
            for f in high:
                lines.append(f"- {f.get('test_name')}: {f.get('details', {}).get('message', 'No details')[:100]}")
            lines.append("")

        lines.extend(["## Recommendations", ""])
        if risk.get("level") in ("critical", "high"):
            lines.append("- Address all critical and high severity issues before deployment")
            lines.append("- Conduct thorough security review")
        else:
            lines.append("- Continue standard security monitoring")
            lines.append("- Re-analyze periodically for drift detection")

        return "\n".join(lines)

    async def _format_report_list(self, context: ResponseContext) -> str:
        """Format list of reports."""
        result = context.analysis_result
        if not result or not isinstance(result, list):
            return "No reports found."

        lines = ["**Recent Reports:**", ""]
        for report in result[:10]:
            risk_level = report.risk_level if hasattr(report, "risk_level") else report.get("risk_level", "unknown")
            target = report.target_name if hasattr(report, "target_name") else report.get("target_name", "unknown")
            timestamp = report.timestamp if hasattr(report, "timestamp") else report.get("timestamp", "unknown")
            lines.append(f"- [{risk_level.upper()}] {target} ({timestamp})")

        return "\n".join(lines)

    async def _format_status(self, context: ResponseContext) -> str:
        """Format system status."""
        result = context.analysis_result or {}

        lines = [
            "**BenderBox Status**",
            "",
            f"Version: {result.get('version', '3.0.0-alpha')}",
            "",
            "**Models:**",
        ]

        models = result.get("models", {})
        for model_type, info in models.items():
            status = "Loaded" if info.get("loaded") else "Not loaded"
            exists = "Found" if info.get("exists") else "Missing"
            lines.append(f"- {model_type}: {status} ({exists})")

        lines.extend([
            "",
            "**Storage:**",
            f"- Reports: {result.get('report_count', 0)}",
            f"- Knowledge entries: {result.get('knowledge_count', 0)}",
        ])

        return "\n".join(lines)

    async def _format_help(self, context: ResponseContext) -> str:
        """Format help message."""
        return """**BenderBox - AI Security Analysis Platform**

**Commands:**
- Analyze a model: "Analyze model.gguf for security issues"
- Analyze MCP server: "Check server.py for vulnerabilities"
- Compare models: "Compare model1.gguf vs model2.gguf"
- Explain findings: "Explain why the risk score is high"
- Query knowledge: "What are jailbreak techniques?"
- List reports: "Show recent analysis reports"
- Get status: "Show system status"

**Analysis Profiles:**
- quick: Fast basic checks
- standard: Balanced analysis (default)
- deep: Comprehensive with semantic analysis

**Examples:**
- "Is mistral-7b.gguf safe for production?"
- "Analyze mcp_server.py with deep profile"
- "What vulnerabilities were found in the last scan?"
"""

    async def _answer_general_question(self, context: ResponseContext) -> str:
        """Answer general questions using LLM."""
        if self._llm_engine is None or not getattr(self._llm_engine, 'is_available', False):
            return (
                "I need an LLM model loaded to answer general questions. "
                "Please ensure llama-cpp-python is installed and a model is available.\n\n"
                "For now, I can help with analysis commands like:\n"
                "- 'analyze model.gguf'\n"
                "- 'status'\n"
                "- 'help'"
            )

        prompt = self._build_general_prompt(context)
        return await self._llm_engine.generate(
            prompt=prompt,
            model_type="analysis",
            max_tokens=512,
            temperature=0.7,
        )

    def _format_error(self, error: str) -> str:
        """Format error message."""
        return f"**Error:** {error}"

    def _build_response_prompt(self, context: ResponseContext) -> str:
        """Build prompt for LLM response generation."""
        history_str = ""
        if context.history:
            for msg in context.history[-5:]:  # Last 5 messages
                history_str += f"{msg.role}: {msg.content}\n"

        return f"""You are BenderBox, an AI security analysis assistant.
Answer the user's question helpfully and concisely.

Conversation history:
{history_str}

User query: {context.user_query}

Response:"""

    def _build_explanation_prompt(self, context: ResponseContext) -> str:
        """Build prompt for explanation generation."""
        result_str = ""
        if context.analysis_result:
            import json
            result_str = json.dumps(context.analysis_result, indent=2)[:2000]

        return f"""You are a security expert explaining analysis results.
Explain the following in clear, non-technical language.

Analysis results:
{result_str}

User question: {context.user_query}

Provide a clear, helpful explanation:"""

    def _build_knowledge_prompt(self, context: ResponseContext) -> str:
        """Build prompt for knowledge query."""
        return f"""You are a cybersecurity expert. Answer the following question
about AI security, jailbreaks, vulnerabilities, or related topics.

Question: {context.user_query}

Provide a clear, accurate answer:"""

    def _build_general_prompt(self, context: ResponseContext) -> str:
        """Build prompt for general questions."""
        return f"""You are BenderBox, an AI security analysis assistant.
Answer the following question helpfully.

Question: {context.user_query}

Answer:"""
