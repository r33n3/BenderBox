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

# Strong system prompt to prevent hallucination
BENDERBOX_SYSTEM_PROMPT = """You are BenderBox, an AI model security analysis assistant.

YOUR PURPOSE (only these tasks):
- Analyze GGUF models for security vulnerabilities
- Test AI models for jailbreaks and prompt injection
- Audit MCP servers for security issues
- Analyze instruction files and system prompts for risks
- Compare models for safety characteristics

YOU ARE NOT:
- A general-purpose AI assistant
- A tool for running models in production
- For self-driving cars, image generation, or other AI applications
- A malware research tool (you analyze security, not create threats)

IMPORTANT RULES:
1. Only answer questions about AI security analysis
2. If asked about unrelated topics, politely explain your purpose
3. Never make up analysis results - only report real data
4. When unsure, suggest using specific BenderBox commands
5. Always ground responses in BenderBox's actual capabilities

AVAILABLE COMMANDS:
- analyze <file> - Analyze model or code for security
- mcp analyze/interrogate - Test MCP servers
- context analyze - Analyze prompts and instructions
- compare - Compare two targets
- help - Show available commands
"""


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
        """Format help message with optional sub-topic."""
        query = context.user_query.lower() if context.user_query else ""

        # Check for sub-help topics
        if "help mcp" in query or "mcp help" in query:
            return self._get_mcp_help()
        elif "help context" in query or "context help" in query or "help prompt" in query:
            return self._get_context_help()
        elif "help model" in query or "model help" in query or "help analyze" in query:
            return self._get_model_help()
        elif "help example" in query or "example" in query:
            return self._get_examples_help()

        return """**BenderBox - AI Security Analysis Platform**

**Core Commands:**
- `status` - Show system status
- `help` - Show this help message
- `help mcp` - MCP server analysis help
- `help context` - Context/prompt analysis help
- `help models` - Model analysis help
- `help examples` - Show example files
- `exit` / `quit` - Exit BenderBox

**Quick Start - Try These Examples:**
```
context analyze examples/prompts/risky_system_prompt.md
mcp analyze examples/mcp_servers/sample_vulnerable_server.py
context scan examples/skills/
```

**Analysis Types:**
1. **MCP Servers** - Test MCP servers for vulnerabilities
2. **Context/Prompts** - Analyze instruction files for risks
3. **Models** - Test AI models for safety/censorship

**Natural Language:**
- "Analyze this file for security issues"
- "Is this MCP server safe?"
- "What jailbreak patterns were found?"
- "Explain why the risk score is high"

**Profiles:** quick | standard | full

Type `help <topic>` for detailed help on: mcp, context, models, examples
"""

    def _get_mcp_help(self) -> str:
        """Get MCP-specific help."""
        return """**MCP Server Security Analysis**

**Commands:**
- `mcp tools <target>` - List tools from MCP server
- `mcp interrogate <target>` - Run security tests
- `mcp analyze <url>` - Static analysis of server code
- `mcp call <target> <tool>` - Call a specific tool

**Target Formats:**
- STDIO: `npx @modelcontextprotocol/server-filesystem .`
- HTTP: `https://mcp.example.com/api`
- Local: `node server.js`

**Try This Example:**
```
mcp analyze examples/mcp_servers/sample_vulnerable_server.py
```
This sample has intentional vulnerabilities (command injection, path traversal, SQL injection) that BenderBox will detect.

**Profiles:**
- `quick` - Fast scan (~15 tests)
- `standard` - Balanced (~30 tests)
- `full` - Comprehensive (~50 tests)

**Example Commands:**
```
mcp tools "npx @modelcontextprotocol/server-filesystem ."
mcp interrogate "npx @server" --profile quick
mcp analyze https://github.com/org/mcp-server
```
"""

    def _get_context_help(self) -> str:
        """Get context/prompt analysis help."""
        return """**Context & Prompt Security Analysis**

**Commands:**
- `context analyze <file>` - Analyze instruction file
- `context scan <dir>` - Scan directory for risky files
- `context output <text>` - Analyze model output

**What It Detects:**
- Jailbreak instructions ("ignore previous instructions")
- Credential exposure (API keys, passwords)
- Code execution risks ("execute any code")
- Data exfiltration patterns
- Safety bypass instructions

**Try These Examples:**
```
# Safe prompt - should have no findings
context analyze examples/prompts/safe_system_prompt.md

# Risky prompt - should flag CRITICAL issues
context analyze examples/prompts/risky_system_prompt.md

# Scan all skill files
context scan examples/skills/ --pattern "*.md"
```

**File Types:**
- System prompts (`.md`, `.txt`)
- Skill definitions (`.md`, `.yaml`)
- Agent instructions
- Model outputs

**Risk Levels:** SAFE | LOW | MEDIUM | HIGH | CRITICAL
"""

    def _get_model_help(self) -> str:
        """Get model analysis help."""
        return """**AI Model Security Analysis**

**Commands:**
- `analyze <model>` - Analyze model for security
- `interrogate <target>` - Test model safety/censorship
- `compare <a> <b>` - Compare two models

**Target Formats:**
- Local GGUF: `./models/llama-7b.gguf`
- API Models: `openai:gpt-4-turbo`, `anthropic:claude-3-sonnet`

**Get a Test Model:**
```
# Download TinyLlama (~700MB)
python bb.py models download tinyllama

# Test the model works
python bb.py models test
```

**Example Commands:**
```
analyze ./models/tinyllama.gguf --profile quick
interrogate openai:gpt-4-turbo --profile quick
compare model1.gguf model2.gguf
```

**Profiles:**
- `quick` - Fast validation (~15 tests)
- `standard` - Balanced analysis (~50 tests)
- `full` - Comprehensive audit (~128 tests)

**What It Tests:**
- Jailbreak resistance
- Content filtering
- Bias detection
- Safety guardrails
"""

    def _get_examples_help(self) -> str:
        """Get examples help."""
        return """**BenderBox Examples**

Example files are in the `examples/` folder:

**Prompt Analysis Examples:**
```
examples/prompts/
├── safe_system_prompt.md      # Well-designed prompt
└── risky_system_prompt.md     # Dangerous patterns (for testing)
```
Try: `context analyze examples/prompts/risky_system_prompt.md`

**MCP Server Examples:**
```
examples/mcp_servers/
└── sample_vulnerable_server.py  # Server with vulnerabilities
```
Try: `mcp analyze examples/mcp_servers/sample_vulnerable_server.py`

**Skill File Examples:**
```
examples/skills/
├── analyze_gguf_model.md
├── analyze_mcp_server.md
├── analyze_skill_security.md
└── ... (7 skill templates)
```
Try: `context analyze examples/skills/analyze_mcp_server.md`

**Quick Test All Examples:**
```
context analyze examples/prompts/risky_system_prompt.md
mcp analyze examples/mcp_servers/sample_vulnerable_server.py
context scan examples/skills/
```

See `examples/README.md` for full documentation.
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

        return f"""{BENDERBOX_SYSTEM_PROMPT}

Conversation history:
{history_str}

User query: {context.user_query}

Provide a helpful response about security analysis. If the question is outside your scope, politely redirect to your capabilities:"""

    def _build_explanation_prompt(self, context: ResponseContext) -> str:
        """Build prompt for explanation generation."""
        result_str = ""
        if context.analysis_result:
            import json
            result_str = json.dumps(context.analysis_result, indent=2)[:2000]

        return f"""{BENDERBOX_SYSTEM_PROMPT}

You are explaining REAL analysis results from BenderBox. Only describe what is in the data below.

Analysis results:
{result_str}

User question: {context.user_query}

Explain these results clearly and accurately. Do not make up findings that are not in the data:"""

    def _build_knowledge_prompt(self, context: ResponseContext) -> str:
        """Build prompt for knowledge query."""
        knowledge_str = ""
        if context.knowledge:
            knowledge_str = "\n".join([
                f"- {entry.name}: {entry.description[:200]}"
                for entry in context.knowledge[:5]
            ])

        return f"""{BENDERBOX_SYSTEM_PROMPT}

You are answering a question about AI security topics within BenderBox's scope.

ONLY answer about:
- AI model security vulnerabilities
- Jailbreak techniques and defenses
- MCP server security
- Prompt injection attacks
- Model safety testing

If the question is outside these topics, explain that BenderBox focuses on AI security analysis.

{f"Relevant knowledge from database:{chr(10)}{knowledge_str}" if knowledge_str else "No specific knowledge entries found for this query."}

Question: {context.user_query}

Provide a focused answer about AI security:"""

    def _build_general_prompt(self, context: ResponseContext) -> str:
        """Build prompt for general questions."""
        history_str = ""
        if context.history:
            for msg in context.history[-3:]:  # Last 3 messages for context
                history_str += f"{msg.role}: {msg.content[:100]}...\n"

        return f"""{BENDERBOX_SYSTEM_PROMPT}

{f"Recent conversation:{chr(10)}{history_str}" if history_str else ""}

User question: {context.user_query}

If this question is about AI security analysis, MCP servers, model testing, or prompt security, answer it.
If it's about something else (self-driving cars, image generation, general coding, etc.), politely explain:
"I'm BenderBox, an AI security analysis tool. I can help you analyze models, test MCP servers, and review prompts for security issues. Try 'help' to see what I can do."

Response:"""
