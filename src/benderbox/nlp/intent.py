"""
Intent Router for BenderBox

Classifies user queries into intents for routing to appropriate handlers.
Uses a combination of fast keyword matching and LLM classification for
ambiguous queries.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents."""

    ANALYZE_MODEL = "analyze_model"
    ANALYZE_INFRASTRUCTURE = "analyze_infrastructure"
    ANALYZE_SKILL = "analyze_skill"
    ANALYZE_CODE = "analyze_code"  # Semantic code analysis
    ANALYZE_BEHAVIOR = "analyze_behavior"  # Behavior analysis
    MCP_TOOLS = "mcp_tools"  # List MCP server tools
    MCP_INTERROGATE = "mcp_interrogate"  # Security test MCP server
    MCP_CALL = "mcp_call"  # Call MCP tool
    CONTEXT_ANALYZE = "context_analyze"  # Analyze instruction file
    CONTEXT_SCAN = "context_scan"  # Scan directory for risky files
    LIST_MODELS = "list_models"  # List available models
    DOWNLOAD_MODEL = "download_model"  # Download a model
    LOAD_MODEL = "load_model"  # Load a model for analysis/nlp
    LOAD_CUSTOM_TESTS = "load_custom_tests"  # Load custom interrogation tests from file
    COMPARE = "compare"
    EXPLAIN = "explain"
    QUERY_KNOWLEDGE = "query_knowledge"
    GENERATE_REPORT = "generate_report"
    LIST_REPORTS = "list_reports"
    VIEW_REPORTS = "view_reports"  # Open report viewer
    VIEW_DOCS = "view_docs"  # Open documentation/instructions
    GET_STATUS = "get_status"
    EXPORT_RESULTS = "export_results"  # Export reports to various formats
    CONFIG_MANAGE = "config_manage"  # Configuration management
    SEMANTIC_SEARCH = "semantic_search"  # Search reports and findings
    SHOW_VERSION = "show_version"  # Show version information
    CHECK_PREREQUISITES = "check_prerequisites"  # Check system prerequisites
    GENERAL_QUESTION = "general_question"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Classified intent with parameters."""

    intent_type: IntentType
    confidence: float  # 0.0 to 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_analysis: bool = False
    requires_llm: bool = False
    raw_query: str = ""

    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is high enough for direct action."""
        return self.confidence >= 0.8


# Keyword patterns for fast intent matching
# Format: (pattern, intent_type, requires_analysis, requires_llm)
KEYWORD_PATTERNS: List[Tuple[re.Pattern, IntentType, bool, bool]] = [
    # MCP server intents (high priority - check first)
    (re.compile(r"\bmcp\s+tools?\b", re.I),
     IntentType.MCP_TOOLS, True, False),
    (re.compile(r"\bmcp\s+interrogate\b", re.I),
     IntentType.MCP_INTERROGATE, True, False),
    (re.compile(r"\bmcp\s+call\b", re.I),
     IntentType.MCP_CALL, True, False),
    (re.compile(r"\b(list|show|get)\b.*\b(mcp|server)\b.*\btools?\b", re.I),
     IntentType.MCP_TOOLS, True, False),
    (re.compile(r"\b(test|interrogate|security)\b.*\b(mcp|server)\b", re.I),
     IntentType.MCP_INTERROGATE, True, False),

    # Context analysis intents
    (re.compile(r"\bcontext\s+analyze\b", re.I),
     IntentType.CONTEXT_ANALYZE, True, False),
    (re.compile(r"\bcontext\s+scan\b", re.I),
     IntentType.CONTEXT_SCAN, True, False),
    (re.compile(r"\b(analyze|check|scan)\b.*\b(instruction|prompt|skill)\s*(file)?\b", re.I),
     IntentType.CONTEXT_ANALYZE, True, False),
    (re.compile(r"\b(scan|check)\b.*\b(directory|folder)\b.*\b(prompt|instruction|skill)\b", re.I),
     IntentType.CONTEXT_SCAN, True, False),

    # Analysis intents
    (re.compile(r"\b(analyze|scan|check|test|examine|inspect)\b.*\b(model|gguf)\b", re.I),
     IntentType.ANALYZE_MODEL, True, False),
    (re.compile(r"\b(analyze|scan|check|test|examine|inspect)\b.*\b(mcp|server|infrastructure)\b", re.I),
     IntentType.ANALYZE_INFRASTRUCTURE, True, False),
    (re.compile(r"\b(analyze|scan|check|test|examine|inspect)\b.*\b(skill)\b", re.I),
     IntentType.ANALYZE_SKILL, True, False),
    (re.compile(r"\bis\s+.+\s+(safe|secure|risky|dangerous)\b", re.I),
     IntentType.ANALYZE_MODEL, True, True),
    # Short model name pattern: "analyze llama", "test mistral", etc.
    (re.compile(r"^(analyze|interrogate|test|scan)\s+([a-zA-Z][a-zA-Z0-9_-]*)(\s|$)", re.I),
     IntentType.ANALYZE_MODEL, True, False),

    # Semantic code analysis intents
    (re.compile(r"\b(analyze|scan|check|review)\b.*\b(code|this code|the code)\b", re.I),
     IntentType.ANALYZE_CODE, True, False),
    (re.compile(r"\b(semantic|security)\s+(analysis|scan)\b", re.I),
     IntentType.ANALYZE_CODE, True, False),
    (re.compile(r"\b(analyze|scan|check)\b.*\b(file|source)\b", re.I),
     IntentType.ANALYZE_CODE, True, False),
    (re.compile(r"\bfind\b.*\b(vulnerabilities|issues|bugs|security)\b.*\b(in|this)\b", re.I),
     IntentType.ANALYZE_CODE, True, False),
    (re.compile(r"\b(is this|is the)\s+(code|file)\s+(safe|secure|vulnerable)\b", re.I),
     IntentType.ANALYZE_CODE, True, True),

    # Behavior analysis intents
    (re.compile(r"\b(analyze|check|assess|evaluate)\b.*\b(behavior|behavioral|safety|responses)\b", re.I),
     IntentType.ANALYZE_BEHAVIOR, True, False),
    (re.compile(r"\b(behavior|capability|capabilities)\s+(analysis|assessment|test)\b", re.I),
     IntentType.ANALYZE_BEHAVIOR, True, False),
    (re.compile(r"\b(detect|find)\b.*\b(inconsistenc|drift|anomal)\b", re.I),
     IntentType.ANALYZE_BEHAVIOR, True, False),

    # Comparison intents
    (re.compile(r"\b(compare|versus|vs\.?|difference between)\b", re.I),
     IntentType.COMPARE, True, True),

    # Explanation intents
    (re.compile(r"\b(explain|why|how come|what does|tell me about)\b.*\b(finding|result|score|risk)\b", re.I),
     IntentType.EXPLAIN, False, True),
    (re.compile(r"\bwhat\s+(is|are|does)\b", re.I),
     IntentType.QUERY_KNOWLEDGE, False, True),

    # Report intents
    (re.compile(r"\b(generate|create|make)\b.*\b(report|card|summary)\b", re.I),
     IntentType.GENERATE_REPORT, False, True),
    (re.compile(r"\b(open|launch|start|view)\b.*\b(reports?|viewer)\b", re.I),
     IntentType.VIEW_REPORTS, False, False),
    (re.compile(r"\breports?\s+view(er)?\b", re.I),
     IntentType.VIEW_REPORTS, False, False),
    (re.compile(r"\bview(er)?\s+reports?\b", re.I),
     IntentType.VIEW_REPORTS, False, False),
    (re.compile(r"\b(show|see)\s+(my\s+)?reports?\b", re.I),
     IntentType.VIEW_REPORTS, False, False),
    # Simple "report" or "reports" should open viewer
    (re.compile(r"^reports?$", re.I),
     IntentType.VIEW_REPORTS, False, False),
    (re.compile(r"^open\s+reports?$", re.I),
     IntentType.VIEW_REPORTS, False, False),
    (re.compile(r"^reports?\s*viewer$", re.I),
     IntentType.VIEW_REPORTS, False, False),
    # Common misspellings of "report"
    (re.compile(r"^(reprot|repotr|repport|rpeort|reoprt|repost|repor)s?$", re.I),
     IntentType.VIEW_REPORTS, False, False),
    (re.compile(r"\b(list|show|get)\s*(reports?|analyses|scans)\b", re.I),
     IntentType.LIST_REPORTS, False, False),
    (re.compile(r"\b(list|show|get)\b.*\b(report|analysis|scan)\b", re.I),
     IntentType.LIST_REPORTS, False, False),
    (re.compile(r"\b(recent|latest|last)\b.*\b(report|analysis|scan)s?\b", re.I),
     IntentType.LIST_REPORTS, False, False),

    # Documentation/instructions intents
    (re.compile(r"\b(open|show|view)\b.*\b(docs?|documentation|instructions?|guide|manual)\b", re.I),
     IntentType.VIEW_DOCS, False, False),
    (re.compile(r"\bdocs?\b$", re.I),
     IntentType.VIEW_DOCS, False, False),
    (re.compile(r"^(documentation|instructions?|guide|manual)$", re.I),
     IntentType.VIEW_DOCS, False, False),
    (re.compile(r"\b(read|see)\s+(the\s+)?(docs?|documentation|instructions?|guide)\b", re.I),
     IntentType.VIEW_DOCS, False, False),
    (re.compile(r"\bhow\s+do\s+i\s+use\s+(this|benderbox)\b", re.I),
     IntentType.VIEW_DOCS, False, False),
    # Misspellings
    (re.compile(r"^(documenation|documention|instrcutions|instrucions|insturctions)$", re.I),
     IntentType.VIEW_DOCS, False, False),

    # Knowledge queries
    (re.compile(r"\b(what|how|why|when|where|who)\b.*\b(jailbreak|vulnerability|attack|exploit)\b", re.I),
     IntentType.QUERY_KNOWLEDGE, False, True),
    (re.compile(r"\btell me about\b", re.I),
     IntentType.QUERY_KNOWLEDGE, False, True),

    # Model listing intents
    (re.compile(r"\b(list|show|what)\b.*\bmodels?\b", re.I),
     IntentType.LIST_MODELS, False, False),
    (re.compile(r"\bmodels?\s+(list|available|downloaded)\b", re.I),
     IntentType.LIST_MODELS, False, False),
    (re.compile(r"\bwhat\b.*\b(analysis|nlp)\s+models?\b", re.I),
     IntentType.LIST_MODELS, False, False),
    (re.compile(r"\bavailable\s+models?\b", re.I),
     IntentType.LIST_MODELS, False, False),
    (re.compile(r"\bmodels?\s+for\s+(analysis|nlp)\b", re.I),
     IntentType.LIST_MODELS, False, False),

    # Model download intents
    (re.compile(r"\bdownload\b.*\b(model|tinyllama|phi|llama|qwen|mistral)\b", re.I),
     IntentType.DOWNLOAD_MODEL, False, False),
    (re.compile(r"\b(get|fetch|install)\b.*\bmodel\b", re.I),
     IntentType.DOWNLOAD_MODEL, False, False),
    (re.compile(r"\bmodels?\s+download\b", re.I),
     IntentType.DOWNLOAD_MODEL, False, False),

        # HuggingFace URL patterns
    (re.compile(r"huggingface\.co/", re.I),
     IntentType.DOWNLOAD_MODEL, False, False),
    (re.compile(r"hf\.co/", re.I),
     IntentType.DOWNLOAD_MODEL, False, False),
    (re.compile(r"download.*from.*(huggingface|hf)", re.I),
     IntentType.DOWNLOAD_MODEL, False, False),
    (re.compile(r"(get|fetch|pull).*from.*(huggingface|hf)", re.I),
     IntentType.DOWNLOAD_MODEL, False, False),

    # Model load intents
    (re.compile(r"\bload\b.*\b(model|for\s+(analysis|nlp))\b", re.I),
     IntentType.LOAD_MODEL, False, False),
    (re.compile(r"\b(use|switch\s+to|select)\b.*\bmodel\b.*\b(for|as)\b", re.I),
     IntentType.LOAD_MODEL, False, False),
    (re.compile(r"\bset\b.*\b(analysis|nlp)\s+model\b", re.I),
     IntentType.LOAD_MODEL, False, False),

    # Custom interrogation tests intents
    (re.compile(r"\b(load|use|import)\b.*\b(custom|my)\s*(tests?|interrogation|prompts?)\b", re.I),
     IntentType.LOAD_CUSTOM_TESTS, False, False),
    (re.compile(r"\b(custom|my)\s*(tests?|interrogation|prompts?)\b.*\b(from|file|\.md|\.yaml)\b", re.I),
     IntentType.LOAD_CUSTOM_TESTS, False, False),
    (re.compile(r"\binterrogat\w*\.md\b", re.I),
     IntentType.LOAD_CUSTOM_TESTS, False, False),
    (re.compile(r"\b(run|execute)\b.*\b(tests?|prompts?)\b.*\b(from|in)\b.*\b(file|\.md)\b", re.I),
     IntentType.LOAD_CUSTOM_TESTS, False, False),

    # Status intents
    (re.compile(r"\b(status|state|info|health)\b", re.I),
     IntentType.GET_STATUS, False, False),

    # Export intents
    (re.compile(r"\bexport\b.*\b(report|result|analysis|finding)\b", re.I),
     IntentType.EXPORT_RESULTS, False, False),
    (re.compile(r"\b(save|convert)\b.*\b(report|result)\b.*\b(to|as)\b", re.I),
     IntentType.EXPORT_RESULTS, False, False),
    (re.compile(r"\b(html|json|csv|sarif|markdown)\s+export\b", re.I),
     IntentType.EXPORT_RESULTS, False, False),

    # Config intents
    (re.compile(r"\b(config|configure|settings?|api\s*key)\b", re.I),
     IntentType.CONFIG_MANAGE, False, False),
    (re.compile(r"\bset\b.*\b(key|config|api)\b", re.I),
     IntentType.CONFIG_MANAGE, False, False),

    # Search intents
    (re.compile(r"\bsearch\b.*\b(report|finding|vulnerability|issue|jailbreak|risk)\b", re.I),
     IntentType.SEMANTIC_SEARCH, False, False),
    (re.compile(r"\bsearch\s+(for|reports?|findings?)\b", re.I),
     IntentType.SEMANTIC_SEARCH, False, False),
    (re.compile(r"\bfind\b.*\b(vulnerability|issue|risk|report|jailbreak)\b", re.I),
     IntentType.SEMANTIC_SEARCH, False, False),

    # Version intents
    (re.compile(r"\b(version|ver)\b", re.I),
     IntentType.SHOW_VERSION, False, False),
    (re.compile(r"\bwhat\s+(version|release)\b", re.I),
     IntentType.SHOW_VERSION, False, False),

    # Prerequisites intents
    (re.compile(r"\b(prerequisites?|dependencies?|requirements?)\b", re.I),
     IntentType.CHECK_PREREQUISITES, False, False),
    (re.compile(r"\bcheck\b.*\b(install|setup|environment|system)\b", re.I),
     IntentType.CHECK_PREREQUISITES, False, False),

    # Help intents
    (re.compile(r"\b(help|usage|how to use|commands|capabilities)\b", re.I),
     IntentType.HELP, False, False),
]

# Parameter extraction patterns
PARAMETER_PATTERNS = {
    "model_path": re.compile(r"['\"]?([^\s'\"]+\.gguf)['\"]?", re.I),
    "model_name": re.compile(r"\b(analyze|test|scan|check)\s+(?:model\s+)?([a-zA-Z0-9_\-]+)\b", re.I),
    "file_path": re.compile(r"['\"]?([^\s'\"]+\.(py|js|ts|yaml|yml|json|c|cpp|h|go|rs|rb|java|php))['\"]?", re.I),
    "code_file": re.compile(r"['\"]?([^\s'\"]+\.(py|js|ts|c|cpp|h|go|rs|rb|java|php))['\"]?", re.I),
    # Profiles: model analysis + infrastructure profiles
    "profile": re.compile(r"\b(quick|standard|full|deep|adversarial|infra-quick|infra-standard|infra-deep)\b", re.I),
    "severity": re.compile(r"\b(critical|high|medium|low|info)\b", re.I),
    "target_type": re.compile(r"\b(model|mcp|server|skill|infrastructure|code|file)\b", re.I),
    "language": re.compile(r"\b(python|javascript|typescript|java|go|rust|ruby|php|c\+\+|c)\b", re.I),
    "model_purpose": re.compile(r"\bfor\s+(analysis|nlp)\b", re.I),
}


class IntentRouter:
    """
    Routes user queries to appropriate handlers by classifying intent.

    Uses a two-stage approach:
    1. Fast path: Keyword pattern matching for common queries
    2. Slow path: LLM classification for ambiguous queries
    """

    def __init__(self, llm_engine=None):
        """
        Initialize IntentRouter.

        Args:
            llm_engine: Optional LocalLLMEngine for LLM-based classification.
        """
        self._llm_engine = llm_engine

    def _set_llm_engine(self, llm_engine) -> None:
        """Set the LLM engine (for lazy initialization)."""
        self._llm_engine = llm_engine

    async def classify(self, query: str) -> Intent:
        """
        Classify user query into an intent.

        Args:
            query: User's natural language query.

        Returns:
            Intent object with classification and parameters.
        """
        query = query.strip()

        if not query:
            return Intent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                raw_query=query,
            )

        # Fast path: keyword matching
        intent = self._keyword_match(query)
        if intent and intent.is_high_confidence:
            logger.debug(f"Fast path match: {intent.intent_type.value} ({intent.confidence:.2f})")
            return intent

        # Slow path: LLM classification (if available and needed)
        # Check both that we have an engine AND that it has llama-cpp available
        if self._llm_engine is not None and getattr(self._llm_engine, 'is_available', False):
            try:
                llm_intent = await self._llm_classify(query)
                if llm_intent.confidence > (intent.confidence if intent else 0):
                    logger.debug(f"LLM classification: {llm_intent.intent_type.value} ({llm_intent.confidence:.2f})")
                    return llm_intent
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")

        # Return best keyword match or unknown
        if intent:
            return intent

        return Intent(
            intent_type=IntentType.GENERAL_QUESTION,
            confidence=0.5,
            requires_llm=True,
            raw_query=query,
        )

    def _keyword_match(self, query: str) -> Optional[Intent]:
        """
        Fast keyword-based intent matching.

        Args:
            query: User query.

        Returns:
            Intent if matched, None otherwise.
        """
        best_match: Optional[Intent] = None
        best_confidence = 0.0

        for pattern, intent_type, requires_analysis, requires_llm in KEYWORD_PATTERNS:
            match = pattern.search(query)
            if match:
                # Calculate confidence based on match quality
                match_ratio = len(match.group()) / len(query)
                confidence = min(0.95, 0.7 + match_ratio * 0.3)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = Intent(
                        intent_type=intent_type,
                        confidence=confidence,
                        parameters=self._extract_parameters(query, intent_type),
                        requires_analysis=requires_analysis,
                        requires_llm=requires_llm,
                        raw_query=query,
                    )

        return best_match

    def _extract_parameters(self, query: str, intent_type: IntentType) -> Dict[str, Any]:
        """
        Extract parameters from query based on intent type.

        Args:
            query: User query.
            intent_type: Classified intent type.

        Returns:
            Dictionary of extracted parameters.
        """
        params: Dict[str, Any] = {}

        # Extract based on intent type
        if intent_type in (IntentType.ANALYZE_MODEL, IntentType.COMPARE):
            # Look for model paths first (e.g., ./model.gguf)
            model_matches = PARAMETER_PATTERNS["model_path"].findall(query)
            if model_matches:
                if len(model_matches) == 1:
                    params["target"] = model_matches[0]
                else:
                    params["targets"] = model_matches
            else:
                # Look for model name (e.g., "analyze llama")
                name_match = PARAMETER_PATTERNS["model_name"].search(query)
                if name_match:
                    params["model_name"] = name_match.group(2)

        if intent_type == IntentType.LIST_MODELS:
            # Extract purpose filter (analysis or nlp)
            purpose_match = PARAMETER_PATTERNS["model_purpose"].search(query)
            if purpose_match:
                params["purpose"] = purpose_match.group(1).lower()
            else:
                params["purpose"] = "all"

        if intent_type in (IntentType.ANALYZE_INFRASTRUCTURE, IntentType.ANALYZE_SKILL):
            # Look for file paths
            file_matches = PARAMETER_PATTERNS["file_path"].findall(query)
            if file_matches:
                params["target"] = file_matches[0][0] if file_matches else None

        if intent_type == IntentType.ANALYZE_CODE:
            # Look for code files
            code_matches = PARAMETER_PATTERNS["code_file"].findall(query)
            if code_matches:
                params["target"] = code_matches[0][0] if code_matches else None
            # Check for language hint
            lang_match = PARAMETER_PATTERNS["language"].search(query)
            if lang_match:
                params["language"] = lang_match.group(1).lower()
            # Default depth for semantic analysis
            if "profile" not in params:
                params["profile"] = "standard"

        # Profile extraction
        profile_match = PARAMETER_PATTERNS["profile"].search(query)
        if profile_match:
            params["profile"] = profile_match.group(1).lower()

        # Severity extraction
        severity_match = PARAMETER_PATTERNS["severity"].search(query)
        if severity_match:
            params["severity"] = severity_match.group(1).lower()

        # Target type extraction
        target_match = PARAMETER_PATTERNS["target_type"].search(query)
        if target_match:
            params["target_type"] = target_match.group(1).lower()

        return params

    async def _llm_classify(self, query: str) -> Intent:
        """
        Use LLM for intent classification.

        Args:
            query: User query.

        Returns:
            Intent from LLM classification.
        """
        if self._llm_engine is None:
            raise RuntimeError("LLM engine not available")

        prompt = self._build_classification_prompt(query)
        response = await self._llm_engine.generate(
            prompt=prompt,
            model_type="analysis",
            max_tokens=100,
            temperature=0.1,  # Low temperature for consistent classification
        )

        return self._parse_llm_response(response, query)

    def _build_classification_prompt(self, query: str) -> str:
        """Build prompt for LLM classification."""
        intent_descriptions = """
ANALYZE_MODEL: User wants to analyze/scan/test an AI model for safety
ANALYZE_INFRASTRUCTURE: User wants to analyze MCP servers or infrastructure
ANALYZE_SKILL: User wants to analyze a skill definition
ANALYZE_CODE: User wants semantic analysis of source code for security issues
ANALYZE_BEHAVIOR: User wants behavioral analysis, capability assessment, or drift detection
COMPARE: User wants to compare multiple models or reports
EXPLAIN: User wants explanation of a finding or result
QUERY_KNOWLEDGE: User wants information about security topics
GENERATE_REPORT: User wants to generate a report or summary
LIST_REPORTS: User wants to see existing reports
GET_STATUS: User wants system status or model info
HELP: User needs help using the system
GENERAL_QUESTION: General question not fitting other categories
"""

        return f"""Classify the following user query into exactly one intent category.

Intent categories:
{intent_descriptions}

User query: "{query}"

Respond with ONLY the intent category name (e.g., ANALYZE_MODEL) and nothing else."""

    def _parse_llm_response(self, response: str, query: str) -> Intent:
        """Parse LLM response into Intent object."""
        response = response.strip().upper()

        # Try to match intent type
        for intent_type in IntentType:
            if intent_type.name in response:
                return Intent(
                    intent_type=intent_type,
                    confidence=0.85,  # LLM classifications get moderate-high confidence
                    parameters=self._extract_parameters(query, intent_type),
                    requires_analysis=intent_type in (
                        IntentType.ANALYZE_MODEL,
                        IntentType.ANALYZE_INFRASTRUCTURE,
                        IntentType.ANALYZE_SKILL,
                        IntentType.ANALYZE_CODE,
                        IntentType.ANALYZE_BEHAVIOR,
                        IntentType.COMPARE,
                    ),
                    requires_llm=intent_type in (
                        IntentType.EXPLAIN,
                        IntentType.QUERY_KNOWLEDGE,
                        IntentType.GENERATE_REPORT,
                        IntentType.COMPARE,
                        IntentType.GENERAL_QUESTION,
                        IntentType.ANALYZE_CODE,  # May use LLM for semantic analysis
                        IntentType.ANALYZE_BEHAVIOR,  # May use LLM for behavior analysis
                    ),
                    raw_query=query,
                )

        # Fallback
        return Intent(
            intent_type=IntentType.GENERAL_QUESTION,
            confidence=0.5,
            requires_llm=True,
            raw_query=query,
        )

    def get_intent_description(self, intent_type: IntentType) -> str:
        """Get human-readable description of an intent type."""
        descriptions = {
            IntentType.ANALYZE_MODEL: "Analyze an AI model for safety and security",
            IntentType.ANALYZE_INFRASTRUCTURE: "Analyze MCP server or infrastructure",
            IntentType.ANALYZE_SKILL: "Analyze a skill definition",
            IntentType.ANALYZE_CODE: "Perform semantic security analysis on code",
            IntentType.ANALYZE_BEHAVIOR: "Perform behavioral analysis or capability assessment",
            IntentType.MCP_TOOLS: "List available tools from an MCP server",
            IntentType.MCP_INTERROGATE: "Run security tests against an MCP server",
            IntentType.MCP_CALL: "Call a specific tool on an MCP server",
            IntentType.CONTEXT_ANALYZE: "Analyze instruction file for security risks",
            IntentType.CONTEXT_SCAN: "Scan directory for risky instruction files",
            IntentType.LIST_MODELS: "List available models for analysis or NLP",
            IntentType.COMPARE: "Compare multiple models or reports",
            IntentType.EXPLAIN: "Explain a finding or result",
            IntentType.QUERY_KNOWLEDGE: "Query security knowledge base",
            IntentType.GENERATE_REPORT: "Generate a report or summary",
            IntentType.LIST_REPORTS: "List existing reports",
            IntentType.GET_STATUS: "Get system status",
            IntentType.HELP: "Get help using BenderBox",
            IntentType.GENERAL_QUESTION: "Answer a general question",
            IntentType.UNKNOWN: "Unknown intent",
        }
        return descriptions.get(intent_type, "Unknown")
