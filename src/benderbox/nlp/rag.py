"""
RAG (Retrieval Augmented Generation) Pipeline for BenderBox

Augments LLM responses with retrieved context from:
- Report database (semantic search)
- Knowledge base (threat intelligence)
- Code patterns and vulnerability data
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from benderbox.storage.vector_store import SearchResult
from benderbox.storage.report_indexer import ReportIndexer
from benderbox.storage.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Context retrieved for RAG augmentation."""

    query: str
    report_results: List[SearchResult] = field(default_factory=list)
    finding_results: List[SearchResult] = field(default_factory=list)
    knowledge_results: List[Any] = field(default_factory=list)

    @property
    def has_context(self) -> bool:
        """Check if any context was retrieved."""
        return bool(
            self.report_results or self.finding_results or self.knowledge_results
        )

    @property
    def total_results(self) -> int:
        """Total number of retrieved items."""
        return (
            len(self.report_results)
            + len(self.finding_results)
            + len(self.knowledge_results)
        )


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""

    response: str
    context: RetrievedContext
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0


class RAGPipeline:
    """
    Retrieval Augmented Generation pipeline.

    Retrieves relevant context from:
    1. Report database (past analysis reports)
    2. Knowledge base (threat intelligence, vulnerabilities)

    Then augments LLM prompts with this context for better responses.
    """

    # Default context limits
    MAX_REPORT_RESULTS = 5
    MAX_FINDING_RESULTS = 10
    MAX_KNOWLEDGE_RESULTS = 5
    MAX_CONTEXT_LENGTH = 4000  # Characters

    def __init__(
        self,
        report_indexer: Optional[ReportIndexer] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
        llm_engine: Optional[Any] = None,
    ):
        """
        Initialize RAG pipeline.

        Args:
            report_indexer: ReportIndexer for report search.
            knowledge_base: KnowledgeBase for threat knowledge.
            llm_engine: LLM engine for generation.
        """
        self._report_indexer = report_indexer
        self._knowledge_base = knowledge_base
        self._llm_engine = llm_engine

    def set_report_indexer(self, indexer: ReportIndexer) -> None:
        """Set the report indexer."""
        self._report_indexer = indexer

    def set_knowledge_base(self, kb: KnowledgeBase) -> None:
        """Set the knowledge base."""
        self._knowledge_base = kb

    def set_llm_engine(self, engine: Any) -> None:
        """Set the LLM engine."""
        self._llm_engine = engine

    async def retrieve(
        self,
        query: str,
        include_reports: bool = True,
        include_findings: bool = True,
        include_knowledge: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievedContext:
        """
        Retrieve relevant context for a query.

        Args:
            query: User query to find context for.
            include_reports: Include report summaries.
            include_findings: Include individual findings.
            include_knowledge: Include knowledge base entries.
            filters: Optional metadata filters.

        Returns:
            RetrievedContext with all retrieved items.
        """
        context = RetrievedContext(query=query)

        # Retrieve from report index
        if include_reports and self._report_indexer:
            try:
                context.report_results = await self._report_indexer.search_reports(
                    query=query,
                    top_k=self.MAX_REPORT_RESULTS,
                )
            except Exception as e:
                logger.warning(f"Report search failed: {e}")

        if include_findings and self._report_indexer:
            try:
                context.finding_results = await self._report_indexer.search_findings(
                    query=query,
                    filters=filters,
                    top_k=self.MAX_FINDING_RESULTS,
                )
            except Exception as e:
                logger.warning(f"Finding search failed: {e}")

        # Retrieve from knowledge base
        if include_knowledge and self._knowledge_base:
            try:
                context.knowledge_results = await self._knowledge_base.search(
                    query=query,
                    top_k=self.MAX_KNOWLEDGE_RESULTS,
                )
            except Exception as e:
                logger.warning(f"Knowledge search failed: {e}")

        logger.debug(
            f"Retrieved {context.total_results} items for query: {query[:50]}..."
        )
        return context

    def build_context_prompt(
        self,
        context: RetrievedContext,
        max_length: int = None,
    ) -> str:
        """
        Build a context string for LLM prompt augmentation.

        Args:
            context: Retrieved context.
            max_length: Maximum context length.

        Returns:
            Formatted context string.
        """
        max_length = max_length or self.MAX_CONTEXT_LENGTH
        sections = []

        # Add report context
        if context.report_results:
            report_section = self._format_report_context(context.report_results)
            if report_section:
                sections.append(("Previous Analysis Reports", report_section))

        # Add finding context
        if context.finding_results:
            finding_section = self._format_finding_context(context.finding_results)
            if finding_section:
                sections.append(("Related Findings", finding_section))

        # Add knowledge context
        if context.knowledge_results:
            knowledge_section = self._format_knowledge_context(context.knowledge_results)
            if knowledge_section:
                sections.append(("Security Knowledge", knowledge_section))

        if not sections:
            return ""

        # Build final context with length limit
        context_parts = []
        current_length = 0

        for title, content in sections:
            section_text = f"## {title}\n{content}\n"
            if current_length + len(section_text) > max_length:
                # Truncate this section
                remaining = max_length - current_length - len(f"## {title}\n\n")
                if remaining > 100:
                    context_parts.append(f"## {title}\n{content[:remaining]}...\n")
                break
            context_parts.append(section_text)
            current_length += len(section_text)

        return "\n".join(context_parts)

    async def generate(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        include_reports: bool = True,
        include_findings: bool = True,
        include_knowledge: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """
        Generate a response with RAG augmentation.

        Args:
            query: User query.
            system_prompt: Optional system prompt override.
            include_reports: Include report context.
            include_findings: Include finding context.
            include_knowledge: Include knowledge context.
            filters: Optional metadata filters.

        Returns:
            RAGResponse with generated content and sources.
        """
        # Retrieve context
        context = await self.retrieve(
            query=query,
            include_reports=include_reports,
            include_findings=include_findings,
            include_knowledge=include_knowledge,
            filters=filters,
        )

        # Build context prompt
        context_prompt = self.build_context_prompt(context)

        # Build sources list
        sources = self._extract_sources(context)

        # Generate with LLM if available
        if self._llm_engine:
            response = await self._generate_with_llm(
                query=query,
                context_prompt=context_prompt,
                system_prompt=system_prompt,
            )
            confidence = 0.8 if context.has_context else 0.5
        else:
            # Fallback: return context summary without LLM
            response = self._generate_fallback(query, context)
            confidence = 0.6 if context.has_context else 0.3

        return RAGResponse(
            response=response,
            context=context,
            sources=sources,
            confidence=confidence,
        )

    async def search(
        self,
        query: str,
        search_type: str = "all",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant information without generation.

        Args:
            query: Search query.
            search_type: Type of search (all, reports, findings, knowledge).
            top_k: Maximum results.
            filters: Optional metadata filters.

        Returns:
            List of search results with metadata.
        """
        results = []

        if search_type in ("all", "reports") and self._report_indexer:
            report_results = await self._report_indexer.search_reports(
                query=query,
                top_k=top_k,
            )
            for r in report_results:
                results.append({
                    "type": "report",
                    "content": r.document.content,
                    "score": r.score,
                    "metadata": r.document.metadata,
                })

        if search_type in ("all", "findings") and self._report_indexer:
            finding_results = await self._report_indexer.search_findings(
                query=query,
                top_k=top_k,
                filters=filters,
            )
            for r in finding_results:
                results.append({
                    "type": "finding",
                    "content": r.document.content,
                    "score": r.score,
                    "metadata": r.document.metadata,
                })

        if search_type in ("all", "knowledge") and self._knowledge_base:
            knowledge_results = await self._knowledge_base.search(
                query=query,
                top_k=top_k,
            )
            for entry in knowledge_results:
                results.append({
                    "type": "knowledge",
                    "content": entry.description if hasattr(entry, "description") else str(entry),
                    "score": 1.0,  # Knowledge entries don't have scores
                    "metadata": {
                        "name": getattr(entry, "name", ""),
                        "category": getattr(entry, "category", ""),
                        "severity": getattr(entry, "severity", ""),
                    },
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # --- Private Methods ---

    def _format_report_context(self, results: List[SearchResult]) -> str:
        """Format report search results for context."""
        lines = []
        for result in results:
            meta = result.document.metadata
            target = meta.get("target_name", "Unknown")
            risk = meta.get("risk_level", "unknown")
            score = meta.get("risk_score", 0)
            timestamp = meta.get("timestamp", "")[:10]  # Just date

            lines.append(
                f"- {target}: {risk.upper()} risk ({score}/100) [{timestamp}]"
            )
            # Add content snippet
            content = result.document.content[:150]
            if len(result.document.content) > 150:
                content += "..."
            lines.append(f"  {content}")

        return "\n".join(lines)

    def _format_finding_context(self, results: List[SearchResult]) -> str:
        """Format finding search results for context."""
        lines = []
        for result in results:
            meta = result.document.metadata
            test_name = meta.get("test_name", "Unknown")
            severity = meta.get("severity", "info")
            status = meta.get("status", "unknown")
            target = meta.get("target_name", "")

            lines.append(
                f"- [{severity.upper()}] {test_name} ({status}) - {target}"
            )

        return "\n".join(lines)

    def _format_knowledge_context(self, entries: List[Any]) -> str:
        """Format knowledge base entries for context."""
        lines = []
        for entry in entries:
            name = getattr(entry, "name", "Unknown")
            category = getattr(entry, "category", "")
            severity = getattr(entry, "severity", "")
            description = getattr(entry, "description", "")[:200]

            lines.append(f"- {name} ({category}, {severity})")
            if description:
                lines.append(f"  {description}")

        return "\n".join(lines)

    def _extract_sources(self, context: RetrievedContext) -> List[Dict[str, Any]]:
        """Extract source citations from context."""
        sources = []

        for result in context.report_results:
            meta = result.document.metadata
            sources.append({
                "type": "report",
                "id": meta.get("report_id", ""),
                "name": meta.get("target_name", ""),
                "relevance": result.score,
            })

        for result in context.finding_results[:5]:  # Limit finding sources
            meta = result.document.metadata
            sources.append({
                "type": "finding",
                "id": meta.get("finding_id", ""),
                "name": meta.get("test_name", ""),
                "report_id": meta.get("report_id", ""),
                "relevance": result.score,
            })

        for entry in context.knowledge_results:
            sources.append({
                "type": "knowledge",
                "id": getattr(entry, "id", ""),
                "name": getattr(entry, "name", ""),
                "category": getattr(entry, "category", ""),
            })

        return sources

    async def _generate_with_llm(
        self,
        query: str,
        context_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate response using LLM with context."""
        default_system = (
            "You are BenderBox, an AI security analysis assistant. "
            "Use the provided context to give accurate, helpful responses "
            "about security analysis, vulnerabilities, and threat intelligence. "
            "If the context doesn't contain relevant information, say so."
        )

        system = system_prompt or default_system

        # Build prompt with context
        if context_prompt:
            full_prompt = f"""Based on the following context from previous analyses and security knowledge:

{context_prompt}

User question: {query}

Provide a helpful response that incorporates relevant information from the context."""
        else:
            full_prompt = f"""User question: {query}

Note: No relevant context was found in the database. Provide a general response."""

        try:
            response = await self._llm_engine.generate(
                prompt=full_prompt,
                system_prompt=system,
                model_type="analysis",
                max_tokens=1000,
                temperature=0.7,
            )
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback(query, RetrievedContext(query=query))

    def _generate_fallback(
        self,
        query: str,
        context: RetrievedContext,
    ) -> str:
        """Generate fallback response without LLM."""
        if not context.has_context:
            return (
                "I couldn't find relevant information in the database for your query. "
                "Try running an analysis first, or refine your search terms."
            )

        # Build a summary response from context
        parts = []

        if context.report_results:
            parts.append(
                f"Found {len(context.report_results)} relevant analysis reports."
            )
            # Add top result
            top = context.report_results[0]
            meta = top.document.metadata
            parts.append(
                f"Most relevant: {meta.get('target_name', 'Unknown')} "
                f"({meta.get('risk_level', 'unknown')} risk)"
            )

        if context.finding_results:
            parts.append(
                f"Found {len(context.finding_results)} related findings."
            )
            # Group by severity
            severities = {}
            for r in context.finding_results:
                sev = r.document.metadata.get("severity", "info")
                severities[sev] = severities.get(sev, 0) + 1
            severity_summary = ", ".join(
                f"{count} {sev}" for sev, count in severities.items()
            )
            parts.append(f"Severities: {severity_summary}")

        if context.knowledge_results:
            parts.append(
                f"Found {len(context.knowledge_results)} knowledge base entries."
            )

        return "\n".join(parts)


async def create_rag_pipeline(
    report_indexer: Optional[ReportIndexer] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
    llm_engine: Optional[Any] = None,
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.

    Args:
        report_indexer: Optional ReportIndexer.
        knowledge_base: Optional KnowledgeBase.
        llm_engine: Optional LLM engine.

    Returns:
        Configured RAGPipeline instance.
    """
    return RAGPipeline(
        report_indexer=report_indexer,
        knowledge_base=knowledge_base,
        llm_engine=llm_engine,
    )
