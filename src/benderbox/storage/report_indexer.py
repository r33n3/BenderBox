"""
Report Indexer for BenderBox

Indexes analysis reports into the vector store for semantic search.
Enables natural language queries across report history.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from benderbox.storage.vector_store import Document, SearchResult, VectorStore
from benderbox.storage.report_db import Finding, ReportFilters, ReportSummary

logger = logging.getLogger(__name__)


# Collection names for different document types
COLLECTION_REPORTS = "reports"
COLLECTION_FINDINGS = "report_findings"


@dataclass
class IndexedReport:
    """Metadata about an indexed report."""

    report_id: str
    target_name: str
    target_type: str
    risk_level: str
    risk_score: float
    finding_count: int
    indexed_at: datetime
    document_ids: List[str] = field(default_factory=list)


@dataclass
class SimilarFinding:
    """A finding similar to a query finding."""

    finding: Finding
    similarity_score: float
    report_id: str
    target_name: str


class ReportIndexer:
    """
    Indexes analysis reports for semantic search.

    Creates searchable documents from:
    - Report summaries (target, risk, profile)
    - Individual findings (test results, vulnerabilities)
    - Aggregated categories and patterns

    Supports:
    - Natural language search across reports
    - Finding similarity search
    - Metadata filtering (date, severity, target)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize ReportIndexer.

        Args:
            vector_store: VectorStore instance for indexing.
            chunk_size: Maximum characters per document chunk.
            chunk_overlap: Overlap between chunks for context.
        """
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._indexed_reports: Dict[str, IndexedReport] = {}

    async def index_report(
        self,
        report: Dict[str, Any],
        report_id: Optional[str] = None,
    ) -> IndexedReport:
        """
        Index a report into the vector store.

        Creates multiple documents:
        1. Report summary document
        2. Individual finding documents
        3. Category aggregation documents

        Args:
            report: Report dictionary (SandboxRunResult format).
            report_id: Optional report ID (generated if not provided).

        Returns:
            IndexedReport with indexing metadata.
        """
        # Extract report metadata
        report_id = report_id or report.get("run_id") or self._generate_id(report)
        target_name = report.get("target_name", "unknown")
        target_type = report.get("target_type", "unknown")
        profile = report.get("profile", "standard")
        timestamp = report.get("timestamp", datetime.now().isoformat())

        summary = report.get("summary", {})
        risk = summary.get("risk", {})
        risk_level = risk.get("level", "unknown")
        risk_score = risk.get("score", 0.0)

        results = report.get("results", [])
        document_ids = []

        # 1. Index report summary
        summary_doc = self._create_summary_document(
            report_id=report_id,
            target_name=target_name,
            target_type=target_type,
            profile=profile,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk.get("factors", []),
            timestamp=timestamp,
            finding_count=len(results),
        )
        await self._vector_store.add_document(summary_doc, COLLECTION_REPORTS)
        document_ids.append(summary_doc.id)

        # 2. Index individual findings
        for idx, result in enumerate(results):
            finding_doc = self._create_finding_document(
                report_id=report_id,
                target_name=target_name,
                finding_idx=idx,
                result=result,
                timestamp=timestamp,
            )
            await self._vector_store.add_document(finding_doc, COLLECTION_FINDINGS)
            document_ids.append(finding_doc.id)

        # 3. Index category aggregations (group findings by category)
        categories = self._aggregate_by_category(results)
        for category, findings in categories.items():
            if findings:
                cat_doc = self._create_category_document(
                    report_id=report_id,
                    target_name=target_name,
                    category=category,
                    findings=findings,
                    timestamp=timestamp,
                )
                await self._vector_store.add_document(cat_doc, COLLECTION_FINDINGS)
                document_ids.append(cat_doc.id)

        # Track indexed report
        indexed = IndexedReport(
            report_id=report_id,
            target_name=target_name,
            target_type=target_type,
            risk_level=risk_level,
            risk_score=risk_score,
            finding_count=len(results),
            indexed_at=datetime.now(),
            document_ids=document_ids,
        )
        self._indexed_reports[report_id] = indexed

        logger.info(
            f"Indexed report {report_id}: {len(document_ids)} documents "
            f"({target_name}, {risk_level}, {len(results)} findings)"
        )

        return indexed

    async def search_reports(
        self,
        query: str,
        filters: Optional[ReportFilters] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search reports using natural language query.

        Args:
            query: Natural language search query.
            filters: Optional filters for metadata.
            top_k: Maximum results to return.

        Returns:
            List of SearchResult with matching documents.
        """
        # Build metadata filter from ReportFilters
        metadata_filter = self._build_metadata_filter(filters) if filters else None

        results = await self._vector_store.search(
            query=query,
            collection=COLLECTION_REPORTS,
            top_k=top_k,
            filter_metadata=metadata_filter,
        )

        return results

    async def search_findings(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
    ) -> List[SearchResult]:
        """
        Search findings using natural language query.

        Args:
            query: Natural language search query.
            filters: Optional metadata filters.
            top_k: Maximum results to return.

        Returns:
            List of SearchResult with matching finding documents.
        """
        results = await self._vector_store.search(
            query=query,
            collection=COLLECTION_FINDINGS,
            top_k=top_k,
            filter_metadata=filters,
        )

        return results

    async def find_similar_findings(
        self,
        finding: Finding,
        top_k: int = 10,
        exclude_same_report: bool = True,
    ) -> List[SimilarFinding]:
        """
        Find findings similar to a given finding.

        Args:
            finding: Finding to find similar matches for.
            top_k: Maximum results to return.
            exclude_same_report: Exclude findings from the same report.

        Returns:
            List of SimilarFinding with similarity scores.
        """
        # Create query from finding content
        query = self._finding_to_query(finding)

        # Search with extra results if excluding same report
        search_k = top_k * 2 if exclude_same_report else top_k

        results = await self._vector_store.search(
            query=query,
            collection=COLLECTION_FINDINGS,
            top_k=search_k,
        )

        similar_findings = []
        for result in results:
            metadata = result.document.metadata

            # Skip same report if requested
            if exclude_same_report and metadata.get("report_id") == finding.report_id:
                continue

            # Reconstruct Finding from metadata
            similar = SimilarFinding(
                finding=Finding(
                    id=metadata.get("finding_id", result.document.id),
                    report_id=metadata.get("report_id", ""),
                    test_name=metadata.get("test_name", ""),
                    category=metadata.get("category", ""),
                    status=metadata.get("status", ""),
                    severity=metadata.get("severity", ""),
                    details={"message": result.document.content},
                ),
                similarity_score=result.score,
                report_id=metadata.get("report_id", ""),
                target_name=metadata.get("target_name", ""),
            )
            similar_findings.append(similar)

            if len(similar_findings) >= top_k:
                break

        return similar_findings

    async def delete_report_index(self, report_id: str) -> int:
        """
        Remove all indexed documents for a report.

        Args:
            report_id: Report ID to remove.

        Returns:
            Number of documents deleted.
        """
        deleted = 0

        # Get tracked document IDs
        indexed = self._indexed_reports.get(report_id)
        if indexed:
            for doc_id in indexed.document_ids:
                # Determine collection from doc_id prefix
                collection = (
                    COLLECTION_REPORTS
                    if doc_id.startswith("report_")
                    else COLLECTION_FINDINGS
                )
                if await self._vector_store.delete_document(doc_id, collection):
                    deleted += 1

            del self._indexed_reports[report_id]
        else:
            # Fall back to searching by report_id metadata
            for collection in [COLLECTION_REPORTS, COLLECTION_FINDINGS]:
                try:
                    # Search for documents with this report_id
                    results = await self._vector_store.search(
                        query="",
                        collection=collection,
                        top_k=1000,
                        filter_metadata={"report_id": report_id},
                    )
                    for result in results:
                        if await self._vector_store.delete_document(
                            result.document.id, collection
                        ):
                            deleted += 1
                except Exception as e:
                    logger.warning(f"Error deleting from {collection}: {e}")

        logger.info(f"Deleted {deleted} documents for report {report_id}")
        return deleted

    async def reindex_all(
        self,
        reports: List[Dict[str, Any]],
        clear_existing: bool = True,
    ) -> int:
        """
        Reindex all reports.

        Args:
            reports: List of report dictionaries.
            clear_existing: Clear existing indexes first.

        Returns:
            Number of reports indexed.
        """
        if clear_existing:
            await self._vector_store.clear_collection(COLLECTION_REPORTS)
            await self._vector_store.clear_collection(COLLECTION_FINDINGS)
            self._indexed_reports.clear()

        indexed_count = 0
        for report in reports:
            try:
                await self.index_report(report)
                indexed_count += 1
            except Exception as e:
                logger.error(f"Failed to index report: {e}")

        logger.info(f"Reindexed {indexed_count} reports")
        return indexed_count

    def get_indexed_reports(self) -> List[IndexedReport]:
        """Get list of currently indexed reports."""
        return list(self._indexed_reports.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        return {
            "indexed_reports": len(self._indexed_reports),
            "total_documents": sum(
                len(r.document_ids) for r in self._indexed_reports.values()
            ),
            "collections": [COLLECTION_REPORTS, COLLECTION_FINDINGS],
        }

    # --- Private Methods ---

    def _generate_id(self, report: Dict[str, Any]) -> str:
        """Generate a deterministic ID for a report."""
        content = f"{report.get('target_name', '')}{report.get('timestamp', '')}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _create_summary_document(
        self,
        report_id: str,
        target_name: str,
        target_type: str,
        profile: str,
        risk_level: str,
        risk_score: float,
        risk_factors: List[str],
        timestamp: str,
        finding_count: int,
    ) -> Document:
        """Create a searchable document for report summary."""
        # Build rich text content for semantic search
        factors_text = ", ".join(risk_factors) if risk_factors else "none identified"

        content = (
            f"Security analysis report for {target_name} ({target_type}). "
            f"Profile: {profile}. "
            f"Risk level: {risk_level} with score {risk_score}/100. "
            f"Risk factors: {factors_text}. "
            f"Total findings: {finding_count}."
        )

        return Document.create(
            content=content,
            metadata={
                "report_id": report_id,
                "target_name": target_name,
                "target_type": target_type,
                "profile": profile,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "timestamp": timestamp,
                "finding_count": finding_count,
                "doc_type": "summary",
            },
            doc_id=f"report_{report_id}",
        )

    def _create_finding_document(
        self,
        report_id: str,
        target_name: str,
        finding_idx: int,
        result: Dict[str, Any],
        timestamp: str,
    ) -> Document:
        """Create a searchable document for a finding."""
        test_name = result.get("test_name", "unknown")
        category = result.get("category", "unknown")
        status = result.get("status", "unknown")
        severity = result.get("severity", "info")
        details = result.get("details", {})
        message = details.get("message", "")

        # Build rich text content
        content = (
            f"Finding: {test_name} in category {category}. "
            f"Status: {status}. Severity: {severity}. "
            f"{message}"
        )

        # Truncate if too long
        if len(content) > self._chunk_size:
            content = content[: self._chunk_size - 3] + "..."

        finding_id = f"finding_{report_id}_{finding_idx}"

        return Document.create(
            content=content,
            metadata={
                "report_id": report_id,
                "target_name": target_name,
                "finding_id": finding_id,
                "test_name": test_name,
                "category": category,
                "status": status,
                "severity": severity,
                "timestamp": timestamp,
                "doc_type": "finding",
            },
            doc_id=finding_id,
        )

    def _create_category_document(
        self,
        report_id: str,
        target_name: str,
        category: str,
        findings: List[Dict[str, Any]],
        timestamp: str,
    ) -> Document:
        """Create a searchable document for category aggregation."""
        # Aggregate findings by status
        passed = sum(1 for f in findings if f.get("status") == "passed")
        failed = sum(1 for f in findings if f.get("status") == "failed")
        warnings = sum(1 for f in findings if f.get("status") == "warning")

        # Get severities
        severities = [f.get("severity", "info") for f in findings]
        max_severity = self._get_max_severity(severities)

        # Build content
        test_names = ", ".join(f.get("test_name", "unknown") for f in findings[:5])
        if len(findings) > 5:
            test_names += f" and {len(findings) - 5} more"

        content = (
            f"Category {category} analysis for {target_name}. "
            f"Tests: {test_names}. "
            f"Results: {passed} passed, {failed} failed, {warnings} warnings. "
            f"Maximum severity: {max_severity}."
        )

        return Document.create(
            content=content,
            metadata={
                "report_id": report_id,
                "target_name": target_name,
                "category": category,
                "passed_count": passed,
                "failed_count": failed,
                "warning_count": warnings,
                "max_severity": max_severity,
                "timestamp": timestamp,
                "doc_type": "category",
            },
            doc_id=f"category_{report_id}_{category}",
        )

    def _aggregate_by_category(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group findings by category."""
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for result in results:
            category = result.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        return categories

    def _get_max_severity(self, severities: List[str]) -> str:
        """Get the highest severity from a list."""
        severity_order = ["critical", "high", "medium", "low", "info"]
        for sev in severity_order:
            if sev in severities:
                return sev
        return "info"

    def _finding_to_query(self, finding: Finding) -> str:
        """Convert a Finding to a search query."""
        return (
            f"{finding.test_name} {finding.category} "
            f"{finding.severity} {finding.details.get('message', '')}"
        )

    def _build_metadata_filter(
        self, filters: ReportFilters
    ) -> Optional[Dict[str, Any]]:
        """Build metadata filter dict from ReportFilters."""
        metadata: Dict[str, Any] = {}

        if filters.target_name:
            metadata["target_name"] = filters.target_name
        if filters.target_type:
            metadata["target_type"] = filters.target_type
        if filters.profile:
            metadata["profile"] = filters.profile
        if filters.risk_level:
            metadata["risk_level"] = filters.risk_level

        # Note: ChromaDB has limited filter support
        # Date range and score range filters would need post-filtering
        # or custom implementation

        return metadata if metadata else None


async def create_report_indexer(
    vector_store: Optional[VectorStore] = None,
) -> ReportIndexer:
    """
    Factory function to create a ReportIndexer.

    Args:
        vector_store: Optional VectorStore (creates new if not provided).

    Returns:
        Configured ReportIndexer instance.
    """
    if vector_store is None:
        vector_store = VectorStore()
        await vector_store.initialize()

    return ReportIndexer(vector_store=vector_store)
