"""
SQLite Report Database for BenderBox

Provides persistent storage for analysis reports with:
- Async operations using aiosqlite
- Automatic schema migration
- Rich querying capabilities
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from benderbox.config import get_config

if TYPE_CHECKING:
    from benderbox.storage.report_indexer import ReportIndexer

logger = logging.getLogger(__name__)

# Current schema version
SCHEMA_VERSION = 1


@dataclass
class Finding:
    """Individual finding from an analysis."""

    id: str
    report_id: str
    test_name: str
    category: str
    status: str  # passed, failed, warning, error
    severity: str  # critical, high, medium, low, info
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class ReportSummary:
    """Summary view of a report."""

    id: str
    run_id: str
    timestamp: datetime
    profile: str
    target_type: str
    target_name: str
    risk_level: str
    risk_score: float
    finding_count: int
    passed_count: int
    failed_count: int


@dataclass
class ReportFilters:
    """Filters for report queries."""

    target_name: Optional[str] = None
    target_type: Optional[str] = None
    profile: Optional[str] = None
    risk_level: Optional[str] = None
    min_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = 100
    offset: int = 0


class ReportDatabase:
    """
    SQLite database for storing analysis reports.

    Uses aiosqlite for async operations.
    Optionally integrates with ReportIndexer for semantic search.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        indexer: Optional["ReportIndexer"] = None,
    ):
        """
        Initialize ReportDatabase.

        Args:
            db_path: Path to SQLite database file. Uses config default if not provided.
            indexer: Optional ReportIndexer for vector store indexing.
        """
        config = get_config().storage
        self.db_path = db_path or config.db_path
        self._initialized = False
        self._indexer = indexer

    def set_indexer(self, indexer: "ReportIndexer") -> None:
        """Set the report indexer for semantic search integration."""
        self._indexer = indexer

    async def _get_connection(self):
        """Get aiosqlite connection."""
        try:
            import aiosqlite
        except ImportError:
            raise ReportDatabaseError(
                "aiosqlite is not installed. Install with:\n"
                "  pip install aiosqlite"
            )

        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        return await aiosqlite.connect(self.db_path)

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with await self._get_connection() as db:
            # Check current schema version
            await db.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                )
            """)

            cursor = await db.execute("SELECT version FROM schema_version")
            row = await cursor.fetchone()
            current_version = row[0] if row else 0

            # Apply migrations
            if current_version < SCHEMA_VERSION:
                await self._migrate(db, current_version)

            await db.commit()

        self._initialized = True
        logger.info(f"Report database initialized at: {self.db_path}")

    async def _migrate(self, db, from_version: int) -> None:
        """Apply schema migrations."""
        if from_version < 1:
            # Initial schema
            await db.executescript("""
                -- Reports table
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Findings table
                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY,
                    report_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT,
                    FOREIGN KEY (report_id) REFERENCES reports(id) ON DELETE CASCADE
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_reports_target_name ON reports(target_name);
                CREATE INDEX IF NOT EXISTS idx_reports_timestamp ON reports(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_reports_risk_level ON reports(risk_level);
                CREATE INDEX IF NOT EXISTS idx_reports_profile ON reports(profile);
                CREATE INDEX IF NOT EXISTS idx_findings_report_id ON findings(report_id);
                CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
                CREATE INDEX IF NOT EXISTS idx_findings_category ON findings(category);
                CREATE INDEX IF NOT EXISTS idx_findings_status ON findings(status);

                -- Update schema version
                DELETE FROM schema_version;
                INSERT INTO schema_version (version) VALUES (1);
            """)

            logger.info("Applied database migration to version 1")

    async def save_report(self, report: Dict[str, Any]) -> str:
        """
        Save an analysis report.

        Args:
            report: Report data (SandboxRunResult format).

        Returns:
            Report ID.
        """
        await self.initialize()

        import uuid

        report_id = str(uuid.uuid4())
        run_id = report.get("run_id", report_id)
        timestamp = report.get("timestamp", datetime.now().isoformat())

        # Extract summary info
        summary = report.get("summary", {})
        risk = summary.get("risk", {})
        profile = report.get("profile", "unknown")
        target_type = report.get("target_type", "unknown")
        target_name = report.get("target_name", "unknown")
        risk_level = risk.get("level", "unknown")
        risk_score = risk.get("score", 0.0)

        async with await self._get_connection() as db:
            # Insert report
            await db.execute(
                """
                INSERT INTO reports (id, run_id, timestamp, profile, target_type,
                                   target_name, risk_level, risk_score, report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    run_id,
                    timestamp,
                    profile,
                    target_type,
                    target_name,
                    risk_level,
                    risk_score,
                    json.dumps(report),
                ),
            )

            # Insert findings
            results = report.get("results", [])
            for result in results:
                finding_id = str(uuid.uuid4())
                await db.execute(
                    """
                    INSERT INTO findings (id, report_id, test_name, category,
                                        status, severity, details, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        finding_id,
                        report_id,
                        result.get("test_name", "unknown"),
                        result.get("category", "unknown"),
                        result.get("status", "unknown"),
                        result.get("severity", "info"),
                        json.dumps(result.get("details", {})),
                        result.get("timestamp"),
                    ),
                )

            await db.commit()

        logger.debug(f"Saved report {report_id} with {len(results)} findings")

        # Index report for semantic search if indexer is configured
        if self._indexer:
            try:
                await self._indexer.index_report(report, report_id=report_id)
                logger.debug(f"Indexed report {report_id} for semantic search")
            except Exception as e:
                # Don't fail the save if indexing fails
                logger.warning(f"Failed to index report {report_id}: {e}")

        return report_id

    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a report by ID.

        Args:
            report_id: Report ID.

        Returns:
            Report data or None if not found.
        """
        await self.initialize()

        async with await self._get_connection() as db:
            cursor = await db.execute(
                "SELECT report_json FROM reports WHERE id = ?",
                (report_id,),
            )
            row = await cursor.fetchone()

            if row:
                return json.loads(row[0])
            return None

    async def get_latest_report(self, target_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent report for a target.

        Args:
            target_name: Target name.

        Returns:
            Report data or None if not found.
        """
        await self.initialize()

        async with await self._get_connection() as db:
            cursor = await db.execute(
                """
                SELECT report_json FROM reports
                WHERE target_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (target_name,),
            )
            row = await cursor.fetchone()

            if row:
                return json.loads(row[0])
            return None

    async def list_reports(
        self, filters: Optional[ReportFilters] = None
    ) -> List[ReportSummary]:
        """
        List reports with optional filters.

        Args:
            filters: Optional filters.

        Returns:
            List of report summaries.
        """
        await self.initialize()

        if filters is None:
            filters = ReportFilters()

        # Build query
        query = """
            SELECT r.id, r.run_id, r.timestamp, r.profile, r.target_type,
                   r.target_name, r.risk_level, r.risk_score,
                   COUNT(f.id) as finding_count,
                   SUM(CASE WHEN f.status = 'passed' THEN 1 ELSE 0 END) as passed_count,
                   SUM(CASE WHEN f.status = 'failed' THEN 1 ELSE 0 END) as failed_count
            FROM reports r
            LEFT JOIN findings f ON r.id = f.report_id
            WHERE 1=1
        """
        params: List[Any] = []

        if filters.target_name:
            query += " AND r.target_name = ?"
            params.append(filters.target_name)

        if filters.target_type:
            query += " AND r.target_type = ?"
            params.append(filters.target_type)

        if filters.profile:
            query += " AND r.profile = ?"
            params.append(filters.profile)

        if filters.risk_level:
            query += " AND r.risk_level = ?"
            params.append(filters.risk_level)

        if filters.min_risk_score is not None:
            query += " AND r.risk_score >= ?"
            params.append(filters.min_risk_score)

        if filters.max_risk_score is not None:
            query += " AND r.risk_score <= ?"
            params.append(filters.max_risk_score)

        if filters.start_date:
            query += " AND r.timestamp >= ?"
            params.append(filters.start_date.isoformat())

        if filters.end_date:
            query += " AND r.timestamp <= ?"
            params.append(filters.end_date.isoformat())

        query += " GROUP BY r.id ORDER BY r.timestamp DESC LIMIT ? OFFSET ?"
        params.extend([filters.limit, filters.offset])

        async with await self._get_connection() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            return [
                ReportSummary(
                    id=row[0],
                    run_id=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    profile=row[3],
                    target_type=row[4],
                    target_name=row[5],
                    risk_level=row[6],
                    risk_score=row[7],
                    finding_count=row[8] or 0,
                    passed_count=row[9] or 0,
                    failed_count=row[10] or 0,
                )
                for row in rows
            ]

    async def search_findings(
        self,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Finding]:
        """
        Search findings across all reports.

        Args:
            severity: Filter by severity level.
            category: Filter by category.
            status: Filter by status.
            limit: Maximum results.

        Returns:
            List of findings.
        """
        await self.initialize()

        query = "SELECT id, report_id, test_name, category, status, severity, details, timestamp FROM findings WHERE 1=1"
        params: List[Any] = []

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        if category:
            query += " AND category = ?"
            params.append(category)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with await self._get_connection() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            return [
                Finding(
                    id=row[0],
                    report_id=row[1],
                    test_name=row[2],
                    category=row[3],
                    status=row[4],
                    severity=row[5],
                    details=json.loads(row[6]),
                    timestamp=datetime.fromisoformat(row[7]) if row[7] else None,
                )
                for row in rows
            ]

    async def get_report_history(
        self, target_name: str, limit: int = 10
    ) -> List[ReportSummary]:
        """
        Get report history for a target.

        Args:
            target_name: Target name.
            limit: Maximum reports.

        Returns:
            List of report summaries, most recent first.
        """
        filters = ReportFilters(target_name=target_name, limit=limit)
        return await self.list_reports(filters)

    async def delete_report(self, report_id: str) -> bool:
        """
        Delete a report and its findings.

        Args:
            report_id: Report ID.

        Returns:
            True if deleted, False if not found.
        """
        await self.initialize()

        async with await self._get_connection() as db:
            # Check if exists
            cursor = await db.execute(
                "SELECT id FROM reports WHERE id = ?", (report_id,)
            )
            if not await cursor.fetchone():
                return False

            # Delete findings first (foreign key)
            await db.execute("DELETE FROM findings WHERE report_id = ?", (report_id,))
            await db.execute("DELETE FROM reports WHERE id = ?", (report_id,))
            await db.commit()

        # Remove from vector store index if indexer is configured
        if self._indexer:
            try:
                await self._indexer.delete_report_index(report_id)
            except Exception as e:
                logger.warning(f"Failed to remove report {report_id} from index: {e}")

        logger.debug(f"Deleted report {report_id}")
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with statistics.
        """
        await self.initialize()

        async with await self._get_connection() as db:
            # Total reports
            cursor = await db.execute("SELECT COUNT(*) FROM reports")
            total_reports = (await cursor.fetchone())[0]

            # Total findings
            cursor = await db.execute("SELECT COUNT(*) FROM findings")
            total_findings = (await cursor.fetchone())[0]

            # Risk level distribution
            cursor = await db.execute(
                "SELECT risk_level, COUNT(*) FROM reports GROUP BY risk_level"
            )
            risk_distribution = dict(await cursor.fetchall())

            # Recent reports
            cursor = await db.execute(
                """
                SELECT timestamp FROM reports
                ORDER BY timestamp DESC LIMIT 1
                """
            )
            row = await cursor.fetchone()
            latest_report = row[0] if row else None

            return {
                "total_reports": total_reports,
                "total_findings": total_findings,
                "risk_distribution": risk_distribution,
                "latest_report": latest_report,
                "db_path": self.db_path,
                "schema_version": SCHEMA_VERSION,
            }

    async def export_report_to_file(
        self,
        report_id: str,
        output_path: Optional[str] = None,
        format: str = "json",
    ) -> Optional[str]:
        """
        Export a report to a file.

        Args:
            report_id: Report ID to export.
            output_path: Output file path. If None, uses reports_path from config.
            format: Output format - 'json' or 'markdown'.

        Returns:
            Path to exported file, or None if report not found.
        """
        report = await self.get_report(report_id)
        if not report:
            return None

        # Determine output path
        if output_path is None:
            config = get_config().storage
            reports_dir = Path(config.reports_path)
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Organize by target name and date
            target_name = report.get("target_name", "unknown").replace("/", "_").replace("\\", "_")
            timestamp = report.get("timestamp", datetime.now().isoformat())[:10]  # YYYY-MM-DD
            ext = "md" if format == "markdown" else "json"
            output_path = str(reports_dir / f"{target_name}_{timestamp}_{report_id[:8]}.{ext}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            content = self._report_to_markdown(report)
            output_file.write_text(content, encoding="utf-8")
        else:
            output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

        logger.info(f"Exported report {report_id} to {output_path}")
        return str(output_file)

    def _report_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert a report to markdown format."""
        summary = report.get("summary", {})
        risk = summary.get("risk", {})
        results = report.get("results", [])

        lines = [
            f"# Security Analysis Report",
            f"",
            f"**Target:** {report.get('target_name', 'Unknown')}",
            f"**Type:** {report.get('target_type', 'Unknown')}",
            f"**Date:** {report.get('timestamp', 'Unknown')}",
            f"**Profile:** {report.get('profile', 'Unknown')}",
            f"",
            f"## Risk Assessment",
            f"",
            f"- **Risk Level:** {risk.get('level', 'Unknown').upper()}",
            f"- **Risk Score:** {risk.get('score', 0)}/100",
            f"",
            f"## Findings Summary",
            f"",
        ]

        # Count by severity
        critical = sum(1 for r in results if r.get("severity") == "critical")
        high = sum(1 for r in results if r.get("severity") == "high")
        medium = sum(1 for r in results if r.get("severity") == "medium")
        low = sum(1 for r in results if r.get("severity") == "low")
        info = sum(1 for r in results if r.get("severity") == "info")

        lines.extend([
            f"| Severity | Count |",
            f"|----------|-------|",
            f"| Critical | {critical} |",
            f"| High     | {high} |",
            f"| Medium   | {medium} |",
            f"| Low      | {low} |",
            f"| Info     | {info} |",
            f"",
            f"## Detailed Findings",
            f"",
        ])

        # Group findings by severity
        for severity in ["critical", "high", "medium", "low", "info"]:
            findings = [r for r in results if r.get("severity") == severity]
            if findings:
                lines.append(f"### {severity.upper()}")
                lines.append("")
                for finding in findings:
                    status_icon = "FAIL" if finding.get("status") == "failed" else "PASS"
                    lines.append(f"- **[{status_icon}]** {finding.get('test_name', 'Unknown')}")
                    details = finding.get("details", {})
                    if details.get("message"):
                        lines.append(f"  - {details['message'][:200]}")
                lines.append("")

        lines.extend([
            f"---",
            f"*Generated by BenderBox v3.0*",
        ])

        return "\n".join(lines)

    async def cleanup_old_reports(
        self,
        days_to_keep: int = 30,
        keep_min: int = 10,
    ) -> int:
        """
        Clean up old reports, keeping the most recent ones.

        Args:
            days_to_keep: Delete reports older than this many days.
            keep_min: Minimum number of reports to keep per target (regardless of age).

        Returns:
            Number of reports deleted.
        """
        await self.initialize()

        from datetime import timedelta
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()

        async with await self._get_connection() as db:
            # Get reports to potentially delete (older than cutoff)
            cursor = await db.execute(
                """
                SELECT id, target_name FROM reports
                WHERE timestamp < ?
                ORDER BY timestamp ASC
                """,
                (cutoff_date,),
            )
            old_reports = await cursor.fetchall()

            # Track reports to delete (respecting keep_min per target)
            target_counts: Dict[str, int] = {}
            to_delete: List[str] = []

            # First, count total reports per target
            cursor = await db.execute(
                "SELECT target_name, COUNT(*) FROM reports GROUP BY target_name"
            )
            for row in await cursor.fetchall():
                target_counts[row[0]] = row[1]

            # Determine which old reports can be deleted
            deleted_per_target: Dict[str, int] = {}
            for report_id, target_name in old_reports:
                current_count = target_counts.get(target_name, 0) - deleted_per_target.get(target_name, 0)
                if current_count > keep_min:
                    to_delete.append(report_id)
                    deleted_per_target[target_name] = deleted_per_target.get(target_name, 0) + 1

            # Delete the reports
            for report_id in to_delete:
                await db.execute("DELETE FROM findings WHERE report_id = ?", (report_id,))
                await db.execute("DELETE FROM reports WHERE id = ?", (report_id,))

            await db.commit()

        logger.info(f"Cleaned up {len(to_delete)} old reports")
        return len(to_delete)

    async def get_targets(self) -> List[Dict[str, Any]]:
        """
        Get a list of all analyzed targets with their report counts.

        Returns:
            List of dictionaries with target info.
        """
        await self.initialize()

        async with await self._get_connection() as db:
            cursor = await db.execute(
                """
                SELECT target_name, target_type,
                       COUNT(*) as report_count,
                       MAX(timestamp) as latest_report,
                       MIN(risk_level) as best_risk,
                       MAX(risk_score) as worst_score
                FROM reports
                GROUP BY target_name
                ORDER BY latest_report DESC
                """
            )
            rows = await cursor.fetchall()

            return [
                {
                    "name": row[0],
                    "type": row[1],
                    "report_count": row[2],
                    "latest_report": row[3],
                    "best_risk_level": row[4],
                    "worst_risk_score": row[5],
                }
                for row in rows
            ]


class ReportDatabaseError(Exception):
    """Exception for report database errors."""

    pass
