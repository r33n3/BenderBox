"""
Export Manager for BenderBox

Manages export operations for analysis results, reports, and model cards.
Supports multiple formats and batch operations.
"""

import json
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from benderbox.reporting.report_generator import ReportGenerator
from benderbox.reporting.model_card import ModelCardGenerator, ModelCard

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    SARIF = "sarif"  # Static Analysis Results Interchange Format


@dataclass
class ExportResult:
    """Result of an export operation."""

    success: bool
    format: ExportFormat
    path: str
    size_bytes: int
    error: Optional[str] = None


@dataclass
class BatchExportResult:
    """Result of a batch export operation."""

    total: int
    successful: int
    failed: int
    results: List[ExportResult]
    archive_path: Optional[str] = None


class ExportManager:
    """
    Manages export operations for BenderBox outputs.

    Supports:
    - Single and batch exports
    - Multiple formats (Markdown, JSON, HTML, CSV, SARIF)
    - Archive creation (ZIP)
    - Model card generation and export
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize ExportManager.

        Args:
            output_dir: Default output directory for exports.
        """
        self.output_dir = Path(output_dir or "./exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._report_generator = ReportGenerator()
        self._model_card_generator = ModelCardGenerator()

    async def export_analysis(
        self,
        analysis_result: Dict[str, Any],
        format: ExportFormat = ExportFormat.MARKDOWN,
        output_path: Optional[str] = None,
        include_model_card: bool = False,
    ) -> ExportResult:
        """
        Export a single analysis result.

        Args:
            analysis_result: Analysis result dictionary.
            format: Export format.
            output_path: Optional output path.
            include_model_card: Include model card in export.

        Returns:
            ExportResult with status and path.
        """
        try:
            # Generate filename if not provided
            if output_path is None:
                target = analysis_result.get("target_name", "analysis")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in target)
                ext = self._get_extension(format)
                output_path = str(self.output_dir / f"{clean_name}_{timestamp}{ext}")

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Generate content based on format
            content = await self._generate_content(analysis_result, format)

            # Write to file
            path.write_text(content, encoding="utf-8")
            size = path.stat().st_size

            logger.info(f"Exported {format.value} to: {path}")

            # Also export model card if requested
            if include_model_card:
                card = self._model_card_generator.generate_from_analysis(analysis_result)
                card_path = path.with_stem(f"{path.stem}_model_card")
                await self._model_card_generator.save_model_card(card, str(card_path), "markdown")

            return ExportResult(
                success=True,
                format=format,
                path=str(path),
                size_bytes=size,
            )

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                format=format,
                path=output_path or "",
                size_bytes=0,
                error=str(e),
            )

    async def export_all_formats(
        self,
        analysis_result: Dict[str, Any],
        output_dir: Optional[str] = None,
        formats: Optional[List[ExportFormat]] = None,
    ) -> BatchExportResult:
        """
        Export analysis in multiple formats.

        Args:
            analysis_result: Analysis result dictionary.
            output_dir: Output directory.
            formats: List of formats (defaults to all).

        Returns:
            BatchExportResult with all export results.
        """
        if formats is None:
            formats = [ExportFormat.MARKDOWN, ExportFormat.JSON, ExportFormat.HTML]

        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        target = analysis_result.get("target_name", "analysis")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in target)

        results = []
        for fmt in formats:
            ext = self._get_extension(fmt)
            path = target_dir / f"{clean_name}_{timestamp}{ext}"
            result = await self.export_analysis(analysis_result, fmt, str(path))
            results.append(result)

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        return BatchExportResult(
            total=len(results),
            successful=successful,
            failed=failed,
            results=results,
        )

    async def batch_export(
        self,
        analyses: List[Dict[str, Any]],
        format: ExportFormat = ExportFormat.JSON,
        output_dir: Optional[str] = None,
        create_archive: bool = False,
    ) -> BatchExportResult:
        """
        Export multiple analyses.

        Args:
            analyses: List of analysis results.
            format: Export format for all.
            output_dir: Output directory.
            create_archive: Create ZIP archive of exports.

        Returns:
            BatchExportResult with all results.
        """
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, analysis in enumerate(analyses):
            target = analysis.get("target_name", f"analysis_{i}")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in target)
            ext = self._get_extension(format)
            path = target_dir / f"{clean_name}_{timestamp}{ext}"

            result = await self.export_analysis(analysis, format, str(path))
            results.append(result)

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        archive_path = None
        if create_archive and successful > 0:
            archive_path = await self._create_archive(
                [r.path for r in results if r.success],
                target_dir,
            )

        return BatchExportResult(
            total=len(results),
            successful=successful,
            failed=failed,
            results=results,
            archive_path=archive_path,
        )

    async def export_model_card(
        self,
        analysis_result: Dict[str, Any],
        output_path: Optional[str] = None,
        format: str = "markdown",
    ) -> ExportResult:
        """
        Export model card for an analysis.

        Args:
            analysis_result: Analysis result dictionary.
            output_path: Optional output path.
            format: Output format (markdown, json).

        Returns:
            ExportResult with status.
        """
        try:
            card = self._model_card_generator.generate_from_analysis(analysis_result)

            if output_path is None:
                target = analysis_result.get("target_name", "model")
                clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in target)
                ext = ".md" if format == "markdown" else ".json"
                output_path = str(self.output_dir / f"{clean_name}_model_card{ext}")

            path = await self._model_card_generator.save_model_card(card, output_path, format)
            size = Path(path).stat().st_size

            return ExportResult(
                success=True,
                format=ExportFormat.MARKDOWN if format == "markdown" else ExportFormat.JSON,
                path=path,
                size_bytes=size,
            )

        except Exception as e:
            logger.error(f"Model card export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.MARKDOWN,
                path=output_path or "",
                size_bytes=0,
                error=str(e),
            )

    async def export_sarif(
        self,
        analysis_result: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> ExportResult:
        """
        Export analysis in SARIF format for tool integration.

        SARIF (Static Analysis Results Interchange Format) is a standard
        format for static analysis tool output.

        Args:
            analysis_result: Analysis result dictionary.
            output_path: Optional output path.

        Returns:
            ExportResult with status.
        """
        try:
            target = analysis_result.get("target_name", "analysis")
            target_path = analysis_result.get("target_path", target)

            if output_path is None:
                clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in target)
                output_path = str(self.output_dir / f"{clean_name}.sarif")

            sarif = {
                "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {
                            "driver": {
                                "name": "BenderBox",
                                "version": "3.0.0-alpha",
                                "informationUri": "https://github.com/benderbox",
                                "rules": self._generate_sarif_rules(analysis_result),
                            }
                        },
                        "results": self._generate_sarif_results(analysis_result, target_path),
                        "invocations": [
                            {
                                "executionSuccessful": True,
                                "endTimeUtc": datetime.now().isoformat() + "Z",
                            }
                        ],
                    }
                ],
            }

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sarif, indent=2), encoding="utf-8")
            size = path.stat().st_size

            logger.info(f"Exported SARIF to: {path}")

            return ExportResult(
                success=True,
                format=ExportFormat.SARIF,
                path=str(path),
                size_bytes=size,
            )

        except Exception as e:
            logger.error(f"SARIF export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.SARIF,
                path=output_path or "",
                size_bytes=0,
                error=str(e),
            )

    async def export_csv(
        self,
        analysis_result: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> ExportResult:
        """
        Export findings as CSV for spreadsheet analysis.

        Args:
            analysis_result: Analysis result dictionary.
            output_path: Optional output path.

        Returns:
            ExportResult with status.
        """
        try:
            target = analysis_result.get("target_name", "analysis")

            if output_path is None:
                clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in target)
                output_path = str(self.output_dir / f"{clean_name}_findings.csv")

            results = analysis_result.get("results", [])

            # Build CSV content
            lines = ["Test Name,Category,Status,Severity,Details"]

            for result in results:
                test_name = result.get("test_name", "").replace(",", ";")
                category = result.get("category", "").replace(",", ";")
                status = result.get("status", "")
                severity = result.get("severity", "")

                details = result.get("details", {})
                if isinstance(details, dict):
                    details_str = details.get("message", "")[:200]
                else:
                    details_str = str(details)[:200]
                details_str = details_str.replace(",", ";").replace("\n", " ")

                lines.append(f"{test_name},{category},{status},{severity},{details_str}")

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines), encoding="utf-8")
            size = path.stat().st_size

            return ExportResult(
                success=True,
                format=ExportFormat.CSV,
                path=str(path),
                size_bytes=size,
            )

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.CSV,
                path=output_path or "",
                size_bytes=0,
                error=str(e),
            )

    async def _generate_content(
        self,
        analysis_result: Dict[str, Any],
        format: ExportFormat,
    ) -> str:
        """Generate content for the specified format."""
        if format == ExportFormat.MARKDOWN:
            return self._report_generator.generate_markdown(analysis_result)
        elif format == ExportFormat.JSON:
            return self._report_generator.generate_json(analysis_result)
        elif format == ExportFormat.HTML:
            return self._report_generator.generate_html(analysis_result)
        elif format == ExportFormat.CSV:
            # Handle CSV separately
            raise ValueError("Use export_csv for CSV format")
        elif format == ExportFormat.SARIF:
            raise ValueError("Use export_sarif for SARIF format")
        else:
            raise ValueError(f"Unknown format: {format}")

    def _get_extension(self, format: ExportFormat) -> str:
        """Get file extension for format."""
        extensions = {
            ExportFormat.MARKDOWN: ".md",
            ExportFormat.JSON: ".json",
            ExportFormat.HTML: ".html",
            ExportFormat.CSV: ".csv",
            ExportFormat.SARIF: ".sarif",
        }
        return extensions.get(format, ".txt")

    async def _create_archive(
        self,
        file_paths: List[str],
        output_dir: Path,
    ) -> str:
        """Create ZIP archive of exported files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = output_dir / f"benderbox_export_{timestamp}.zip"

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in file_paths:
                path = Path(file_path)
                if path.exists():
                    zf.write(path, path.name)

        logger.info(f"Created archive: {archive_path}")
        return str(archive_path)

    def _generate_sarif_rules(
        self,
        analysis_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate SARIF rule definitions from analysis."""
        rules = []
        seen_tests = set()

        for result in analysis_result.get("results", []):
            test_name = result.get("test_name", "unknown")
            if test_name in seen_tests:
                continue
            seen_tests.add(test_name)

            severity = result.get("severity", "info").lower()
            level = {
                "critical": "error",
                "high": "error",
                "medium": "warning",
                "low": "note",
                "info": "note",
            }.get(severity, "note")

            rules.append({
                "id": test_name,
                "name": test_name.replace("_", " ").title(),
                "shortDescription": {
                    "text": f"BenderBox {result.get('category', 'security')} test",
                },
                "defaultConfiguration": {
                    "level": level,
                },
            })

        return rules

    def _generate_sarif_results(
        self,
        analysis_result: Dict[str, Any],
        target_path: str,
    ) -> List[Dict[str, Any]]:
        """Generate SARIF results from analysis."""
        sarif_results = []

        for result in analysis_result.get("results", []):
            if result.get("status") not in ("failed", "warning"):
                continue

            severity = result.get("severity", "info").lower()
            level = {
                "critical": "error",
                "high": "error",
                "medium": "warning",
                "low": "note",
                "info": "note",
            }.get(severity, "note")

            details = result.get("details", {})
            message = details.get("message", "") if isinstance(details, dict) else str(details)

            sarif_results.append({
                "ruleId": result.get("test_name", "unknown"),
                "level": level,
                "message": {
                    "text": message[:500] if message else f"Finding from {result.get('test_name')}",
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": target_path,
                            },
                        },
                    }
                ],
            })

        return sarif_results
