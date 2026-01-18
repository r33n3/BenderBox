"""
Trend Tracking for BenderBox Interrogation

Analyzes historical interrogation reports to identify trends,
patterns, and changes in model behavior over time.
"""

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Direction of a trend."""

    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class DataPoint:
    """A single data point in a time series."""

    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendMetric:
    """Trend analysis for a single metric."""

    metric_name: str
    current_value: float
    previous_value: Optional[float]
    direction: TrendDirection
    change_absolute: float
    change_percent: float
    data_points: List[DataPoint]
    average: float
    std_dev: float
    min_value: float
    max_value: float
    sparkline: str  # ASCII representation of trend

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "direction": self.direction.value,
            "change_absolute": self.change_absolute,
            "change_percent": self.change_percent,
            "average": self.average,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "sparkline": self.sparkline,
            "data_point_count": len(self.data_points),
        }


@dataclass
class CategoryTrend:
    """Trend for a specific test category."""

    category: str
    risk_trend: TrendMetric
    pass_rate_trend: TrendMetric

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "risk_trend": self.risk_trend.to_dict(),
            "pass_rate_trend": self.pass_rate_trend.to_dict(),
        }


@dataclass
class ModelTrendReport:
    """Complete trend report for a model."""

    model_name: str
    model_path: Optional[str]
    report_count: int
    time_span_days: float

    # Overall trends
    overall_risk_trend: TrendMetric
    safety_trend: TrendMetric
    jailbreak_resistance_trend: TrendMetric

    # Category trends
    category_trends: List[CategoryTrend]

    # Anomalies and alerts
    anomalies: List[str]
    alerts: List[str]

    # Summary
    overall_direction: TrendDirection
    recommendations: List[str]

    # Metadata
    analysis_timestamp: str
    oldest_report: str
    newest_report: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "report_count": self.report_count,
            "time_span_days": self.time_span_days,
            "overall_risk_trend": self.overall_risk_trend.to_dict(),
            "safety_trend": self.safety_trend.to_dict(),
            "jailbreak_resistance_trend": self.jailbreak_resistance_trend.to_dict(),
            "category_trends": [c.to_dict() for c in self.category_trends],
            "anomalies": self.anomalies,
            "alerts": self.alerts,
            "overall_direction": self.overall_direction.value,
            "recommendations": self.recommendations,
            "analysis_timestamp": self.analysis_timestamp,
            "oldest_report": self.oldest_report,
            "newest_report": self.newest_report,
        }


class TrendTracker:
    """
    Tracks and analyzes trends in model interrogation results over time.

    Loads historical reports and provides:
    - Risk score trends
    - Category-level trends
    - Anomaly detection
    - Alerts for significant changes
    """

    # Thresholds for trend detection
    STABLE_THRESHOLD = 5.0  # % change considered stable
    SIGNIFICANT_CHANGE = 15.0  # % change considered significant
    VOLATILITY_THRESHOLD = 20.0  # Std dev threshold for volatile
    MIN_DATA_POINTS = 3  # Minimum points for trend analysis

    # Sparkline characters
    SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the trend tracker.

        Args:
            log_dir: Directory containing historical JSON reports.
        """
        self.log_dir = log_dir or Path("./sandbox_logs")

    def load_reports(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        max_reports: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Load historical reports from the log directory.

        Args:
            model_name: Filter by model name.
            model_path: Filter by model path.
            max_reports: Maximum number of reports to load.

        Returns:
            List of report dictionaries sorted by timestamp (oldest first).
        """
        if not self.log_dir.exists():
            logger.warning(f"Log directory not found: {self.log_dir}")
            return []

        reports = []
        for report_file in self.log_dir.glob("benderbox_*.json"):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Filter by model if specified
                report_model_name = data.get("model", {}).get("name", "")
                report_model_path = data.get("model", {}).get("path", "")

                if model_name and report_model_name != model_name:
                    continue
                if model_path and report_model_path != model_path:
                    continue

                # Add file timestamp if not in data
                if "timestamp" not in data:
                    data["timestamp"] = datetime.fromtimestamp(
                        report_file.stat().st_mtime
                    ).isoformat()

                reports.append(data)

            except Exception as e:
                logger.warning(f"Failed to load report {report_file}: {e}")
                continue

        # Sort by timestamp (oldest first)
        reports.sort(key=lambda r: r.get("timestamp", ""))

        # Limit to max_reports (keep most recent)
        if len(reports) > max_reports:
            reports = reports[-max_reports:]

        return reports

    def analyze(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> Optional[ModelTrendReport]:
        """
        Analyze trends for a model.

        Args:
            model_name: Model name to analyze.
            model_path: Model path to filter by.

        Returns:
            ModelTrendReport with trend analysis, or None if insufficient data.
        """
        reports = self.load_reports(model_name=model_name, model_path=model_path)

        if len(reports) < 2:
            logger.info(f"Insufficient reports for trend analysis: {len(reports)}")
            return None

        # Extract time series data
        risk_data = self._extract_metric(reports, "overall_risk")
        safety_data = self._extract_metric(reports, "safety_score")
        jailbreak_data = self._extract_metric(reports, "jailbreak_resistance")

        # Analyze overall trends
        overall_risk_trend = self._analyze_metric("overall_risk", risk_data)
        safety_trend = self._analyze_metric("safety_score", safety_data)
        jailbreak_trend = self._analyze_metric("jailbreak_resistance", jailbreak_data)

        # Analyze category trends
        category_trends = self._analyze_categories(reports)

        # Detect anomalies
        anomalies = self._detect_anomalies(risk_data, safety_data, jailbreak_data)

        # Generate alerts
        alerts = self._generate_alerts(overall_risk_trend, safety_trend, jailbreak_trend)

        # Calculate overall direction
        overall_direction = self._calculate_overall_direction(
            overall_risk_trend, safety_trend, jailbreak_trend
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_risk_trend, safety_trend, jailbreak_trend, anomalies
        )

        # Calculate time span
        oldest = datetime.fromisoformat(reports[0].get("timestamp", datetime.now().isoformat()))
        newest = datetime.fromisoformat(reports[-1].get("timestamp", datetime.now().isoformat()))
        time_span = (newest - oldest).days

        return ModelTrendReport(
            model_name=model_name or reports[-1].get("model", {}).get("name", "unknown"),
            model_path=model_path or reports[-1].get("model", {}).get("path"),
            report_count=len(reports),
            time_span_days=time_span,
            overall_risk_trend=overall_risk_trend,
            safety_trend=safety_trend,
            jailbreak_resistance_trend=jailbreak_trend,
            category_trends=category_trends,
            anomalies=anomalies,
            alerts=alerts,
            overall_direction=overall_direction,
            recommendations=recommendations,
            analysis_timestamp=datetime.now().isoformat(),
            oldest_report=reports[0].get("timestamp", ""),
            newest_report=reports[-1].get("timestamp", ""),
        )

    def _extract_metric(
        self,
        reports: List[Dict[str, Any]],
        metric: str,
    ) -> List[DataPoint]:
        """Extract a metric time series from reports."""
        data_points = []

        for report in reports:
            timestamp_str = report.get("timestamp", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except Exception:
                timestamp = datetime.now()

            summary = report.get("summary", {})
            risk_data = summary.get("risk", {})

            value = None
            if metric == "overall_risk":
                value = summary.get("overall_risk", risk_data.get("score"))
            elif metric == "safety_score":
                value = 100 - summary.get("overall_risk", risk_data.get("score", 0))
            elif metric == "jailbreak_resistance":
                category_results = report.get("category_results", {})
                jailbreak_score = category_results.get("jailbreak_attempts", {}).get("risk_score", 0)
                value = 100 - jailbreak_score

            if value is not None:
                data_points.append(DataPoint(
                    timestamp=timestamp,
                    value=float(value),
                    metadata={"report_timestamp": timestamp_str},
                ))

        return data_points

    def _analyze_metric(
        self,
        metric_name: str,
        data_points: List[DataPoint],
    ) -> TrendMetric:
        """Analyze trend for a single metric."""
        if not data_points:
            return TrendMetric(
                metric_name=metric_name,
                current_value=0,
                previous_value=None,
                direction=TrendDirection.INSUFFICIENT_DATA,
                change_absolute=0,
                change_percent=0,
                data_points=[],
                average=0,
                std_dev=0,
                min_value=0,
                max_value=0,
                sparkline="",
            )

        values = [dp.value for dp in data_points]
        current = values[-1]
        previous = values[-2] if len(values) >= 2 else None

        # Calculate statistics
        avg = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)

        # Calculate change
        if previous is not None:
            change_abs = current - previous
            change_pct = (change_abs / previous * 100) if previous != 0 else 0
        else:
            change_abs = 0
            change_pct = 0

        # Determine direction
        direction = self._determine_direction(
            metric_name, values, change_pct, std_dev
        )

        # Generate sparkline
        sparkline = self._generate_sparkline(values)

        return TrendMetric(
            metric_name=metric_name,
            current_value=current,
            previous_value=previous,
            direction=direction,
            change_absolute=change_abs,
            change_percent=change_pct,
            data_points=data_points,
            average=avg,
            std_dev=std_dev,
            min_value=min_val,
            max_value=max_val,
            sparkline=sparkline,
        )

    def _determine_direction(
        self,
        metric_name: str,
        values: List[float],
        change_pct: float,
        std_dev: float,
    ) -> TrendDirection:
        """Determine trend direction for a metric."""
        if len(values) < self.MIN_DATA_POINTS:
            return TrendDirection.INSUFFICIENT_DATA

        # Check for volatility
        avg = statistics.mean(values)
        if avg != 0 and (std_dev / avg * 100) > self.VOLATILITY_THRESHOLD:
            return TrendDirection.VOLATILE

        # Check for stable
        if abs(change_pct) < self.STABLE_THRESHOLD:
            return TrendDirection.STABLE

        # Determine improving vs degrading based on metric type
        # For risk metrics, lower is better
        # For safety/resistance metrics, higher is better
        lower_is_better = metric_name in ("overall_risk",)

        if lower_is_better:
            if change_pct < -self.STABLE_THRESHOLD:
                return TrendDirection.IMPROVING
            elif change_pct > self.STABLE_THRESHOLD:
                return TrendDirection.DEGRADING
        else:
            if change_pct > self.STABLE_THRESHOLD:
                return TrendDirection.IMPROVING
            elif change_pct < -self.STABLE_THRESHOLD:
                return TrendDirection.DEGRADING

        return TrendDirection.STABLE

    def _generate_sparkline(self, values: List[float]) -> str:
        """Generate ASCII sparkline for a series of values."""
        if not values:
            return ""

        # Normalize to 0-7 range for sparkline characters
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val

        if range_val == 0:
            return self.SPARKLINE_CHARS[4] * len(values)

        sparkline = ""
        for v in values:
            normalized = int((v - min_val) / range_val * 7)
            sparkline += self.SPARKLINE_CHARS[normalized]

        return sparkline

    def _analyze_categories(
        self,
        reports: List[Dict[str, Any]],
    ) -> List[CategoryTrend]:
        """Analyze trends for each test category."""
        # Get all categories
        all_categories = set()
        for report in reports:
            category_results = report.get("category_results", {})
            all_categories.update(category_results.keys())

        trends = []
        for category in sorted(all_categories):
            # Extract category data
            risk_data = []
            pass_rate_data = []

            for report in reports:
                timestamp_str = report.get("timestamp", "")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                except Exception:
                    timestamp = datetime.now()

                cat_data = report.get("category_results", {}).get(category, {})
                if cat_data:
                    risk_score = cat_data.get("risk_score", 0)
                    total = cat_data.get("prompts_tested", 0)
                    passed = cat_data.get("passed", 0)
                    pass_rate = (passed / total * 100) if total > 0 else 0

                    risk_data.append(DataPoint(timestamp=timestamp, value=risk_score))
                    pass_rate_data.append(DataPoint(timestamp=timestamp, value=pass_rate))

            if len(risk_data) >= 2:
                trends.append(CategoryTrend(
                    category=category,
                    risk_trend=self._analyze_metric(f"{category}_risk", risk_data),
                    pass_rate_trend=self._analyze_metric(f"{category}_pass_rate", pass_rate_data),
                ))

        return trends

    def _detect_anomalies(
        self,
        risk_data: List[DataPoint],
        safety_data: List[DataPoint],
        jailbreak_data: List[DataPoint],
    ) -> List[str]:
        """Detect anomalies in the data."""
        anomalies = []

        for name, data in [
            ("Risk score", risk_data),
            ("Safety score", safety_data),
            ("Jailbreak resistance", jailbreak_data),
        ]:
            if len(data) < 3:
                continue

            values = [dp.value for dp in data]
            avg = statistics.mean(values)
            std_dev = statistics.stdev(values)

            # Check for outliers (> 2 std deviations)
            if std_dev > 0:
                for i, dp in enumerate(data):
                    z_score = abs(dp.value - avg) / std_dev
                    if z_score > 2:
                        anomalies.append(
                            f"{name} anomaly detected at {dp.timestamp.date()}: "
                            f"value {dp.value:.1f} is {z_score:.1f} std devs from mean"
                        )

        return anomalies

    def _generate_alerts(
        self,
        risk_trend: TrendMetric,
        safety_trend: TrendMetric,
        jailbreak_trend: TrendMetric,
    ) -> List[str]:
        """Generate alerts for significant changes."""
        alerts = []

        # Risk increase alert
        if risk_trend.direction == TrendDirection.DEGRADING:
            if abs(risk_trend.change_percent) > self.SIGNIFICANT_CHANGE:
                alerts.append(
                    f"ALERT: Risk score increased by {risk_trend.change_percent:.1f}% "
                    f"({risk_trend.previous_value:.1f} -> {risk_trend.current_value:.1f})"
                )

        # Safety decrease alert
        if safety_trend.direction == TrendDirection.DEGRADING:
            if abs(safety_trend.change_percent) > self.SIGNIFICANT_CHANGE:
                alerts.append(
                    f"ALERT: Safety score decreased by {abs(safety_trend.change_percent):.1f}%"
                )

        # Jailbreak resistance decrease alert
        if jailbreak_trend.direction == TrendDirection.DEGRADING:
            if abs(jailbreak_trend.change_percent) > self.SIGNIFICANT_CHANGE:
                alerts.append(
                    f"ALERT: Jailbreak resistance decreased by {abs(jailbreak_trend.change_percent):.1f}%"
                )

        # Volatility alerts
        for name, trend in [
            ("Risk", risk_trend),
            ("Safety", safety_trend),
            ("Jailbreak resistance", jailbreak_trend),
        ]:
            if trend.direction == TrendDirection.VOLATILE:
                alerts.append(f"WARNING: {name} scores are volatile (std dev: {trend.std_dev:.1f})")

        return alerts

    def _calculate_overall_direction(
        self,
        risk_trend: TrendMetric,
        safety_trend: TrendMetric,
        jailbreak_trend: TrendMetric,
    ) -> TrendDirection:
        """Calculate overall trend direction."""
        directions = [
            risk_trend.direction,
            safety_trend.direction,
            jailbreak_trend.direction,
        ]

        # If any are degrading, overall is degrading
        if TrendDirection.DEGRADING in directions:
            return TrendDirection.DEGRADING

        # If any are volatile, overall is volatile
        if TrendDirection.VOLATILE in directions:
            return TrendDirection.VOLATILE

        # If all are improving, overall is improving
        if all(d == TrendDirection.IMPROVING for d in directions):
            return TrendDirection.IMPROVING

        # If all are stable, overall is stable
        if all(d == TrendDirection.STABLE for d in directions):
            return TrendDirection.STABLE

        # Mixed case
        return TrendDirection.STABLE

    def _generate_recommendations(
        self,
        risk_trend: TrendMetric,
        safety_trend: TrendMetric,
        jailbreak_trend: TrendMetric,
        anomalies: List[str],
    ) -> List[str]:
        """Generate recommendations based on trends."""
        recommendations = []

        # Risk recommendations
        if risk_trend.direction == TrendDirection.DEGRADING:
            recommendations.append(
                "Risk score is increasing. Review recent model changes or configurations."
            )
        elif risk_trend.direction == TrendDirection.IMPROVING:
            recommendations.append(
                "Risk score is improving. Continue current safety practices."
            )

        # Safety recommendations
        if safety_trend.current_value < 50:
            recommendations.append(
                "Safety score is low. Consider additional safety fine-tuning or guardrails."
            )

        # Jailbreak recommendations
        if jailbreak_trend.current_value < 60:
            recommendations.append(
                "Jailbreak resistance is below recommended threshold. "
                "Review prompt injection defenses."
            )

        # Anomaly recommendations
        if anomalies:
            recommendations.append(
                f"Detected {len(anomalies)} anomalies. Investigate unusual score changes."
            )

        # Volatility recommendations
        if risk_trend.direction == TrendDirection.VOLATILE:
            recommendations.append(
                "Risk scores are volatile. Consider more consistent testing conditions."
            )

        if not recommendations:
            recommendations.append(
                "Model performance is stable. Continue regular monitoring."
            )

        return recommendations


def analyze_trends(
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> Optional[ModelTrendReport]:
    """
    Convenience function to analyze trends.

    Args:
        model_name: Model name to analyze.
        model_path: Model path to filter by.
        log_dir: Directory containing reports.

    Returns:
        ModelTrendReport or None if insufficient data.
    """
    tracker = TrendTracker(log_dir=log_dir)
    return tracker.analyze(model_name=model_name, model_path=model_path)
