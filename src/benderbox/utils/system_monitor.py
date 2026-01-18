"""
System Monitor for BenderBox

Provides real-time system metrics monitoring during analysis operations.
Shows CPU, memory, and GPU usage updating periodically.
"""

import asyncio
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Current system metrics snapshot."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    cpu_cores: int = 1
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    gpu_name: Optional[str] = None
    gpu_util_percent: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "cpu_cores": self.cpu_cores,
            "ram_used_gb": round(self.ram_used_gb, 2),
            "ram_total_gb": round(self.ram_total_gb, 2),
            "ram_percent": round(self.ram_percent, 1),
            "gpu_name": self.gpu_name,
            "gpu_util_percent": round(self.gpu_util_percent, 1),
            "vram_used_gb": round(self.vram_used_gb, 2),
            "vram_total_gb": round(self.vram_total_gb, 2),
        }

    def format_display(self) -> str:
        """Format metrics for display."""
        lines = []
        lines.append(f"CPU: {self.cpu_percent:.0f}% ({self.cpu_cores} cores)")
        lines.append(f"RAM: {self.ram_used_gb:.1f}/{self.ram_total_gb:.1f} GB ({self.ram_percent:.0f}%)")
        if self.gpu_name:
            lines.append(f"GPU: {self.gpu_util_percent:.0f}% | VRAM: {self.vram_used_gb:.1f}/{self.vram_total_gb:.1f} GB")
        return " | ".join(lines)


class SystemMonitor:
    """
    Real-time system metrics monitor.

    Can run in background and periodically report metrics via callback.
    """

    def __init__(self, update_interval: float = 2.0):
        """
        Initialize system monitor.

        Args:
            update_interval: Seconds between metric updates.
        """
        self.update_interval = update_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._metrics_history: List[SystemMetrics] = []
        self._callbacks: List[Callable[[SystemMetrics], None]] = []
        self._has_psutil = False
        self._has_nvidia = False
        self._cached_cpu_percent: float = 0.0  # Cached CPU value for non-blocking reads
        self._cpu_lock = threading.Lock()

        # Check available monitoring tools
        self._check_capabilities()

        # Initialize CPU percent baseline (first call returns 0.0, so prime it)
        if self._has_psutil:
            try:
                import psutil
                psutil.cpu_percent(interval=None)  # Prime the CPU counter
            except Exception:
                pass

    def _check_capabilities(self) -> None:
        """Check what monitoring capabilities are available."""
        try:
            import psutil
            self._has_psutil = True
        except ImportError:
            logger.warning("psutil not available - CPU/RAM monitoring limited")

        try:
            result = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True, timeout=5
            )
            self._has_nvidia = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    def get_metrics(self, blocking: bool = False) -> SystemMetrics:
        """
        Get current system metrics snapshot.

        Args:
            blocking: If True, use blocking CPU measurement (more accurate but slower).
                      If False (default), use non-blocking cached value for responsiveness.
        """
        metrics = SystemMetrics()

        # CPU and RAM via psutil
        if self._has_psutil:
            try:
                import psutil

                if blocking:
                    # Blocking measurement for accuracy (used in background thread)
                    metrics.cpu_percent = psutil.cpu_percent(interval=0.5)
                    # Update cache
                    with self._cpu_lock:
                        self._cached_cpu_percent = metrics.cpu_percent
                else:
                    # Non-blocking: use cached value, update in background
                    # This call returns CPU since last call (non-blocking)
                    instant_cpu = psutil.cpu_percent(interval=None)
                    with self._cpu_lock:
                        # Use cached value if available and instant reading seems off
                        if self._cached_cpu_percent > 0:
                            # Blend for smoother updates
                            metrics.cpu_percent = self._cached_cpu_percent
                        else:
                            metrics.cpu_percent = instant_cpu

                metrics.cpu_cores = psutil.cpu_count() or 1

                mem = psutil.virtual_memory()
                metrics.ram_used_gb = mem.used / (1024**3)
                metrics.ram_total_gb = mem.total / (1024**3)
                metrics.ram_percent = mem.percent
            except Exception as e:
                logger.debug(f"psutil error: {e}")
        else:
            # Fallback for CPU count
            import os
            metrics.cpu_cores = os.cpu_count() or 1

        # GPU via nvidia-smi
        if self._has_nvidia:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 4:
                        metrics.gpu_name = parts[0].strip()
                        metrics.vram_used_gb = float(parts[1].strip()) / 1024
                        metrics.vram_total_gb = float(parts[2].strip()) / 1024
                        metrics.gpu_util_percent = float(parts[3].strip())
            except Exception as e:
                logger.debug(f"nvidia-smi error: {e}")

        return metrics

    def add_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """Add a callback to receive metric updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug("System monitor started")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.debug("System monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Use blocking=True for accurate CPU measurement in background
                metrics = self.get_metrics(blocking=True)
                self._metrics_history.append(metrics)

                # Keep last 100 samples
                if len(self._metrics_history) > 100:
                    self._metrics_history = self._metrics_history[-100:]

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.debug(f"Callback error: {e}")

            except Exception as e:
                logger.debug(f"Monitor loop error: {e}")

            time.sleep(self.update_interval)

    def get_history(self) -> List[SystemMetrics]:
        """Get metrics history."""
        return self._metrics_history.copy()

    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak values from metrics history."""
        if not self._metrics_history:
            return {}

        return {
            "peak_cpu_percent": max(m.cpu_percent for m in self._metrics_history),
            "peak_ram_percent": max(m.ram_percent for m in self._metrics_history),
            "peak_ram_gb": max(m.ram_used_gb for m in self._metrics_history),
            "peak_gpu_percent": max(m.gpu_util_percent for m in self._metrics_history),
            "peak_vram_gb": max(m.vram_used_gb for m in self._metrics_history),
        }

    def __enter__(self) -> "SystemMonitor":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


class AnalysisProgressTracker:
    """
    Tracks analysis progress with real-time metrics display.

    Combines progress messages with system metrics for comprehensive
    analysis monitoring.
    """

    def __init__(self, ui=None, update_interval: float = 2.0):
        """
        Initialize progress tracker.

        Args:
            ui: Optional terminal UI for display.
            update_interval: Seconds between metric updates.
        """
        self.ui = ui
        self.monitor = SystemMonitor(update_interval)
        self.start_time: Optional[datetime] = None
        self.current_phase: str = ""
        self.phases_completed: List[str] = []
        self._last_metrics_line: str = ""

    def start(self, phase: str = "Initializing") -> None:
        """Start tracking with initial phase."""
        self.start_time = datetime.now()
        self.current_phase = phase
        self.phases_completed = []

        # Add display callback
        self.monitor.add_callback(self._on_metrics_update)
        self.monitor.start()

        self._display_phase(phase)

    def update_phase(self, phase: str) -> None:
        """Update current phase."""
        if self.current_phase:
            self.phases_completed.append(self.current_phase)
        self.current_phase = phase
        self._display_phase(phase)

    def stop(self) -> Dict[str, Any]:
        """Stop tracking and return summary."""
        self.monitor.stop()

        elapsed = None
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()

        return {
            "elapsed_seconds": elapsed,
            "phases_completed": self.phases_completed + [self.current_phase],
            "peak_metrics": self.monitor.get_peak_metrics(),
        }

    def _display_phase(self, phase: str) -> None:
        """Display phase update."""
        if self.ui and hasattr(self.ui, 'print_info'):
            self.ui.print_info(f"[{phase}]")
        else:
            print(f"[{phase}]")

    def _on_metrics_update(self, metrics: SystemMetrics) -> None:
        """Handle metrics update - display to user."""
        metrics_line = metrics.format_display()

        # Only update if metrics changed significantly
        if metrics_line != self._last_metrics_line:
            self._last_metrics_line = metrics_line

            if self.ui and hasattr(self.ui, 'print_metrics'):
                self.ui.print_metrics(metrics_line)
            elif self.ui and hasattr(self.ui, 'console'):
                # Use rich console for inline update
                try:
                    self.ui.console.print(f"  {metrics_line}", end="\r")
                except Exception:
                    pass

    def __enter__(self) -> "AnalysisProgressTracker":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.stop()


def save_analysis_json(
    result: Dict[str, Any],
    output_dir: Optional[Path] = None,
    filename_prefix: str = "analysis"
) -> Path:
    """
    Save analysis result to JSON file.

    Args:
        result: Analysis result dictionary.
        output_dir: Directory for output (default: reports/).
        filename_prefix: Prefix for filename.

    Returns:
        Path to saved JSON file.
    """
    import json

    if output_dir is None:
        output_dir = Path("reports")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp and target name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = result.get("target_name", "unknown")
    # Sanitize target name for filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in target_name)[:50]

    filename = f"{filename_prefix}_{safe_name}_{timestamp}.json"
    output_path = output_dir / filename

    # Add metadata
    result_with_meta = {
        "saved_at": datetime.now().isoformat(),
        "benderbox_version": result.get("version", "unknown"),
        **result
    }

    output_path.write_text(json.dumps(result_with_meta, indent=2, default=str), encoding="utf-8")
    logger.info(f"Analysis JSON saved to: {output_path}")

    return output_path


def get_analysis_summary(
    result: Dict[str, Any],
    analysis_type: str = "general"
) -> str:
    """
    Generate type-specific analysis summary.

    Args:
        result: Analysis result dictionary.
        analysis_type: Type of analysis (interrogation, security, triage, general).

    Returns:
        Formatted summary string.
    """
    summary = result.get("summary", {})
    risk = summary.get("risk", {})
    risk_level = risk.get("level", "unknown").upper()
    risk_score = risk.get("score", 0)
    target_name = result.get("target_name", "Unknown")
    results = result.get("results", [])

    # Count results
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    warnings = sum(1 for r in results if r.get("status") == "warning")

    lines = []

    if analysis_type == "interrogation":
        # Interrogation-specific summary
        lines.append(f"Interrogation Complete: {target_name}")
        lines.append(f"Risk Assessment: {risk_level} ({risk_score}/100)")
        lines.append(f"Tests Executed: {len(results)}")
        lines.append(f"  Passed: {passed} | Failed: {failed} | Warnings: {warnings}")

        # Highlight jailbreak-specific findings
        jailbreak_findings = [r for r in results if "jailbreak" in r.get("category", "").lower()]
        if jailbreak_findings:
            jb_failed = sum(1 for r in jailbreak_findings if r.get("status") == "failed")
            lines.append(f"Jailbreak Tests: {jb_failed}/{len(jailbreak_findings)} vulnerabilities found")

    elif analysis_type == "security":
        # Security scan summary
        lines.append(f"Security Scan: {target_name}")
        lines.append(f"Overall Risk: {risk_level} ({risk_score}/100)")

        # Group by severity
        critical = [r for r in results if r.get("severity") == "critical"]
        high = [r for r in results if r.get("severity") == "high"]
        medium = [r for r in results if r.get("severity") == "medium"]

        lines.append(f"Findings: {len(critical)} critical, {len(high)} high, {len(medium)} medium")

    elif analysis_type == "triage":
        # Quick triage summary
        lines.append(f"Triage: {target_name}")
        lines.append(f"Quick Assessment: {risk_level}")

        if risk_score >= 70:
            lines.append("Recommendation: Requires detailed review")
        elif risk_score >= 40:
            lines.append("Recommendation: Review high-severity items")
        else:
            lines.append("Recommendation: Low risk, standard monitoring")

    else:
        # General summary
        lines.append(f"Analysis: {target_name}")
        lines.append(f"Risk: {risk_level} ({risk_score}/100)")
        lines.append(f"Results: {passed} passed, {failed} failed, {warnings} warnings")

    return "\n".join(lines)
