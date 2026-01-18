"""
Resource Checker for BenderBox

Provides system resource assessment for dual-model operations
like judge analysis during interrogation. Helps prevent memory
exhaustion on resource-constrained systems.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Minimum RAM thresholds for different operations
RAM_THRESHOLDS = {
    "single_model_min": 4,      # Minimum RAM to run a single small model
    "single_model_comfortable": 8,   # Comfortable RAM for single model
    "dual_model_min": 12,       # Minimum RAM for dual model (target + judge)
    "dual_model_comfortable": 16,    # Comfortable RAM for dual model
    "dual_model_large": 24,     # For larger models with judge
}

# Model size to RAM multiplier (conservative estimates)
# GGUF models typically need 1.2-1.5x file size in RAM when loaded
MODEL_RAM_MULTIPLIER = 1.3


@dataclass
class ResourceAssessment:
    """Assessment of system resources for a given operation."""

    # System info
    total_ram_gb: int
    available_ram_gb: int
    has_gpu: bool
    gpu_vram_gb: Optional[int]

    # Operation-specific
    operation: str  # "single_model", "dual_model", "judge_analysis"
    target_model_ram_gb: float
    judge_model_ram_gb: float
    total_required_ram_gb: float

    # Assessment results
    can_run: bool
    is_comfortable: bool
    risk_level: str  # "low", "medium", "high", "critical"
    warnings: List[str]
    recommendations: List[str]

    def format_summary(self) -> str:
        """Format assessment as a summary string."""
        lines = []

        # Header
        risk_indicator = {
            "low": "[OK]",
            "medium": "[WARN]",
            "high": "[RISK]",
            "critical": "[CRITICAL]"
        }.get(self.risk_level, "[?]")

        lines.append(f"Resource Assessment: {risk_indicator}")
        lines.append(f"  System RAM: {self.total_ram_gb}GB (available: ~{self.available_ram_gb}GB)")

        if self.has_gpu and self.gpu_vram_gb:
            lines.append(f"  GPU VRAM: {self.gpu_vram_gb}GB")

        lines.append(f"  Required RAM: ~{self.total_required_ram_gb:.1f}GB")

        if self.operation == "dual_model":
            lines.append(f"    - Target model: ~{self.target_model_ram_gb:.1f}GB")
            lines.append(f"    - Judge model: ~{self.judge_model_ram_gb:.1f}GB")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")

        if self.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")

        return "\n".join(lines)


class ResourceChecker:
    """
    Checks system resources for BenderBox operations.

    Helps prevent memory exhaustion by warning users before
    starting resource-intensive operations like dual-model
    interrogation with judge analysis.
    """

    def __init__(self, model_manager=None):
        """
        Initialize ResourceChecker.

        Args:
            model_manager: Optional ModelManager instance for model analysis.
        """
        self._model_manager = model_manager

    def _get_model_manager(self):
        """Get or create ModelManager."""
        if self._model_manager is None:
            from benderbox.utils.model_manager import ModelManager
            self._model_manager = ModelManager()
        return self._model_manager

    def get_system_info(self) -> Dict:
        """
        Get current system resource information.

        Returns:
            Dictionary with RAM, GPU, and CPU info.
        """
        mm = self._get_model_manager()

        ram_gb = mm.get_system_ram_gb()
        gpu_info = mm.get_gpu_info()

        # Estimate available RAM (leave room for OS and other processes)
        # On Windows, assume ~4GB for OS; on Linux, ~2GB
        import os
        os_overhead = 4 if os.name == "nt" else 2
        available_ram_gb = max(ram_gb - os_overhead, 2)

        return {
            "total_ram_gb": ram_gb,
            "available_ram_gb": available_ram_gb,
            "has_gpu": gpu_info.get("has_nvidia", False),
            "gpu_vram_gb": gpu_info.get("vram_gb"),
            "gpu_name": gpu_info.get("gpu_name"),
            "cuda_available": gpu_info.get("cuda_available", False),
        }

    def estimate_model_ram(self, model_path: str) -> float:
        """
        Estimate RAM required to load a model.

        Args:
            model_path: Path to GGUF model file.

        Returns:
            Estimated RAM in GB.
        """
        path = Path(model_path)

        if not path.exists():
            # Default estimate for unknown models
            return 4.0

        file_size_mb = path.stat().st_size / (1024 * 1024)
        file_size_gb = file_size_mb / 1024

        # Conservative multiplier for GGUF models
        return file_size_gb * MODEL_RAM_MULTIPLIER

    def assess_judge_mode(
        self,
        target_model_path: Optional[str] = None,
        judge_model_path: Optional[str] = None,
        target_model_ram_gb: Optional[float] = None,
    ) -> ResourceAssessment:
        """
        Assess if system can handle judge mode (dual model).

        Args:
            target_model_path: Path to target model being interrogated.
            judge_model_path: Path to judge model for analysis.
            target_model_ram_gb: Pre-calculated RAM for target (if known).

        Returns:
            ResourceAssessment with warnings and recommendations.
        """
        system = self.get_system_info()

        # Estimate RAM for both models
        if target_model_ram_gb is not None:
            target_ram = target_model_ram_gb
        elif target_model_path:
            target_ram = self.estimate_model_ram(target_model_path)
        else:
            target_ram = 4.0  # Default for unknown

        if judge_model_path:
            judge_ram = self.estimate_model_ram(judge_model_path)
        else:
            judge_ram = 2.0  # Default for typical judge model (TinyLlama ~1.5GB)

        total_required = target_ram + judge_ram
        available = system["available_ram_gb"]

        warnings = []
        recommendations = []

        # Assess risk level
        if total_required > available:
            risk_level = "critical"
            can_run = False
            warnings.append(
                f"Dual-model mode requires ~{total_required:.1f}GB RAM, "
                f"but only {available}GB is available"
            )
            warnings.append("System may become unresponsive or crash")
            recommendations.append("Run without --judge flag to use heuristic analysis only")
            recommendations.append("Or use a smaller judge model")
        elif total_required > available * 0.9:
            risk_level = "high"
            can_run = True
            warnings.append(
                f"Dual-model mode will use ~{total_required:.1f}GB of {available}GB available RAM"
            )
            warnings.append("System may slow down significantly during analysis")
            recommendations.append("Close other applications before running")
            recommendations.append("Consider using --judge only for important tests")
        elif total_required > available * 0.7:
            risk_level = "medium"
            can_run = True
            warnings.append(
                f"Dual-model mode will use ~{int(total_required / available * 100)}% of available RAM"
            )
            recommendations.append("Close memory-intensive applications for best performance")
        else:
            risk_level = "low"
            can_run = True

        # Check if system meets comfortable thresholds
        is_comfortable = (
            system["total_ram_gb"] >= RAM_THRESHOLDS["dual_model_comfortable"]
        )

        # GPU-specific recommendations
        if system["has_gpu"] and system["cuda_available"]:
            if risk_level in ("high", "critical"):
                recommendations.append(
                    f"GPU detected ({system.get('gpu_name', 'NVIDIA')}). "
                    "Consider using --gpu-layers to offload model to GPU"
                )

        return ResourceAssessment(
            total_ram_gb=system["total_ram_gb"],
            available_ram_gb=available,
            has_gpu=system["has_gpu"],
            gpu_vram_gb=system["gpu_vram_gb"],
            operation="dual_model",
            target_model_ram_gb=target_ram,
            judge_model_ram_gb=judge_ram,
            total_required_ram_gb=total_required,
            can_run=can_run,
            is_comfortable=is_comfortable,
            risk_level=risk_level,
            warnings=warnings,
            recommendations=recommendations,
        )

    def assess_single_model(
        self,
        model_path: Optional[str] = None,
        model_ram_gb: Optional[float] = None,
    ) -> ResourceAssessment:
        """
        Assess if system can handle a single model load.

        Args:
            model_path: Path to model file.
            model_ram_gb: Pre-calculated RAM requirement.

        Returns:
            ResourceAssessment with warnings and recommendations.
        """
        system = self.get_system_info()

        if model_ram_gb is not None:
            required_ram = model_ram_gb
        elif model_path:
            required_ram = self.estimate_model_ram(model_path)
        else:
            required_ram = 4.0

        available = system["available_ram_gb"]

        warnings = []
        recommendations = []

        if required_ram > available:
            risk_level = "critical"
            can_run = False
            warnings.append(f"Model requires ~{required_ram:.1f}GB but only {available}GB available")
            recommendations.append("Use a smaller/more quantized model")
        elif required_ram > available * 0.9:
            risk_level = "high"
            can_run = True
            warnings.append(f"Model will use most available RAM ({required_ram:.1f}GB of {available}GB)")
        elif required_ram > available * 0.7:
            risk_level = "medium"
            can_run = True
        else:
            risk_level = "low"
            can_run = True

        is_comfortable = (
            system["total_ram_gb"] >= RAM_THRESHOLDS["single_model_comfortable"]
        )

        return ResourceAssessment(
            total_ram_gb=system["total_ram_gb"],
            available_ram_gb=available,
            has_gpu=system["has_gpu"],
            gpu_vram_gb=system["gpu_vram_gb"],
            operation="single_model",
            target_model_ram_gb=required_ram,
            judge_model_ram_gb=0.0,
            total_required_ram_gb=required_ram,
            can_run=can_run,
            is_comfortable=is_comfortable,
            risk_level=risk_level,
            warnings=warnings,
            recommendations=recommendations,
        )

    def get_judge_requirements(
        self,
        target_model_path: Optional[str] = None,
    ) -> Dict:
        """
        Get requirements for using judge mode with a target model.

        Args:
            target_model_path: Path to model being interrogated.

        Returns:
            Dictionary with minimum requirements and recommendations.
        """
        if target_model_path:
            target_ram = self.estimate_model_ram(target_model_path)
        else:
            target_ram = 4.0

        # Assume typical judge model (TinyLlama Q4)
        judge_ram = 2.0

        total_min = target_ram + judge_ram
        total_comfortable = total_min * 1.3  # 30% headroom

        return {
            "target_model_ram_gb": round(target_ram, 1),
            "judge_model_ram_gb": round(judge_ram, 1),
            "minimum_ram_gb": round(total_min + 4, 0),  # +4GB for OS
            "recommended_ram_gb": round(total_comfortable + 4, 0),
            "description": (
                f"Judge mode loads two models simultaneously:\n"
                f"  - Target model: ~{target_ram:.1f}GB\n"
                f"  - Judge model: ~{judge_ram:.1f}GB\n"
                f"  - OS overhead: ~4GB\n"
                f"  - Minimum system RAM: {int(total_min + 4)}GB\n"
                f"  - Recommended RAM: {int(total_comfortable + 4)}GB"
            ),
        }

    def is_low_resource_system(self) -> bool:
        """
        Check if this is a low-resource system.

        Returns:
            True if system RAM is below comfortable threshold.
        """
        system = self.get_system_info()
        return system["total_ram_gb"] < RAM_THRESHOLDS["single_model_comfortable"]

    def should_warn_judge_mode(
        self,
        target_model_path: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Quick check if we should warn about judge mode.

        Args:
            target_model_path: Path to target model.

        Returns:
            Tuple of (should_warn, warning_message).
        """
        assessment = self.assess_judge_mode(target_model_path=target_model_path)

        if assessment.risk_level in ("high", "critical"):
            return True, assessment.format_summary()
        return False, None


def check_judge_resources(
    target_model_path: Optional[str] = None,
    judge_model_path: Optional[str] = None,
) -> ResourceAssessment:
    """
    Convenience function to check resources for judge mode.

    Args:
        target_model_path: Path to model being interrogated.
        judge_model_path: Path to judge model.

    Returns:
        ResourceAssessment with full analysis.
    """
    checker = ResourceChecker()
    return checker.assess_judge_mode(
        target_model_path=target_model_path,
        judge_model_path=judge_model_path,
    )


def get_judge_requirements(target_model_path: Optional[str] = None) -> Dict:
    """
    Convenience function to get judge mode requirements.

    Args:
        target_model_path: Path to target model.

    Returns:
        Dictionary with requirements info.
    """
    checker = ResourceChecker()
    return checker.get_judge_requirements(target_model_path)
