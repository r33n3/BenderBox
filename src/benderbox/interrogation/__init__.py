"""
BenderBox Model Interrogation Module

Framework for testing AI models to detect unwanted outputs,
validate censorship claims, and assess unknown source risks.

Supports LLM-as-judge analysis and multiple display modes
for interactive research sessions.
"""

from .prompts.schema import TestPrompt, PromptCategory, ExpectedBehavior
from .prompts.loader import PromptLibrary
from .runner.base import ModelRunner, GenerationResult, RunnerConfig
from .runner.llama_cpp import LlamaCppRunner
from .analyzer.refusal import RefusalDetector, RefusalResult
from .analyzer.harm import HarmClassifier, HarmCategory
from .analyzer.results import AnalysisResult, ResponseClassification
from .validator.censorship import CensorshipValidator, CensorshipLevel, MislabelingReport
from .scoring.risk import RiskScorer, InterrogationRiskScore

# Judge and display components
from .judge import JudgeAnalyzer, JudgeVerdict
from .display import (
    DisplayHandler,
    DisplayConfig,
    DisplayMode,
    InteractiveChoice,
    SkipCategoryException,
    AbortInterrogationException,
    create_display_handler,
)
from .followup import FollowupGenerator, FollowupResult

# Variant generation
from .variants import (
    VariantGenerator,
    VariantTechnique,
    VariantConfig,
    generate_variants,
)

# Persistent probing (multi-variant testing)
from .persistent_probe import (
    PersistentProbe,
    ProbeResult,
    ProbeAttempt,
    create_prober_from_profile,
)

# Comparative analysis
from .comparison import (
    ComparativeAnalyzer,
    ComparisonReport,
    ComparisonDimension,
    DimensionComparison,
    CategoryComparison,
    ModelResult,
    compare_models,
    compare_reports,
    compare_and_format,
    compare_and_export,
)

# Trend tracking
from .trend import (
    TrendTracker,
    TrendDirection,
    ModelTrendReport,
    analyze_trends,
)

__all__ = [
    # Prompts
    "TestPrompt",
    "PromptCategory",
    "ExpectedBehavior",
    "PromptLibrary",
    # Runner
    "ModelRunner",
    "GenerationResult",
    "RunnerConfig",
    "LlamaCppRunner",
    # Analyzer
    "RefusalDetector",
    "RefusalResult",
    "HarmClassifier",
    "HarmCategory",
    "AnalysisResult",
    "ResponseClassification",
    # Validator
    "CensorshipValidator",
    "CensorshipLevel",
    "MislabelingReport",
    # Scoring
    "RiskScorer",
    "InterrogationRiskScore",
    # Judge
    "JudgeAnalyzer",
    "JudgeVerdict",
    # Display
    "DisplayHandler",
    "DisplayConfig",
    "DisplayMode",
    "InteractiveChoice",
    "SkipCategoryException",
    "AbortInterrogationException",
    "create_display_handler",
    # Follow-up
    "FollowupGenerator",
    "FollowupResult",
    # Variants
    "VariantGenerator",
    "VariantTechnique",
    "VariantConfig",
    "generate_variants",
    # Persistent Probing
    "PersistentProbe",
    "ProbeResult",
    "ProbeAttempt",
    "create_prober_from_profile",
    # Comparison
    "ComparativeAnalyzer",
    "ComparisonReport",
    "ComparisonDimension",
    "DimensionComparison",
    "CategoryComparison",
    "ModelResult",
    "compare_models",
    "compare_reports",
    "compare_and_format",
    "compare_and_export",
    # Trend Tracking
    "TrendTracker",
    "TrendDirection",
    "ModelTrendReport",
    "analyze_trends",
]
