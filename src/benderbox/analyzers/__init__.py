"""
BenderBox Analyzers Module - Semantic and Behavioral Analysis

This module provides:
- SemanticAnalyzer: LLM-powered code and configuration analysis
- BehaviorAnalyzer: Model output pattern and safety analysis
- ComparativeAnalyzer: Cross-analysis comparison and ranking
- Test integration for sandbox framework
"""

__all__ = [
    # Semantic Analysis
    "SemanticAnalyzer",
    "SemanticAnalysisResult",
    "SemanticFinding",
    "FindingSeverity",
    "FindingCategory",
    # Behavior Analysis
    "BehaviorAnalyzer",
    "BehaviorProfile",
    "OutputAnalysis",
    "BehaviorSignal",
    "BehaviorCategory",
    "RiskLevel",
    # Comparative Analysis
    "ComparativeAnalyzer",
    "ComparisonResult",
    "ComparisonMetric",
    "RankingResult",
    "ComparisonType",
    # Test Integration
    "SemanticSecurityTest",
    "BehaviorAnalysisTest",
    "register_analyzer_tests",
    "add_to_profiles",
]


def __getattr__(name):
    """Lazy import for module attributes."""
    # Semantic Analysis
    if name in ("SemanticAnalyzer", "SemanticAnalysisResult", "SemanticFinding", "FindingSeverity", "FindingCategory"):
        from benderbox.analyzers.semantic import (
            SemanticAnalyzer,
            SemanticAnalysisResult,
            SemanticFinding,
            FindingSeverity,
            FindingCategory,
        )
        return locals()[name]

    # Behavior Analysis
    if name in ("BehaviorAnalyzer", "BehaviorProfile", "OutputAnalysis", "BehaviorSignal", "BehaviorCategory", "RiskLevel"):
        from benderbox.analyzers.behavior import (
            BehaviorAnalyzer,
            BehaviorProfile,
            OutputAnalysis,
            BehaviorSignal,
            BehaviorCategory,
            RiskLevel,
        )
        return locals()[name]

    # Comparative Analysis
    if name in ("ComparativeAnalyzer", "ComparisonResult", "ComparisonMetric", "RankingResult", "ComparisonType"):
        from benderbox.analyzers.comparative import (
            ComparativeAnalyzer,
            ComparisonResult,
            ComparisonMetric,
            RankingResult,
            ComparisonType,
        )
        return locals()[name]

    # Test Integration
    if name in ("SemanticSecurityTest", "BehaviorAnalysisTest", "register_analyzer_tests", "add_to_profiles"):
        from benderbox.analyzers.test_integration import (
            SemanticSecurityTest,
            BehaviorAnalysisTest,
            register_analyzer_tests,
            add_to_profiles,
        )
        return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
