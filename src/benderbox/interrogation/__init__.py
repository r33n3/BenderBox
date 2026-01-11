"""
BenderBox Model Interrogation Module

Framework for testing AI models to detect unwanted outputs,
validate censorship claims, and assess unknown source risks.
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
]
