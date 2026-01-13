"""
Sprint 4: Integration & HTML - Test Suite

Tests for:
- M1.3: Behavior Integration
- M2.3: Comparison UI/Reporting
- M3.1: HTML Template System
- M3.2: Report Sections
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_behavior_analyzer_intent():
    """Test 1: ANALYZE_BEHAVIOR intent exists."""
    try:
        from benderbox.nlp.intent import IntentType, IntentRouter

        # Check enum exists
        assert hasattr(IntentType, "ANALYZE_BEHAVIOR"), "Missing ANALYZE_BEHAVIOR enum"

        # Check router can classify behavior queries
        router = IntentRouter()

        # Test pattern matching - "capability assessment" should match
        intent = router._keyword_match("capability assessment for safety")
        assert intent is not None, "No intent matched for 'capability assessment for safety'"
        assert intent.intent_type == IntentType.ANALYZE_BEHAVIOR, f"Expected ANALYZE_BEHAVIOR, got {intent.intent_type}"

        # Check intent description exists
        desc = router.get_intent_description(IntentType.ANALYZE_BEHAVIOR)
        assert "behavior" in desc.lower(), f"Intent description missing 'behavior': {desc}"

        # Verify intent is in classification descriptions
        classification_prompt = router._build_classification_prompt("test")
        assert "ANALYZE_BEHAVIOR" in classification_prompt

        print("PASS: ANALYZE_BEHAVIOR intent exists and classifies correctly")
        return True
    except AssertionError as e:
        print(f"FAIL: ANALYZE_BEHAVIOR intent error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: ANALYZE_BEHAVIOR intent error: {e}")
        return False


def test_analysis_bridge_behavior():
    """Test 2: AnalysisBridge has analyze_behavior method."""
    try:
        from benderbox.nlp.analysis_bridge import AnalysisBridge

        bridge = AnalysisBridge()

        # Check method exists
        assert hasattr(bridge, "analyze_behavior")
        assert callable(getattr(bridge, "analyze_behavior"))

        # Check helper methods
        assert hasattr(bridge, "_get_behavior_test_prompts")
        assert hasattr(bridge, "_score_to_risk_level")

        # Test prompt generation
        quick_prompts = bridge._get_behavior_test_prompts("quick")
        assert len(quick_prompts) > 0

        standard_prompts = bridge._get_behavior_test_prompts("standard")
        assert len(standard_prompts) > len(quick_prompts)

        # Test score conversion
        assert bridge._score_to_risk_level(85) == "low"
        assert bridge._score_to_risk_level(65) == "medium"
        assert bridge._score_to_risk_level(45) == "high"
        assert bridge._score_to_risk_level(25) == "critical"

        print("PASS: AnalysisBridge has analyze_behavior method")
        return True
    except Exception as e:
        print(f"FAIL: AnalysisBridge behavior method error: {e}")
        return False


def test_interrogation_report_behavior():
    """Test 3: InterrogationReport includes behavior_profile field."""
    try:
        from benderbox.interrogation.reports.generator import (
            InterrogationReport,
            ReportGenerator,
        )
        import dataclasses

        # Check field exists
        fields = {f.name for f in dataclasses.fields(InterrogationReport)}
        assert "behavior_profile" in fields

        # Check ReportGenerator accepts behavior_profile
        import inspect
        sig = inspect.signature(ReportGenerator.generate)
        assert "behavior_profile" in sig.parameters

        # Create report with behavior profile
        report = InterrogationReport(
            model_path="test.gguf",
            model_name="test-model",
            behavior_profile=None,  # Can be None
        )
        assert report is not None

        print("PASS: InterrogationReport includes behavior_profile")
        return True
    except Exception as e:
        print(f"FAIL: InterrogationReport behavior_profile error: {e}")
        return False


def test_cli_behavior_command():
    """Test 4: CLI has behavior command."""
    try:
        from benderbox.ui.app import cli
        from click.testing import CliRunner

        runner = CliRunner()

        # Test help for behavior command
        result = runner.invoke(cli, ["behavior", "--help"])
        assert result.exit_code == 0, f"Exit code: {result.exit_code}"
        assert "behavioral analysis" in result.output.lower()

        print("PASS: CLI behavior command exists")
        return True
    except Exception as e:
        print(f"FAIL: CLI behavior command error: {e}")
        return False


def test_cli_compare_command():
    """Test 5: CLI has compare command."""
    try:
        from benderbox.ui.app import cli
        from click.testing import CliRunner

        runner = CliRunner()

        # Test help for compare command
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0, f"Exit code: {result.exit_code}"
        assert "compare" in result.output.lower()

        print("PASS: CLI compare command exists")
        return True
    except Exception as e:
        print(f"FAIL: CLI compare command error: {e}")
        return False


def test_comparison_markdown_generator():
    """Test 6: ReportGenerator has comparison markdown method."""
    try:
        from benderbox.reporting.report_generator import ReportGenerator

        generator = ReportGenerator()

        # Check method exists
        assert hasattr(generator, "generate_comparison_markdown")

        # Test generation
        test_comparison = {
            "targets": ["model1.gguf", "model2.gguf"],
            "results": [
                {"risk_score": 30, "risk_level": "low", "finding_count": 5},
                {"risk_score": 60, "risk_level": "medium", "finding_count": 10},
            ],
            "summary": {"safest": "model1.gguf", "riskiest": "model2.gguf"},
        }

        markdown = generator.generate_comparison_markdown(test_comparison)

        assert "# Model Comparison Report" in markdown
        assert "Side-by-Side Comparison" in markdown
        assert "model1.gguf" in markdown
        assert "model2.gguf" in markdown

        print("PASS: ReportGenerator has comparison markdown method")
        return True
    except Exception as e:
        print(f"FAIL: Comparison markdown generator error: {e}")
        return False


def test_comparison_html_generator():
    """Test 7: ReportGenerator has comparison HTML method."""
    try:
        from benderbox.reporting.report_generator import ReportGenerator

        generator = ReportGenerator()

        # Check method exists
        assert hasattr(generator, "generate_comparison_html")

        # Test generation
        test_comparison = {
            "targets": ["model1.gguf", "model2.gguf"],
            "results": [
                {"risk_score": 30, "risk_level": "low", "finding_count": 5},
                {"risk_score": 60, "risk_level": "medium", "finding_count": 10},
            ],
            "summary": {"safest": "model1.gguf"},
        }

        html = generator.generate_comparison_html(test_comparison)

        assert "<!DOCTYPE html>" in html
        assert "Model Comparison Report" in html
        assert "comparison-table" in html

        print("PASS: ReportGenerator has comparison HTML method")
        return True
    except Exception as e:
        print(f"FAIL: Comparison HTML generator error: {e}")
        return False


def test_html_report_generator_import():
    """Test 8: HTMLReportGenerator can be imported."""
    try:
        from benderbox.reporting.html_generator import (
            HTMLReportGenerator,
            HTMLSection,
            HTML_TEMPLATE,
        )

        generator = HTMLReportGenerator()
        assert generator is not None

        # Check template exists
        assert "<!DOCTYPE html>" in HTML_TEMPLATE
        assert "container" in HTML_TEMPLATE
        assert "responsive" in HTML_TEMPLATE.lower() or "@media" in HTML_TEMPLATE

        print("PASS: HTMLReportGenerator imports successfully")
        return True
    except ImportError as e:
        print(f"FAIL: HTMLReportGenerator import error: {e}")
        return False


def test_html_report_generation():
    """Test 9: HTMLReportGenerator generates valid HTML."""
    try:
        from benderbox.reporting.html_generator import HTMLReportGenerator

        generator = HTMLReportGenerator()

        # Test data
        test_result = {
            "target_name": "test-model.gguf",
            "target_type": "model",
            "profile": "standard",
            "timestamp": "2026-01-12T10:00:00",
            "summary": {
                "risk": {
                    "level": "medium",
                    "score": 45,
                    "factors": ["Some risk factor"],
                }
            },
            "results": [
                {
                    "test_name": "jailbreak_test",
                    "category": "jailbreak",
                    "status": "failed",
                    "severity": "high",
                    "details": {"message": "Test failed"},
                },
            ],
        }

        html = generator.generate(test_result)

        # Verify HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "test-model.gguf" in html
        assert "Risk Assessment" in html
        assert "Recommendations" in html

        # Check CSS is embedded
        assert "<style>" in html
        assert "var(--primary)" in html

        print("PASS: HTMLReportGenerator generates valid HTML")
        return True
    except Exception as e:
        print(f"FAIL: HTML report generation error: {e}")
        return False


def test_html_report_sections():
    """Test 10: HTMLReportGenerator has all required sections."""
    try:
        from benderbox.reporting.html_generator import HTMLReportGenerator

        generator = HTMLReportGenerator()

        # Check section methods exist
        assert hasattr(generator, "_generate_header")
        assert hasattr(generator, "_generate_risk_overview")
        assert hasattr(generator, "_generate_summary_stats")
        assert hasattr(generator, "_generate_findings_section")
        assert hasattr(generator, "_generate_recommendations")
        assert hasattr(generator, "_generate_behavior_section")
        assert hasattr(generator, "_generate_footer")

        # Test with behavior profile
        test_result = {
            "target_name": "test-model",
            "target_type": "model",
            "summary": {"risk": {"level": "low", "score": 20}},
            "results": [],
            "behavior_profile": {
                "overall_safety_score": 85.0,
                "total_outputs": 10,
                "safe_outputs": 9,
                "jailbreak_success_rate": 0.1,
            },
        }

        html = generator.generate(test_result)
        assert "Behavior Analysis" in html
        assert "Safety Score" in html

        print("PASS: HTMLReportGenerator has all required sections")
        return True
    except Exception as e:
        print(f"FAIL: HTML report sections error: {e}")
        return False


def main():
    """Run all Sprint 4 tests."""
    print("=" * 60)
    print("Sprint 4: Integration & HTML - Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Test 1: ANALYZE_BEHAVIOR intent", test_behavior_analyzer_intent),
        ("Test 2: AnalysisBridge behavior method", test_analysis_bridge_behavior),
        ("Test 3: InterrogationReport behavior_profile", test_interrogation_report_behavior),
        ("Test 4: CLI behavior command", test_cli_behavior_command),
        ("Test 5: CLI compare command", test_cli_compare_command),
        ("Test 6: Comparison markdown generator", test_comparison_markdown_generator),
        ("Test 7: Comparison HTML generator", test_comparison_html_generator),
        ("Test 8: HTMLReportGenerator import", test_html_report_generator_import),
        ("Test 9: HTML report generation", test_html_report_generation),
        ("Test 10: HTML report sections", test_html_report_sections),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAIL: Unexpected error: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    if failed > 0:
        print(f"\n{failed} test(s) FAILED")
        return 1
    else:
        print("\nAll tests PASSED!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
