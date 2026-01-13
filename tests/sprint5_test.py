"""
Sprint 5: HTML Polish - Test Suite

Tests for:
- M3.3: Interactive Features
- M3.4: Comparison Reports
- M3.5: CLI Integration
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_html_interactive_features():
    """Test 1: HTML template has interactive features."""
    try:
        from benderbox.reporting.html_generator import HTMLReportGenerator, HTML_TEMPLATE

        # Check for theme toggle support in template
        assert 'data-theme' in HTML_TEMPLATE, "Missing theme toggle support"

        # Check for print styles (PDF export)
        assert "@media print" in HTML_TEMPLATE, "Missing print media query for PDF export"

        # Check toolbar is generated correctly
        generator = HTMLReportGenerator()
        toolbar = generator._generate_toolbar()
        assert 'class="toolbar"' in toolbar, "Missing toolbar class"
        assert 'search-box' in toolbar, "Missing search box"
        assert 'toggleTheme' in toolbar, "Missing theme toggle function"
        assert 'exportPDF' in toolbar, "Missing PDF export function"

        print("PASS: HTML template has interactive features")
        return True
    except AssertionError as e:
        print(f"FAIL: HTML interactive features error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: HTML interactive features error: {e}")
        return False


def test_html_dark_light_mode():
    """Test 2: HTML supports dark/light mode."""
    try:
        from benderbox.reporting.html_generator import HTML_TEMPLATE

        # Check CSS variables for theme
        assert "--bg:" in HTML_TEMPLATE or "--bg :" in HTML_TEMPLATE, "Missing background CSS variable"
        assert "--text:" in HTML_TEMPLATE or "--text :" in HTML_TEMPLATE, "Missing text CSS variable"

        # Check dark theme styles
        assert '[data-theme="dark"]' in HTML_TEMPLATE, "Missing dark theme selector"

        print("PASS: HTML supports dark/light mode")
        return True
    except AssertionError as e:
        print(f"FAIL: Dark/light mode error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: Dark/light mode error: {e}")
        return False


def test_html_table_of_contents():
    """Test 3: HTMLReportGenerator creates table of contents."""
    try:
        from benderbox.reporting.html_generator import HTMLReportGenerator

        generator = HTMLReportGenerator()

        # Check TOC method exists
        assert hasattr(generator, "_generate_toc"), "Missing _generate_toc method"

        # Test TOC generation (section_id, title, badge)
        sections = [
            ("risk-assessment", "Risk Overview", ""),
            ("findings", "Findings", "5"),
            ("recommendations", "Recommendations", ""),
        ]
        toc = generator._generate_toc(sections)

        assert "toc" in toc.lower() or "table" in toc.lower() or "contents" in toc.lower(), "TOC output missing expected markers"
        assert "Risk Overview" in toc, "Risk Overview missing from TOC"
        assert "Findings" in toc, "Findings missing from TOC"
        assert 'href="#findings"' in toc, "Missing anchor links"

        print("PASS: HTMLReportGenerator creates table of contents")
        return True
    except AssertionError as e:
        print(f"FAIL: Table of contents error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: Table of contents error: {e}")
        return False


def test_html_svg_charts():
    """Test 4: HTMLReportGenerator creates SVG charts."""
    try:
        from benderbox.reporting.html_generator import HTMLReportGenerator

        generator = HTMLReportGenerator()

        # Check chart methods exist
        assert hasattr(generator, "_generate_pie_chart_svg"), "Missing _generate_pie_chart_svg method"
        assert hasattr(generator, "_generate_comparison_chart"), "Missing _generate_comparison_chart method"

        # Test with empty data (should handle gracefully)
        empty_result = generator._generate_pie_chart_svg([], 0)
        assert "No data" in empty_result, "Should handle empty data"

        # Test that generate() method creates a full HTML report with charts
        test_result = {
            "target_name": "test-model",
            "target_type": "model",
            "summary": {"risk": {"level": "low", "score": 20}},
            "results": [
                {"test_name": "test1", "status": "passed", "severity": "info"},
                {"test_name": "test2", "status": "failed", "severity": "high"},
            ],
        }

        full_html = generator.generate(test_result)
        assert "<svg" in full_html, "Generated HTML should contain SVG charts"

        print("PASS: HTMLReportGenerator creates SVG charts")
        return True
    except AssertionError as e:
        print(f"FAIL: SVG charts error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: SVG charts error: {e}")
        return False


def test_html_comparison_report():
    """Test 5: HTMLReportGenerator generates comparison reports."""
    try:
        from benderbox.reporting.html_generator import HTMLReportGenerator

        generator = HTMLReportGenerator()

        # Check comparison method exists
        assert hasattr(generator, "generate_comparison"), "Missing generate_comparison method"

        # Test comparison generation
        test_comparison = {
            "targets": ["model1.gguf", "model2.gguf"],
            "results": [
                {"risk_score": 30, "risk_level": "low", "finding_count": 5},
                {"risk_score": 60, "risk_level": "medium", "finding_count": 10},
            ],
            "summary": {"safest": "model1.gguf"},
        }

        html = generator.generate_comparison(test_comparison)

        assert "<!DOCTYPE html>" in html, "Missing DOCTYPE"
        assert "model1.gguf" in html, "Missing first model"
        assert "model2.gguf" in html, "Missing second model"
        assert "comparison" in html.lower(), "Missing comparison markers"

        print("PASS: HTMLReportGenerator generates comparison reports")
        return True
    except AssertionError as e:
        print(f"FAIL: Comparison report error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: Comparison report error: {e}")
        return False


def test_html_diff_highlighting():
    """Test 6: Comparison reports have diff highlighting."""
    try:
        from benderbox.reporting.html_generator import HTMLReportGenerator

        generator = HTMLReportGenerator()

        # Check diff method exists
        assert hasattr(generator, "_generate_comparison_table"), "Missing _generate_comparison_table method"

        # Use correct method signature: targets and results as separate args
        targets = ["model1", "model2"]
        results = [
            {"risk_score": 30, "risk_level": "low", "finding_count": 5},
            {"risk_score": 60, "risk_level": "medium", "finding_count": 10},
        ]

        table_html = generator._generate_comparison_table(targets, results)

        # Check for diff styling
        assert "diff" in table_html.lower() or "risk-low" in table_html.lower() or "risk-medium" in table_html.lower(), "Missing diff highlighting"

        print("PASS: Comparison reports have diff highlighting")
        return True
    except AssertionError as e:
        print(f"FAIL: Diff highlighting error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: Diff highlighting error: {e}")
        return False


def test_cli_analyze_format_option():
    """Test 7: CLI analyze command has --format option with html choice."""
    try:
        from benderbox.ui.app import cli
        from click.testing import CliRunner

        runner = CliRunner()

        # Test help for analyze command
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0, f"Exit code: {result.exit_code}"
        assert "--format" in result.output or "-f" in result.output, "Missing --format option"
        assert "html" in result.output, "Missing html format choice"

        print("PASS: CLI analyze command has --format option with html")
        return True
    except AssertionError as e:
        print(f"FAIL: CLI analyze format option error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: CLI analyze format option error: {e}")
        return False


def test_cli_analyze_open_option():
    """Test 8: CLI analyze command has --open option."""
    try:
        from benderbox.ui.app import cli
        from click.testing import CliRunner

        runner = CliRunner()

        # Test help for analyze command
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0, f"Exit code: {result.exit_code}"
        assert "--open" in result.output, "Missing --open option"
        assert "browser" in result.output.lower(), "Missing browser description"

        print("PASS: CLI analyze command has --open option")
        return True
    except AssertionError as e:
        print(f"FAIL: CLI analyze --open option error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: CLI analyze --open option error: {e}")
        return False


def test_cli_export_command():
    """Test 9: CLI has export command group."""
    try:
        from benderbox.ui.app import cli
        from click.testing import CliRunner

        runner = CliRunner()

        # Test help for export command
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0, f"Exit code: {result.exit_code}"
        assert "export" in result.output.lower(), "Missing export description"

        # Test export subcommands exist
        result = runner.invoke(cli, ["export", "report", "--help"])
        assert result.exit_code == 0, f"Export report exit code: {result.exit_code}"

        result = runner.invoke(cli, ["export", "batch", "--help"])
        assert result.exit_code == 0, f"Export batch exit code: {result.exit_code}"

        print("PASS: CLI has export command group")
        return True
    except AssertionError as e:
        print(f"FAIL: CLI export command error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: CLI export command error: {e}")
        return False


def test_cli_interrogate_format_option():
    """Test 10: CLI interrogate command has --format and --open options."""
    try:
        from benderbox.ui.app import cli
        from click.testing import CliRunner

        runner = CliRunner()

        # Test help for interrogate command
        result = runner.invoke(cli, ["interrogate", "--help"])
        assert result.exit_code == 0, f"Exit code: {result.exit_code}"
        assert "--format" in result.output or "-f" in result.output, "Missing --format option"
        assert "html" in result.output, "Missing html format choice"
        assert "--open" in result.output, "Missing --open option"

        print("PASS: CLI interrogate command has --format and --open options")
        return True
    except AssertionError as e:
        print(f"FAIL: CLI interrogate format option error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: CLI interrogate format option error: {e}")
        return False


def main():
    """Run all Sprint 5 tests."""
    print("=" * 60)
    print("Sprint 5: HTML Polish - Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Test 1: HTML interactive features", test_html_interactive_features),
        ("Test 2: Dark/light mode support", test_html_dark_light_mode),
        ("Test 3: Table of contents generation", test_html_table_of_contents),
        ("Test 4: SVG chart generation", test_html_svg_charts),
        ("Test 5: Comparison report generation", test_html_comparison_report),
        ("Test 6: Diff highlighting", test_html_diff_highlighting),
        ("Test 7: CLI analyze --format option", test_cli_analyze_format_option),
        ("Test 8: CLI analyze --open option", test_cli_analyze_open_option),
        ("Test 9: CLI export command group", test_cli_export_command),
        ("Test 10: CLI interrogate format options", test_cli_interrogate_format_option),
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
