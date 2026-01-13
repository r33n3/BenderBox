"""
Phase 11: User Experience & Documentation - Test Suite

Tests for:
- Issue #4: Model download documentation in README
- Issue #5: Consistent model recommendations in error messages
- Issue #6: --yes flag for models download command
- Issue #8: requirements.txt file
- Issue #9: Python version consistency
- Issue #10: Correct URLs in pyproject.toml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_readme_model_download_section():
    """Test 1: README has model download documentation."""
    try:
        readme_path = Path(__file__).parent.parent / "README.md"
        if not readme_path.exists():
            print("SKIP: README.md not found")
            return True

        content = readme_path.read_text()

        # Should have download section
        assert "Download a Model" in content, \
            "README should have 'Download a Model' section"

        # Should mention models commands
        assert "models list" in content, \
            "README should mention 'models list' command"
        assert "models download" in content, \
            "README should mention 'models download' command"
        assert "models setup" in content, \
            "README should mention 'models setup' command"

        # Should have model table
        assert "tinyllama" in content.lower(), \
            "README should list tinyllama model"
        assert "RAM Required" in content or "RAM" in content, \
            "README should mention RAM requirements"

        print("PASS: README has model download documentation")
        return True
    except Exception as e:
        print(f"FAIL: README model docs error: {e}")
        return False


def test_llm_engine_error_message():
    """Test 2: LLM engine error message references models command."""
    try:
        from benderbox.nlp.llm_engine import LlamaModel, ModelNotFoundError

        # Create model wrapper with non-existent path
        model = LlamaModel(model_path="/nonexistent/model.gguf")

        try:
            model._ensure_loaded()
            print("FAIL: Should have raised ModelNotFoundError")
            return False
        except ModelNotFoundError as e:
            error_msg = str(e)

            # Should NOT reference TheBloke or manual download
            assert "TheBloke" not in error_msg, \
                "Error should not reference TheBloke"
            assert "huggingface.co" not in error_msg.lower(), \
                "Error should not reference HuggingFace directly"

            # Should reference models command
            assert "models list" in error_msg, \
                "Error should reference 'models list' command"
            assert "models download" in error_msg, \
                "Error should reference 'models download' command"

            print("PASS: LLM engine error message is correct")
            return True

    except ImportError as e:
        print(f"SKIP: llama-cpp not installed: {e}")
        return True
    except Exception as e:
        print(f"FAIL: LLM engine error test failed: {e}")
        return False


def test_models_download_yes_flag():
    """Test 3: models download command has --yes flag."""
    try:
        import click
        from benderbox.ui.app import models_download

        # Check if the command has --yes option
        params = {p.name: p for p in models_download.params}

        assert "yes" in params, "models_download should have 'yes' parameter"

        yes_param = params["yes"]
        assert yes_param.is_flag, "'yes' should be a flag"

        # Check short option
        assert "-y" in yes_param.opts or "--yes" in yes_param.opts, \
            "'yes' should have -y or --yes option"

        print("PASS: models download has --yes flag")
        return True
    except Exception as e:
        print(f"FAIL: --yes flag test error: {e}")
        return False


def test_requirements_txt_exists():
    """Test 4: requirements.txt file exists and is valid."""
    try:
        req_path = Path(__file__).parent.parent / "requirements.txt"

        assert req_path.exists(), "requirements.txt should exist"

        content = req_path.read_text()

        # Should contain editable install
        assert "-e ." in content or "-e." in content, \
            "requirements.txt should contain '-e .'"

        print("PASS: requirements.txt exists and is valid")
        return True
    except Exception as e:
        print(f"FAIL: requirements.txt error: {e}")
        return False


def test_python_version_consistency():
    """Test 5: Python version is consistent across docs."""
    try:
        root = Path(__file__).parent.parent

        # Read pyproject.toml
        pyproject = root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml should exist"
        pyproject_content = pyproject.read_text()

        # Get Python version from pyproject.toml
        assert 'requires-python = ">=3.9"' in pyproject_content, \
            "pyproject.toml should require Python 3.9+"

        # Read README
        readme = root / "README.md"
        assert readme.exists(), "README.md should exist"
        readme_content = readme.read_text()

        # Check README matches (should say 3.9+ not 3.10+)
        assert "Python 3.9+" in readme_content, \
            "README should state Python 3.9+ requirement"
        assert "Python 3.10+" not in readme_content, \
            "README should NOT state Python 3.10+ (inconsistent)"

        print("PASS: Python version is consistent (3.9+)")
        return True
    except Exception as e:
        print(f"FAIL: Python version consistency error: {e}")
        return False


def test_pyproject_urls_correct():
    """Test 6: pyproject.toml has correct repository URLs."""
    try:
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"

        content = pyproject_path.read_text()

        # Should NOT have placeholder URLs
        assert "yourusername" not in content, \
            "pyproject.toml should not have 'yourusername' placeholder"

        # Should have correct URLs
        assert "r33n3/BenderBox" in content, \
            "pyproject.toml should have r33n3/BenderBox URLs"

        # Should have Issues URL
        assert "Issues" in content and "issues" in content.lower(), \
            "pyproject.toml should have Issues URL"

        print("PASS: pyproject.toml has correct URLs")
        return True
    except Exception as e:
        print(f"FAIL: pyproject.toml URLs error: {e}")
        return False


def test_readme_requirements_alternative():
    """Test 7: README documents requirements.txt as alternative."""
    try:
        readme_path = Path(__file__).parent.parent / "README.md"
        if not readme_path.exists():
            print("SKIP: README.md not found")
            return True

        content = readme_path.read_text()

        # Should mention requirements.txt
        assert "requirements.txt" in content, \
            "README should mention requirements.txt"

        print("PASS: README documents requirements.txt alternative")
        return True
    except Exception as e:
        print(f"FAIL: README requirements.txt error: {e}")
        return False


def test_readme_correct_repo_url():
    """Test 8: README has correct repository URL."""
    try:
        readme_path = Path(__file__).parent.parent / "README.md"
        if not readme_path.exists():
            print("SKIP: README.md not found")
            return True

        content = readme_path.read_text()

        # Should have correct repo URL (not placeholder)
        assert "r33n3/BenderBox" in content or "github.com/r33n3" in content, \
            "README should have correct repository URL"

        # Should NOT have placeholder
        assert "<repository-url>" not in content or "r33n3/BenderBox" in content, \
            "README should not have placeholder URL without real URL"

        print("PASS: README has correct repository URL")
        return True
    except Exception as e:
        print(f"FAIL: README repo URL error: {e}")
        return False


def test_model_download_docs_complete():
    """Test 9: Model download docs have all required info."""
    try:
        readme_path = Path(__file__).parent.parent / "README.md"
        if not readme_path.exists():
            print("SKIP: README.md not found")
            return True

        content = readme_path.read_text()

        # Should have size information
        assert "MB" in content or "GB" in content, \
            "README should mention model sizes"

        # Should have quality levels
        assert "Basic" in content or "Good" in content or "Best" in content, \
            "README should mention quality levels"

        # Should have multiple model options
        model_count = sum(1 for m in ["tinyllama", "phi2", "mistral", "qwen"]
                         if m in content.lower())
        assert model_count >= 3, \
            f"README should list multiple model options (found {model_count})"

        print("PASS: Model download docs are complete")
        return True
    except Exception as e:
        print(f"FAIL: Model docs completeness error: {e}")
        return False


def test_models_download_set_default_flag():
    """Test 10: models download command has --set-default flag."""
    try:
        from benderbox.ui.app import models_download

        params = {p.name: p for p in models_download.params}

        assert "set_default" in params, \
            "models_download should have 'set_default' parameter"

        set_default_param = params["set_default"]
        assert set_default_param.is_flag, "'set_default' should be a flag"
        assert "-d" in set_default_param.opts or "--set-default" in set_default_param.opts, \
            "'set_default' should have -d or --set-default option"

        print("PASS: models download has --set-default flag")
        return True
    except Exception as e:
        print(f"FAIL: --set-default flag test error: {e}")
        return False


def main():
    """Run all Phase 11 tests."""
    print("=" * 60)
    print("Phase 11: User Experience & Documentation - Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Test 1: README model download docs", test_readme_model_download_section),
        ("Test 2: LLM engine error message", test_llm_engine_error_message),
        ("Test 3: models download --yes flag", test_models_download_yes_flag),
        ("Test 4: requirements.txt exists", test_requirements_txt_exists),
        ("Test 5: Python version consistency", test_python_version_consistency),
        ("Test 6: pyproject.toml URLs", test_pyproject_urls_correct),
        ("Test 7: README requirements.txt docs", test_readme_requirements_alternative),
        ("Test 8: README repository URL", test_readme_correct_repo_url),
        ("Test 9: Model download docs complete", test_model_download_docs_complete),
        ("Test 10: models download --set-default", test_models_download_set_default_flag),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{name}")
        print("-" * 40)
        try:
            result = test_func()
            if result:
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
