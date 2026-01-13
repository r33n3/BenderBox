"""
Phase 9: Critical Model Infrastructure - Test Suite

Tests for:
- Issue #12: cache_dir -> cache_path fix
- Issue #1: Directory auto-creation on init
- Issue #2: Base path defaults to ~/.benderbox/
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_model_manager_default_base_path():
    """Test 1: ModelManager defaults to ~/.benderbox/ not cwd."""
    try:
        from benderbox.utils.model_manager import ModelManager

        # Create without explicit path
        manager = ModelManager()

        # Should use home directory, not cwd
        expected_base = Path.home() / ".benderbox"
        assert manager.base_path == expected_base, \
            f"Expected {expected_base}, got {manager.base_path}"

        print("PASS: ModelManager defaults to ~/.benderbox/")
        return True
    except Exception as e:
        print(f"FAIL: Default base path error: {e}")
        return False


def test_model_manager_explicit_base_path():
    """Test 2: ModelManager respects explicit base path."""
    try:
        from benderbox.utils.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = Path(tmpdir) / "custom_benderbox"
            manager = ModelManager(base_path=custom_path)

            assert manager.base_path == custom_path, \
                f"Expected {custom_path}, got {manager.base_path}"

            print("PASS: ModelManager respects explicit base path")
            return True
    except Exception as e:
        print(f"FAIL: Explicit base path error: {e}")
        return False


def test_directories_created_on_init():
    """Test 3: All required directories are created on init."""
    try:
        from benderbox.utils.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_benderbox"
            manager = ModelManager(base_path=base_path)

            # Check all directories exist
            expected_dirs = [
                manager.base_path,
                manager.models_dir,
                manager.data_models_dir,
                manager.analysis_model_dir,
                manager.code_model_dir,
                manager.nlp_model_dir,
            ]

            for dir_path in expected_dirs:
                assert dir_path.exists(), f"Directory not created: {dir_path}"
                assert dir_path.is_dir(), f"Not a directory: {dir_path}"

            print(f"PASS: All {len(expected_dirs)} directories created on init")
            return True
    except Exception as e:
        print(f"FAIL: Directory creation error: {e}")
        return False


def test_nlp_model_dir_exists():
    """Test 4: NLP model directory is defined and created."""
    try:
        from benderbox.utils.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_benderbox"
            manager = ModelManager(base_path=base_path)

            # Check nlp_model_dir attribute exists
            assert hasattr(manager, "nlp_model_dir"), "Missing nlp_model_dir attribute"
            assert manager.nlp_model_dir == base_path / "models" / "nlp"
            assert manager.nlp_model_dir.exists()

            print("PASS: NLP model directory exists")
            return True
    except Exception as e:
        print(f"FAIL: NLP model dir error: {e}")
        return False


def test_ensure_directories_method():
    """Test 5: _ensure_directories method exists and works."""
    try:
        from benderbox.utils.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_benderbox"
            manager = ModelManager(base_path=base_path)

            # Check method exists
            assert hasattr(manager, "_ensure_directories"), \
                "Missing _ensure_directories method"

            # Call it again (should be idempotent)
            manager._ensure_directories()

            # Directories should still exist
            assert manager.models_dir.exists()

            print("PASS: _ensure_directories method works")
            return True
    except Exception as e:
        print(f"FAIL: _ensure_directories error: {e}")
        return False


def test_model_source_handler_cache_path():
    """Test 6: ModelSourceHandler uses cache_path parameter."""
    try:
        from benderbox.utils.model_source import ModelSourceHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache"

            # Should work with cache_path parameter
            handler = ModelSourceHandler(cache_path=cache_path)

            assert handler.cache_path == cache_path
            assert handler.cache_path.exists()

            print("PASS: ModelSourceHandler uses cache_path parameter")
            return True
    except TypeError as e:
        if "cache_dir" in str(e):
            print(f"FAIL: Still using old 'cache_dir' parameter: {e}")
        else:
            print(f"FAIL: TypeError: {e}")
        return False
    except Exception as e:
        print(f"FAIL: ModelSourceHandler error: {e}")
        return False


def test_model_manager_download_method_signature():
    """Test 7: download_model uses correct ModelSourceHandler init."""
    try:
        from benderbox.utils.model_manager import ModelManager
        import inspect

        # Get the source code of download_model
        source = inspect.getsource(ModelManager.download_model)

        # Check it uses cache_path, not cache_dir
        assert "cache_path=" in source, "Should use cache_path parameter"
        assert "cache_dir=" not in source, "Should not use old cache_dir parameter"

        print("PASS: download_model uses correct parameter name")
        return True
    except Exception as e:
        print(f"FAIL: Method signature check error: {e}")
        return False


def test_model_manager_directory_structure():
    """Test 8: Directory structure matches expected layout."""
    try:
        from benderbox.utils.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_benderbox"
            manager = ModelManager(base_path=base_path)

            # Check expected structure
            expected = {
                "base": base_path,
                "models": base_path / "models",
                "data/models": base_path / "data" / "models",
                "models/analysis": base_path / "models" / "analysis",
                "models/code": base_path / "models" / "code",
                "models/nlp": base_path / "models" / "nlp",
            }

            for name, path in expected.items():
                assert path.exists(), f"Missing: {name} at {path}"

            print("PASS: Directory structure is correct")
            return True
    except Exception as e:
        print(f"FAIL: Directory structure error: {e}")
        return False


def test_get_model_manager_function():
    """Test 9: get_model_manager helper uses correct defaults."""
    try:
        from benderbox.utils.model_manager import get_model_manager

        manager = get_model_manager()

        # Should use home directory
        expected = Path.home() / ".benderbox"
        assert manager.base_path == expected

        print("PASS: get_model_manager uses correct defaults")
        return True
    except Exception as e:
        print(f"FAIL: get_model_manager error: {e}")
        return False


def test_model_manager_idempotent_init():
    """Test 10: Multiple inits don't cause issues."""
    try:
        from benderbox.utils.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "test_benderbox"

            # Create multiple managers with same path
            m1 = ModelManager(base_path=base_path)
            m2 = ModelManager(base_path=base_path)
            m3 = ModelManager(base_path=base_path)

            # All should work fine
            assert m1.base_path == m2.base_path == m3.base_path
            assert m1.models_dir.exists()

            print("PASS: Multiple inits are idempotent")
            return True
    except Exception as e:
        print(f"FAIL: Idempotent init error: {e}")
        return False


def main():
    """Run all Phase 9 tests."""
    print("=" * 60)
    print("Phase 9: Critical Model Infrastructure - Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Test 1: Default base path is ~/.benderbox/", test_model_manager_default_base_path),
        ("Test 2: Explicit base path works", test_model_manager_explicit_base_path),
        ("Test 3: Directories created on init", test_directories_created_on_init),
        ("Test 4: NLP model directory exists", test_nlp_model_dir_exists),
        ("Test 5: _ensure_directories method", test_ensure_directories_method),
        ("Test 6: ModelSourceHandler cache_path", test_model_source_handler_cache_path),
        ("Test 7: download_model parameter name", test_model_manager_download_method_signature),
        ("Test 8: Directory structure correct", test_model_manager_directory_structure),
        ("Test 9: get_model_manager defaults", test_get_model_manager_function),
        ("Test 10: Idempotent initialization", test_model_manager_idempotent_init),
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
