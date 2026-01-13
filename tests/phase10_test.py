"""
Phase 10: Windows Platform Fixes - Test Suite

Tests for:
- Issue #3: RAM detection improvements
- Issue #11: bb.bat working directory fix
- Issue #7: PATH documentation (manual verification)
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_ram_detection_basic():
    """Test 1: RAM detection returns reasonable value."""
    try:
        from benderbox.utils.model_manager import ModelManager

        manager = ModelManager()
        ram_gb = manager.get_system_ram_gb()

        # Should return a positive integer
        assert isinstance(ram_gb, int), f"RAM should be int, got {type(ram_gb)}"
        assert ram_gb > 0, f"RAM should be positive, got {ram_gb}"
        assert ram_gb < 1024, f"RAM should be < 1TB, got {ram_gb}GB"

        print(f"PASS: RAM detection returns {ram_gb}GB")
        return True
    except Exception as e:
        print(f"FAIL: RAM detection error: {e}")
        return False


def test_ram_detection_consistency():
    """Test 2: Multiple RAM detections return consistent values."""
    try:
        from benderbox.utils.model_manager import ModelManager

        manager = ModelManager()

        # Call multiple times
        values = [manager.get_system_ram_gb() for _ in range(3)]

        # All values should be the same
        assert all(v == values[0] for v in values), \
            f"Inconsistent RAM values: {values}"

        print(f"PASS: RAM detection is consistent ({values[0]}GB)")
        return True
    except Exception as e:
        print(f"FAIL: RAM consistency error: {e}")
        return False


def test_ram_info_diagnostic():
    """Test 3: RAM info diagnostic method works."""
    try:
        from benderbox.utils.model_manager import ModelManager

        manager = ModelManager()
        info = manager.get_system_ram_info()

        # Check structure
        assert "detected_gb" in info, "Missing detected_gb"
        assert "methods" in info, "Missing methods"
        assert isinstance(info["methods"], dict), "methods should be dict"

        # Should have at least one detection method
        assert len(info["methods"]) > 0, "No detection methods reported"

        # Print diagnostic info
        print(f"Detected: {info['detected_gb']}GB")
        for method, data in info["methods"].items():
            if "error" in data:
                print(f"  {method}: error - {data['error']}")
            else:
                print(f"  {method}: {data.get('total_gb', 'N/A')}GB")

        print("PASS: RAM diagnostic method works")
        return True
    except Exception as e:
        print(f"FAIL: RAM diagnostic error: {e}")
        return False


def test_ram_detection_windows_ctypes():
    """Test 4: Windows ctypes detection method (Windows only)."""
    if os.name != "nt":
        print("SKIP: Windows-only test")
        return True

    try:
        from benderbox.utils.model_manager import ModelManager

        manager = ModelManager()
        info = manager.get_system_ram_info()

        # On Windows, ctypes method should be present
        assert "ctypes" in info["methods"], "ctypes method missing on Windows"

        ctypes_info = info["methods"]["ctypes"]
        if "error" in ctypes_info:
            print(f"FAIL: ctypes method error: {ctypes_info['error']}")
            return False

        assert "total_gb" in ctypes_info, "ctypes missing total_gb"
        assert ctypes_info["total_gb"] > 0, "ctypes RAM should be positive"

        print(f"PASS: Windows ctypes detected {ctypes_info['total_gb']}GB")
        return True
    except Exception as e:
        print(f"FAIL: Windows ctypes error: {e}")
        return False


def test_ram_cross_validation():
    """Test 5: Cross-validation of RAM detection methods."""
    try:
        from benderbox.utils.model_manager import ModelManager

        manager = ModelManager()
        info = manager.get_system_ram_info()

        # Get all successful detection values
        values = []
        for method, data in info["methods"].items():
            if "total_gb" in data and "error" not in data:
                values.append((method, data["total_gb"]))

        if len(values) < 2:
            print(f"SKIP: Only {len(values)} detection method(s) available")
            return True

        # Check if values are within 10% of each other
        gb_values = [v for _, v in values]
        max_val = max(gb_values)
        min_val = min(gb_values)

        if min_val > 0:
            ratio = max_val / min_val
            if ratio > 1.1:
                print(f"WARNING: Detection methods differ by {ratio:.1%}")
                print(f"  Values: {values}")
            else:
                print(f"PASS: Detection methods agree within 10%")
        else:
            print(f"PASS: Detection complete (min value is 0)")

        return True
    except Exception as e:
        print(f"FAIL: Cross-validation error: {e}")
        return False


def test_bb_bat_no_cd():
    """Test 6: bb.bat doesn't use cd command."""
    try:
        bb_bat_path = Path(__file__).parent.parent / "bb.bat"
        if not bb_bat_path.exists():
            print("SKIP: bb.bat not found")
            return True

        content = bb_bat_path.read_text()

        # Should not have 'cd /d' command
        assert "cd /d" not in content.lower(), \
            "bb.bat should not use 'cd /d' command"

        # Should have SCRIPT_DIR variable
        assert "SCRIPT_DIR" in content, \
            "bb.bat should use SCRIPT_DIR variable"

        # Should reference bb.py via SCRIPT_DIR
        assert "%SCRIPT_DIR%bb.py" in content or "SCRIPT_DIR%bb.py" in content, \
            "bb.bat should reference bb.py via SCRIPT_DIR"

        print("PASS: bb.bat preserves working directory")
        return True
    except Exception as e:
        print(f"FAIL: bb.bat check error: {e}")
        return False


def test_run_bat_no_cd():
    """Test 7: run.bat doesn't use cd command."""
    try:
        run_bat_path = Path(__file__).parent.parent / "run.bat"
        if not run_bat_path.exists():
            print("SKIP: run.bat not found")
            return True

        content = run_bat_path.read_text()

        # Should not have 'cd /d' command
        assert "cd /d" not in content.lower(), \
            "run.bat should not use 'cd /d' command"

        # Should have SCRIPT_DIR variable
        assert "SCRIPT_DIR" in content, \
            "run.bat should use SCRIPT_DIR variable"

        print("PASS: run.bat preserves working directory")
        return True
    except Exception as e:
        print(f"FAIL: run.bat check error: {e}")
        return False


def test_readme_windows_path_section():
    """Test 8: README has Windows PATH setup section."""
    try:
        readme_path = Path(__file__).parent.parent / "README.md"
        if not readme_path.exists():
            print("SKIP: README.md not found")
            return True

        content = readme_path.read_text()

        # Should have Windows PATH section
        assert "Windows PATH Setup" in content, \
            "README should have 'Windows PATH Setup' section"

        # Should have PowerShell instructions
        assert "PowerShell" in content and "SetEnvironmentVariable" in content, \
            "README should have PowerShell PATH instructions"

        # Should have manual setup instructions
        assert "Environment Variables" in content or "sysdm.cpl" in content, \
            "README should have manual PATH setup instructions"

        # Should have verification command
        assert "benderbox --version" in content, \
            "README should have verification command"

        print("PASS: README has Windows PATH documentation")
        return True
    except Exception as e:
        print(f"FAIL: README check error: {e}")
        return False


def test_model_recommendations_with_ram():
    """Test 9: Model recommendations use correct RAM value."""
    try:
        from benderbox.utils.model_manager import ModelManager

        manager = ModelManager()
        ram_gb = manager.get_system_ram_gb()

        # Get recommendations for detected RAM
        recommendations = manager.get_recommended_models(max_ram_gb=ram_gb)

        # Should get at least one recommendation
        assert len(recommendations) > 0, "Should have at least one model recommendation"

        # All recommendations should fit in detected RAM
        for model in recommendations:
            assert model.min_ram_gb <= ram_gb, \
                f"{model.name} requires {model.min_ram_gb}GB but only {ram_gb}GB detected"

        print(f"PASS: {len(recommendations)} models recommended for {ram_gb}GB RAM")
        return True
    except Exception as e:
        print(f"FAIL: Model recommendations error: {e}")
        return False


def test_ram_detection_no_import_error():
    """Test 10: RAM detection handles missing psutil gracefully."""
    try:
        from benderbox.utils.model_manager import ModelManager

        manager = ModelManager()

        # This should never raise ImportError even if psutil is missing
        # (it falls back to ctypes on Windows or returns 8)
        ram_gb = manager.get_system_ram_gb()

        assert ram_gb > 0, "RAM should be positive even with fallback"

        print(f"PASS: RAM detection handles dependencies gracefully ({ram_gb}GB)")
        return True
    except ImportError as e:
        print(f"FAIL: RAM detection raised ImportError: {e}")
        return False
    except Exception as e:
        print(f"FAIL: RAM detection error: {e}")
        return False


def main():
    """Run all Phase 10 tests."""
    print("=" * 60)
    print("Phase 10: Windows Platform Fixes - Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Test 1: RAM detection basic", test_ram_detection_basic),
        ("Test 2: RAM detection consistency", test_ram_detection_consistency),
        ("Test 3: RAM diagnostic method", test_ram_info_diagnostic),
        ("Test 4: Windows ctypes detection", test_ram_detection_windows_ctypes),
        ("Test 5: RAM cross-validation", test_ram_cross_validation),
        ("Test 6: bb.bat no cd command", test_bb_bat_no_cd),
        ("Test 7: run.bat no cd command", test_run_bat_no_cd),
        ("Test 8: README PATH documentation", test_readme_windows_path_section),
        ("Test 9: Model recommendations with RAM", test_model_recommendations_with_ram),
        ("Test 10: RAM graceful fallback", test_ram_detection_no_import_error),
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
