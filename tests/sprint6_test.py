"""
Sprint 6: TUI Foundation - Test Suite

Tests for:
- L1.1: TUI Application Shell
- L1.2: Dashboard Screen
- L1.3: Analysis Screen
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_tui_module_import():
    """Test 1: TUI module can be imported."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE, main

        assert main is not None, "main function should exist"

        # TEXTUAL_AVAILABLE should be a boolean
        assert isinstance(TEXTUAL_AVAILABLE, bool), "TEXTUAL_AVAILABLE should be boolean"

        print(f"PASS: TUI module imports (Textual available: {TEXTUAL_AVAILABLE})")
        return True
    except ImportError as e:
        print(f"FAIL: TUI module import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: TUI module error: {e}")
        return False


def test_css_theme_exists():
    """Test 2: CSS theme is defined."""
    try:
        from benderbox.ui.tui import BENDERBOX_CSS

        assert BENDERBOX_CSS is not None, "CSS theme should exist"
        assert len(BENDERBOX_CSS) > 100, "CSS theme should have content"

        # Check for key CSS elements
        assert "Screen" in BENDERBOX_CSS, "CSS should style Screen"
        assert "Header" in BENDERBOX_CSS, "CSS should style Header"
        assert "Footer" in BENDERBOX_CSS, "CSS should style Footer"
        assert ".widget" in BENDERBOX_CSS, "CSS should have widget class"
        assert ".dashboard-grid" in BENDERBOX_CSS, "CSS should have dashboard grid"

        print("PASS: CSS theme exists with required styles")
        return True
    except ImportError as e:
        print(f"FAIL: CSS theme import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: CSS theme error: {e}")
        return False


def test_benderbox_tui_class():
    """Test 3: BenderBoxTUI class exists with correct structure."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - BenderBoxTUI class test skipped")
            return True

        from benderbox.ui.tui import BenderBoxTUI

        assert BenderBoxTUI is not None, "BenderBoxTUI should exist"

        # Check class attributes
        assert hasattr(BenderBoxTUI, "TITLE"), "Should have TITLE"
        assert hasattr(BenderBoxTUI, "CSS"), "Should have CSS"
        assert hasattr(BenderBoxTUI, "BINDINGS"), "Should have BINDINGS"
        assert hasattr(BenderBoxTUI, "SCREENS"), "Should have SCREENS"

        # Check screens are registered
        screens = BenderBoxTUI.SCREENS
        assert "dashboard" in screens, "Should have dashboard screen"
        assert "analysis" in screens, "Should have analysis screen"
        assert "reports" in screens, "Should have reports screen"
        assert "chat" in screens, "Should have chat screen"

        print("PASS: BenderBoxTUI class has correct structure")
        return True
    except ImportError as e:
        print(f"FAIL: BenderBoxTUI import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: BenderBoxTUI error: {e}")
        return False


def test_dashboard_screen():
    """Test 4: DashboardScreen class exists."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - DashboardScreen test skipped")
            return True

        from benderbox.ui.tui import DashboardScreen

        assert DashboardScreen is not None, "DashboardScreen should exist"
        assert hasattr(DashboardScreen, "compose"), "Should have compose method"
        assert hasattr(DashboardScreen, "BINDINGS"), "Should have keybindings"

        print("PASS: DashboardScreen class exists")
        return True
    except ImportError as e:
        print(f"FAIL: DashboardScreen import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: DashboardScreen error: {e}")
        return False


def test_analysis_screen():
    """Test 5: AnalysisScreen class exists."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - AnalysisScreen test skipped")
            return True

        from benderbox.ui.tui import AnalysisScreen

        assert AnalysisScreen is not None, "AnalysisScreen should exist"
        assert hasattr(AnalysisScreen, "compose"), "Should have compose method"
        assert hasattr(AnalysisScreen, "run_analysis"), "Should have run_analysis method"

        print("PASS: AnalysisScreen class exists")
        return True
    except ImportError as e:
        print(f"FAIL: AnalysisScreen import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: AnalysisScreen error: {e}")
        return False


def test_reports_screen():
    """Test 6: ReportsScreen class exists."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ReportsScreen test skipped")
            return True

        from benderbox.ui.tui import ReportsScreen

        assert ReportsScreen is not None, "ReportsScreen should exist"
        assert hasattr(ReportsScreen, "compose"), "Should have compose method"

        print("PASS: ReportsScreen class exists")
        return True
    except ImportError as e:
        print(f"FAIL: ReportsScreen import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: ReportsScreen error: {e}")
        return False


def test_chat_screen():
    """Test 7: ChatScreen class exists."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ChatScreen test skipped")
            return True

        from benderbox.ui.tui import ChatScreen

        assert ChatScreen is not None, "ChatScreen should exist"
        assert hasattr(ChatScreen, "compose"), "Should have compose method"
        assert hasattr(ChatScreen, "send_message"), "Should have send_message method"

        print("PASS: ChatScreen class exists")
        return True
    except ImportError as e:
        print(f"FAIL: ChatScreen import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: ChatScreen error: {e}")
        return False


def test_status_bar_widget():
    """Test 8: StatusBar widget exists."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - StatusBar test skipped")
            return True

        from benderbox.ui.tui import StatusBar

        assert StatusBar is not None, "StatusBar should exist"
        assert hasattr(StatusBar, "llm_status"), "Should have llm_status reactive"
        assert hasattr(StatusBar, "db_status"), "Should have db_status reactive"
        assert hasattr(StatusBar, "last_scan"), "Should have last_scan reactive"

        print("PASS: StatusBar widget exists")
        return True
    except ImportError as e:
        print(f"FAIL: StatusBar import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: StatusBar error: {e}")
        return False


def test_stats_widget():
    """Test 9: StatsWidget exists."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - StatsWidget test skipped")
            return True

        from benderbox.ui.tui import StatsWidget

        assert StatsWidget is not None, "StatsWidget should exist"
        assert hasattr(StatsWidget, "update_stats"), "Should have update_stats method"

        print("PASS: StatsWidget exists")
        return True
    except ImportError as e:
        print(f"FAIL: StatsWidget import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: StatsWidget error: {e}")
        return False


def test_entry_point():
    """Test 10: TUI entry point is configured."""
    try:
        # Check pyproject.toml has the entry point
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            assert "benderbox-tui" in content, "Should have benderbox-tui entry point"
            assert "benderbox.ui.tui:main" in content, "Should point to tui:main"

        # Check main function exists and is callable
        from benderbox.ui.tui import main
        assert callable(main), "main should be callable"

        print("PASS: TUI entry point is configured")
        return True
    except Exception as e:
        print(f"FAIL: Entry point error: {e}")
        return False


def main():
    """Run all Sprint 6 tests."""
    print("=" * 60)
    print("Sprint 6: TUI Foundation - Test Suite")
    print("=" * 60)
    print()

    # Check if textual is available
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE
        print(f"Textual installed: {TEXTUAL_AVAILABLE}")
    except ImportError:
        print("Textual installed: False (import failed)")
    print()

    tests = [
        ("Test 1: TUI module import", test_tui_module_import),
        ("Test 2: CSS theme exists", test_css_theme_exists),
        ("Test 3: BenderBoxTUI class", test_benderbox_tui_class),
        ("Test 4: DashboardScreen class", test_dashboard_screen),
        ("Test 5: AnalysisScreen class", test_analysis_screen),
        ("Test 6: ReportsScreen class", test_reports_screen),
        ("Test 7: ChatScreen class", test_chat_screen),
        ("Test 8: StatusBar widget", test_status_bar_widget),
        ("Test 9: StatsWidget widget", test_stats_widget),
        ("Test 10: Entry point configured", test_entry_point),
    ]

    passed = 0
    failed = 0
    skipped = 0

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
