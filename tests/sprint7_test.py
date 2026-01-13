"""
Sprint 7: TUI Completion - Test Suite

Tests for:
- L1.4: Reports Screen enhancements
- L1.5: Chat Screen enhancements
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_reports_screen_enhancements():
    """Test 1: ReportsScreen has enhanced features."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ReportsScreen enhancement test skipped")
            return True

        from benderbox.ui.tui import ReportsScreen

        assert ReportsScreen is not None, "ReportsScreen should exist"
        assert hasattr(ReportsScreen, "compose"), "Should have compose method"
        assert hasattr(ReportsScreen, "load_reports"), "Should have load_reports method"
        assert hasattr(ReportsScreen, "update_reports_table"), "Should have update_reports_table"
        assert hasattr(ReportsScreen, "show_report_details"), "Should have show_report_details"
        assert hasattr(ReportsScreen, "export_selected_report"), "Should have export_selected_report"

        print("PASS: ReportsScreen has enhanced features")
        return True
    except ImportError as e:
        print(f"FAIL: ReportsScreen import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: ReportsScreen error: {e}")
        return False


def test_reports_screen_bindings():
    """Test 2: ReportsScreen has correct keybindings."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ReportsScreen bindings test skipped")
            return True

        from benderbox.ui.tui import ReportsScreen

        bindings = ReportsScreen.BINDINGS
        binding_keys = [b.key for b in bindings]

        assert "escape" in binding_keys, "Should have escape binding"
        assert "s" in binding_keys, "Should have search binding"
        assert "e" in binding_keys, "Should have export binding"
        assert "r" in binding_keys, "Should have refresh binding"

        print("PASS: ReportsScreen has correct keybindings")
        return True
    except Exception as e:
        print(f"FAIL: ReportsScreen bindings error: {e}")
        return False


def test_reports_screen_filter_attribute():
    """Test 3: ReportsScreen has filter attributes."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ReportsScreen filter test skipped")
            return True

        from benderbox.ui.tui import ReportsScreen

        # Check class has expected attributes
        assert hasattr(ReportsScreen, "reports"), "Should have reports list"
        assert hasattr(ReportsScreen, "selected_report"), "Should have selected_report"
        assert hasattr(ReportsScreen, "current_filter"), "Should have current_filter"

        print("PASS: ReportsScreen has filter attributes")
        return True
    except Exception as e:
        print(f"FAIL: ReportsScreen filter error: {e}")
        return False


def test_chat_screen_enhancements():
    """Test 4: ChatScreen has enhanced features."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ChatScreen enhancement test skipped")
            return True

        from benderbox.ui.tui import ChatScreen

        assert ChatScreen is not None, "ChatScreen should exist"
        assert hasattr(ChatScreen, "compose"), "Should have compose method"
        assert hasattr(ChatScreen, "send_message"), "Should have send_message method"
        assert hasattr(ChatScreen, "process_message"), "Should have process_message"
        assert hasattr(ChatScreen, "handle_command"), "Should have handle_command"
        assert hasattr(ChatScreen, "update_context_display"), "Should have update_context_display"

        print("PASS: ChatScreen has enhanced features")
        return True
    except ImportError as e:
        print(f"FAIL: ChatScreen import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: ChatScreen error: {e}")
        return False


def test_chat_screen_commands():
    """Test 5: ChatScreen has command definitions."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ChatScreen commands test skipped")
            return True

        from benderbox.ui.tui import ChatScreen

        assert hasattr(ChatScreen, "COMMANDS"), "Should have COMMANDS list"
        commands = ChatScreen.COMMANDS

        assert len(commands) > 0, "Should have at least one command"

        # Check for expected commands
        cmd_names = [c[0] for c in commands]
        assert "/help" in cmd_names, "Should have /help command"
        assert "/analyze" in cmd_names, "Should have /analyze command"
        assert "/search" in cmd_names, "Should have /search command"
        assert "/export" in cmd_names, "Should have /export command"

        print("PASS: ChatScreen has command definitions")
        return True
    except Exception as e:
        print(f"FAIL: ChatScreen commands error: {e}")
        return False


def test_chat_screen_suggestions():
    """Test 6: ChatScreen has suggested queries."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ChatScreen suggestions test skipped")
            return True

        from benderbox.ui.tui import ChatScreen

        assert hasattr(ChatScreen, "SUGGESTIONS"), "Should have SUGGESTIONS list"
        suggestions = ChatScreen.SUGGESTIONS

        assert len(suggestions) > 0, "Should have at least one suggestion"
        assert all(isinstance(s, str) for s in suggestions), "Suggestions should be strings"

        print("PASS: ChatScreen has suggested queries")
        return True
    except Exception as e:
        print(f"FAIL: ChatScreen suggestions error: {e}")
        return False


def test_chat_screen_context():
    """Test 7: ChatScreen has context tracking."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ChatScreen context test skipped")
            return True

        from benderbox.ui.tui import ChatScreen

        # Check context-related attributes
        assert hasattr(ChatScreen, "current_context"), "Should have current_context"
        assert hasattr(ChatScreen, "current_model"), "Should have current_model"
        assert hasattr(ChatScreen, "message_history"), "Should have message_history"
        assert hasattr(ChatScreen, "show_command_suggestions"), "Should have show_command_suggestions"

        print("PASS: ChatScreen has context tracking")
        return True
    except Exception as e:
        print(f"FAIL: ChatScreen context error: {e}")
        return False


def test_chat_screen_bindings():
    """Test 8: ChatScreen has correct keybindings."""
    try:
        from benderbox.ui.tui import TEXTUAL_AVAILABLE

        if not TEXTUAL_AVAILABLE:
            print("SKIP: Textual not installed - ChatScreen bindings test skipped")
            return True

        from benderbox.ui.tui import ChatScreen

        bindings = ChatScreen.BINDINGS
        binding_keys = [b.key for b in bindings]

        assert "escape" in binding_keys, "Should have escape binding"
        assert "ctrl+l" in binding_keys, "Should have clear binding"
        assert "ctrl+k" in binding_keys, "Should have focus input binding"
        assert "tab" in binding_keys, "Should have tab for suggestions"

        print("PASS: ChatScreen has correct keybindings")
        return True
    except Exception as e:
        print(f"FAIL: ChatScreen bindings error: {e}")
        return False


def test_css_chat_enhancements():
    """Test 9: CSS has chat enhancement styles."""
    try:
        from benderbox.ui.tui import BENDERBOX_CSS

        assert BENDERBOX_CSS is not None, "CSS should exist"

        # Check for chat-specific CSS
        assert "#chat-context" in BENDERBOX_CSS, "Should have chat-context style"
        assert "#chat-suggestions" in BENDERBOX_CSS, "Should have chat-suggestions style"
        assert "#command-suggestions" in BENDERBOX_CSS, "Should have command-suggestions style"
        assert ".suggestion-btn" in BENDERBOX_CSS, "Should have suggestion-btn style"
        assert ".context-indicator" in BENDERBOX_CSS, "Should have context-indicator style"

        print("PASS: CSS has chat enhancement styles")
        return True
    except Exception as e:
        print(f"FAIL: CSS chat enhancements error: {e}")
        return False


def test_css_reports_enhancements():
    """Test 10: CSS has reports enhancement styles."""
    try:
        from benderbox.ui.tui import BENDERBOX_CSS

        assert BENDERBOX_CSS is not None, "CSS should exist"

        # Check for reports-specific CSS
        assert "#reports-container" in BENDERBOX_CSS, "Should have reports-container style"
        assert "#reports-toolbar" in BENDERBOX_CSS, "Should have reports-toolbar style"
        assert "#reports-list" in BENDERBOX_CSS, "Should have reports-list style"
        assert "#report-detail-panel" in BENDERBOX_CSS, "Should have report-detail-panel style"
        assert ".finding-item" in BENDERBOX_CSS, "Should have finding-item style"

        print("PASS: CSS has reports enhancement styles")
        return True
    except Exception as e:
        print(f"FAIL: CSS reports enhancements error: {e}")
        return False


def main():
    """Run all Sprint 7 tests."""
    print("=" * 60)
    print("Sprint 7: TUI Completion - Test Suite")
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
        ("Test 1: ReportsScreen enhancements", test_reports_screen_enhancements),
        ("Test 2: ReportsScreen keybindings", test_reports_screen_bindings),
        ("Test 3: ReportsScreen filter attributes", test_reports_screen_filter_attribute),
        ("Test 4: ChatScreen enhancements", test_chat_screen_enhancements),
        ("Test 5: ChatScreen commands", test_chat_screen_commands),
        ("Test 6: ChatScreen suggestions", test_chat_screen_suggestions),
        ("Test 7: ChatScreen context tracking", test_chat_screen_context),
        ("Test 8: ChatScreen keybindings", test_chat_screen_bindings),
        ("Test 9: CSS chat enhancements", test_css_chat_enhancements),
        ("Test 10: CSS reports enhancements", test_css_reports_enhancements),
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
