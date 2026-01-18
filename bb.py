#!/usr/bin/env python3
"""
BenderBox - Quick launcher with interactive mode support.

Usage:
    python bb.py                    # Start interactive chat
    python bb.py --interactive      # Same as above
    python bb.py -i                 # Same as above
    python bb.py analyze ./file     # Run specific command
    python bb.py --help             # Show all commands
"""

import os
import sys

# Suppress llama.cpp C++ level logging (context size warnings, etc.)
# Must be set before any llama_cpp import
os.environ.setdefault("LLAMA_LOG_LEVEL", "ERROR")

# Add src directory to path so imports work without installation
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def _suppress_llama_logs():
    """Suppress llama.cpp C++ warnings using log callback."""
    try:
        import ctypes
        import llama_cpp

        # Define the log callback type matching the C signature
        # typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
        log_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

        # Create a null callback that does nothing
        @log_callback_type
        def _null_log_callback(level, text, user_data):
            pass  # Suppress all llama.cpp log output

        # Keep reference to prevent garbage collection
        _suppress_llama_logs._callback_ref = _null_log_callback
        llama_cpp.llama_log_set(_null_log_callback, None)
    except (ImportError, AttributeError, OSError):
        pass


def main():
    # Suppress llama.cpp warnings early
    _suppress_llama_logs()
    # Check for interactive flags or no arguments
    if len(sys.argv) == 1 or sys.argv[1] in ("--interactive", "-i"):
        # Remove the flag if present so Click doesn't complain
        if len(sys.argv) > 1 and sys.argv[1] in ("--interactive", "-i"):
            sys.argv.pop(1)
        # Insert 'chat' command
        sys.argv.insert(1, "chat")

    try:
        from benderbox.ui.app import cli
        cli()
    except ImportError as e:
        print(f"Error: {e}")
        print()
        print("Missing dependencies. You can either:")
        print()
        print("  1. Run the setup wizard (recommended):")
        print("       python setup_wizard.py")
        print()
        print("  2. Install manually:")
        print("       pip install -e .")
        print()
        print("  3. Install with NLP features:")
        print("       pip install -e '.[nlp]'")
        print()

        # Check if setup_wizard.py exists and offer to run it
        setup_wizard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup_wizard.py")
        if os.path.exists(setup_wizard_path):
            try:
                response = input("Run setup wizard now? [Y/n]: ").strip().lower()
                if response != 'n':
                    import subprocess
                    subprocess.run([sys.executable, setup_wizard_path])
                    sys.exit(0)
            except (EOFError, KeyboardInterrupt):
                pass

        sys.exit(1)


if __name__ == "__main__":
    main()
