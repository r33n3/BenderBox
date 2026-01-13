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

import sys
import os

# Add src directory to path so imports work without installation
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def main():
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
