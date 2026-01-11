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
        print("Missing dependencies. Install with:")
        print("    pip install -e .")
        print()
        print("Or install just the core dependencies:")
        print("    pip install pyyaml click rich httpx huggingface-hub")
        sys.exit(1)


if __name__ == "__main__":
    main()
