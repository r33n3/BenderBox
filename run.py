#!/usr/bin/env python3
"""
BenderBox - AI Security Analysis Platform

Quick launcher script. Run from the project root:
    python run.py [command] [options]

Examples:
    python run.py --help
    python run.py chat
    python run.py interrogate openai:gpt-3.5-turbo
    python run.py config api-keys
    python run.py prerequisites check

For full installation, run: pip install -e .
Then use: benderbox [command]
"""

import sys
import os

# Add src directory to path so imports work without installation
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def main():
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
