#!/usr/bin/env python3
"""
BenderBox Sandbox - Interactive Menu Functions

This module contains all interactive menu functions for the self-contained
BenderBox sandbox CLI. Import these into benderbox_sandbox_cli.py.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def interactive_menu(log_dir: Path, test_registry: Dict, sandbox_analyze_fn) -> int:
    """
    Interactive menu for BenderBox Sandbox.
    Provides a self-contained interface for all operations.
    """
    while True:
        print("\n" + "=" * 70)
        print("BenderBox Model & Infrastructure Sandbox - Interactive Mode v2.0")
        print("=" * 70)
        print("\nMODEL ANALYSIS")
        print("  1. Analyze a GGUF model")
        print("  2. Query model metadata (natural language)")
        print("  3. Compare two models")
        print("\nINFRASTRUCTURE ANALYSIS (v2.0)")
        print("  4. Analyze MCP server security")
        print("  5. Analyze skill security")
        print("  6. Compare infrastructure components")
        print("\nREPORTS & INFORMATION")
        print("  7. View latest report")
        print("  8. View all reports")
        print("  9. List available tests")
        print("  10. Help & Documentation")
        print("\n  0. Exit")
        print()

        choice = input("Select option (0-10): ").strip()

        if choice == "0":
            print("\n[BenderBox] Goodbye!")
            return 0

        elif choice == "1":
            return analyze_model_interactive(log_dir, test_registry, sandbox_analyze_fn)

        elif choice == "2":
            query_metadata_interactive(log_dir)

        elif choice == "3":
            compare_models_interactive(log_dir)

        elif choice == "4":
            return analyze_mcp_server_interactive(log_dir, test_registry, sandbox_analyze_fn)

        elif choice == "5":
            return analyze_skill_interactive(log_dir, test_registry, sandbox_analyze_fn)

        elif choice == "6":
            compare_infrastructure_interactive(log_dir)

        elif choice == "7":
            view_latest_report(log_dir)

        elif choice == "8":
            view_all_reports(log_dir)

        elif choice == "9":
            list_tests_interactive(test_registry)

        elif choice == "10":
            show_help()

        else:
            print("\n[BenderBox] Invalid choice. Please try again.")


def analyze_mcp_server_interactive(log_dir: Path, test_registry: Dict, sandbox_analyze_fn) -> int:
    """Interactive MCP server analysis workflow."""
    print("\n" + "=" * 70)
    print("Analyze MCP Server Security")
    print("=" * 70)

    # Find available MCP servers
    servers_dir = Path(".")
    py_files = sorted(servers_dir.glob("*.py"))

    # Filter out known system files
    exclude = {"sandbox_cli.py", "interactive.py", "mcp_server.py",
               "dynamic_tests.py", "infrastructure_tests.py",
               "mcp_analyzer.py", "skill_analyzer.py", "__init__.py"}
    py_files = [f for f in py_files if f.name not in exclude]

    if not py_files:
        print(f"\n[BenderBox] No MCP server files found in {servers_dir}")
        print("Tip: Place MCP server .py files in the current directory")
        input("\nPress ENTER to continue...")
        return 0

    # Show available servers
    print("\nAvailable MCP servers:")
    for i, server in enumerate(py_files, 1):
        size_kb = server.stat().st_size / 1024
        print(f"  {i}. {server.name} ({size_kb:.1f} KB)")

    # Select server
    while True:
        server_choice = input(f"\nSelect server (1-{len(py_files)}) or 'q' to cancel: ").strip()
        if server_choice.lower() == 'q':
            return 0
        try:
            server_idx = int(server_choice) - 1
            if 0 <= server_idx < len(py_files):
                server_path = py_files[server_idx]
                break
            else:
                print(f"Invalid choice. Please enter 1-{len(py_files)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")

    # Select profile
    print("\nAvailable profiles:")
    print("  1. infra-quick    - Fast security scan (~10-20s)")
    print("  2. infra-standard - Standard security audit (~30-60s) [Recommended]")
    print("  3. infra-deep     - Comprehensive analysis (~2-5min)")

    while True:
        profile_choice = input("\nSelect profile (1-3) or 'q' to cancel: ").strip()
        if profile_choice.lower() == 'q':
            return 0
        if profile_choice == '1':
            profile = 'infra-quick'
            break
        elif profile_choice == '2':
            profile = 'infra-standard'
            break
        elif profile_choice == '3':
            profile = 'infra-deep'
            break
        else:
            print("Invalid choice. Please enter 1-3")

    # Confirm and run
    print("\n" + "-" * 70)
    print(f"MCP Server: {server_path.name}")
    print(f"Profile: {profile}")
    print("-" * 70)

    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\n[BenderBox] Analysis cancelled.")
        return 0

    # Run analysis
    print("\n[BenderBox] Starting security analysis...")
    return sandbox_analyze_fn(
        model_path=None,
        mcp_server_path=server_path,
        skill_path=None,
        profile=profile,
        log_dir=log_dir,
        tests_override=None,
        format_mode="both",
        no_fail_on_test_errors=False,
    )


def analyze_skill_interactive(log_dir: Path, test_registry: Dict, sandbox_analyze_fn) -> int:
    """Interactive skill analysis workflow."""
    print("\n" + "=" * 70)
    print("Analyze Skill Security")
    print("=" * 70)

    # Find available skills
    skills_dir = Path("skills")
    if not skills_dir.exists():
        print(f"\n[BenderBox] Skills directory not found: {skills_dir}")
        input("\nPress ENTER to continue...")
        return 0

    md_files = sorted(skills_dir.glob("*.md"))
    if not md_files:
        print(f"\n[BenderBox] No skill files found in {skills_dir}")
        input("\nPress ENTER to continue...")
        return 0

    # Show available skills
    print("\nAvailable skills:")
    for i, skill in enumerate(md_files, 1):
        size_kb = skill.stat().st_size / 1024
        print(f"  {i}. {skill.name} ({size_kb:.1f} KB)")

    # Select skill
    while True:
        skill_choice = input(f"\nSelect skill (1-{len(md_files)}) or 'q' to cancel: ").strip()
        if skill_choice.lower() == 'q':
            return 0
        try:
            skill_idx = int(skill_choice) - 1
            if 0 <= skill_idx < len(md_files):
                skill_path = md_files[skill_idx]
                break
            else:
                print(f"Invalid choice. Please enter 1-{len(md_files)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")

    # Select profile
    print("\nAvailable profiles:")
    print("  1. infra-quick    - Fast security check (~5-10s) [Recommended]")
    print("  2. infra-standard - Standard security audit (~15-30s)")
    print("  3. infra-deep     - Comprehensive analysis (~1-2min)")

    while True:
        profile_choice = input("\nSelect profile (1-3) or 'q' to cancel: ").strip()
        if profile_choice.lower() == 'q':
            return 0
        if profile_choice == '1':
            profile = 'infra-quick'
            break
        elif profile_choice == '2':
            profile = 'infra-standard'
            break
        elif profile_choice == '3':
            profile = 'infra-deep'
            break
        else:
            print("Invalid choice. Please enter 1-3")

    # Confirm and run
    print("\n" + "-" * 70)
    print(f"Skill: {skill_path.name}")
    print(f"Profile: {profile}")
    print("-" * 70)

    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\n[BenderBox] Analysis cancelled.")
        return 0

    # Run analysis
    print("\n[BenderBox] Starting security analysis...")
    return sandbox_analyze_fn(
        model_path=None,
        mcp_server_path=None,
        skill_path=skill_path,
        profile=profile,
        log_dir=log_dir,
        tests_override=None,
        format_mode="both",
        no_fail_on_test_errors=False,
    )


def compare_infrastructure_interactive(log_dir: Path):
    """Compare two infrastructure components side by side."""
    print("\n" + "=" * 70)
    print("Compare Infrastructure Security")
    print("=" * 70)

    if not log_dir.exists():
        print(f"\n[BenderBox] No reports directory: {log_dir}")
        input("\nPress ENTER to continue...")
        return

    # Find infrastructure reports
    reports = sorted(log_dir.glob("benderbox_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    infra_reports = []

    for report in reports:
        try:
            with report.open("r") as f:
                data = json.load(f)
            # Check if it's an infrastructure report
            if data.get("infrastructure"):
                infra_reports.append(report)
        except Exception:
            continue

    if len(infra_reports) < 2:
        print(f"\n[BenderBox] Need at least 2 infrastructure reports to compare. Found: {len(infra_reports)}")
        print("Tip: Run infrastructure analysis first (options 4 or 5)")
        input("\nPress ENTER to continue...")
        return

    print(f"\nAvailable infrastructure reports:\n")
    for i, report in enumerate(infra_reports, 1):
        try:
            with report.open("r") as f:
                data = json.load(f)
            infra = data.get("infrastructure", {})

            if "mcp_server" in infra:
                comp_name = infra["mcp_server"].get("filename", "Unknown")
                comp_type = "MCP Server"
            elif "skill" in infra:
                comp_name = infra["skill"].get("filename", "Unknown")
                comp_type = "Skill"
            else:
                comp_name = "Unknown"
                comp_type = "Unknown"

            risk = data.get("overall_risk", {}).get("level", "Unknown")
            profile = data.get("profile", "Unknown")
            print(f"  {i}. [{comp_type}] {comp_name} (profile: {profile}, risk: {risk})")
        except Exception:
            print(f"  {i}. {report.name} (unreadable)")

    # Select two components
    comp1_idx = None
    comp2_idx = None

    while comp1_idx is None:
        choice = input(f"\nSelect first component (1-{len(infra_reports)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(infra_reports):
                comp1_idx = idx
        except ValueError:
            pass

    while comp2_idx is None:
        choice = input(f"Select second component (1-{len(infra_reports)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(infra_reports) and idx != comp1_idx:
                comp2_idx = idx
            elif idx == comp1_idx:
                print("Please select a different component.")
        except ValueError:
            pass

    # Load and compare
    try:
        with infra_reports[comp1_idx].open("r") as f:
            data1 = json.load(f)
        with infra_reports[comp2_idx].open("r") as f:
            data2 = json.load(f)

        infra1 = data1.get("infrastructure", {})
        infra2 = data2.get("infrastructure", {})

        print("\n" + "=" * 70)
        print("Infrastructure Security Comparison")
        print("=" * 70)

        # Determine component names and types
        if "mcp_server" in infra1:
            name1 = infra1["mcp_server"].get("filename", "N/A")
            type1 = "MCP Server"
        elif "skill" in infra1:
            name1 = infra1["skill"].get("filename", "N/A")
            type1 = "Skill"
        else:
            name1 = "Unknown"
            type1 = "Unknown"

        if "mcp_server" in infra2:
            name2 = infra2["mcp_server"].get("filename", "N/A")
            type2 = "MCP Server"
        elif "skill" in infra2:
            name2 = infra2["skill"].get("filename", "N/A")
            type2 = "Skill"
        else:
            name2 = "Unknown"
            type2 = "Unknown"

        print(f"\n{'Property':<25} {'Component 1':<25} {'Component 2':<25}")
        print("-" * 75)
        print(f"{'Type':<25} {type1:<25} {type2:<25}")
        print(f"{'Name':<25} {name1:<25} {name2:<25}")

        risk1 = data1.get("overall_risk", {})
        risk2 = data2.get("overall_risk", {})

        print(f"{'Risk Level':<25} {risk1.get('level', 'N/A'):<25} {risk2.get('level', 'N/A'):<25}")
        print(f"{'Risk Score':<25} {risk1.get('score', 'N/A'):<25} {risk2.get('score', 'N/A'):<25}")

        # Count findings by severity
        findings1 = data1.get("findings", [])
        findings2 = data2.get("findings", [])

        critical1 = sum(1 for f in findings1 if f.get("severity") == "CRITICAL")
        critical2 = sum(1 for f in findings2 if f.get("severity") == "CRITICAL")
        high1 = sum(1 for f in findings1 if f.get("severity") == "HIGH")
        high2 = sum(1 for f in findings2 if f.get("severity") == "HIGH")
        medium1 = sum(1 for f in findings1 if f.get("severity") == "MEDIUM")
        medium2 = sum(1 for f in findings2 if f.get("severity") == "MEDIUM")

        print(f"{'Critical Findings':<25} {critical1:<25} {critical2:<25}")
        print(f"{'High Findings':<25} {high1:<25} {high2:<25}")
        print(f"{'Medium Findings':<25} {medium1:<25} {medium2:<25}")

        # Recommendation
        print("\n" + "=" * 70)
        print("Recommendation")
        print("=" * 70)

        score1 = risk1.get("score", 100)
        score2 = risk2.get("score", 100)

        if score1 < score2:
            print(f"\n✅ Component 1 ({name1}) is safer")
            print(f"   Risk score: {score1} vs {score2}")
            print(f"   Critical findings: {critical1} vs {critical2}")
        elif score2 < score1:
            print(f"\n✅ Component 2 ({name2}) is safer")
            print(f"   Risk score: {score2} vs {score1}")
            print(f"   Critical findings: {critical2} vs {critical1}")
        else:
            print(f"\n⚖️  Both components have similar risk levels")
            print(f"   Risk score: {score1} (both)")
            print(f"   Review specific findings to determine which is better for your use case")

    except Exception as e:
        print(f"[BenderBox] Error comparing components: {e}")

    input("\nPress ENTER to continue...")


def analyze_model_interactive(log_dir: Path, test_registry: Dict, sandbox_analyze_fn) -> int:
    """Interactive model analysis workflow."""
    print("\n" + "=" * 70)
    print("Analyze Model")
    print("=" * 70)

    # Find available models
    models_dir = Path("models")
    if not models_dir.exists():
        print(f"\n[BenderBox] Models directory not found: {models_dir}")
        input("\nPress ENTER to continue...")
        return 0

    gguf_files = sorted(models_dir.glob("*.gguf"))
    if not gguf_files:
        print(f"\n[BenderBox] No GGUF models found in {models_dir}")
        input("\nPress ENTER to continue...")
        return 0

    # Show available models
    print("\nAvailable models:")
    for i, model in enumerate(gguf_files, 1):
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"  {i}. {model.name} ({size_mb:.1f} MB)")

    # Select model
    while True:
        model_choice = input(f"\nSelect model (1-{len(gguf_files)}) or 'q' to cancel: ").strip()
        if model_choice.lower() == 'q':
            return 0
        try:
            model_idx = int(model_choice) - 1
            if 0 <= model_idx < len(gguf_files):
                model_path = gguf_files[model_idx]
                break
            else:
                print(f"Invalid choice. Please enter 1-{len(gguf_files)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")

    # Select profile
    print("\nAvailable profiles:")
    print("  1. quick    - Fast GGUF metadata only (~5-10s)")
    print("  2. standard - Common static tests (~10-15s) [Recommended]")
    print("  3. deep     - All available tests (~30s+)")
    print("  4. custom   - Choose specific tests")

    while True:
        profile_choice = input("\nSelect profile (1-4) or 'q' to cancel: ").strip()
        if profile_choice.lower() == 'q':
            return 0
        if profile_choice == '1':
            profile = 'quick'
            break
        elif profile_choice == '2':
            profile = 'standard'
            break
        elif profile_choice == '3':
            profile = 'deep'
            break
        elif profile_choice == '4':
            profile = 'custom'
            break
        else:
            print("Invalid choice. Please enter 1-4")

    # Custom test selection
    tests_override = None
    if profile == 'custom':
        print("\nAvailable tests:")
        for name, factory in test_registry.items():
            test = factory()
            print(f"  - {name} [{test.category}]")

        tests_input = input("\nEnter test names (comma-separated): ").strip()
        if tests_input:
            tests_override = [t.strip() for t in tests_input.split(",") if t.strip()]

    # Confirm and run
    print("\n" + "-" * 70)
    print(f"Model: {model_path.name}")
    print(f"Profile: {profile}")
    if tests_override:
        print(f"Tests: {', '.join(tests_override)}")
    print("-" * 70)

    confirm = input("\nProceed with analysis? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\n[BenderBox] Analysis cancelled.")
        return 0

    # Run analysis
    print("\n[BenderBox] Starting analysis...")
    return sandbox_analyze_fn(
        model_path=model_path,
        profile=profile,
        log_dir=log_dir,
        tests_override=tests_override,
        format_mode="both",
        no_fail_on_test_errors=False,
    )


def view_latest_report(log_dir: Path):
    """View the most recent sandbox report."""
    print("\n" + "=" * 70)
    print("Latest Sandbox Report")
    print("=" * 70)

    if not log_dir.exists():
        print(f"\n[BenderBox] No reports directory: {log_dir}")
        input("\nPress ENTER to continue...")
        return

    reports = sorted(log_dir.glob("benderbox_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        print(f"\n[BenderBox] No reports found in {log_dir}")
        input("\nPress ENTER to continue...")
        return

    latest = reports[0]
    display_report(latest)
    input("\nPress ENTER to continue...")


def view_all_reports(log_dir: Path):
    """View all sandbox reports."""
    print("\n" + "=" * 70)
    print("All Sandbox Reports")
    print("=" * 70)

    if not log_dir.exists():
        print(f"\n[BenderBox] No reports directory: {log_dir}")
        input("\nPress ENTER to continue...")
        return

    reports = sorted(log_dir.glob("benderbox_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        print(f"\n[BenderBox] No reports found in {log_dir}")
        input("\nPress ENTER to continue...")
        return

    print(f"\nFound {len(reports)} report(s):\n")
    for i, report in enumerate(reports, 1):
        try:
            with report.open("r") as f:
                data = json.load(f)
            model_name = data.get("model", {}).get("name", "Unknown")
            profile = data.get("profile", "Unknown")
            risk = data.get("overall_risk", {}).get("level", "Unknown")
            timestamp = report.stem.split("_")[1] if "_" in report.stem else "Unknown"
            print(f"  {i}. [{timestamp}] {model_name} (profile: {profile}, risk: {risk})")
        except Exception:
            print(f"  {i}. {report.name} (unreadable)")

    # Allow selection
    while True:
        choice = input(f"\nSelect report to view (1-{len(reports)}) or 'q' to go back: ").strip()
        if choice.lower() == 'q':
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(reports):
                display_report(reports[idx])
                input("\nPress ENTER to continue...")
                return
            else:
                print(f"Invalid choice. Please enter 1-{len(reports)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")


def display_report(report_path: Path):
    """Display a formatted sandbox report."""
    try:
        with report_path.open("r") as f:
            data = json.load(f)

        print(f"\n{'=' * 70}")
        print(f"Report: {report_path.name}")
        print(f"{'=' * 70}\n")

        # Basic info
        print(f"Run ID: {data.get('run_id', 'N/A')}")
        print(f"Profile: {data.get('profile', 'N/A')}")
        print(f"Timestamp: {data.get('timestamp_utc', 'N/A')}")
        print()

        # Model info
        model = data.get("model", {})
        print("=== Model Information ===")
        print(f"Name: {model.get('name', 'N/A')}")
        print(f"Size: {model.get('size_bytes', 0) / (1024**3):.2f} GB")
        print(f"SHA256: {model.get('fingerprint', 'N/A')[:32]}...")
        print()

        # GGUF metadata
        metadata = model.get("metadata", {})
        if metadata and "architecture" in metadata:
            print("=== GGUF Metadata ===")
            print(f"Architecture: {metadata.get('architecture', 'N/A')}")
            print(f"Parameters: {metadata.get('parameter_count', 'N/A')}")
            print(f"Quantization: {metadata.get('quantization', 'N/A')}")
            ctx = metadata.get('context_length')
            if ctx:
                print(f"Context Length: {ctx:,} tokens")
            print(f"Layers: {metadata.get('layers', 'N/A')}")
            print(f"Embedding Dim: {metadata.get('embedding_length', 'N/A')}")
            print(f"Format: {metadata.get('format', 'N/A')}")
            print()

        # Risk assessment
        risk = data.get("overall_risk", {})
        print("=== Risk Assessment ===")
        print(f"Level: {risk.get('level', 'N/A')}")
        print(f"Score: {risk.get('score', 'N/A')}/100")
        factors = risk.get('primary_factors', [])
        if factors:
            print("Primary Factors:")
            for factor in factors:
                print(f"  - {factor}")
        print()

        # Test results
        print("=== Test Results ===")
        tests = data.get("tests", [])
        for test in tests:
            status = test['status']
            severity = test['severity']
            name = test['name']
            category = test['category']
            print(f"  [{status}/{severity}] {name} ({category})")

        print()
        print(f"Full report: {report_path}")

    except Exception as e:
        print(f"[BenderBox] Error reading report: {e}")


def query_metadata_interactive(log_dir: Path):
    """Interactive metadata query using natural language."""
    print("\n" + "=" * 70)
    print("Query Model Metadata (Natural Language)")
    print("=" * 70)

    if not log_dir.exists():
        print(f"\n[BenderBox] No reports directory: {log_dir}")
        input("\nPress ENTER to continue...")
        return

    reports = sorted(log_dir.glob("benderbox_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        print(f"\n[BenderBox] No reports found in {log_dir}")
        input("\nPress ENTER to continue...")
        return

    latest = reports[0]

    try:
        with latest.open("r") as f:
            data = json.load(f)

        metadata = data.get("model", {}).get("metadata", {})
        model_name = data.get("model", {}).get("name", "Unknown")

        print(f"\nQuerying metadata for: {model_name}")
        print("\nExamples:")
        print("  - What architecture is this model?")
        print("  - How many parameters?")
        print("  - What's the context length?")
        print("  - Is it quantized?")
        print("  - Can it run on 8GB VRAM?")
        print("\nType 'quit' to exit.\n")

        while True:
            query = input("Query: ").strip()
            if query.lower() in ['quit', 'q', 'exit']:
                break

            if not query:
                continue

            # Simple keyword matching for common questions
            query_lower = query.lower()

            if 'architecture' in query_lower or 'arch' in query_lower:
                print(f"→ Architecture: {metadata.get('architecture', 'Unknown')}\n")

            elif 'parameter' in query_lower or 'param' in query_lower or 'size' in query_lower:
                print(f"→ Parameters: {metadata.get('parameter_count', 'Unknown')}\n")

            elif 'context' in query_lower or 'window' in query_lower:
                ctx = metadata.get('context_length', 'Unknown')
                print(f"→ Context Length: {ctx:,} tokens\n" if isinstance(ctx, int) else f"→ Context Length: {ctx}\n")

            elif 'quantiz' in query_lower or 'quant' in query_lower:
                quant = metadata.get('quantization', 'Unknown')
                bits = metadata.get('quantization_bits', 'Unknown')
                print(f"→ Quantization: {quant} ({bits} bits)\n")

            elif 'layer' in query_lower:
                print(f"→ Layers: {metadata.get('layers', 'Unknown')}\n")

            elif 'embed' in query_lower:
                print(f"→ Embedding Dimension: {metadata.get('embedding_length', 'Unknown')}\n")

            elif 'vocab' in query_lower:
                vocab = metadata.get('vocab_size', 'Unknown')
                print(f"→ Vocabulary Size: {vocab:,} tokens\n" if isinstance(vocab, int) else f"→ Vocabulary Size: {vocab}\n")

            elif 'vram' in query_lower or 'memory' in query_lower or 'gpu' in query_lower:
                params = metadata.get('parameter_count', '')
                quant = metadata.get('quantization', '')

                if '7B' in params or '8B' in params:
                    if 'Q4' in quant:
                        print(f"→ Estimated VRAM: ~4-6 GB (with {quant} quantization)\n")
                    elif 'Q8' in quant:
                        print(f"→ Estimated VRAM: ~8-10 GB (with {quant} quantization)\n")
                    else:
                        print(f"→ Estimated VRAM: ~4-10 GB depending on quantization\n")
                elif '13B' in params:
                    print(f"→ Estimated VRAM: ~12-16 GB\n")
                elif '70B' in params or '65B' in params:
                    print(f"→ Estimated VRAM: ~40-80 GB\n")
                else:
                    print(f"→ Cannot estimate VRAM for {params} parameters\n")

            elif 'what' in query_lower and 'inside' in query_lower:
                # "What's inside this file?"
                print(f"→ This is a {metadata.get('architecture', 'unknown').upper()} model with:")
                print(f"   - Parameters: {metadata.get('parameter_count', 'Unknown')}")
                print(f"   - Quantization: {metadata.get('quantization', 'Unknown')}")
                ctx = metadata.get('context_length', 'Unknown')
                print(f"   - Context: {ctx:,} tokens" if isinstance(ctx, int) else f"   - Context: {ctx}")
                print(f"   - Format: {metadata.get('format', 'Unknown')}\n")

            else:
                print("→ I'm not sure how to answer that. Try asking about:")
                print("   architecture, parameters, context length, quantization,")
                print("   layers, embedding, vocabulary, or VRAM requirements.\n")

    except Exception as e:
        print(f"[BenderBox] Error querying metadata: {e}")

    input("\nPress ENTER to continue...")


def compare_models_interactive(log_dir: Path):
    """Compare two models side by side."""
    print("\n" + "=" * 70)
    print("Compare Models")
    print("=" * 70)

    if not log_dir.exists():
        print(f"\n[BenderBox] No reports directory: {log_dir}")
        input("\nPress ENTER to continue...")
        return

    reports = sorted(log_dir.glob("benderbox_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(reports) < 2:
        print(f"\n[BenderBox] Need at least 2 reports to compare. Found: {len(reports)}")
        input("\nPress ENTER to continue...")
        return

    print(f"\nAvailable reports:\n")
    for i, report in enumerate(reports, 1):
        try:
            with report.open("r") as f:
                data = json.load(f)
            model_name = data.get("model", {}).get("name", "Unknown")
            profile = data.get("profile", "Unknown")
            print(f"  {i}. {model_name} (profile: {profile})")
        except Exception:
            print(f"  {i}. {report.name} (unreadable)")

    # Select two models
    model1_idx = None
    model2_idx = None

    while model1_idx is None:
        choice = input(f"\nSelect first model (1-{len(reports)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(reports):
                model1_idx = idx
        except ValueError:
            pass

    while model2_idx is None:
        choice = input(f"Select second model (1-{len(reports)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(reports) and idx != model1_idx:
                model2_idx = idx
            elif idx == model1_idx:
                print("Please select a different model.")
        except ValueError:
            pass

    # Load and compare
    try:
        with reports[model1_idx].open("r") as f:
            data1 = json.load(f)
        with reports[model2_idx].open("r") as f:
            data2 = json.load(f)

        m1 = data1.get("model", {})
        m2 = data2.get("model", {})
        meta1 = m1.get("metadata", {})
        meta2 = m2.get("metadata", {})

        print("\n" + "=" * 70)
        print("Model Comparison")
        print("=" * 70)

        print(f"\n{'Property':<25} {'Model 1':<25} {'Model 2':<25}")
        print("-" * 75)
        print(f"{'Name':<25} {m1.get('name', 'N/A'):<25} {m2.get('name', 'N/A'):<25}")
        print(f"{'Size (GB)':<25} {m1.get('size_bytes', 0) / (1024**3):<25.2f} {m2.get('size_bytes', 0) / (1024**3):<25.2f}")
        print(f"{'Architecture':<25} {meta1.get('architecture', 'N/A'):<25} {meta2.get('architecture', 'N/A'):<25}")
        print(f"{'Parameters':<25} {meta1.get('parameter_count', 'N/A'):<25} {meta2.get('parameter_count', 'N/A'):<25}")
        print(f"{'Quantization':<25} {meta1.get('quantization', 'N/A'):<25} {meta2.get('quantization', 'N/A'):<25}")

        ctx1 = meta1.get('context_length', 'N/A')
        ctx2 = meta2.get('context_length', 'N/A')
        ctx1_str = f"{ctx1:,}" if isinstance(ctx1, int) else str(ctx1)
        ctx2_str = f"{ctx2:,}" if isinstance(ctx2, int) else str(ctx2)
        print(f"{'Context Length':<25} {ctx1_str:<25} {ctx2_str:<25}")

        print(f"{'Layers':<25} {meta1.get('layers', 'N/A'):<25} {meta2.get('layers', 'N/A'):<25}")
        print(f"{'Embedding Dim':<25} {meta1.get('embedding_length', 'N/A'):<25} {meta2.get('embedding_length', 'N/A'):<25}")

        risk1 = data1.get("overall_risk", {}).get("level", "N/A")
        risk2 = data2.get("overall_risk", {}).get("level", "N/A")
        print(f"{'Risk Level':<25} {risk1:<25} {risk2:<25}")

    except Exception as e:
        print(f"[BenderBox] Error comparing models: {e}")

    input("\nPress ENTER to continue...")


def list_tests_interactive(test_registry: Dict):
    """List available tests interactively."""
    print("\nAvailable tests:")
    for name, factory in test_registry.items():
        test = factory()
        print(f"  - {name} [{test.category}]")
    input("\nPress ENTER to continue...")


def show_help():
    """Display help and documentation."""
    print("\n" + "=" * 70)
    print("BenderBox Sandbox v2.0 - Help & Documentation")
    print("=" * 70)

    help_text = """
OVERVIEW
--------
BenderBox Sandbox is an AI Model & Infrastructure security analysis tool:
  - GGUF model analysis - metadata, safety, capabilities
  - MCP server security analysis - command injection, data exfiltration
  - Skill security analysis - prompt injection, credential harvesting
  - Detailed security reporting in JSON format

MODEL ANALYSIS PROFILES
-----------------------
  quick    - Fast GGUF metadata only (~5-10s)
             Use when: You want to know "what's inside this file?"

  standard - Common static tests (~10-15s) [Recommended]
             Use when: Normal model analysis before use

  deep     - All available tests (~30s+)
             Use when: Comprehensive pre-deployment validation

  attack   - Security-focused testing (~45-60s)
             Use when: Adversarial testing

INFRASTRUCTURE ANALYSIS PROFILES (v2.0)
---------------------------------------
  infra-quick    - Fast security scan (~10-20s)
                   Use when: Quick security check

  infra-standard - Standard security audit (~30-60s) [Recommended]
                   Use when: Pre-deployment validation

  infra-deep     - Comprehensive analysis (~2-5min)
                   Use when: High-security environments

COMMON WORKFLOWS
----------------
MODEL ANALYSIS:
1. Quick model inspection:
   Menu > 1 > Select model > Profile 1 (quick)

2. Full model analysis:
   Menu > 1 > Select model > Profile 2 (standard)

3. Query model metadata:
   Menu > 2 > "What architecture is this model?"

INFRASTRUCTURE ANALYSIS (v2.0):
4. Check MCP server security:
   Menu > 4 > Select server > Profile 2 (infra-standard)

5. Check skill security:
   Menu > 5 > Select skill > Profile 1 (infra-quick)

6. Compare components:
   Menu > 6 > Select two infrastructure reports

VIEW RESULTS:
7. View latest report:
   Menu > 7

8. View all reports:
   Menu > 8

NATURAL LANGUAGE QUERIES
-------------------------
For models, you can ask questions like:
  - What architecture is this model?
  - How many parameters?
  - What's the context length?
  - Is it quantized?
  - Can it run on 8GB VRAM?

SECURITY FINDINGS
-----------------
Infrastructure analysis detects:
  - Command injection vulnerabilities
  - Data exfiltration patterns
  - Credential harvesting attempts
  - Prompt injection patterns
  - Backdoor patterns
  - Obfuscation techniques

FILES & LOCATIONS
-----------------
  Reports:      ./sandbox_logs/benderbox_*.json
  Models:       ./models/*.gguf
  Examples:     ./examples/ (skills, MCP servers, prompts)
  Docs:         ./README.md
                ./QUICK_REFERENCE.md
                ./INFRASTRUCTURE_ANALYSIS_GUIDE.md

COMMAND LINE USAGE
------------------
You can also run from command line:

  # Model analysis
  python benderbox_sandbox_cli.py --model model.gguf --profile quick

  # MCP server analysis (v2.0)
  python benderbox_sandbox_cli.py --mcp-server server.py --profile infra-standard

  # Skill analysis (v2.0)
  python benderbox_sandbox_cli.py --skill skill.md --profile infra-quick

  # View summary
  python benderbox_sandbox_cli.py --summary

  # List tests
  python benderbox_sandbox_cli.py --list-tests

  # Interactive mode (this menu)
  python benderbox_sandbox_cli.py --interactive

For more information, see the documentation files.
"""
    print(help_text)
    input("\nPress ENTER to continue...")
