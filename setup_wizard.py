#!/usr/bin/env python3
"""
BenderBox Setup Wizard

Interactive terminal-based installer that guides users through
feature selection and dependency installation.

Usage:
    python setup_wizard.py
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Feature definitions
FEATURES = {
    "core": {
        "name": "Core (Required)",
        "description": "CLI, model downloads, basic analysis",
        "packages": ["pyyaml>=6.0", "aiosqlite>=0.19.0", "click>=8.0.0", "rich>=13.0.0", "httpx>=0.25.0", "huggingface-hub>=0.20.0"],
        "required": True,
        "python_min": (3, 9),
        "python_max": None,
    },
    "nlp": {
        "name": "NLP / Local LLM",
        "description": "Interactive chat with local GGUF models (TinyLlama, Mistral, etc.)",
        "packages": ["llama-cpp-python>=0.2.0"],
        "required": False,
        "python_min": (3, 9),
        "python_max": (3, 12),  # Pre-built wheels only up to 3.12
        "requires_build_tools": True,
        "build_tools_note": "Python 3.13+ requires Visual Studio Build Tools to compile",
    },
    "nlp_advanced": {
        "name": "NLP Advanced",
        "description": "Vector database and embeddings for knowledge base features",
        "packages": ["chromadb>=0.4.0", "sentence-transformers>=2.2.0"],
        "required": False,
        "python_min": (3, 9),
        "python_max": (3, 12),
        "depends_on": ["nlp"],
    },
    "mcp": {
        "name": "MCP Server",
        "description": "Model Context Protocol server for AI assistant integration",
        "packages": ["mcp>=0.1.0"],
        "required": False,
        "python_min": (3, 10),
        "python_max": None,
    },
    "tui": {
        "name": "Terminal UI",
        "description": "Rich terminal user interface with Textual",
        "packages": ["textual>=0.40.0"],
        "required": False,
        "python_min": (3, 9),
        "python_max": None,
    },
    "web": {
        "name": "Web Interface",
        "description": "FastAPI-based web interface",
        "packages": ["fastapi>=0.100.0", "uvicorn>=0.23.0"],
        "required": False,
        "python_min": (3, 9),
        "python_max": None,
    },
    "dev": {
        "name": "Development Tools",
        "description": "Testing, linting, and type checking tools",
        "packages": ["pytest>=7.0", "pytest-cov", "pytest-asyncio", "black", "flake8", "mypy"],
        "required": False,
        "python_min": (3, 9),
        "python_max": None,
    },
}


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def print_banner():
    """Print the BenderBox banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
 ██████╗ ███████╗███╗   ██╗██████╗ ███████╗██████╗ ██████╗  ██████╗ ██╗  ██╗
 ██╔══██╗██╔════╝████╗  ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔═══██╗╚██╗██╔╝
 ██████╔╝█████╗  ██╔██╗ ██║██║  ██║█████╗  ██████╔╝██████╔╝██║   ██║ ╚███╔╝
 ██╔══██╗██╔══╝  ██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗██╔══██╗██║   ██║ ██╔██╗
 ██████╔╝███████╗██║ ╚████║██████╔╝███████╗██║  ██║██████╔╝╚██████╔╝██╔╝ ██╗
 ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝
{Colors.END}
{Colors.BOLD}                    Setup Wizard v1.0{Colors.END}
                 AI Security Analysis Platform
"""
    print(banner)


def get_python_version():
    """Get the current Python version as a tuple."""
    return (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)


def check_python_version():
    """Check Python version and return compatibility info."""
    version = get_python_version()
    version_str = f"{version[0]}.{version[1]}.{version[2]}"

    issues = []
    warnings = []

    if version < (3, 9):
        issues.append(f"Python {version_str} is not supported. Minimum required: 3.9")
    elif version >= (3, 13):
        warnings.append(f"Python {version_str} detected. Some packages may require compilation.")
        warnings.append("Pre-built wheels for llama-cpp-python are only available for Python 3.12 and earlier.")
    elif version >= (3, 9) and version < (3, 11):
        warnings.append(f"Python {version_str} detected. Python 3.11 or 3.12 recommended for best compatibility.")

    return {
        "version": version,
        "version_str": version_str,
        "issues": issues,
        "warnings": warnings,
        "supported": version >= (3, 9),
        "optimal": (3, 11) <= version <= (3, 12, 99),
    }


def check_build_tools():
    """Check if Visual Studio Build Tools are available on Windows."""
    if platform.system() != 'Windows':
        return {"available": True, "message": "Not Windows, build tools check skipped"}

    # Check for cl.exe (MSVC compiler)
    cl_path = shutil.which("cl")
    if cl_path:
        return {"available": True, "message": f"MSVC compiler found: {cl_path}"}

    # Check for common VS Build Tools paths
    vs_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC",
    ]

    for path in vs_paths:
        if os.path.exists(path):
            return {"available": True, "message": f"Build Tools found at: {path}"}

    return {
        "available": False,
        "message": "Visual Studio Build Tools not found",
        "install_cmd": 'winget install Microsoft.VisualStudio.2022.BuildTools\ncurl -L -o vs_buildtools.exe "https://aka.ms/vs/17/release/vs_buildtools.exe"\n.\\vs_buildtools.exe --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait'
    }


def check_package_installed(package_name):
    """Check if a package is installed."""
    # Extract package name without version specifier
    name = package_name.split(">=")[0].split("==")[0].split("<")[0].replace("-", "_")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def install_packages(packages, verbose=True):
    """Install a list of packages using pip."""
    if not packages:
        return True, []

    failed = []
    for package in packages:
        if verbose:
            print(f"  Installing {package}...", end=" ", flush=True)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "--quiet"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per package
            )

            if result.returncode == 0:
                if verbose:
                    print(f"{Colors.GREEN}OK{Colors.END}")
            else:
                if verbose:
                    print(f"{Colors.RED}FAILED{Colors.END}")
                failed.append((package, result.stderr))
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"{Colors.RED}TIMEOUT{Colors.END}")
            failed.append((package, "Installation timed out"))
        except Exception as e:
            if verbose:
                print(f"{Colors.RED}ERROR{Colors.END}")
            failed.append((package, str(e)))

    return len(failed) == 0, failed


def print_system_check(python_info, build_tools_info):
    """Print system compatibility check results."""
    print(f"\n{Colors.BOLD}System Check{Colors.END}")
    print("=" * 50)

    # Python version
    if python_info["optimal"]:
        status = f"{Colors.GREEN}OPTIMAL{Colors.END}"
    elif python_info["supported"]:
        status = f"{Colors.YELLOW}SUPPORTED{Colors.END}"
    else:
        status = f"{Colors.RED}NOT SUPPORTED{Colors.END}"

    print(f"Python Version: {python_info['version_str']} [{status}]")

    for issue in python_info["issues"]:
        print(f"  {Colors.RED}ERROR:{Colors.END} {issue}")

    for warning in python_info["warnings"]:
        print(f"  {Colors.YELLOW}WARNING:{Colors.END} {warning}")

    # Build tools (Windows only)
    if platform.system() == 'Windows':
        if build_tools_info["available"]:
            print(f"Build Tools: {Colors.GREEN}Available{Colors.END}")
        else:
            print(f"Build Tools: {Colors.YELLOW}Not Found{Colors.END}")
            print(f"  {Colors.YELLOW}NOTE:{Colors.END} Required for compiling llama-cpp-python on Python 3.13+")

    print()


def print_feature_menu(python_info, build_tools_info):
    """Print the feature selection menu with compatibility info."""
    print(f"{Colors.BOLD}Available Features{Colors.END}")
    print("=" * 50)

    feature_status = {}

    for idx, (key, feature) in enumerate(FEATURES.items(), 1):
        # Check compatibility
        compatible = True
        notes = []

        py_version = python_info["version"][:2]

        if feature.get("python_min") and py_version < feature["python_min"]:
            compatible = False
            notes.append(f"Requires Python {feature['python_min'][0]}.{feature['python_min'][1]}+")

        if feature.get("python_max") and py_version > feature["python_max"]:
            if feature.get("requires_build_tools"):
                if not build_tools_info["available"]:
                    notes.append(feature.get("build_tools_note", "May require compilation"))

        # Check dependencies
        if feature.get("depends_on"):
            notes.append(f"Requires: {', '.join(feature['depends_on'])}")

        feature_status[key] = {"compatible": compatible, "notes": notes}

        # Print feature
        if feature.get("required"):
            marker = f"{Colors.GREEN}[Required]{Colors.END}"
        elif not compatible:
            marker = f"{Colors.RED}[Incompatible]{Colors.END}"
        elif notes:
            marker = f"{Colors.YELLOW}[Warning]{Colors.END}"
        else:
            marker = ""

        print(f"\n  {Colors.CYAN}{idx}.{Colors.END} {Colors.BOLD}{feature['name']}{Colors.END} {marker}")
        print(f"     {feature['description']}")

        for note in notes:
            print(f"     {Colors.YELLOW}*{Colors.END} {note}")

    print()
    return feature_status


def get_user_selection(feature_status):
    """Get user's feature selection."""
    print(f"{Colors.BOLD}Select Features to Install{Colors.END}")
    print("=" * 50)
    print("Enter the numbers of features you want to install, separated by commas.")
    print("Example: 1,2,3 or 'all' for all compatible features, 'minimal' for core only")
    print()

    feature_keys = list(FEATURES.keys())

    while True:
        try:
            selection = input(f"{Colors.CYAN}Your selection: {Colors.END}").strip().lower()

            if selection == 'all':
                # Select all compatible features
                selected = [key for key, status in feature_status.items()
                           if status["compatible"] or FEATURES[key].get("required")]
                break
            elif selection == 'minimal':
                selected = ["core"]
                break
            elif selection == 'recommended':
                selected = ["core", "nlp"]
                break
            else:
                # Parse comma-separated numbers
                indices = [int(x.strip()) for x in selection.split(",")]
                selected = []

                for idx in indices:
                    if 1 <= idx <= len(feature_keys):
                        key = feature_keys[idx - 1]
                        selected.append(key)
                    else:
                        print(f"{Colors.RED}Invalid selection: {idx}{Colors.END}")

                # Always include core
                if "core" not in selected:
                    selected.insert(0, "core")

                # Add dependencies
                for key in selected.copy():
                    deps = FEATURES[key].get("depends_on", [])
                    for dep in deps:
                        if dep not in selected:
                            selected.append(dep)
                            print(f"  {Colors.YELLOW}Adding dependency:{Colors.END} {FEATURES[dep]['name']}")

                break

        except ValueError:
            print(f"{Colors.RED}Invalid input. Please enter numbers separated by commas.{Colors.END}")
        except KeyboardInterrupt:
            print("\n\nInstallation cancelled.")
            sys.exit(0)

    return selected


def run_installation(selected_features, build_tools_info):
    """Run the installation for selected features."""
    print(f"\n{Colors.BOLD}Installing Selected Features{Colors.END}")
    print("=" * 50)

    all_packages = []
    for key in selected_features:
        feature = FEATURES[key]
        print(f"\n{Colors.CYAN}{feature['name']}{Colors.END}")

        # Check if build tools warning needed
        if feature.get("requires_build_tools") and not build_tools_info["available"]:
            py_version = get_python_version()[:2]
            if feature.get("python_max") and py_version > feature["python_max"]:
                print(f"  {Colors.YELLOW}WARNING:{Colors.END} This feature may require compilation.")
                print(f"  If installation fails, install Visual Studio Build Tools first:")
                print(f"  {Colors.CYAN}{build_tools_info.get('install_cmd', 'See README.md')}{Colors.END}")
                print()

        success, failed = install_packages(feature["packages"])

        if failed:
            print(f"\n  {Colors.RED}Some packages failed to install:{Colors.END}")
            for pkg, error in failed:
                print(f"    - {pkg}: {error[:100]}...")

    return True


def install_editable():
    """Install the package in editable mode."""
    print(f"\n{Colors.BOLD}Installing BenderBox in Development Mode{Colors.END}")
    print("=" * 50)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print(f"  {Colors.GREEN}BenderBox installed successfully!{Colors.END}")
            return True
        else:
            print(f"  {Colors.RED}Installation failed:{Colors.END}")
            print(result.stderr[:500])
            return False
    except Exception as e:
        print(f"  {Colors.RED}Error: {e}{Colors.END}")
        return False


def verify_installation():
    """Verify the installation by importing key modules."""
    print(f"\n{Colors.BOLD}Verifying Installation{Colors.END}")
    print("=" * 50)

    checks = [
        ("Core", "benderbox"),
        ("CLI", "benderbox.ui.app"),
        ("Config", "benderbox.config"),
    ]

    # Check for optional modules
    optional_checks = [
        ("NLP/LLM", "llama_cpp"),
        ("MCP", "mcp"),
        ("TUI", "textual"),
        ("Web", "fastapi"),
    ]

    all_ok = True

    for name, module in checks:
        try:
            __import__(module)
            print(f"  {name}: {Colors.GREEN}OK{Colors.END}")
        except ImportError as e:
            print(f"  {name}: {Colors.RED}FAILED{Colors.END} - {e}")
            all_ok = False

    print(f"\n  {Colors.CYAN}Optional Features:{Colors.END}")
    for name, module in optional_checks:
        try:
            __import__(module)
            print(f"  {name}: {Colors.GREEN}Available{Colors.END}")
        except ImportError:
            print(f"  {name}: {Colors.YELLOW}Not Installed{Colors.END}")

    return all_ok


def offer_model_download():
    """Offer to download TinyLlama model if NLP is installed."""
    print(f"\n{Colors.BOLD}Model Download{Colors.END}")
    print("=" * 50)
    print(f"""
The NLP feature requires a GGUF model to work.
TinyLlama (~700MB) is recommended for getting started.
""")

    choice = input(f"{Colors.CYAN}Download TinyLlama now? [Y/n]: {Colors.END}").strip().lower()
    if choice != 'n':
        print(f"\n{Colors.CYAN}Downloading TinyLlama...{Colors.END}")
        print("This may take a few minutes depending on your connection.\n")

        try:
            result = subprocess.run(
                [sys.executable, "bb.py", "models", "download", "tinyllama"],
                timeout=600,  # 10 minute timeout
                cwd=Path(__file__).parent
            )

            if result.returncode == 0:
                print(f"\n{Colors.GREEN}TinyLlama downloaded successfully!{Colors.END}")
                return True
            else:
                print(f"\n{Colors.YELLOW}Download failed. You can try again later:{Colors.END}")
                print(f"  python bb.py models download tinyllama")
                return False
        except subprocess.TimeoutExpired:
            print(f"\n{Colors.RED}Download timed out.{Colors.END}")
            print(f"You can retry: python bb.py models download tinyllama")
            return False
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.END}")
            print(f"You can retry: python bb.py models download tinyllama")
            return False
    else:
        print(f"\n{Colors.YELLOW}Skipped.{Colors.END} Download later with:")
        print(f"  python bb.py models download tinyllama")
        return False


def print_next_steps(nlp_installed=False, model_downloaded=False):
    """Print next steps after installation."""
    print(f"\n{Colors.BOLD}Installation Complete!{Colors.END}")
    print("=" * 50)

    if nlp_installed and not model_downloaded:
        print(f"""
{Colors.YELLOW}Model Required:{Colors.END}
   Download a model to use the NLP features:
   {Colors.GREEN}python bb.py models download tinyllama{Colors.END}
""")

    print(f"""
{Colors.CYAN}Getting Started:{Colors.END}

1. Start interactive mode:
   {Colors.GREEN}python bb.py -i{Colors.END}

2. Or run directly:
   {Colors.GREEN}python bb.py --help{Colors.END}

{Colors.CYAN}Try These Examples:{Colors.END}
   python bb.py context analyze examples/prompts/risky_system_prompt.md
   python bb.py mcp analyze examples/mcp_servers/sample_vulnerable_server.py

{Colors.CYAN}Quick Commands:{Colors.END}
   python bb.py models list      - List available models
   python bb.py config api-keys  - Configure API keys
   python bb.py status           - Check system status

{Colors.YELLOW}Documentation:{Colors.END} See README.md for full usage guide
""")


def main():
    """Main setup wizard entry point."""
    clear_screen()
    print_banner()

    # System checks
    python_info = check_python_version()
    build_tools_info = check_build_tools()

    print_system_check(python_info, build_tools_info)

    # Check if Python is supported
    if not python_info["supported"]:
        print(f"{Colors.RED}Python version not supported. Please install Python 3.9 or later.{Colors.END}")
        sys.exit(1)

    # Show feature menu
    feature_status = print_feature_menu(python_info, build_tools_info)

    # Get user selection
    print(f"\n{Colors.BOLD}Installation Options:{Colors.END}")
    print(f"  {Colors.CYAN}all{Colors.END}         - Install all compatible features")
    print(f"  {Colors.CYAN}recommended{Colors.END} - Core + NLP (most users)")
    print(f"  {Colors.CYAN}minimal{Colors.END}     - Core only")
    print(f"  {Colors.CYAN}1,2,3{Colors.END}       - Select specific features by number")
    print()

    selected = get_user_selection(feature_status)

    print(f"\n{Colors.BOLD}Selected Features:{Colors.END}")
    for key in selected:
        print(f"  - {FEATURES[key]['name']}")

    # Confirm
    print()
    confirm = input(f"{Colors.CYAN}Proceed with installation? [Y/n]: {Colors.END}").strip().lower()
    if confirm == 'n':
        print("Installation cancelled.")
        sys.exit(0)

    # Run installation
    run_installation(selected, build_tools_info)

    # Install in editable mode
    install_editable()

    # Verify
    verify_installation()

    # Check if NLP was installed
    nlp_installed = "nlp" in selected
    model_downloaded = False

    # Offer model download if NLP was installed
    if nlp_installed:
        try:
            import llama_cpp
            model_downloaded = offer_model_download()
        except ImportError:
            print(f"\n{Colors.YELLOW}NLP installation may have failed.{Colors.END}")
            print("Try installing manually: pip install llama-cpp-python")

    # Next steps
    print_next_steps(nlp_installed=nlp_installed, model_downloaded=model_downloaded)


if __name__ == "__main__":
    main()
