#!/usr/bin/env python3
"""
BenderBox Setup Script

Cross-platform setup script for installing BenderBox and its prerequisites.
Can be run standalone or imported as a module.

Usage:
    python scripts/setup.py --all
    python scripts/setup.py --check
    python scripts/setup.py --install llama-cli
"""

import argparse
import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ANSI colors (disabled on Windows cmd without ANSI support)
class Colors:
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            ENABLED = True
        except Exception:
            ENABLED = False
    else:
        ENABLED = True

    if ENABLED:
        RED = "\033[0;31m"
        GREEN = "\033[0;32m"
        YELLOW = "\033[1;33m"
        CYAN = "\033[0;36m"
        NC = "\033[0m"
    else:
        RED = GREEN = YELLOW = CYAN = NC = ""


def print_header():
    """Print script header."""
    print()
    print(f"{Colors.CYAN}================================================{Colors.NC}")
    print(f"{Colors.CYAN}  BenderBox Setup Script ({platform.system()}){Colors.NC}")
    print(f"{Colors.CYAN}================================================{Colors.NC}")
    print()


def print_success(message: str):
    print(f"{Colors.GREEN}[OK]{Colors.NC} {message}")


def print_warning(message: str):
    print(f"{Colors.YELLOW}[!]{Colors.NC} {message}")


def print_error(message: str):
    print(f"{Colors.RED}[X]{Colors.NC} {message}")


def print_info(message: str):
    print(f"{Colors.CYAN}[*]{Colors.NC} {message}")


@dataclass
class CheckResult:
    """Result of a prerequisite check."""
    name: str
    installed: bool
    version: Optional[str] = None
    path: Optional[str] = None
    message: str = ""


def get_tools_dir() -> Path:
    """Get the BenderBox tools directory."""
    return Path.home() / ".benderbox" / "tools"


def check_python() -> CheckResult:
    """Check Python installation."""
    try:
        result = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            version = result.stdout.strip().replace("Python ", "")
            parts = version.split(".")
            major, minor = int(parts[0]), int(parts[1])

            if major >= 3 and minor >= 9:
                return CheckResult(
                    name="python",
                    installed=True,
                    version=version,
                    path=sys.executable,
                    message=f"Python {version} is installed",
                )
            else:
                return CheckResult(
                    name="python",
                    installed=False,
                    version=version,
                    message=f"Python {version} found, but 3.9+ is required",
                )
    except Exception as e:
        pass

    return CheckResult(
        name="python",
        installed=False,
        message="Python 3.9+ is required. Visit https://www.python.org/downloads/",
    )


def check_git() -> CheckResult:
    """Check Git installation."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            import re
            match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
            version = match.group(1) if match else "unknown"

            return CheckResult(
                name="git",
                installed=True,
                version=version,
                path=shutil.which("git"),
                message=f"Git {version} is installed",
            )
    except Exception:
        pass

    return CheckResult(
        name="git",
        installed=False,
        message="Git is not installed",
    )


def check_llama_cli() -> CheckResult:
    """Check llama-cli installation."""
    # Check tools directory
    tools_dir = get_tools_dir()
    if platform.system() == "Windows":
        llama_path = tools_dir / "llama-cli.exe"
    else:
        llama_path = tools_dir / "llama-cli"

    if llama_path.exists():
        return CheckResult(
            name="llama-cli",
            installed=True,
            path=str(llama_path),
            message="llama-cli is installed",
        )

    # Check PATH
    which_path = shutil.which("llama-cli")
    if which_path:
        return CheckResult(
            name="llama-cli",
            installed=True,
            path=which_path,
            message="llama-cli is installed",
        )

    return CheckResult(
        name="llama-cli",
        installed=False,
        message="llama-cli is not installed",
    )


def get_latest_llama_release() -> Optional[str]:
    """Get the latest llama.cpp release tag."""
    try:
        url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})

        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
            tag = data.get("tag_name", "").lstrip("v")
            return tag
    except Exception as e:
        print_error(f"Could not fetch latest release: {e}")
        return None


def download_file(url: str, dest: Path, show_progress: bool = True) -> bool:
    """Download a file with progress indication."""
    try:
        req = urllib.request.Request(url)

        with urllib.request.urlopen(req, timeout=300) as response:
            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if show_progress and total > 0:
                        percent = (downloaded / total) * 100
                        bar_len = 40
                        filled = int(bar_len * downloaded / total)
                        bar = "=" * filled + "-" * (bar_len - filled)
                        print(f"\r  [{bar}] {percent:.1f}%", end="", flush=True)

            if show_progress:
                print()  # Newline after progress bar

        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False


def install_llama_cli(force: bool = False) -> bool:
    """Install llama-cli."""
    print_info("Installing llama-cli...")

    # Check if already installed
    check = check_llama_cli()
    if check.installed and not force:
        print_success(f"llama-cli is already installed at {check.path}")
        return True

    # Get latest release
    release = get_latest_llama_release()
    if not release:
        return False

    print_info(f"Latest release: {release}")

    # Determine download URL
    system = platform.system().lower()
    arch = platform.machine().lower()

    if system == "windows":
        download_url = f"https://github.com/ggerganov/llama.cpp/releases/download/b{release}/llama-b{release}-bin-win-avx2-x64.zip"
        exe_name = "llama-cli.exe"
    elif system == "darwin":
        if arch == "arm64":
            download_url = f"https://github.com/ggerganov/llama.cpp/releases/download/b{release}/llama-b{release}-bin-macos-arm64.zip"
        else:
            download_url = f"https://github.com/ggerganov/llama.cpp/releases/download/b{release}/llama-b{release}-bin-macos-x64.zip"
        exe_name = "llama-cli"
    elif system == "linux":
        download_url = f"https://github.com/ggerganov/llama.cpp/releases/download/b{release}/llama-b{release}-bin-ubuntu-x64.zip"
        exe_name = "llama-cli"
    else:
        print_error(f"Unsupported platform: {system}")
        return False

    print_info(f"Downloading from {download_url}...")

    # Create tools directory
    tools_dir = get_tools_dir()
    tools_dir.mkdir(parents=True, exist_ok=True)

    # Download to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / "llama.zip"
        extract_path = tmpdir / "extracted"
        extract_path.mkdir()

        if not download_file(download_url, zip_path):
            return False

        print_info("Extracting archive...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)

        # Find llama-cli
        llama_exe = None
        for root, dirs, files in os.walk(extract_path):
            for f in files:
                if f == exe_name or f == "main" or f == "main.exe":
                    llama_exe = Path(root) / f
                    break
            if llama_exe:
                break

        if not llama_exe:
            print_error("Could not find llama-cli in archive")
            return False

        # Copy to tools directory
        dest_path = tools_dir / exe_name
        shutil.copy2(llama_exe, dest_path)

        # Make executable on Unix
        if system != "windows":
            dest_path.chmod(0o755)

        print_success(f"llama-cli installed to {dest_path}")

        # PATH suggestion
        print_info("To add to PATH:")
        if system == "windows":
            print(f"  $env:PATH += \";{tools_dir}\"")
        else:
            print(f"  export PATH=\"$PATH:{tools_dir}\"")

        return True


def install_python_deps() -> bool:
    """Install Python dependencies."""
    print_info("Installing Python dependencies...")

    try:
        # Find project root (where pyproject.toml is)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent

        if not (project_root / "pyproject.toml").exists():
            # Maybe we're running from a different location
            project_root = Path.cwd()

        if not (project_root / "pyproject.toml").exists():
            print_error("Could not find pyproject.toml")
            return False

        # Upgrade pip
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )

        # Install package with all extras
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", f"{project_root}[all,dev]"],
            check=True,
        )

        print_success("Python dependencies installed")
        return True

    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def check_all() -> Dict[str, CheckResult]:
    """Check all prerequisites."""
    return {
        "python": check_python(),
        "git": check_git(),
        "llama-cli": check_llama_cli(),
    }


def print_check_results(results: Dict[str, CheckResult]):
    """Print check results in a table format."""
    print()
    print(f"{'Package':<15} {'Status':<15} {'Version':<12} {'Path'}")
    print("-" * 70)

    for name, result in results.items():
        status = f"{Colors.GREEN}INSTALLED{Colors.NC}" if result.installed else f"{Colors.RED}NOT INSTALLED{Colors.NC}"
        version = result.version or "-"
        path = result.path or result.message

        # Truncate path if too long
        if len(path) > 35:
            path = "..." + path[-32:]

        print(f"{name:<15} {status:<24} {version:<12} {path}")

    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BenderBox Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py --check              Check all prerequisites
  python setup.py --all                Install all prerequisites
  python setup.py --install llama-cli  Install specific package
  python setup.py --install deps       Install Python dependencies
        """,
    )

    parser.add_argument(
        "--check", "-c",
        action="store_true",
        help="Check prerequisite status",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Install all prerequisites",
    )
    parser.add_argument(
        "--install", "-i",
        metavar="PACKAGE",
        help="Install specific package (llama-cli, deps)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reinstall",
    )

    args = parser.parse_args()

    print_header()

    # Default to check if no arguments
    if not any([args.check, args.all, args.install]):
        args.check = True

    if args.check:
        results = check_all()
        print_check_results(results)

        installed = sum(1 for r in results.values() if r.installed)
        total = len(results)

        if installed == total:
            print_success(f"All {total} prerequisites are installed!")
        else:
            print_warning(f"{installed}/{total} prerequisites are installed")

        return 0 if installed == total else 1

    if args.install:
        pkg = args.install.lower()

        if pkg == "llama-cli":
            success = install_llama_cli(force=args.force)
        elif pkg == "deps" or pkg == "dependencies":
            success = install_python_deps()
        else:
            print_error(f"Unknown package: {pkg}")
            print_info("Available packages: llama-cli, deps")
            return 1

        return 0 if success else 1

    if args.all:
        results = {}

        # Check/install Python
        check = check_python()
        if not check.installed:
            print_error("Python 3.9+ is required but not installed")
            print_info("Please install Python first: https://www.python.org/downloads/")
            return 1
        results["python"] = True

        # Check Git (optional)
        check = check_git()
        results["git"] = check.installed
        if not check.installed:
            print_warning("Git is not installed (optional)")

        # Install llama-cli
        results["llama-cli"] = install_llama_cli(force=args.force)

        # Install Python dependencies
        results["deps"] = install_python_deps()

        # Summary
        print()
        print(f"{Colors.CYAN}{'='*50}{Colors.NC}")
        print(f"{Colors.CYAN}  Installation Summary{Colors.NC}")
        print(f"{Colors.CYAN}{'='*50}{Colors.NC}")

        for name, success in results.items():
            if success:
                print_success(name)
            else:
                print_error(name)

        failed = sum(1 for s in results.values() if not s)
        if failed == 0:
            print()
            print_success("All prerequisites installed successfully!")
            return 0
        else:
            print()
            print_warning(f"{failed} installation(s) need attention")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
