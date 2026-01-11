"""
Prerequisites Management for BenderBox

Manages installation and verification of external tools and dependencies
required for BenderBox functionality.
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PackageStatus(Enum):
    """Status of a package installation."""

    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    OUTDATED = "outdated"
    VERSION_CONFLICT = "version_conflict"
    ERROR = "error"


class InstallMethod(Enum):
    """Package installation method."""

    BINARY = "binary"  # Pre-built binary download
    SOURCE = "source"  # Build from source
    PIP = "pip"  # Python package
    SYSTEM = "system"  # System package manager (apt, brew, choco)
    MANUAL = "manual"  # Manual installation required


@dataclass
class VersionInfo:
    """Version information for a package."""

    major: int = 0
    minor: int = 0
    patch: int = 0
    build: str = ""

    @classmethod
    def parse(cls, version_str: str) -> "VersionInfo":
        """Parse version string into VersionInfo."""
        if not version_str:
            return cls()

        # Clean version string
        version_str = version_str.strip().lstrip("v")

        # Try to parse semver-like format
        match = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[.-](.+))?", version_str)
        if match:
            return cls(
                major=int(match.group(1) or 0),
                minor=int(match.group(2) or 0),
                patch=int(match.group(3) or 0),
                build=match.group(4) or "",
            )

        return cls()

    def __str__(self) -> str:
        """Convert to version string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            version += f"-{self.build}"
        return version

    def __lt__(self, other: "VersionInfo") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: "VersionInfo") -> bool:
        return (self.major, self.minor, self.patch) <= (
            other.major,
            other.minor,
            other.patch,
        )

    def __gt__(self, other: "VersionInfo") -> bool:
        return (self.major, self.minor, self.patch) > (
            other.major,
            other.minor,
            other.patch,
        )

    def __ge__(self, other: "VersionInfo") -> bool:
        return (self.major, self.minor, self.patch) >= (
            other.major,
            other.minor,
            other.patch,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VersionInfo):
            return False
        return (self.major, self.minor, self.patch) == (
            other.major,
            other.minor,
            other.patch,
        )


@dataclass
class PackageDefinition:
    """Definition of an installable package."""

    name: str
    description: str
    required: bool = False
    version_command: str = ""  # Command to check version
    version_pattern: str = ""  # Regex to extract version
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    install_methods: Dict[str, "InstallConfig"] = field(default_factory=dict)
    check_command: str = ""  # Command to check if installed
    dependencies: List[str] = field(default_factory=list)

    def get_install_config(self, system: str) -> Optional["InstallConfig"]:
        """Get install config for current system."""
        # Try exact match first
        if system in self.install_methods:
            return self.install_methods[system]

        # Try platform-agnostic
        if "any" in self.install_methods:
            return self.install_methods["any"]

        return None


@dataclass
class InstallConfig:
    """Installation configuration for a specific platform."""

    method: InstallMethod
    url: str = ""  # Download URL (for binary/source)
    package_name: str = ""  # Package name (for pip/system)
    commands: List[str] = field(default_factory=list)  # Install commands
    post_install: List[str] = field(default_factory=list)  # Post-install commands
    env_vars: Dict[str, str] = field(default_factory=dict)  # Environment variables
    extract_path: str = ""  # Path within archive to extract
    copy_patterns: List[str] = field(default_factory=list)  # Additional patterns to copy (e.g., ["*.dll"])


@dataclass
class PackageCheckResult:
    """Result of checking a package's installation status."""

    package: str
    status: PackageStatus
    installed_version: Optional[VersionInfo] = None
    required_version: Optional[str] = None
    message: str = ""
    path: Optional[str] = None


@dataclass
class InstallResult:
    """Result of a package installation."""

    package: str
    success: bool
    message: str
    version: Optional[VersionInfo] = None
    path: Optional[str] = None


class PrerequisiteManager:
    """
    Manages BenderBox prerequisites and external dependencies.

    Provides:
    - Package detection and version checking
    - Automated installation from multiple sources
    - Version conflict detection
    - User-configurable package definitions
    """

    # Default package definitions
    DEFAULT_PACKAGES: Dict[str, PackageDefinition] = {
        "llama-cli": PackageDefinition(
            name="llama-cli",
            description="llama.cpp command-line tool for GGUF model inspection",
            required=False,
            version_command="llama-cli --version",
            version_pattern=r"version:\s*(\d+)",
            check_command="llama-cli --help",
            install_methods={
                "windows": InstallConfig(
                    method=InstallMethod.BINARY,
                    url="https://github.com/ggerganov/llama.cpp/releases/download/b{release}/llama-b{release}-bin-win-cpu-x64.zip",
                    extract_path="llama-cli.exe",
                    copy_patterns=["*.dll"],  # Copy all DLLs needed by llama-cli
                ),
                "linux": InstallConfig(
                    method=InstallMethod.BINARY,
                    url="https://github.com/ggerganov/llama.cpp/releases/download/b{release}/llama-b{release}-bin-ubuntu-x64.tar.gz",
                    extract_path="llama-cli",
                    copy_patterns=["*.so", "*.so.*"],  # Copy shared libraries
                ),
                "darwin": InstallConfig(
                    method=InstallMethod.BINARY,
                    url="https://github.com/ggerganov/llama.cpp/releases/download/b{release}/llama-b{release}-bin-macos-arm64.tar.gz",
                    extract_path="llama-cli",
                    copy_patterns=["*.dylib"],  # Copy dynamic libraries
                ),
            },
        ),
        "python": PackageDefinition(
            name="python",
            description="Python interpreter",
            required=True,
            version_command="python --version",
            version_pattern=r"Python\s+(\d+\.\d+\.\d+)",
            min_version="3.9.0",
            check_command="python --version",
            install_methods={
                "any": InstallConfig(
                    method=InstallMethod.MANUAL,
                    commands=["Visit https://www.python.org/downloads/"],
                ),
            },
        ),
        "git": PackageDefinition(
            name="git",
            description="Git version control",
            required=False,
            version_command="git --version",
            version_pattern=r"git version\s+(\d+\.\d+\.\d+)",
            check_command="git --version",
            install_methods={
                "windows": InstallConfig(
                    method=InstallMethod.SYSTEM,
                    package_name="git",
                    commands=["winget install --id Git.Git -e --source winget"],
                ),
                "linux": InstallConfig(
                    method=InstallMethod.SYSTEM,
                    package_name="git",
                    commands=["sudo apt-get install -y git"],
                ),
                "darwin": InstallConfig(
                    method=InstallMethod.SYSTEM,
                    package_name="git",
                    commands=["brew install git"],
                ),
            },
        ),
    }

    def __init__(
        self,
        tools_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize PrerequisiteManager.

        Args:
            tools_path: Path to store downloaded tools.
            config_path: Path to user package configuration.
        """
        self.tools_path = tools_path or Path.home() / ".benderbox" / "tools"
        self.config_path = config_path or Path.home() / ".benderbox" / "prerequisites.json"

        # Ensure paths exist
        self.tools_path.mkdir(parents=True, exist_ok=True)

        # Load packages (defaults + user-defined)
        self.packages = dict(self.DEFAULT_PACKAGES)
        self._load_user_packages()

        # Detect system
        self.system = self._detect_system()

    def _detect_system(self) -> str:
        """Detect the current operating system."""
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "darwin":
            return "darwin"
        elif system == "linux":
            return "linux"
        return "unknown"

    def _load_user_packages(self) -> None:
        """Load user-defined package configurations."""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path) as f:
                config = json.load(f)

            for name, pkg_data in config.get("packages", {}).items():
                # Parse install methods
                install_methods = {}
                for platform_name, method_data in pkg_data.get("install_methods", {}).items():
                    install_methods[platform_name] = InstallConfig(
                        method=InstallMethod(method_data.get("method", "manual")),
                        url=method_data.get("url", ""),
                        package_name=method_data.get("package_name", ""),
                        commands=method_data.get("commands", []),
                        post_install=method_data.get("post_install", []),
                        env_vars=method_data.get("env_vars", {}),
                        extract_path=method_data.get("extract_path", ""),
                    )

                self.packages[name] = PackageDefinition(
                    name=name,
                    description=pkg_data.get("description", ""),
                    required=pkg_data.get("required", False),
                    version_command=pkg_data.get("version_command", ""),
                    version_pattern=pkg_data.get("version_pattern", ""),
                    min_version=pkg_data.get("min_version"),
                    max_version=pkg_data.get("max_version"),
                    check_command=pkg_data.get("check_command", ""),
                    dependencies=pkg_data.get("dependencies", []),
                    install_methods=install_methods,
                )

            logger.info(f"Loaded {len(config.get('packages', {}))} user-defined packages")
        except Exception as e:
            logger.warning(f"Could not load user packages: {e}")

    def save_user_packages(self, packages: Dict[str, PackageDefinition]) -> None:
        """Save user-defined packages to config."""
        config = {"packages": {}}

        for name, pkg in packages.items():
            if name in self.DEFAULT_PACKAGES:
                continue  # Don't save defaults

            install_methods = {}
            for platform_name, method in pkg.install_methods.items():
                install_methods[platform_name] = {
                    "method": method.method.value,
                    "url": method.url,
                    "package_name": method.package_name,
                    "commands": method.commands,
                    "post_install": method.post_install,
                    "env_vars": method.env_vars,
                    "extract_path": method.extract_path,
                }

            config["packages"][name] = {
                "description": pkg.description,
                "required": pkg.required,
                "version_command": pkg.version_command,
                "version_pattern": pkg.version_pattern,
                "min_version": pkg.min_version,
                "max_version": pkg.max_version,
                "check_command": pkg.check_command,
                "dependencies": pkg.dependencies,
                "install_methods": install_methods,
            }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def check_package(self, package_name: str) -> PackageCheckResult:
        """
        Check if a package is installed and its version status.

        Args:
            package_name: Name of the package to check.

        Returns:
            PackageCheckResult with status information.
        """
        if package_name not in self.packages:
            return PackageCheckResult(
                package=package_name,
                status=PackageStatus.ERROR,
                message=f"Unknown package: {package_name}",
            )

        pkg = self.packages[package_name]

        # Check if binary exists in tools path
        tool_path = self._find_tool_path(package_name)

        # Try to get version
        version = None
        found_path = None

        if pkg.version_command:
            try:
                # Check tools path first
                if tool_path and tool_path.exists():
                    found_path = str(tool_path)
                    cmd = pkg.version_command.replace(package_name, str(tool_path))
                else:
                    cmd = pkg.version_command

                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                output = result.stdout + result.stderr

                if pkg.version_pattern:
                    match = re.search(pkg.version_pattern, output)
                    if match:
                        version = VersionInfo.parse(match.group(1))

                # If no version found but command succeeded, package exists
                if result.returncode == 0 and not found_path:
                    found_path = shutil.which(package_name.split()[0])

            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                logger.debug(f"Version check failed for {package_name}: {e}")

        # If version check failed, try simple existence check
        if not version and not found_path:
            if pkg.check_command:
                try:
                    # Check tools path first
                    if tool_path and tool_path.exists():
                        found_path = str(tool_path)
                        cmd = pkg.check_command.replace(package_name, str(tool_path))
                    else:
                        cmd = pkg.check_command

                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        found_path = found_path or shutil.which(package_name.split("-")[0])
                except Exception:
                    pass

        # Determine status
        if not found_path and not version:
            return PackageCheckResult(
                package=package_name,
                status=PackageStatus.NOT_INSTALLED,
                message=f"{package_name} is not installed",
            )

        # Check version constraints
        if version and pkg.min_version:
            min_ver = VersionInfo.parse(pkg.min_version)
            if version < min_ver:
                return PackageCheckResult(
                    package=package_name,
                    status=PackageStatus.OUTDATED,
                    installed_version=version,
                    required_version=pkg.min_version,
                    message=f"{package_name} {version} is older than required {pkg.min_version}",
                    path=found_path,
                )

        if version and pkg.max_version:
            max_ver = VersionInfo.parse(pkg.max_version)
            if version > max_ver:
                return PackageCheckResult(
                    package=package_name,
                    status=PackageStatus.VERSION_CONFLICT,
                    installed_version=version,
                    required_version=f"<= {pkg.max_version}",
                    message=f"{package_name} {version} is newer than max supported {pkg.max_version}",
                    path=found_path,
                )

        return PackageCheckResult(
            package=package_name,
            status=PackageStatus.INSTALLED,
            installed_version=version,
            message=f"{package_name} is installed" + (f" (version {version})" if version else ""),
            path=found_path,
        )

    def check_all_packages(self) -> List[PackageCheckResult]:
        """Check all registered packages."""
        results = []
        for name in self.packages:
            results.append(self.check_package(name))
        return results

    def _find_tool_path(self, package_name: str) -> Optional[Path]:
        """Find a tool in the tools directory."""
        # Check for executable
        if self.system == "windows":
            exe_name = f"{package_name}.exe"
        else:
            exe_name = package_name

        tool_path = self.tools_path / exe_name
        if tool_path.exists():
            return tool_path

        # Check subdirectories
        for subdir in self.tools_path.iterdir():
            if subdir.is_dir():
                tool_in_subdir = subdir / exe_name
                if tool_in_subdir.exists():
                    return tool_in_subdir

        return None

    async def install_package(
        self,
        package_name: str,
        force: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> InstallResult:
        """
        Install a package.

        Args:
            package_name: Name of the package to install.
            force: Force reinstall even if already installed.
            progress_callback: Callback for progress updates (message, percent).

        Returns:
            InstallResult with installation outcome.
        """
        if package_name not in self.packages:
            return InstallResult(
                package=package_name,
                success=False,
                message=f"Unknown package: {package_name}",
            )

        pkg = self.packages[package_name]

        # Check current status
        if not force:
            check = self.check_package(package_name)
            if check.status == PackageStatus.INSTALLED:
                return InstallResult(
                    package=package_name,
                    success=True,
                    message=f"{package_name} is already installed",
                    version=check.installed_version,
                    path=check.path,
                )

        # Get install config for this system
        install_config = pkg.get_install_config(self.system)
        if not install_config:
            return InstallResult(
                package=package_name,
                success=False,
                message=f"No installation method available for {package_name} on {self.system}",
            )

        if progress_callback:
            progress_callback(f"Installing {package_name}...", 0.0)

        try:
            if install_config.method == InstallMethod.BINARY:
                result = await self._install_binary(pkg, install_config, progress_callback)
            elif install_config.method == InstallMethod.PIP:
                result = await self._install_pip(pkg, install_config, progress_callback)
            elif install_config.method == InstallMethod.SYSTEM:
                result = await self._install_system(pkg, install_config, progress_callback)
            elif install_config.method == InstallMethod.MANUAL:
                return InstallResult(
                    package=package_name,
                    success=False,
                    message=f"Manual installation required: {', '.join(install_config.commands)}",
                )
            else:
                return InstallResult(
                    package=package_name,
                    success=False,
                    message=f"Unsupported install method: {install_config.method}",
                )

            # Run post-install commands
            if result.success and install_config.post_install:
                for cmd in install_config.post_install:
                    subprocess.run(cmd, shell=True)

            if progress_callback:
                progress_callback(f"{package_name} installation complete", 1.0)

            return result

        except Exception as e:
            logger.error(f"Installation failed for {package_name}: {e}")
            return InstallResult(
                package=package_name,
                success=False,
                message=f"Installation failed: {e}",
            )

    async def _install_binary(
        self,
        pkg: PackageDefinition,
        config: InstallConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> InstallResult:
        """Install a package from binary download."""
        import tarfile
        import urllib.request

        # Get latest release version for URL templating
        url = config.url
        if "{release}" in url:
            release = await self._get_latest_release("ggerganov/llama.cpp")
            if release:
                url = url.replace("{release}", release)
            else:
                return InstallResult(
                    package=pkg.name,
                    success=False,
                    message="Could not determine latest release version",
                )

        if progress_callback:
            progress_callback(f"Downloading {pkg.name}...", 0.1)

        # Determine file extension from URL
        if url.endswith(".tar.gz"):
            ext = ".tar.gz"
        elif url.endswith(".zip"):
            ext = ".zip"
        else:
            ext = Path(url).suffix or ".zip"

        # Download file
        with tempfile.TemporaryDirectory() as tmpdir:
            download_path = Path(tmpdir) / f"download{ext}"

            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "BenderBox/1.0"},
                )

                with urllib.request.urlopen(req, timeout=300) as response:
                    total = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(download_path, "wb") as f:
                        while True:
                            chunk = response.read(8192)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback and total:
                                progress = 0.1 + (downloaded / total) * 0.6
                                progress_callback(f"Downloading {pkg.name}...", progress)

            except Exception as e:
                return InstallResult(
                    package=pkg.name,
                    success=False,
                    message=f"Download failed: {e}",
                )

            if progress_callback:
                progress_callback(f"Extracting {pkg.name}...", 0.7)

            # Extract
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()

            try:
                if url.endswith(".tar.gz") or url.endswith(".tgz"):
                    with tarfile.open(download_path, "r:gz") as tf:
                        tf.extractall(extract_dir)
                elif url.endswith(".zip"):
                    with zipfile.ZipFile(download_path, "r") as zf:
                        zf.extractall(extract_dir)
                else:
                    # Move as-is for non-archive files
                    shutil.copy(download_path, extract_dir / download_path.name)
            except Exception as e:
                return InstallResult(
                    package=pkg.name,
                    success=False,
                    message=f"Extraction failed: {e}",
                )

            if progress_callback:
                progress_callback(f"Installing {pkg.name}...", 0.8)

            # Find and copy the target file
            target_name = config.extract_path or pkg.name
            if self.system == "windows" and not target_name.endswith(".exe"):
                target_name += ".exe"

            # Search for target file
            target_file = None
            for root, dirs, files in os.walk(extract_dir):
                for f in files:
                    if f == target_name or f == config.extract_path:
                        target_file = Path(root) / f
                        break
                if target_file:
                    break

            if not target_file:
                # List what was found
                found_files = list(extract_dir.rglob("*"))
                return InstallResult(
                    package=pkg.name,
                    success=False,
                    message=f"Could not find {target_name} in archive. Found: {[f.name for f in found_files[:10]]}",
                )

            # Copy to tools directory
            dest_path = self.tools_path / target_file.name
            shutil.copy2(target_file, dest_path)

            # Make executable on Unix
            if self.system != "windows":
                dest_path.chmod(0o755)

            # Copy additional files matching patterns (e.g., DLLs)
            if config.copy_patterns:
                import fnmatch
                copied_count = 0
                for root, dirs, files in os.walk(extract_dir):
                    for f in files:
                        for pattern in config.copy_patterns:
                            if fnmatch.fnmatch(f, pattern):
                                src_file = Path(root) / f
                                dst_file = self.tools_path / f
                                if not dst_file.exists() or dst_file.stat().st_mtime < src_file.stat().st_mtime:
                                    shutil.copy2(src_file, dst_file)
                                    copied_count += 1
                                break
                if copied_count > 0:
                    logger.info(f"Copied {copied_count} additional files for {pkg.name}")

            if progress_callback:
                progress_callback(f"{pkg.name} installed successfully", 1.0)

            # Verify installation
            check = self.check_package(pkg.name)

            return InstallResult(
                package=pkg.name,
                success=True,
                message=f"Installed {pkg.name} to {dest_path}",
                version=check.installed_version,
                path=str(dest_path),
            )

    async def _install_pip(
        self,
        pkg: PackageDefinition,
        config: InstallConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> InstallResult:
        """Install a package via pip."""
        package_name = config.package_name or pkg.name

        if progress_callback:
            progress_callback(f"Installing {package_name} via pip...", 0.3)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                return InstallResult(
                    package=pkg.name,
                    success=False,
                    message=f"pip install failed: {result.stderr}",
                )

            # Verify installation
            check = self.check_package(pkg.name)

            return InstallResult(
                package=pkg.name,
                success=True,
                message=f"Installed {package_name} via pip",
                version=check.installed_version,
            )

        except subprocess.TimeoutExpired:
            return InstallResult(
                package=pkg.name,
                success=False,
                message="Installation timed out",
            )

    async def _install_system(
        self,
        pkg: PackageDefinition,
        config: InstallConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> InstallResult:
        """Install a package via system package manager."""
        if not config.commands:
            return InstallResult(
                package=pkg.name,
                success=False,
                message="No system install commands defined",
            )

        if progress_callback:
            progress_callback(f"Installing {pkg.name} via system package manager...", 0.3)

        for cmd in config.commands:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                if result.returncode != 0:
                    return InstallResult(
                        package=pkg.name,
                        success=False,
                        message=f"Command failed: {cmd}\n{result.stderr}",
                    )

            except subprocess.TimeoutExpired:
                return InstallResult(
                    package=pkg.name,
                    success=False,
                    message=f"Command timed out: {cmd}",
                )

        # Verify installation
        check = self.check_package(pkg.name)

        return InstallResult(
            package=pkg.name,
            success=check.status == PackageStatus.INSTALLED,
            message=f"Installed {pkg.name} via system package manager",
            version=check.installed_version,
            path=check.path,
        )

    async def _get_latest_release(self, repo: str) -> Optional[str]:
        """Get latest release tag from GitHub."""
        import urllib.request

        try:
            url = f"https://api.github.com/repos/{repo}/releases/latest"
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "BenderBox/1.0",
                },
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                tag = data.get("tag_name", "").lstrip("vb")  # Remove 'v' or 'b' prefix
                return tag
        except Exception as e:
            logger.warning(f"Could not fetch latest release: {e}")

        return None

    def add_package(self, package: PackageDefinition) -> None:
        """Add or update a package definition."""
        self.packages[package.name] = package
        self._save_modified_packages()

    def remove_package(self, package_name: str) -> bool:
        """Remove a user-defined package."""
        if package_name in self.DEFAULT_PACKAGES:
            return False  # Can't remove defaults

        if package_name in self.packages:
            del self.packages[package_name]
            self._save_modified_packages()
            return True

        return False

    def _save_modified_packages(self) -> None:
        """Save non-default packages to config."""
        user_packages = {
            name: pkg
            for name, pkg in self.packages.items()
            if name not in self.DEFAULT_PACKAGES
        }
        self.save_user_packages(user_packages)

    def get_install_instructions(self, package_name: str) -> str:
        """Get human-readable install instructions for a package."""
        if package_name not in self.packages:
            return f"Unknown package: {package_name}"

        pkg = self.packages[package_name]
        config = pkg.get_install_config(self.system)

        if not config:
            return f"No installation method available for {package_name} on {self.system}"

        instructions = [f"# Installing {pkg.name}", f"# {pkg.description}", ""]

        if config.method == InstallMethod.BINARY:
            instructions.extend([
                "# Automatic installation available:",
                f"benderbox prerequisites install {package_name}",
                "",
                "# Or manual download:",
                f"# Download from: {config.url}",
                f"# Extract to: {self.tools_path}",
            ])
        elif config.method == InstallMethod.PIP:
            instructions.extend([
                f"pip install {config.package_name or package_name}",
            ])
        elif config.method == InstallMethod.SYSTEM:
            instructions.extend(config.commands)
        elif config.method == InstallMethod.MANUAL:
            instructions.extend(config.commands)

        return "\n".join(instructions)

    def generate_install_script(self, packages: Optional[List[str]] = None) -> str:
        """
        Generate a shell script to install prerequisites.

        Args:
            packages: List of package names, or None for all required packages.

        Returns:
            Shell script content.
        """
        if packages is None:
            packages = [name for name, pkg in self.packages.items() if pkg.required]

        if self.system == "windows":
            return self._generate_windows_script(packages)
        else:
            return self._generate_unix_script(packages)

    def _generate_windows_script(self, packages: List[str]) -> str:
        """Generate Windows batch/PowerShell script."""
        lines = [
            "@echo off",
            "REM BenderBox Prerequisites Installation Script",
            "REM Generated: " + datetime.now().isoformat(),
            "",
            "echo BenderBox Prerequisites Installer",
            "echo ================================",
            "",
        ]

        for pkg_name in packages:
            if pkg_name not in self.packages:
                continue

            pkg = self.packages[pkg_name]
            config = pkg.get_install_config("windows")

            lines.append(f"echo.")
            lines.append(f"echo Checking {pkg.name}...")

            if config:
                if config.method == InstallMethod.SYSTEM:
                    for cmd in config.commands:
                        lines.append(f"echo Running: {cmd}")
                        lines.append(cmd)
                elif config.method == InstallMethod.BINARY:
                    lines.append(f"echo Manual download required: {config.url}")
                elif config.method == InstallMethod.PIP:
                    lines.append(f"pip install {config.package_name or pkg_name}")

        lines.extend([
            "",
            "echo.",
            "echo Installation complete!",
            "pause",
        ])

        return "\n".join(lines)

    def _generate_unix_script(self, packages: List[str]) -> str:
        """Generate Unix shell script."""
        lines = [
            "#!/bin/bash",
            "# BenderBox Prerequisites Installation Script",
            f"# Generated: {datetime.now().isoformat()}",
            "",
            "set -e",
            "",
            'echo "BenderBox Prerequisites Installer"',
            'echo "================================"',
            "",
        ]

        for pkg_name in packages:
            if pkg_name not in self.packages:
                continue

            pkg = self.packages[pkg_name]
            config = pkg.get_install_config(self.system)

            lines.append(f'echo ""')
            lines.append(f'echo "Checking {pkg.name}..."')

            if pkg.check_command:
                lines.append(f"if command -v {pkg_name.split()[0]} &> /dev/null; then")
                lines.append(f'    echo "{pkg.name} is already installed"')
                lines.append("else")

                if config:
                    if config.method == InstallMethod.SYSTEM:
                        for cmd in config.commands:
                            lines.append(f"    {cmd}")
                    elif config.method == InstallMethod.BINARY:
                        lines.append(f'    echo "Manual download required: {config.url}"')
                    elif config.method == InstallMethod.PIP:
                        lines.append(f"    pip install {config.package_name or pkg_name}")

                lines.append("fi")

        lines.extend([
            "",
            'echo ""',
            'echo "Installation complete!"',
        ])

        return "\n".join(lines)


def create_default_config() -> Dict[str, Any]:
    """Create default prerequisites configuration."""
    return {
        "packages": {},
        "settings": {
            "tools_path": str(Path.home() / ".benderbox" / "tools"),
            "auto_install": False,
            "check_updates": True,
        },
    }
