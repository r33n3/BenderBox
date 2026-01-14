#!/usr/bin/env python3
"""
BenderBox Model Sandbox CLI

Profiles:
  - quick  : light static + minimal dynamic probes
  - deep   : full static + broad dynamic + jailbreak suite
  - attack : focused jailbreak / abuse testing
  - custom : specific tests via --tests

Exit codes:
  0 : pipeline executed successfully (even if some tests FAIL)
  1 : bad arguments / configuration error
  2 : runtime error (e.g., I/O, llama.cpp start failure)
  3 : internal bug / unhandled exception
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Callable, Any

# Import dynamic tests if available
try:
    from benderbox.dynamic_tests import BasicJailbreakTest, BackdoorDetectionTest
    DYNAMIC_TESTS_AVAILABLE = True
except ImportError:
    DYNAMIC_TESTS_AVAILABLE = False
    BasicJailbreakTest = None
    BackdoorDetectionTest = None

# Import infrastructure tests if available
try:
    from benderbox.infrastructure_tests import (
        MCPStaticAnalysisTest,
        SkillStaticAnalysisTest,
        InfrastructureComprehensiveTest
    )
    INFRASTRUCTURE_TESTS_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_TESTS_AVAILABLE = False
    MCPStaticAnalysisTest = None
    SkillStaticAnalysisTest = None
    InfrastructureComprehensiveTest = None

RUNNER_VERSION = "2.0.0"  # Infrastructure analysis extension
SCHEMA_VERSION = "0.2.0"  # Extended schema


# ---------- Data Models ----------

@dataclass
class TestResult:
    name: str
    category: str
    status: str = "PASS"   # PASS, FAIL, WARN, ERROR, SKIP
    severity: str = "INFO" # INFO, LOW, MEDIUM, HIGH, CRITICAL
    score: Optional[float] = None
    details: str = ""
    metrics: Optional[Dict[str, float]] = None
    artifacts: Optional[List[Dict[str, str]]] = None


@dataclass
class SandboxRunResult:
    schema_version: str
    run_id: str
    timestamp_utc: str
    profile: str
    model: Dict[str, Any]
    environment: Dict[str, Any]
    overall_risk: Dict[str, Any]
    capability_fingerprint: Dict[str, Any]
    safety: Dict[str, Any]
    tests: List[TestResult]
    errors: List[str]
    notes: str = ""


# ---------- Test Registry & Profiles ----------

class SandboxTest:
    """
    Base class for sandbox tests. Override `run` in subclasses.

    `context` will hold:
      - model_info
      - profile
      - global settings
      - utilities for llama.cpp calls (later)
    """

    name: str = "base_test"
    category: str = "generic"

    def run(self, context: Dict[str, Any]) -> TestResult:
        raise NotImplementedError


class DummyStaticMetadataTest(SandboxTest):
    """
    Example stub test to show wiring; replace with real GGUF metadata checks.
    """
    name = "static_metadata_basic"
    category = "static"

    def run(self, context: Dict[str, Any]) -> TestResult:
        model_info = context.get("model_info", {})
        path = model_info.get("path", "unknown")
        size_bytes = model_info.get("size_bytes", 0)

        details = f"Model path: {path}\nFile size: {size_bytes} bytes."
        metrics = {"size_mb": size_bytes / (1024 * 1024)} if size_bytes else {}

        return TestResult(
            name=self.name,
            category=self.category,
            status="PASS",
            severity="INFO",
            details=details,
            metrics=metrics,
        )


class GGUFMetadataSanityTest(SandboxTest):
    """
    GGUF-aware static test that inspects and validates GGUF metadata.

    This test:
    - Extracts GGUF metadata using llama-cli
    - Validates expected fields are present
    - Checks for potential issues (missing metadata, unsupported formats, etc.)
    - Summarizes key model characteristics
    - Records metadata for downstream analysis
    """
    name = "gguf_metadata_sanity"
    category = "static"

    def run(self, context: Dict[str, Any]) -> TestResult:
        model_info = context.get("model_info", {})
        metadata = model_info.get("metadata", {})
        path = model_info.get("path", "unknown")
        size_bytes = model_info.get("size_bytes", 0)

        # Check if metadata extraction succeeded
        if "error" in metadata:
            return TestResult(
                name=self.name,
                category=self.category,
                status="ERROR",
                severity="MEDIUM",
                details=f"Failed to extract GGUF metadata: {metadata.get('error')}\n"
                        f"Note: {metadata.get('note', 'N/A')}",
                metrics={"size_mb": size_bytes / (1024 * 1024)},
            )

        # Build a detailed summary of the model
        summary_parts = [
            f"Model: {model_info.get('name')}",
            f"Path: {path}",
            f"Size: {size_bytes / (1024 * 1024):.2f} MB",
            f"SHA256: {model_info.get('fingerprint', 'N/A')[:16]}...",
            "",
            "=== GGUF Metadata ===",
        ]

        # Key fields to report
        key_fields = [
            ("Architecture", "architecture"),
            ("Parameter Count", "parameter_count"),
            ("Quantization", "quantization"),
            ("Quantization Bits", "quantization_bits"),
            ("Context Length", "context_length"),
            ("Embedding Length", "embedding_length"),
            ("Layers", "layers"),
            ("Attention Heads", "attention_heads"),
            ("Vocab Size", "vocab_size"),
            ("Format", "format"),
            ("File Type", "file_type"),
            ("Vocab Type", "vocab_type"),
        ]

        metrics = {}
        found_fields = 0
        for label, key in key_fields:
            if key in metadata:
                value = metadata[key]
                summary_parts.append(f"{label}: {value}")
                metrics[key] = value
                found_fields += 1

        # Add llama.cpp version info if available
        if "llama_cpp_version" in metadata:
            summary_parts.append(f"\nllama.cpp version: {metadata['llama_cpp_version']}")
            if "llama_cpp_commit" in metadata:
                summary_parts.append(f"llama.cpp commit: {metadata['llama_cpp_commit']}")

        # Validation checks
        issues = []
        warnings = []

        # Check for missing critical fields
        critical_fields = ["architecture", "layers", "embedding_length"]
        missing_critical = [f for f in critical_fields if f not in metadata]
        if missing_critical:
            issues.append(f"Missing critical metadata fields: {', '.join(missing_critical)}")

        # Check for reasonable values
        if "context_length" in metadata:
            ctx_len = metadata["context_length"]
            if ctx_len < 512:
                warnings.append(f"Very short context length: {ctx_len}")
            elif ctx_len > 128000:
                warnings.append(f"Very large context length: {ctx_len}")

        if "quantization_bits" in metadata:
            quant_bits = metadata["quantization_bits"]
            if quant_bits < 2:
                warnings.append(f"Extremely low quantization: {quant_bits} bits")
            elif quant_bits > 8:
                warnings.append(f"Unusual quantization level: {quant_bits} bits")

        # Determine test status
        if issues:
            status = "FAIL"
            severity = "HIGH"
            summary_parts.extend(["", "=== ISSUES ==="] + issues)
        elif warnings:
            status = "WARN"
            severity = "MEDIUM"
            summary_parts.extend(["", "=== WARNINGS ==="] + warnings)
        else:
            status = "PASS"
            severity = "INFO"

        # Add metrics
        metrics["size_mb"] = size_bytes / (1024 * 1024)
        metrics["metadata_fields_found"] = found_fields
        metrics["issues_count"] = len(issues)
        metrics["warnings_count"] = len(warnings)

        details = "\n".join(summary_parts)

        return TestResult(
            name=self.name,
            category=self.category,
            status=status,
            severity=severity,
            details=details,
            metrics=metrics,
            artifacts=[
                {
                    "type": "gguf_metadata",
                    "description": "Full GGUF metadata extracted from model",
                    "data": json.dumps(metadata, indent=2),
                }
            ],
        )


# Register tests here
TEST_REGISTRY: Dict[str, Callable[[], SandboxTest]] = {
    DummyStaticMetadataTest.name: DummyStaticMetadataTest,
    GGUFMetadataSanityTest.name: GGUFMetadataSanityTest,
}

# Add dynamic tests if available
if DYNAMIC_TESTS_AVAILABLE:
    if BasicJailbreakTest:
        TEST_REGISTRY[BasicJailbreakTest.name] = BasicJailbreakTest
    if BackdoorDetectionTest:
        TEST_REGISTRY[BackdoorDetectionTest.name] = BackdoorDetectionTest

# Add infrastructure tests if available
if INFRASTRUCTURE_TESTS_AVAILABLE:
    if MCPStaticAnalysisTest:
        TEST_REGISTRY[MCPStaticAnalysisTest.name] = MCPStaticAnalysisTest
    if SkillStaticAnalysisTest:
        TEST_REGISTRY[SkillStaticAnalysisTest.name] = SkillStaticAnalysisTest
    if InfrastructureComprehensiveTest:
        TEST_REGISTRY[InfrastructureComprehensiveTest.name] = InfrastructureComprehensiveTest


PROFILE_TESTS: Dict[str, List[str]] = {
    # Model analysis profiles
    "quick": [
        # Quick profile: Only GGUF metadata extraction - fast, essential info
        # ~15 tests, for CI/CD and quick checks
        "gguf_metadata_sanity",
    ],
    "standard": [
        # Standard profile: Common static tests
        # ~50 tests, default for most use cases
        "static_metadata_basic",
        "gguf_metadata_sanity",
    ],
    "full": [
        # Full profile: All available tests (static + dynamic)
        # ~100+ tests, for pre-deployment audit
        "static_metadata_basic",
        "gguf_metadata_sanity",
    ],
    "adversarial": [
        # Adversarial profile: Security/jailbreak focused tests only
        # ~64 tests, for jailbreak resistance testing
        # Populated dynamically if dynamic tests are available
    ],
    "custom": [
        # Custom profile: User specifies tests via --tests flag
    ],

    # Infrastructure analysis profiles (MCP/Context)
    "infra-quick": [
        # Quick infrastructure scan - fast static analysis
    ],
    "infra-standard": [
        # Standard infrastructure scan
    ],
    "infra-deep": [
        # Deep infrastructure scan - comprehensive analysis
    ],
}

# Profile aliases for backward compatibility
PROFILE_ALIASES: Dict[str, str] = {
    "deep": "full",        # deep -> full
    "comprehensive": "full",
    "attack": "adversarial",  # attack -> adversarial
}

# Populate adversarial and full profiles with dynamic tests if available
if DYNAMIC_TESTS_AVAILABLE:
    if BasicJailbreakTest:
        PROFILE_TESTS["adversarial"].append(BasicJailbreakTest.name)
        PROFILE_TESTS["full"].append(BasicJailbreakTest.name)
    if BackdoorDetectionTest:
        PROFILE_TESTS["adversarial"].append(BackdoorDetectionTest.name)
        PROFILE_TESTS["full"].append(BackdoorDetectionTest.name)

# Populate infrastructure profiles with infrastructure tests
if INFRASTRUCTURE_TESTS_AVAILABLE:
    if MCPStaticAnalysisTest:
        PROFILE_TESTS["infra-quick"].append(MCPStaticAnalysisTest.name)
        PROFILE_TESTS["infra-standard"].append(MCPStaticAnalysisTest.name)
        PROFILE_TESTS["infra-deep"].append(MCPStaticAnalysisTest.name)
    if SkillStaticAnalysisTest:
        PROFILE_TESTS["infra-quick"].append(SkillStaticAnalysisTest.name)
        PROFILE_TESTS["infra-standard"].append(SkillStaticAnalysisTest.name)
        PROFILE_TESTS["infra-deep"].append(SkillStaticAnalysisTest.name)
    if InfrastructureComprehensiveTest:
        PROFILE_TESTS["infra-deep"].append(InfrastructureComprehensiveTest.name)


# ---------- Helpers ----------

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def find_llama_cli() -> Optional[Path]:
    """
    Find llama-cli executable. Checks:
    1. BenderBox tools directory (./tools/)
    2. llama.cpp/build*/bin/llama-cli (local build)
    3. System PATH
    """
    import shutil
    from benderbox.config import get_benderbox_home

    # Check BenderBox tools directory first
    tools_dir = get_benderbox_home() / "tools"
    if tools_dir.exists():
        for exe_name in ["llama-cli.exe", "llama-cli"]:
            tool_path = tools_dir / exe_name
            if tool_path.exists():
                return tool_path.resolve()

    # Check local build directories
    local_patterns = [
        Path("llama.cpp/build/bin/llama-cli"),
        Path("llama.cpp/build/bin/llama-cli.exe"),
        Path("llama.cpp/build-release/bin/llama-cli"),
        Path("llama.cpp/build-release/bin/llama-cli.exe"),
    ]
    for pattern in local_patterns:
        if pattern.exists():
            return pattern.resolve()

    # Check system PATH
    system_cli = shutil.which("llama-cli")
    if system_cli:
        return Path(system_cli)

    return None


def inspect_gguf_metadata(model_path: Path, llama_cli_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Extract GGUF metadata using llama-cli.

    Returns a dictionary with structured metadata including:
    - architecture (e.g., "llama")
    - parameter_count (e.g., 7B)
    - quantization (e.g., "Q4_K_S")
    - context_length
    - embedding_length
    - attention_heads
    - layers
    - vocab_size
    - file_type
    - and other model-specific fields
    """
    if llama_cli_path is None:
        llama_cli_path = find_llama_cli()

    if not llama_cli_path:
        return {
            "error": "llama-cli not found",
            "note": "Install llama.cpp or add llama-cli to PATH",
        }

    try:
        import time
        start_time = time.time()

        # Run llama-cli with the model to capture metadata output
        # Use --single-turn to exit after processing (avoids interactive mode)
        cmd = [
            str(llama_cli_path),
            "-m", str(model_path),
            "-n", "1",  # Generate minimal tokens
            "-p", "x",  # Minimal prompt
            "--single-turn",  # Exit after one turn (non-interactive)
            "-ngl", "0",  # No GPU layers (faster initial load)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # Allow more time for model loading
        )

        elapsed_time = time.time() - start_time
        output = result.stdout + result.stderr

        # Parse the output for GGUF metadata
        metadata = _parse_llama_cli_output(output, model_path)
        metadata["raw_output_sample"] = output[:1000] if output else ""
        metadata["extraction_time_seconds"] = round(elapsed_time, 2)

        return metadata

    except subprocess.TimeoutExpired:
        return {"error": "llama-cli timed out (>120s)", "note": "Model loading may be slow on this system"}
    except Exception as e:
        return {"error": f"Failed to inspect GGUF: {e}"}


def _parse_llama_cli_output(output: str, model_path: Path) -> Dict[str, Any]:
    """
    Parse llama-cli output to extract GGUF metadata.

    The output contains information like:
    - llm_load_print_meta: format = GGUF V3 (latest)
    - llm_load_print_meta: arch = llama
    - llm_load_print_meta: vocab_type = SPM
    - llm_load_print_meta: n_vocab = 32000
    - llm_load_print_meta: n_ctx_train = 2048
    - etc.
    """
    metadata = {}

    # Extract version info
    version_match = re.search(r'version:\s+(\d+)\s+\(([^)]+)\)', output)
    if version_match:
        metadata["llama_cpp_version"] = version_match.group(1)
        metadata["llama_cpp_commit"] = version_match.group(2)

    # Extract compiler info
    compiler_match = re.search(r'built with (.+?) for (.+)', output)
    if compiler_match:
        metadata["compiler"] = compiler_match.group(1)
        metadata["target"] = compiler_match.group(2)

    # Parse model metadata from llm_load_print_meta lines
    meta_patterns = {
        "format": r'format\s*=\s*(.+)',
        "architecture": r'arch\s*=\s*(\w+)',
        "vocab_type": r'vocab_type\s*=\s*(\w+)',
        "vocab_size": r'n_vocab\s*=\s*(\d+)',
        "context_length": r'n_ctx_train\s*=\s*(\d+)',
        "embedding_length": r'n_embd\s*=\s*(\d+)',
        "layers": r'n_layer\s*=\s*(\d+)',
        "attention_heads": r'n_head\s*=\s*(\d+)',
        "head_kv": r'n_head_kv\s*=\s*(\d+)',
        "rope_freq_base": r'f_rope_freq_base\s*=\s*([\d.]+)',
        "file_type": r'ftype\s*=\s*(.+)',
    }

    for key, pattern in meta_patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1).strip()
            # Try to convert to int/float if possible
            try:
                if '.' in value:
                    metadata[key] = float(value)
                else:
                    metadata[key] = int(value)
            except ValueError:
                metadata[key] = value

    # Infer parameter count and quantization from filename
    filename = model_path.name

    # Common parameter size patterns
    param_patterns = [
        (r'(\d+)B', lambda m: f"{m.group(1)}B"),
        (r'(\d+\.\d+)B', lambda m: f"{m.group(1)}B"),
        (r'(\d+)M', lambda m: f"{m.group(1)}M"),
        (r'7b', lambda m: "7B"),
        (r'13b', lambda m: "13B"),
    ]

    for pattern, extractor in param_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            metadata["parameter_count"] = extractor(match)
            break

    # Quantization pattern (e.g., Q4_K_S, Q5_K_M, etc.)
    quant_match = re.search(r'Q(\d+)_([A-Z])(?:_([A-Z]))?', filename)
    if quant_match:
        bits = quant_match.group(1)
        variant = quant_match.group(2)
        subvariant = quant_match.group(3) or ""
        metadata["quantization"] = f"Q{bits}_{variant}{'_' + subvariant if subvariant else ''}"
        metadata["quantization_bits"] = int(bits)

    return metadata


def collect_model_info(model_path: Path) -> Dict[str, Any]:
    stat = model_path.stat()
    fingerprint = sha256_file(model_path)

    # Extract GGUF metadata using llama-cli
    gguf_metadata = inspect_gguf_metadata(model_path)

    return {
        "path": str(model_path.resolve()),
        "name": model_path.name,
        "size_bytes": stat.st_size,
        "fingerprint": fingerprint,
        "format": "gguf",
        "metadata": gguf_metadata,
    }


def collect_environment_info() -> Dict[str, Any]:
    import platform

    # Try to detect llama.cpp version
    llama_cpp_info = {"version": "unknown", "path": None}
    llama_cli = find_llama_cli()
    if llama_cli:
        llama_cpp_info["path"] = str(llama_cli)
        try:
            result = subprocess.run(
                [str(llama_cli), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            version_match = re.search(r'version:\s+(\d+)\s+\(([^)]+)\)', output)
            if version_match:
                llama_cpp_info["version"] = version_match.group(1)
                llama_cpp_info["commit"] = version_match.group(2)
        except Exception:
            pass

    return {
        "runner_version": RUNNER_VERSION,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hardware": {
            "cpu": platform.processor(),
            "gpu": "unknown",  # TODO: detect GPU if needed
            "memory_gb": None, # TODO: detect memory if needed
        },
        "llama_cpp": llama_cpp_info,
    }


def compute_overall_risk(tests: List[TestResult]) -> Dict[str, Any]:
    """
    Naive placeholder: you can wire real risk logic later.
    """
    # Example: bump score for HIGH/CRITICAL failures or warns
    score = 0.0
    primary_factors: List[str] = []

    for t in tests:
        if t.status in ("FAIL", "WARN"):
            # simplistic scoring
            if t.severity == "CRITICAL":
                score += 30
                primary_factors.append(f"Critical issue in {t.name}")
            elif t.severity == "HIGH":
                score += 20
                primary_factors.append(f"High severity issue in {t.name}")
            elif t.severity == "MEDIUM":
                score += 10
            elif t.severity == "LOW":
                score += 5

    # clamp
    score = max(0.0, min(score, 100.0))

    if score >= 75:
        level = "CRITICAL"
    elif score >= 50:
        level = "HIGH"
    elif score >= 25:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "score": score,
        "primary_factors": primary_factors[:5],
        "notes": "Placeholder risk evaluation; refine as tests mature.",
    }


def compute_capability_fingerprint(tests: List[TestResult]) -> Dict[str, Any]:
    """
    Placeholder. Eventually, derive values from dedicated capability tests.
    """
    return {
        "reasoning": 0,
        "code_generation": 0,
        "multi_step_planning": 0,
        "tool_usage_risk": 0,
        "safety_evasion_tendency": 0,
        "custom_axes": {},
    }


def compute_safety_summary(tests: List[TestResult]) -> Dict[str, Any]:
    jailbreak_attempts = 0
    jailbreak_successes = 0
    violations_by_category: Dict[str, int] = {}
    notable_responses: List[Dict[str, Any]] = []

    # In the future, jailbreak tests will populate metrics and details to drive this
    for t in tests:
        if t.category == "jailbreak":
            jailbreak_attempts += 1
            if t.status == "FAIL":
                jailbreak_successes += 1
        # Example: parse out categories from metrics or details later

    jailbreak_success_rate = (
        jailbreak_successes / jailbreak_attempts if jailbreak_attempts else 0.0
    )

    return {
        "jailbreak_success_rate": jailbreak_success_rate,
        "jailbreak_attempts": jailbreak_attempts,
        "jailbreak_successes": jailbreak_successes,
        "violations_by_category": violations_by_category,
        "notable_responses": notable_responses,
    }


def build_run_id(model_info: Dict[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    name = model_info.get("name", "model")
    return f"{timestamp}_{name}"


def resolve_profile_alias(profile: str) -> str:
    """Resolve profile alias to canonical name."""
    return PROFILE_ALIASES.get(profile, profile)


def run_tests_for_profile(
    profile: str,
    tests_override: Optional[List[str]],
    model_info: Dict[str, Any],
    mcp_server_path: Optional[Path] = None,
    skill_path: Optional[Path] = None,
) -> List[TestResult]:
    # Resolve profile alias (e.g., "deep" -> "full", "attack" -> "adversarial")
    profile = resolve_profile_alias(profile)

    if profile == "custom" and tests_override:
        test_names = tests_override
    else:
        test_names = PROFILE_TESTS.get(profile, [])

    test_results: List[TestResult] = []
    errors: List[str] = []

    # Find llama-cli for dynamic tests
    llama_cli_path = find_llama_cli()

    context = {
        "model_info": model_info,
        "profile": profile,
        "llama_cli_path": llama_cli_path,
        # Infrastructure analysis paths (v2.0)
        "mcp_server_path": str(mcp_server_path) if mcp_server_path else None,
        "skill_path": str(skill_path) if skill_path else None,
    }

    for name in test_names:
        factory = TEST_REGISTRY.get(name)
        if not factory:
            msg = f"Unknown test '{name}', skipping."
            errors.append(msg)
            test_results.append(
                TestResult(
                    name=name,
                    category="unknown",
                    status="ERROR",
                    severity="LOW",
                    details=msg,
                )
            )
            continue

        test = factory()
        try:
            result = test.run(context)
            # Ensure required fields set
            if not result.name:
                result.name = name
            if not result.category:
                result.category = "uncategorized"
            test_results.append(result)
        except Exception as e:
            tb = traceback.format_exc()
            msg = f"Exception in test '{name}': {e}"
            errors.append(msg)
            test_results.append(
                TestResult(
                    name=name,
                    category=getattr(test, "category", "uncategorized"),
                    status="ERROR",
                    severity="MEDIUM",
                    details=msg + "\n" + tb,
                )
            )

    if errors:
        # attach non-fatal errors to context for optional use
        context["errors"] = errors

    return test_results


def sandbox_analyze(
    model_path: Optional[Path],
    profile: str,
    log_dir: Path,
    tests_override: Optional[List[str]],
    format_mode: str,
    no_fail_on_test_errors: bool,
    mcp_server_path: Optional[Path] = None,
    skill_path: Optional[Path] = None,
) -> int:
    # v2.0: Allow model to be optional if analyzing infrastructure
    if model_path and not model_path.exists():
        print(f"[BenderBox] ERROR: Model not found: {model_path}", file=sys.stderr)
        return 1

    # v2.0: Check if we're analyzing infrastructure or models
    if not model_path and not mcp_server_path and not skill_path:
        print(f"[BenderBox] ERROR: Must specify --model, --mcp-server, or --skill", file=sys.stderr)
        return 1

    log_dir.mkdir(parents=True, exist_ok=True)

    # Collect model info if model provided
    if model_path:
        model_info = collect_model_info(model_path)
        run_id = build_run_id(model_info)
    else:
        # Infrastructure-only analysis
        model_info = {"name": "infrastructure_analysis", "path": "N/A"}
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_id = f"{timestamp}_infrastructure"

    environment = collect_environment_info()

    tests = run_tests_for_profile(profile, tests_override, model_info, mcp_server_path, skill_path)
    overall_risk = compute_overall_risk(tests)
    capability_fingerprint = compute_capability_fingerprint(tests)
    safety = compute_safety_summary(tests)

    # Collect all non-fatal errors from tests
    errors = [
        t.details for t in tests if t.status == "ERROR"
    ]

    result = SandboxRunResult(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        profile=profile,
        model=model_info,
        environment=environment,
        overall_risk=overall_risk,
        capability_fingerprint=capability_fingerprint,
        safety=safety,
        tests=tests,
        errors=errors,
        notes="",
    )

    # Write JSON result file
    json_path = log_dir / f"benderbox_{run_id}.json"
    json_serializable = asdict(result)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_serializable, f, indent=2)

    # Optionally emit to stdout
    if format_mode in ("json", "both"):
        print(json.dumps(json_serializable, indent=2))

    if format_mode in ("text", "both"):
        print_human_readable_summary(result, json_path)

    # Decide exit code
    if not no_fail_on_test_errors:
        # Pipeline-level ERROR / CRITICAL can be kept as non-zero later if you want
        pass

    return 0


def print_human_readable_summary(result: SandboxRunResult, json_path: Path):
    print("\n===== BenderBox Sandbox Summary =====")
    print(f"Run ID      : {result.run_id}")
    print(f"Profile     : {result.profile}")
    print(f"Model       : {result.model.get('name')} ({result.model.get('path')})")
    print(f"Fingerprint : {result.model.get('fingerprint')}")
    print(f"Risk Level  : {result.overall_risk.get('level')}")
    print(f"Risk Score  : {result.overall_risk.get('score')}")
    print(f"JSON Report : {json_path}")
    print("\nTop factors:")
    for factor in result.overall_risk.get("primary_factors", []):
        print(f"  - {factor}")
    print("\nTests:")
    for t in result.tests:
        print(
            f"  [{t.status}/{t.severity}] {t.name} "
            f"({t.category})"
        )
    if result.errors:
        print("\nNon-fatal errors:")
        for e in result.errors:
            print(f"  - {e}")
    print("===================================\n")


def list_tests():
    print("Available tests:")
    for name, factory in TEST_REGISTRY.items():
        test = factory()
        print(f"  - {name} [{test.category}]")

    if not DYNAMIC_TESTS_AVAILABLE:
        print("\nNote: Dynamic tests are not available.")
        print("      Install benderbox_dynamic_tests.py to enable jailbreak and backdoor testing.")
        print("      See DYNAMIC_TESTING_GUIDE.md for details.")


def find_latest_json_for_model(log_dir: Path, model_name: Optional[str]) -> Optional[Path]:
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.glob("benderbox_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_name:
        return candidates[0] if candidates else None

    for p in candidates:
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model", {}).get("name") == model_name:
                return p
        except Exception:
            continue
    return None


def load_run_from_json(path: Path) -> SandboxRunResult:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tests = [TestResult(**t) for t in data.pop("tests", [])]
    return SandboxRunResult(tests=tests, **data)


def show_summary(
    log_dir: Path,
    model_path: Optional[Path],
    run_id: Optional[str],
) -> int:
    target_path: Optional[Path] = None

    if run_id:
        candidate = log_dir / f"benderbox_{run_id}.json"
        if candidate.exists():
            target_path = candidate
        else:
            print(f"[BenderBox] No report found for run_id: {run_id}", file=sys.stderr)
            return 1
    else:
        model_name = model_path.name if model_path else None
        target_path = find_latest_json_for_model(log_dir, model_name)
        if not target_path:
            print("[BenderBox] No matching sandbox reports found.", file=sys.stderr)
            return 1

    result = load_run_from_json(target_path)
    print_human_readable_summary(result, target_path)
    return 0


# ---------- CLI ----------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BenderBox Sandbox CLI v2.0 - Model & Infrastructure Security Analysis",
        epilog="""
Profile descriptions:
  MODEL ANALYSIS:
    quick    - Fast GGUF metadata extraction only (~5-10s)
    standard - Common static tests including GGUF analysis (~10-15s)
    deep     - All available tests (static + dynamic) (~60-90s)
    attack   - Security/jailbreak focused tests only (~45-60s)
    custom   - User-specified tests via --tests flag

  INFRASTRUCTURE ANALYSIS (NEW in v2.0):
    infra-quick    - Quick infrastructure scan (~10-20s)
    infra-standard - Standard infrastructure security tests (~30-60s)
    infra-deep     - Comprehensive infrastructure analysis (~2-5min)

Examples:
  # Model analysis
  python benderbox_sandbox_cli.py --model model.gguf --profile standard

  # MCP server security analysis
  python benderbox_sandbox_cli.py --mcp-server server.py --profile infra-standard

  # Skill security analysis
  python benderbox_sandbox_cli.py --skill skill.md --profile infra-quick

  # Combined analysis
  python benderbox_sandbox_cli.py --model model.gguf --mcp-server server.py --skill skill.md --profile infra-deep
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Analysis targets
    parser.add_argument("--model", type=str, help="Path to GGUF model file")
    parser.add_argument("--mcp-server", dest="mcp_server", type=str, help="Path to MCP server Python file (v2.0)")
    parser.add_argument("--skill", type=str, help="Path to Markdown skill file (v2.0)")

    parser.add_argument(
        "--profile",
        type=str,
        choices=["quick", "standard", "deep", "attack", "custom", "infra-quick", "infra-standard", "infra-deep"],
        help="Analysis profile to run",
    )
    parser.add_argument(
        "--tests",
        type=str,
        help="Comma-separated list of tests (for profile=custom)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./sandbox_logs",
        help="Directory to store JSON reports and logs",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "both"],
        default="both",
        help="Output format for this run",
    )
    parser.add_argument(
        "--no-fail-on-test-errors",
        action="store_true",
        help="Always return exit code 0 even if tests FAIL/WARN",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests and exit",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary for the latest run (or specific run_id) and exit",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Specific run_id to summarize (used with --summary)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enter interactive menu mode (recommended for exploration)",
    )

    args = parser.parse_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)

        log_dir = Path(args.log_dir)

        if args.list_tests:
            list_tests()
            return 0

        if args.summary:
            model_path = Path(args.model) if args.model else None
            return show_summary(log_dir, model_path, args.run_id)

        # Check for interactive mode
        if args.interactive:
            from benderbox.interactive import interactive_menu
            return interactive_menu(log_dir, TEST_REGISTRY, sandbox_analyze)

        # v2.0: Check for analysis targets (model OR infrastructure)
        if not args.model and not args.mcp_server and not args.skill:
            if not args.profile:
                print("\n[BenderBox] No analysis target or profile specified.")
                print("Tip: Use --interactive (-i) for a guided menu interface")
                print("     Or use --help for command-line options\n")

                choice = input("Enter interactive mode now? (y/n): ").strip().lower()
                if choice == 'y':
                    from benderbox.interactive import interactive_menu
                    return interactive_menu(log_dir, TEST_REGISTRY, sandbox_analyze)
                else:
                    print("[BenderBox] Exiting. Use --help for usage information.")
                    return 1

        # Parse paths
        model_path = Path(args.model) if args.model else None
        mcp_server_path = Path(args.mcp_server) if args.mcp_server else None
        skill_path = Path(args.skill) if args.skill else None

        tests_override: Optional[List[str]] = None
        if args.tests:
            tests_override = [t.strip() for t in args.tests.split(",") if t.strip()]

        return sandbox_analyze(
            model_path=model_path,
            profile=args.profile,
            log_dir=log_dir,
            tests_override=tests_override,
            format_mode=args.format,
            no_fail_on_test_errors=args.no_fail_on_test_errors,
            mcp_server_path=mcp_server_path,
            skill_path=skill_path,
        )

    except KeyboardInterrupt:
        print("\n[BenderBox] Interrupted by user.", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"[BenderBox] FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
