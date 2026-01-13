"""
Sprint 8: RLM Integration - Test Suite

Tests for:
- R1.1: RLM Infrastructure
- R1.2: Codebase-Scale Analysis
- R1.3: Batch Response Analysis
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_rlm_module_import():
    """Test 1: RLM module can be imported."""
    try:
        from benderbox.nlp.rlm import (
            RLMController,
            RLMConfig,
            RLMContext,
            RLMStats,
            DecompositionStrategy,
            AggregationStrategy,
        )

        assert RLMController is not None
        assert RLMConfig is not None
        assert RLMContext is not None
        assert RLMStats is not None

        print("PASS: RLM module imports successfully")
        return True
    except ImportError as e:
        print(f"FAIL: RLM module import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: RLM module error: {e}")
        return False


def test_rlm_controller_creation():
    """Test 2: RLMController can be created with config."""
    try:
        from benderbox.nlp.rlm import RLMController, RLMConfig

        # Create with default config
        controller = RLMController()
        assert controller is not None
        assert controller.config is not None
        assert controller.config.max_depth == 5

        # Create with custom config
        config = RLMConfig(
            max_depth=3,
            max_chunk_size=2000,
            max_parallel=2,
        )
        controller2 = RLMController(config=config)
        assert controller2.config.max_depth == 3
        assert controller2.config.max_chunk_size == 2000

        print("PASS: RLMController creation works")
        return True
    except Exception as e:
        print(f"FAIL: RLMController creation error: {e}")
        return False


def test_rlm_decomposition():
    """Test 3: RLM decomposition strategies work."""
    try:
        from benderbox.nlp.rlm import RLMController, RLMConfig, DecompositionStrategy

        # Test fixed decomposition
        config = RLMConfig(
            decomposition=DecompositionStrategy.CHUNK_FIXED,
            max_chunk_size=100,  # Small chunk size for testing
        )
        controller = RLMController(config=config)

        # Create content larger than chunk size
        content = "def foo():\n    pass\n\n" * 50  # ~1000 chars

        chunks = controller._decompose(content)
        assert len(chunks) > 1, "Should decompose into multiple chunks"

        for chunk in chunks:
            assert chunk.content is not None
            assert chunk.index >= 0
            assert chunk.total == len(chunks)

        # Test semantic decomposition
        config2 = RLMConfig(
            decomposition=DecompositionStrategy.CHUNK_SEMANTIC,
            max_chunk_size=100,
        )
        controller2 = RLMController(config=config2)
        chunks2 = controller2._decompose(content)
        assert len(chunks2) > 0, "Semantic decomposition should produce chunks"

        print("PASS: RLM decomposition strategies work")
        return True
    except Exception as e:
        print(f"FAIL: RLM decomposition error: {e}")
        return False


def test_rlm_context():
    """Test 4: RLM context tracking works."""
    try:
        from benderbox.nlp.rlm import RLMContext

        # Create context
        ctx = RLMContext()
        assert ctx.depth == 0
        assert ctx.path == []
        assert ctx.variables == {}
        assert ctx.accumulated_cost == 0.0

        # Create child context
        child_ctx = RLMContext(
            depth=ctx.depth + 1,
            path=ctx.path + ["chunk_0"],
            variables=ctx.variables.copy(),
            accumulated_cost=10.0,
        )
        assert child_ctx.depth == 1
        assert child_ctx.path == ["chunk_0"]
        assert child_ctx.accumulated_cost == 10.0

        print("PASS: RLM context tracking works")
        return True
    except Exception as e:
        print(f"FAIL: RLM context error: {e}")
        return False


def test_rlm_stats():
    """Test 5: RLM statistics tracking works."""
    try:
        from benderbox.nlp.rlm import RLMController

        controller = RLMController()

        # Check initial stats
        stats = controller.get_stats()
        assert stats["total_calls"] == 0
        assert stats["chunks_processed"] == 0
        assert stats["cache_hits"] == 0

        # Stats should have expected keys
        expected_keys = [
            "total_calls",
            "max_depth_reached",
            "total_tokens_processed",
            "total_cost",
            "cache_hits",
            "cache_misses",
            "chunks_processed",
            "aggregations_performed",
            "duration_seconds",
            "cache_hit_rate",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing stats key: {key}"

        print("PASS: RLM statistics tracking works")
        return True
    except Exception as e:
        print(f"FAIL: RLM stats error: {e}")
        return False


def test_codebase_analyzer_import():
    """Test 6: CodebaseAnalyzer can be imported."""
    try:
        from benderbox.nlp.rlm import (
            CodebaseAnalyzer,
            CodebaseAnalysisResult,
            FileInfo,
        )

        assert CodebaseAnalyzer is not None
        assert CodebaseAnalysisResult is not None
        assert FileInfo is not None

        # Create analyzer
        analyzer = CodebaseAnalyzer()
        assert analyzer is not None
        assert analyzer.rlm is not None

        print("PASS: CodebaseAnalyzer imports successfully")
        return True
    except ImportError as e:
        print(f"FAIL: CodebaseAnalyzer import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: CodebaseAnalyzer error: {e}")
        return False


def test_codebase_file_discovery():
    """Test 7: Codebase file discovery works."""
    try:
        from benderbox.nlp.rlm import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer()

        # Test on current project src directory
        src_path = Path(__file__).parent.parent / "src" / "benderbox"
        if src_path.exists():
            files = analyzer._discover_files(src_path)
            assert len(files) > 0, "Should discover files"

            # Check file info
            for f in files[:5]:
                assert f.path.exists()
                assert f.size > 0
                assert f.language in ["python", "javascript", "typescript", "unknown"]

            print(f"PASS: File discovery works ({len(files)} files found)")
            return True
        else:
            print("SKIP: Source directory not found")
            return True
    except Exception as e:
        print(f"FAIL: File discovery error: {e}")
        return False


def test_codebase_file_prioritization():
    """Test 8: File prioritization works."""
    try:
        from benderbox.nlp.rlm import CodebaseAnalyzer, FileInfo

        analyzer = CodebaseAnalyzer()

        # Create test files
        files = [
            FileInfo(path=Path("test.py"), size=1000, language="python"),
            FileInfo(path=Path("main.py"), size=500, language="python", is_entry_point=True),
            FileInfo(path=Path("config.yaml"), size=200, language="yaml", is_config=True),
            FileInfo(path=Path("auth.py"), size=2000, language="python"),
        ]

        prioritized = analyzer._prioritize_files(files)

        # Entry points should be first
        assert prioritized[0].is_entry_point or prioritized[0].is_config, \
            "High priority files should be first"

        # All files should have priorities set
        for f in prioritized:
            assert f.priority >= 0

        print("PASS: File prioritization works")
        return True
    except Exception as e:
        print(f"FAIL: File prioritization error: {e}")
        return False


def test_batch_response_analyzer_import():
    """Test 9: BatchResponseAnalyzer can be imported."""
    try:
        from benderbox.nlp.rlm import (
            BatchResponseAnalyzer,
            BatchResponseResult,
        )

        assert BatchResponseAnalyzer is not None
        assert BatchResponseResult is not None

        # Create analyzer
        analyzer = BatchResponseAnalyzer()
        assert analyzer is not None
        assert analyzer.rlm is not None

        print("PASS: BatchResponseAnalyzer imports successfully")
        return True
    except ImportError as e:
        print(f"FAIL: BatchResponseAnalyzer import error: {e}")
        return False
    except Exception as e:
        print(f"FAIL: BatchResponseAnalyzer error: {e}")
        return False


def test_analysis_bridge_rlm_methods():
    """Test 10: AnalysisBridge has RLM methods."""
    try:
        from benderbox.nlp.analysis_bridge import AnalysisBridge

        bridge = AnalysisBridge()

        # Check RLM methods exist
        assert hasattr(bridge, "analyze_codebase"), "Should have analyze_codebase"
        assert hasattr(bridge, "analyze_responses_batch"), "Should have analyze_responses_batch"

        # Check callable
        assert callable(bridge.analyze_codebase)
        assert callable(bridge.analyze_responses_batch)

        # Check status includes RLM info
        status = bridge.get_status()
        assert "rlm_analysis" in status, "Status should include rlm_analysis"
        assert status["rlm_analysis"]["available"] is True

        print("PASS: AnalysisBridge has RLM methods")
        return True
    except Exception as e:
        print(f"FAIL: AnalysisBridge RLM methods error: {e}")
        return False


def main():
    """Run all Sprint 8 tests."""
    print("=" * 60)
    print("Sprint 8: RLM Integration - Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Test 1: RLM module import", test_rlm_module_import),
        ("Test 2: RLMController creation", test_rlm_controller_creation),
        ("Test 3: RLM decomposition strategies", test_rlm_decomposition),
        ("Test 4: RLM context tracking", test_rlm_context),
        ("Test 5: RLM statistics tracking", test_rlm_stats),
        ("Test 6: CodebaseAnalyzer import", test_codebase_analyzer_import),
        ("Test 7: File discovery", test_codebase_file_discovery),
        ("Test 8: File prioritization", test_codebase_file_prioritization),
        ("Test 9: BatchResponseAnalyzer import", test_batch_response_analyzer_import),
        ("Test 10: AnalysisBridge RLM methods", test_analysis_bridge_rlm_methods),
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
