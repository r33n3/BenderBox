"""
Sprint 2: Vector Store - Test Suite

Tests for:
- H2.1: Report Indexing Pipeline
- H2.2: Semantic Search API
- H2.3: UI Integration
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_report_indexer_import():
    """Test 1: ReportIndexer can be imported."""
    try:
        from benderbox.storage.report_indexer import (
            ReportIndexer,
            IndexedReport,
            SimilarFinding,
            COLLECTION_REPORTS,
            COLLECTION_FINDINGS,
        )
        print("PASS: ReportIndexer imports successfully")
        return True
    except ImportError as e:
        print(f"FAIL: ReportIndexer import error: {e}")
        return False


def test_report_indexer_creation():
    """Test 2: ReportIndexer can be instantiated."""
    try:
        from benderbox.storage.report_indexer import ReportIndexer
        from benderbox.storage.vector_store import VectorStore

        # Create mock vector store
        class MockVectorStore:
            async def add_document(self, doc, collection):
                return doc.id
            async def search(self, query, collection, top_k=5, filter_metadata=None):
                return []
            async def delete_document(self, doc_id, collection):
                return True
            async def clear_collection(self, collection):
                return 0

        indexer = ReportIndexer(vector_store=MockVectorStore())
        assert indexer is not None
        print("PASS: ReportIndexer instantiates successfully")
        return True
    except Exception as e:
        print(f"FAIL: ReportIndexer creation error: {e}")
        return False


def test_index_report():
    """Test 3: ReportIndexer.index_report() works."""
    async def run_test():
        from benderbox.storage.report_indexer import ReportIndexer

        indexed_docs = []

        class MockVectorStore:
            async def add_document(self, doc, collection):
                indexed_docs.append((doc, collection))
                return doc.id
            async def search(self, query, collection, top_k=5, filter_metadata=None):
                return []

        indexer = ReportIndexer(vector_store=MockVectorStore())

        # Test report
        test_report = {
            "run_id": "test-001",
            "timestamp": "2026-01-12T10:00:00",
            "profile": "standard",
            "target_type": "model",
            "target_name": "test-model.gguf",
            "summary": {
                "risk": {
                    "level": "medium",
                    "score": 45.0,
                    "factors": ["Some risk factor"],
                }
            },
            "results": [
                {
                    "test_name": "jailbreak_test_1",
                    "category": "jailbreak",
                    "status": "failed",
                    "severity": "high",
                    "details": {"message": "Model failed jailbreak test"},
                },
                {
                    "test_name": "safety_test_1",
                    "category": "safety",
                    "status": "passed",
                    "severity": "info",
                    "details": {"message": "Safety test passed"},
                },
            ],
        }

        result = await indexer.index_report(test_report, report_id="test-001")

        # Should have indexed: 1 summary + 2 findings + 2 categories
        assert len(indexed_docs) >= 3, f"Expected at least 3 docs, got {len(indexed_docs)}"
        assert result.report_id == "test-001"
        assert result.target_name == "test-model.gguf"
        assert result.risk_level == "medium"
        return True

    try:
        result = asyncio.run(run_test())
        print("PASS: index_report() works correctly")
        return result
    except Exception as e:
        print(f"FAIL: index_report() error: {e}")
        return False


def test_rag_pipeline_import():
    """Test 4: RAGPipeline can be imported."""
    try:
        from benderbox.nlp.rag import (
            RAGPipeline,
            RetrievedContext,
            RAGResponse,
        )
        print("PASS: RAGPipeline imports successfully")
        return True
    except ImportError as e:
        print(f"FAIL: RAGPipeline import error: {e}")
        return False


def test_rag_pipeline_search():
    """Test 5: RAGPipeline.search() works."""
    async def run_test():
        from benderbox.nlp.rag import RAGPipeline
        from benderbox.storage.report_indexer import ReportIndexer
        from benderbox.storage.vector_store import Document, SearchResult

        class MockVectorStore:
            async def add_document(self, doc, collection):
                return doc.id
            async def search(self, query, collection, top_k=5, filter_metadata=None):
                # Return mock results
                doc = Document.create(
                    content=f"Mock result for {query}",
                    metadata={"target_name": "test", "test_name": "mock_test"},
                )
                return [SearchResult(document=doc, score=0.9, distance=0.1)]

        indexer = ReportIndexer(vector_store=MockVectorStore())
        rag = RAGPipeline(report_indexer=indexer)

        results = await rag.search(query="test query", search_type="all", top_k=5)

        assert len(results) > 0, "Expected at least one result"
        assert results[0]["score"] == 0.9
        return True

    try:
        result = asyncio.run(run_test())
        print("PASS: RAGPipeline.search() works correctly")
        return result
    except Exception as e:
        print(f"FAIL: RAGPipeline.search() error: {e}")
        return False


def test_chat_ui_search_command():
    """Test 6: ChatUI has SEARCH command type."""
    try:
        from benderbox.ui.chat_ui import ChatUI, CommandType, ParsedCommand

        # Check SEARCH enum exists
        assert hasattr(CommandType, "SEARCH"), "Missing SEARCH command type"

        # Check search aliases exist
        assert "search" in ChatUI.COMMAND_ALIASES, "Missing search aliases"
        assert "find" in ChatUI.COMMAND_ALIASES["search"]

        # Test parsing search command
        ui = ChatUI()
        cmd = ui.parse_input("search SQL injection")
        assert cmd.command_type == CommandType.SEARCH
        assert "SQL" in cmd.args or "SQL injection" in " ".join(cmd.args)

        print("PASS: ChatUI has SEARCH command type")
        return True
    except Exception as e:
        print(f"FAIL: ChatUI SEARCH command error: {e}")
        return False


def test_cli_search_command():
    """Test 7: CLI has search command."""
    try:
        from benderbox.ui.app import cli
        from click.testing import CliRunner

        runner = CliRunner()

        # Test help for search command
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0, f"Exit code: {result.exit_code}"
        assert "semantic search" in result.output.lower()

        print("PASS: CLI search command exists")
        return True
    except Exception as e:
        print(f"FAIL: CLI search command error: {e}")
        return False


def test_report_db_indexer_hook():
    """Test 8: ReportDatabase has indexer hook."""
    try:
        from benderbox.storage.report_db import ReportDatabase

        # Check set_indexer method exists
        assert hasattr(ReportDatabase, "set_indexer"), "Missing set_indexer method"

        # Check __init__ accepts indexer
        import inspect
        sig = inspect.signature(ReportDatabase.__init__)
        assert "indexer" in sig.parameters, "Missing indexer parameter in __init__"

        print("PASS: ReportDatabase has indexer hook")
        return True
    except Exception as e:
        print(f"FAIL: ReportDatabase indexer hook error: {e}")
        return False


def test_help_includes_search():
    """Test 9: Help text includes search command."""
    try:
        from benderbox.ui.terminal import TerminalUI

        ui = TerminalUI()

        # Capture help text by checking print_help method
        # (We can't easily capture stdout, but we can check the method exists
        # and contains search info by reading the file)
        import benderbox.ui.terminal as term_module
        import inspect

        source = inspect.getsource(term_module.TerminalUI.print_help)
        assert "search" in source.lower(), "Help text doesn't mention search"
        assert "Semantic Search" in source, "Help text missing Semantic Search section"

        print("PASS: Help text includes search command")
        return True
    except Exception as e:
        print(f"FAIL: Help text search check error: {e}")
        return False


def test_find_similar_findings():
    """Test 10: find_similar_findings() works."""
    async def run_test():
        from benderbox.storage.report_indexer import ReportIndexer, SimilarFinding
        from benderbox.storage.report_db import Finding
        from benderbox.storage.vector_store import Document, SearchResult

        class MockVectorStore:
            async def add_document(self, doc, collection):
                return doc.id
            async def search(self, query, collection, top_k=5, filter_metadata=None):
                doc = Document.create(
                    content="Similar finding content",
                    metadata={
                        "report_id": "other-report",
                        "target_name": "other-model",
                        "finding_id": "finding-002",
                        "test_name": "similar_test",
                        "category": "jailbreak",
                        "status": "failed",
                        "severity": "high",
                    },
                )
                return [SearchResult(document=doc, score=0.85, distance=0.15)]

        indexer = ReportIndexer(vector_store=MockVectorStore())

        test_finding = Finding(
            id="finding-001",
            report_id="test-report",
            test_name="jailbreak_test",
            category="jailbreak",
            status="failed",
            severity="high",
            details={"message": "Test failed"},
        )

        similar = await indexer.find_similar_findings(test_finding, top_k=5)

        assert len(similar) > 0, "Expected similar findings"
        assert similar[0].similarity_score == 0.85
        assert similar[0].report_id == "other-report"
        return True

    try:
        result = asyncio.run(run_test())
        print("PASS: find_similar_findings() works correctly")
        return result
    except Exception as e:
        print(f"FAIL: find_similar_findings() error: {e}")
        return False


def main():
    """Run all Sprint 2 tests."""
    print("=" * 60)
    print("Sprint 2: Vector Store - Test Suite")
    print("=" * 60)
    print()

    tests = [
        ("Test 1: ReportIndexer import", test_report_indexer_import),
        ("Test 2: ReportIndexer creation", test_report_indexer_creation),
        ("Test 3: index_report()", test_index_report),
        ("Test 4: RAGPipeline import", test_rag_pipeline_import),
        ("Test 5: RAGPipeline.search()", test_rag_pipeline_search),
        ("Test 6: ChatUI SEARCH command", test_chat_ui_search_command),
        ("Test 7: CLI search command", test_cli_search_command),
        ("Test 8: ReportDatabase indexer hook", test_report_db_indexer_hook),
        ("Test 9: Help includes search", test_help_includes_search),
        ("Test 10: find_similar_findings()", test_find_similar_findings),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{name}")
        print("-" * 40)
        try:
            if test_func():
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
