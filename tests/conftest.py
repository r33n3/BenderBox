"""
BenderBox Test Configuration and Fixtures

Shared fixtures for all test modules.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Generator, Optional
from unittest.mock import MagicMock, AsyncMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Event Loop Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root) -> Path:
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def models_dir(data_dir) -> Path:
    """Return the models directory."""
    return data_dir / "models"


@pytest.fixture
def test_model_path(models_dir) -> Optional[Path]:
    """Return path to a test model if one exists."""
    # Look for any .gguf file in models directory
    for pattern in ["**/*.gguf"]:
        models = list(models_dir.glob(pattern))
        if models:
            return models[0]
    return None


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_terminal_ui():
    """Create a mock TerminalUI that captures output."""
    mock = MagicMock()
    mock.output = []

    def capture_print(msg, *args, **kwargs):
        mock.output.append(str(msg))

    mock.print_info = capture_print
    mock.print_success = capture_print
    mock.print_warning = capture_print
    mock.print_error = capture_print
    mock.print_header = capture_print
    mock.print_markdown = capture_print
    mock.console = None

    return mock


@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager with test data."""
    mock = MagicMock()

    mock.get_downloaded_models.return_value = [
        {"name": "test-model.gguf", "path": "/fake/path/test-model.gguf", "size_mb": 100},
        {"name": "phi-2.Q2_K.gguf", "path": "/fake/path/phi-2.Q2_K.gguf", "size_mb": 1119},
    ]
    mock.list_analysis_models.return_value = []
    mock.list_nlp_models.return_value = []
    mock.find_model_by_name.return_value = None

    return mock


@pytest.fixture
def mock_llm_engine():
    """Create a mock LLM engine."""
    mock = MagicMock()
    mock.is_available = False
    mock.generate = AsyncMock(return_value="Mock LLM response")
    return mock


# ============================================================================
# NLP Component Fixtures
# ============================================================================

@pytest.fixture
def command_mapper(mock_model_manager):
    """Create a CommandMapper instance with mock model manager."""
    from benderbox.nlp.command_mapper import CommandMapper
    return CommandMapper(mock_model_manager)


@pytest.fixture
def intent_router(mock_llm_engine):
    """Create an IntentRouter instance."""
    from benderbox.nlp.intent import IntentRouter
    return IntentRouter(mock_llm_engine)


@pytest.fixture
def conversation_manager(mock_llm_engine):
    """Create a ConversationManager instance."""
    from benderbox.nlp.conversation import ConversationManager
    return ConversationManager(llm_engine=mock_llm_engine)


@pytest.fixture
def persona():
    """Create a BenderPersona instance."""
    from benderbox.nlp.persona import BenderPersona
    return BenderPersona()


# ============================================================================
# UI Component Fixtures
# ============================================================================

@pytest.fixture
def chat_ui(mock_terminal_ui, mock_model_manager, conversation_manager):
    """Create a ChatUI instance with mocks."""
    from benderbox.ui.chat_ui import ChatUI

    ui = ChatUI(
        conversation_manager=conversation_manager,
        terminal_ui=mock_terminal_ui,
        model_manager=mock_model_manager,
    )
    return ui


# ============================================================================
# Analysis Fixtures
# ============================================================================

@pytest.fixture
def analysis_bridge(tmp_path):
    """Create an AnalysisBridge instance with temp log directory."""
    from benderbox.nlp.analysis_bridge import AnalysisBridge
    return AnalysisBridge(log_dir=str(tmp_path / "logs"))


# ============================================================================
# Helper Functions
# ============================================================================

def run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)
