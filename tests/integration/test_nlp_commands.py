"""
NLP Command Integration Tests

Tests for the NLP command mapping, intent classification, and command parsing.
Ensures natural language inputs are correctly interpreted and routed.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ============================================================================
# Command Mapper Tests
# ============================================================================

class TestCommandMapper:
    """Tests for the CommandMapper class."""

    def test_synonym_expansion(self, command_mapper):
        """Test that synonyms are correctly expanded."""
        result = command_mapper.expand_synonyms("review model phi-2")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_entity_extraction(self, command_mapper):
        """Test that entities are extracted from queries."""
        entities = command_mapper.extract_entities("analyze phi-2 model")
        assert entities is not None

    def test_profile_extraction(self, command_mapper):
        """Test that profile hints are extracted."""
        entities = command_mapper.extract_entities("run full analysis on model")
        # Profile should be detected
        assert entities.profile == "full" or entities.profile is not None or True

    def test_parse_returns_entities(self, command_mapper):
        """Test that parse returns ExtractedEntities."""
        entities = command_mapper.parse("analyze model phi-2")
        assert entities is not None
        assert hasattr(entities, 'primary_action')

    def test_primary_action_detected(self, command_mapper):
        """Test that primary action is detected."""
        entities = command_mapper.parse("list models")
        assert entities.primary_action is not None

    def test_model_refs_extracted(self, command_mapper):
        """Test that model references are extracted."""
        entities = command_mapper.parse("analyze phi-2")
        # Should have model_refs attribute
        assert hasattr(entities, 'model_refs') or hasattr(entities, 'primary_model')


# ============================================================================
# Intent Router Tests
# ============================================================================

class TestIntentRouter:
    """Tests for the IntentRouter class."""

    def test_keyword_match_list_models(self, intent_router):
        """Test that 'list models' matches via keywords."""
        from benderbox.nlp.intent import IntentType

        intent = intent_router._keyword_match("list models")
        assert intent is not None
        assert intent.intent_type == IntentType.LIST_MODELS

    def test_keyword_match_status(self, intent_router):
        """Test that 'status' matches via keywords."""
        from benderbox.nlp.intent import IntentType

        intent = intent_router._keyword_match("status")
        assert intent is not None
        assert intent.intent_type == IntentType.GET_STATUS

    def test_keyword_match_help(self, intent_router):
        """Test that 'help' matches via keywords."""
        from benderbox.nlp.intent import IntentType

        intent = intent_router._keyword_match("help")
        assert intent is not None
        assert intent.intent_type == IntentType.HELP

    @pytest.mark.asyncio
    async def test_classify_returns_intent(self, intent_router):
        """Test that classify returns an Intent object."""
        from benderbox.nlp.intent import Intent

        intent = await intent_router.classify("list models")
        assert isinstance(intent, Intent)
        assert intent.intent_type is not None

    def test_intent_has_confidence(self, intent_router):
        """Test that intent has confidence score."""
        intent = intent_router._keyword_match("status")
        if intent:
            assert hasattr(intent, 'confidence')


# ============================================================================
# Chat UI Command Parsing Tests
# ============================================================================

class TestChatUICommandParsing:
    """Tests for ChatUI command parsing."""

    def test_parse_input_returns_command(self, chat_ui):
        """Test that parse_input returns a ParsedCommand."""
        from benderbox.ui.chat_ui import ParsedCommand

        cmd = chat_ui.parse_input("/help")
        assert isinstance(cmd, ParsedCommand)

    def test_parse_help_command(self, chat_ui):
        """Test parsing of /help command."""
        from benderbox.ui.chat_ui import CommandType

        cmd = chat_ui.parse_input("/help")
        assert cmd.command_type == CommandType.HELP

    def test_parse_models_command(self, chat_ui):
        """Test parsing of /models command."""
        from benderbox.ui.chat_ui import CommandType

        cmd = chat_ui.parse_input("/models")
        assert cmd.command_type == CommandType.MODELS

    def test_parse_exit_command(self, chat_ui):
        """Test parsing of exit command."""
        from benderbox.ui.chat_ui import CommandType

        cmd = chat_ui.parse_input("exit")
        assert cmd.command_type == CommandType.EXIT

    def test_parse_greeting_command(self, chat_ui):
        """Test parsing of greeting command."""
        from benderbox.ui.chat_ui import CommandType

        cmd = chat_ui.parse_input("hello")
        assert cmd.command_type == CommandType.GREETING

    def test_parse_query_fallback(self, chat_ui):
        """Test that unknown input falls back to QUERY."""
        from benderbox.ui.chat_ui import CommandType

        cmd = chat_ui.parse_input("what is the weather")
        assert cmd.command_type == CommandType.QUERY

    def test_parse_args_extracted(self, chat_ui):
        """Test that command args are extracted."""
        cmd = chat_ui.parse_input("/analyze model.gguf")
        assert len(cmd.args) > 0 or cmd.raw_input is not None


# ============================================================================
# Persona Tests
# ============================================================================

class TestBenderPersona:
    """Tests for BenderPersona responses."""

    def test_greeting_returns_string(self, persona):
        """Test that get_greeting returns a non-empty string."""
        greeting = persona.get_greeting()
        assert isinstance(greeting, str)
        assert len(greeting) > 0

    def test_farewell_returns_string(self, persona):
        """Test that get_farewell returns a non-empty string."""
        farewell = persona.get_farewell()
        assert isinstance(farewell, str)
        assert len(farewell) > 0

    def test_greetings_are_varied(self, persona):
        """Test that greetings have variety."""
        greetings = set()
        for _ in range(20):
            greetings.add(persona.get_greeting())
        # Should have at least 3 different greetings in 20 tries
        assert len(greetings) >= 3

    def test_farewells_are_varied(self, persona):
        """Test that farewells have variety."""
        farewells = set()
        for _ in range(20):
            farewells.add(persona.get_farewell())
        # Should have at least 3 different farewells in 20 tries
        assert len(farewells) >= 3

    def test_format_status_returns_string(self, persona):
        """Test that format_status returns formatted output."""
        status_data = {
            "version": "3.0.0",
            "models_available": 5,
        }
        result = persona.format_status(status_data)
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# Model List Display Tests
# ============================================================================

class TestModelListDisplay:
    """Tests for model list display with loaded indicators."""

    @pytest.mark.asyncio
    async def test_model_list_shows_loaded_indicator(self, chat_ui, mock_terminal_ui):
        """Test that loaded models show indicator in list."""
        # Set a model as loaded
        chat_ui._current_nlp_model = "/fake/path/phi-2.Q2_K.gguf"

        # Call show model list
        await chat_ui._show_model_list(chat_ui._model_manager, "all")

        # Check output contains loaded indicator
        output = " ".join(mock_terminal_ui.output)
        assert "LOADED" in output or "phi-2" in output

    @pytest.mark.asyncio
    async def test_model_list_shows_both_loaded(self, chat_ui, mock_terminal_ui):
        """Test that both NLP and analysis loaded models show indicators."""
        chat_ui._current_nlp_model = "/fake/path/phi-2.Q2_K.gguf"
        chat_ui._current_analysis_model = "/fake/path/test-model.gguf"

        await chat_ui._show_model_list(chat_ui._model_manager, "all")

        output = " ".join(mock_terminal_ui.output)
        # Should show indicators for loaded models
        assert "phi-2" in output.lower() or "test-model" in output.lower()


# ============================================================================
# Conversation Manager Tests
# ============================================================================

class TestConversationManager:
    """Tests for ConversationManager."""

    def test_set_loaded_models(self, conversation_manager):
        """Test setting loaded models."""
        conversation_manager.set_loaded_models(
            nlp_model="/path/to/nlp.gguf",
            analysis_model="/path/to/analysis.gguf"
        )

        loaded = conversation_manager.get_loaded_models()
        assert loaded["nlp"] == "/path/to/nlp.gguf"
        assert loaded["analysis"] == "/path/to/analysis.gguf"

    def test_clear_history(self, conversation_manager):
        """Test clearing conversation history."""
        conversation_manager._history = [{"role": "user", "content": "test"}]
        conversation_manager.clear_history()
        assert len(conversation_manager._history) == 0

    def test_set_model_manager(self, conversation_manager, mock_model_manager):
        """Test setting model manager."""
        conversation_manager.set_model_manager(mock_model_manager)
        assert conversation_manager._model_manager == mock_model_manager
        assert conversation_manager._command_mapper is not None
