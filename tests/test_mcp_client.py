"""
Tests for MCP Client module.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from benderbox.analyzers.mcp_client import (
    MCPClient,
    MCPClientManager,
    MCPToolInfo,
    MCPToolResult,
    MCPTransport,
    MCPConnectionState,
)


class TestMCPToolInfo:
    """Tests for MCPToolInfo dataclass."""

    def test_create_tool_info(self):
        """Test creating a tool info object."""
        tool = MCPToolInfo(
            name="test_tool",
            description="A test tool",
            input_schema={"param1": {"type": "string"}},
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "param1" in tool.input_schema

    def test_tool_info_defaults(self):
        """Test tool info with default fields."""
        tool = MCPToolInfo(name="minimal_tool")

        assert tool.name == "minimal_tool"
        assert tool.description == ""
        assert tool.input_schema == {}


class TestMCPToolResult:
    """Tests for MCPToolResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful result."""
        result = MCPToolResult(
            success=True,
            tool_name="test_tool",
            content="Success output",
        )

        assert result.content == "Success output"
        assert result.success is True
        assert result.error is None

    def test_create_error_result(self):
        """Test creating an error result."""
        result = MCPToolResult(
            success=False,
            tool_name="test_tool",
            error="Something went wrong",
        )

        assert result.error == "Something went wrong"
        assert result.success is False


class TestMCPTransport:
    """Tests for MCPTransport enum."""

    def test_transport_values(self):
        """Test transport enum values."""
        assert MCPTransport.AUTO.value == "auto"
        assert MCPTransport.HTTP.value == "http"
        assert MCPTransport.STDIO.value == "stdio"

    def test_transport_from_string(self):
        """Test creating transport from string."""
        assert MCPTransport("http") == MCPTransport.HTTP
        assert MCPTransport("stdio") == MCPTransport.STDIO


class TestMCPClient:
    """Tests for MCPClient class."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = MCPClient()

        assert client.is_connected is False
        assert client.target is None
        assert client.tools == []
        assert client.state == MCPConnectionState.DISCONNECTED

    def test_client_detect_transport_http(self):
        """Test HTTP transport detection."""
        client = MCPClient()

        transport = client._detect_transport("https://example.com/api")
        assert transport == MCPTransport.HTTP

        transport = client._detect_transport("http://localhost:8080")
        assert transport == MCPTransport.HTTP

    def test_client_detect_transport_stdio(self):
        """Test STDIO transport detection."""
        client = MCPClient()

        transport = client._detect_transport("npx @org/mcp-server")
        assert transport == MCPTransport.STDIO

        transport = client._detect_transport("node server.js")
        assert transport == MCPTransport.STDIO

    @pytest.mark.asyncio
    async def test_client_not_connected_tools(self):
        """Test list_tools when not connected."""
        client = MCPClient()

        # With mocked _send_request returning None
        with patch.object(client, '_send_request', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = None
            tools = await client.list_tools()
            assert tools == []

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test async context manager."""
        async with MCPClient() as client:
            assert client is not None
            # Not connected, but should not raise

    @pytest.mark.asyncio
    async def test_client_disconnect_when_not_connected(self):
        """Test disconnect when not connected."""
        client = MCPClient()
        # Should not raise
        await client.disconnect()
        assert client.is_connected is False
        assert client.state == MCPConnectionState.DISCONNECTED


class TestMCPClientHTTP:
    """Tests for HTTP transport functionality."""

    @pytest.mark.asyncio
    async def test_http_connect_mock(self):
        """Test HTTP connection with mocked response."""
        client = MCPClient()

        # This is just a structural test showing how to mock
        with patch.object(client, '_connect_http', new_callable=AsyncMock):
            with patch.object(client, '_initialize', new_callable=AsyncMock):
                with patch.object(client, 'refresh_tools', new_callable=AsyncMock) as mock_refresh:
                    mock_refresh.return_value = []

                    await client.connect("http://test.com", MCPTransport.HTTP)

                    assert client._transport == MCPTransport.HTTP

    @pytest.mark.asyncio
    async def test_http_list_tools_mock(self):
        """Test listing tools via HTTP with mocked response."""
        client = MCPClient()
        client._state = MCPConnectionState.CONNECTED
        client._transport = MCPTransport.HTTP

        with patch.object(client, '_send_request', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {
                "tools": [
                    {"name": "tool1", "description": "First tool"},
                    {"name": "tool2", "description": "Second tool"},
                ]
            }

            tools = await client.refresh_tools()

            assert len(tools) == 2
            assert tools[0].name == "tool1"


class TestMCPClientManager:
    """Tests for MCPClientManager class."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = MCPClientManager()

        assert manager._clients == {}
        assert manager._active_client is None

    @pytest.mark.asyncio
    async def test_manager_connect(self):
        """Test connecting via manager."""
        manager = MCPClientManager()

        # Mock the MCPClient
        with patch('benderbox.analyzers.mcp_client.MCPClient') as MockClient:
            mock_client = MagicMock()
            mock_client.connect = AsyncMock(return_value=True)
            MockClient.return_value = mock_client

            client = await manager.connect("test", "http://example.com")

            assert "test" in manager._clients

    @pytest.mark.asyncio
    async def test_manager_disconnect(self):
        """Test disconnecting via manager."""
        manager = MCPClientManager()

        mock_client = MagicMock()
        mock_client.disconnect = AsyncMock()
        manager._clients["test"] = mock_client

        await manager.disconnect("test")

        mock_client.disconnect.assert_called_once()
        assert "test" not in manager._clients

    @pytest.mark.asyncio
    async def test_manager_disconnect_all(self):
        """Test disconnecting all clients."""
        manager = MCPClientManager()

        mock_client1 = MagicMock()
        mock_client1.disconnect = AsyncMock()
        mock_client2 = MagicMock()
        mock_client2.disconnect = AsyncMock()

        manager._clients["c1"] = mock_client1
        manager._clients["c2"] = mock_client2

        await manager.disconnect_all()

        mock_client1.disconnect.assert_called_once()
        mock_client2.disconnect.assert_called_once()
        assert manager._clients == {}

    def test_manager_get_client(self):
        """Test getting a client."""
        manager = MCPClientManager()

        mock_client = MagicMock()
        manager._clients["test"] = mock_client

        client = manager.get_client("test")
        assert client == mock_client

        client = manager.get_client("nonexistent")
        assert client is None

    def test_manager_get_connection_names(self):
        """Test getting connection names."""
        manager = MCPClientManager()

        mock_client = MagicMock()
        mock_client._target = "http://example.com"
        manager._clients["test"] = mock_client

        names = manager.get_connection_names()
        assert "test" in names

    @pytest.mark.asyncio
    async def test_manager_context_manager(self):
        """Test async context manager."""
        async with MCPClientManager() as manager:
            assert manager is not None


class TestMCPClientCallTool:
    """Tests for tool calling functionality."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        client = MCPClient()

        with patch.object(client, '_send_request', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = {
                "content": [
                    {"type": "text", "text": "Hello world"}
                ]
            }

            result = await client.call_tool("test_tool", {"arg": "value"})

            assert result.success is True
            assert result.tool_name == "test_tool"
            assert "Hello" in result.content

    @pytest.mark.asyncio
    async def test_call_tool_failure(self):
        """Test failed tool call."""
        client = MCPClient()

        with patch.object(client, '_send_request', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = None

            result = await client.call_tool("test_tool", {})

            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_call_tool_exception(self):
        """Test tool call with exception."""
        client = MCPClient()

        with patch.object(client, '_send_request', new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("Network error")

            result = await client.call_tool("test_tool", {})

            assert result.success is False
            assert "Network error" in result.error


class TestMCPClientIntegration:
    """Integration-style tests (still with mocks for external calls)."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a full connect -> list tools -> call tool -> disconnect workflow."""
        client = MCPClient()

        with patch.object(client, '_connect_http', new_callable=AsyncMock):
            with patch.object(client, '_initialize', new_callable=AsyncMock):
                with patch.object(client, '_send_request', new_callable=AsyncMock) as mock_send:
                    # Mock tools list response
                    mock_send.return_value = {
                        "tools": [{"name": "test_tool", "description": "A test"}]
                    }

                    await client.connect("http://test.com", MCPTransport.HTTP)

                    assert client.is_connected
                    assert len(client.tools) == 1

                    # Mock tool call
                    mock_send.return_value = {
                        "content": [{"type": "text", "text": "Result"}]
                    }

                    result = await client.call_tool("test_tool", {})
                    assert result.success

                    await client.disconnect()
                    assert not client.is_connected

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios."""
        client = MCPClient()

        # Test connection error
        with patch.object(client, '_connect_http', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await client.connect("http://example.com", MCPTransport.HTTP)

            assert client.state == MCPConnectionState.DISCONNECTED


class TestMCPToolMethods:
    """Tests for tool-related methods."""

    def test_get_tool_found(self):
        """Test getting an existing tool."""
        client = MCPClient()
        client._tools = [
            MCPToolInfo(name="tool1", description="First"),
            MCPToolInfo(name="tool2", description="Second"),
        ]

        tool = client.get_tool("tool1")
        assert tool is not None
        assert tool.name == "tool1"

    def test_get_tool_not_found(self):
        """Test getting a non-existent tool."""
        client = MCPClient()
        client._tools = [MCPToolInfo(name="tool1")]

        tool = client.get_tool("nonexistent")
        assert tool is None


class TestMCPClientManagerAdvanced:
    """Advanced tests for MCPClientManager."""

    def test_set_active(self):
        """Test setting active client."""
        manager = MCPClientManager()
        mock_client = MagicMock()
        manager._clients["test"] = mock_client

        result = manager.set_active("test")
        assert result is True
        assert manager._active_client == "test"

        result = manager.set_active("nonexistent")
        assert result is False

    def test_active_client_property(self):
        """Test active client property."""
        manager = MCPClientManager()
        mock_client = MagicMock()
        manager._clients["test"] = mock_client
        manager._active_client = "test"

        assert manager.active_client == mock_client

    @pytest.mark.asyncio
    async def test_manager_call_tool_no_connection(self):
        """Test calling tool with no active connection."""
        manager = MCPClientManager()

        result = await manager.call_tool("test_tool", {})

        assert result.success is False
        assert "No active" in result.error

    def test_list_all_tools(self):
        """Test listing tools from all connections."""
        manager = MCPClientManager()

        mock_client = MagicMock()
        mock_client.tools = [MCPToolInfo(name="tool1")]
        mock_client.is_connected = True
        manager._clients["server1"] = mock_client

        all_tools = manager.list_all_tools()

        assert "server1" in all_tools
        assert len(all_tools["server1"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
