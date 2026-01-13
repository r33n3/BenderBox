"""
MCP Client for BenderBox

Provides persistent connections to MCP (Model Context Protocol) servers
for live interrogation and tool testing.

Supports:
- HTTP/SSE transport (remote servers)
- STDIO transport (local server processes)
- Connection pooling and management
"""

import asyncio
import json
import logging
import re
import shlex
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class MCPTransport(Enum):
    """MCP transport types."""
    HTTP = "http"
    STDIO = "stdio"
    AUTO = "auto"


class MCPConnectionState(Enum):
    """Connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""
    name: str
    description: str = ""
    input_schema: Dict = field(default_factory=dict)
    annotations: Dict = field(default_factory=dict)


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""
    success: bool
    tool_name: str
    content: Any = None
    error: Optional[str] = None
    raw_response: Optional[Dict] = None
    duration_ms: float = 0


@dataclass
class MCPServerCapabilities:
    """Capabilities reported by MCP server."""
    tools: bool = False
    resources: bool = False
    prompts: bool = False
    logging: bool = False


class MCPClient:
    """
    Persistent MCP server connection.

    Supports both HTTP and STDIO transports for connecting to
    MCP servers and invoking their tools.
    """

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize MCP client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.timeout = timeout
        self.max_retries = max_retries

        self._state = MCPConnectionState.DISCONNECTED
        self._transport: Optional[MCPTransport] = None
        self._target: Optional[str] = None

        # HTTP transport
        self._http_client: Optional[httpx.AsyncClient] = None
        self._endpoint: Optional[str] = None

        # STDIO transport
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0

        # Cached data
        self._tools: List[MCPToolInfo] = []
        self._capabilities: Optional[MCPServerCapabilities] = None
        self._server_info: Dict = {}

    @property
    def state(self) -> MCPConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == MCPConnectionState.CONNECTED

    @property
    def transport(self) -> Optional[MCPTransport]:
        """Get current transport type."""
        return self._transport

    @property
    def target(self) -> Optional[str]:
        """Get connection target."""
        return self._target

    @property
    def tools(self) -> List[MCPToolInfo]:
        """Get cached tools list."""
        return self._tools

    def _detect_transport(self, target: str) -> MCPTransport:
        """
        Auto-detect transport type from target.

        Args:
            target: Connection target (URL or command)

        Returns:
            Detected transport type
        """
        # Check for HTTP/HTTPS URLs
        if target.startswith(("http://", "https://")):
            return MCPTransport.HTTP

        # Check for common command patterns
        if target.startswith(("npx ", "uvx ", "python ", "node ")):
            return MCPTransport.STDIO

        # Check if it looks like a command (has spaces and common patterns)
        if " " in target or target.endswith((".js", ".py", ".ts")):
            return MCPTransport.STDIO

        # Default to HTTP if it looks like a domain
        if "." in target and "/" not in target:
            return MCPTransport.HTTP

        # Default to STDIO for anything else
        return MCPTransport.STDIO

    async def connect(
        self,
        target: str,
        transport: MCPTransport = MCPTransport.AUTO,
    ) -> bool:
        """
        Connect to an MCP server.

        Args:
            target: Server URL or command to spawn
            transport: Transport type (auto-detect if AUTO)

        Returns:
            True if connection successful
        """
        if self.is_connected:
            await self.disconnect()

        self._state = MCPConnectionState.CONNECTING
        self._target = target

        # Detect transport if auto
        if transport == MCPTransport.AUTO:
            self._transport = self._detect_transport(target)
        else:
            self._transport = transport

        logger.info(f"Connecting to MCP server: {target} (transport: {self._transport.value})")

        try:
            if self._transport == MCPTransport.HTTP:
                await self._connect_http(target)
            else:
                await self._connect_stdio(target)

            # Initialize connection
            await self._initialize()

            # Discover tools
            await self.refresh_tools()

            self._state = MCPConnectionState.CONNECTED
            logger.info(f"Connected to MCP server with {len(self._tools)} tools")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._state = MCPConnectionState.ERROR
            await self.disconnect()
            raise

    async def _connect_http(self, url: str) -> None:
        """Connect via HTTP transport."""
        # Ensure URL has protocol
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        self._endpoint = url
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
        )

    async def _connect_stdio(self, command: str) -> None:
        """Connect via STDIO transport."""
        # Parse command
        if sys.platform == "win32":
            # On Windows, use shell=True for commands like "npx"
            self._process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            # On Unix, split the command properly
            args = shlex.split(command)
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        # Give process time to start
        await asyncio.sleep(0.5)

        # Check if process is still running
        if self._process.returncode is not None:
            stderr = await self._process.stderr.read()
            raise RuntimeError(f"Process exited immediately: {stderr.decode()}")

    async def _initialize(self) -> None:
        """Initialize the MCP connection."""
        # Send initialize request
        response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
            },
            "clientInfo": {
                "name": "BenderBox",
                "version": "3.0.0",
            },
        })

        if response:
            self._server_info = response.get("serverInfo", {})
            caps = response.get("capabilities", {})
            self._capabilities = MCPServerCapabilities(
                tools="tools" in caps,
                resources="resources" in caps,
                prompts="prompts" in caps,
                logging="logging" in caps,
            )

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
            except Exception:
                pass
            self._process = None

        self._state = MCPConnectionState.DISCONNECTED
        self._tools = []
        self._capabilities = None
        self._server_info = {}
        self._endpoint = None
        self._target = None
        self._transport = None

        logger.info("Disconnected from MCP server")

    def _next_request_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    async def _send_request(
        self,
        method: str,
        params: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Send a JSON-RPC request.

        Args:
            method: RPC method name
            params: Method parameters

        Returns:
            Response result or None on error
        """
        request_id = self._next_request_id()
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        if self._transport == MCPTransport.HTTP:
            return await self._send_http_request(request)
        else:
            return await self._send_stdio_request(request)

    async def _send_notification(
        self,
        method: str,
        params: Optional[Dict] = None,
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        if self._transport == MCPTransport.HTTP:
            # HTTP notifications are fire-and-forget
            try:
                await self._http_client.post(
                    self._endpoint,
                    json=notification,
                )
            except Exception:
                pass
        else:
            # STDIO notification
            if self._process and self._process.stdin:
                message = json.dumps(notification) + "\n"
                self._process.stdin.write(message.encode())
                await self._process.stdin.drain()

    async def _send_http_request(self, request: Dict) -> Optional[Dict]:
        """Send request via HTTP."""
        try:
            response = await self._http_client.post(
                self._endpoint,
                json=request,
            )
            response.raise_for_status()

            result = response.json()
            if "error" in result:
                logger.warning(f"RPC error: {result['error']}")
                return None

            return result.get("result")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    async def _send_stdio_request(self, request: Dict) -> Optional[Dict]:
        """Send request via STDIO."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError("STDIO transport not connected")

        # Send request
        message = json.dumps(request) + "\n"
        self._process.stdin.write(message.encode())
        await self._process.stdin.drain()

        # Read response
        try:
            response_line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self.timeout,
            )

            if not response_line:
                return None

            result = json.loads(response_line.decode())

            if "error" in result:
                logger.warning(f"RPC error: {result['error']}")
                return None

            return result.get("result")

        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            return None

    async def refresh_tools(self) -> List[MCPToolInfo]:
        """
        Refresh the list of available tools.

        Returns:
            List of available tools
        """
        result = await self._send_request("tools/list", {})

        if result and "tools" in result:
            self._tools = [
                MCPToolInfo(
                    name=tool.get("name", ""),
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    annotations=tool.get("annotations", {}),
                )
                for tool in result["tools"]
            ]
        else:
            self._tools = []

        return self._tools

    async def list_tools(self) -> List[MCPToolInfo]:
        """
        Get list of available tools.

        Returns:
            List of tools (from cache or refreshed)
        """
        if not self._tools:
            await self.refresh_tools()
        return self._tools

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict] = None,
    ) -> MCPToolResult:
        """
        Call an MCP tool.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        import time

        start_time = time.time()

        try:
            result = await self._send_request("tools/call", {
                "name": name,
                "arguments": arguments or {},
            })

            duration_ms = (time.time() - start_time) * 1000

            if result is None:
                return MCPToolResult(
                    success=False,
                    tool_name=name,
                    error="Request failed",
                    duration_ms=duration_ms,
                )

            # Extract content from result
            content = result.get("content", [])
            if isinstance(content, list) and content:
                # Combine text content
                text_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_content.append(item)
                content = "\n".join(text_content) if text_content else content

            return MCPToolResult(
                success=True,
                tool_name=name,
                content=content,
                raw_response=result,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return MCPToolResult(
                success=False,
                tool_name=name,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def call_tool_stream(
        self,
        name: str,
        arguments: Optional[Dict] = None,
    ) -> AsyncIterator[str]:
        """
        Call a tool and stream the response.

        Note: Currently falls back to non-streaming for STDIO.
        HTTP streaming support depends on server implementation.

        Args:
            name: Tool name
            arguments: Tool arguments

        Yields:
            Response chunks
        """
        if self._transport == MCPTransport.HTTP:
            # Try streaming via HTTP
            request = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments or {},
                },
            }

            try:
                async with self._http_client.stream(
                    "POST",
                    self._endpoint,
                    json=request,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "result" in data:
                                    content = data["result"].get("content", [])
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            yield item["text"]
                            except json.JSONDecodeError:
                                yield line
            except Exception:
                # Fall back to non-streaming
                result = await self.call_tool(name, arguments)
                if result.success and result.content:
                    yield str(result.content)
        else:
            # STDIO doesn't support streaming, use regular call
            result = await self.call_tool(name, arguments)
            if result.success and result.content:
                yield str(result.content)

    def get_tool(self, name: str) -> Optional[MCPToolInfo]:
        """Get tool info by name."""
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    async def __aenter__(self) -> "MCPClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.disconnect()


class MCPClientManager:
    """
    Manages multiple MCP client connections.

    Useful for chat integration where multiple servers
    may be connected simultaneously.
    """

    def __init__(self):
        """Initialize client manager."""
        self._clients: Dict[str, MCPClient] = {}
        self._active_client: Optional[str] = None

    @property
    def active_client(self) -> Optional[MCPClient]:
        """Get the active client."""
        if self._active_client and self._active_client in self._clients:
            return self._clients[self._active_client]
        return None

    @property
    def connections(self) -> Dict[str, MCPClient]:
        """Get all connections."""
        return self._clients.copy()

    def get_connection_names(self) -> List[str]:
        """Get names of all connections."""
        return list(self._clients.keys())

    async def connect(
        self,
        name: str,
        target: str,
        transport: MCPTransport = MCPTransport.AUTO,
        set_active: bool = True,
    ) -> MCPClient:
        """
        Connect to an MCP server.

        Args:
            name: Connection name for reference
            target: Server URL or command
            transport: Transport type
            set_active: Set as active connection

        Returns:
            Connected client
        """
        # Disconnect existing connection with same name
        if name in self._clients:
            await self.disconnect(name)

        client = MCPClient()
        await client.connect(target, transport)

        self._clients[name] = client

        if set_active:
            self._active_client = name

        return client

    async def disconnect(self, name: Optional[str] = None) -> None:
        """
        Disconnect a client.

        Args:
            name: Connection name (disconnects active if None)
        """
        if name is None:
            name = self._active_client

        if name and name in self._clients:
            await self._clients[name].disconnect()
            del self._clients[name]

            if self._active_client == name:
                self._active_client = None
                # Set another client as active if available
                if self._clients:
                    self._active_client = next(iter(self._clients))

    async def disconnect_all(self) -> None:
        """Disconnect all clients."""
        for name in list(self._clients.keys()):
            await self.disconnect(name)

    def set_active(self, name: str) -> bool:
        """
        Set the active connection.

        Args:
            name: Connection name

        Returns:
            True if successful
        """
        if name in self._clients:
            self._active_client = name
            return True
        return False

    def get_client(self, name: str) -> Optional[MCPClient]:
        """Get a client by name."""
        return self._clients.get(name)

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict] = None,
        client_name: Optional[str] = None,
    ) -> MCPToolResult:
        """
        Call a tool on a connected server.

        Args:
            tool_name: Tool to call
            arguments: Tool arguments
            client_name: Specific client (uses active if None)

        Returns:
            Tool result
        """
        client = self.get_client(client_name) if client_name else self.active_client

        if not client:
            return MCPToolResult(
                success=False,
                tool_name=tool_name,
                error="No active MCP connection",
            )

        return await client.call_tool(tool_name, arguments)

    def list_all_tools(self) -> Dict[str, List[MCPToolInfo]]:
        """
        List tools from all connected servers.

        Returns:
            Dict mapping connection name to tool list
        """
        return {
            name: client.tools
            for name, client in self._clients.items()
            if client.is_connected
        }

    async def __aenter__(self) -> "MCPClientManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.disconnect_all()


# Convenience function
async def connect_mcp(
    target: str,
    transport: MCPTransport = MCPTransport.AUTO,
) -> MCPClient:
    """
    Connect to an MCP server.

    Args:
        target: Server URL or command
        transport: Transport type

    Returns:
        Connected client
    """
    client = MCPClient()
    await client.connect(target, transport)
    return client
