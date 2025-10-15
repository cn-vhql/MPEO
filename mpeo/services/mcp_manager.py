"""
MCP服务管理模块 - 提供统一、抽象的MCP服务调用接口
参考mcp_chatbot-master项目的简洁设计，优化MPEO的MCP服务调用逻辑
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None

# Optional aiohttp imports for HTTP client
try:
    import aiohttp
    from aiohttp import ClientTimeout, TCPConnector
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None
    ClientTimeout = None
    TCPConnector = None

from ..models import MCPServiceConfig, TaskNode
from ..utils.logging import get_logger


@dataclass
class MCPTool:
    """MCP工具信息"""
    name: str
    description: str
    input_schema: Dict[str, Any]

    def format_for_llm(self) -> str:
        """格式化工具信息供LLM使用"""
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


@dataclass
class MCPResult:
    """MCP调用结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: Optional[str] = None
    service_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "tool_name": self.tool_name,
            "service_name": self.service_name
        }


@dataclass
class MCPConnectionConfig:
    """MCP连接配置"""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 2
    retry_delay: float = 1.0

    @classmethod
    def from_service_config(cls, config: MCPServiceConfig) -> "MCPConnectionConfig":
        """从MCPServiceConfig创建连接配置"""
        # 支持基于命令的MCP服务
        if config.service_type == "stdio":
            # 对于STDIO类型，从endpoint_url解析命令信息
            # 例如: stdio://npx/@modelcontextprotocol/server-fetch
            if config.endpoint_url.startswith("stdio://"):
                # 提取命令信息
                endpoint_info = config.endpoint_url[7:]  # 去掉 "stdio://"
                parts = endpoint_info.split("/", 1)
                command = parts[0] if parts else "npx"
                args = parts[1].split("/") if len(parts) > 1 and parts[1] else []
                if command == "npx" and args:
                    args = ["-y"] + args
            else:
                command = "npx"
                args = []

            return cls(
                name=config.service_name,
                command=command,
                args=args,
                env={},
                timeout=config.timeout
            )

        # 对于HTTP/SSE服务，使用HTTP客户端
        return cls(
            name=config.service_name,
            command="curl",  # 标识为HTTP类型
            args=[config.endpoint_url],
            timeout=config.timeout
        )


class MCPClient:
    """MCP客户端，基于mcp_chatbot-master的简洁设计"""

    def __init__(self, name: str, config: MCPConnectionConfig, logger: Optional[logging.Logger] = None):
        if not MCP_AVAILABLE:
            raise ImportError("MCP library is not installed. Install with: pip install mcp")

        self.name = name
        self.config = config
        self.logger = logger or get_logger(f"mcp_client.{name}")
        self.stdio_context: Any = None
        self.session: Optional[ClientSession] = None
        self._cleanup_lock = asyncio.Lock()
        self.exit_stack = AsyncExitStack()
        self._available_tools: List[MCPTool] = []
        self._is_initialized = False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()

    async def initialize(self) -> None:
        """初始化服务器连接"""
        if self._is_initialized:
            return

        if not MCP_AVAILABLE:
            raise ImportError("MCP library is not installed")

        self.logger.info(f"Initializing MCP client for {self.name}")

        command = (
            shutil.which("npx")
            if self.config.command == "npx"
            else self.config.command
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config.args,
            env={**os.environ, **self.config.env}
            if self.config.env
            else None,
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            self._is_initialized = True
            self.logger.info(f"MCP client {self.name} initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> List[MCPTool]:
        """获取可用工具列表"""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        # 如果已经缓存了工具列表，直接返回
        if self._available_tools:
            return self._available_tools

        try:
            self.logger.debug(f"Calling list_tools on session for {self.name}")
            tools_response = await self.session.list_tools()
            self.logger.debug(f"Raw tools response from {self.name}: {type(tools_response)} - {str(tools_response)[:200]}")

            tools = []

            # 处理不同类型的响应格式
            if hasattr(tools_response, '__iter__'):
                # 如果是可迭代对象
                for item in tools_response:
                    self.logger.debug(f"Processing item: {type(item)} - {str(item)[:100]}")

                    if isinstance(item, tuple) and item[0] == "tools":
                        # 标准MCP格式: ("tools", [tool1, tool2, ...])
                        for tool in item[1]:
                            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                                mcp_tool = MCPTool(
                                    tool.name,
                                    tool.description,
                                    getattr(tool, 'inputSchema', {})
                                )
                                tools.append(mcp_tool)
                                self.logger.debug(f"Added tool: {tool.name}")
                    elif hasattr(item, 'name') and hasattr(item, 'description'):
                        # 直接的工具对象
                        mcp_tool = MCPTool(
                            item.name,
                            item.description,
                            getattr(item, 'inputSchema', {})
                        )
                        tools.append(mcp_tool)
                        self.logger.debug(f"Added tool directly: {item.name}")
            elif isinstance(tools_response, list):
                # 如果是列表格式
                for item in tools_response:
                    if hasattr(item, 'name') and hasattr(item, 'description'):
                        mcp_tool = MCPTool(
                            item.name,
                            item.description,
                            getattr(item, 'inputSchema', {})
                        )
                        tools.append(mcp_tool)
                        self.logger.debug(f"Added tool from list: {item.name}")

            self._available_tools = tools
            self.logger.info(f"Retrieved {len(tools)} tools from {self.name}")

            # 记录找到的工具详情
            if tools:
                tool_names = [tool.name for tool in tools]
                self.logger.debug(f"Available tools in {self.name}: {tool_names}")
            else:
                self.logger.warning(f"No tools found in {self.name} - checking response format")
                self.logger.debug(f"Response type: {type(tools_response)}")
                if hasattr(tools_response, '__len__'):
                    self.logger.debug(f"Response length: {len(tools_response)}")

            return tools

        except Exception as e:
            self.logger.error(f"Failed to list tools from {self.name}: {str(e)}")
            self.logger.debug(f"Error details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """调用指定的MCP工具，带重试机制"""
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        start_time = asyncio.get_event_loop().time()

        attempt = 0
        while attempt < self.config.max_retries:
            try:
                self.logger.debug(f"Calling tool {tool_name} on {self.name}, attempt {attempt + 1}")

                result = await self.session.call_tool(tool_name, arguments)
                execution_time = asyncio.get_event_loop().time() - start_time

                self.logger.info(f"Tool {tool_name} call completed in {execution_time:.2f}s")

                return MCPResult(
                    success=True,
                    data=result,
                    execution_time=execution_time,
                    tool_name=tool_name,
                    service_name=self.name
                )

            except Exception as e:
                attempt += 1
                self.logger.warning(f"Error executing tool {tool_name}: {e}. Attempt {attempt} of {self.config.max_retries}")

                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    execution_time = asyncio.get_event_loop().time() - start_time
                    self.logger.error(f"Tool {tool_name} failed after {self.config.max_retries} attempts")

                    return MCPResult(
                        success=False,
                        error=f"Tool {tool_name} failed after {self.config.max_retries} attempts: {str(e)}",
                        execution_time=execution_time,
                        tool_name=tool_name,
                        service_name=self.name
                    )

    async def cleanup(self) -> None:
        """清理服务器资源"""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
                self._is_initialized = False
                self._available_tools = []
                self.logger.info(f"MCP client {self.name} cleaned up")
            except Exception as e:
                self.logger.error(f"Error during cleanup of server {self.name}: {e}")


class HTTPMCPClient:
    """HTTP MCP客户端，用于处理基于HTTP的MCP服务"""

    def __init__(self, name: str, config: MCPConnectionConfig, logger: Optional[logging.Logger] = None):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp library is not installed. Install with: pip install aiohttp")

        self.name = name
        self.config = config
        self.logger = logger or get_logger(f"http_mcp_client.{name}")
        self.session: Optional[aiohttp.ClientSession] = None
        self._is_initialized = False
        self._available_tools: List[MCPTool] = []

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def initialize(self):
        """初始化HTTP客户端"""
        if self._is_initialized:
            return

        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp library is not installed")

        self.logger.info(f"Initializing HTTP MCP client for {self.name}")

        # 创建TCP连接器
        connector = TCPConnector(
            limit=100,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )

        # 创建HTTP会话
        timeout = ClientTimeout(
            total=self.config.timeout,
            connect=min(self.config.timeout // 3, 20),
            sock_read=max(self.config.timeout - 10, 40)
        )

        headers = {
            "User-Agent": "MPEO-MCP-Client/2.0",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )

        self._is_initialized = True
        self.logger.info(f"HTTP MCP client {self.name} initialized successfully")

    async def close(self):
        """关闭HTTP客户端"""
        if self.session:
            await self.session.close()
            self.session = None
        self._is_initialized = False
        self.logger.info(f"HTTP MCP client {self.name} closed")

    async def list_tools(self) -> List[MCPTool]:
        """获取可用工具列表"""
        if not self._is_initialized:
            await self.initialize()

        if self._available_tools:
            return self._available_tools

        tools = []
        try:
            # 从args中获取endpoint URL
            endpoint_url = self.config.args[0] if self.config.args else "http://localhost:8080"
            self.logger.debug(f"HTTP MCP client listing tools from: {endpoint_url}")

            async with self.session.post(endpoint_url, json={"action": "list_tools"}) as response:
                self.logger.debug(f"HTTP response status: {response.status}")

                if response.status == 200:
                    response_text = await response.text()
                    self.logger.debug(f"HTTP response text (first 500 chars): {response_text[:500]}")

                    try:
                        data = json.loads(response_text)
                        self.logger.debug(f"Parsed JSON data type: {type(data)}")

                        if isinstance(data, list):
                            self.logger.debug(f"Processing tools list with {len(data)} items")
                            for tool_data in data:
                                if isinstance(tool_data, dict) and "name" in tool_data:
                                    mcp_tool = MCPTool(
                                        name=tool_data.get("name", ""),
                                        description=tool_data.get("description", ""),
                                        input_schema=tool_data.get("input_schema", {})
                                    )
                                    tools.append(mcp_tool)
                                    self.logger.debug(f"Added HTTP tool: {mcp_tool.name}")
                        elif isinstance(data, dict) and "tools" in data:
                            self.logger.debug(f"Processing tools dict with {len(data.get('tools', []))} items")
                            for tool_data in data["tools"]:
                                if isinstance(tool_data, dict) and "name" in tool_data:
                                    mcp_tool = MCPTool(
                                        name=tool_data.get("name", ""),
                                        description=tool_data.get("description", ""),
                                        input_schema=tool_data.get("input_schema", {})
                                    )
                                    tools.append(mcp_tool)
                                    self.logger.debug(f"Added HTTP tool from dict: {mcp_tool.name}")
                        else:
                            self.logger.warning(f"Unexpected JSON structure in tools response: {type(data)}")

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON response when listing tools: {e}")
                        self.logger.debug(f"Response text that failed JSON parsing: {response_text[:200]}")

                    self._available_tools = tools
                    self.logger.info(f"Retrieved {len(tools)} tools from {self.name}")

                    if tools:
                        tool_names = [tool.name for tool in tools]
                        self.logger.debug(f"Available HTTP tools: {tool_names}")

                else:
                    self.logger.warning(f"HTTP request failed with status {response.status}")
                    error_text = await response.text()
                    self.logger.error(f"HTTP error response: {error_text[:200]}")

        except Exception as e:
            self.logger.error(f"Failed to list tools from {self.name}: {str(e)}")
            self.logger.debug(f"HTTP tools listing error details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")

        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """调用指定的MCP工具"""
        if not self._is_initialized:
            await self.initialize()

        start_time = asyncio.get_event_loop().time()

        try:
            # 从args中获取endpoint URL
            endpoint_url = self.config.args[0] if self.config.args else "http://localhost:8080"

            payload = {
                "tool": tool_name,
                "arguments": arguments
            }

            async with self.session.post(endpoint_url, json=payload) as response:
                execution_time = asyncio.get_event_loop().time() - start_time

                if response.status == 200:
                    response_text = await response.text()
                    try:
                        result_data = json.loads(response_text)
                        self.logger.info(f"Tool {tool_name} call completed in {execution_time:.2f}s")
                        return MCPResult(
                            success=True,
                            data=result_data,
                            execution_time=execution_time,
                            tool_name=tool_name,
                            service_name=self.name
                        )
                    except json.JSONDecodeError:
                        return MCPResult(
                            success=True,
                            data=response_text,
                            execution_time=execution_time,
                            tool_name=tool_name,
                            service_name=self.name
                        )
                else:
                    error_text = await response.text()
                    return MCPResult(
                        success=False,
                        error=f"Tool call failed with status {response.status}: {error_text}",
                        execution_time=execution_time,
                        tool_name=tool_name,
                        service_name=self.name
                    )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"Tool {tool_name} failed: {str(e)}")
            return MCPResult(
                success=False,
                error=f"Tool call failed: {str(e)}",
                execution_time=execution_time,
                tool_name=tool_name,
                service_name=self.name
            )


class JSONRPCMCPClient:
    """JSON-RPC MCP客户端，用于处理基于JSON-RPC会话的MCP服务"""

    def __init__(self, name: str, config: MCPConnectionConfig, logger: Optional[logging.Logger] = None):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp library is not installed. Install with: pip install aiohttp")

        self.name = name
        self.config = config
        self.logger = logger or get_logger(f"jsonrpc_mcp_client.{name}")
        self.session: Optional[aiohttp.ClientSession] = None
        self._is_initialized = False
        self._available_tools: List[MCPTool] = []
        self._session_id: Optional[str] = None
        self._endpoint_url: Optional[str] = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def initialize(self):
        """初始化JSON-RPC客户端"""
        if self._is_initialized:
            return

        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp library is not installed")

        self.logger.info(f"Initializing JSON-RPC MCP client for {self.name}")

        # 从args中获取endpoint URL
        self._endpoint_url = self.config.args[0] if self.config.args else "http://localhost:8080"
        self.logger.debug(f"JSON-RPC endpoint: {self._endpoint_url}")

        # 创建TCP连接器
        connector = TCPConnector(
            limit=100,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )

        # 创建HTTP会话
        timeout = ClientTimeout(
            total=self.config.timeout,
            connect=min(self.config.timeout // 3, 20),
            sock_read=max(self.config.timeout - 10, 40),
            sock_connect=10
        )

        headers = {
            "User-Agent": "MPEO-MCP-Client/2.0",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Accept-Encoding": "gzip, deflate",  # 支持压缩
            "Connection": "keep-alive"
        }

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )

        # 初始化MCP会话
        await self._initialize_mcp_session()

        self._is_initialized = True
        self.logger.info(f"JSON-RPC MCP client {self.name} initialized successfully")

    async def _initialize_mcp_session(self):
        """初始化MCP会话"""
        try:
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "MPEO",
                        "version": "2.0"
                    }
                }
            }

            self.logger.debug(f"Sending MCP initialize request to {self._endpoint_url}")

            # 初始化请求不需要session-id头部
            async with self.session.post(self._endpoint_url, json=init_payload) as response:
                if response.status == 200:
                    response_text = await response.text()
                    self.logger.debug(f"MCP initialize response: {response_text[:200]}")

                    try:
                        data = json.loads(response_text)
                        if "result" in data:
                            result = data["result"]
                            server_info = result.get("serverInfo", {})

                            # 尝试从响应中获取会话ID
                            self._session_id = None

                            # 方法1: 从响应headers中获取（如果有）
                            response_headers = response.headers
                            if "mcp-session-id" in response_headers:
                                self._session_id = response_headers["mcp-session-id"]
                                self.logger.debug(f"Got session ID from headers: {self._session_id}")

                            # 方法2: 从result中获取（如果有）
                            if not self._session_id and "sessionId" in result:
                                self._session_id = result["sessionId"]
                                self.logger.debug(f"Got session ID from result: {self._session_id}")

                            # 方法3: 使用固定的session ID进行无状态通信
                            if not self._session_id:
                                self._session_id = "mpeo-client"
                                self.logger.debug(f"Using default session ID: {self._session_id}")

                            self.logger.info(f"MCP session initialized: {server_info}")

                            # 发送 initialized 通知（根据MCP协议规范）
                            await self._send_initialized_notification()

                        else:
                            self.logger.warning(f"Unexpected MCP initialize response: {data}")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Invalid JSON in MCP initialize response: {e}")
                        raise
                else:
                    error_text = await response.text()
                    self.logger.error(f"MCP initialize failed with status {response.status}: {error_text}")
                    raise RuntimeError(f"MCP initialize failed: {error_text}")

        except Exception as e:
            self.logger.error(f"Failed to initialize MCP session: {e}")
            raise

    async def _send_initialized_notification(self):
        """发送initialized通知"""
        try:
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }

            headers = {}
            if self._session_id:
                headers["mcp-session-id"] = self._session_id

            self.logger.debug("Sending MCP initialized notification")

            async with self.session.post(self._endpoint_url, json=initialized_notification, headers=headers) as response:
                if response.status in [200, 202]:  # 202 Accepted is also valid for notifications
                    self.logger.debug("MCP initialized notification sent successfully")
                else:
                    response_text = await response.text()
                    self.logger.warning(f"Initialized notification returned status {response.status}: {response_text}")
                    # 继续执行，不是致命错误

        except Exception as e:
            self.logger.warning(f"Failed to send initialized notification: {e}")
            # 继续执行，不是致命错误

    async def close(self):
        """关闭JSON-RPC客户端"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            self._is_initialized = False
            self._session_id = None
            self.logger.info(f"JSON-RPC MCP client {self.name} closed")
        except Exception as e:
            self.logger.error(f"Error closing JSON-RPC client {self.name}: {e}")
            # 确保状态重置
            self.session = None
            self._is_initialized = False
            self._session_id = None

    async def _send_jsonrpc_request(self, method: str, params: Optional[Dict] = None, max_retries: int = 3) -> Any:
        """发送JSON-RPC请求，带重试机制"""
        if not self._is_initialized:
            await self.initialize()

        request_id = getattr(self, '_request_counter', 1)
        setattr(self, '_request_counter', request_id + 1)

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }

        # 只有当params不为None时才添加
        if params is not None:
            payload["params"] = params

        headers = {}
        if self._session_id:
            headers["mcp-session-id"] = self._session_id

        self.logger.debug(f"Sending JSON-RPC request: {method} (ID: {request_id}) with params: {params}")

        for attempt in range(max_retries):
            try:
                # 使用更长的超时时间和更好的错误处理
                timeout = aiohttp.ClientTimeout(
                    total=60,  # 更长的总超时
                    connect=15,  # 连接超时
                    sock_read=45  # 读取超时
                )

                async with self.session.post(
                    self._endpoint_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=True
                ) as response:

                    if response.status == 200:
                        try:
                            # 尝试读取响应，处理ContentLengthError
                            response_text = await response.text()
                            self.logger.debug(f"JSON-RPC response (attempt {attempt + 1}): {response_text[:200]}")

                            if not response_text:
                                raise RuntimeError("Empty response from server")

                            try:
                                data = json.loads(response_text)
                                if "error" in data:
                                    error = data["error"]
                                    self.logger.error(f"JSON-RPC error: {error}")
                                    raise RuntimeError(f"JSON-RPC error: {error}")
                                return data.get("result")
                            except json.JSONDecodeError as e:
                                self.logger.error(f"Invalid JSON in JSON-RPC response: {e}")
                                self.logger.error(f"Response text: {response_text[:500]}")
                                raise RuntimeError(f"Invalid JSON response: {e}")

                        except aiohttp.ClientPayloadError as e:
                            self.logger.warning(f"Payload error on attempt {attempt + 1}: {e}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(1 * (attempt + 1))  # 指数退避
                                continue
                            else:
                                raise RuntimeError(f"Payload error after {max_retries} attempts: {e}")

                    else:
                        error_text = await response.text()
                        self.logger.error(f"JSON-RPC request failed with status {response.status}: {error_text}")
                        raise RuntimeError(f"JSON-RPC request failed: HTTP {response.status}")

            except aiohttp.ClientConnectorError as e:
                self.logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                else:
                    raise RuntimeError(f"Connection failed after {max_retries} attempts: {e}")

            except asyncio.TimeoutError as e:
                self.logger.warning(f"Timeout error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                else:
                    raise RuntimeError(f"Request timeout after {max_retries} attempts: {e}")

            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                else:
                    raise

        raise RuntimeError(f"Failed to send JSON-RPC request {method} after {max_retries} attempts")

    async def list_tools(self) -> List[MCPTool]:
        """获取可用工具列表"""
        if not self._is_initialized:
            await self.initialize()

        if self._available_tools:
            return self._available_tools

        tools = []
        try:
            self.logger.debug(f"JSON-RPC MCP client listing tools from: {self._endpoint_url}")

            # 使用正确的MCP协议方法
            result = await self._send_jsonrpc_request("tools/list", None)
            self.logger.debug(f"Raw tools response from {self.name}: {type(result)} - {str(result)[:200]}")

            if result is None:
                self.logger.warning("tools/list returned None, returning empty list")
                return []

            if isinstance(result, dict) and "tools" in result:
                tools_list = result["tools"]
                self.logger.debug(f"Processing {len(tools_list)} tools from JSON-RPC response")

                for tool_data in tools_list:
                    if isinstance(tool_data, dict):
                        mcp_tool = MCPTool(
                            name=tool_data.get("name", ""),
                            description=tool_data.get("description", ""),
                            input_schema=tool_data.get("inputSchema", {})
                        )
                        tools.append(mcp_tool)
                        self.logger.debug(f"Added JSON-RPC tool: {mcp_tool.name}")

                self._available_tools = tools
                self.logger.info(f"Retrieved {len(tools)} tools from {self.name}")

                if tools:
                    tool_names = [tool.name for tool in tools]
                    self.logger.debug(f"Available JSON-RPC tools: {tool_names}")

            else:
                self.logger.warning(f"Unexpected JSON-RPC tools response format: {type(result)}")

        except Exception as e:
            self.logger.error(f"Failed to list tools from {self.name}: {str(e)}")
            self.logger.debug(f"JSON-RPC tools listing error details: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")

        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """调用指定的MCP工具"""
        if not self._is_initialized:
            await self.initialize()

        start_time = asyncio.get_event_loop().time()

        try:
            params = {
                "name": tool_name,
                "arguments": arguments
            }

            self.logger.debug(f"Calling JSON-RPC tool: {tool_name} with arguments: {arguments}")

            result = await self._send_jsonrpc_request("tools/call", params)
            execution_time = asyncio.get_event_loop().time() - start_time

            self.logger.info(f"JSON-RPC tool {tool_name} call completed in {execution_time:.2f}s")

            return MCPResult(
                success=True,
                data=result,
                execution_time=execution_time,
                tool_name=tool_name,
                service_name=self.name
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.logger.error(f"JSON-RPC tool {tool_name} failed: {str(e)}")
            return MCPResult(
                success=False,
                error=f"JSON-RPC tool call failed: {str(e)}",
                execution_time=execution_time,
                tool_name=tool_name,
                service_name=self.name
            )


class MCPServiceManager:
    """MCP服务管理器，统一管理多种类型的MCP客户端"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("mcp_service_manager")
        self.clients: Dict[str, Union[MCPClient, HTTPMCPClient, JSONRPCMCPClient]] = {}
        self.stack = AsyncExitStack()
        self._is_initialized = False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def initialize(self):
        """初始化服务管理器"""
        if self._is_initialized:
            return

        self.logger.info("Initializing MCP Service Manager")

        # 自动加载配置文件中的MCP服务
        await self._load_services_from_config()

        self._is_initialized = True

    async def _load_services_from_config(self):
        """从配置文件加载MCP服务"""
        try:
            # 尝试加载配置文件
            config_path = "/vda1/data/MPEO/config/mcp_services.json"
            if os.path.exists(config_path):
                self.logger.info(f"Loading MCP services from {config_path}")

                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                if "mcpServices" in config_data:
                    services = config_data["mcpServices"]
                    self.logger.info(f"Found {len(services)} MCP services in config")

                    for service_name, service_config in services.items():
                        try:
                            # 创建MCPServiceConfig
                            mcp_service_config = MCPServiceConfig(
                                service_name=service_name,
                                service_type=service_config.get("type", "http"),
                                endpoint_url=service_config.get("url", ""),
                                timeout=service_config.get("timeout", 30)
                            )

                            # 注册服务
                            success = await self.register_from_service_config(mcp_service_config)
                            if success:
                                self.logger.info(f"Successfully loaded MCP service: {service_name}")
                            else:
                                self.logger.warning(f"Failed to load MCP service: {service_name}")

                        except Exception as e:
                            self.logger.error(f"Error loading MCP service {service_name}: {e}")

                else:
                    self.logger.warning("No 'mcpServices' section found in config file")
            else:
                self.logger.info(f"MCP config file not found at {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load MCP services config: {e}")

    async def close(self):
        """关闭服务管理器"""
        self.logger.info("Closing MCP Service Manager")
        await self.stack.aclose()
        self.clients.clear()
        self._is_initialized = False

    async def register_service(self, config: MCPConnectionConfig) -> bool:
        """注册MCP服务"""
        try:
            self.logger.info(f"Registering MCP service: {config.name}")

            # 根据配置选择客户端类型
            if config.command == "curl" or (config.args and config.args[0].startswith('http')):
                # 检查是否为JSON-RPC会话式服务
                endpoint_url = config.args[0] if config.args else ""

                # 更精确的JSON-RPC检测逻辑
                is_jsonrpc = (
                    endpoint_url.endswith("/mcp") or  # ModelScope MCP服务
                    ("mcp.api-inference.modelscope.net" in endpoint_url and "/mcp" in endpoint_url) or
                    (endpoint_url.count('/') == 4 and endpoint_url.endswith('/mcp'))  # 格式: https://domain/.../mcp
                )

                if is_jsonrpc:
                    # JSON-RPC会话式客户端
                    if not AIOHTTP_AVAILABLE:
                        self.logger.error(f"aiohttp library not available for JSON-RPC service: {config.name}")
                        self.logger.error("Install aiohttp library with: pip install aiohttp")
                        return False
                    client = JSONRPCMCPClient(config.name, config, self.logger)
                    self.logger.info(f"Using JSON-RPC client for {config.name} at {endpoint_url}")
                else:
                    # 普通HTTP客户端
                    if not AIOHTTP_AVAILABLE:
                        self.logger.error(f"aiohttp library not available for HTTP service: {config.name}")
                        self.logger.error("Install aiohttp library with: pip install aiohttp")
                        return False
                    client = HTTPMCPClient(config.name, config, self.logger)
                    self.logger.info(f"Using HTTP client for {config.name} at {endpoint_url}")
            else:
                # STDIO客户端
                if not MCP_AVAILABLE:
                    self.logger.error(f"MCP library not available for STDIO service: {config.name}")
                    self.logger.error("Install MCP library with: pip install mcp")
                    return False
                client = MCPClient(config.name, config, self.logger)
                self.logger.info(f"Using STDIO client for {config.name}")

            # 将客户端添加到堆栈中管理
            managed_client = await self.stack.enter_async_context(client)

            self.clients[config.name] = managed_client
            self.logger.info(f"Successfully registered MCP service: {config.name}")
            return True

        except ImportError as e:
            self.logger.error(f"Failed to register MCP service {config.name}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to register MCP service {config.name}: {str(e)}")
            return False

    async def register_from_service_config(self, service_config: MCPServiceConfig) -> bool:
        """从MCPServiceConfig注册服务"""
        connection_config = MCPConnectionConfig.from_service_config(service_config)
        return await self.register_service(connection_config)

    async def get_available_tools(self, service_name: Optional[str] = None) -> Dict[str, List[MCPTool]]:
        """获取可用工具列表"""
        tools_dict = {}

        services_to_check = [service_name] if service_name else list(self.clients.keys())

        for name in services_to_check:
            if name in self.clients:
                try:
                    tools = await self.clients[name].list_tools()
                    tools_dict[name] = tools
                except Exception as e:
                    self.logger.error(f"Failed to get tools from {name}: {str(e)}")
                    tools_dict[name] = []

        return tools_dict

    async def call_tool(self, service_name: str, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """调用指定服务的工具"""
        if service_name not in self.clients:
            return MCPResult(
                success=False,
                error=f"Service {service_name} not registered",
                tool_name=tool_name,
                service_name=service_name
            )

        return await self.clients[service_name].call_tool(tool_name, arguments)

    async def find_and_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """在所有注册的服务中查找并调用工具"""
        for service_name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                for tool in tools:
                    if tool.name == tool_name:
                        return await client.call_tool(tool_name, arguments)
            except Exception as e:
                self.logger.warning(f"Failed to check tools in {service_name}: {str(e)}")
                continue

        return MCPResult(
            success=False,
            error=f"Tool {tool_name} not found in any registered service",
            tool_name=tool_name
        )

    def get_registered_services(self) -> List[str]:
        """获取已注册的服务列表"""
        return list(self.clients.keys())

    def is_service_registered(self, service_name: str) -> bool:
        """检查服务是否已注册"""
        return service_name in self.clients

    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """获取服务状态"""
        if service_name not in self.clients:
            return {
                "service": service_name,
                "status": "not_registered",
                "tools_count": 0
            }

        try:
            tools = await self.clients[service_name].list_tools()
            return {
                "service": service_name,
                "status": "active",
                "tools_count": len(tools),
                "tools": [tool.to_dict() for tool in tools]
            }
        except Exception as e:
            return {
                "service": service_name,
                "status": "error",
                "error": str(e),
                "tools_count": 0
            }