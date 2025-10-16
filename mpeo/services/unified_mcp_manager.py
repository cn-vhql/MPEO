"""
统一的MCP服务管理器 - 整合所有MCP相关功能
替代原有的多个MCP管理器实现，提供统一、高效的接口
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, _AsyncGeneratorContextManager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Optional imports with graceful fallback
try:
    import mcp.types
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    stdio_client = None
    sse_client = None
    streamablehttp_client = None

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


# 统一的数据结构定义
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
    service_type: str = "http"  # stdio, sse, streamable_http, jsonrpc
    endpoint_url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = None
    env: Dict[str, str] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}
        if self.headers is None:
            self.headers = {}

    @classmethod
    def from_service_config(cls, service_config) -> "MCPConnectionConfig":
        """从MCPServiceConfig创建连接配置"""
        if service_config.service_type == "stdio":
            # STDIO类型服务
            if service_config.endpoint_url.startswith("stdio://"):
                endpoint_info = service_config.endpoint_url[7:]
                parts = endpoint_info.split("/", 1)
                command = parts[0] if parts else "npx"
                args = parts[1].split("/") if len(parts) > 1 and parts[1] else []
                if command == "npx" and args:
                    args = ["-y"] + args
            else:
                command = "npx"
                args = []

            return cls(
                name=service_config.service_name,
                service_type="stdio",
                command=command,
                args=args,
                env={},
                timeout=service_config.timeout or 60
            )
        else:
            # HTTP/SSE类型服务
            service_type = service_config.service_type.lower()
            if service_type not in ["sse", "streamable_http", "http", "jsonrpc"]:
                service_type = "http"

            # 检查是否为JSON-RPC会话式服务
            endpoint_url = service_config.endpoint_url
            if service_type == "jsonrpc" or (endpoint_url and "mcp.api-inference.modelscope.net" in endpoint_url):
                service_type = "streamable_http"

            return cls(
                name=service_config.service_name,
                service_type=service_type,
                endpoint_url=endpoint_url,
                timeout=service_config.timeout or 60,
                headers=service_config.headers or {}
            )


# 工具注册表
class ToolRegistry:
    """工具注册表，管理工具名称到服务的映射"""

    def __init__(self):
        self._tools: Dict[str, str] = {}  # tool_name -> service_name
        self._service_tools: Dict[str, List[MCPTool]] = {}  # service_name -> tools

    def register_tools(self, service_name: str, tools: List[MCPTool]):
        """注册服务的工具"""
        self._service_tools[service_name] = tools
        for tool in tools:
            self._tools[tool.name] = service_name

    def find_service(self, tool_name: str) -> Optional[str]:
        """查找工具所属的服务"""
        return self._tools.get(tool_name)

    def get_service_tools(self, service_name: str) -> List[MCPTool]:
        """获取服务的所有工具"""
        return self._service_tools.get(service_name, [])

    def get_all_tools(self) -> Dict[str, List[MCPTool]]:
        """获取所有服务的工具"""
        return self._service_tools.copy()

    def clear_service(self, service_name: str):
        """清除服务的工具注册"""
        if service_name in self._service_tools:
            for tool in self._service_tools[service_name]:
                self._tools.pop(tool.name, None)
            del self._service_tools[service_name]


# 抽象客户端基类
class MCPClientBase(ABC):
    """MCP客户端基类"""

    def __init__(self, name: str, config: MCPConnectionConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"mcp_client.{name}")
        self.stack = AsyncExitStack()
        self._is_connected = False
        self._cached_tools: List[MCPTool] = []

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    @abstractmethod
    async def connect(self) -> None:
        """连接到MCP服务器"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭连接"""
        pass

    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """获取可用工具列表"""
        pass

    def _validate_connection(self) -> None:
        """验证连接状态"""
        if not self._is_connected:
            raise RuntimeError(f"Client {self.name} not connected")


# STDIO客户端实现
class StdioClient(MCPClientBase):
    """STDIO有状态客户端"""

    def __init__(self, name: str, config: MCPConnectionConfig):
        if not MCP_AVAILABLE:
            raise ImportError("MCP library is not installed. Install with: pip install mcp")
        super().__init__(name, config)
        self._session: Optional[ClientSession] = None
        self._client: Optional[Any] = None

    async def connect(self) -> None:
        """连接到STDIO服务器"""
        if self._is_connected:
            return

        self.logger.info(f"Connecting STDIO client {self.name}")

        command = (
            shutil.which("npx")
            if self.config.command == "npx"
            else self.config.command
        )

        if command is None:
            raise ValueError("Command must be a valid string")

        try:
            # 创建STDIO客户端
            self._client = stdio_client({
                "command": command,
                "args": self.config.args,
                "env": {**os.environ, **self.config.env} if self.config.env else None,
            })

            # 进入上下文
            context = await self.stack.enter_async_context(self._client)
            read_stream, write_stream = context[0], context[1]

            # 创建会话
            self._session = ClientSession(read_stream, write_stream)
            await self.stack.enter_async_context(self._session)

            # 初始化会话
            await self._session.initialize()

            self._is_connected = True
            self.logger.info(f"STDIO client {self.name} connected successfully")

        except Exception as e:
            self.logger.error(f"Failed to connect STDIO client {self.name}: {e}")
            await self.close()
            raise

    async def close(self) -> None:
        """关闭连接"""
        if not self._is_connected:
            return

        try:
            await self.stack.aclose()
            self._session = None
            self._client = None
            self._is_connected = False
            self._cached_tools.clear()
            self.logger.info(f"STDIO client {self.name} closed")
        except Exception as e:
            self.logger.error(f"Error closing STDIO client {self.name}: {e}")

    async def list_tools(self) -> List[MCPTool]:
        """获取工具列表"""
        self._validate_connection()

        if self._cached_tools:
            return self._cached_tools

        try:
            response = await self._session.list_tools()
            tools = []

            for tool in response.tools:
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=getattr(tool, 'inputSchema', {})
                )
                tools.append(mcp_tool)

            self._cached_tools = tools
            self.logger.info(f"Retrieved {len(tools)} tools from {self.name}")
            return tools

        except Exception as e:
            self.logger.error(f"Failed to list tools from {self.name}: {e}")
            return []


# HTTP客户端实现
class HttpClient(MCPClientBase):
    """HTTP无状态客户端"""

    def __init__(self, name: str, config: MCPConnectionConfig):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp library is not installed. Install with: pip install aiohttp")
        super().__init__(name, config)
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_id: Optional[str] = None

    def get_session_id(self) -> Optional[str]:
        """获取当前会话ID"""
        return self._session_id

    def set_session_id(self, session_id: str) -> None:
        """设置会话ID"""
        self._session_id = session_id

    async def connect(self) -> None:
        """连接检查（无状态客户端不需要持久连接）"""
        if self._is_connected:
            return

        # 创建TCP连接器 - 优化配置以处理大文件下载
        connector = TCPConnector(
            limit=50,  # 减少并发连接数，避免资源竞争
            limit_per_host=5,  # 减少每个主机的并发连接
            ttl_dns_cache=600,  # 增加DNS缓存时间
            use_dns_cache=True,
            keepalive_timeout=120,  # 增加keepalive时间
            enable_cleanup_closed=True,
            force_close=False,  # 允许连接复用
            family=0,  # 允许IPv4和IPv6
            ssl=False  # 对于HTTP连接，不需要SSL
        )

        # 创建HTTP会话 - 增加超时时间以处理大文件下载
        timeout = ClientTimeout(
            total=self.config.timeout,
            connect=min(self.config.timeout // 4, 30),  # 连接超时
            sock_read=max(self.config.timeout - 30, 60)  # 读取超时，给更多时间处理大文件
        )

        headers = {
            "User-Agent": "MPEO-MCP-Client/2.0",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.config.headers
        }

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )

        self._is_connected = True
        self.logger.info(f"HTTP client {self.name} ready (stateless)")

    async def close(self) -> None:
        """关闭连接"""
        if not self._is_connected:
            return

        if self.session:
            await self.session.close()
            self.session = None

        self._is_connected = False
        self._cached_tools.clear()
        self.logger.info(f"HTTP client {self.name} closed")

    async def list_tools(self) -> List[MCPTool]:
        """获取工具列表"""
        if self._cached_tools:
            return self._cached_tools

        tools = []
        try:
            endpoint_url = self.config.endpoint_url
            self.logger.debug(f"HTTP client listing tools from: {endpoint_url}")

            # 检查是否为JSON-RPC端点
            if self.config.service_type == "streamable_http" or "mcp.api-inference.modelscope.net" in endpoint_url:
                # JSON-RPC方式
                tools = await self._list_tools_jsonrpc()
            else:
                # 简单HTTP方式
                tools = await self._list_tools_http()

            self._cached_tools = tools
            self.logger.info(f"Retrieved {len(tools)} tools from {self.name}")
            return tools

        except Exception as e:
            self.logger.error(f"Failed to list tools from {self.name}: {str(e)}")
            return []

    async def _list_tools_http(self) -> List[MCPTool]:
        """简单HTTP方式获取工具列表"""
        async with self.session.post(self.config.endpoint_url, json={"action": "list_tools"}) as response:
            if response.status == 200:
                response_text = await response.text()
                try:
                    data = json.loads(response_text)
                    tools = []

                    if isinstance(data, list):
                        for tool_data in data:
                            if isinstance(tool_data, dict) and "name" in tool_data:
                                tools.append(MCPTool(
                                    name=tool_data.get("name", ""),
                                    description=tool_data.get("description", ""),
                                    input_schema=tool_data.get("input_schema", {})
                                ))
                    elif isinstance(data, dict) and "tools" in data:
                        for tool_data in data["tools"]:
                            if isinstance(tool_data, dict) and "name" in tool_data:
                                tools.append(MCPTool(
                                    name=tool_data.get("name", ""),
                                    description=tool_data.get("description", ""),
                                    input_schema=tool_data.get("input_schema", {})
                                ))

                    return tools
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON response when listing tools")
                    return []
            else:
                self.logger.warning(f"HTTP request failed with status {response.status}")
                return []

    async def _list_tools_jsonrpc(self) -> List[MCPTool]:
        """会话式MCP协议方式获取工具列表"""
        try:
            # 初始化会话 - 不提供session ID，让服务器分配
            session_id = self._session_id  # 可能为None
            self.logger.debug(f"Using MCP session: {session_id or 'new session'}")

            # 初始化会话
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "MPEO", "version": "2.0"}
                }
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }

            # 如果有现有会话ID，则使用它
            if session_id:
                headers["mcp-session-id"] = session_id

            async with self.session.post(self.config.endpoint_url, json=init_payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"MCP initialize failed: HTTP {response.status}, Response: {error_text}")
                    raise RuntimeError(f"MCP initialize failed: HTTP {response.status}")

                init_response = await response.json()
                self.logger.debug(f"Initialize response: {init_response}")

                # 检查初始化响应
                if "error" in init_response:
                    raise RuntimeError(f"MCP initialize error: {init_response['error']}")

                # 从响应头中提取会话ID
                response_session_id = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
                if response_session_id:
                    self.logger.info(f"Server provided session ID: {response_session_id}")
                    session_id = response_session_id
                    self.set_session_id(session_id)
                elif not session_id:
                    # 如果服务器没有提供会话ID且我们没有现有会话，这可能是问题
                    self.logger.warning("No session ID provided by server")

            # 发送 initialized 通知
            init_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            if session_id:
                headers["mcp-session-id"] = session_id

            async with self.session.post(self.config.endpoint_url, json=init_notification, headers=headers) as response:
                if response.status not in [200, 202, 204]:  # 通知可能返回204或202
                    self.logger.warning(f"Initialized notification returned status {response.status}")

            # 获取工具列表
            tools_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }

            if session_id:
                headers["mcp-session-id"] = session_id

            async with self.session.post(self.config.endpoint_url, json=tools_payload, headers=headers) as response:
                if response.status == 200:
                    response_text = await response.text()
                    data = json.loads(response_text)
                    self.logger.debug(f"Tools list response: {response_text[:500]}")

                    if "error" in data:
                        raise RuntimeError(f"Tools list error: {data['error']}")

                    if "result" in data and "tools" in data["result"]:
                        tools_list = data["result"]["tools"]
                        tools = []
                        for tool_data in tools_list:
                            if isinstance(tool_data, dict):
                                tools.append(MCPTool(
                                    name=tool_data.get("name", ""),
                                    description=tool_data.get("description", ""),
                                    input_schema=tool_data.get("inputSchema", {})
                                ))
                        self.logger.info(f"Successfully retrieved {len(tools)} tools via session protocol")
                        return tools
                    else:
                        self.logger.warning(f"Unexpected MCP tools response structure: {data}")
                        return []
                else:
                    error_text = await response.text()
                    self.logger.error(f"Tools list failed: HTTP {response.status}, Response: {error_text}")
                    return []

        except Exception as e:
            self.logger.error(f"Session-based MCP tools listing failed: {str(e)}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return []


# 统一的MCP服务管理器
class UnifiedMCPManager:
    """统一的MCP服务管理器，整合所有MCP功能"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("unified_mcp_manager")
        self.clients: Dict[str, MCPClientBase] = {}
        self.tool_registry = ToolRegistry()
        self.stack = AsyncExitStack()
        self._is_initialized = False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def initialize(self) -> None:
        """初始化服务管理器"""
        if self._is_initialized:
            return

        self.logger.info("Initializing Unified MCP Service Manager")
        await self._load_services_from_config()
        self._is_initialized = True

    async def _load_services_from_config(self) -> None:
        """从配置文件加载MCP服务"""
        try:
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
                            mcp_service_config = MCPServiceConfig(
                                service_name=service_name,
                                service_type=service_config.get("type", "http"),
                                endpoint_url=service_config.get("url", ""),
                                timeout=service_config.get("timeout", 30),
                                headers=service_config.get("headers", {})
                            )

                            success = await self.register_service(mcp_service_config)
                            if success:
                                self.logger.info(f"Successfully loaded MCP service: {service_name}")
                            else:
                                self.logger.warning(f"Failed to load MCP service: {service_name}")

                        except Exception as e:
                            self.logger.error(f"Error loading MCP service {service_name}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to load MCP services config: {e}")

    async def close(self) -> None:
        """关闭服务管理器"""
        self.logger.info("Closing Unified MCP Service Manager")
        await self.stack.aclose()
        self.clients.clear()
        self.tool_registry._tools.clear()
        self.tool_registry._service_tools.clear()
        self._is_initialized = False

    async def register_service(self, service_config: MCPServiceConfig) -> bool:
        """注册MCP服务"""
        try:
            self.logger.info(f"Registering MCP service: {service_config.service_name}")

            # 创建连接配置
            connection_config = MCPConnectionConfig.from_service_config(service_config)

            # 根据服务类型创建客户端
            if connection_config.service_type == "stdio":
                if not MCP_AVAILABLE:
                    self.logger.error(f"MCP library not available for STDIO service: {service_config.service_name}")
                    return False
                client = StdioClient(connection_config.name, connection_config)
            else:
                if not AIOHTTP_AVAILABLE:
                    self.logger.error(f"aiohttp library not available for HTTP service: {service_config.service_name}")
                    return False
                client = HttpClient(connection_config.name, connection_config)

            # 将客户端添加到堆栈中管理
            managed_client = await self.stack.enter_async_context(client)
            self.clients[connection_config.name] = managed_client

            # 获取并注册工具
            try:
                tools = await managed_client.list_tools()
                self.tool_registry.register_tools(connection_config.name, tools)
                self.logger.info(f"Registered {len(tools)} tools for service {connection_config.name}")
            except Exception as e:
                self.logger.warning(f"Failed to load tools for {connection_config.name}: {e}")

            self.logger.info(f"Successfully registered MCP service: {connection_config.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register MCP service {service_config.service_name}: {e}")
            return False

    async def get_available_tools(self, service_name: Optional[str] = None) -> Dict[str, List[MCPTool]]:
        """获取可用工具列表"""
        if service_name:
            if service_name in self.clients:
                return {service_name: self.tool_registry.get_service_tools(service_name)}
            else:
                return {}
        else:
            return self.tool_registry.get_all_tools()

    async def call_tool(self, service_name: str, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """调用指定服务的工具"""
        if service_name not in self.clients:
            return MCPResult(
                success=False,
                error=f"Service {service_name} not registered",
                tool_name=tool_name,
                service_name=service_name
            )

        client = self.clients[service_name]
        start_time = asyncio.get_event_loop().time()
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # 这里需要实现具体的工具调用逻辑
                # 由于不同客户端的调用方式不同，需要在基类中定义统一的调用接口
                if hasattr(client, 'call_tool'):
                    result = await client.call_tool(tool_name, arguments)
                else:
                    # 简单实现，实际应该根据客户端类型进行调用
                    result = await self._call_tool_generic(client, tool_name, arguments)

                execution_time = asyncio.get_event_loop().time() - start_time
                self.logger.info(f"Tool {tool_name} call completed in {execution_time:.2f}s")

                return MCPResult(
                    success=True,
                    data=result,
                    execution_time=execution_time,
                    tool_name=tool_name,
                    service_name=service_name
                )

            except Exception as e:
                execution_time = asyncio.get_event_loop().time() - start_time
                error_str = str(e)

                # 检查是否为可重试的错误
                if attempt < max_retries - 1 and self._is_retryable_error(error_str):
                    self.logger.warning(f"Tool {tool_name} call failed (attempt {attempt + 1}/{max_retries}): {error_str}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避
                    continue

                self.logger.error(f"Failed to call tool {tool_name}: {error_str}")
                return MCPResult(
                    success=False,
                    error=f"Tool call failed: {error_str}",
                    execution_time=execution_time,
                    tool_name=tool_name,
                    service_name=service_name
                )

    def _is_retryable_error(self, error_str: str) -> bool:
        """判断错误是否可重试"""
        retryable_errors = [
            "ContentLengthError",
            "ConnectionError",
            "TimeoutError",
            "ServerDisconnectedError",
            "ClientPayloadError",
            "HTTP 500",  # 服务器内部错误
            "HTTP 502",  # 网关错误
            "HTTP 503",  # 服务不可用
            "HTTP 504",  # 网关超时
        ]

        return any(error in error_str for error in retryable_errors)

    async def _call_tool_generic(self, client: MCPClientBase, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """通用工具调用方法（需要根据实际客户端类型实现）"""
        # 这是一个简化实现，实际应该根据客户端类型进行不同的调用
        if isinstance(client, HttpClient):
            # HTTP客户端调用
            if client.config.service_type == "streamable_http" or "mcp.api-inference.modelscope.net" in client.config.endpoint_url:
                # 会话式MCP调用
                session_id = client.get_session_id() or str(uuid.uuid4())
                headers = {
                    "mcp-session-id": session_id,
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }

                # 初始化会话
                init_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "MPEO", "version": "2.0"}
                    }
                }

                async with client.session.post(client.config.endpoint_url, json=init_payload, headers=headers) as response:
                    if response.status != 200:
                        raise RuntimeError(f"MCP session initialize failed: HTTP {response.status}")

                    # 从响应头中提取会话ID
                    response_session_id = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
                    if response_session_id and response_session_id != session_id:
                        client.logger.info(f"Server provided new session ID for tool call: {response_session_id}")
                        session_id = response_session_id
                        client.set_session_id(session_id)
                        headers["mcp-session-id"] = session_id

                # 发送 initialized通知
                init_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }

                async with client.session.post(client.config.endpoint_url, json=init_notification, headers=headers) as response:
                    if response.status not in [200, 202, 204]:
                        client.logger.warning(f"Initialized notification returned status {response.status}")

                # 调用工具
                payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                }

                async with client.session.post(client.config.endpoint_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            if "error" in data:
                                raise RuntimeError(f"Tool call error: {data['error']}")
                            return data.get("result")
                        except aiohttp.ClientPayloadError as e:
                            if "ContentLengthError" in str(e):
                                client.logger.warning(f"Content length error for {tool_name}: {str(e)}")
                                # 尝试获取响应内容的一部分，至少提供部分结果
                                try:
                                    partial_text = await response.text()
                                    return {
                                        "error": "ContentLengthError",
                                        "partial_content": partial_text[:1000] + "..." if len(partial_text) > 1000 else partial_text,
                                        "message": "Received incomplete response from server"
                                    }
                                except Exception:
                                    return {
                                        "error": "ContentLengthError",
                                        "message": "Server returned incomplete data, unable to retrieve content"
                                    }
                            else:
                                raise
                        except json.JSONDecodeError as e:
                            client.logger.error(f"JSON decode error for {tool_name}: {str(e)}")
                            # 尝试获取原始文本响应
                            try:
                                raw_text = await response.text()
                                return {
                                    "error": "JSONDecodeError",
                                    "raw_response": raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text,
                                    "message": "Failed to decode server response as JSON"
                                }
                            except Exception:
                                return {
                                    "error": "JSONDecodeError",
                                    "message": "Failed to decode server response and could not retrieve raw content"
                                }
                    else:
                        error_text = await response.text()
                        raise RuntimeError(f"Tool call failed: HTTP {response.status}, {error_text}")
            else:
                # 简单HTTP调用
                payload = {"tool": tool_name, "arguments": arguments}
                async with client.session.post(client.config.endpoint_url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise RuntimeError(f"Tool call failed: HTTP {response.status}")
        else:
            # STDIO客户端调用
            if hasattr(client, '_session') and client._session:
                return await client._session.call_tool(tool_name, arguments)
            else:
                raise RuntimeError("Client session not available")

    async def find_and_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """在所有注册的服务中查找并调用工具"""
        service_name = self.tool_registry.find_service(tool_name)
        if service_name:
            return await self.call_tool(service_name, tool_name, arguments)
        else:
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
            tools = self.tool_registry.get_service_tools(service_name)
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

    async def health_check(self) -> Dict[str, Any]:
        """健康检查所有注册的服务"""
        results = {}

        for service_name in list(self.clients.keys()):
            try:
                status = await self.get_service_status(service_name)
                results[service_name] = {
                    "healthy": status["status"] == "active",
                    "tools_count": status["tools_count"],
                    "status": status["status"],
                    "error": status.get("error")
                }
            except Exception as e:
                results[service_name] = {
                    "healthy": False,
                    "tools_count": 0,
                    "status": "error",
                    "error": str(e)
                }

        total_services = len(results)
        healthy_services = sum(1 for r in results.values() if r["healthy"])

        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": total_services - healthy_services,
            "services": results
        }

    async def refresh_tools(self, service_name: Optional[str] = None) -> None:
        """刷新工具缓存"""
        services_to_refresh = [service_name] if service_name else list(self.clients.keys())

        for name in services_to_refresh:
            if name in self.clients:
                try:
                    client = self.clients[name]
                    tools = await client.list_tools()
                    self.tool_registry.register_tools(name, tools)
                    self.logger.info(f"Refreshed {len(tools)} tools for service {name}")
                except Exception as e:
                    self.logger.error(f"Failed to refresh tools for {name}: {e}")


# 为了向后兼容，提供别名
MCPServiceManager = UnifiedMCPManager
OptimizedMCPServiceManager = UnifiedMCPManager