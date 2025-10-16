"""
优化的MCP客户端模块 - 基于mcp_demo的优秀设计模式
提供统一、可靠的MCP服务调用接口
"""

import asyncio
import json
import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, _AsyncGeneratorContextManager
from typing import Any, Callable, Dict, List, Optional, Union

# MCP SDK imports
import mcp.types
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client

MCP_AVAILABLE = True

from ..models import MCPServiceConfig, TaskNode
from ..utils.logging import get_logger
from .mcp_common import MCPTool, MCPResult, MCPConnectionConfig, MCPClientBase


class MCPToolFunction:
    """MCP工具函数包装器，类似于mcp_demo的实现"""

    def __init__(
        self,
        mcp_name: str,
        tool: Any,
        wrap_tool_result: bool = True,
        client_gen: Optional[Callable[..., _AsyncGeneratorContextManager[Any]]] = None,
        session: Optional[ClientSession] = None,
    ):
        """初始化MCP工具函数"""
        self.mcp_name = mcp_name
        self.name = tool.name
        self.description = tool.description
        self.input_schema = getattr(tool, 'inputSchema', {})
        self.wrap_tool_result = wrap_tool_result

        # 验证参数
        if (client_gen is None and session is None) or (client_gen is not None and session is not None):
            raise ValueError("Either client_gen or session must be provided, but not both.")

        self.client_gen = client_gen
        self.session = session
        self.logger = get_logger(f"mcp_tool.{mcp_name}.{tool.name}")

    async def __call__(self, **kwargs: Any) -> MCPResult:
        """调用MCP工具函数"""
        start_time = asyncio.get_event_loop().time()

        try:
            if self.client_gen:
                # 无状态客户端，每次调用都创建新会话
                async with self.client_gen() as cli:
                    read_stream, write_stream = cli[0], cli[1]
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.call_tool(self.name, arguments=kwargs)
            else:
                # 有状态客户端，使用现有会话
                if not self.session:
                    raise RuntimeError("Session not initialized for stateful client")
                result = await self.session.call_tool(self.name, arguments=kwargs)

            execution_time = asyncio.get_event_loop().time() - start_time

            # 转换结果
            if self.wrap_tool_result:
                data = self._convert_mcp_content_to_blocks(result.content)
            else:
                data = result

            self.logger.info(f"Tool {self.name} call completed in {execution_time:.2f}s")

            return MCPResult(
                success=True,
                data=data,
                execution_time=execution_time,
                tool_name=self.name,
                service_name=self.mcp_name
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_msg = f"Tool {self.name} call failed: {str(e)}"
            self.logger.error(error_msg)

            return MCPResult(
                success=False,
                error=error_msg,
                execution_time=execution_time,
                tool_name=self.name,
                service_name=self.mcp_name
            )

    def _convert_mcp_content_to_blocks(self, mcp_content_blocks: List[Any]) -> List[Dict[str, Any]]:
        """转换MCP内容为统一格式"""
        content = []
        for block in mcp_content_blocks:
            if isinstance(block, mcp.types.TextContent):
                content.append({
                    "type": "text",
                    "text": block.text
                })
            elif isinstance(block, mcp.types.ImageContent):
                content.append({
                    "type": "image",
                    "mime_type": block.mimeType,
                    "data": block.data
                })
            elif isinstance(block, mcp.types.AudioContent):
                content.append({
                    "type": "audio",
                    "mime_type": block.mimeType,
                    "data": block.data
                })
            else:
                self.logger.warning(f"Unsupported content type: {type(block)}")
        return content




class StdioStatefulClient(MCPClientBase):
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

        from mcp import StdioServerParameters
        server_params = StdioServerParameters(
            command=command,
            args=self.config.args,
            env={**os.environ, **self.config.env} if self.config.env else None,
        )

        try:
            # 创建STDIO客户端
            self._client = stdio_client(server_params)

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


class HttpStatelessClient(MCPClientBase):
    """HTTP无状态客户端"""

    def __init__(self, name: str, config: MCPConnectionConfig):
        super().__init__(name, config)
        self.client_config = {
            "url": config.endpoint_url,
            "headers": config.headers,
            "timeout": config.timeout,
        }

    async def connect(self) -> None:
        """连接检查（无状态客户端不需要持久连接）"""
        self._is_connected = True
        self.logger.info(f"HTTP client {self.name} ready (stateless)")

    async def close(self) -> None:
        """关闭连接"""
        self._is_connected = False
        self._cached_tools.clear()
        self.logger.info(f"HTTP client {self.name} closed")

    def _get_client(self) -> _AsyncGeneratorContextManager[Any]:
        """获取客户端实例"""
        if not MCP_AVAILABLE:
            raise ImportError("MCP library is not installed. Install with: pip install mcp")

        if self.config.service_type == "sse":
            if sse_client is None:
                raise ImportError("SSE client not available in MCP library")
            return sse_client(**self.client_config)
        elif self.config.service_type == "streamable_http":
            if streamablehttp_client is None:
                raise ImportError("Streamable HTTP client not available in MCP library")
            return streamablehttp_client(**self.client_config)
        else:
            raise ValueError(f"Unsupported transport type: {self.config.service_type}")

    async def list_tools(self) -> List[MCPTool]:
        """获取工具列表"""
        if self._cached_tools:
            return self._cached_tools

        try:
            async with self._get_client() as cli:
                read_stream, write_stream = cli[0], cli[1]
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    response = await session.list_tools()

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

    async def get_callable_function(self, func_name: str, wrap_tool_result: bool = True) -> MCPToolFunction:
        """获取可调用的工具函数"""
        if not self._cached_tools:
            await self.list_tools()

        # 查找工具
        target_tool = None
        for tool in self._cached_tools:
            if tool.name == func_name:
                target_tool = tool
                break

        if not target_tool:
            raise ValueError(f"Tool '{func_name}' not found in service '{self.name}'")

        # 创建工具对象
        tool_obj = type('Tool', (), {
            'name': target_tool.name,
            'description': target_tool.description,
            'inputSchema': target_tool.input_schema
        })()

        return MCPToolFunction(
            mcp_name=self.name,
            tool=tool_obj,
            wrap_tool_result=wrap_tool_result,
            client_gen=self._get_client
        )


class OptimizedMCPServiceManager:
    """优化的MCP服务管理器，参照mcp_demo的优秀设计模式"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("optimized_mcp_manager")
        self.clients: Dict[str, MCPClientBase] = {}
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

        self.logger.info("Initializing Optimized MCP Service Manager")
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
                            # 创建MCPServiceConfig
                            mcp_service_config = MCPServiceConfig(
                                service_name=service_name,
                                service_type=service_config.get("type", "http"),
                                endpoint_url=service_config.get("url", ""),
                                timeout=service_config.get("timeout", 30),
                                headers=service_config.get("headers", {})
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

    async def close(self) -> None:
        """关闭服务管理器"""
        self.logger.info("Closing Optimized MCP Service Manager")
        await self.stack.aclose()
        self.clients.clear()
        self._is_initialized = False

    async def register_service(self, config: MCPConnectionConfig) -> bool:
        """注册MCP服务"""
        try:
            self.logger.info(f"Registering MCP service: {config.name}")

            # 根据服务类型创建客户端
            if config.service_type == "stdio":
                client = StdioStatefulClient(config.name, config)

            elif config.service_type in ["sse", "streamable_http", "http", "jsonrpc"]:
                # HTTP类型服务，根据URL判断是否为MCP标准端点
                if config.endpoint_url and config.endpoint_url.endswith("/mcp"):
                    # MCP标准JSON-RPC端点，转换为合适的类型
                    config.service_type = "streamable_http"
                elif config.service_type == "jsonrpc":
                    # JSON-RPC类型的HTTP服务，使用streamable_http
                    config.service_type = "streamable_http"

                client = HttpStatelessClient(config.name, config)
            else:
                self.logger.error(f"Unsupported service type: {config.service_type}")
                return False

            # 将客户端添加到堆栈中管理
            managed_client = await self.stack.enter_async_context(client)

            self.clients[config.name] = managed_client
            self.logger.info(f"Successfully registered MCP service: {config.name} ({config.service_type})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register MCP service {config.name}: {e}")
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
                    self.logger.error(f"Failed to get tools from {name}: {e}")
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

        try:
            # 获取工具函数
            tool_func = await self.clients[service_name].get_callable_function(tool_name)
            # 调用工具
            return await tool_func(**arguments)

        except Exception as e:
            self.logger.error(f"Failed to call tool {tool_name} from {service_name}: {e}")
            return MCPResult(
                success=False,
                error=f"Tool call failed: {str(e)}",
                tool_name=tool_name,
                service_name=service_name
            )

    async def find_and_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """在所有注册的服务中查找并调用工具"""
        for service_name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                for tool in tools:
                    if tool.name == tool_name:
                        return await self.call_tool(service_name, tool_name, arguments)
            except Exception as e:
                self.logger.warning(f"Failed to check tools in {service_name}: {e}")
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