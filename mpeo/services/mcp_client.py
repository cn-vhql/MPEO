"""
MCP客户端模块 - 提供统一的MCP服务调用接口
参考mcp_demo.py的设计模式，实现更可靠的MCP调用逻辑
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from aiohttp import ClientTimeout, ClientConnectorError, ClientPayloadError

from ..models import MCPServiceConfig, TaskNode
from ..utils.logging import get_logger


@dataclass
class MCPTool:
    """MCP工具信息"""
    name: str
    description: str
    input_schema: Dict[str, Any]

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "tool_name": self.tool_name
        }


@dataclass
class MCPConnectionConfig:
    """MCP连接配置"""
    name: str
    endpoint_url: str
    service_type: str = "sse"
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_service_config(cls, config: MCPServiceConfig) -> "MCPConnectionConfig":
        """从MCPServiceConfig创建连接配置"""
        return cls(
            name=config.service_name,
            endpoint_url=config.endpoint_url,
            service_type=config.service_type,
            timeout=config.timeout or 60,
            headers=config.headers or {}
        )


class EnhancedMCPClient:
    """增强的MCP客户端，支持多种连接方式和错误恢复"""

    def __init__(self, config: MCPConnectionConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or get_logger(f"mcp_client.{config.name}")
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector: Optional[aiohttp.TCPConnector] = None
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
        """初始化客户端连接"""
        if self._is_initialized:
            return

        self.logger.info(f"Initializing MCP client for {self.config.name}")

        # 创建TCP连接器
        self.connector = aiohttp.TCPConnector(
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
            **self.config.headers
        }

        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers=headers
        )

        self._is_initialized = True
        self.logger.info(f"MCP client {self.config.name} initialized successfully")

    async def close(self):
        """关闭客户端连接"""
        if self.session:
            await self.session.close()
            self.session = None
        if self.connector:
            await self.connector.close()
            self.connector = None
        self._is_initialized = False
        self.logger.info(f"MCP client {self.config.name} closed")

    async def list_tools(self) -> List[MCPTool]:
        """获取可用工具列表"""
        if not self._is_initialized:
            await self.initialize()

        # 如果已经缓存了工具列表，直接返回
        if self._available_tools:
            return self._available_tools

        try:
            if self.config.service_type.lower() == "sse":
                tools = await self._list_sse_tools()
            else:
                tools = await self._list_http_tools()

            self._available_tools = tools
            self.logger.info(f"Retrieved {len(tools)} tools from {self.config.name}")
            return tools

        except Exception as e:
            self.logger.error(f"Failed to list tools from {self.config.name}: {str(e)}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """调用指定的MCP工具"""
        if not self._is_initialized:
            await self.initialize()

        start_time = time.time()

        # 重试逻辑
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Calling tool {tool_name} on {self.config.name}, attempt {attempt + 1}")

                if self.config.service_type.lower() == "sse":
                    result = await self._call_sse_tool(tool_name, arguments)
                else:
                    result = await self._call_http_tool(tool_name, arguments)

                execution_time = time.time() - start_time
                result.execution_time = execution_time

                self.logger.info(f"Tool {tool_name} call completed in {execution_time:.2f}s")
                return result

            except asyncio.TimeoutError:
                error_msg = f"Tool {tool_name} timeout on attempt {attempt + 1}"
                self.logger.warning(error_msg)
                if attempt == self.config.max_retries - 1:
                    return MCPResult(
                        success=False,
                        error=f"Tool call timeout after {self.config.max_retries} attempts",
                        execution_time=time.time() - start_time,
                        tool_name=tool_name
                    )
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

            except Exception as e:
                error_msg = f"Tool {tool_name} failed on attempt {attempt + 1}: {str(e)}"
                self.logger.warning(error_msg)
                if attempt == self.config.max_retries - 1:
                    return MCPResult(
                        success=False,
                        error=f"Tool call failed after {self.config.max_retries} attempts: {str(e)}",
                        execution_time=time.time() - start_time,
                        tool_name=tool_name
                    )
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

    async def _list_sse_tools(self) -> List[MCPTool]:
        """通过SSE连接获取工具列表"""
        tools = []

        try:
            # 建立SSE连接
            headers = {
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }

            timeout = ClientTimeout(total=30, connect=15, sock_read=15)

            async with self.session.get(
                self.config.endpoint_url,
                headers=headers,
                timeout=timeout
            ) as response:
                if response.status != 200:
                    raise Exception(f"SSE connection failed with status {response.status}")

                content_type = response.headers.get('Content-Type', '')
                if 'text/event-stream' not in content_type.lower():
                    # 尝试作为HTTP响应处理
                    response_text = await response.text()
                    try:
                        data = json.loads(response_text)
                        if isinstance(data, list):
                            for tool_data in data:
                                if isinstance(tool_data, dict) and "name" in tool_data:
                                    tools.append(MCPTool(
                                        name=tool_data.get("name", ""),
                                        description=tool_data.get("description", ""),
                                        input_schema=tool_data.get("input_schema", {})
                                    ))
                        return tools
                    except json.JSONDecodeError:
                        return tools

                # 解析SSE流获取工具信息
                tools_data = []
                line_count = 0

                try:
                    async with asyncio.timeout(20):
                        async for line in response.content:
                            if line:
                                line_text = line.decode('utf-8').strip()
                                line_count += 1

                                if line_text.startswith('data: '):
                                    event_data = line_text[6:].strip()
                                    try:
                                        tool_data = json.loads(event_data)
                                        if isinstance(tool_data, dict) and "name" in tool_data:
                                            tools_data.append(tool_data)
                                    except json.JSONDecodeError:
                                        continue

                                if line_count >= 100:  # 限制最大行数
                                    break

                except asyncio.TimeoutError:
                    self.logger.warning("SSE stream read timeout while listing tools")

                # 转换为MCPTool对象
                for tool_data in tools_data:
                    tools.append(MCPTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("input_schema", {})
                    ))

        except Exception as e:
            self.logger.error(f"Failed to list SSE tools: {str(e)}")

        return tools

    async def _list_http_tools(self) -> List[MCPTool]:
        """通过HTTP连接获取工具列表"""
        tools = []

        try:
            payload = {"action": "list_tools"}

            async with self.session.post(
                self.config.endpoint_url,
                json=payload
            ) as response:
                if response.status == 200:
                    response_text = await response.text()
                    try:
                        data = json.loads(response_text)
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
                    except json.JSONDecodeError:
                        self.logger.warning("Invalid JSON response when listing tools")

        except Exception as e:
            self.logger.error(f"Failed to list HTTP tools: {str(e)}")

        return tools

    async def _call_sse_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """通过SSE调用工具"""
        try:
            # 步骤1：建立SSE连接获取会话信息
            headers = {
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }

            timeout = ClientTimeout(total=30, connect=15, sock_read=15)

            async with self.session.get(
                self.config.endpoint_url,
                headers=headers,
                timeout=timeout
            ) as response:
                if response.status != 200:
                    return MCPResult(
                        success=False,
                        error=f"SSE connection failed with status {response.status}",
                        tool_name=tool_name
                    )

                content_type = response.headers.get('Content-Type', '')
                if 'text/event-stream' not in content_type.lower():
                    # 直接作为HTTP响应处理
                    response_text = await response.text()
                    try:
                        result_data = json.loads(response_text)
                        return MCPResult(
                            success=True,
                            data=result_data,
                            tool_name=tool_name
                        )
                    except json.JSONDecodeError:
                        return MCPResult(
                            success=True,
                            data=response_text,
                            tool_name=tool_name
                        )

                # 解析SSE流获取端点信息
                endpoint_info = None
                line_count = 0

                try:
                    async with asyncio.timeout(20):
                        async for line in response.content:
                            if line:
                                line_text = line.decode('utf-8').strip()
                                line_count += 1

                                if line_text.startswith('data: '):
                                    event_data = line_text[6:].strip()
                                    if event_data.startswith('/messages/?session_id='):
                                        endpoint_info = event_data
                                        break
                                    elif event_data.startswith('/'):
                                        endpoint_info = event_data
                                        break

                                if line_count >= 50:
                                    break

                except asyncio.TimeoutError:
                    return MCPResult(
                        success=False,
                        error="SSE stream read timeout",
                        tool_name=tool_name
                    )

                if not endpoint_info:
                    return MCPResult(
                        success=False,
                        error="No endpoint information received from SSE stream",
                        tool_name=tool_name
                    )

                # 步骤2：发送工具调用请求
                base_url = self.config.endpoint_url.rstrip('/')
                messages_url = f"{base_url}{endpoint_info}"

                payload = {
                    "tool": tool_name,
                    "arguments": arguments
                }

                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }

                timeout = ClientTimeout(total=60, connect=10, sock_read=50)

                async with self.session.post(
                    messages_url,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                ) as tool_response:
                    if tool_response.status == 200:
                        response_text = await tool_response.text()
                        try:
                            result_data = json.loads(response_text)
                            return MCPResult(
                                success=True,
                                data=result_data,
                                tool_name=tool_name
                            )
                        except json.JSONDecodeError:
                            return MCPResult(
                                success=True,
                                data=response_text,
                                tool_name=tool_name
                            )
                    else:
                        error_text = await tool_response.text()
                        return MCPResult(
                            success=False,
                            error=f"Tool call failed with status {tool_response.status}: {error_text}",
                            tool_name=tool_name
                        )

        except Exception as e:
            return MCPResult(
                success=False,
                error=f"SSE tool call failed: {str(e)}",
                tool_name=tool_name
            )

    async def _call_http_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """通过HTTP调用工具"""
        try:
            payload = {
                "tool": tool_name,
                "arguments": arguments
            }

            async with self.session.post(
                self.config.endpoint_url,
                json=payload
            ) as response:
                if response.status == 200:
                    response_text = await response.text()
                    try:
                        result_data = json.loads(response_text)
                        return MCPResult(
                            success=True,
                            data=result_data,
                            tool_name=tool_name
                        )
                    except json.JSONDecodeError:
                        return MCPResult(
                            success=True,
                            data=response_text,
                            tool_name=tool_name
                        )
                else:
                    error_text = await response.text()
                    return MCPResult(
                        success=False,
                        error=f"Tool call failed with status {response.status}: {error_text}",
                        tool_name=tool_name
                    )

        except Exception as e:
            return MCPResult(
                success=False,
                error=f"HTTP tool call failed: {str(e)}",
                tool_name=tool_name
            )


class MCPServiceManager:
    """MCP服务管理器，统一管理多个MCP客户端"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger("mcp_service_manager")
        self.clients: Dict[str, EnhancedMCPClient] = {}
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
        self._is_initialized = True

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

            # 创建客户端
            client = EnhancedMCPClient(config, self.logger)

            # 将客户端添加到堆栈中管理
            managed_client = await self.stack.enter_async_context(client)

            self.clients[config.name] = managed_client
            self.logger.info(f"Successfully registered MCP service: {config.name}")
            return True

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
                tool_name=tool_name
            )

        return await self.clients[service_name].call_tool(tool_name, arguments)

    async def find_and_call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """在所有注册的服务中查找并调用工具"""
        for service_name, client in self.clients.items():
            tools = await client.list_tools()
            for tool in tools:
                if tool.name == tool_name:
                    return await client.call_tool(tool_name, arguments)

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