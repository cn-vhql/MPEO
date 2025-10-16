"""
MCP通用定义 - 避免循环导入
"""

from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, _AsyncGeneratorContextManager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union


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
    service_type: str = "http"  # stdio, sse, streamable_http
    endpoint_url: Optional[str] = None
    command: Optional[str] = None
    args: list = None
    env: dict = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    headers: dict = None

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

            return cls(
                name=service_config.service_name,
                service_type=service_type,
                endpoint_url=service_config.endpoint_url,
                timeout=service_config.timeout or 60,
                headers=service_config.headers or {}
            )


class MCPClientBase(ABC):
    """MCP客户端基类，参照mcp_demo的设计模式"""

    def __init__(self, name: str, config: MCPConnectionConfig):
        from ..utils.logging import get_logger
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

    async def get_callable_function(self, func_name: str, wrap_tool_result: bool = True):
        """获取可调用的工具函数"""
        if not self._is_connected:
            raise RuntimeError(f"Client {self.name} not connected")

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

        # Import here to avoid circular import
        from .optimized_mcp_manager import MCPToolFunction

        return MCPToolFunction(
            mcp_name=self.name,
            tool=tool_obj,
            wrap_tool_result=wrap_tool_result,
            session=getattr(self, '_session', None)
        )

    def _validate_connection(self) -> None:
        """验证连接状态"""
        if not self._is_connected:
            raise RuntimeError(f"Client {self.name} not connected")