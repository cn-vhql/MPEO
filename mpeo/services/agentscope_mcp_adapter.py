"""
AgentScope MCP Adapter - 将MCP服务适配为AgentScope工具系统
提供统一的工具接口，兼容AgentScope的工具调用机制
"""

import asyncio
import functools
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass

# AgentScope imports
try:
    from agentscope.tool import Toolkit
    from agentscope.message import Msg
    AGENTSCOPE_AVAILABLE = True
except ImportError:
    AGENTSCOPE_AVAILABLE = False
    Toolkit = None
    Msg = None

# MPEO imports
from .unified_mcp_manager import UnifiedMCPManager, MCPTool, MCPResult
from ..utils.logging import get_logger


@dataclass
class AgentScopeToolConfig:
    """AgentScope工具配置"""
    name: str
    description: str
    service_name: str
    mcp_tool_name: str
    parameters_schema: Dict[str, Any]
    enabled: bool = True
    cache_results: bool = True
    timeout: int = 30


class MCPToolFunction:
    """MCP工具函数包装器，使其符合AgentScope工具函数规范"""

    def __init__(self,
                 service_name: str,
                 mcp_tool: MCPTool,
                 mcp_manager: UnifiedMCPManager,
                 config: AgentScopeToolConfig = None):
        self.service_name = service_name
        self.mcp_tool = mcp_tool
        self.mcp_manager = mcp_manager
        self.config = config or AgentScopeToolConfig(
            name=mcp_tool.name,
            description=mcp_tool.description,
            service_name=service_name,
            mcp_tool_name=mcp_tool.name,
            parameters_schema=mcp_tool.input_schema
        )
        self.logger = get_logger(f"mcp_tool.{service_name}.{mcp_tool.name}")
        self._call_cache = {}

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        AgentScope工具函数调用接口
        返回格式化的结果字典
        """
        # 检查缓存
        if self.config.cache_results:
            cache_key = self._generate_cache_key(kwargs)
            if cache_key in self._call_cache:
                self.logger.debug(f"Using cached result for {self.config.name}")
                return self._call_cache[cache_key]

        # 执行MCP调用
        try:
            # 验证参数
            validated_args = self._validate_parameters(kwargs)

            # 异步调用MCP工具
            result = asyncio.run(self._call_mcp_tool(validated_args))

            # 格式化结果
            formatted_result = self._format_result(result)

            # 缓存结果
            if self.config.cache_results and result.success:
                cache_key = self._generate_cache_key(kwargs)
                self._call_cache[cache_key] = formatted_result

            return formatted_result

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "tool_name": self.config.name,
                "service_name": self.service_name,
                "error_type": "tool_execution_error"
            }
            self.logger.error(f"Tool execution failed: {str(e)}")
            return error_result

    async def _call_mcp_tool(self, arguments: Dict[str, Any]) -> MCPResult:
        """异步调用MCP工具"""
        try:
            self.logger.debug(f"Calling MCP tool {self.mcp_tool.name} with args: {arguments}")

            result = await self.mcp_manager.call_tool(
                service_name=self.service_name,
                tool_name=self.mcp_tool.name,
                arguments=arguments
            )

            self.logger.debug(f"MCP tool result: success={result.success}, execution_time={result.execution_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"MCP tool call failed: {str(e)}")
            return MCPResult(
                success=False,
                error=str(e),
                tool_name=self.mcp_tool.name,
                service_name=self.service_name
            )

    def _validate_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """验证和标准化参数"""
        validated = {}

        if not self.mcp_tool.input_schema:
            return kwargs

        properties = self.mcp_tool.input_schema.get("properties", {})
        required = self.mcp_tool.input_schema.get("required", [])

        # 为缺失的必需参数添加默认值
        for param in required:
            if param not in kwargs:
                # 尝试从properties中获取默认值
                if param in properties and "default" in properties[param]:
                    kwargs[param] = properties[param]["default"]
                else:
                    # 为常见参数提供合理的默认值
                    if param == "format":
                        kwargs[param] = "YYYY-MM-DD HH:mm:ss"
                    elif param == "timezone":
                        kwargs[param] = "UTC"
                    elif param == "url":
                        # 对于URL参数，如果缺失则返回错误，因为这个需要用户提供
                        raise ValueError(f"Missing required parameter: {param}")
                    else:
                        kwargs[param] = ""

        # 验证并转换参数
        for param_name, param_value in kwargs.items():
            if param_name in properties:
                param_schema = properties[param_name]
                validated[param_name] = self._validate_parameter_value(
                    param_name, param_value, param_schema
                )
            else:
                # 保留未定义的参数，但记录警告
                self.logger.warning(f"Unknown parameter: {param_name}")
                validated[param_name] = param_value

        return validated

    def _validate_parameter_value(self, param_name: str, value: Any, schema: Dict[str, Any]) -> Any:
        """验证单个参数值"""
        param_type = schema.get("type", "string")

        try:
            if param_type == "string":
                return str(value)
            elif param_type == "integer":
                return int(value)
            elif param_type == "number":
                return float(value)
            elif param_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ["true", "1", "yes", "on"]
                return bool(value)
            elif param_type == "array":
                if isinstance(value, str):
                    # 尝试解析JSON数组
                    try:
                        return json.loads(value)
                    except:
                        return [value]
                elif not isinstance(value, list):
                    return [value]
                return value
            elif param_type == "object":
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except:
                        return {"value": value}
                elif not isinstance(value, dict):
                    return {"value": value}
                return value
            else:
                return value

        except (ValueError, TypeError) as e:
            self.logger.warning(f"Parameter validation failed for {param_name}: {str(e)}")
            return value

    def _format_result(self, mcp_result: MCPResult) -> Dict[str, Any]:
        """格式化MCP结果为AgentScope兼容格式"""
        if mcp_result.success:
            return {
                "success": True,
                "data": mcp_result.data,
                "tool_name": self.config.name,
                "service_name": self.service_name,
                "execution_time": mcp_result.execution_time,
                "metadata": {
                    "mcp_tool": self.mcp_tool.name,
                    "original_result": mcp_result.to_dict()
                }
            }
        else:
            return {
                "success": False,
                "error": mcp_result.error,
                "tool_name": self.config.name,
                "service_name": self.service_name,
                "error_type": "mcp_call_failed",
                "metadata": {
                    "mcp_tool": self.mcp_tool.name,
                    "original_result": mcp_result.to_dict()
                }
            }

    def _generate_cache_key(self, kwargs: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        key_data = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(f"{self.service_name}.{self.mcp_tool.name}:{key_data}".encode()).hexdigest()

    @property
    def __doc__(self) -> str:
        """生成工具函数的文档字符串，符合AgentScope规范"""
        doc = f"{self.config.description}\n\n"

        if self.mcp_tool.input_schema and "properties" in self.mcp_tool.input_schema:
            doc += "参数:\n"
            properties = self.mcp_tool.input_schema["properties"]
            required = self.mcp_tool.input_schema.get("required", [])

            for param_name, param_schema in properties.items():
                param_desc = param_schema.get("description", "无描述")
                param_type = param_schema.get("type", "string")
                is_required = param_name in required

                doc += f"  {param_name} ({param_type})"
                if is_required:
                    doc += " [必需]"
                doc += f": {param_desc}\n"

                if "default" in param_schema:
                    doc += f"    默认值: {param_schema['default']}\n"

        doc += f"\n返回值: Dict[str, Any] - 包含success、data等字段的结果字典"
        return doc


class AgentScopeMCPAdapter:
    """AgentScope MCP适配器，统一管理MCP工具到AgentScope的转换"""

    def __init__(self, mcp_manager: UnifiedMCPManager):
        if not AGENTSCOPE_AVAILABLE:
            raise ImportError("AgentScope is not installed. Install with: pip install agentscope")

        self.mcp_manager = mcp_manager
        self.logger = get_logger("agentscope_mcp_adapter")
        self.tool_configs: Dict[str, AgentScopeToolConfig] = {}
        self.tool_functions: Dict[str, MCPToolFunction] = {}
        self.toolkits: Dict[str, Toolkit] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """初始化适配器，加载所有MCP工具"""
        if self._initialized:
            return

        self.logger.info("Initializing AgentScope MCP Adapter")

        try:
            # 确保MCP管理器已初始化
            if not hasattr(self.mcp_manager, '_is_initialized') or not self.mcp_manager._is_initialized:
                await self.mcp_manager.initialize()

            # 加载所有MCP工具
            await self._load_mcp_tools()

            # 创建默认工具包
            await self._create_default_toolkit()

            self._initialized = True
            self.logger.info(f"AgentScope MCP Adapter initialized with {len(self.tool_functions)} tools")

        except Exception as e:
            self.logger.error(f"Failed to initialize AgentScope MCP Adapter: {str(e)}")
            raise

    async def _load_mcp_tools(self) -> None:
        """加载所有MCP工具并创建对应的AgentScope工具函数"""
        try:
            # 获取所有可用工具
            all_tools = await self.mcp_manager.get_available_tools()

            for service_name, tools in all_tools.items():
                self.logger.debug(f"Loading {len(tools)} tools from service: {service_name}")

                for mcp_tool in tools:
                    try:
                        # 创建工具配置
                        config = AgentScopeToolConfig(
                            name=f"mcp_{service_name}_{mcp_tool.name}",
                            description=f"[MCP-{service_name}] {mcp_tool.description}",
                            service_name=service_name,
                            mcp_tool_name=mcp_tool.name,
                            parameters_schema=mcp_tool.input_schema
                        )

                        # 创建工具函数
                        tool_func = MCPToolFunction(
                            service_name=service_name,
                            mcp_tool=mcp_tool,
                            mcp_manager=self.mcp_manager,
                            config=config
                        )

                        # 存储工具
                        self.tool_configs[config.name] = config
                        self.tool_functions[config.name] = tool_func

                        self.logger.debug(f"Loaded MCP tool: {config.name}")

                    except Exception as e:
                        self.logger.error(f"Failed to load tool {mcp_tool.name} from {service_name}: {str(e)}")
                        continue

        except Exception as e:
            self.logger.error(f"Failed to load MCP tools: {str(e)}")
            raise

    async def _create_default_toolkit(self) -> None:
        """创建包含所有工具的默认工具包"""
        if not AGENTSCOPE_AVAILABLE:
            return

        try:
            default_toolkit = Toolkit()

            for tool_name, tool_func in self.tool_functions.items():
                if tool_func.config.enabled:
                    try:
                        # 注册工具到工具包
                        default_toolkit.register_tool_function(
                            tool_func,
                            group_name=tool_func.service_name,
                            func_description=tool_func.config.description
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to register tool {tool_name}: {str(e)}")
                        continue

            self.toolkits["default"] = default_toolkit
            self.logger.info(f"Created default toolkit with {len(default_toolkit._tools)} tools")

        except Exception as e:
            self.logger.error(f"Failed to create default toolkit: {str(e)}")
            # 创建空工具包作为后备
            if AGENTSCOPE_AVAILABLE:
                self.toolkits["default"] = Toolkit()
                self.logger.info("Created empty default toolkit as fallback")

    async def create_toolkit(self,
                           service_names: Optional[List[str]] = None,
                           tool_names: Optional[List[str]] = None,
                           group_name: str = "custom") -> Toolkit:
        """创建自定义工具包"""
        if not AGENTSCOPE_AVAILABLE:
            raise RuntimeError("AgentScope not available")

        if not self._initialized:
            await self.initialize()

        toolkit = Toolkit()

        for tool_name, tool_func in self.tool_functions.items():
            # 根据条件筛选工具
            include_tool = True

            if service_names and tool_func.service_name not in service_names:
                include_tool = False

            if tool_names and tool_func.config.name not in tool_names:
                include_tool = False

            if include_tool and tool_func.config.enabled:
                toolkit.register_tool_function(
                    tool_func,
                    group_name=group_name,
                    func_description=tool_func.config.description
                )

        # 缓存工具包
        cache_key = f"custom_{group_name}_{hash(tuple(service_names or []))}_{hash(tuple(tool_names or []))}"
        self.toolkits[cache_key] = toolkit

        self.logger.info(f"Created custom toolkit '{group_name}' with {len(toolkit._tools)} tools")
        return toolkit

    async def get_relevant_tools(self, query: str, max_tools: int = 10) -> List[MCPToolFunction]:
        """根据查询获取相关工具"""
        if not self._initialized:
            await self.initialize()

        query_lower = query.lower()
        scored_tools = []

        for tool_func in self.tool_functions.values():
            if not tool_func.config.enabled:
                continue

            score = 0

            # 工具名称匹配
            if tool_func.mcp_tool.name.lower() in query_lower:
                score += 10

            # 服务名称匹配
            if tool_func.service_name.lower() in query_lower:
                score += 5

            # 描述匹配
            desc_lower = tool_func.config.description.lower()
            query_words = query_lower.split()
            for word in query_words:
                if word in desc_lower:
                    score += 2

            # 特定关键词匹配
            keywords_map = {
                "fetch": ["fetch", "get", "获取", "抓取", "下载"],
                "search": ["search", "查找", "搜索", "find"],
                "time": ["time", "时间", "date", "日期"],
                "docs": ["docs", "documentation", "文档", "api"]
            }

            for keyword, matches in keywords_map.items():
                if keyword in tool_func.mcp_tool.name.lower() and any(match in query_lower for match in matches):
                    score += 8

            if score > 0:
                scored_tools.append((tool_func, score))

        # 按分数排序并返回前N个
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool_func for tool_func, _ in scored_tools[:max_tools]]

    async def create_contextual_toolkit(self, query: str, max_tools: int = 5) -> Toolkit:
        """根据查询上下文创建工具包"""
        relevant_tools = await self.get_relevant_tools(query, max_tools)

        if not AGENTSCOPE_AVAILABLE:
            raise RuntimeError("AgentScope not available")

        toolkit = Toolkit()

        for tool_func in relevant_tools:
            toolkit.register_tool_function(
                tool_func,
                group_name="contextual",
                func_description=tool_func.config.description
            )

        # 生成查询哈希作为缓存键
        import hashlib
        cache_key = f"contextual_{hashlib.md5(query.encode()).hexdigest()}"
        self.toolkits[cache_key] = toolkit

        self.logger.info(f"Created contextual toolkit with {len(relevant_tools)} tools for query: {query[:50]}...")
        return toolkit

    def get_toolkit(self, name: str = "default") -> Optional[Toolkit]:
        """获取已创建的工具包"""
        return self.toolkits.get(name)

    def get_tool_function(self, tool_name: str) -> Optional[MCPToolFunction]:
        """获取特定的工具函数"""
        return self.tool_functions.get(tool_name)

    def list_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """列出所有可用工具的信息"""
        tools_info = {}

        for tool_name, tool_func in self.tool_functions.items():
            if tool_func.config.enabled:
                tools_info[tool_name] = {
                    "name": tool_func.config.name,
                    "description": tool_func.config.description,
                    "service_name": tool_func.service_name,
                    "mcp_tool_name": tool_func.mcp_tool.name,
                    "parameters_schema": tool_func.mcp_tool.input_schema,
                    "cache_enabled": tool_func.config.cache_results
                }

        return tools_info

    def enable_tool(self, tool_name: str) -> bool:
        """启用工具"""
        if tool_name in self.tool_functions:
            self.tool_functions[tool_name].config.enabled = True
            self.logger.info(f"Enabled tool: {tool_name}")
            return True
        return False

    def disable_tool(self, tool_name: str) -> bool:
        """禁用工具"""
        if tool_name in self.tool_functions:
            self.tool_functions[tool_name].config.enabled = False
            self.logger.info(f"Disabled tool: {tool_name}")
            return True
        return False

    def clear_cache(self, tool_name: Optional[str] = None) -> None:
        """清空工具缓存"""
        if tool_name:
            if tool_name in self.tool_functions:
                self.tool_functions[tool_name]._call_cache.clear()
                self.logger.info(f"Cleared cache for tool: {tool_name}")
        else:
            for tool_func in self.tool_functions.values():
                tool_func._call_cache.clear()
            self.logger.info("Cleared cache for all tools")

    async def refresh_tools(self) -> None:
        """刷新工具列表"""
        self.logger.info("Refreshing MCP tools")

        # 清空现有工具
        self.tool_configs.clear()
        self.tool_functions.clear()
        self.toolkits.clear()

        # 重新加载
        await self._load_mcp_tools()
        await self._create_default_toolkit()

        self.logger.info(f"Tools refreshed: {len(self.tool_functions)} tools loaded")

    async def close(self) -> None:
        """关闭适配器"""
        self.logger.info("Closing AgentScope MCP Adapter")

        # 清空缓存
        self.clear_cache()

        # 清空工具
        self.tool_configs.clear()
        self.tool_functions.clear()
        self.toolkits.clear()

        self._initialized = False


# 便利函数
async def create_mcp_adapter(mcp_manager: UnifiedMCPManager) -> AgentScopeMCPAdapter:
    """创建并初始化MCP适配器"""
    adapter = AgentScopeMCPAdapter(mcp_manager)
    await adapter.initialize()
    return adapter


async def create_contextual_toolkit(mcp_manager: UnifiedMCPManager, query: str) -> Toolkit:
    """为查询创建上下文工具包的便利函数"""
    adapter = await create_mcp_adapter(mcp_manager)
    return await adapter.create_contextual_toolkit(query)