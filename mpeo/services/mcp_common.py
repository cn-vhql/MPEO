"""
MCP通用定义 - 向后兼容性支持
从统一MCP管理器导入数据结构，避免重复定义
"""

# 从统一管理器导入所有数据结构
from .unified_mcp_manager import (
    MCPTool,
    MCPResult,
    MCPConnectionConfig,
    UnifiedMCPManager,
    ToolRegistry
)

# 为了向后兼容，保留一些别名
MCPClientBase = object  # 基类现在在unified_mcp_manager中定义