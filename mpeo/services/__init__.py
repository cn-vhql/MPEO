"""
MPEO 服务模块

包含各种服务组件：
- database: 数据库服务
- unified_mcp_manager: 统一MCP服务管理器
- configuration_loader: 配置加载器
- mcp_common: MCP通用定义（向后兼容）
"""

from .database import DatabaseManager
from .unified_mcp_manager import UnifiedMCPManager
from .configuration_loader import get_config_loader

# 为了向后兼容，保留原有的导入路径
from .mcp_common import MCPTool, MCPResult, MCPConnectionConfig

__all__ = [
    "DatabaseManager",
    "UnifiedMCPManager",
    "get_config_loader",
    "MCPTool",
    "MCPResult",
    "MCPConnectionConfig"
]