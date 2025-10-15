"""
MPEO 数据模型模块

包含所有数据结构定义：
- task: 任务相关模型
- session: 会话相关模型
- config: 配置相关模型
"""

from .task import (
    TaskNode, TaskEdge, TaskGraph, TaskType, TaskStatus, DependencyType,
    ExecutionResult, ExecutionResults
)
from .session import TaskSession
from .config import SystemConfig, MCPServiceConfig

__all__ = [
    # Task models
    "TaskNode",
    "TaskEdge",
    "TaskGraph",
    "TaskType",
    "TaskStatus",
    "DependencyType",
    "ExecutionResult",
    "ExecutionResults",

    # Session models
    "TaskSession",

    # Config models
    "SystemConfig",
    "MCPServiceConfig"
]