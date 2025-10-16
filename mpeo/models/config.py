"""
配置相关数据模型
"""

from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """System configuration"""
    max_parallel_tasks: int = Field(default=4, ge=1, description="Maximum parallel tasks")
    mcp_service_timeout: int = Field(default=30, ge=1, description="MCP service timeout in seconds")
    task_retry_count: int = Field(default=3, ge=0, description="Task retry count")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    database_path: str = Field(default_factory=lambda: "data/databases/mpeo.db", description="SQLite database path")


class MCPServiceConfig(BaseModel):
    """MCP service configuration"""
    service_name: str = Field(..., description="Service name")
    service_type: str = Field(default="http", description="Service type (http, sse, websocket)")
    endpoint_url: str = Field(..., description="Service endpoint URL")
    timeout: int = Field(default=30, description="Request timeout")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Request headers")
    auth_config: Optional[Dict[str, Any]] = Field(default=None, description="Authentication config")