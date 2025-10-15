"""
配置工具模块
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from ..models.config import SystemConfig, MCPServiceConfig
from .exceptions import ConfigurationError


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    加载系统配置

    Args:
        config_path: 配置文件路径，可选

    Returns:
        SystemConfig实例

    Raises:
        ConfigurationError: 配置加载失败
    """
    try:
        # 加载环境变量
        load_dotenv(override=True)

        # 默认配置
        config_data = {}

        # 从配置文件加载
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config_data.update(file_config)

        # 从环境变量覆盖
        env_mappings = {
            'MPEO_MAX_PARALLEL_TASKS': 'max_parallel_tasks',
            'MPEO_MCP_TIMEOUT': 'mcp_service_timeout',
            'MPEO_RETRY_COUNT': 'task_retry_count',
            'OPENAI_MODEL': 'openai_model',
            'MPEO_DATABASE_PATH': 'database_path'
        }

        for env_key, config_key in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                # 类型转换
                if config_key in ['max_parallel_tasks', 'mcp_service_timeout', 'task_retry_count']:
                    config_data[config_key] = int(env_value)
                else:
                    config_data[config_key] = env_value

        return SystemConfig(**config_data)

    except Exception as e:
        raise ConfigurationError(f"配置加载失败: {str(e)}")


def load_mcp_services(config_path: str = "config/mcp_services.json") -> Dict[str, MCPServiceConfig]:
    """
    加载MCP服务配置

    Args:
        config_path: MCP服务配置文件路径

    Returns:
        MCP服务配置字典

    Raises:
        ConfigurationError: 配置加载失败
    """
    try:
        if not Path(config_path).exists():
            return {}

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        mcp_services = {}
        if "mcpServices" in config_data:
            for service_name, service_config in config_data["mcpServices"].items():
                mcp_services[service_name] = MCPServiceConfig(
                    service_name=service_name,
                    service_type=service_config.get("type", "http"),
                    endpoint_url=service_config.get("url", ""),
                    timeout=service_config.get("timeout", 30),
                    headers=service_config.get("headers", {})
                )

        return mcp_services

    except Exception as e:
        raise ConfigurationError(f"MCP服务配置加载失败: {str(e)}")


def save_mcp_services(services: Dict[str, MCPServiceConfig],
                     config_path: str = "config/mcp_services.json"):
    """
    保存MCP服务配置

    Args:
        services: MCP服务配置字典
        config_path: 配置文件路径

    Raises:
        ConfigurationError: 配置保存失败
    """
    try:
        # 确保配置目录存在
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "mcpServices": {
                name: {
                    "type": service.service_type,
                    "url": service.endpoint_url,
                    "timeout": service.timeout,
                    "headers": service.headers or {}
                }
                for name, service in services.items()
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        raise ConfigurationError(f"MCP服务配置保存失败: {str(e)}")