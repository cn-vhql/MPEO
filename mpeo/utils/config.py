"""
配置工具模块
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from ..models.config import SystemConfig, MCPServiceConfig
from ..models.agent_config import MultiAgentConfig, AgentModelConfig, OpenAIApiConfig
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


def load_agent_config(config_path: str = "config/agent_models.json") -> MultiAgentConfig:
    """
    加载智能体模型配置

    Args:
        config_path: 智能体配置文件路径

    Returns:
        MultiAgentConfig实例

    Raises:
        ConfigurationError: 配置加载失败
    """
    try:
        # 如果配置文件不存在，使用默认配置
        if not Path(config_path).exists():
            default_config = get_default_agent_config()
            return _apply_env_overrides(default_config)

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # 处理全局OpenAI配置
        global_openai_config = config_data.get('global_openai_config', {})

        # 应用配置到各个智能体
        for agent_name in ['planner', 'executor', 'output']:
            if agent_name in config_data and 'openai_config' not in config_data[agent_name]:
                # 如果智能体没有单独的OpenAI配置，使用全局配置
                if global_openai_config:
                    config_data[agent_name]['openai_config'] = global_openai_config.copy()

        agent_config = MultiAgentConfig(**config_data)
        return _apply_env_overrides(agent_config)

    except Exception as e:
        raise ConfigurationError(f"智能体配置加载失败: {str(e)}")


def _apply_env_overrides(agent_config: MultiAgentConfig) -> MultiAgentConfig:
    """
    应用环境变量覆盖配置（仅在配置文件中缺少配置时使用）

    Args:
        agent_config: 智能体配置

    Returns:
        应用环境变量后的配置
    """
    # 为每个智能体检查配置完整性
    for agent_name in ['planner', 'executor', 'output']:
        agent = getattr(agent_config, agent_name)

        # 如果智能体没有OpenAI配置，创建一个空配置
        if not agent.openai_config:
            agent.openai_config = OpenAIApiConfig()

        # 只有当配置为空时才使用环境变量
        if not agent.openai_config.api_key:
            # 先尝试智能体特定的环境变量
            agent_specific_key = os.getenv(f"OPENAI_API_KEY_{agent_name.upper()}")
            if agent_specific_key:
                agent.openai_config.api_key = agent_specific_key
            else:
                # 使用全局环境变量
                global_api_key = os.getenv("OPENAI_API_KEY")
                if global_api_key:
                    agent.openai_config.api_key = global_api_key

        if not agent.openai_config.base_url:
            # 先尝试智能体特定的环境变量
            agent_specific_base = os.getenv(f"OPENAI_API_BASE_{agent_name.upper()}")
            if agent_specific_base:
                agent.openai_config.base_url = agent_specific_base
            else:
                # 使用全局环境变量
                global_base_url = os.getenv("OPENAI_API_BASE")
                if global_base_url:
                    agent.openai_config.base_url = global_base_url

        if not agent.openai_config.organization:
            # 先尝试智能体特定的环境变量
            agent_specific_org = os.getenv(f"OPENAI_ORGANIZATION_{agent_name.upper()}")
            if agent_specific_org:
                agent.openai_config.organization = agent_specific_org
            else:
                # 使用全局环境变量
                global_organization = os.getenv("OPENAI_ORGANIZATION")
                if global_organization:
                    agent.openai_config.organization = global_organization

    return agent_config


def load_agent_preset(preset_name: str, presets_path: str = "config/agent_models.presets.json") -> MultiAgentConfig:
    """
    加载智能体预设配置

    Args:
        preset_name: 预设名称
        presets_path: 预设配置文件路径

    Returns:
        MultiAgentConfig实例

    Raises:
        ConfigurationError: 配置加载失败
    """
    try:
        if not Path(presets_path).exists():
            raise ConfigurationError(f"预设配置文件不存在: {presets_path}")

        with open(presets_path, 'r', encoding='utf-8') as f:
            presets_data = json.load(f)

        if "presets" not in presets_data or preset_name not in presets_data["presets"]:
            available_presets = list(presets_data.get("presets", {}).keys())
            raise ConfigurationError(f"预设 '{preset_name}' 不存在。可用预设: {available_presets}")

        preset_config = presets_data["presets"][preset_name]

        # 合并全局配置
        if "global_model" in preset_config:
            preset_config.setdefault("global_model", preset_config["global_model"])
        if "global_timeout" in preset_config:
            preset_config.setdefault("global_timeout", preset_config["global_timeout"])
        if "global_retry_count" in preset_config:
            preset_config.setdefault("global_retry_count", preset_config["global_retry_count"])

        return MultiAgentConfig(**preset_config)

    except Exception as e:
        raise ConfigurationError(f"预设配置加载失败: {str(e)}")


def get_default_agent_config() -> MultiAgentConfig:
    """
    获取默认智能体配置

    Returns:
        MultiAgentConfig实例
    """
    return MultiAgentConfig(
        planner=AgentModelConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=2000,
            timeout=60,
            system_prompt="你是一个专业的任务规划专家，擅长将复杂需求分解为结构化的任务图。"
        ),
        executor=AgentModelConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1500,
            timeout=30,
            system_prompt="你是一个高效的任务执行专家，能够准确执行各种计算任务和数据处理。"
        ),
        output=AgentModelConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.4,
            max_tokens=2500,
            timeout=45,
            system_prompt="你是一个专业的内容整合专家，擅长将多个任务的执行结果整合为完整的答案。"
        ),
        global_model="gpt-3.5-turbo",
        global_timeout=60,
        global_retry_count=3,
        model_selection_strategy="balanced"
    )


def save_agent_config(agent_config: MultiAgentConfig, config_path: str = "config/agent_models.json"):
    """
    保存智能体配置

    Args:
        agent_config: 智能体配置
        config_path: 配置文件路径

    Raises:
        ConfigurationError: 配置保存失败
    """
    try:
        # 确保配置目录存在
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)

        config_data = agent_config.model_dump(exclude_none=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        raise ConfigurationError(f"智能体配置保存失败: {str(e)}")


def list_available_presets(presets_path: str = "config/agent_models.presets.json") -> Dict[str, str]:
    """
    列出所有可用的预设配置

    Args:
        presets_path: 预设配置文件路径

    Returns:
        预设名称到描述的映射
    """
    try:
        if not Path(presets_path).exists():
            return {}

        with open(presets_path, 'r', encoding='utf-8') as f:
            presets_data = json.load(f)

        presets = {}
        if "presets" in presets_data:
            for name, config in presets_data["presets"].items():
                presets[name] = config.get("description", f"预设配置: {name}")

        return presets

    except Exception as e:
        raise ConfigurationError(f"预设列表获取失败: {str(e)}")