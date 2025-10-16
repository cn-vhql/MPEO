"""
统一配置加载器 - 简化配置管理逻辑
整合所有配置相关的加载和管理功能
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

from ..models.config import SystemConfig, MCPServiceConfig
from ..models.agent_config import MultiAgentConfig, AgentModelConfig, OpenAIApiConfig
from ..utils.logging import get_logger


class ConfigurationLoader:
    """统一配置加载器，处理所有配置相关的操作"""

    def __init__(self):
        self.logger = get_logger("config_loader")
        self._system_config: Optional[SystemConfig] = None
        self._agent_config: Optional[MultiAgentConfig] = None
        self._mcp_services: Dict[str, MCPServiceConfig] = {}

    def load_system_config(self, config_path: Optional[str] = None) -> SystemConfig:
        """加载系统配置"""
        if self._system_config is not None:
            return self._system_config

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

            self._system_config = SystemConfig(**config_data)
            self.logger.info("System configuration loaded successfully")
            return self._system_config

        except Exception as e:
            self.logger.error(f"Failed to load system configuration: {str(e)}")
            # 返回默认配置
            self._system_config = SystemConfig()
            return self._system_config

    def load_agent_config(self, config_path: str = "config/agent_models.json") -> MultiAgentConfig:
        """加载智能体配置"""
        if self._agent_config is not None:
            return self._agent_config

        try:
            # 如果配置文件不存在，使用默认配置
            if not Path(config_path).exists():
                self._agent_config = self._get_default_agent_config()
                self._agent_config = self._apply_env_overrides(self._agent_config)
                return self._agent_config

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
            self._agent_config = self._apply_env_overrides(agent_config)
            self.logger.info("Agent configuration loaded successfully")
            return self._agent_config

        except Exception as e:
            self.logger.error(f"Failed to load agent configuration: {str(e)}")
            # 返回默认配置
            self._agent_config = self._get_default_agent_config()
            self._agent_config = self._apply_env_overrides(self._agent_config)
            return self._agent_config

    def load_mcp_services(self, config_path: str = "config/mcp_services.json") -> Dict[str, MCPServiceConfig]:
        """加载MCP服务配置"""
        if self._mcp_services:
            return self._mcp_services

        try:
            if not Path(config_path).exists():
                self.logger.info(f"MCP config file not found at {config_path}")
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

            self._mcp_services = mcp_services
            self.logger.info(f"Loaded {len(mcp_services)} MCP services")
            return mcp_services

        except Exception as e:
            self.logger.error(f"Failed to load MCP services configuration: {str(e)}")
            return {}

    def save_mcp_services(self, services: Dict[str, MCPServiceConfig],
                         config_path: str = "config/mcp_services.json"):
        """保存MCP服务配置"""
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

            self.logger.info("MCP services configuration saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save MCP services configuration: {str(e)}")

    def save_agent_config(self, agent_config: MultiAgentConfig, config_path: str = "config/agent_models.json"):
        """保存智能体配置"""
        try:
            # 确保配置目录存在
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)

            config_data = agent_config.model_dump(exclude_none=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

            self.logger.info("Agent configuration saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save agent configuration: {str(e)}")

    def add_mcp_service(self, service_config: MCPServiceConfig, config_path: str = "config/mcp_services.json"):
        """添加单个MCP服务配置"""
        mcp_services = self.load_mcp_services(config_path)
        mcp_services[service_config.service_name] = service_config
        self.save_mcp_services(mcp_services, config_path)
        self._mcp_services = mcp_services  # 更新缓存

    def remove_mcp_service(self, service_name: str, config_path: str = "config/mcp_services.json"):
        """移除MCP服务配置"""
        mcp_services = self.load_mcp_services(config_path)
        if service_name in mcp_services:
            del mcp_services[service_name]
            self.save_mcp_services(mcp_services, config_path)
            self._mcp_services = mcp_services  # 更新缓存
            self.logger.info(f"Removed MCP service: {service_name}")
        else:
            self.logger.warning(f"MCP service not found: {service_name}")

    def list_mcp_services(self, config_path: str = "config/mcp_services.json") -> List[str]:
        """列出所有MCP服务名称"""
        mcp_services = self.load_mcp_services(config_path)
        return list(mcp_services.keys())

    def get_mcp_service(self, service_name: str, config_path: str = "config/mcp_services.json") -> Optional[MCPServiceConfig]:
        """获取指定MCP服务配置"""
        mcp_services = self.load_mcp_services(config_path)
        return mcp_services.get(service_name)

    def _get_default_agent_config(self) -> MultiAgentConfig:
        """获取默认智能体配置"""
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

    def _apply_env_overrides(self, agent_config: MultiAgentConfig) -> MultiAgentConfig:
        """应用环境变量覆盖配置"""
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

    def reload_configs(self):
        """重新加载所有配置"""
        self._system_config = None
        self._agent_config = None
        self._mcp_services = {}
        self.logger.info("All configurations reset and will be reloaded on next access")

    def validate_configs(self) -> Dict[str, Any]:
        """验证配置的完整性和正确性"""
        validation_result = {
            "system_config": {"valid": True, "errors": []},
            "agent_config": {"valid": True, "errors": []},
            "mcp_services": {"valid": True, "errors": []}
        }

        try:
            # 验证系统配置
            system_config = self.load_system_config()
            if system_config.max_parallel_tasks < 1:
                validation_result["system_config"]["errors"].append("max_parallel_tasks must be >= 1")
                validation_result["system_config"]["valid"] = False

            if system_config.mcp_service_timeout < 1:
                validation_result["system_config"]["errors"].append("mcp_service_timeout must be >= 1")
                validation_result["system_config"]["valid"] = False

            # 验证智能体配置
            agent_config = self.load_agent_config()
            for agent_name in ['planner', 'executor', 'output']:
                agent = getattr(agent_config, agent_name)
                if not agent.model_name:
                    validation_result["agent_config"]["errors"].append(f"{agent_name} model_name is required")
                    validation_result["agent_config"]["valid"] = False

                if not agent.openai_config or not agent.openai_config.api_key:
                    # 检查环境变量
                    if not os.getenv("OPENAI_API_KEY") and not os.getenv(f"OPENAI_API_KEY_{agent_name.upper()}"):
                        validation_result["agent_config"]["errors"].append(f"{agent_name} OpenAI API key is required")
                        validation_result["agent_config"]["valid"] = False

            # 验证MCP服务配置
            mcp_services = self.load_mcp_services()
            for service_name, service_config in mcp_services.items():
                if not service_config.service_name:
                    validation_result["mcp_services"]["errors"].append(f"Service {service_name} name is required")
                    validation_result["mcp_services"]["valid"] = False

                if not service_config.endpoint_url:
                    validation_result["mcp_services"]["errors"].append(f"Service {service_name} endpoint_url is required")
                    validation_result["mcp_services"]["valid"] = False

            self.logger.info("Configuration validation completed")
            return validation_result

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            validation_result["system_config"]["valid"] = False
            validation_result["system_config"]["errors"].append(str(e))
            return validation_result


# 全局配置加载器实例
_config_loader: Optional[ConfigurationLoader] = None


def get_config_loader() -> ConfigurationLoader:
    """获取全局配置加载器实例"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigurationLoader()
    return _config_loader