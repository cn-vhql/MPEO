"""
项目路径配置模块
"""

import os
from pathlib import Path
from typing import Optional

class ProjectPaths:
    """项目路径配置类，统一管理所有硬编码路径"""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # 自动检测项目根目录
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = base_dir

    @property
    def data_dir(self) -> Path:
        """数据目录"""
        return self.base_dir / "data"

    @property
    def databases_dir(self) -> Path:
        """数据库目录"""
        return self.data_dir / "databases"

    @property
    def logs_dir(self) -> Path:
        """日志目录"""
        return self.data_dir / "logs"

    @property
    def config_dir(self) -> Path:
        """配置目录"""
        return self.base_dir / "config"

    @property
    def default_database_path(self) -> str:
        """默认数据库路径"""
        return str(self.databases_dir / "mpeo.db")

    @property
    def mcp_services_config_path(self) -> str:
        """MCP服务配置文件路径"""
        return str(self.config_dir / "mcp_services.json")

    @property
    def agent_models_config_path(self) -> str:
        """智能体模型配置文件路径"""
        return str(self.config_dir / "agent_models.json")

    def ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            self.data_dir,
            self.databases_dir,
            self.logs_dir,
            self.config_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# 全局路径实例
paths = ProjectPaths()

# 向后兼容的函数
def get_project_paths() -> ProjectPaths:
    """获取项目路径配置实例"""
    return paths

def ensure_project_structure():
    """确保项目目录结构存在"""
    paths.ensure_directories()