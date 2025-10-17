"""
Main entry point for the Multi-model collaborative task processing system
多模型协作任务处理系统主入口 (支持AgentScope增强版)
"""

import asyncio
import sys
import argparse
import os
from pathlib import Path
from typing import Optional

# Ensure proper package structure
if __name__ == "__main__" and __package__ is None:
    # Only add to path if running as script and not in package context
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from mpeo.core import CLIInterface
from mpeo.models import SystemConfig, AgentModelConfig
from mpeo.services import get_config_loader
from mpeo.utils.logging import get_logger


class EnhancedCLIInterface:
    """增强的CLI界面，支持AgentScope集成"""

    def __init__(self):
        self.logger = get_logger("enhanced_cli")
        self.cli_interface = CLIInterface()
        self.config_loader = get_config_loader()
        self.agentscope_config = None

    async def initialize(self, config: SystemConfig, enable_agentscope: bool = True) -> bool:
        """初始化系统，支持AgentScope集成"""
        try:
            self.logger.info("Initializing MPEO system with AgentScope support...")

            # 检查AgentScope配置和可用性
            if enable_agentscope:
                self.agentscope_config = self.config_loader.load_agentscope_config()
                if not self.agentscope_config.enabled:
                    self.logger.warning("AgentScope is disabled in configuration, falling back to legacy mode")
                    enable_agentscope = False
                elif not self.config_loader.is_agentscope_enabled():
                    self.logger.warning("AgentScope is not available, falling back to legacy mode")
                    enable_agentscope = False
                else:
                    self.logger.info("AgentScope integration enabled")


            # 创建AgentScope配置参数
            planner_config = None
            executor_config = None

            if enable_agentscope and self.agentscope_config:
                # 获取模型映射
                model_mapping = self.config_loader.get_model_mapping()

                # 获取配置覆盖
                config_overrides = self.agentscope_config.model_integration.get("openai", {}).get("config_overrides", {})

                planner_config = AgentModelConfig(
                    model_name=model_mapping.get("planner", "gpt-4"),
                    temperature=config_overrides.get("planner", {}).get("temperature", 0.3),
                    max_tokens=config_overrides.get("planner", {}).get("max_tokens", 2000)
                )

                executor_config = AgentModelConfig(
                    model_name=model_mapping.get("executor", "gpt-3.5-turbo"),
                    temperature=config_overrides.get("executor", {}).get("temperature", 0.2),
                    max_tokens=config_overrides.get("executor", {}).get("max_tokens", 1500)
                )

            # 使用现有的CLI接口初始化，传递AgentScope配置
            success = self.cli_interface.initialize(
                config=config,
                planner_config=planner_config,
                executor_config=executor_config,
                enable_agentscope=enable_agentscope
            )
            if not success:
                return False

            self.logger.info(f"System initialized successfully (AgentScope: {'enabled' if enable_agentscope else 'disabled'})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            return False

    async def run_interactive_mode(self):
        """运行交互模式"""
        print("\n" + "="*60)
        print("🤖 MPEO - 多模型协作任务处理系统")
        print("="*60)

        if self.agentscope_config and self.agentscope_config.enabled:
            print("✅ AgentScope集成已启用")

            # 获取模型映射
            model_mapping = self.config_loader.get_model_mapping()

            print(f"📊 规划器模型: {model_mapping.get('planner', 'gpt-4')}")
            print(f"⚡ 执行器模型: {model_mapping.get('executor', 'gpt-3.5-turbo')}")
        else:
            print("⚠️  AgentScope未启用，使用传统模式")

        print(f"🔧 系统状态: 已启动")
        print(f"💾 数据库: {self.cli_interface.coordinator.config.database_path}")
        print("\n🎯 特色功能:")
        if self.agentscope_config and self.agentscope_config.enabled:
            print("  • 智能任务规划和分解")
            print("  • 并行任务执行")
            print("  • 上下文工具选择")
            print("  • 自动错误恢复")
        else:
            print("  • 传统任务处理模式")
        print("\n输入 'help' 或 '帮助' 查看可用命令")
        print("输入 'quit' 或 '退出' 退出系统")
        print("-"*60)

        # 使用现有的CLI接口运行交互模式
        await self.cli_interface.run_interactive_mode()

    async def cleanup(self):
        """清理资源"""
        try:
            # 清理由CLIInterface内部处理
            if self.cli_interface and self.cli_interface.coordinator:
                await self.cli_interface.coordinator.cleanup()

            self.logger.info("System cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")


def main():
    """Main entry point with AgentScope support"""
    parser = argparse.ArgumentParser(description="多模型协作任务处理系统 (支持AgentScope增强版)")
    parser.add_argument("--config", help="配置文件路径", default=None)
    parser.add_argument("--max-parallel", type=int, help="最大并行任务数", default=4)
    parser.add_argument("--timeout", type=int, help="MCP服务超时时间(秒)", default=30)
    parser.add_argument("--retries", type=int, help="任务重试次数", default=3)
    parser.add_argument("--model", help="OpenAI模型名称", default="gpt-3.5-turbo")
    parser.add_argument("--db-path", help="数据库路径", default="data/databases/mpeo.db")
    parser.add_argument("--no-agentscope", action="store_true", help="禁用AgentScope集成", default=False)

    args = parser.parse_args()

    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 错误: OPENAI_API_KEY 环境变量未设置")
        print("请设置您的OpenAI API密钥:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # 创建系统配置
    config = SystemConfig(
        max_parallel_tasks=args.max_parallel,
        mcp_service_timeout=args.timeout,
        task_retry_count=args.retries,
        openai_model=args.model,
        database_path=args.db_path
    )

    # 初始化增强CLI界面
    cli = EnhancedCLIInterface()

    # 运行系统
    try:
        # 初始化系统
        enable_agentscope = not args.no_agentscope
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        initialized = loop.run_until_complete(cli.initialize(config, enable_agentscope))

        if not initialized:
            print("❌ 系统初始化失败，请检查配置")
            sys.exit(1)

        # 运行交互模式
        loop.run_until_complete(cli.run_interactive_mode())

    except KeyboardInterrupt:
        print("\n\n👋 系统已停止")
    except Exception as e:
        print(f"❌ 系统错误: {str(e)}")
        sys.exit(1)
    finally:
        # 清理资源
        try:
            loop.run_until_complete(cli.cleanup())
            loop.close()
        except:
            pass


if __name__ == "__main__":
    main()