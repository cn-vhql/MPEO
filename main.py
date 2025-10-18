"""
Main entry point for the Multi-model collaborative task processing system
多模型协作任务处理系统主入口
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

from mpeo.core.cli import NewCLIInterface
from mpeo.core.coordinator import SystemCoordinatorFactory
from mpeo.models import SystemConfig, AgentModelConfig
from mpeo.services import get_config_loader
from mpeo.utils.logging import get_logger


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="多模型协作任务处理系统")
    parser.add_argument("--config", help="配置文件路径", default=None)
    parser.add_argument("--max-parallel", type=int, help="最大并行任务数", default=4)
    parser.add_argument("--timeout", type=int, help="MCP服务超时时间(秒)", default=30)
    parser.add_argument("--retries", type=int, help="任务重试次数", default=3)
    parser.add_argument("--model", help="OpenAI模型名称", default="gpt-3.5-turbo")
    parser.add_argument("--db-path", help="数据库路径", default="data/databases/mpeo.db")

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

    # 初始化CLI界面
    cli = NewCLIInterface()

    # 运行系统
    try:
        # 初始化系统
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 异步初始化系统
        initialized = loop.run_until_complete(cli.initialize(config))

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
            if cli.coordinator:
                loop.run_until_complete(cli.cleanup())
            loop.close()
        except:
            pass


if __name__ == "__main__":
    main()
