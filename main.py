"""
Main entry point for the Multi-model collaborative task processing system
多模型协作任务处理系统主入口
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from mpeo.coordinator import CLIInterface
from mpeo.models import SystemConfig


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="多模型协作任务处理系统")
    parser.add_argument("--config", help="配置文件路径", default=None)
    parser.add_argument("--max-parallel", type=int, help="最大并行任务数", default=4)
    parser.add_argument("--timeout", type=int, help="MCP服务超时时间(秒)", default=30)
    parser.add_argument("--retries", type=int, help="任务重试次数", default=3)
    parser.add_argument("--model", help="OpenAI模型名称", default="gpt-3.5-turbo")
    parser.add_argument("--db-path", help="数据库路径", default="mpeo.db")
    
    args = parser.parse_args()
    
    # Create system configuration
    config = SystemConfig(
        max_parallel_tasks=args.max_parallel,
        mcp_service_timeout=args.timeout,
        task_retry_count=args.retries,
        openai_model=args.model,
        database_path=args.db_path
    )
    
    # Initialize CLI interface
    cli = CLIInterface()
    
    # Initialize system
    if not cli.initialize(config):
        print("系统初始化失败，请检查配置")
        sys.exit(1)
    
    # Run interactive mode
    try:
        asyncio.run(cli.run_interactive_mode())
    except KeyboardInterrupt:
        print("\n系统已停止")
    except Exception as e:
        print(f"系统错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()