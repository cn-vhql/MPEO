"""
MCP服务优化使用示例
展示如何使用新的MCP服务管理器和优化后的executor
"""

import asyncio
import logging
from pprint import pprint

from mpeo.core.executor import TaskExecutor
from mpeo.models import MCPServiceConfig, SystemConfig, TaskNode, TaskType
from mpeo.services.database import DatabaseManager
from mpeo.services.mcp_manager import MCPServiceManager, MCPConnectionConfig
from openai import OpenAI


async def example_mcp_service_usage():
    """演示MCP服务管理器的使用"""

    # 1. 创建MCP服务管理器
    async with MCPServiceManager() as mcp_manager:
        print("=== MCP服务管理器初始化完成 ===")

        # 2. 注册STDIO类型的MCP服务（类似mcp_chatbot-master的方式）
        stdio_config = MCPConnectionConfig(
            name="fetch_service",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-fetch"],
            timeout=30
        )

        success = await mcp_manager.register_service(stdio_config)
        print(f"STDIO服务注册: {'成功' if success else '失败'}")

        # 3. 注册HTTP类型的MCP服务
        http_config = MCPConnectionConfig(
            name="web_search_service",
            command="curl",  # 标识为HTTP类型
            args=["http://localhost:8080/mcp"],  # 端点URL
            timeout=60
        )

        success = await mcp_manager.register_service(http_config)
        print(f"HTTP服务注册: {'成功' if success else '失败'}")

        # 4. 获取所有可用工具
        tools = await mcp_manager.get_available_tools()
        print("\n=== 可用工具列表 ===")
        for service_name, service_tools in tools.items():
            print(f"\n服务: {service_name}")
            for tool in service_tools:
                print(f"  - {tool.name}: {tool.description}")
                print(f"    参数: {tool.input_schema}")

        # 5. 调用工具示例
        print("\n=== 工具调用示例 ===")
        try:
            result = await mcp_manager.call_tool(
                service_name="fetch_service",
                tool_name="fetch",
                arguments={
                    "url": "https://example.com",
                    "method": "GET"
                }
            )

            if result.success:
                print(f"工具调用成功，耗时: {result.execution_time:.2f}s")
                print(f"结果: {result.data}")
            else:
                print(f"工具调用失败: {result.error}")

        except Exception as e:
            print(f"工具调用异常: {str(e)}")


async def example_optimized_executor():
    """演示优化后的executor使用"""

    # 创建数据库管理器
    db_manager = DatabaseManager()

    # 创建系统配置
    config = SystemConfig(
        openai_model="gpt-3.5-turbo",
        max_parallel_tasks=4,
        task_retry_count=2
    )

    # 创建OpenAI客户端（需要设置OPENAI_API_KEY环境变量）
    try:
        openai_client = OpenAI()
    except Exception as e:
        print(f"OpenAI客户端初始化失败: {e}")
        print("请确保设置了OPENAI_API_KEY环境变量")
        return

    # 创建优化后的executor
    executor = TaskExecutor(
        openai_client=openai_client,
        database=db_manager,
        config=config
    )

    print("=== 优化后的TaskExecutor初始化完成 ===")

    # 注册MCP服务
    mcp_service_config = MCPServiceConfig(
        service_name="fetch_service",
        service_type="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-fetch"],
        timeout=30
    )

    await executor.register_mcp_service(mcp_service_config)
    print("MCP服务注册完成")

    # 创建一个MCP调用任务
    task = TaskNode(
        task_id="test_mcp_task",
        task_desc="使用MCP服务获取网页内容",
        task_type=TaskType.MCP_CALL,
        expected_output="网页的HTML内容",
        priority=1
    )

    print("\n=== 执行MCP任务 ===")
    try:
        input_data = {"url": "https://example.com"}
        result = await executor._execute_mcp_call(
            task=task,
            input_data=input_data,
            session_id="test_session"
        )

        print(f"MCP任务执行成功: {result}")

    except Exception as e:
        print(f"MCP任务执行失败: {str(e)}")

    finally:
        # 清理资源
        await executor.cleanup_mcp_manager()
        print("资源清理完成")


def main():
    """主函数"""
    print("MCP服务优化示例")
    print("=" * 50)

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 运行示例
    try:
        asyncio.run(example_mcp_service_usage())
        print("\n" + "=" * 50)
        asyncio.run(example_optimized_executor())
    except KeyboardInterrupt:
        print("\n示例被用户中断")
    except Exception as e:
        print(f"\n示例运行出错: {str(e)}")


if __name__ == "__main__":
    main()