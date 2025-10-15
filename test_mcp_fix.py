#!/usr/bin/env python3
"""
测试MCP服务修复效果的脚本
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from mpeo.coordinator import CLIInterface
from mpeo.models import SystemConfig


async def test_mcp_service():
    """测试MCP服务调用是否正常工作"""
    print("=" * 60)
    print("测试MCP服务修复效果")
    print("=" * 60)
    
    try:
        # 初始化系统
        config = SystemConfig(
            max_parallel_tasks=1,
            mcp_service_timeout=30,
            task_retry_count=1,
            openai_model="gpt-3.5-turbo",
            database_path="test_mcp_fix.db"
        )
        
        cli = CLIInterface()
        if not cli.initialize(config):
            print("❌ 系统初始化失败")
            return False
        
        print("✅ 系统初始化成功")
        
        # 检查MCP服务注册情况
        mcp_services = cli.coordinator.executor.mcp_services
        print(f"📋 注册的MCP服务: {list(mcp_services.keys())}")
        
        if not mcp_services:
            print("⚠️  没有注册MCP服务，无法测试")
            return False
        
        # 测试一个简单的查询，触发MCP调用
        print("\n🔍 开始测试MCP服务调用...")
        test_query = "获取今天的天气信息"
        
        result = await cli.coordinator.process_user_query(test_query)
        
        print(f"\n📝 测试结果:")
        print(f"结果长度: {len(result)} 字符")
        print(f"结果预览: {result[:200]}...")
        
        # 检查结果中是否包含错误信息
        if "405" in result and "Method Not Allowed" in result:
            print("❌ MCP服务仍然返回405错误，修复失败")
            return False
        elif "失败" in result and "MCP" in result:
            print("⚠️  MCP服务调用仍有问题，但可能不是405错误")
            return False
        else:
            print("✅ MCP服务调用测试通过，没有405错误")
            return True
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主函数"""
    success = await test_mcp_service()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 MCP服务修复验证成功！")
    else:
        print("💥 MCP服务修复验证失败！")
    print("=" * 60)
    
    # 清理测试数据库
    try:
        if os.path.exists("test_mcp_fix.db"):
            os.remove("test_mcp_fix.db")
            print("🧹 清理测试数据库完成")
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())