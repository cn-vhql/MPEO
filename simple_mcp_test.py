#!/usr/bin/env python3
"""
简化的MCP服务验证脚本
"""

import asyncio
import aiohttp
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mpeo.models import MCPServiceConfig


async def test_mcp_service():
    """Test MCP service configuration"""
    print("🚀 开始验证MCP服务配置...")
    
    # Test configuration from user feedback
    service_config = MCPServiceConfig(
        service_name="fetch",
        endpoint_url="https://mcp.api-inference.modelscope.net/f72f96405fff4b/sse",
        timeout=30,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
    )
    
    print(f"\n🔍 测试MCP服务: {service_config.service_name}")
    print(f"   URL: {service_config.endpoint_url}")
    print(f"   类型: SSE (Server-Sent Events)")
    print(f"   超时: {service_config.timeout}s")
    print(f"   请求头: {service_config.headers}")
    
    # Prepare test payload
    test_payload = {
        "task_id": "test_task",
        "input_data": {
            "query": "https://zhuanlan.zhihu.com/p/1901715821740421904",
            "test": True
        },
        "timeout": service_config.timeout
    }
    
    print(f"\n📡 测试请求内容:")
    print(f"   {json.dumps(test_payload, ensure_ascii=False, indent=2)}")
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"\n📡 发送测试请求...")
            
            async with session.post(
                service_config.endpoint_url,
                json=test_payload,
                headers=service_config.headers or {},
                timeout=aiohttp.ClientTimeout(total=service_config.timeout)
            ) as response:
                print(f"\n📊 响应状态: {response.status}")
                print(f"   响应头: {dict(response.headers)}")
                
                if response.status == 200:
                    print(f"   ✅ 服务连接成功!")
                    
                    # Check content type
                    content_type = response.headers.get('Content-Type', '')
                    print(f"   📋 Content-Type: {content_type}")
                    
                    if 'text/event-stream' in content_type:
                        print(f"   🔄 SSE流响应，开始读取...")
                        line_count = 0
                        async for line in response.content:
                            if line:
                                line_text = line.decode('utf-8').strip()
                                if line_text and line_text != 'data:':
                                    print(f"   📝 {line_text}")
                                    line_count += 1
                                    if line_count >= 3:  # Limit output for testing
                                        break
                        print(f"   ✅ 成功读取SSE流数据")
                    else:
                        # Try to read as text
                        response_text = await response.text()
                        print(f"   📄 响应内容 (前500字符):")
                        print(f"   {response_text[:500]}")
                        
                        # Try to parse as JSON
                        try:
                            response_data = json.loads(response_text)
                            print(f"   📦 JSON响应解析成功")
                            print(f"   {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                        except json.JSONDecodeError:
                            print(f"   📄 响应为非JSON格式")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"   ❌ 服务返回错误: {response.status}")
                    print(f"   错误内容: {error_text[:500]}")
                    return False
                    
    except asyncio.TimeoutError:
        print(f"   ⏰ 请求超时 ({service_config.timeout}s)")
        return False
    except Exception as e:
        print(f"   ❌ 请求失败: {str(e)}")
        print(f"   错误类型: {type(e).__name__}")
        return False


async def test_config_file():
    """Test configuration file loading"""
    print(f"\n📁 测试配置文件加载...")
    
    config_file_path = "mcp_config.json"
    if not os.path.exists(config_file_path):
        print(f"   ❌ 配置文件不存在: {config_file_path}")
        return False
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        print(f"   ✅ 配置文件加载成功: {config_file_path}")
        print(f"   📋 配置内容:")
        print(f"   {json.dumps(config_data, ensure_ascii=False, indent=2)}")
        
        if "mcpServices" in config_data:
            services = config_data["mcpServices"]
            print(f"   📋 发现 {len(services)} 个MCP服务配置")
            
            for service_name, service_config in services.items():
                print(f"   🔍 服务: {service_name}")
                print(f"      URL: {service_config.get('url', 'N/A')}")
                print(f"      类型: {service_config.get('type', 'N/A')}")
                print(f"      超时: {service_config.get('timeout', 'N/A')}")
            
            return True
        else:
            print(f"   ❌ 配置文件中缺少 'mcpServices' 字段")
            return False
            
    except Exception as e:
        print(f"   ❌ 配置文件解析失败: {str(e)}")
        return False


async def main():
    """Main test function"""
    print("=" * 60)
    print("MCP服务配置验证工具")
    print("=" * 60)
    
    # Test 1: Configuration file loading
    config_success = await test_config_file()
    
    # Test 2: MCP service connection
    service_success = await test_mcp_service()
    
    # Final results
    print(f"\n" + "=" * 60)
    print("📋 验证结果")
    print("=" * 60)
    print(f"配置文件加载: {'✅ 成功' if config_success else '❌ 失败'}")
    print(f"MCP服务连接: {'✅ 成功' if service_success else '❌ 失败'}")
    
    if config_success and service_success:
        print(f"\n🎉 所有验证通过！MCP服务配置正确且可以正常连接。")
        print(f"\n💡 建议:")
        print(f"   1. 配置文件格式正确")
        print(f"   2. MCP服务可以正常连接")
        print(f"   3. SSE流响应正常")
        print(f"   4. 系统可以正常使用此服务")
        return 0
    else:
        print(f"\n⚠️  部分验证失败，请检查:")
        if not config_success:
            print(f"   ❌ 配置文件格式或路径")
        if not service_success:
            print(f"   ❌ MCP服务连接或响应")
        print(f"\n💡 可能的解决方案:")
        print(f"   1. 检查网络连接")
        print(f"   2. 验证API端点URL")
        print(f"   3. 确认请求头配置")
        print(f"   4. 检查服务是否在线")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)