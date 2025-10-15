#!/usr/bin/env python3
"""
调试MCP API接口，找出正确的请求方式
"""

import asyncio
import aiohttp
import json
import sys
import os

async def test_mcp_api_methods():
    """测试不同的HTTP方法和参数格式"""

    # 测试的MCP服务URL
    test_urls = [
        "https://mcp.api-inference.modelscope.net/06178f20f6b343/sse",
        "https://mcp.api-inference.modelscope.net/709e37c4ef5847/sse"
    ]

    # 测试不同的请求方式
    test_methods = [
        {"method": "GET", "params": True, "data": False},
        {"method": "POST", "params": False, "data": True, "json": True},
        {"method": "POST", "params": True, "data": True, "form": True}
    ]

    # 测试headers
    test_headers = [
        {"Accept": "text/event-stream", "Content-Type": "application/json"},
        {"Accept": "application/json", "Content-Type": "application/json"},
        {"Accept": "*/*", "Content-Type": "application/json"},
        {"Accept": "text/event-stream", "Content-Type": "application/x-www-form-urlencoded"}
    ]

    # 测试payload
    test_payloads = [
        {"task": "test", "input": "hello"},
        {"messages": [{"role": "user", "content": "hello"}]},
        {"query": "test query"},
        {}
    ]

    print("🔍 开始测试MCP API接口...")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        for url in test_urls:
            print(f"\n📍 测试URL: {url}")
            print("-" * 40)

            for method_config in test_methods:
                method = method_config["method"]
                use_params = method_config.get("params", False)
                use_data = method_config.get("data", False)
                use_json = method_config.get("json", False)
                use_form = method_config.get("form", False)

                for headers in test_headers:
                    print(f"\n📤 尝试: {method} 请求")
                    print(f"   Headers: {headers}")

                    try:
                        # 准备请求参数
                        kwargs = {"headers": headers}

                        if use_params and test_payloads:
                            kwargs["params"] = test_payloads[0]

                        if use_data:
                            if use_json and test_payloads:
                                kwargs["json"] = test_payloads[0]
                            elif use_form and test_payloads:
                                kwargs["data"] = test_payloads[0]

                        kwargs["timeout"] = aiohttp.ClientTimeout(total=30)

                        # 发送请求
                        async with session.request(method, url, **kwargs) as response:
                            print(f"   ✅ 状态码: {response.status}")
                            print(f"   📋 响应头: {dict(response.headers)}")

                            # 读取响应内容
                            content_type = response.headers.get('Content-Type', '')
                            if 'application/json' in content_type:
                                response_data = await response.json()
                                print(f"   📄 响应数据: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
                            else:
                                response_text = await response.text()
                                print(f"   📄 响应文本: {response_text[:200]}...")

                            # 如果请求成功，打印成功信息
                            if response.status == 200:
                                print("   🎉 请求成功！这个配置可能有效")
                                return True

                    except asyncio.TimeoutError:
                        print("   ⏰ 请求超时")
                    except aiohttp.ClientError as e:
                        print(f"   ❌ 客户端错误: {str(e)}")
                    except Exception as e:
                        print(f"   ❌ 其他错误: {str(e)}")

                    print("   " + "-" * 30)

    print("\n❌ 所有测试都失败了，可能需要查看MCP服务的官方文档")
    return False

async def test_sse_connection():
    """专门测试SSE连接"""
    print("\n🌊 测试SSE连接...")
    print("=" * 60)

    url = "https://mcp.api-inference.modelscope.net/06178f20f6b343/sse"
    headers = {
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as response:
                print(f"状态码: {response.status}")
                print(f"响应头: {dict(response.headers)}")

                if response.status == 200:
                    print("✅ SSE连接成功！")
                    line_count = 0
                    async for line in response.content:
                        if line:
                            line_text = line.decode('utf-8').strip()
                            if line_text:
                                print(f"📡 行 {line_count}: {line_text}")
                                line_count += 1
                                if line_count > 10:  # 限制输出行数
                                    break
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ SSE连接失败: {error_text}")
                    return False

    except Exception as e:
        print(f"❌ SSE连接异常: {str(e)}")
        return False

async def main():
    """主函数"""
    print("🧪 MCP API接口调试工具")

    # 测试不同的请求方式
    success1 = await test_mcp_api_methods()

    # 测试SSE连接
    success2 = await test_sse_connection()

    if success1 or success2:
        print("\n🎉 找到了可用的API调用方式！")
    else:
        print("\n💥 所有API调用方式都失败了")
        print("建议：")
        print("1. 查看ModelScope MCP服务的官方文档")
        print("2. 检查是否需要特殊的认证方式")
        print("3. 确认API端点URL是否正确")

if __name__ == "__main__":
    asyncio.run(main())