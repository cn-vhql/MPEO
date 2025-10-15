#!/usr/bin/env python3
"""
MCP服务测试脚本 - 用于检测是MCP服务问题还是调用代码问题
"""

import asyncio
import aiohttp
import json
import time
import urllib.parse

async def test_mcp_service_direct():
    """直接测试MCP服务端点"""
    print("🔧️ 开始直接测试MCP服务端点...")
    
    # 测试配置 - 从mcp_config.json中获取
    test_services = [
        {
            "name": "default",
            "url": "https://mcp.api-inference.modelscope.net/06178f20f6b343/sse",
            "type": "sse"
        },
        {
            "name": "fetch", 
            "url": "https://mcp.api-inference.modelscope.net/06178f20f6b343/sse",
            "type": "sse"
        },
        {
            "name": "bing-cn-mcp-server",
            "url": "https://mcp.api-inference.modelscope.net/709e37c4ef5847/sse", 
            "type": "sse"
        }
    ]
    
    # 测试数据
    test_input = {
        "task_id": "test_task_001",
        "task_desc": "测试MCP服务连接",
        "expected_output": "服务响应结果",
        "timeout": 30
    }
    
    for service in test_services:
        print(f"\n📡 测试服务: {service['name']}")
        print(f"   URL: {service['url']}")
        print(f"   类型: {service['type']}")
        
        # 测试1: 基本连接测试
        print(f"   🔄 测试1: 基本连接测试...")
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(
                    service['url'],
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    elapsed = time.time() - start_time
                    print(f"      ✅ 连接成功! 状态: {response.status}, 耗时: {elapsed:.2f}s")
                    print(f"      📋 响应头: {dict(response.headers)}")
                    
                    # 对于SSE流，不尝试读取全部内容（会导致超时）
                    # 只检查响应头和连接状态
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/event-stream' in content_type:
                        print(f"      📄 检测到SSE流，跳过内容读取（避免超时）")
                    else:
                        # 非SSE响应，可以安全读取
                        response_text = await response.text()
                        print(f"      📄 响应内容 (前100字符): {response_text[:100]}...")
                    
        except Exception as e:
            print(f"      ❌ 连接失败: {type(e).__name__}: {str(e)}")
            continue
        
        # 测试2: 带查询参数的SSE请求
        print(f"   🔄 测试2: 带查询参数的SSE请求...")
        try:
            # 构建查询参数（与executor.py中的逻辑相同）
            params = {
                "task_id": test_input["task_id"],
                "timeout": str(test_input["timeout"])
            }
            
            # 添加输入数据作为查询参数
            for key, value in test_input.items():
                if isinstance(value, dict):
                    params[f"input_{key}"] = json.dumps(value)
                else:
                    params[f"input_{key}"] = str(value)
            
            # 构建URL
            url = service['url']
            if params:
                query_string = urllib.parse.urlencode(params)
                url = f"{url}?{query_string}"
            
            print(f"      📡 请求URL: {url[:100]}...")
            print(f"      📋 查询参数: {params}")
            
            # 准备SSE头
            headers = {
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    elapsed = time.time() - start_time
                    print(f"      ✅ 请求成功! 状态: {response.status}, 耗时: {elapsed:.2f}s")
                    print(f"      📋 响应头: {dict(response.headers)}")
                    
                    content_type = response.headers.get('Content-Type', '')
                    print(f"      📋 Content-Type: {content_type}")
                    
                    if 'text/event-stream' in content_type:
                        print(f"      🔄 检测到SSE流，开始读取...")
                        line_count = 0
                        event_count = 0
                        async for line in response.content:
                            if line:
                                line_text = line.decode('utf-8').strip()
                                if line_text.startswith('data:'):
                                    event_count += 1
                                    if line_text.startswith('data: '):
                                        event_data = line_text[6:]
                                    else:
                                        event_data = line_text[5:]
                                    print(f"      📝 事件 {event_count}: {event_data[:80]}...")
                                    line_count += 1
                                    if line_count >= 3:  # 限制输出
                                        break
                        print(f"      ✅ SSE流读取完成，共 {event_count} 个事件")
                    else:
                        response_text = await response.text()
                        print(f"      📄 非SSE响应 (前200字符): {response_text[:200]}...")
                        
        except asyncio.TimeoutError:
            print(f"      ⏰ 请求超时 (30秒)")
        except Exception as e:
            print(f"      ❌ 请求失败: {type(e).__name__}: {str(e)}")
        
        # 测试3: 简化的POST请求（对比测试）
        print(f"   🔄 测试3: 简化的POST请求...")
        try:
            payload = {
                "task_id": test_input["task_id"],
                "input_data": test_input,
                "timeout": test_input["timeout"]
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post(
                    service['url'],
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    elapsed = time.time() - start_time
                    print(f"      ✅ POST请求成功! 状态: {response.status}, 耗时: {elapsed:.2f}s")
                    print(f"      📋 响应头: {dict(response.headers)}")
                    
                    response_text = await response.text()
                    print(f"      📄 响应内容 (前200字符): {response_text[:200]}...")
                        
        except asyncio.TimeoutError:
            print(f"      ⏰ POST请求超时 (30秒)")
        except Exception as e:
            print(f"      ❌ POST请求失败: {type(e).__name__}: {str(e)}")
        
        print(f"   ✅ 服务 {service['name']} 测试完成")
        print(f"   {'='*60}")

async def test_different_parameters():
    """测试不同参数组合"""
    print(f"\n🔧️ 测试不同参数组合...")
    
    base_url = "https://mcp.api-inference.modelscope.net/709e37c4ef5847/sse"
    
    # 测试不同的参数组合
    test_cases = [
        {
            "name": "最小参数",
            "params": {"task_id": "test", "timeout": "30"}
        },
        {
            "name": "带简单输入",
            "params": {
                "task_id": "test", 
                "timeout": "30",
                "input_task_desc": "测试任务"
            }
        },
        {
            "name": "带复杂输入",
            "params": {
                "task_id": "test",
                "timeout": "30", 
                "input_task_desc": "测试任务",
                "input_expected_output": "测试输出"
            }
        },
        {
            "name": "带JSON输入",
            "params": {
                "task_id": "test",
                "timeout": "30",
                "input_data": json.dumps({"query": "test", "test": True})
            }
        }
    ]
    
    for case in test_cases:
        print(f"\n   📡 测试: {case['name']}")
        
        # 构建URL
        url = base_url
        if case['params']:
            query_string = urllib.parse.urlencode(case['params'])
            url = f"{url}?{query_string}"
        
        print(f"      URL: {url[:100]}...")
        
        try:
            headers = {
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            }
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15)  # 缩短超时时间
                ) as response:
                    elapsed = time.time() - start_time
                    print(f"      ✅ 成功! 状态: {response.status}, 耗时: {elapsed:.2f}s")
                    
                    content_type = response.headers.get('Content-Type', '')
                    print(f"      Content-Type: {content_type}")
                    
                    if 'text/event-stream' in content_type:
                        print(f"      🔄 检测到SSE流")
                        # 只读取第一个事件
                        async for line in response.content:
                            if line:
                                line_text = line.decode('utf-8').strip()
                                if line_text.startswith('data:'):
                                    if line_text.startswith('data: '):
                                        event_data = line_text[6:]
                                    else:
                                        event_data = line_text[5:]
                                    print(f"      📝 事件: {event_data[:100]}...")
                                    break
                    else:
                        response_text = await response.text()
                        print(f"      📄 响应: {response_text[:100]}...")
                        
        except asyncio.TimeoutError:
            print(f"      ⏰ 超时 (15秒)")
        except Exception as e:
            print(f"      ❌ 失败: {type(e).__name__}: {str(e)}")

async def main():
    """主测试函数"""
    print("=" * 60)
    print("MCP服务诊断工具")
    print("=" * 60)
    
    # 测试1: 直接服务测试
    await test_mcp_service_direct()
    
    # 测试2: 不同参数测试
    await test_different_parameters()
    
    print(f"\n" + "=" * 60)
    print("📋 诊断结果分析")
    print("=" * 60)
    
    print("""
🔍 问题分析指南：

1. **如果所有测试都超时**：
   - ❌ 可能是网络连接问题
   - ❌ 可能是MCP服务端点完全不可用
   - ❌ 可能是防火墙或代理问题

2. **如果基本连接成功但SSE请求超时**：
   - ⚠️ 可能是查询参数格式问题
   - ⚠️ 可能是SSE端点需要特定参数
   - ⚠️ 可能是URL编码问题

3. **如果SSE请求成功但内容异常**：
   - 🔧 可能是SSE事件格式解析问题
   - 🔧 可能是响应内容格式不符合预期
   - 🔧 可能需要不同的请求头

4. **如果POST请求成功但GET失败**：
   - 📡 说明端点支持POST但不支持GET
   - 📡 需要修改调用代码使用POST方法

5. **如果某些参数组合成功**：
   - ✨ 说明是参数格式问题
   - ✨ 需要调整参数构造逻辑

💡 建议的解决步骤：
1. 首先确认网络连接和端点可用性
2. 检查正确的HTTP方法（GET vs POST）
3. 验证必需的参数和格式
4. 测试不同的请求头组合
5. 根据测试结果调整调用代码
    """)

if __name__ == "__main__":
    asyncio.run(main())