# MPEO 配置文件说明

本目录包含MPEO系统的各种配置文件。

## 文件说明

### MCP服务配置

- **`mcp_services.example.json`** - MCP服务配置模板文件
- **`mcp_services.json`** - 实际使用的MCP服务配置（需要自己创建）

### 使用方法

1. **复制配置模板**
   ```bash
   cp config/mcp_services.example.json config/mcp_services.json
   ```

2. **根据需要修改配置**
   - 启用或禁用特定的MCP服务
   - 调整超时时间和重试参数
   - 修改服务URL（如果使用不同的服务端点）

3. **配置文件结构**
   ```json
   {
     "mcpServices": {
       "service-name": {
         "type": "sse",
         "url": "service-endpoint",
         "timeout": 30,
         "headers": {
           "Content-Type": "application/json",
           "Accept": "text/event-stream"
         }
       }
     },
     "configuration": {
       "globalTimeout": 30,
       "retryAttempts": 3
     }
   }
   ```

## MCP服务列表

| 服务名称 | 描述 | 用途 |
|---------|------|------|
| `fetch` | Web内容抓取 | 获取网页内容、API数据 |
| `tavily-server` | 网络搜索 | 实时网络搜索功能 |
| `context7-mcp-server` | 上下文处理 | 文本理解、上下文分析 |
| `bing-cn-mcp-server` | 必应搜索 | 中文搜索优化 |

## 安全注意事项

⚠️ **重要提醒**：
- 配置文件可能包含敏感信息（URL、API密钥等）
- **不要将实际的 `mcp_services.json` 提交到版本控制系统**
- 生产环境中应使用环境变量或安全的密钥管理系统
- 定期轮换服务端点的认证信息

## 故障排除

### 常见问题

1. **服务连接超时**
   - 检查网络连接
   - 增加超时时间配置
   - 验证服务端点是否可用

2. **服务响应错误**
   - 检查服务配置是否正确
   - 验证headers设置
   - 查看系统日志获取详细错误信息

3. **性能问题**
   - 调整并发请求数量
   - 优化超时和重试设置
   - 监控服务响应时间

## 开发建议

- 在开发环境中使用测试端点
- 为不同环境创建不同的配置文件
- 使用配置验证工具确保格式正确
- 定期备份配置文件