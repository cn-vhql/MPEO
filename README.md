# 多模型协作任务处理系统 (MPEO)

一个支持"模型自动处理 + 人工干预确认"的多模型协作系统，通过规划模型、人工反馈环节、执行模型、输出模型的协同，将用户输入的问题/需求转化为结构化处理流程，经人工确认后执行，最终生成符合预期的结果。

## 🚀 功能特性

### 核心组件

1. **规划模型 (Planner Model)**
   - 智能分析用户需求，自动分解为可执行任务
   - 生成有向无环图 (DAG) 形式的任务结构
   - 支持任务依赖关系定义和优先级分配

2. **人工反馈环节 (Human Feedback Interface)**
   - 可视化展示任务图，支持确认和修改
   - 提供任务的增删改查操作
   - 自动校验DAG无环性，防止循环依赖

3. **执行模型 (Executor Model)**
   - 基于DAG的智能任务调度
   - 支持串行和并行执行模式
   - 集成MCP服务调用和本地计算
   - 任务失败自动重试机制

4. **输出模型 (Output Model)**
   - 智能整合多任务执行结果
   - 自动检测和解决结果冲突
   - 支持多种输出格式（文本、表格、报告等）

### 技术特点

- 🤖 **AI驱动**: 使用OpenAI GPT模型进行智能任务分解和结果整合
- 🔄 **异步执行**: 支持多任务并行处理，提高执行效率
- 💾 **数据持久化**: 使用SQLite存储会话历史和执行结果
- 🎛️ **可配置**: 支持灵活的系统配置和MCP服务注册
- 📊 **可视化**: 基于Rich库的美观命令行界面
- 🔍 **日志追踪**: 完整的操作日志记录和错误追踪

## 📦 安装和配置

### 环境要求

- Python 3.11+
- OpenAI API Key

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd MPEO
```

2. **安装依赖**
```bash
pip install -e .
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，添加你的 OpenAI API Key
```

4. **验证安装**
```bash
python main.py --help
```

## 🎯 使用方法

### 基本使用

启动交互式界面：
```bash
python main.py
```

### 命令行参数

```bash
python main.py [选项]

选项:
  --config CONFIG       配置文件路径
  --max-parallel MAX    最大并行任务数 (默认: 4)
  --timeout TIMEOUT     MCP服务超时时间(秒) (默认: 30)
  --retries RETRIES     任务重试次数 (默认: 3)
  --model MODEL         OpenAI模型名称 (默认: gpt-3.5-turbo)
  --db-path DB_PATH     数据库路径 (默认: mpeo.db)
```

### 交互式命令

在系统运行时，可以使用以下命令：

- `help` / `帮助` - 显示帮助信息
- `quit` / `exit` / `退出` - 退出系统
- `status` / `状态` - 显示系统状态
- `history` / `历史` - 显示会话历史
- `logs` / `日志` - 显示系统日志
- `config set <key> <value>` - 设置配置
- `config get <key>` - 获取配置
- `mcp register <service_name> <url>` - 注册MCP服务

### 使用示例

1. **简单查询**
```
请帮我分析一下当前的市场趋势
```

2. **复杂数据处理**
```
请帮我分析最近一个月的销售数据，生成包括趋势分析、异常检测和预测报告的完整分析结果
```

3. **指定输出格式**
```
请以表格形式展示产品对比分析结果
```

## 🏗️ 系统架构

```
用户输入
    ↓
规划模型 (Planner Model)
    ↓
生成任务图 (DAG)
    ↓
人工反馈环节 (Human Feedback)
    ↓
确认/修改任务图
    ↓
执行模型 (Executor Model)
    ↓
任务调度执行
    ↓
输出模型 (Output Model)
    ↓
最终结果
```

## 📊 数据模型

### 任务图 (Task Graph)

```json
{
  "task_graph": {
    "nodes": [
      {
        "task_id": "T1",
        "task_desc": "数据收集",
        "task_type": "mcp调用",
        "expected_output": "原始数据集",
        "priority": 5
      }
    ],
    "edges": [
      {
        "from_task_id": "T1",
        "to_task_id": "T2",
        "dependency_type": "数据依赖"
      }
    ]
  }
}
```

### 执行结果 (Execution Results)

```json
{
  "execution_results": [
    {
      "task_id": "T1",
      "status": "success",
      "output": "处理结果",
      "execution_time": 2.5,
      "error_msg": null
    }
  ],
  "total_execution_time": 5.2,
  "success_count": 3,
  "failed_count": 0
}
```

## 🔧 MCP服务集成

系统支持与外部MCP服务集成，用于执行特定领域的任务：

### 注册MCP服务

```bash
# 在交互界面中执行
mcp register weather_service http://localhost:8080/weather
mcp register data_analyzer http://localhost:8081/analyze
```

### MCP服务接口规范

MCP服务需要实现以下HTTP接口：

```http
POST /endpoint
Content-Type: application/json

{
  "task_id": "T1",
  "input_data": {...},
  "timeout": 30
}
```

响应格式：
```json
{
  "status": "success",
  "result": {...},
  "message": "处理完成"
}
```

## 📝 配置说明

### 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| OPENAI_API_KEY | OpenAI API密钥 | 必填 |
| OPENAI_MODEL | OpenAI模型名称 | gpt-3.5-turbo |
| MAX_PARALLEL_TASKS | 最大并行任务数 | 4 |
| MCP_SERVICE_TIMEOUT | MCP服务超时时间(秒) | 30 |
| TASK_RETRY_COUNT | 任务重试次数 | 3 |
| DATABASE_PATH | 数据库路径 | mpeo.db |

### 系统配置

系统支持运行时配置修改：

```bash
# 设置最大并行任务数
config set max_parallel_tasks 8

# 查看当前配置
config get openai_model
```

## 🗄️ 数据存储

系统使用SQLite数据库存储以下数据：

- **会话记录**: 用户查询和系统响应
- **任务图**: 各版本的任务图数据
- **执行结果**: 详细的任务执行记录
- **系统日志**: 完整的操作和错误日志
- **配置信息**: 系统和MCP服务配置

### 数据库结构

```sql
-- 会话表
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_query TEXT NOT NULL,
    task_graph TEXT,
    execution_results TEXT,
    final_output TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT DEFAULT 'created'
);

-- 任务图表
CREATE TABLE task_graphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    graph_version INTEGER DEFAULT 1,
    graph_data TEXT NOT NULL,
    is_final BOOLEAN DEFAULT FALSE,
    created_at TEXT NOT NULL
);

-- 日志表
CREATE TABLE logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    component TEXT NOT NULL,
    operation TEXT NOT NULL,
    details TEXT,
    timestamp TEXT NOT NULL
);
```

## 🐛 故障排除

### 常见问题

1. **OpenAI API错误**
   - 检查API密钥是否正确设置
   - 确认API配额是否充足
   - 检查网络连接

2. **任务执行失败**
   - 查看系统日志了解详细错误信息
   - 检查MCP服务是否正常运行
   - 调整任务重试次数和超时时间

3. **数据库错误**
   - 确认数据库文件权限
   - 检查磁盘空间是否充足
   - 重新初始化数据库

### 日志查看

```bash
# 查看系统日志
logs

# 查看特定会话日志
logs <session_id>
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 支持

如果您遇到问题或有建议，请：

1. 查看本文档的故障排除部分
2. 检查项目的 Issues 页面
3. 创建新的 Issue 描述您的问题

## 🔄 更新日志

### v0.1.0 (2024-10-14)
- 初始版本发布
- 实现基本的多模型协作功能
- 支持任务图生成和人工确认
- 集成OpenAI API和MCP服务调用
- 提供完整的命令行界面

---

**多模型协作任务处理系统** - 让AI协作更智能、更可控！