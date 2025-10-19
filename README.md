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

- 🤖 **AI驱动**: 使用大语言模型进行智能任务分解和结果整合
- 🔄 **异步执行**: 支持多任务并行处理，提高执行效率
- 💾 **数据持久化**: 使用SQLite存储会话历史和执行结果
- 🎛️ **可配置**: 支持灵活的系统配置和MCP服务注册
- 📊 **可视化**: 基于Rich库的美观命令行界面
- 🔍 **日志追踪**: 完整的操作日志记录和错误追踪
- 🔌 **MCP集成**: 完整支持Model Context Protocol标准

## 📦 安装和配置

### 环境要求

- Python 3.11+
- ModelScope API 密钥（推荐）或 OpenAI 兼容 API 密钥
- 支持 uv（可选，用于更快的依赖安装）

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd MPEO
```

2. **安装依赖**
```bash
pip install -e .
# 或者使用 uv (如果可用)
uv pip install -e .
```

3. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，添加你的 API 密钥

# 对于 ModelScope API（推荐）：
OPENAI_API_KEY=your-modelscope-api-key
OPENAI_API_BASE=https://api-inference.modelscope.cn/v1

# 对于 OpenAI API：
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_API_BASE=https://api.openai.com/v1
```

4. **配置智能体模型（可选）**
```bash
# 编辑 config/agent_models.json 文件来配置不同智能体的模型
# 系统会自动使用环境变量中的配置，无需额外设置
```

5. **验证安装**
```bash
# 运行测试脚本验证配置
python test_fix.py

# 或直接查看帮助信息
python main.py --help
```

## 🎯 使用方法

### 基本使用

**方式一：使用启动脚本（推荐）**
```bash
# 使用预配置的启动脚本
./start.sh
```

**方式二：直接运行**
```bash
# 确保环境变量正确设置后启动
python main.py
```

**方式三：自定义参数**
```bash
python main.py [选项]

选项:
  --config CONFIG       配置文件路径
  --max-parallel MAX    最大并行任务数 (默认: 4)
  --timeout TIMEOUT     MCP服务超时时间(秒) (默认: 30)
  --retries RETRIES     任务重试次数 (默认: 3)
  --model MODEL         大模型名称 (默认: Qwen/Qwen3-30B-A3B-Instruct-2507)
  --db-path DB_PATH     数据库路径 (默认: data/databases/mpeo.db)
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

1. **时间查询**
```
当前时间
```

2. **网页抓取**
```
请抓取 https://example.com 的内容
```

3. **复杂数据处理**
```
请帮我分析最近一个月的销售数据，生成包括趋势分析、异常检测和预测报告的完整分析结果
```

4. **趋势分析**
```
请分析当前最热门的编程语言和框架趋势
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

## 🔧 MCP服务集成

系统已预配置了多个MCP服务，支持以下功能：

### 预配置服务

1. **fetch** - 网页抓取工具
   - 获取URL内容并转换为markdown格式
   - 支持多种内容格式

2. **context7-mcp** - 文档查询工具
   - 获取编程库的最新文档
   - 支持代码示例和API参考

3. **time** - 时间处理工具
   - 当前时间查询
   - 时区转换
   - 时间戳处理
   - 日期计算

4. **weather** - 天气查询工具（自定义）
   - 实时天气数据获取
   - 天气预报查询
   - 支持多种数据源整合

### MCP服务配置

MCP服务配置位于 `config/mcp_services.json`：

```json
{
  "mcpServices": {
    "time": {
      "type": "streamable_http",
      "url": "https://mcp.api-inference.modelscope.net/0edb46e720b744/mcp",
      "timeout": 30,
      "description": "时间处理工具 - 当前时间、时区转换、时间戳等功能"
    },
    "fetch": {
      "type": "streamable_http",
      "url": "https://mcp.api-inference.modelscope.net/9ccb10acb11f4d/mcp",
      "timeout": 30,
      "description": "网页抓取工具 - 获取URL内容并转换为markdown格式"
    },
    "context7-mcp": {
      "type": "streamable_http",
      "url": "https://mcp.api-inference.modelscope.net/d49e76846b6647/mcp",
      "timeout": 30,
      "description": "Context7文档查询工具 - 获取编程库的最新文档"
    },
    "weather": {
      "type": "streamable_http",
      "url": "http://192.168.3.86:20080/e/az10dnd0els2jb54/mcp",
      "timeout": 60,
      "enabled": true,
      "headers": {
        "Content-Type": "application/json",
        "User-Agent": "MPEO-MCP-Client/2.0"
      },
      "description": "天气查询工具 - 实时天气数据和天气预报"
    }
  }
}
```

### 动态注册MCP服务

```bash
# 在交互界面中执行
mcp register custom_service http://localhost:8080/mcp
```

## 📊 数据模型

### 任务图 (Task Graph)

系统使用DAG结构表示任务依赖关系：

- **任务节点**: 包含任务描述、类型、优先级等信息
- **依赖边**: 定义任务间的数据依赖关系
- **执行调度**: 根据依赖关系自动调度任务执行

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

## 📝 配置说明

### 环境变量配置

| 变量名 | 说明 | 默认值 | 推荐值 |
|--------|------|--------|--------|
| OPENAI_API_KEY | 大模型API密钥 | 必填 | ModelScope API Key |
| OPENAI_BASE_URL | 大模型API端点 | https://api.openai.com/v1 | https://api-inference.modelscope.cn/v1 |
| OPENAI_MODEL | 大模型名称 | gpt-3.5-turbo | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| MAX_PARALLEL_TASKS | 最大并行任务数 | 4 | 4-8 |
| MCP_SERVICE_TIMEOUT | MCP服务超时时间(秒) | 30 | 30-60 |
| TASK_RETRY_COUNT | 任务重试次数 | 3 | 2-3 |
| DATABASE_PATH | 数据库路径 | data/databases/mpeo.db | data/databases/mpeo.db |

### 智能体独立配置

支持为不同的智能体配置不同的模型和参数：

```json
{
  "planner": {
    "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "temperature": 0.3,
    "openai_config": {
      "api_key": "your-planner-api-key",
      "base_url": "https://api-inference.modelscope.cn/v1"
    }
  },
  "executor": {
    "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "temperature": 0.2,
    "openai_config": {
      "api_key": "your-executor-api-key",
      "base_url": "https://api-inference.modelscope.cn/v1"
    }
  }
}
```

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

## 🚀 快速开始

### 1. 基础配置

确保你的环境变量中有正确的API配置：

```bash
# 方式一：使用启动脚本（推荐）
./start.sh

# 方式二：手动设置
export OPENAI_API_KEY="your-modelscope-api-key-here"
export OPENAI_BASE_URL="https://api-inference.modelscope.cn/v1"
python main.py
```

### 2. 验证安装

运行系统自带的测试脚本来验证配置：

```bash
python test_fix.py
```

预期输出：
```
🧪 MPEO 系统配置测试
==================================================
🔗 测试 ModelScope API 连接...
✅ API 连接成功: 连接成功

🧠 测试任务规划功能...
✅ 任务规划成功，生成了 X 个任务:

🎉 所有测试通过！系统配置正确。
```

### 3. 启动系统

```bash
# 使用启动脚本
./start.sh

# 或者直接运行
python main.py
```

### 4. 简单测试

在交互界面中输入：

```
当前时间是什么？
```

系统将：
1. 分析你的需求
2. 生成任务图（时间查询任务）
3. 等待你确认
4. 执行时间查询
5. 返回结果

### 5. 复杂任务示例

```
请帮我分析当前最热门的三个AI编程工具，并获取每个工具的官方文档信息
```

这将演示系统的多任务协作能力。

### 6. 天气查询示例

```
杭州天气
```

系统将生成多个任务：
- 调用天气API获取实时数据
- 整合多个数据源
- 验证结果完整性
- 生成最终天气报告

## 🐛 故障排除

### 常见问题

1. **API认证错误 (401)**
   - 检查环境变量中的API密钥是否正确
   - 确认API密钥是否有效且未过期
   - 验证API端点URL是否正确
   - **解决方案**: 使用 `./start.sh` 脚本确保环境变量正确设置

2. **ModelScope API 认证失败**
   - 确认使用的是有效的 ModelScope API Token
   - 检查 `config/agent_models.json` 中的 API 密钥配置
   - 验证 API 密钥格式：`ms-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
   - **解决方案**: 运行 `python test_fix.py` 验证配置

3. **MCP服务连接失败**
   - 检查网络连接和服务可用性
   - 验证服务URL配置是否正确
   - 查看系统日志了解详细错误信息
   - **解决方案**: 检查 `config/mcp_services.json` 配置文件

4. **任务执行失败**
   - 查看系统日志了解详细错误信息
   - 检查MCP服务是否正常运行
   - 调整任务重试次数和超时时间
   - **解决方案**: 使用交互式命令 `logs` 查看详细错误信息

5. **数据库错误**
   - 确认数据库文件权限
   - 检查磁盘空间是否充足
   - 重新初始化数据库
   - **解决方案**: 删除 `data/databases/mpeo.db` 让系统重新创建

6. **环境变量冲突**
   - 系统环境变量可能覆盖 `.env` 文件配置
   - 不同 shell 会话可能使用不同的环境变量
   - **解决方案**: 使用 `./start.sh` 脚本或手动设置 `export` 命令

### 日志查看

```bash
# 在交互界面中查看系统日志
logs

# 查看特定会话日志
logs <session_id>
```

日志文件位置：`data/logs/YYYY-MM-DD.log`

### 环境检查

检查系统状态：

```bash
# 在交互界面中执行
status
```

## 📁 项目文件说明

### 新增文件

为了解决配置问题和简化使用流程，新增了以下文件：

#### `start.sh` - 启动脚本
```bash
#!/bin/bash
# MPEO 启动脚本 - 确保使用正确的环境配置
export OPENAI_API_KEY="your-modelscope-api-key-here"
export OPENAI_API_BASE="https://api-inference.modelscope.cn/v1"
python main.py "$@"
```

#### `test_fix.py` - 系统测试脚本
- 验证 ModelScope API 连接
- 测试任务规划功能
- 检查配置加载和数据库连接
- 提供详细的错误诊断信息

### 配置文件

#### `config/agent_models.json` - 智能体配置
- 配置规划、执行、输出三个智能体的模型参数
- 支持不同的 OpenAI 兼容 API
- 支持自定义模型名称和参数

#### `config/mcp_services.json` - MCP服务配置
- 配置预置的 MCP 服务
- 支持动态添加新的服务
- 包含时间、网页抓取、文档查询、天气等服务

### 数据目录

#### `data/databases/` - 数据库文件
- SQLite 数据库存储会话和任务数据
- 自动创建和初始化
- 支持数据持久化和查询

#### `data/logs/` - 日志文件
- 按日期组织的系统日志
- 详细的操作记录和错误追踪
- 支持日志查看和故障诊断

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📚 相关文档

- [CLAUDE.md](CLAUDE.md) - Claude Code 开发指导文档
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南

## 📞 支持

如果您遇到问题或有建议，请：

1. 查看本文档的故障排除部分
2. 检查项目的 Issues 页面
3. 创建新的 Issue 描述您的问题
4. 作者联系方式
   - 邮箱：[yl_zhangqiang@foxmail.com](mailto:yl_zhangqiang@foxmail.com)

