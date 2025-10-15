# MPEO 项目重构计划

## 🎯 目标
重新组织项目文件结构，使其更符合Python项目最佳实践，提高可维护性和可扩展性。

## 📁 新的目录结构

```
mpeo/
├── README.md                    # 主要项目文档
├── QUICKSTART.md                # 快速开始指南
├── CLAUDE.md                    # Claude Code 开发指南
├── pyproject.toml              # 项目配置
├── .env.example                # 环境变量模板
├── .gitignore                  # Git忽略文件
├── .python-version             # Python版本
├── uv.lock                     # 依赖锁定文件
│
├── mpeo/                       # 主包目录
│   ├── __init__.py            # 包初始化
│   │
│   ├── core/                  # 核心模块
│   │   ├── __init__.py
│   │   ├── coordinator.py     # 系统协调器
│   │   ├── planner.py         # 规划模型
│   │   ├── executor.py        # 执行模型
│   │   └── output.py          # 输出模型
│   │
│   ├── models/                # 数据模型
│   │   ├── __init__.py
│   │   ├── task.py           # 任务相关模型
│   │   ├── session.py        # 会话相关模型
│   │   └── config.py         # 配置相关模型
│   │
│   ├── services/              # 服务层
│   │   ├── __init__.py
│   │   ├── database.py        # 数据库服务
│   │   ├── mcp_client.py      # MCP客户端
│   │   └── openai_client.py   # OpenAI客户端
│   │
│   ├── interfaces/            # 用户界面
│   │   ├── __init__.py
│   │   ├── cli.py            # 命令行界面
│   │   └── interactive.py    # 交互式界面
│   │
│   └── utils/                 # 工具模块
│       ├── __init__.py
│       ├── logging.py        # 日志工具
│       ├── config.py         # 配置工具
│       └── exceptions.py     # 自定义异常
│
├── config/                    # 配置文件目录
│   ├── mcp_services.json     # MCP服务配置
│   ├── default.env           # 默认环境配置
│   └── logging.yaml          # 日志配置
│
├── data/                      # 数据目录
│   ├── databases/            # 数据库文件
│   └── logs/                 # 日志文件
│
├── tests/                     # 测试目录
│   ├── __init__.py
│   ├── unit/                 # 单元测试
│   ├── integration/          # 集成测试
│   └── fixtures/             # 测试数据
│
├── scripts/                   # 脚本目录
│   ├── setup_mcp.py         # MCP设置脚本
│   ├── test_mcp.py          # MCP测试脚本
│   └── migrate_db.py        # 数据库迁移脚本
│
├── docs/                      # 文档目录
│   ├── api/                  # API文档
│   ├── user_guide/           # 用户指南
│   └── development/          # 开发文档
│
└── examples/                  # 示例目录
    ├── basic_usage.py        # 基础使用示例
    └── advanced/             # 高级示例
```

## 🔄 迁移步骤

1. **创建新目录结构**
2. **移动核心文件到对应模块**
   - `models.py` → `models/task.py`, `models/session.py`, `models/config.py`
   - `coordinator.py` → `core/coordinator.py`
   - `planner.py` → `core/planner.py`
   - `executor.py` → `core/executor.py`
   - `output.py` → `core/output.py`
   - `database.py` → `services/database.py`
   - `interface.py` → `interfaces/cli.py`
3. **更新导入路径**
4. **创建工具模块**
5. **移动配置和数据文件**
6. **重组测试和示例文件**
7. **更新文档**

## 📋 文件拆分详情

### models.py 拆分为:
- `models/task.py` - TaskNode, TaskEdge, TaskGraph, ExecutionResult 等
- `models/session.py` - TaskSession, UserQuery 等
- `models/config.py` - SystemConfig, MCPServiceConfig 等

### 新增工具模块:
- `utils/logging.py` - 日志配置和工具函数
- `utils/config.py` - 配置加载和验证
- `utils/exceptions.py` - 自定义异常类

### 服务层重组:
- `services/mcp_client.py` - MCP客户端相关逻辑从executor中提取
- `services/openai_client.py` - OpenAI客户端相关逻辑

## 🎯 预期收益

1. **更好的代码组织** - 按功能模块清晰分类
2. **更高的可维护性** - 单一职责，易于修改
3. **更强的可扩展性** - 新功能容易添加
4. **更好的测试覆盖** - 模块化便于单元测试
5. **更清晰的依赖关系** - 减少循环依赖
6. **更好的开发体验** - 文件结构符合直觉