# 智能体模型配置指南

本文档介绍如何为MPEO系统中的不同智能体配置独立的大模型。

## 概述

MPEO系统支持为以下智能体配置独立的模型：
- **规划智能体 (Planner)** - 负责任务分解和DAG生成
- **执行智能体 (Executor)** - 负责任务执行和MCP服务调用
- **输出智能体 (Output)** - 负责结果整合和最终答案生成

## 配置文件

### 主要配置文件

1. **`config/agent_models.json`** - 主要的智能体模型配置文件
2. **`config/agent_models.example.json`** - 配置模板文件
3. **`config/agent_models.presets.json`** - 预设配置文件

### 配置文件结构

### 完整配置示例

```json
{
  "planner": {
    "model_name": "gpt-4",
    "temperature": 0.3,
    "max_tokens": 2000,
    "timeout": 60,
    "retry_count": 3,
    "system_prompt": "你是一个专业的任务规划专家...",
    "openai_config": {
      "api_key": "sk-your-planner-api-key",
      "base_url": "https://api.openai.com/v1",
      "organization": "org-your-planner-org",
      "timeout": 60,
      "max_retries": 3,
      "custom_headers": {
        "User-Agent": "MPEO-Planner/1.0"
      }
    }
  },
  "executor": {
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.2,
    "max_tokens": 1500,
    "timeout": 30,
    "retry_count": 2,
    "system_prompt": "你是一个高效的任务执行专家...",
    "openai_config": {
      "api_key": "sk-your-executor-api-key",
      "base_url": "https://api.openai.com/v1",
      "organization": "org-your-executor-org",
      "timeout": 30,
      "max_retries": 2
    }
  },
  "output": {
    "model_name": "gpt-4",
    "temperature": 0.4,
    "max_tokens": 2500,
    "timeout": 45,
    "retry_count": 2,
    "system_prompt": "你是一个专业的内容整合专家...",
    "openai_config": {
      "api_key": "sk-your-output-api-key",
      "base_url": "https://api.openai.com/v1",
      "organization": "org-your-output-org",
      "timeout": 45,
      "max_retries": 2
    }
  },
  "global_openai_config": {
    "api_key": "sk-your-global-api-key",
    "base_url": "https://api.openai.com/v1",
    "organization": "org-your-global-org",
    "timeout": 60,
    "max_retries": 3
  },
  "global_model": "gpt-3.5-turbo",
  "global_timeout": 60,
  "global_retry_count": 3,
  "enable_model_caching": true,
  "model_selection_strategy": "quality"
}
```

### 最简配置示例

```json
{
  "planner": {
    "model_name": "gpt-4",
    "temperature": 0.3
  },
  "executor": {
    "model_name": "gpt-3.5-turbo",
    "temperature": 0.2
  },
  "output": {
    "model_name": "gpt-4",
    "temperature": 0.4
  }
}
```

## OpenAI API配置

### 配置方式

MPEO支持多种配置OpenAI API的方式：

#### 1. 环境变量配置（推荐）

```bash
# 全局配置（所有智能体共享）
export OPENAI_API_KEY=sk-your-openai-api-key
export OPENAI_API_BASE=https://api.openai.com/v1
export OPENAI_ORGANIZATION=org-your-organization-id

# 智能体单独配置（可选）
export OPENAI_API_KEY_PLANNER=sk-your-planner-key
export OPENAI_API_BASE_PLANNER=https://api.openai.com/v1
export OPENAI_ORGANIZATION_PLANNER=org-your-planner-org
```

#### 2. 配置文件配置

```json
{
  "planner": {
    "model_name": "gpt-4",
    "openai_config": {
      "api_key": "sk-your-planner-key",
      "base_url": "https://api.openai.com/v1",
      "organization": "org-your-planner-org"
    }
  }
}
```

#### 3. 全局配置配置

```json
{
  "global_openai_config": {
    "api_key": "sk-your-global-key",
    "base_url": "https://api.openai.com/v1",
    "organization": "org-your-global-org"
  }
}
```

### 配置优先级

配置优先级从高到低：
1. **智能体单独的环境变量** (如 `OPENAI_API_KEY_PLANNER`)
2. **智能体单独的配置文件** (`planner.openai_config`)
3. **全局环境变量** (`OPENAI_API_KEY`)
4. **全局配置文件** (`global_openai_config`)

### 不同API地址配置

#### OpenAI官方API
```json
{
  "openai_config": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-..."
  }
}
```

#### Azure OpenAI
```json
{
  "openai_config": {
    "base_url": "https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2024-02-15-preview",
    "api_key": "your-azure-api-key",
    "custom_headers": {
      "api-key": "your-azure-api-key"
    }
  }
}
```

#### 自定义代理或兼容API
```json
{
  "openai_config": {
    "base_url": "https://your-proxy-domain.com/v1",
    "api_key": "your-proxy-key",
    "custom_headers": {
      "Authorization": "Bearer your-token",
      "X-Custom-Header": "custom-value"
    }
  }
}
```

## 使用方法

### 1. 基本使用

```bash
# 复制配置模板
cp config/agent_models.example.json config/agent_models.json

# 复制环境变量模板
cp .env.example .env

# 编辑配置文件和环境变量
vim config/agent_models.json
vim .env

# 运行系统（会自动加载配置）
python main.py
```

### 2. 使用预设配置

```bash
# 查看可用预设
python -c "
from mpeo.services import get_config_loader
loader = get_config_loader()
# 注意：预设功能将在未来版本中实现
print('预设功能即将推出')
"

# 在代码中直接使用配置加载器
from mpeo.services import get_config_loader
from mpeo.core.coordinator import SystemCoordinator

# 加载智能体配置
config_loader = get_config_loader()
agent_config = config_loader.load_agent_config()
coordinator = SystemCoordinator(agent_config=agent_config)
```

### 3. 编程方式配置

```python
from mpeo.models.agent_config import MultiAgentConfig, AgentModelConfig
from mpeo.core.coordinator import SystemCoordinator

# 创建自定义配置
agent_config = MultiAgentConfig(
    planner=AgentModelConfig(
        model_name="gpt-4-turbo",
        temperature=0.2,
        max_tokens=3000
    ),
    executor=AgentModelConfig(
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=2000
    ),
    output=AgentModelConfig(
        model_name="gpt-4",
        temperature=0.3,
        max_tokens=4000
    )
)

# 使用自定义配置
coordinator = SystemCoordinator(agent_config=agent_config)
```

## 预设配置

系统提供以下预设配置：

| 预设名称 | 描述 | 适用场景 |
|---------|------|----------|
| `high_quality` | 高质量配置 | 复杂任务，追求最佳结果 |
| `balanced` | 平衡配置 | 性能和质量的最佳平衡 |
| `speed_optimized` | 速度优化配置 | 简单任务，快速响应 |
| `cost_optimized` | 成本优化配置 | 预算有限的情况 |
| `creative_tasks` | 创意任务配置 | 需要创造力的任务 |
| `analytical_tasks` | 分析任务配置 | 数据分析和逻辑推理 |

## 配置参数说明

### AgentModelConfig 参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `model_name` | str | 模型名称 | 必填 |
| `temperature` | float | 温度参数 (0-2) | 0.7 |
| `max_tokens` | int | 最大令牌数 | None |
| `top_p` | float | Top-p采样 (0-1) | 1.0 |
| `timeout` | int | 请求超时时间(秒) | 60 |
| `retry_count` | int | 重试次数 | 3 |
| `system_prompt` | str | 系统提示词 | None |
| `openai_config` | OpenAIApiConfig | OpenAI API配置 | None |

### OpenAIApiConfig 参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `api_key` | str | OpenAI API密钥 | None (从环境变量读取) |
| `base_url` | str | OpenAI API基础URL | None (从环境变量读取) |
| `organization` | str | OpenAI组织ID | None (从环境变量读取) |
| `timeout` | int | 请求超时时间(秒) | 60 |
| `max_retries` | int | 最大重试次数 | 3 |
| `custom_headers` | dict | 自定义请求头 | None |

### MultiAgentConfig 参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `planner` | AgentModelConfig | 规划智能体配置 | 必填 |
| `executor` | AgentModelConfig | 执行智能体配置 | 必填 |
| `output` | AgentModelConfig | 输出智能体配置 | 必填 |
| `enable_model_caching` | bool | 启用模型缓存 | true |
| `model_selection_strategy` | str | 模型选择策略 | "balanced" |

## 模型选择建议

### 规划智能体 (Planner)
- **推荐模型**: GPT-4, GPT-4 Turbo
- **温度**: 0.2-0.4 (较低的随机性，确保结构化输出)
- **特点**: 需要强大的逻辑推理和结构化思维能力

### 执行智能体 (Executor)
- **推荐模型**: GPT-3.5 Turbo, GPT-4
- **温度**: 0.1-0.3 (低随机性，确保准确性)
- **特点**: 需要精确执行，避免创造性偏差

### 输出智能体 (Output)
- **推荐模型**: GPT-4, GPT-4 Turbo
- **温度**: 0.3-0.5 (适中的创造性，生成流畅文本)
- **特点**: 需要良好的语言组织和总结能力

## 性能优化

### 1. 成本控制
```json
{
  "daily_token_limit": 100000,
  "cost_alert_threshold": 10.0,
  "model_selection_strategy": "cost"
}
```

### 2. 速度优化
```json
{
  "enable_parallel_execution": true,
  "max_concurrent_requests": 5,
  "enable_model_caching": true
}
```

### 3. 质量优化
```json
{
  "model_selection_strategy": "quality",
  "global_timeout": 90,
  "global_retry_count": 3
}
```

## 故障排除

### 常见问题

1. **配置文件加载失败**
   - 检查JSON格式是否正确
   - 确认文件路径存在
   - 验证必需字段是否完整

2. **模型调用失败**
   - 检查API密钥是否正确
   - 验证模型名称是否支持
   - 确认网络连接正常

3. **性能问题**
   - 调整超时时间
   - 减少max_tokens设置
   - 启用模型缓存

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看详细的模型调用信息
coordinator = SystemCoordinator()
```

## 最佳实践

1. **任务复杂度匹配**: 根据任务复杂度选择合适的模型配置
2. **成本监控**: 设置合理的令牌限制和成本告警
3. **性能测试**: 在生产环境前充分测试配置
4. **配置备份**: 保存配置文件的备份版本
5. **渐进式优化**: 从预设配置开始，逐步调优

## 更新配置

系统支持热更新配置：

```python
# 更新智能体配置
coordinator.agent_config = new_agent_config

# 重新初始化组件
coordinator.planner = PlannerModel(
    coordinator.openai_client,
    coordinator.database,
    coordinator.agent_config.planner
)
```