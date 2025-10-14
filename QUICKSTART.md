# 🚀 快速开始指南

## 1. 环境准备

确保您已安装 Python 3.11 或更高版本：

```bash
python --version
```

## 2. 安装系统

```bash
# 进入项目目录
cd MPEO

# 安装依赖
pip install -e .

# 或手动安装依赖
pip install openai aiohttp pydantic rich networkx typing-extensions python-dotenv
```

## 3. 配置 OpenAI API

```bash
# 复制环境配置文件
cp .env.example .env

# 编辑 .env 文件，添加您的 OpenAI API Key
# OPENAI_API_KEY=your_openai_api_key_here
```

## 4. 验证安装

```bash
# 运行测试脚本
python test_system.py
```

如果所有测试通过，您应该看到：
```
🎉 All tests passed! System is ready to use.
```

## 5. 启动系统

```bash
# 使用默认配置启动
python main.py

# 或使用自定义配置
python main.py --max-parallel 8 --model gpt-4 --timeout 60
```

## 6. 首次使用

启动后，您将看到交互式界面。输入您的第一个问题：

```
请帮我分析一下当前的市场趋势
```

系统将：
1. 分析您的需求
2. 生成任务图
3. 等待您的确认
4. 执行任务
5. 返回整合后的结果

## 7. 常用命令

在系统运行时，您可以使用以下命令：

- `help` - 显示帮助
- `status` - 查看系统状态
- `history` - 查看历史记录
- `quit` - 退出系统

## 8. 示例查询

### 简单查询
```
今天天气怎么样？
```

### 数据分析
```
请分析最近一个月的销售数据，生成趋势报告
```

### 复杂任务
```
请帮我做一个完整的市场调研，包括：
1. 竞争对手分析
2. 市场规模评估  
3. 用户需求调研
4. 风险评估
请以报告形式输出
```

## 🔧 故障排除

### 问题：OpenAI API 错误
**解决方案**：检查 `.env` 文件中的 `OPENAI_API_KEY` 是否正确设置

### 问题：导入错误
**解决方案**：确保所有依赖已正确安装，运行 `pip install -e .`

### 问题：数据库错误
**解决方案**：检查数据库文件权限，或删除现有数据库文件重新初始化

## 📚 更多信息

- 完整文档：查看 `README.md`
- 系统架构：了解各组件如何协同工作
- MCP集成：如何集成外部服务

---

🎉 **恭喜！** 您已成功设置多模型协作任务处理系统。开始探索AI协作的无限可能吧！