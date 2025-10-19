#!/bin/bash

# MPEO 启动脚本 - 确保使用正确的环境配置
# 此脚本会设置必要的环境变量并启动系统

echo "🚀 启动 MPEO 系统..."

# 设置 ModelScope API 配置（覆盖系统环境变量）
export OPENAI_API_KEY="your-modelscope-api-key-here"
export OPENAI_API_BASE="https://api-inference.modelscope.cn/v1"

echo "✅ 环境变量已配置"
echo "   API Key: ${OPENAI_API_KEY:0:20}..."
echo "   Base URL: $OPENAI_API_BASE"

# 启动主程序
echo "📝 启动主程序..."
python main.py "$@"