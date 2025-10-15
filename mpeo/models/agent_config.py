"""
智能体配置相关数据模型
"""

from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field


class OpenAIApiConfig(BaseModel):
    """OpenAI API配置"""
    api_key: Optional[str] = Field(default=None, description="OpenAI API密钥")
    base_url: Optional[str] = Field(default=None, description="OpenAI API基础URL")
    organization: Optional[str] = Field(default=None, description="OpenAI组织ID")
    timeout: Optional[int] = Field(default=60, ge=1, description="请求超时时间（秒）")
    max_retries: Optional[int] = Field(default=3, ge=0, description="最大重试次数")
    custom_headers: Optional[Dict[str, str]] = Field(default=None, description="自定义请求头")


class AgentModelConfig(BaseModel):
    """单个智能体的模型配置"""
    model_name: str = Field(..., description="模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="最大令牌数")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Top-p采样")
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    timeout: Optional[int] = Field(default=60, ge=1, description="请求超时时间（秒）")
    retry_count: Optional[int] = Field(default=3, ge=0, description="重试次数")

    # 智能体特定配置
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="响应格式配置")

    # OpenAI API配置
    openai_config: Optional[OpenAIApiConfig] = Field(default=None, description="OpenAI API配置")

    # 向后兼容的配置
    custom_headers: Optional[Dict[str, str]] = Field(default=None, description="自定义请求头（已废弃，使用openai_config.custom_headers）")


class MultiAgentConfig(BaseModel):
    """多智能体模型配置"""

    # 各个智能体的模型配置
    planner: AgentModelConfig = Field(..., description="规划智能体配置")
    executor: AgentModelConfig = Field(..., description="执行智能体配置")
    output: AgentModelConfig = Field(..., description="输出智能体配置")

    # 全局配置
    global_model: Optional[str] = Field(default=None, description="全局默认模型")
    global_timeout: int = Field(default=60, ge=1, description="全局超时时间")
    global_retry_count: int = Field(default=3, ge=0, description="全局重试次数")

    # 性能配置
    enable_model_caching: bool = Field(default=True, description="启用模型缓存")
    enable_parallel_execution: bool = Field(default=True, description="启用并行执行")
    max_concurrent_requests: int = Field(default=5, ge=1, description="最大并发请求数")

    # 成本控制
    daily_token_limit: Optional[int] = Field(default=None, ge=1, description="每日令牌限制")
    cost_alert_threshold: Optional[float] = Field(default=None, ge=0, description="成本告警阈值")

    # 模型选择策略
    model_selection_strategy: str = Field(default="quality", description="模型选择策略 (quality/speed/cost)")


class AgentPerformanceMetrics(BaseModel):
    """智能体性能指标"""
    agent_name: str = Field(..., description="智能体名称")
    model_name: str = Field(..., description="使用的模型")
    total_requests: int = Field(default=0, ge=0, description="总请求数")
    successful_requests: int = Field(default=0, ge=0, description="成功请求数")
    failed_requests: int = Field(default=0, ge=0, description="失败请求数")
    average_response_time: float = Field(default=0.0, ge=0, description="平均响应时间（秒）")
    total_tokens_used: int = Field(default=0, ge=0, description="总令牌消耗")
    total_cost: float = Field(default=0.0, ge=0, description="总成本")

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class ModelCostConfig(BaseModel):
    """模型成本配置"""
    model_name: str = Field(..., description="模型名称")
    input_token_cost: float = Field(..., ge=0, description="输入令牌单价")
    output_token_cost: float = Field(..., ge=0, description="输出令牌单价")
    currency: str = Field(default="USD", description="货币单位")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """计算成本"""
        return (input_tokens * self.input_token_cost + output_tokens * self.output_token_cost)


class AgentCapabilityConfig(BaseModel):
    """智能体能力配置"""
    agent_name: str = Field(..., description="智能体名称")
    supported_tasks: List[str] = Field(default_factory=list, description="支持的任务类型")
    preferred_models: List[str] = Field(default_factory=list, description="偏好模型列表")
    fallback_models: List[str] = Field(default_factory=list, description="备用模型列表")

    # 能力评分（0-100）
    reasoning_score: int = Field(default=50, ge=0, le=100, description="推理能力评分")
    creativity_score: int = Field(default=50, ge=0, le=100, description="创造能力评分")
    analysis_score: int = Field(default=50, ge=0, le=100, description="分析能力评分")
    coding_score: int = Field(default=50, ge=0, le=100, description="编程能力评分")

    def get_suitable_model(self, task_type: str, available_models: List[str]) -> str:
        """根据任务类型获取合适的模型"""
        # 检查偏好模型
        for model in self.preferred_models:
            if model in available_models:
                return model

        # 检查备用模型
        for model in self.fallback_models:
            if model in available_models:
                return model

        # 返回第一个可用模型
        return available_models[0] if available_models else "gpt-3.5-turbo"