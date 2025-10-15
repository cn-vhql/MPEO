"""
会话相关数据模型
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from .task import TaskGraph, ExecutionResults


class TaskSession(BaseModel):
    """Task processing session"""
    session_id: str = Field(..., description="Unique session identifier")
    user_query: str = Field(..., description="Original user query")
    task_graph: Optional[TaskGraph] = Field(default=None, description="Generated task graph")
    execution_results: Optional[ExecutionResults] = Field(default=None, description="Execution results")
    final_output: Optional[str] = Field(default=None, description="Final output")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="created", description="Session status")