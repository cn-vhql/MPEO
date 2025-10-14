"""
Data models for the multi-model collaborative task processing system
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class TaskType(str, Enum):
    """Task type enumeration"""
    LOCAL_COMPUTE = "本地计算"
    MCP_CALL = "mcp调用"
    DATA_PROCESSING = "数据处理"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "待执行"
    RUNNING = "执行中"
    SUCCESS = "成功"
    FAILED = "失败"


class DependencyType(str, Enum):
    """Dependency type enumeration"""
    DATA_DEPENDENCY = "数据依赖"
    RESULT_DEPENDENCY = "结果依赖"


class TaskNode(BaseModel):
    """Task node in the DAG"""
    task_id: str = Field(..., description="Unique task identifier")
    task_desc: str = Field(..., description="Task description")
    task_type: TaskType = Field(..., description="Task type")
    expected_output: str = Field(..., description="Expected output format/content")
    priority: int = Field(default=3, ge=1, le=5, description="Priority (1-5)")


class TaskEdge(BaseModel):
    """Task dependency edge in the DAG"""
    from_task_id: str = Field(..., description="Source task ID")
    to_task_id: str = Field(..., description="Target task ID")
    dependency_type: DependencyType = Field(..., description="Dependency type")


class TaskGraph(BaseModel):
    """Directed Acyclic Graph (DAG) for tasks"""
    nodes: List[TaskNode] = Field(default_factory=list)
    edges: List[TaskEdge] = Field(default_factory=list)

    @validator('nodes')
    def validate_unique_task_ids(cls, v):
        task_ids = [node.task_id for node in v]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task IDs must be unique")
        return v

    @validator('edges')
    def validate_edge_references(cls, v, values):
        if 'nodes' not in values:
            return v
        
        node_ids = {node.task_id for node in values['nodes']}
        for edge in v:
            if edge.from_task_id not in node_ids:
                raise ValueError(f"Source task ID '{edge.from_task_id}' not found in nodes")
            if edge.to_task_id not in node_ids:
                raise ValueError(f"Target task ID '{edge.to_task_id}' not found in nodes")
        return v

    def has_cycle(self) -> bool:
        """Check if the graph has cycles using DFS"""
        try:
            import networkx as nx
            G = nx.DiGraph()
            G.add_nodes_from([node.task_id for node in self.nodes])
            G.add_edges_from([(edge.from_task_id, edge.to_task_id) for edge in self.edges])
            return not nx.is_directed_acyclic_graph(G)
        except ImportError:
            # Fallback to simple cycle detection without networkx
            return self._simple_cycle_detection()

    def _simple_cycle_detection(self) -> bool:
        """Simple cycle detection without external dependencies"""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Find all outgoing edges from this node
            for edge in self.edges:
                if edge.from_task_id == node_id:
                    if edge.to_task_id not in visited:
                        if dfs(edge.to_task_id):
                            return True
                    elif edge.to_task_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in self.nodes:
            if node.task_id not in visited:
                if dfs(node.task_id):
                    return True
        return False


class ExecutionResult(BaseModel):
    """Single task execution result"""
    task_id: str = Field(..., description="Task ID from DAG")
    status: TaskStatus = Field(..., description="Execution status")
    output: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Task output")
    execution_time: float = Field(..., description="Execution time in seconds")
    error_msg: Optional[str] = Field(default=None, description="Error message if failed")


class ExecutionResults(BaseModel):
    """Collection of all task execution results"""
    execution_results: List[ExecutionResult] = Field(default_factory=list)
    total_execution_time: float = Field(default=0.0, description="Total execution time")
    success_count: int = Field(default=0, description="Number of successful tasks")
    failed_count: int = Field(default=0, description="Number of failed tasks")

    def add_result(self, result: ExecutionResult):
        """Add an execution result and update statistics"""
        self.execution_results.append(result)
        if result.status == TaskStatus.SUCCESS:
            self.success_count += 1
        elif result.status == TaskStatus.FAILED:
            self.failed_count += 1


class SystemConfig(BaseModel):
    """System configuration"""
    max_parallel_tasks: int = Field(default=4, ge=1, description="Maximum parallel tasks")
    mcp_service_timeout: int = Field(default=30, ge=1, description="MCP service timeout in seconds")
    task_retry_count: int = Field(default=3, ge=0, description="Task retry count")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    database_path: str = Field(default="mpeo.db", description="SQLite database path")


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


class MCPServiceConfig(BaseModel):
    """MCP service configuration"""
    service_name: str = Field(..., description="Service name")
    endpoint_url: str = Field(..., description="Service endpoint URL")
    timeout: int = Field(default=30, description="Request timeout")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Request headers")
    auth_config: Optional[Dict[str, Any]] = Field(default=None, description="Authentication config")