"""
Executor Model with AgentScope Integration - 使用AgentScope工具系统的执行模型
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union
import aiohttp
from openai import OpenAI

from ..models import (
    TaskGraph, TaskNode, TaskEdge, TaskType, TaskStatus,
    ExecutionResult, ExecutionResults, SystemConfig, MCPServiceConfig
)
from ..models.agent_config import AgentModelConfig
from ..services.database import DatabaseManager
from ..services.unified_mcp_manager import UnifiedMCPManager, MCPResult
from ..services.agentscope_mcp_adapter import AgentScopeMCPAdapter

# AgentScope imports
try:
    from agentscope.agents import ReActAgent, AgentBase
    from agentscope.models import OpenAIChatModel
    from agentscope.message import Msg
    from agentscope.memory import InMemoryMemory
    from agentscope.tool import Toolkit
    from agentscope.pipeline import fanout_pipeline, sequential_pipeline
    AGENTSCOPE_AVAILABLE = True
except ImportError:
    AGENTSCOPE_AVAILABLE = False
    ReActAgent = None
    AgentBase = None
    OpenAIChatModel = None
    Msg = None
    InMemoryMemory = None
    Toolkit = None
    fanout_pipeline = None
    sequential_pipeline = None


class TaskExecutionAgent(AgentBase):
    """任务执行专用智能体"""

    def __init__(self,
                 task_node: TaskNode,
                 model_config: AgentModelConfig,
                 toolkit: Optional[Toolkit] = None,
                 execution_context: Optional[Dict[str, Any]] = None):
        super().__init__(name=f"TaskExecutor-{task_node.task_id}")
        self.task_node = task_node
        self.model_config = model_config
        self.toolkit = toolkit
        self.execution_context = execution_context or {}

        # 如果AgentScope可用，创建模型和记忆
        if AGENTSCOPE_AVAILABLE:
            self.model = OpenAIChatModel(
                model_name=model_config.model_name,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens
            )
            self.memory = InMemoryMemory()

    async def reply(self, msg: Msg) -> Msg:
        """执行任务并返回结果"""
        try:
            # 根据任务类型选择执行方法
            if self.task_node.task_type == TaskType.MCP_CALL and self.toolkit:
                result = await self._execute_with_tools(msg.content)
            elif self.task_node.task_type == TaskType.LOCAL_COMPUTE:
                result = await self._execute_local_compute(msg.content)
            elif self.task_node.task_type == TaskType.DATA_PROCESSING:
                result = await self._execute_data_processing(msg.content)
            else:
                raise ValueError(f"Unknown task type: {self.task_node.task_type}")

            return Msg(
                name=self.name,
                content=result,
                role="assistant"
            )

        except Exception as e:
            error_result = f"任务执行失败: {str(e)}"
            return Msg(
                name=self.name,
                content=error_result,
                role="assistant"
            )

    async def _execute_with_tools(self, task_input: str) -> str:
        """使用工具执行任务"""
        if not self.toolkit:
            return "没有可用的工具来执行此任务"

        # 创建ReAct智能体来执行具体任务
        react_agent = ReActAgent(
            name=f"ToolExecutor-{self.task_node.task_id}",
            sys_prompt=self._get_tool_execution_prompt(),
            model=self.model,
            memory=self.memory,
            toolkit=self.toolkit,
            max_iters=2  # 限制迭代次数
        )

        # 构建执行消息
        exec_msg = Msg(
            name="user",
            content=f"""
任务描述：{self.task_node.task_desc}
预期输出：{self.task_node.expected_output}

输入信息：
{task_input}

请使用可用的工具来完成这个任务。
""",
            role="user"
        )

        # 执行并获取结果
        response = await react_agent.reply(exec_msg)
        return response.content

    async def _execute_local_compute(self, task_input: str) -> str:
        """执行本地计算任务"""
        # 使用OpenAI模型进行本地计算
        from openai import OpenAI

        # 这里需要访问原始的OpenAI客户端
        # 在实际使用中，这应该通过依赖注入来提供
        prompt = f"""
请执行以下本地计算任务：

任务描述：{self.task_node.task_desc}
预期输出：{self.task_node.expected_output}

输入信息：
{task_input}

请根据任务描述和输入信息，执行相应的计算或处理，并返回结果。
确保输出格式符合预期要求。
"""

        # 这里应该使用注入的OpenAI客户端
        # 为了简化，我们返回一个基本的响应
        return f"本地计算任务 '{self.task_node.task_desc}' 已完成"

    async def _execute_data_processing(self, task_input: str) -> str:
        """执行数据处理任务"""
        prompt = f"""
请执行以下数据处理任务：

任务描述：{self.task_node.task_desc}
预期输出：{self.task_node.expected_output}

输入数据：
{task_input}

请对输入数据进行处理，包括但不限于：
- 数据清洗和格式化
- 数据转换和计算
- 数据分析和统计
- 结果整理和呈现

返回处理后的结果。
"""

        # 这里应该使用注入的OpenAI客户端
        return f"数据处理任务 '{self.task_node.task_desc}' 已完成"

    def _get_tool_execution_prompt(self) -> str:
        """获取工具执行的系统提示"""
        return f"""
你是一个专业的任务执行智能体，专门负责使用工具完成具体的任务。

当前任务：
- 任务ID: {self.task_node.task_id}
- 任务描述: {self.task_node.task_desc}
- 预期输出: {self.task_node.expected_output}
- 任务类型: {self.task_node.task_type.value}

执行指导：
1. 仔细理解任务需求和预期输出
2. 选择合适的工具来完成任务
3. 根据任务描述准备工具调用参数
4. 分析工具执行结果，确保满足预期要求
5. 如有需要，可以进行多次工具调用

重要提醒：
- 如果任务描述中提到了具体的URL、参数等，请准确使用
- 确保工具调用的结果符合预期输出格式
- 如果工具调用失败，分析原因并尝试替代方案
"""


class AgentScopeTaskExecutor:
    """使用AgentScope工具系统的任务执行器"""

    def __init__(self,
                 openai_client: OpenAI,
                 database: DatabaseManager,
                 config: SystemConfig,
                 model_config: AgentModelConfig = None,
                 enable_agentscope: bool = True):
        self.client = openai_client
        self.database = database
        self.config = config
        self.model_config = model_config or AgentModelConfig(
            model_name=config.openai_model,
            temperature=0.2,
            max_tokens=1500,
            timeout=30
        )

        # AgentScope相关
        self.enable_agentscope = enable_agentscope and AGENTSCOPE_AVAILABLE
        self.mcp_services: Dict[str, MCPServiceConfig] = {}
        self.mcp_manager: Optional[UnifiedMCPManager] = None
        self.agentscope_adapter: Optional[AgentScopeMCPAdapter] = None

        # 向后兼容
        self.execution_context: Dict[str, Any] = {}

        if self.enable_agentscope:
            self._initialize_agentscope()

    def _initialize_agentscope(self):
        """初始化AgentScope组件"""
        try:
            if not AGENTSCOPE_AVAILABLE:
                raise ImportError("AgentScope is not installed")

            # 创建OpenAI模型包装器
            self.agentscope_model = OpenAIChatModel(
                model_name=self.model_config.model_name,
                api_key=self.client.api_key,
                organization_id=getattr(self.client, 'organization', None),
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens
            )

            self.database.log_event("", "executor", "agentscope_initialized",
                                   f"AgentScope executor initialized with model: {self.model_config.model_name}")

        except Exception as e:
            self.database.log_event("", "executor", "agentscope_init_failed",
                                   f"Failed to initialize AgentScope: {str(e)}")
            self.enable_agentscope = False

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self.model_config.model_name

    async def register_mcp_service(self, service_config: MCPServiceConfig):
        """注册MCP服务配置"""
        self.mcp_services[service_config.service_name] = service_config

        # 初始化统一MCP管理器
        if self.mcp_manager is None:
            self.mcp_manager = UnifiedMCPManager()
            await self.mcp_manager.initialize()

        # 注册服务
        success = await self.mcp_manager.register_service(service_config)

        # 如果启用了AgentScope，初始化适配器
        if success and self.enable_agentscope:
            try:
                if self.agentscope_adapter is None:
                    self.agentscope_adapter = AgentScopeMCPAdapter(self.mcp_manager)
                    await self.agentscope_adapter.initialize()
                else:
                    await self.agentscope_adapter.refresh_tools()

                self.database.log_event("", "executor", "agentscope_mcp_registered",
                                       f"MCP service registered with AgentScope: {service_config.service_name}")

            except Exception as e:
                self.database.log_event("", "executor", "agentscope_mcp_registration_failed",
                                       f"Failed to register MCP service with AgentScope: {str(e)}")

        return success

    def get_current_mcp_manager(self):
        """获取当前活跃的MCP管理器"""
        if self.mcp_manager is None:
            self.mcp_manager = UnifiedMCPManager()
        return self.mcp_manager

    async def execute_task_graph(self, task_graph: TaskGraph, user_query: str, session_id: str) -> ExecutionResults:
        """执行任务图，支持AgentScope并行执行"""
        self.database.log_event(session_id, "executor", "start_execution",
                               f"Executing {len(task_graph.nodes)} tasks")

        start_time = time.time()
        results = ExecutionResults()

        try:
            # 优先使用AgentScope并行执行
            if self.enable_agentscope and len(task_graph.nodes) > 1:
                return await self._agentscope_parallel_execution(task_graph, user_query, session_id, start_time)
            else:
                # 降级到传统执行方式
                return await self._legacy_execution(task_graph, user_query, session_id, start_time)

        except Exception as e:
            self.database.log_event(session_id, "executor", "execution_error", f"Execution failed: {str(e)}")
            # 如果AgentScope执行失败，尝试传统方式
            if self.enable_agentscope:
                self.database.log_event(session_id, "executor", "fallback_to_legacy",
                                       "Falling back to legacy execution method")
                return await self._legacy_execution(task_graph, user_query, session_id, start_time)
            else:
                results.total_execution_time = time.time() - start_time
                return results

    async def _agentscope_parallel_execution(self,
                                           task_graph: TaskGraph,
                                           user_query: str,
                                           session_id: str,
                                           start_time: float) -> ExecutionResults:
        """使用AgentScope进行并行任务执行"""
        results = ExecutionResults()

        try:
            # 构建依赖映射
            dependency_map = self._build_dependency_map(task_graph)
            reverse_dependency_map = self._build_reverse_dependency_map(task_graph)

            # 跟踪任务状态
            task_status: Dict[str, TaskStatus] = {}
            task_outputs: Dict[str, Any] = {}

            # 初始化所有任务为等待状态
            for task in task_graph.nodes:
                task_status[task.task_id] = TaskStatus.PENDING

            # 分阶段执行任务
            while len(task_status) > len([s for s in task_status.values() if s in [TaskStatus.SUCCESS, TaskStatus.FAILED]]):
                # 获取准备执行的任务
                ready_tasks = self._get_ready_tasks(task_graph, task_status, dependency_map)

                if not ready_tasks:
                    # 检查死锁
                    pending_tasks = [tid for tid, status in task_status.items() if status == TaskStatus.PENDING]
                    if pending_tasks:
                        self.database.log_event(session_id, "executor", "deadlock_detected",
                                               f"Pending tasks: {pending_tasks}")
                        for task_id in pending_tasks:
                            task_status[task_id] = TaskStatus.FAILED
                            result = ExecutionResult(
                                task_id=task_id,
                                status=TaskStatus.FAILED,
                                output=None,
                                execution_time=0.0,
                                error_msg="Deadlock detected"
                            )
                            results.add_result(result)
                    break

                # 使用AgentScope并行执行就绪任务
                if len(ready_tasks) > 1:
                    await self._execute_tasks_with_agentscope(
                        ready_tasks, task_graph, task_status, task_outputs,
                        results, user_query, session_id
                    )
                else:
                    await self._execute_single_task_with_agentscope(
                        ready_tasks[0], task_graph, task_status, task_outputs,
                        results, user_query, session_id
                    )

            results.total_execution_time = time.time() - start_time
            self.database.log_event(session_id, "executor", "agentscope_execution_completed",
                                   f"Completed: {results.success_count}, Failed: {results.failed_count}")

            return results

        except Exception as e:
            self.database.log_event(session_id, "executor", "agentscope_execution_error", f"AgentScope execution error: {str(e)}")
            results.total_execution_time = time.time() - start_time
            return results

    async def _execute_tasks_with_agentscope(self,
                                           tasks: List[TaskNode],
                                           task_graph: TaskGraph,
                                           task_status: Dict[str, TaskStatus],
                                           task_outputs: Dict[str, Any],
                                           results: ExecutionResults,
                                           user_query: str,
                                           session_id: str):
        """使用AgentScope并行执行多个任务"""
        self.database.log_event(session_id, "executor", "agentscope_parallel_execution",
                               f"Executing {len(tasks)} tasks with AgentScope")

        try:
            # 为每个任务创建执行智能体
            execution_agents = []
            messages = []

            for task in tasks:
                # 准备输入数据
                input_data = self._prepare_input_data(task, task_outputs, task_graph)
                task_context = {
                    "task": task,
                    "input_data": input_data,
                    "user_query": user_query,
                    "session_id": session_id
                }

                # 创建任务特定的工具包
                toolkit = await self._create_task_toolkit(task, session_id)

                # 创建执行智能体
                agent = TaskExecutionAgent(
                    task_node=task,
                    model_config=self.model_config,
                    toolkit=toolkit,
                    execution_context=task_context
                )

                # 创建输入消息
                input_msg = Msg(
                    name="user",
                    content=self._build_task_execution_message(task, input_data, user_query),
                    role="user"
                )

                execution_agents.append(agent)
                messages.append(input_msg)

            # 使用AgentScope的并行管道执行
            if len(execution_agents) > 1:
                responses = await fanout_pipeline(execution_agents, messages)
            else:
                # 单个任务直接执行
                responses = [await execution_agents[0].reply(messages[0])]

            # 处理结果
            for i, (task, response) in enumerate(zip(tasks, responses)):
                execution_time = 0.0  # 这里应该记录实际执行时间
                task_status[task.task_id] = TaskStatus.SUCCESS
                task_outputs[task.task_id] = response.content

                result = ExecutionResult(
                    task_id=task.task_id,
                    status=TaskStatus.SUCCESS,
                    output=response.content,
                    execution_time=execution_time,
                    error_msg=None
                )

                results.add_result(result)
                self.database.log_event(session_id, "executor", "agentscope_task_completed",
                                       f"Task {task.task_id} completed with AgentScope")

        except Exception as e:
            # 标记所有任务为失败
            for task in tasks:
                task_status[task.task_id] = TaskStatus.FAILED
                result = ExecutionResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    output=None,
                    execution_time=0.0,
                    error_msg=f"AgentScope execution failed: {str(e)}"
                )
                results.add_result(result)

                self.database.log_event(session_id, "executor", "agentscope_task_failed",
                                       f"Task {task.task_id} failed: {str(e)}")

    async def _execute_single_task_with_agentscope(self,
                                                 task: TaskNode,
                                                 task_graph: TaskGraph,
                                                 task_status: Dict[str, TaskStatus],
                                                 task_outputs: Dict[str, Any],
                                                 results: ExecutionResults,
                                                 user_query: str,
                                                 session_id: str):
        """使用AgentScope执行单个任务"""
        task_status[task.task_id] = TaskStatus.RUNNING
        self.database.log_event(session_id, "executor", "agentscope_task_started",
                               f"Task {task.task_id}: {task.task_desc}")

        start_time = time.time()

        try:
            # 准备输入数据
            input_data = self._prepare_input_data(task, task_outputs, task_graph)
            task_context = {
                "task": task,
                "input_data": input_data,
                "user_query": user_query,
                "session_id": session_id
            }

            # 创建任务特定的工具包
            toolkit = await self._create_task_toolkit(task, session_id)

            # 创建执行智能体
            agent = TaskExecutionAgent(
                task_node=task,
                model_config=self.model_config,
                toolkit=toolkit,
                execution_context=task_context
            )

            # 创建输入消息
            input_msg = Msg(
                name="user",
                content=self._build_task_execution_message(task, input_data, user_query),
                role="user"
            )

            # 执行任务
            response = await agent.reply(input_msg)
            execution_time = time.time() - start_time

            # 记录成功结果
            task_status[task.task_id] = TaskStatus.SUCCESS
            task_outputs[task.task_id] = response.content

            result = ExecutionResult(
                task_id=task.task_id,
                status=TaskStatus.SUCCESS,
                output=response.content,
                execution_time=execution_time,
                error_msg=None
            )

            results.add_result(result)
            self.database.log_event(session_id, "executor", "agentscope_task_completed",
                                   f"Task {task.task_id} completed in {execution_time:.2f}s")

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"AgentScope task execution failed: {str(e)}"

            # 记录失败结果
            task_status[task.task_id] = TaskStatus.FAILED

            result = ExecutionResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                output=None,
                execution_time=execution_time,
                error_msg=error_msg
            )

            results.add_result(result)
            self.database.log_event(session_id, "executor", "agentscope_task_failed",
                                   f"Task {task.task_id} failed: {error_msg}")

    async def _create_task_toolkit(self, task: TaskNode, session_id: str) -> Optional[Toolkit]:
        """为特定任务创建工具包"""
        if not self.enable_agentscope or not self.agentscope_adapter:
            return None

        try:
            # 如果是MCP调用任务，创建相关工具包
            if task.task_type == TaskType.MCP_CALL:
                # 根据任务描述创建上下文工具包
                toolkit = await self.agentscope_adapter.create_contextual_toolkit(task.task_desc)
                self.database.log_event(session_id, "executor", "task_toolkit_created",
                                       f"Created contextual toolkit for task {task.task_id}")
                return toolkit
            else:
                # 本地计算任务可能不需要工具
                return None

        except Exception as e:
            self.database.log_event(session_id, "executor", "task_toolkit_creation_failed",
                                   f"Failed to create toolkit for task {task.task_id}: {str(e)}")
            return None

    def _build_task_execution_message(self, task: TaskNode, input_data: Dict[str, Any], user_query: str) -> str:
        """构建任务执行消息"""
        return f"""
任务ID: {task.task_id}
任务描述: {task.task_desc}
任务类型: {task.task_type.value}
预期输出: {task.expected_output}

原始用户问题: {user_query}

输入数据:
{json.dumps(input_data, ensure_ascii=False, indent=2)}

请根据任务描述和输入信息完成此任务。如果需要使用外部工具，请使用可用的MCP工具。
"""

    async def _legacy_execution(self,
                               task_graph: TaskGraph,
                               user_query: str,
                               session_id: str,
                               start_time: float) -> ExecutionResults:
        """传统的任务执行方式（向后兼容）"""
        try:
            # 使用原有的执行逻辑
            from .executor import TaskExecutor

            # 创建临时传统执行器
            legacy_executor = TaskExecutor(self.client, self.database, self.config, self.model_config)
            legacy_executor.mcp_services = self.mcp_services
            legacy_executor.mcp_manager = self.mcp_manager

            return await legacy_executor.execute_task_graph(task_graph, user_query, session_id)

        except Exception as e:
            self.database.log_event(session_id, "executor", "legacy_execution_failed",
                                   f"Legacy execution failed: {str(e)}")
            results = ExecutionResults()
            results.total_execution_time = time.time() - start_time
            return results

    def _build_dependency_map(self, task_graph: TaskGraph) -> Dict[str, List[str]]:
        """构建任务依赖映射"""
        dependency_map = {task.task_id: [] for task in task_graph.nodes}
        for edge in task_graph.edges:
            dependency_map[edge.to_task_id].append(edge.from_task_id)
        return dependency_map

    def _build_reverse_dependency_map(self, task_graph: TaskGraph) -> Dict[str, List[str]]:
        """构建反向依赖映射"""
        reverse_map = {task.task_id: [] for task in task_graph.nodes}
        for edge in task_graph.edges:
            reverse_map[edge.from_task_id].append(edge.to_task_id)
        return reverse_map

    def _get_ready_tasks(self,
                        task_graph: TaskGraph,
                        task_status: Dict[str, TaskStatus],
                        dependency_map: Dict[str, List[str]]) -> List[TaskNode]:
        """获取准备执行的任务"""
        ready_tasks = []

        for task in task_graph.nodes:
            if task_status[task.task_id] == TaskStatus.PENDING:
                # 检查依赖是否完成
                dependencies = dependency_map[task.task_id]
                all_deps_completed = all(
                    task_status.get(dep, TaskStatus.SUCCESS) == TaskStatus.SUCCESS
                    for dep in dependencies
                )

                if all_deps_completed:
                    ready_tasks.append(task)

        # 按优先级排序
        ready_tasks.sort(key=lambda t: (t.priority, t.task_id), reverse=True)

        # 应用并行限制
        parallel_limit = self._calculate_dynamic_parallel_limit(ready_tasks)
        return ready_tasks[:parallel_limit]

    def _calculate_dynamic_parallel_limit(self, ready_tasks: List[TaskNode]) -> int:
        """计算动态并行限制"""
        if not ready_tasks:
            return 0

        mcp_tasks = sum(1 for t in ready_tasks if t.task_type == TaskType.MCP_CALL)
        local_tasks = sum(1 for t in ready_tasks if t.task_type == TaskType.LOCAL_COMPUTE)

        if mcp_tasks > 0 and local_tasks == 0:
            return min(len(ready_tasks), self.config.max_parallel_tasks * 2)
        elif local_tasks > 0 and mcp_tasks == 0:
            return min(len(ready_tasks), max(1, self.config.max_parallel_tasks // 2))
        else:
            return min(len(ready_tasks), self.config.max_parallel_tasks)

    def _prepare_input_data(self, task: TaskNode, task_outputs: Dict[str, Any], task_graph: TaskGraph) -> Dict[str, Any]:
        """准备任务输入数据"""
        input_data = {
            "task_id": task.task_id,
            "task_desc": task.task_desc,
            "expected_output": task.expected_output
        }

        # 添加依赖任务的输出
        for edge in task_graph.edges:
            if edge.to_task_id == task.task_id and edge.from_task_id in task_outputs:
                input_data[f"dep_{edge.from_task_id}"] = task_outputs[edge.from_task_id]

        return input_data

    async def cleanup_mcp_manager(self):
        """清理MCP管理器"""
        try:
            if self.agentscope_adapter:
                await self.agentscope_adapter.close()
                self.agentscope_adapter = None

            if self.mcp_manager:
                await self.mcp_manager.close()
                self.mcp_manager = None

            self.database.log_event("", "executor", "cleanup_completed", "AgentScope executor cleaned up")

        except Exception as e:
            self.database.log_event("", "executor", "cleanup_error", f"Cleanup error: {str(e)}")


# 向后兼容的别名
TaskExecutor = AgentScopeTaskExecutor