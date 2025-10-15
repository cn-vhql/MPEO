"""
Executor Model - Task scheduling and execution
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
from openai import OpenAI

from ..models import (
    TaskGraph, TaskNode, TaskEdge, TaskType, TaskStatus,
    ExecutionResult, ExecutionResults, SystemConfig, MCPServiceConfig
)
from ..models.agent_config import AgentModelConfig
from ..services.database import DatabaseManager
from ..services.mcp_manager import MCPServiceManager, MCPResult


class TaskExecutor:
    """Task executor with parallel and serial execution capabilities"""

    def __init__(self,
                 openai_client: OpenAI,
                 database: DatabaseManager,
                 config: SystemConfig,
                 model_config: AgentModelConfig = None):
        self.client = openai_client
        self.database = database
        self.config = config
        self.model_config = model_config or AgentModelConfig(
            model_name=config.openai_model,
            temperature=0.2,
            max_tokens=1500,
            timeout=30
        )
        self.mcp_services: Dict[str, MCPServiceConfig] = {}
        self.mcp_manager: Optional[MCPServiceManager] = None
        self.execution_context: Dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self.model_config.model_name
    
    async def register_mcp_service(self, service_config: MCPServiceConfig):
        """Register an MCP service configuration"""
        self.mcp_services[service_config.service_name] = service_config

        # Initialize MCP manager if needed
        if self.mcp_manager is None:
            self.mcp_manager = MCPServiceManager()
            await self.mcp_manager.initialize()

        # Register service with manager
        success = await self.mcp_manager.register_from_service_config(service_config)

        if success:
            self.database.log_event(None, "executor", "mcp_service_registered",
                                   f"Service: {service_config.service_name}")
        else:
            self.database.log_event(None, "executor", "mcp_service_registration_failed",
                                   f"Service: {service_config.service_name}")
    
    async def execute_task_graph(self, task_graph: TaskGraph, user_query: str, session_id: str) -> ExecutionResults:
        """
        Execute all tasks in the task graph according to dependencies
        
        Args:
            task_graph: Task graph to execute
            user_query: Original user query
            session_id: Session identifier
            
        Returns:
            ExecutionResults: Results of all task executions
        """
        self.database.log_event(session_id, "executor", "start_execution", 
                               f"Executing {len(task_graph.nodes)} tasks")
        
        start_time = time.time()
        results = ExecutionResults()
        
        try:
            # Build dependency map
            dependency_map = self._build_dependency_map(task_graph)
            reverse_dependency_map = self._build_reverse_dependency_map(task_graph)
            
            # Track task status
            task_status: Dict[str, TaskStatus] = {}
            task_outputs: Dict[str, Any] = {}
            
            # Initialize all tasks as pending
            for task in task_graph.nodes:
                task_status[task.task_id] = TaskStatus.PENDING
            
            # Execute tasks based on dependencies
            while len(task_status) > len([s for s in task_status.values() if s in [TaskStatus.SUCCESS, TaskStatus.FAILED]]):
                # Find tasks ready to execute (all dependencies completed)
                ready_tasks = self._get_ready_tasks(task_graph, task_status, dependency_map)
                
                if not ready_tasks:
                    # Check for deadlock (shouldn't happen with valid DAG)
                    pending_tasks = [tid for tid, status in task_status.items() if status == TaskStatus.PENDING]
                    if pending_tasks:
                        self.database.log_event(session_id, "executor", "deadlock_detected", 
                                               f"Pending tasks: {pending_tasks}")
                        # Mark remaining tasks as failed
                        for task_id in pending_tasks:
                            task_status[task_id] = TaskStatus.FAILED
                            result = ExecutionResult(
                                task_id=task_id,
                                status=TaskStatus.FAILED,
                                output=None,
                                execution_time=0.0,
                                error_msg="Deadlock detected - unable to resolve dependencies"
                            )
                            results.add_result(result)
                    break
                
                # Execute ready tasks in parallel
                if len(ready_tasks) > 1:
                    # Parallel execution
                    await self._execute_tasks_parallel(
                        ready_tasks, task_graph, task_status, task_outputs, 
                        results, user_query, session_id
                    )
                else:
                    # Single task execution
                    await self._execute_single_task(
                        ready_tasks[0], task_graph, task_status, task_outputs,
                        results, user_query, session_id
                    )
            
            # Calculate total execution time
            results.total_execution_time = time.time() - start_time
            
            self.database.log_event(session_id, "executor", "execution_completed", 
                                   f"Completed: {results.success_count}, Failed: {results.failed_count}")
            
            return results
            
        except Exception as e:
            self.database.log_event(session_id, "executor", "execution_error", f"Execution failed: {str(e)}")
            results.total_execution_time = time.time() - start_time
            return results
    
    def _build_dependency_map(self, task_graph: TaskGraph) -> Dict[str, List[str]]:
        """Build map of task -> its dependencies"""
        dependency_map = {task.task_id: [] for task in task_graph.nodes}
        for edge in task_graph.edges:
            dependency_map[edge.to_task_id].append(edge.from_task_id)
        return dependency_map
    
    def _build_reverse_dependency_map(self, task_graph: TaskGraph) -> Dict[str, List[str]]:
        """Build map of task -> tasks that depend on it"""
        reverse_map = {task.task_id: [] for task in task_graph.nodes}
        for edge in task_graph.edges:
            reverse_map[edge.from_task_id].append(edge.to_task_id)
        return reverse_map
    
    def _get_ready_tasks(self, task_graph: TaskGraph, task_status: Dict[str, TaskStatus], 
                        dependency_map: Dict[str, List[str]]) -> List[TaskNode]:
        """Get tasks that are ready to execute (all dependencies completed)"""
        ready_tasks = []
        
        for task in task_graph.nodes:
            if task_status[task.task_id] == TaskStatus.PENDING:
                # Check if all dependencies are completed successfully
                dependencies = dependency_map[task.task_id]
                all_deps_completed = all(
                    task_status.get(dep, TaskStatus.FAILED) == TaskStatus.SUCCESS 
                    for dep in dependencies
                )
                
                if all_deps_completed:
                    ready_tasks.append(task)
        
        # Sort by priority (higher priority first)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Limit by max parallel tasks
        return ready_tasks[:self.config.max_parallel_tasks]
    
    async def _execute_tasks_parallel(self, tasks: List[TaskNode], task_graph: TaskGraph,
                                    task_status: Dict[str, TaskStatus], task_outputs: Dict[str, Any],
                                    results: ExecutionResults, user_query: str, session_id: str):
        """Execute multiple tasks in parallel"""
        self.database.log_event(session_id, "executor", "parallel_execution", 
                               f"Executing {len(tasks)} tasks in parallel")
        
        async def execute_task_wrapper(task: TaskNode):
            return await self._execute_single_task(
                task, task_graph, task_status, task_outputs, results, user_query, session_id
            )
        
        # Create tasks for parallel execution
        coroutines = [execute_task_wrapper(task) for task in tasks]
        await asyncio.gather(*coroutines, return_exceptions=True)
    
    async def _execute_single_task(self, task: TaskNode, task_graph: TaskGraph,
                                 task_status: Dict[str, TaskStatus], task_outputs: Dict[str, Any],
                                 results: ExecutionResults, user_query: str, session_id: str) -> ExecutionResult:
        """Execute a single task"""
        task_status[task.task_id] = TaskStatus.RUNNING
        self.database.log_event(session_id, "executor", "task_started", f"Task {task.task_id}: {task.task_desc}")
        
        start_time = time.time()
        
        # Prepare input data from dependent tasks
        input_data = self._prepare_input_data(task, task_outputs, task_graph)
        
        # Execute with retry logic
        for attempt in range(self.config.task_retry_count + 1):
            try:
                if attempt > 0:
                    self.database.log_event(session_id, "executor", "task_retry", 
                                           f"Retrying task {task.task_id}, attempt {attempt + 1}")
                
                # Execute based on task type
                if task.task_type == TaskType.LOCAL_COMPUTE:
                    output = await self._execute_local_compute(task, input_data, user_query, session_id)
                elif task.task_type == TaskType.MCP_CALL:
                    output = await self._execute_mcp_call(task, input_data, user_query, session_id)
                elif task.task_type == TaskType.DATA_PROCESSING:
                    output = await self._execute_data_processing(task, input_data, session_id)
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                # Task completed successfully
                execution_time = time.time() - start_time
                task_status[task.task_id] = TaskStatus.SUCCESS
                task_outputs[task.task_id] = output
                
                result = ExecutionResult(
                    task_id=task.task_id,
                    status=TaskStatus.SUCCESS,
                    output=output,
                    execution_time=execution_time,
                    error_msg=None
                )
                
                results.add_result(result)
                self.database.log_event(session_id, "executor", "task_completed", 
                                       f"Task {task.task_id} completed in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                error_msg = f"Task execution failed (attempt {attempt + 1}): {str(e)}"
                self.database.log_event(session_id, "executor", "task_error", 
                                       f"Task {task.task_id}: {error_msg}")
                
                if attempt == self.config.task_retry_count:
                    # All retries failed
                    execution_time = time.time() - start_time
                    task_status[task.task_id] = TaskStatus.FAILED
                    
                    result = ExecutionResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        output=None,
                        execution_time=execution_time,
                        error_msg=error_msg
                    )
                    
                    results.add_result(result)
                    self.database.log_event(session_id, "executor", "task_failed", 
                                           f"Task {task.task_id} failed after {attempt + 1} attempts")
                    
                    return result
                
                # Wait before retry
                await asyncio.sleep(1.0 * (attempt + 1))
    
    def _prepare_input_data(self, task: TaskNode, task_outputs: Dict[str, Any], 
                           task_graph: TaskGraph) -> Dict[str, Any]:
        """Prepare input data for task based on its dependencies"""
        input_data = {
            "task_id": task.task_id,
            "task_desc": task.task_desc,
            "expected_output": task.expected_output
        }
        
        # Add outputs from dependent tasks
        for edge in task_graph.edges:
            if edge.to_task_id == task.task_id and edge.from_task_id in task_outputs:
                input_data[f"dep_{edge.from_task_id}"] = task_outputs[edge.from_task_id]
        
        return input_data
    
    async def _execute_local_compute(self, task: TaskNode, input_data: Dict[str, Any], 
                                   user_query: str, session_id: str) -> Any:
        """Execute local computation task using OpenAI"""
        prompt = f"""
        请执行以下本地计算任务：

        任务描述：{task.task_desc}
        预期输出：{task.expected_output}
        
        原始用户问题：{user_query}
        
        输入数据：
        {json.dumps(input_data, ensure_ascii=False, indent=2)}
        
        请根据任务描述和输入数据，执行相应的计算或处理，并返回结果。
        确保输出格式符合预期要求。
        """
        
        messages = [
            {"role": "system", "content": self.model_config.system_prompt or "你是一个专业的计算和处理助手，能够执行各种本地计算任务。"},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            top_p=self.model_config.top_p,
            frequency_penalty=self.model_config.frequency_penalty,
            presence_penalty=self.model_config.presence_penalty
        )
        
        return response.choices[0].message.content
    
    async def _execute_mcp_call(self, task: TaskNode, input_data: Dict[str, Any],
                              user_query: str, session_id: str) -> Any:
        """Execute MCP service call using the new MCP service manager"""
        if self.mcp_manager is None:
            raise ValueError("MCP service manager not initialized")

        # Extract service name from task description or use default
        service_name = self._extract_service_name(task.task_desc)

        self.database.log_event(session_id, "executor", "mcp_call_start",
                               f"Task: {task.task_id}, Service: {service_name}")

        # Check if service is registered
        if not self.mcp_manager.is_service_registered(service_name):
            available_services = self.mcp_manager.get_registered_services()
            self.database.log_event(session_id, "executor", "mcp_service_not_found",
                                   f"Service '{service_name}' not registered. Available: {available_services}")
            raise ValueError(f"MCP service '{service_name}' not registered")

        try:
            # Extract tool name from task description
            tool_name = self._extract_tool_name(task.task_desc)

            # Prepare arguments for the tool call
            arguments = await self._prepare_tool_arguments(tool_name, task, input_data, user_query)

            self.database.log_event(session_id, "executor", "mcp_tool_call",
                                   f"Calling tool: {tool_name} on service: {service_name}")

            # Call the tool through the MCP manager
            result: MCPResult = await self.mcp_manager.call_tool(
                service_name=service_name,
                tool_name=tool_name,
                arguments=arguments
            )

            # Log the result
            if result.success:
                self.database.log_event(session_id, "executor", "mcp_call_success",
                                       f"Tool {tool_name} completed in {result.execution_time:.2f}s")
                return result.data
            else:
                self.database.log_event(session_id, "executor", "mcp_call_failed",
                                       f"Tool {tool_name} failed: {result.error}")
                raise Exception(f"MCP tool call failed: {result.error}")

        except Exception as e:
            self.database.log_event(session_id, "executor", "mcp_call_error",
                                   f"MCP call failed: {str(e)}")
            raise

    def _extract_service_name(self, task_desc: str) -> str:
        """Extract MCP service name from task description"""
        # Try to extract service name from task description
        if "mcp_" in task_desc.lower():
            # Try to extract service name after "mcp_"
            parts = task_desc.lower().split("mcp_")
            if len(parts) > 1:
                service_name = parts[1].split()[0]
                # Check if this service is registered
                if self.mcp_manager and self.mcp_manager.is_service_registered(service_name):
                    return service_name

        # Try to find service name in description
        if self.mcp_manager:
            for service_name in self.mcp_manager.get_registered_services():
                if service_name.lower() in task_desc.lower():
                    return service_name

        # Enhanced service name extraction
        # Check for common MCP service patterns
        if "fetch" in task_desc.lower() or "获取" in task_desc:
            if self.mcp_manager and self.mcp_manager.is_service_registered("fetch"):
                return "fetch"

        if "bing" in task_desc.lower() or "搜索" in task_desc:
            if self.mcp_manager and self.mcp_manager.is_service_registered("bing-cn-mcp-server"):
                return "bing-cn-mcp-server"

        # Use first available service as default
        if self.mcp_manager:
            services = self.mcp_manager.get_registered_services()
            if services:
                return services[0]

        # No services available
        return "default"

    def _extract_tool_name(self, task_desc: str) -> str:
        """Extract tool name from task description"""
        # Look for common tool patterns in task description
        task_desc_lower = task_desc.lower()

        # Common tool name patterns
        if "fetch" in task_desc_lower or "get" in task_desc_lower or "获取" in task_desc_lower:
            return "fetch"
        elif "search" in task_desc_lower or "查找" in task_desc_lower or "搜索" in task_desc_lower:
            return "search"
        elif "process" in task_desc_lower or "处理" in task_desc_lower:
            return "process"
        elif "analyze" in task_desc_lower or "分析" in task_desc_lower:
            return "analyze"
        elif "calculate" in task_desc_lower or "计算" in task_desc_lower:
            return "calculate"
        elif "translate" in task_desc_lower or "翻译" in task_desc_lower:
            return "translate"
        elif "summarize" in task_desc_lower or "总结" in task_desc_lower or "摘要" in task_desc_lower:
            return "summarize"
        else:
            # Default tool name
            return "execute"

    async def _prepare_tool_arguments(self, tool_name: str, task: TaskNode,
                                   input_data: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Prepare arguments for MCP tool call based on tool requirements"""

        # Get tool schema from MCP manager
        tool_schema = None
        try:
            if self.mcp_manager:
                # Get the service that contains this tool
                service_name = self._extract_service_name(task.task_desc)
                if service_name and self.mcp_manager.is_service_registered(service_name):
                    tools = await self.mcp_manager.get_available_tools(service_name)
                    if service_name in tools:
                        for tool in tools[service_name]:
                            if tool.name == tool_name:
                                tool_schema = tool.input_schema
                                break
        except Exception as e:
            self.database.log_event(None, "executor", "tool_schema_error",
                                   f"Failed to get schema for {tool_name}: {str(e)}")

        # Prepare arguments based on tool schema
        if tool_name == "fetch" and tool_schema:
            arguments = {}

            # Extract URL from task description, user query, or input data
            url = self._extract_url_from_context(task.task_desc, user_query, input_data)
            if url:
                arguments["url"] = url
            else:
                # Try to extract from the task description itself
                url = self._extract_url_from_text(task.task_desc)
                if url:
                    arguments["url"] = url

            # Add optional parameters if defaults are available
            if "properties" in tool_schema:
                for param_name, param_info in tool_schema["properties"].items():
                    if param_name not in arguments and "default" in param_info:
                        arguments[param_name] = param_info["default"]

            return arguments

        # Default argument preparation for other tools
        return {
            "task_id": task.task_id,
            "task_desc": task.task_desc,
            "expected_output": task.expected_output,
            "input_data": input_data
        }

    def _extract_url_from_context(self, task_desc: str, user_query: str, input_data: Dict[str, Any]) -> Optional[str]:
        """Extract URL from task description, user query, or input data"""

        # First, try to extract from user query
        url = self._extract_url_from_text(user_query)
        if url:
            return url

        # Then, try to extract from task description
        url = self._extract_url_from_text(task_desc)
        if url:
            return url

        # Finally, try to extract from input data
        for key, value in input_data.items():
            if isinstance(value, str):
                url = self._extract_url_from_text(value)
                if url:
                    return url
            elif isinstance(value, dict):
                # recursively search in dict values
                dict_result = self._extract_url_from_context("", "", value)
                if dict_result:
                    return dict_result

        return None

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        """Extract URL from text using regex patterns"""
        import re

        # Common URL patterns
        url_patterns = [
            r'https?://[^\s\)\]}]+',  # Standard URLs
            r'(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s\)\]}]+)?',  # More lenient
        ]

        for pattern in url_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the first match that looks like a complete URL
                for match in matches:
                    if match.startswith('http'):
                        return match
                    elif '.' in match and len(match) > 10:  # Basic validation
                        return f"https://{match}"

        return None

    async def cleanup_mcp_manager(self):
        """Cleanup MCP service manager"""
        if self.mcp_manager:
            await self.mcp_manager.close()
            self.mcp_manager = None

    async def _execute_data_processing(self, task: TaskNode, input_data: Dict[str, Any],
                                     session_id: str) -> Any:
        """Execute data processing task"""
        prompt = f"""
        请执行以下数据处理任务：

        任务描述：{task.task_desc}
        预期输出：{task.expected_output}
        
        输入数据：
        {json.dumps(input_data, ensure_ascii=False, indent=2)}
        
        请对输入数据进行处理，包括但不限于：
        - 数据清洗和格式化
        - 数据转换和计算
        - 数据分析和统计
        - 结果整理和呈现
        
        返回处理后的结果。
        """
        
        messages = [
            {"role": "system", "content": self.model_config.system_prompt or "你是一个专业的数据处理专家，擅长各种数据分析和处理任务。"},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            top_p=self.model_config.top_p,
            frequency_penalty=self.model_config.frequency_penalty,
            presence_penalty=self.model_config.presence_penalty
        )
        
        return response.choices[0].message.content