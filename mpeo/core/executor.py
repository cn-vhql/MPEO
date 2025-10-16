"""
Executor Model - Task scheduling and execution
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
        self.mcp_manager: Optional[UnifiedMCPManager] = None
        self.execution_context: Dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self.model_config.model_name
    
    async def register_mcp_service(self, service_config: MCPServiceConfig):
        """Register an MCP service configuration"""
        self.mcp_services[service_config.service_name] = service_config

        # Initialize unified MCP manager if needed
        if self.mcp_manager is None:
            self.mcp_manager = UnifiedMCPManager()
            await self.mcp_manager.initialize()

        # Register service with unified manager
        success = await self.mcp_manager.register_service(service_config)

        if success:
            self.database.log_event(None, "executor", "mcp_service_registered",
                                   f"Service: {service_config.service_name}")
        else:
            self.database.log_event(None, "executor", "mcp_service_registration_failed",
                                   f"Service: {service_config.service_name}")

    def get_current_mcp_manager(self):
        """Get the currently active MCP manager, initializing if necessary"""
        if self.mcp_manager is None:
            self.mcp_manager = UnifiedMCPManager()
            # Note: We'll initialize it lazily when needed
        return self.mcp_manager
    
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
        """Get tasks that are ready to execute (all dependencies completed) with improved logic"""
        ready_tasks = []
        failed_dependencies = set()

        for task in task_graph.nodes:
            if task_status[task.task_id] == TaskStatus.PENDING:
                # Check if all dependencies are completed successfully
                dependencies = dependency_map[task.task_id]

                # Skip tasks with failed dependencies
                has_failed_dep = any(
                    task_status.get(dep, TaskStatus.FAILED) == TaskStatus.FAILED
                    for dep in dependencies
                )

                if has_failed_dep:
                    failed_dependencies.add(task.task_id)
                    continue

                # Check if all dependencies completed successfully
                all_deps_completed = all(
                    task_status.get(dep, TaskStatus.SUCCESS) == TaskStatus.SUCCESS
                    for dep in dependencies
                )

                if all_deps_completed:
                    ready_tasks.append(task)

        # Mark tasks with failed dependencies as failed
        for task_id in failed_dependencies:
            task_status[task_id] = TaskStatus.FAILED
            self.database.log_event("", "executor", "task_skipped_failed_deps",
                                   f"Task {task_id} skipped due to failed dependencies")

        # Sort by priority (higher priority first), then by task ID for consistency
        ready_tasks.sort(key=lambda t: (t.priority, t.task_id), reverse=True)

        # Apply smart parallelism limit based on task types
        parallel_limit = self._calculate_dynamic_parallel_limit(ready_tasks)
        return ready_tasks[:parallel_limit]

    def _calculate_dynamic_parallel_limit(self, ready_tasks: List[TaskNode]) -> int:
        """Calculate dynamic parallel limit based on task types and system load"""
        if not ready_tasks:
            return 0

        # Count task types
        mcp_tasks = sum(1 for t in ready_tasks if t.task_type == TaskType.MCP_CALL)
        local_tasks = sum(1 for t in ready_tasks if t.task_type == TaskType.LOCAL_COMPUTE)

        # MCP calls are I/O bound, can run more in parallel
        # Local compute tasks are CPU bound, limit concurrency
        if mcp_tasks > 0 and local_tasks == 0:
            # All MCP tasks - can run more in parallel
            return min(len(ready_tasks), self.config.max_parallel_tasks * 2)
        elif local_tasks > 0 and mcp_tasks == 0:
            # All local compute tasks - conservative parallelism
            return min(len(ready_tasks), max(1, self.config.max_parallel_tasks // 2))
        else:
            # Mixed tasks - use standard limit
            return min(len(ready_tasks), self.config.max_parallel_tasks)
    
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
        """Execute MCP service call using the unified MCP service manager with improved error handling"""
        current_manager = self.get_current_mcp_manager()

        # Step 1: Ensure the manager is initialized
        try:
            if not hasattr(current_manager, '_is_initialized') or not current_manager._is_initialized:
                await current_manager.initialize()
                self.database.log_event(session_id, "executor", "mcp_manager_initialized",
                                       f"Initialized UnifiedMCPManager")
        except Exception as e:
            error_msg = f"Failed to initialize MCP manager: {str(e)}"
            self.database.log_event(session_id, "executor", "mcp_manager_init_failed", error_msg)
            # Return a meaningful error instead of raising
            return f"[MCP服务初始化失败] {error_msg}"

        # Step 2: Extract and validate service name
        try:
            service_name = self._extract_service_name(task.task_desc)
            if not service_name or service_name == "default":
                # Fallback to first available service
                available_services = current_manager.get_registered_services()
                if available_services:
                    service_name = available_services[0]
                    self.database.log_event(session_id, "executor", "mcp_service_fallback",
                                           f"Using fallback service: {service_name}")
                else:
                    return "[MCP服务不可用] 没有注册的MCP服务"
        except Exception as e:
            error_msg = f"Failed to extract service name: {str(e)}"
            self.database.log_event(session_id, "executor", "service_extraction_failed", error_msg)
            return f"[MCP服务提取失败] {error_msg}"

        # Step 3: Check if service is registered
        if not current_manager.is_service_registered(service_name):
            available_services = current_manager.get_registered_services()
            error_msg = f"Service '{service_name}' not registered. Available: {available_services}"
            self.database.log_event(session_id, "executor", "mcp_service_not_found", error_msg)
            # Try to use the first available service as fallback
            if available_services:
                service_name = available_services[0]
                self.database.log_event(session_id, "executor", "mcp_service_fallback_used",
                                       f"Using fallback service: {service_name}")
            else:
                return "[MCP服务不可用] 没有注册的MCP服务"

        # Step 4: Extract tool name with fallback
        try:
            tool_name = self._extract_tool_name(task.task_desc)
            if not tool_name or tool_name == "execute":
                # Get available tools for the service
                tools = await current_manager.get_available_tools(service_name)
                if service_name in tools and tools[service_name]:
                    tool_name = tools[service_name][0].name  # Use first available tool
                    self.database.log_event(session_id, "executor", "tool_fallback_used",
                                           f"Using fallback tool: {tool_name}")
                else:
                    return f"[MCP工具不可用] 服务 '{service_name}' 没有可用工具"
        except Exception as e:
            error_msg = f"Failed to extract tool name: {str(e)}"
            self.database.log_event(session_id, "executor", "tool_extraction_failed", error_msg)
            return f"[MCP工具提取失败] {error_msg}"

        # Step 5: Prepare arguments with validation
        try:
            arguments = await self._prepare_tool_arguments(tool_name, task, input_data, user_query)
            if not arguments:
                arguments = {}  # Ensure arguments is not None
        except Exception as e:
            error_msg = f"Failed to prepare tool arguments: {str(e)}"
            self.database.log_event(session_id, "executor", "argument_preparation_failed", error_msg)
            arguments = {"task_desc": task.task_desc, "user_query": user_query}  # Basic fallback

        # Step 6: Execute the MCP call with comprehensive error handling
        try:
            self.database.log_event(session_id, "executor", "mcp_tool_call_start",
                                   f"Calling tool: {tool_name} on service: {service_name}")

            result: MCPResult = await current_manager.call_tool(
                service_name=service_name,
                tool_name=tool_name,
                arguments=arguments
            )

            # Step 7: Process the result
            if result.success:
                self.database.log_event(session_id, "executor", "mcp_call_success",
                                       f"Tool {tool_name} completed in {result.execution_time:.2f}s")
                return self._format_mcp_result(result.data)
            else:
                # Handle specific error types
                error_data = result.data if isinstance(result.data, dict) else {}
                error_type = error_data.get("error", "UnknownError")
                error_msg = result.error or "Unknown error occurred"

                self.database.log_event(session_id, "executor", "mcp_call_failed",
                                       f"Tool {tool_name} failed: {error_type} - {error_msg}")

                # Handle known error types gracefully
                if error_type in ["ContentLengthError", "JSONDecodeError", "TimeoutError"]:
                    return self._format_mcp_result(error_data)
                else:
                    return f"[MCP调用失败] {error_type}: {error_msg}"

        except asyncio.TimeoutError:
            error_msg = f"MCP call timeout for tool {tool_name}"
            self.database.log_event(session_id, "executor", "mcp_timeout", error_msg)
            return f"[MCP调用超时] 工具 {tool_name} 调用超时"

        except ConnectionError as e:
            error_msg = f"MCP connection error: {str(e)}"
            self.database.log_event(session_id, "executor", "mcp_connection_error", error_msg)
            return f"[MCP连接错误] 无法连接到MCP服务: {str(e)}"

        except Exception as e:
            error_msg = f"Unexpected MCP call error: {str(e)}"
            self.database.log_event(session_id, "executor", "mcp_unexpected_error", error_msg)
            # Try to provide more context if available
            if hasattr(e, '__cause__') and e.__cause__:
                return f"[MCP调用异常] {error_msg} (原因: {str(e.__cause__)})"
            return f"[MCP调用异常] {error_msg}"

    def _extract_service_name(self, task_desc: str) -> str:
        """Extract MCP service name from task description using tool registry"""
        current_manager = self.get_current_mcp_manager()
        if not current_manager:
            return "default"

        task_desc_lower = task_desc.lower()

        # Try to extract service name from task description
        if "mcp_" in task_desc_lower:
            # Try to extract service name after "mcp_"
            parts = task_desc_lower.split("mcp_")
            if len(parts) > 1:
                service_name = parts[1].split()[0]
                # Check if this service is registered
                if current_manager.is_service_registered(service_name):
                    return service_name

        # Use tool registry to find service for the tool
        tool_name = self._extract_tool_name(task_desc)
        service_name = current_manager.tool_registry.find_service(tool_name)
        if service_name:
            return service_name

        # Fallback: try to find service name in description
        for service_name in current_manager.get_registered_services():
            if service_name.lower() in task_desc_lower:
                return service_name

        # Use first available service as default
        services = current_manager.get_registered_services()
        if services:
            return services[0]

        # No services available, return default
        return "default"

    def _extract_tool_name(self, task_desc: str) -> str:
        """Extract tool name from task description - simplified version"""
        task_desc_lower = task_desc.lower()

        # Common tool keywords mapping
        tool_keywords = {
            "current_time": ["时间", "time", "当前时间", "现在几点", "获取时间", "current time"],
            "fetch": ["fetch", "get", "获取", "抓取", "retrieve"],
            "search": ["search", "查找", "搜索", "find"],
            "process": ["process", "处理", "handle"],
            "analyze": ["analyze", "分析", "analysis"],
            "calculate": ["calculate", "计算", "compute"],
            "translate": ["translate", "翻译", "translation"],
            "summarize": ["summarize", "总结", "摘要", "summary"],
            "get_library_docs": ["docs", "文档", "documentation", "library"]
        }

        # Find matching tool
        for tool_name, keywords in tool_keywords.items():
            if any(keyword in task_desc_lower for keyword in keywords):
                return tool_name

        # Default tool name
        return "execute"

    async def _prepare_tool_arguments(self, tool_name: str, task: TaskNode,
                                   input_data: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Prepare arguments for MCP tool call based on tool requirements"""

        # Get tool schema from current MCP manager
        tool_schema = None
        try:
            current_manager = self.get_current_mcp_manager()
            if current_manager:
                # Get the service that contains this tool
                service_name = self._extract_service_name(task.task_desc)
                if service_name and current_manager.is_service_registered(service_name):
                    tools = await current_manager.get_available_tools(service_name)
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

        # Special handling for time-related tools
        if tool_name == "current_time":
            arguments = {}
            # Use default format based on tool schema
            arguments["format"] = "YYYY-MM-DD HH:mm:ss"

            # Check if timezone is mentioned
            if "timezone" in task.task_desc.lower() or "时区" in task.task_desc:
                # Try to extract timezone from task description
                import re
                timezone_match = re.search(r'(?:timezone|时区)[:\s]*([a-zA-Z_/\-+]+)', task.task_desc, re.IGNORECASE)
                if timezone_match:
                    arguments["timezone"] = timezone_match.group(1)
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

    def _format_mcp_result(self, mcp_data: Any) -> Union[str, Dict[str, Any]]:
        """Format MCP result data to match ExecutionResult expectations"""
        if isinstance(mcp_data, str):
            return mcp_data
        elif isinstance(mcp_data, dict):
            # Handle error responses with partial content
            if "error" in mcp_data:
                error_type = mcp_data.get("error", "Unknown")

                # Try to get partial content for better fallback
                if "partial_content" in mcp_data:
                    partial = mcp_data.get("partial_content", "")
                    if partial and partial.strip():
                        # Add a prefix to indicate partial content
                        return f"[部分内容获取成功 - 原因: {error_type}]\n\n{partial}"

                # Try raw response for JSON decode errors
                if "raw_response" in mcp_data:
                    raw = mcp_data.get("raw_response", "")
                    if raw and raw.strip():
                        return f"[原始响应 - 原因: {error_type}]\n\n{raw}"

                # Return error message as fallback
                message = mcp_data.get("message", f"发生错误: {error_type}")
                return f"[数据获取失败: {error_type}]\n\n详细信息: {message}"

            return mcp_data
        elif isinstance(mcp_data, list):
            # Handle list format from MCP tools
            if not mcp_data:
                return "Empty result"

            # If list contains text content blocks, extract and concatenate text
            if len(mcp_data) == 1 and isinstance(mcp_data[0], dict):
                item = mcp_data[0]
                if item.get("type") == "text" and "text" in item:
                    return item["text"]

            # If multiple items, try to extract text from all text blocks
            text_parts = []
            for item in mcp_data:
                if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])

            if text_parts:
                return " ".join(text_parts)

            # Fallback: convert list to string representation
            return str(mcp_data)
        else:
            # Fallback for other data types
            return str(mcp_data)

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