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

from .models import (
    TaskGraph, TaskNode, TaskEdge, TaskType, TaskStatus, 
    ExecutionResult, ExecutionResults, SystemConfig, MCPServiceConfig
)
from .database import DatabaseManager


class TaskExecutor:
    """Task executor with parallel and serial execution capabilities"""
    
    def __init__(self, 
                 openai_client: OpenAI,
                 database: DatabaseManager,
                 config: SystemConfig):
        self.client = openai_client
        self.database = database
        self.config = config
        self.mcp_services: Dict[str, MCPServiceConfig] = {}
        self.execution_context: Dict[str, Any] = {}
    
    def register_mcp_service(self, service_config: MCPServiceConfig):
        """Register an MCP service configuration"""
        self.mcp_services[service_config.service_name] = service_config
        self.database.log_event(None, "executor", "mcp_service_registered", 
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
                    output = await self._execute_mcp_call(task, input_data, session_id)
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
        
        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": "你是一个专业的计算和处理助手，能够执行各种本地计算任务。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    async def _execute_mcp_call(self, task: TaskNode, input_data: Dict[str, Any], 
                              session_id: str) -> Any:
        """Execute MCP service call"""
        # Extract service name from task description or use default
        service_name = self._extract_service_name(task.task_desc)
        
        self.database.log_event(session_id, "executor", "mcp_call_debug", 
                               f"Task: {task.task_id}, Extracted service name: {service_name}")
        
        if service_name not in self.mcp_services:
            available_services = list(self.mcp_services.keys())
            self.database.log_event(session_id, "executor", "mcp_call_debug", 
                                   f"Service '{service_name}' not registered. Available services: {available_services}")
            raise ValueError(f"MCP service '{service_name}' not registered")
        
        service_config = self.mcp_services[service_name]
        
        self.database.log_event(session_id, "executor", "mcp_call_debug", 
                               f"Service config: name={service_config.service_name}, type={service_config.service_type}, url={service_config.endpoint_url}")
        self.database.log_event(session_id, "executor", "mcp_call_debug", 
                               f"Input data: {json.dumps(input_data, ensure_ascii=False)}")
        
        # Make HTTP request to MCP service based on service type
        async with aiohttp.ClientSession() as session:
            try:
                if service_config.service_type == "sse":
                    # Handle SSE (Server-Sent Events) service
                    self.database.log_event(session_id, "executor", "mcp_call_debug", 
                                           f"Routing to SSE call method")
                    return await self._execute_sse_call(session, service_config, task, input_data, session_id)
                else:
                    # Handle regular HTTP service
                    self.database.log_event(session_id, "executor", "mcp_call_debug", 
                                           f"Routing to HTTP call method")
                    return await self._execute_http_call(session, service_config, task, input_data, session_id)
                        
            except asyncio.TimeoutError:
                self.database.log_event(session_id, "executor", "mcp_call_debug",
                                       f"Timeout after 60 seconds")
                raise Exception(f"MCP service call timeout after 60 seconds")
            except aiohttp.ClientError as e:
                self.database.log_event(session_id, "executor", "mcp_call_debug",
                                       f"HTTP client error: {type(e).__name__}: {str(e)}")
                raise Exception(f"MCP service HTTP error: {str(e)}")
            except Exception as e:
                self.database.log_event(session_id, "executor", "mcp_call_debug",
                                       f"Exception: {type(e).__name__}: {str(e)}")
                import traceback
                self.database.log_event(session_id, "executor", "mcp_call_debug",
                                       f"Traceback: {traceback.format_exc()}")
                raise Exception(f"MCP service call failed: {str(e)}")
    
    async def _execute_http_call(self, session: aiohttp.ClientSession, service_config: MCPServiceConfig,
                                task: TaskNode, input_data: Dict[str, Any], session_id: str) -> Any:
        """Execute regular HTTP MCP service call"""
        # Prepare request payload
        payload = {
            "task_id": task.task_id,
            "input_data": input_data,
            "timeout": service_config.timeout
        }
        
        headers = service_config.headers or {}
        
        self.database.log_event(session_id, "executor", "http_call_debug", 
                               f"POST URL: {service_config.endpoint_url}")
        self.database.log_event(session_id, "executor", "http_call_debug", 
                               f"Headers: {json.dumps(headers, ensure_ascii=False)}")
        self.database.log_event(session_id, "executor", "http_call_debug", 
                               f"Payload: {json.dumps(payload, ensure_ascii=False)}")
        
        async with session.post(
            service_config.endpoint_url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=service_config.timeout)
        ) as response:
            self.database.log_event(session_id, "executor", "http_call_debug", 
                                   f"Response status: {response.status}")
            self.database.log_event(session_id, "executor", "http_call_debug", 
                                   f"Response headers: {dict(response.headers)}")
            
            if response.status == 200:
                response_text = await response.text()
                self.database.log_event(session_id, "executor", "http_call_debug", 
                                       f"Response body: {response_text[:500]}...")
                try:
                    result = json.loads(response_text)
                    self.database.log_event(session_id, "executor", "mcp_call_success", 
                                           f"Service: {service_config.service_name}, Task: {task.task_id}")
                    return result
                except json.JSONDecodeError as e:
                    self.database.log_event(session_id, "executor", "http_call_debug", 
                                           f"JSON parse error: {str(e)}")
                    return response_text
            else:
                error_text = await response.text()
                self.database.log_event(session_id, "executor", "http_call_debug", 
                                       f"Error response: {error_text}")
                raise Exception(f"MCP service returned status {response.status}: {error_text}")
    
    async def _execute_sse_call(self, session: aiohttp.ClientSession, service_config: MCPServiceConfig,
                              task: TaskNode, input_data: Dict[str, Any], session_id: str) -> Any:
        """Execute SSE (Server-Sent Events) MCP service call"""

        # Prepare headers for SSE - use standard MCP protocol headers
        headers = service_config.headers or {}
        headers.setdefault("Accept", "text/event-stream")
        headers.setdefault("Cache-Control", "no-cache")
        headers.setdefault("Connection", "keep-alive")

        # According to MCP protocol, use GET request for SSE streams
        # Prepare query parameters
        params = {}

        # Add task information as query parameters
        if task.task_id:
            params["task_id"] = task.task_id

        # Add input data as JSON string parameter
        if input_data:
            params["input"] = json.dumps(input_data, ensure_ascii=False)

        # Add task description if available
        if task.task_desc:
            params["task"] = task.task_desc

        self.database.log_event(session_id, "executor", "sse_call_start",
                               f"Service: {service_config.service_name}, Task: {task.task_id}")
        self.database.log_event(session_id, "executor", "sse_call_debug",
                               f"GET URL: {service_config.endpoint_url}")
        self.database.log_event(session_id, "executor", "sse_call_debug",
                               f"Headers: {json.dumps(headers, ensure_ascii=False)}")
        self.database.log_event(session_id, "executor", "sse_call_debug",
                               f"Params: {json.dumps(params, ensure_ascii=False)}")

        # Use a more flexible timeout configuration
        timeout = aiohttp.ClientTimeout(
            total=60,  # Increase to 60 seconds total
            connect=15,  # 15 seconds to establish connection
            sock_read=45  # 45 seconds for reading
        )

        async with session.get(
            service_config.endpoint_url,
            params=params,
            headers=headers,
            timeout=timeout
        ) as response:
            self.database.log_event(session_id, "executor", "sse_call_debug", 
                                   f"Response status: {response.status}")
            self.database.log_event(session_id, "executor", "sse_call_debug", 
                                   f"Response headers: {dict(response.headers)}")
            
            if response.status == 200:
                # Check if response is SSE
                content_type = response.headers.get('Content-Type', '')
                self.database.log_event(session_id, "executor", "sse_call_debug", 
                                       f"Content-Type: {content_type}")
                
                if 'text/event-stream' in content_type:
                    # Handle SSE stream with timeout
                    self.database.log_event(session_id, "executor", "sse_call_debug",
                                           f"Processing SSE stream...")
                    events = []
                    line_count = 0

                    try:
                        # Create a timeout task for reading
                        async with asyncio.timeout(30):  # 30 second timeout for reading
                            async for line in response.content:
                                if line:
                                    line_text = line.decode('utf-8').strip()
                                    self.database.log_event(session_id, "executor", "sse_call_debug",
                                                           f"Raw line {line_count}: {line_text[:100]}...")

                                    # Handle both 'data:' and 'data: ' formats
                                    if line_text.startswith('data:'):
                                        if line_text.startswith('data: '):
                                            event_data = line_text[6:]  # Remove 'data: ' prefix
                                        else:
                                            event_data = line_text[5:]  # Remove 'data:' prefix

                                        if event_data:
                                            self.database.log_event(session_id, "executor", "sse_call_debug",
                                                                   f"Event {line_count}: {event_data[:100]}...")
                                            line_count += 1

                                            # Handle special SSE events
                                            if event_data == '[DONE]':
                                                self.database.log_event(session_id, "executor", "sse_call_debug",
                                                                       f"Stream completed with [DONE] marker")
                                                break

                                            try:
                                                # Try to parse as JSON
                                                event_json = json.loads(event_data)
                                                events.append(event_json)
                                            except json.JSONDecodeError:
                                                # Keep as text if not JSON
                                                events.append(event_data)

                                    # Stop after receiving some events to avoid hanging
                                    if line_count >= 5:  # Limit to 5 events
                                        self.database.log_event(session_id, "executor", "sse_call_debug",
                                                               f"Stopping after 5 events to avoid hanging")
                                        break

                    except asyncio.TimeoutError:
                        self.database.log_event(session_id, "executor", "sse_call_debug",
                                               f"SSE read timeout after 30 seconds")

                    self.database.log_event(session_id, "executor", "sse_call_debug",
                                           f"Total events received: {len(events)}")

                    if events:
                        # Return the last event or all events
                        result = events[-1] if len(events) == 1 else {"events": events}
                        self.database.log_event(session_id, "executor", "sse_call_success",
                                               f"Service: {service_config.service_name}, Task: {task.task_id}")
                        return result
                    else:
                        self.database.log_event(session_id, "executor", "sse_call_debug",
                                               f"No events received from SSE stream")
                        # Instead of raising an exception, return a default response
                        return {"status": "completed", "message": "SSE connection established but no events received"}
                else:
                    # Handle as regular response
                    response_text = await response.text()
                    self.database.log_event(session_id, "executor", "sse_call_debug", 
                                           f"Non-SSE response: {response_text[:500]}...")
                    try:
                        result = json.loads(response_text)
                        self.database.log_event(session_id, "executor", "sse_call_success", 
                                               f"Service: {service_config.service_name}, Task: {task.task_id}")
                        return result
                    except json.JSONDecodeError as e:
                        self.database.log_event(session_id, "executor", "sse_call_debug", 
                                               f"JSON parse error: {str(e)}")
                        return response_text
            else:
                error_text = await response.text()
                self.database.log_event(session_id, "executor", "sse_call_debug", 
                                       f"Error response: {error_text}")
                raise Exception(f"MCP SSE service returned status {response.status}: {error_text}")
    
    def _extract_service_name(self, task_desc: str) -> str:
        """Extract MCP service name from task description"""
        # Try to extract service name from task description
        if "mcp_" in task_desc.lower():
            # Try to extract service name after "mcp_"
            parts = task_desc.lower().split("mcp_")
            if len(parts) > 1:
                service_name = parts[1].split()[0]
                # Check if this service is registered
                if service_name in self.mcp_services:
                    return service_name

        # Try to find service name in description
        for service_name in self.mcp_services:
            if service_name.lower() in task_desc.lower():
                return service_name

        # Enhanced service name extraction
        # Check for common MCP service patterns
        if "fetch" in task_desc.lower() or "获取" in task_desc:
            if "fetch" in self.mcp_services:
                return "fetch"

        if "bing" in task_desc.lower() or "搜索" in task_desc:
            if "bing-cn-mcp-server" in self.mcp_services:
                return "bing-cn-mcp-server"

        # Use first available service as default
        if self.mcp_services:
            first_service = list(self.mcp_services.keys())[0]
            return first_service

        # No services available
        return "default"
    
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
        
        response = self.client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": "你是一个专业的数据处理专家，擅长各种数据分析和处理任务。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content