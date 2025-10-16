"""
规划器辅助模块 - 拆分长方法
"""

import json
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI

from ..models import TaskGraph, TaskNode, TaskEdge, TaskType, DependencyType
from ..models.agent_config import AgentModelConfig
from ..services.database import DatabaseManager


class QueryAnalyzer:
    """查询分析器"""

    def __init__(self, client: OpenAI, model_config: AgentModelConfig, database: DatabaseManager):
        self.client = client
        self.model_config = model_config
        self.database = database

    async def analyze_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """分析用户查询"""

        prompt = f"""
        请分析以下用户需求，提取关键信息并结构化输出：

        用户需求：{user_query}

        请按照以下JSON格式输出分析结果：
        {{
            "core_objective": "核心目标描述",
            "domain": "领域类型（如：数据分析、报告生成、信息查询等）",
            "complexity": "复杂度等级（简单/中等/复杂）",
            "constraints": [
                "约束条件1",
                "约束条件2"
            ],
            "expected_output_format": "预期输出格式",
            "required_data_sources": [
                "数据源1",
                "数据源2"
            ],
            "key_requirements": [
                "关键需求1",
                "关键需求2"
            ]
        }}
        """

        try:
            messages = [
                {"role": "system", "content": self.model_config.system_prompt or "你是一个专业的需求分析师，擅长将用户需求结构化。"},
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

            analysis_text = response.choices[0].message.content
            self.database.log_event(session_id, "planner", "query_analyzed", f"Analysis: {analysis_text[:200]}...")

        except Exception as e:
            print(f"[ERROR] Planner - API call failed: {str(e)}")
            raise

        try:
            # Extract JSON from response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_content = analysis_text[json_start:json_end]
                return json.loads(json_content)
            else:
                # Fallback if JSON parsing fails
                return self._create_fallback_analysis(user_query)
        except json.JSONDecodeError:
            self.database.log_event(session_id, "planner", "json_parse_error", "Failed to parse analysis JSON")
            return self._create_fallback_analysis(user_query)

    def _create_fallback_analysis(self, user_query: str) -> Dict[str, Any]:
        """创建备用分析结果"""
        return {
            "core_objective": user_query,
            "domain": "通用",
            "complexity": "中等",
            "constraints": [],
            "expected_output_format": "文本",
            "required_data_sources": [],
            "key_requirements": [user_query]
        }


class TaskGenerator:
    """任务生成器"""

    def __init__(self, client: OpenAI, model_config: AgentModelConfig, database: DatabaseManager):
        self.client = client
        self.model_config = model_config
        self.database = database

    async def generate_tasks(self, analysis_result: Dict[str, Any], available_mcp_tools: Dict[str, Any],
                           session_id: str) -> List[TaskNode]:
        """生成任务节点"""

        # 构建工具信息字符串
        tools_summary = self._build_tools_summary(available_mcp_tools)

        prompt = f"""
        基于以下需求分析结果和可用的MCP工具，将需求分解为具体的可执行任务：

        需求分析：
        {json.dumps(analysis_result, ensure_ascii=False, indent=2)}

        可用的MCP工具：
        {tools_summary}

        请按照以下JSON格式输出任务列表：
        {{
            "tasks": [
                {{
                    "task_desc": "任务具体描述，如果使用MCP工具请在描述中包含服务名和工具名",
                    "task_type": "任务类型（本地计算/mcp调用/数据处理）",
                    "expected_output": "预期输出描述",
                    "priority": 优先级(1-5)
                }}
            ]
        }}

        重要指导原则：
        1. 任务粒度要适中，每个任务应该是可独立完成的
        2. 任务描述要具体明确
        3. 合理分配优先级（1-5，数字越大优先级越高）
        4. 确保所有任务覆盖完整需求
        5. 根据可用的MCP工具合理规划任务
        6. 在任务描述中明确指出使用的MCP服务名和工具名
        7. 优先使用相关性高的MCP工具
        8. 避免创建无法执行的任务
        9. 注意工具的参数要求
        10. 对于fetch工具，必须在任务描述中包含要获取的URL
        11. 如果用户查询中包含URL，直接使用该URL
        """

        try:
            messages = [
                {"role": "system", "content": self.model_config.system_prompt or "你是一个专业的任务分解专家，擅长将复杂需求分解为可执行的任务。"},
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

            response_text = response.choices[0].message.content
            self.database.log_event(session_id, "planner", "tasks_generated", f"Generated tasks: {response_text[:200]}...")

        except Exception as e:
            print(f"[ERROR] Planner - Task generation API call failed: {str(e)}")
            raise

        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                tasks_data = json.loads(json_content)

                # Convert to TaskNode objects
                tasks = []
                for i, task_data in enumerate(tasks_data.get("tasks", [])):
                    task_node = TaskNode(
                        task_id=f"T{i+1}",
                        task_desc=task_data.get("task_desc", f"任务{i+1}"),
                        task_type=TaskType(task_data.get("task_type", "本地计算")),
                        expected_output=task_data.get("expected_output", "完成处理"),
                        priority=task_data.get("priority", 3)
                    )
                    tasks.append(task_node)

                return tasks
            else:
                # Fallback: create a single task
                return [self._create_fallback_task(analysis_result)]

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.database.log_event(session_id, "planner", "task_generation_error", f"Failed to generate tasks: {str(e)}")
            # Fallback: create a single task
            return [self._create_fallback_task(analysis_result)]

    def _build_tools_summary(self, available_mcp_tools: Dict[str, Any]) -> str:
        """构建工具信息摘要"""
        if not available_mcp_tools:
            return "暂无可用MCP工具"

        summary = []
        for service_name, tools in available_mcp_tools.items():
            if tools:
                summary.append(f"服务 '{service_name}' 提供以下工具:")
                for tool in tools:
                    tool_info = f"  - {tool.name}: {tool.description}"
                    if tool.input_schema.get("properties"):
                        params = []
                        for param_name, param_info in tool.input_schema["properties"].items():
                            required = param_name in tool.input_schema.get("required", [])
                            req_str = " (必需)" if required else " (可选)"
                            params.append(f"{param_name}{req_str}")
                        if params:
                            tool_info += f"\n    参数: {', '.join(params)}"
                    summary.append(tool_info)

        return "\n".join(summary) if summary else "暂无可用MCP工具"

    def _create_fallback_task(self, analysis_result: Dict[str, Any]) -> TaskNode:
        """创建备用任务"""
        return TaskNode(
            task_id="T1",
            task_desc=analysis_result.get("core_objective", "处理用户需求"),
            task_type=TaskType.LOCAL_COMPUTE,
            expected_output=analysis_result.get("expected_output_format", "处理结果"),
            priority=3
        )


class DependencyGenerator:
    """依赖关系生成器"""

    def __init__(self, client: OpenAI, model_config: AgentModelConfig, database: DatabaseManager):
        self.client = client
        self.model_config = model_config
        self.database = database

    async def generate_dependencies(self, tasks: List[TaskNode], analysis_result: Dict[str, Any],
                                  session_id: str) -> List[TaskEdge]:
        """生成任务依赖关系"""
        if len(tasks) <= 1:
            return []

        prompt = f"""
        基于以下任务列表和需求分析，定义任务之间的依赖关系：

        任务列表：
        {json.dumps([{"task_id": task.task_id, "task_desc": task.task_desc, "task_type": task.task_type.value} for task in tasks], ensure_ascii=False, indent=2)}

        需求分析：
        {json.dumps(analysis_result, ensure_ascii=False, indent=2)}

        请按照以下JSON格式输出依赖关系：
        {{
            "dependencies": [
                {{
                    "from_task_id": "前置任务ID",
                    "to_task_id": "依赖任务ID",
                    "dependency_type": "依赖类型（数据依赖/结果依赖）"
                }}
            ]
        }}

        注意：
        1. 只定义确实需要的依赖关系
        2. 避免循环依赖
        3. 确保依赖关系合理且必要
        4. 如果任务可以并行执行，不要创建依赖关系
        """

        messages = [
            {"role": "system", "content": self.model_config.system_prompt or "你是一个专业的任务调度专家，擅长定义任务间的合理依赖关系。"},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model_config.model_name,
            messages=messages,
            temperature=self.model_config.temperature * 0.7,  # 依赖关系生成使用更低的温度
            max_tokens=self.model_config.max_tokens,
            top_p=self.model_config.top_p,
            frequency_penalty=self.model_config.frequency_penalty,
            presence_penalty=self.model_config.presence_penalty
        )

        response_text = response.choices[0].message.content
        self.database.log_event(session_id, "planner", "dependencies_generated", f"Generated dependencies: {response_text[:200]}...")

        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                deps_data = json.loads(json_content)

                # Convert to TaskEdge objects
                dependencies = []
                for dep_data in deps_data.get("dependencies", []):
                    try:
                        task_edge = TaskEdge(
                            from_task_id=dep_data.get("from_task_id"),
                            to_task_id=dep_data.get("to_task_id"),
                            dependency_type=DependencyType(dep_data.get("dependency_type", "结果依赖"))
                        )
                        dependencies.append(task_edge)
                    except (ValueError, KeyError):
                        # Skip invalid dependencies
                        continue

                return dependencies
            else:
                # Fallback: create simple sequential dependencies
                return self._create_sequential_dependencies(tasks)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.database.log_event(session_id, "planner", "dependency_generation_error", f"Failed to generate dependencies: {str(e)}")
            # Fallback: create simple sequential dependencies
            return self._create_sequential_dependencies(tasks)

    def _create_sequential_dependencies(self, tasks: List[TaskNode]) -> List[TaskEdge]:
        """创建顺序依赖关系"""
        dependencies = []
        for i in range(len(tasks) - 1):
            dependencies.append(TaskEdge(
                from_task_id=tasks[i].task_id,
                to_task_id=tasks[i+1].task_id,
                dependency_type=DependencyType.RESULT_DEPENDENCY
            ))
        return dependencies