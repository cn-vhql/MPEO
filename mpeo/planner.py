"""
Planner Model - Task decomposition and DAG generation
"""

import json
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .models import TaskGraph, TaskNode, TaskEdge, TaskType, DependencyType
from .database import DatabaseManager


class PlannerModel:
    """Planner model for task decomposition and DAG generation"""
    
    def __init__(self, openai_client: OpenAI, database: DatabaseManager, model_name: str = "gpt-3.5-turbo"):
        self.client = openai_client
        self.database = database
        self.model_name = model_name
    
    def analyze_and_decompose(self, user_query: str, session_id: str) -> TaskGraph:
        """
        Analyze user query and decompose into task graph
        
        Args:
            user_query: User's original query/question
            session_id: Session identifier for logging
            
        Returns:
            TaskGraph: Generated task graph
        """
        self.database.log_event(session_id, "planner", "start_analysis", f"Query: {user_query[:100]}...")
        
        try:
            # Step 1: Analyze the query and extract requirements
            analysis_result = self._analyze_query(user_query, session_id)
            
            # Step 2: Generate task decomposition
            tasks = self._generate_tasks(analysis_result, session_id)
            
            # Step 3: Generate dependencies between tasks
            dependencies = self._generate_dependencies(tasks, analysis_result, session_id)
            
            # Step 4: Create task graph
            task_graph = TaskGraph(nodes=tasks, edges=dependencies)
            
            # Validate the graph
            if task_graph.has_cycle():
                self.database.log_event(session_id, "planner", "cycle_detected", "Generated DAG has cycles, regenerating...")
                return self._regenerate_graph(user_query, session_id)
            
            self.database.log_event(session_id, "planner", "graph_generated", 
                                  f"Generated {len(tasks)} tasks with {len(dependencies)} dependencies")
            
            return task_graph
            
        except Exception as e:
            self.database.log_event(session_id, "planner", "error", f"Failed to generate task graph: {str(e)}")
            raise
    
    def _analyze_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """Analyze user query to extract requirements and constraints"""
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
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的需求分析师，擅长将用户需求结构化。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        analysis_text = response.choices[0].message.content
        self.database.log_event(session_id, "planner", "query_analyzed", f"Analysis: {analysis_text[:200]}...")
        
        try:
            # Extract JSON from response
            json_start = analysis_text.find('{')
            json_end = analysis_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_content = analysis_text[json_start:json_end]
                return json.loads(json_content)
            else:
                # Fallback if JSON parsing fails
                return {
                    "core_objective": user_query,
                    "domain": "通用",
                    "complexity": "中等",
                    "constraints": [],
                    "expected_output_format": "文本",
                    "required_data_sources": [],
                    "key_requirements": [user_query]
                }
        except json.JSONDecodeError:
            self.database.log_event(session_id, "planner", "json_parse_error", "Failed to parse analysis JSON")
            return {
                "core_objective": user_query,
                "domain": "通用",
                "complexity": "中等",
                "constraints": [],
                "expected_output_format": "文本",
                "required_data_sources": [],
                "key_requirements": [user_query]
            }
    
    def _generate_tasks(self, analysis_result: Dict[str, Any], session_id: str) -> List[TaskNode]:
        """Generate task nodes based on analysis"""
        prompt = f"""
        基于以下需求分析结果，将需求分解为具体的可执行任务：

        需求分析：
        {json.dumps(analysis_result, ensure_ascii=False, indent=2)}

        请按照以下JSON格式输出任务列表：
        {{
            "tasks": [
                {{
                    "task_desc": "任务具体描述",
                    "task_type": "任务类型（本地计算/mcp调用/数据处理）",
                    "expected_output": "预期输出描述",
                    "priority": 优先级(1-5)
                }}
            ]
        }}

        注意：
        1. 任务粒度要适中，每个任务应该是可独立完成的
        2. 任务描述要具体明确
        3. 合理分配优先级（1-5，数字越大优先级越高）
        4. 确保所有任务覆盖完整需求
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的任务分解专家，擅长将复杂需求分解为可执行的任务。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content
        self.database.log_event(session_id, "planner", "tasks_generated", f"Generated tasks: {response_text[:200]}...")
        
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
                return [TaskNode(
                    task_id="T1",
                    task_desc=analysis_result.get("core_objective", "处理用户需求"),
                    task_type=TaskType.LOCAL_COMPUTE,
                    expected_output=analysis_result.get("expected_output_format", "处理结果"),
                    priority=3
                )]
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.database.log_event(session_id, "planner", "task_generation_error", f"Failed to generate tasks: {str(e)}")
            # Fallback: create a single task
            return [TaskNode(
                task_id="T1",
                task_desc=analysis_result.get("core_objective", "处理用户需求"),
                task_type=TaskType.LOCAL_COMPUTE,
                expected_output=analysis_result.get("expected_output_format", "处理结果"),
                priority=3
            )]
    
    def _generate_dependencies(self, tasks: List[TaskNode], analysis_result: Dict[str, Any], session_id: str) -> List[TaskEdge]:
        """Generate dependencies between tasks"""
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
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的任务调度专家，擅长定义任务间的合理依赖关系。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
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
                dependencies = []
                for i in range(len(tasks) - 1):
                    dependencies.append(TaskEdge(
                        from_task_id=tasks[i].task_id,
                        to_task_id=tasks[i+1].task_id,
                        dependency_type=DependencyType.RESULT_DEPENDENCY
                    ))
                return dependencies
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.database.log_event(session_id, "planner", "dependency_generation_error", f"Failed to generate dependencies: {str(e)}")
            # Fallback: create simple sequential dependencies
            dependencies = []
            for i in range(len(tasks) - 1):
                dependencies.append(TaskEdge(
                    from_task_id=tasks[i].task_id,
                    to_task_id=tasks[i+1].task_id,
                    dependency_type=DependencyType.RESULT_DEPENDENCY
                ))
            return dependencies
    
    def _regenerate_graph(self, user_query: str, session_id: str) -> TaskGraph:
        """Regenerate task graph if cycle detected"""
        self.database.log_event(session_id, "planner", "regenerating_graph", "Cycle detected, regenerating with sequential execution")
        
        # Create a simple sequential task graph as fallback
        tasks = [
            TaskNode(
                task_id="T1",
                task_desc="分析用户需求",
                task_type=TaskType.LOCAL_COMPUTE,
                expected_output="需求分析结果",
                priority=5
            ),
            TaskNode(
                task_id="T2",
                task_desc="执行核心处理",
                task_type=TaskType.LOCAL_COMPUTE,
                expected_output="处理结果",
                priority=3
            ),
            TaskNode(
                task_id="T3",
                task_desc="生成最终输出",
                task_type=TaskType.LOCAL_COMPUTE,
                expected_output="最终答案",
                priority=1
            )
        ]
        
        dependencies = [
            TaskEdge(
                from_task_id="T1",
                to_task_id="T2",
                dependency_type=DependencyType.RESULT_DEPENDENCY
            ),
            TaskEdge(
                from_task_id="T2",
                to_task_id="T3",
                dependency_type=DependencyType.RESULT_DEPENDENCY
            )
        ]
        
        return TaskGraph(nodes=tasks, edges=dependencies)