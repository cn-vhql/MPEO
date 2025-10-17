"""
Planner Model - Task decomposition and DAG generation with AgentScope support
"""

import json
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
from ..models import TaskGraph, TaskNode, TaskEdge, TaskType, DependencyType
from ..models.agent_config import AgentModelConfig
from ..services.database import DatabaseManager
from ..services.unified_mcp_manager import UnifiedMCPManager, MCPTool
from ..services.agentscope_mcp_adapter import AgentScopeMCPAdapter

# AgentScope imports
try:
    from agentscope.agents import ReActAgent
    from agentscope.models import OpenAIChatModel
    from agentscope.message import Msg
    from agentscope.memory import InMemoryMemory
    from agentscope.tool import Toolkit
    from agentscope import init as agentscope_init
    AGENTSCOPE_AVAILABLE = True
except ImportError:
    AGENTSCOPE_AVAILABLE = False
    ReActAgent = None
    OpenAIChatModel = None
    Msg = None
    InMemoryMemory = None
    Toolkit = None


class PlannerModel:
    """Planner model for task decomposition and DAG generation with AgentScope support"""

    def __init__(self, openai_client: OpenAI, database: DatabaseManager,
                 model_config: AgentModelConfig = None, enable_agentscope: bool = True):
        self.client = openai_client
        self.database = database
        self.model_config = model_config or AgentModelConfig(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=2000,
            timeout=60
        )
        self.mcp_manager: Optional[UnifiedMCPManager] = None
        self.available_mcp_tools: Dict[str, List[MCPTool]] = {}

        # AgentScope相关
        self.enable_agentscope = enable_agentscope and AGENTSCOPE_AVAILABLE
        self.agentscope_adapter: Optional[AgentScopeMCPAdapter] = None
        self.planner_agent: Optional[ReActAgent] = None
        self.agentscope_model = None
        self.memory = None

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

            # 创建记忆系统
            self.memory = InMemoryMemory()

            self.database.log_event("", "planner", "agentscope_initialized",
                                   f"AgentScope planner initialized with model: {self.model_config.model_name}")

        except Exception as e:
            self.database.log_event("", "planner", "agentscope_init_failed",
                                   f"Failed to initialize AgentScope: {str(e)}")
            self.enable_agentscope = False

    @property
    def model_name(self) -> str:
        """获取模型名称"""
        return self.model_config.model_name

    async def set_mcp_manager(self, mcp_manager: UnifiedMCPManager):
        """设置MCP服务管理器"""
        self.mcp_manager = mcp_manager

        if self.enable_agentscope:
            # 初始化AgentScope适配器
            try:
                self.agentscope_adapter = AgentScopeMCPAdapter(mcp_manager)
                await self.agentscope_adapter.initialize()

                # 创建规划智能体
                await self._create_planner_agent()

                self.database.log_event("", "planner", "agentscope_adapter_initialized",
                                       f"AgentScope adapter initialized with {len(self.agentscope_adapter.tool_functions)} tools")

            except Exception as e:
                self.database.log_event("", "planner", "agentscope_adapter_init_failed",
                                       f"Failed to initialize AgentScope adapter: {str(e)}")
                # 降级到传统模式
                self.enable_agentscope = False

        # 兼容性：刷新传统MCP工具缓存
        await self.refresh_mcp_tools()

    async def refresh_mcp_tools(self, force_refresh: bool = False):
        """刷新可用的MCP工具列表，支持AgentScope和传统模式"""
        if not self.mcp_manager:
            return

        try:
            import time

            # 检查缓存
            if not force_refresh and hasattr(self, '_tools_cache_timestamp'):
                if time.time() - self._tools_cache_timestamp < 300:  # 5分钟缓存
                    return

            # AgentScope模式：刷新适配器
            if self.enable_agentscope and self.agentscope_adapter:
                await self.agentscope_adapter.refresh_tools()
                # 转换为传统格式以保持兼容性
                self._convert_agentscope_tools_to_legacy_format()

            # 传统模式：直接获取MCP工具
            else:
                self.available_mcp_tools = await self.mcp_manager.get_available_tools()

            self._tools_cache_timestamp = time.time()

        except Exception as e:
            self.database.log_event("", "planner", "tools_refresh_failed", f"Failed to refresh tools: {str(e)}")
            if not hasattr(self, 'available_mcp_tools'):
                self.available_mcp_tools = {}

    def _convert_agentscope_tools_to_legacy_format(self):
        """将AgentScope工具转换为传统格式以保持兼容性"""
        if not self.agentscope_adapter:
            return

        try:
            self.available_mcp_tools = {}

            # 按服务名分组工具
            for tool_name, tool_func in self.agentscope_adapter.tool_functions.items():
                service_name = tool_func.service_name

                if service_name not in self.available_mcp_tools:
                    self.available_mcp_tools[service_name] = []

                # 创建兼容的工具对象
                legacy_tool = LegacyMCPTool(
                    name=tool_func.mcp_tool.name,
                    description=tool_func.mcp_tool.description,
                    input_schema=tool_func.mcp_tool.input_schema
                )

                self.available_mcp_tools[service_name].append(legacy_tool)

        except Exception as e:
            self.database.log_event("", "planner", "tool_conversion_failed",
                                   f"Failed to convert AgentScope tools: {str(e)}")

    def invalidate_tools_cache(self):
        """使工具缓存失效，强制下次刷新"""
        if hasattr(self, '_tools_cache_timestamp'):
            delattr(self, '_tools_cache_timestamp')

        # 清空AgentScope适配器缓存
        if self.agentscope_adapter:
            self.agentscope_adapter.clear_cache()

    async def _create_planner_agent(self):
        """创建AgentScope规划智能体"""
        if not self.enable_agentscope or not self.agentscope_adapter:
            return

        try:
            # 创建工具包
            toolkit = await self.agentscope_adapter.create_contextual_toolkit("任务规划工具发现")

            # 创建ReAct智能体
            self.planner_agent = ReActAgent(
                name="MPEO规划师",
                sys_prompt=self._get_planning_system_prompt(),
                model=self.agentscope_model,
                memory=self.memory,
                toolkit=toolkit,
                max_iters=3  # 限制迭代次数，提高效率
            )

            self.database.log_event("", "planner", "planner_agent_created", "AgentScope planner agent created")

        except Exception as e:
            self.database.log_event("", "planner", "planner_agent_creation_failed",
                                   f"Failed to create planner agent: {str(e)}")
            raise

    def _get_planning_system_prompt(self) -> str:
        """获取规划智能体的系统提示"""
        return """
你是一个专业的AI任务规划专家，专门负责将复杂的用户需求分解为可执行的任务图。

你的主要职责：
1. 分析用户需求，理解核心目标和约束条件
2. 发现和评估可用的MCP工具
3. 将需求分解为具体的、可执行的任务
4. 定义任务间的依赖关系，确保执行顺序合理
5. 生成结构化的任务图（DAG）

工具使用指导：
- 使用工具发现功能来了解当前可用的MCP服务
- 根据任务需求选择合适的工具
- 在任务描述中明确指定使用的服务名和工具名
- 确保任务描述具体、可执行

任务分解原则：
1. 任务粒度适中，每个任务可独立完成
2. 优先使用MCP工具处理外部数据获取
3. 合理设置任务优先级（1-5，数字越大优先级越高）
4. 避免循环依赖
5. 确保任务覆盖完整需求

输出格式：
严格按照JSON格式输出任务列表和依赖关系。
"""

    def get_tools_summary(self) -> str:
        """获取MCP工具的摘要信息，用于prompt"""
        if not self.available_mcp_tools:
            return "暂无可用MCP工具"

        summary = []
        for service_name, tools in self.available_mcp_tools.items():
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

    def find_relevant_tools(self, query: str) -> List[Dict[str, Any]]:
        """根据查询查找相关的MCP工具"""
        relevant_tools = []
        query_lower = query.lower()

        for service_name, tools in self.available_mcp_tools.items():
            for tool in tools:
                # 简单的关键词匹配
                relevance_score = 0

                # 检查工具名称匹配
                if tool.name.lower() in query_lower:
                    relevance_score += 10

                # 检查工具描述匹配
                desc_lower = tool.description.lower()
                query_words = query_lower.split()
                for word in query_words:
                    if word in desc_lower:
                        relevance_score += 3

                # 检查特定关键词匹配
                if any(keyword in query_lower for keyword in ["fetch", "get", "获取", "下载"]) and "fetch" in tool.name.lower():
                    relevance_score += 8
                elif any(keyword in query_lower for keyword in ["search", "查找", "搜索"]) and "search" in tool.name.lower():
                    relevance_score += 8
                elif any(keyword in query_lower for keyword in ["process", "处理"]) and "process" in tool.name.lower():
                    relevance_score += 8

                if relevance_score > 0:
                    relevant_tools.append({
                        "service_name": service_name,
                        "tool": tool,
                        "relevance_score": relevance_score
                    })

        # 按相关性排序
        relevant_tools.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_tools

    # 保持向后兼容的方法
    def update_mcp_services(self, mcp_services: List[str]):
        """Update available MCP services for planning (deprecated, use set_mcp_manager instead)"""
        print(f"[DEBUG] Planner - update_mcp_services called with: {mcp_services}")
        print(f"[DEBUG] Planner - This method is deprecated, please use set_mcp_manager instead")
    
    async def analyze_and_decompose(self, user_query: str, session_id: str) -> TaskGraph:
        """
        使用AgentScope进行需求分析和任务分解
        """
        self.database.log_event(session_id, "planner", "start_analysis", f"Query: {user_query[:100]}...")

        try:
            # 优先使用AgentScope智能体
            if self.enable_agentscope and self.planner_agent:
                return await self._agentscope_analyze_and_decompose(user_query, session_id)
            else:
                # 降级到传统方式
                return await self._legacy_analyze_and_decompose(user_query, session_id)

        except Exception as e:
            self.database.log_event(session_id, "planner", "analysis_error", f"Analysis failed: {str(e)}")
            # 如果AgentScope失败，尝试传统方式
            if self.enable_agentscope:
                self.database.log_event(session_id, "planner", "fallback_to_legacy",
                                       "Falling back to legacy planning method")
                return await self._legacy_analyze_and_decompose(user_query, session_id)
            else:
                raise

    async def _legacy_analyze_and_decompose(self, user_query: str, session_id: str) -> TaskGraph:
        """传统的需求分析和任务分解（向后兼容）"""
        try:
            # Initialize helper classes
            from .planner_helpers import QueryAnalyzer, TaskGenerator, DependencyGenerator

            query_analyzer = QueryAnalyzer(self.client, self.model_config, self.database)
            task_generator = TaskGenerator(self.client, self.model_config, self.database)
            dependency_generator = DependencyGenerator(self.client, self.model_config, self.database)

            # Step 1: Analyze the query and extract requirements
            analysis_result = await query_analyzer.analyze_query(user_query, session_id)

            # Step 2: Generate task decomposition
            tasks = await task_generator.generate_tasks(analysis_result, self.available_mcp_tools, session_id)

            # Step 3: Generate dependencies between tasks
            dependencies = await dependency_generator.generate_dependencies(tasks, analysis_result, session_id)

            # Step 4: Create task graph
            task_graph = TaskGraph(nodes=tasks, edges=dependencies)

            # Validate the graph
            if task_graph.has_cycle():
                self.database.log_event(session_id, "planner", "cycle_detected", "Generated DAG has cycles, regenerating...")
                return await self._regenerate_graph(user_query, session_id)

            self.database.log_event(session_id, "planner", "graph_generated",
                                  f"Generated {len(tasks)} tasks with {len(dependencies)} dependencies")

            return task_graph

        except Exception as e:
            self.database.log_event(session_id, "planner", "error", f"Failed to generate task graph: {str(e)}")
            raise
    
    def _analyze_query(self, user_query: str, session_id: str) -> Dict[str, Any]:
        """Analyze user query to extract requirements and constraints"""
        print(f"[DEBUG] Planner - Analyzing query: {user_query}")
        print(f"[DEBUG] Planner - Using model: {self.model_name}")
        
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
            print(f"[DEBUG] Planner - Making API call to analyze query...")
            print(f"[DEBUG] Planner - Using model: {self.model_config.model_name}, temperature: {self.model_config.temperature}")

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
            print(f"[DEBUG] Planner - API call successful, response status: {response.choices[0].finish_reason if response.choices else 'No choices'}")
            
            analysis_text = response.choices[0].message.content
            print(f"[DEBUG] Planner - Analysis response length: {len(analysis_text)} characters")
            print(f"[DEBUG] Planner - Analysis preview: {analysis_text[:100]}...")
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
    
    async def _generate_tasks(self, analysis_result: Dict[str, Any], session_id: str) -> List[TaskNode]:
        """Generate task nodes based on analysis"""
        print(f"[DEBUG] Planner - Generating tasks from analysis")
        print(f"[DEBUG] Planner - Analysis result: {analysis_result}")

        # 刷新MCP工具信息
        await self.refresh_mcp_tools()

        # 获取相关工具
        user_query = analysis_result.get("core_objective", "")
        relevant_tools = self.find_relevant_tools(user_query)

        print(f"[DEBUG] Planner - Found {len(relevant_tools)} relevant MCP tools")
        for tool_info in relevant_tools[:3]:  # 只显示前3个最相关的
            tool = tool_info["tool"]
            print(f"[DEBUG] Planner -   - {tool_info['service_name']}.{tool.name} (score: {tool_info['relevance_score']})")

        # 构建工具信息字符串
        tools_summary = self.get_tools_summary()

        # 如果有相关工具，添加详细信息
        if relevant_tools:
            tools_summary += "\n\n特别相关的工具："
            for tool_info in relevant_tools[:5]:  # 只显示前5个最相关的
                tool = tool_info["tool"]
                tools_summary += f"\n- {tool_info['service_name']}.{tool.name}: {tool.description}"
                tools_summary += f"\n  格式化信息:\n{tool.format_for_llm()}"

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
        5. 根据可用的MCP工具合理规划任务：
           - 如果需要网络数据获取，使用mcp调用类型的任务
           - 如果需要复杂计算，使用本地计算类型的任务
           - 如果需要数据处理，使用数据处理类型的任务
        6. 在任务描述中明确指出使用的MCP服务名和工具名，格式为："使用[服务名]的[工具名]获取..."
        7. 优先使用相关性高的MCP工具
        8. 避免创建无法执行的任务，确保任务描述与可用工具匹配
        9. 注意工具的参数要求，在任务描述中明确需要的参数
        10. **特别注意**：对于fetch工具，必须在任务描述中包含要获取的URL，例如："使用fetch服务的fetch工具获取URL: https://example.com 的内容"
        11. **URL处理**：如果用户查询中包含URL，直接使用该URL；如果需要生成URL，提供完整的URL地址
        """
        
        try:
            print(f"[DEBUG] Planner - Making API call to generate tasks...")
            print(f"[DEBUG] Planner - Using model: {self.model_config.model_name}, temperature: {self.model_config.temperature}")

            messages = [
                {"role": "system", "content": self.model_config.system_prompt or "你是一个专业的任务分解专家，擅长将复杂需求分解为可执行的任务，并且能够根据可用的MCP服务合理规划任务。"},
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
            print(f"[DEBUG] Planner - Task generation API call successful")
            
            response_text = response.choices[0].message.content
            print(f"[DEBUG] Planner - Task generation response length: {len(response_text)} characters")
            print(f"[DEBUG] Planner - Task generation preview: {response_text[:100]}...")
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
    
    async def _regenerate_graph(self, user_query: str, session_id: str) -> TaskGraph:
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

    # AgentScope增强的分析和分解方法
    async def _agentscope_analyze_and_decompose(self, user_query: str, session_id: str) -> TaskGraph:
        """使用AgentScope智能体进行分析和分解"""
        try:
            # 确保工具是最新的
            await self.refresh_mcp_tools()

            # 创建上下文工具包
            contextual_toolkit = await self.agentscope_adapter.create_contextual_toolkit(user_query)

            # 更新智能体的工具包
            self.planner_agent.toolkit = contextual_toolkit

            # 构建规划提示
            planning_prompt = self._build_planning_prompt(user_query)

            # 创建输入消息
            input_msg = Msg(
                name="user",
                content=planning_prompt,
                role="user"
            )

            # 调用AgentScope智能体
            response = await self.planner_agent.reply(input_msg)

            # 解析响应并生成任务图
            task_graph = await self._parse_agentscope_response(response.content, user_query, session_id)

            # 验证任务图
            if task_graph.has_cycle():
                self.database.log_event(session_id, "planner", "cycle_detected", "Generated DAG has cycles, regenerating...")
                return await self._regenerate_graph(user_query, session_id)

            self.database.log_event(session_id, "planner", "agentscope_graph_generated",
                                   f"Generated {len(task_graph.nodes)} tasks with {len(task_graph.edges)} dependencies")

            return task_graph

        except Exception as e:
            self.database.log_event(session_id, "planner", "agentscope_planning_failed",
                                   f"AgentScope planning failed: {str(e)}")
            raise

    def _build_planning_prompt(self, user_query: str) -> str:
        """构建规划提示"""
        tools_summary = self.get_tools_summary()

        return f"""
请为以下用户查询生成详细的任务执行计划：

用户查询：{user_query}

当前可用的MCP工具：
{tools_summary}

请按照以下步骤进行分析和规划：

1. **需求分析**：理解用户的核心目标、约束条件和预期输出
2. **工具评估**：使用工具发现功能了解当前可用的MCP服务，评估哪些工具对完成任务有帮助
3. **任务分解**：将复杂需求分解为具体的、可执行的任务
4. **依赖分析**：定义任务间的逻辑依赖关系
5. **输出格式化**：生成标准的JSON格式任务图

输出要求：
- 严格按照JSON格式输出
- 任务描述要具体明确，包含使用的服务名和工具名
- 合理设置优先级（1-5，数字越大优先级越高）
- 确保任务间依赖关系合理，避免循环依赖

JSON输出格式：
{{
    "analysis": {{
        "core_objective": "核心目标",
        "domain": "领域类型",
        "complexity": "复杂度",
        "key_requirements": ["需求1", "需求2"]
    }},
    "tasks": [
        {{
            "task_id": "T1",
            "task_desc": "具体任务描述，包含使用的MCP服务名和工具名",
            "task_type": "任务类型（本地计算/mcp调用/数据处理）",
            "expected_output": "预期输出",
            "priority": 3
        }}
    ],
    "dependencies": [
        {{
            "from_task_id": "前置任务ID",
            "to_task_id": "依赖任务ID",
            "dependency_type": "依赖类型（数据依赖/结果依赖）"
        }}
    ]
}}

重要提醒：
1. 请先使用工具发现功能查看当前可用的MCP服务
2. 在任务描述中明确指出使用的MCP服务名和工具名
3. 确保任务描述与可用工具匹配
4. 对于fetch工具，必须在任务描述中包含完整的URL地址
"""

    async def _parse_agentscope_response(self, response_content: str, user_query: str, session_id: str) -> TaskGraph:
        """解析AgentScope智能体的响应"""
        try:
            # 提取JSON部分
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1

            if json_start == -1 or json_end <= json_start:
                raise ValueError("No valid JSON found in response")

            json_content = response_content[json_start:json_end]
            data = json.loads(json_content)

            # 提取任务列表
            tasks_data = data.get("tasks", [])
            tasks = []

            for i, task_data in enumerate(tasks_data):
                task_node = TaskNode(
                    task_id=task_data.get("task_id", f"T{i+1}"),
                    task_desc=task_data.get("task_desc", f"任务{i+1}"),
                    task_type=TaskType(task_data.get("task_type", "本地计算")),
                    expected_output=task_data.get("expected_output", "完成处理"),
                    priority=task_data.get("priority", 3)
                )
                tasks.append(task_node)

            # 提取依赖关系
            deps_data = data.get("dependencies", [])
            dependencies = []

            for dep_data in deps_data:
                try:
                    task_edge = TaskEdge(
                        from_task_id=dep_data.get("from_task_id"),
                        to_task_id=dep_data.get("to_task_id"),
                        dependency_type=DependencyType(dep_data.get("dependency_type", "结果依赖"))
                    )
                    dependencies.append(task_edge)
                except (ValueError, KeyError):
                    # 跳过无效依赖
                    continue

            # 创建任务图
            task_graph = TaskGraph(nodes=tasks, edges=dependencies)

            self.database.log_event(session_id, "planner", "agentscope_response_parsed",
                                   f"Parsed {len(tasks)} tasks and {len(dependencies)} dependencies")

            return task_graph

        except json.JSONDecodeError as e:
            self.database.log_event(session_id, "planner", "agentscope_json_parse_error",
                                   f"Failed to parse JSON: {str(e)}")
            raise ValueError(f"Invalid JSON in AgentScope response: {str(e)}")
        except Exception as e:
            self.database.log_event(session_id, "planner", "agentscope_response_parse_error",
                                   f"Failed to parse response: {str(e)}")
            raise

    async def cleanup(self):
        """清理资源"""
        try:
            if self.agentscope_adapter:
                await self.agentscope_adapter.close()
                self.agentscope_adapter = None

            self.planner_agent = None
            self.memory = None

            self.database.log_event("", "planner", "cleanup_completed", "AgentScope planner cleaned up")

        except Exception as e:
            self.database.log_event("", "planner", "cleanup_error", f"Cleanup error: {str(e)}")


# 兼容性类：用于向后兼容
class LegacyMCPTool:
    """传统MCP工具类，保持向后兼容"""
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def format_for_llm(self) -> str:
        """格式化工具信息供LLM使用"""
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }