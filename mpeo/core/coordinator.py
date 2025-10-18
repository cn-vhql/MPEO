"""
重构后的SystemCoordinator - 使用依赖注入和事件系统
"""

import asyncio
import uuid
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

from ..models import TaskSession, TaskGraph, ExecutionResults, SystemConfig, MCPServiceConfig
from ..models.agent_config import MultiAgentConfig, OpenAIApiConfig
from ..services import DatabaseManager
from ..services.unified_mcp_manager import UnifiedMCPManager
from ..services.configuration_loader import get_config_loader
from .planner import PlannerModel
from .executor import TaskExecutor
from .output import OutputModel
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cli import HumanFeedbackInterface
from .container import DIContainer, LifetimeScope, get_container, singleton_service
from .events import EventBus, get_event_bus, Event, TaskStartedEvent, TaskCompletedEvent, TaskFailedEvent


class ISystemCoordinator:
    """系统协调器接口"""
    
    async def process_user_query(self, user_query: str) -> str:
        """处理用户查询"""
        return ""
    
    async def register_mcp_service(self, service_config: MCPServiceConfig) -> None:
        """注册MCP服务"""
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {}
    
    async def cleanup(self):
        """清理资源"""
        pass


@singleton_service(ISystemCoordinator)
class SystemCoordinator(ISystemCoordinator):
    """重构后的系统协调器 - 使用依赖注入和事件系统"""
    
    def __init__(
        self,
        container: DIContainer,
        event_bus: EventBus,
        database: DatabaseManager,
        config: SystemConfig,
        agent_config: MultiAgentConfig,
        openai_clients: Dict[str, OpenAI],
        mcp_manager: UnifiedMCPManager,
        planner: PlannerModel,
        executor: TaskExecutor,
        output_model: OutputModel,
        interface: Any  # 使用Any避免循环导入
    ):
        self.container = container
        self.event_bus = event_bus
        self.database = database
        self.config = config
        self.agent_config = agent_config
        self.openai_clients = openai_clients
        self.mcp_manager = mcp_manager
        self.planner = planner
        self.executor = executor
        self.output_model = output_model
        self.interface = interface
        
        self._mcp_manager_initialized = False
        self._setup_logging()
        self._setup_event_handlers()
        
        # 加载配置
        self._load_configuration()
    
    def _setup_logging(self):
        """设置日志系统"""
        from ..utils.logging import setup_logging
        from ..utils.paths import get_project_paths

        project_paths = get_project_paths()
        project_paths.ensure_directories()
        setup_logging(log_dir=str(project_paths.logs_dir))
    
    def _setup_event_handlers(self):
        """设置事件处理器"""
        # 订阅任务相关事件
        self.event_bus.subscribe("TaskStartedEvent", self._on_task_started)
        self.event_bus.subscribe("TaskCompletedEvent", self._on_task_completed)
        self.event_bus.subscribe("TaskFailedEvent", self._on_task_failed)
    
    async def _on_task_started(self, event: TaskStartedEvent):
        """任务开始事件处理"""
        logging.info(f"Task started: {event.task_id} - {event.task_name}")
        self.database.log_event(
            event.task_id, 
            "coordinator", 
            "task_started", 
            f"Task {event.task_name} started"
        )
    
    async def _on_task_completed(self, event: TaskCompletedEvent):
        """任务完成事件处理"""
        logging.info(f"Task completed: {event.task_id} - {event.task_name} in {event.duration:.3f}s")
        self.database.log_event(
            event.task_id,
            "coordinator", 
            "task_completed", 
            f"Task {event.task_name} completed in {event.duration:.3f}s"
        )
    
    async def _on_task_failed(self, event: TaskFailedEvent):
        """任务失败事件处理"""
        logging.error(f"Task failed: {event.task_id} - {event.task_name} - {event.error}")
        self.database.log_event(
            event.task_id,
            "coordinator", 
            "task_failed", 
            f"Task {event.task_name} failed: {event.error}"
        )
    
    async def cleanup(self):
        """清理资源"""
        if hasattr(self, 'mcp_manager') and self.mcp_manager:
            await self.mcp_manager.close()
            self._mcp_manager_initialized = False
    
    async def _ensure_mcp_manager_initialized(self):
        """确保MCP管理器已初始化"""
        if not self._mcp_manager_initialized:
            await self._initialize_mcp_manager()
            self._mcp_manager_initialized = True
    
    async def _initialize_mcp_manager(self):
        """初始化MCP管理器"""
        await self.mcp_manager.initialize()
        
        # 与组件共享MCP管理器
        await self.planner.set_mcp_manager(self.mcp_manager)
        self.executor.mcp_manager = self.mcp_manager
        
        # 从数据库加载MCP服务
        await self._load_mcp_services_from_database()
        
        # 从配置文件加载MCP服务
        await self._load_mcp_services_from_config()
        
        logging.info("MCP manager initialized and shared with components")
    
    async def _load_mcp_services_from_database(self):
        """从数据库加载MCP服务"""
        try:
            mcp_services = self.database.load_config("mcp_services", {})
            for service_name, service_config in mcp_services.items():
                try:
                    mcp_config = MCPServiceConfig.parse_obj(service_config)
                    await self.mcp_manager.register_service(mcp_config)
                    logging.info(f"MCP service '{service_name}' loaded from database")
                except Exception as e:
                    logging.error(f"Failed to load MCP service '{service_name}' from database: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load MCP services from database: {str(e)}")
    
    async def _load_mcp_services_from_config(self):
        """从配置文件加载MCP服务"""
        logging.info("MCP services configuration loading delegated to UnifiedMCPManager")
    
    def _load_configuration(self):
        """加载系统配置"""
        pass
    
    async def process_user_query(self, user_query: str) -> str:
        """处理用户查询"""
        # 创建会话
        session_id = str(uuid.uuid4())
        session = TaskSession(
            session_id=session_id,
            user_query=user_query,
            status="created"
        )
        
        self.database.log_event(session_id, "coordinator", "session_started", f"Query: {user_query[:100]}...")

        try:
            # 确保MCP管理器已初始化
            await self._ensure_mcp_manager_initialized()

            # 发布查询开始事件
            await self.event_bus.publish(Event(
                data={"session_id": session_id, "query": user_query},
                event_type="QueryProcessingStarted"
            ))

            # 阶段1: 规划阶段
            self.database.log_event(session_id, "coordinator", "planning_phase_started")
            task_graph = await self._planning_phase(user_query, session_id)
            if not task_graph:
                return "抱歉，无法为您的需求生成有效的任务计划。请提供更详细的信息。"
            
            session.task_graph = task_graph
            session.status = "planned"
            self.database.save_session(session)
            
            # 阶段2: 人工反馈阶段
            self.database.log_event(session_id, "coordinator", "feedback_phase_started")
            confirmed, final_graph = await self._feedback_phase(task_graph, user_query, session_id)
            if not confirmed:
                session.status = "cancelled"
                self.database.save_session(session)
                return "操作已取消。"
            
            session.task_graph = final_graph
            session.status = "confirmed"
            self.database.save_session(session)
            
            # 保存最终任务图
            if final_graph:
                self.database.save_task_graph(session_id, final_graph, is_final=True)
            
            # 阶段3: 执行阶段
            self.database.log_event(session_id, "coordinator", "execution_phase_started")
            if final_graph:
                execution_results = await self._execution_phase(final_graph, user_query, session_id)
            else:
                execution_results = ExecutionResults()
            session.execution_results = execution_results
            session.status = "executed"
            self.database.save_session(session)
            
            # 阶段4: 输出生成阶段
            self.database.log_event(session_id, "coordinator", "output_phase_started")
            final_output = self._output_phase(execution_results, user_query, final_graph or task_graph, session_id)
            session.final_output = final_output
            session.status = "completed"
            self.database.save_session(session)
            
            # 显示执行结果
            self.interface.display_execution_results(execution_results, session_id)
            
            # 发布查询完成事件
            await self.event_bus.publish(Event(
                data={"session_id": session_id, "query": user_query, "result": final_output},
                event_type="QueryProcessingCompleted"
            ))
            
            self.database.log_event(session_id, "coordinator", "session_completed", "All phases completed successfully")
            
            return final_output
            
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            self.database.log_event(session_id, "coordinator", "error", error_msg)
            session.status = "error"
            session.final_output = error_msg
            self.database.save_session(session)
            
            # 发布查询失败事件
            await self.event_bus.publish(Event(
                data={"session_id": session_id, "query": user_query, "error": error_msg},
                event_type="QueryProcessingFailed"
            ))
            
            return error_msg
    
    async def _planning_phase(self, user_query: str, session_id: str) -> Optional[TaskGraph]:
        """规划阶段"""
        try:
            logging.debug(f"Starting planning phase for query: {user_query}")
            
            task_graph = await self.planner.analyze_and_decompose(user_query, session_id)
            
            logging.debug(f"Task graph generated successfully with {len(task_graph.nodes)} tasks")
            
            self.database.save_task_graph(session_id, task_graph, is_final=False)
            
            self.database.log_event(session_id, "coordinator", "planning_completed", 
                                   f"Generated {len(task_graph.nodes)} tasks")
            
            return task_graph
            
        except Exception as e:
            logging.error(f"Planning phase failed: {str(e)}")
            self.database.log_event(session_id, "coordinator", "planning_error", f"Planning failed: {str(e)}")
            return None
    
    async def _feedback_phase(self, task_graph: TaskGraph, user_query: str, session_id: str) -> tuple[bool, Optional[TaskGraph]]:
        """人工反馈阶段"""
        try:
            confirmed, modified_graph = await self.interface.present_task_graph(
                task_graph, user_query, session_id, self.mcp_manager
            )
            
            if confirmed and modified_graph:
                self.database.save_task_graph(session_id, modified_graph, is_final=False)
                self.database.log_event(session_id, "coordinator", "feedback_completed", "User confirmed DAG")
                return True, modified_graph
            else:
                self.database.log_event(session_id, "coordinator", "feedback_cancelled", "User cancelled or rejected DAG")
                return False, None
                
        except Exception as e:
            self.database.log_event(session_id, "coordinator", "feedback_error", f"Feedback phase failed: {str(e)}")
            return False, None
    
    async def _execution_phase(self, task_graph: TaskGraph, user_query: str, session_id: str) -> ExecutionResults:
        """执行阶段"""
        try:
            execution_results = await self.executor.execute_task_graph(task_graph, user_query, session_id)
            
            self.database.log_event(session_id, "coordinator", "execution_completed", 
                                   f"Executed {execution_results.success_count}/{len(execution_results.execution_results)} tasks successfully")
            
            return execution_results
            
        except Exception as e:
            self.database.log_event(session_id, "coordinator", "execution_error", f"Execution failed: {str(e)}")
            return ExecutionResults()
    
    def _output_phase(self, execution_results: ExecutionResults, user_query: str, 
                     task_graph: TaskGraph, session_id: str) -> str:
        """输出生成阶段"""
        try:
            final_output = self.output_model.generate_final_output(
                execution_results, user_query, task_graph, session_id
            )
            
            self.database.log_event(session_id, "coordinator", "output_completed", "Final output generated")
            
            return final_output
            
        except Exception as e:
            self.database.log_event(session_id, "coordinator", "output_error", f"Output generation failed: {str(e)}")
            return f"生成最终输出时发生错误: {str(e)}"
    
    async def register_mcp_service(self, service_config: MCPServiceConfig):
        """注册MCP服务"""
        await self._ensure_mcp_manager_initialized()
        
        await self.mcp_manager.register_service(service_config)
        
        config_loader = get_config_loader()
        config_loader.add_mcp_service(service_config)
        
        await self.planner.refresh_mcp_tools(force_refresh=True)
        
        # 发布服务注册事件
        await self.event_bus.publish(Event(
            data={"service_name": getattr(service_config, 'service_name', 'unknown'), "service_type": getattr(service_config, 'connection_type', 'unknown')},
            event_type="MCPServiceRegistered"
        ))
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        recent_sessions = self.database.list_sessions(limit=10)
        recent_logs = self.database.get_logs(limit=20)
        
        total_sessions = len(self.database.list_sessions(limit=1000))
        recent_success_rate = 0
        if recent_sessions:
            completed_sessions = [s for s in recent_sessions if s.status == "completed"]
            recent_success_rate = len(completed_sessions) / len(recent_sessions) * 100
        
        return {
            "system_status": "running",
            "total_sessions": total_sessions,
            "recent_success_rate": f"{recent_success_rate:.1f}%",
            "registered_mcp_services": len(self.executor.mcp_services),
            "config": self.config.dict(),
            "recent_sessions": [
                {
                    "session_id": s.session_id[:8] + "...",
                    "status": s.status,
                    "created_at": s.created_at.isoformat(),
                    "user_query": s.user_query[:50] + "..." if len(s.user_query) > 50 else s.user_query
                }
                for s in recent_sessions[:5]
            ]
        }


class SystemCoordinatorFactory:
    """系统协调器工厂"""
    
    @staticmethod
    def create_coordinator(
        config: Optional[SystemConfig] = None,
        agent_config: Optional[MultiAgentConfig] = None,
        agent_config_path: Optional[str] = None,
        openai_client: Optional[OpenAI] = None
    ) -> SystemCoordinator:
        """创建系统协调器实例"""
        
        # 加载环境变量
        load_dotenv(override=True)
        
        # 获取容器和事件总线
        container = get_container()
        event_bus = get_event_bus()
        
        # 初始化配置
        config = config or SystemConfig()
        
        # 加载代理配置
        config_loader = get_config_loader()
        if agent_config:
            final_agent_config = agent_config
        elif agent_config_path:
            final_agent_config = config_loader.load_agent_config(agent_config_path)
        else:
            final_agent_config = config_loader.load_agent_config()
        
        # 从环境变量覆盖模型
        openai_model = os.getenv("OPENAI_MODEL")
        if openai_model:
            config.openai_model = openai_model
        
        # 创建OpenAI客户端
        if openai_client:
            openai_clients = {
                'planner': openai_client,
                'executor': openai_client,
                'output': openai_client
            }
        else:
            openai_clients = SystemCoordinatorFactory._create_openai_clients(final_agent_config)
        
        # 初始化数据库
        database = DatabaseManager(config.database_path)
        
        # 初始化MCP管理器
        mcp_manager = UnifiedMCPManager()
        
        # 初始化传统组件
        planner = PlannerModel(
            openai_clients['planner'],
            database,
            final_agent_config.planner
        )
        
        executor = TaskExecutor(
            openai_clients['executor'],
            database,
            config,
            final_agent_config.executor
        )
        
        output_model = OutputModel(
            openai_clients['output'],
            database,
            final_agent_config.output
        )
        
        # 延迟导入避免循环依赖
        from .cli import HumanFeedbackInterface
        interface = HumanFeedbackInterface(database)
        
        # 注册服务到容器
        container.register_singleton(DatabaseManager, instance=database)
        container.register_singleton(UnifiedMCPManager, instance=mcp_manager)
        container.register_singleton(PlannerModel, instance=planner)
        container.register_singleton(TaskExecutor, instance=executor)
        container.register_singleton(OutputModel, instance=output_model)
        container.register_singleton(HumanFeedbackInterface, instance=interface)
        container.register_singleton(SystemConfig, instance=config)
        container.register_singleton(MultiAgentConfig, instance=final_agent_config)
        
        # 创建协调器
        coordinator = SystemCoordinator(
            container=container,
            event_bus=event_bus,
            database=database,
            config=config,
            agent_config=final_agent_config,
            openai_clients=openai_clients,
            mcp_manager=mcp_manager,
            planner=planner,
            executor=executor,
            output_model=output_model,
            interface=interface
        )
        
        return coordinator
    
    @staticmethod
    def _create_openai_clients(agent_config: MultiAgentConfig) -> Dict[str, OpenAI]:
        """创建OpenAI客户端"""
        clients = {}
        agent_names = ['planner', 'executor', 'output']
        
        for agent_name in agent_names:
            agent_model_config = getattr(agent_config, agent_name)
            clients[agent_name] = SystemCoordinatorFactory._create_openai_client(agent_name, agent_model_config)
        
        return clients
    
    @staticmethod
    def _create_openai_client(agent_name: str, agent_config) -> OpenAI:
        """创建单个OpenAI客户端"""
        openai_config = agent_config.openai_config or OpenAIApiConfig()
        
        if not openai_config.api_key:
            global_api_key = os.getenv("OPENAI_API_KEY")
            if not global_api_key:
                raise ValueError(f"OPENAI_API_KEY environment variable is required for {agent_name}")
            openai_config.api_key = global_api_key
        
        if not openai_config.base_url:
            global_base_url = os.getenv("OPENAI_API_BASE")
            if global_base_url:
                openai_config.base_url = global_base_url
        
        if not openai_config.organization:
            global_organization = os.getenv("OPENAI_ORGANIZATION")
            if global_organization:
                openai_config.organization = global_organization
        
        try:
            client_kwargs = {
                'api_key': openai_config.api_key,
                'timeout': openai_config.timeout or 60,
                'max_retries': openai_config.max_retries or 3
            }
            
            if openai_config.base_url:
                client_kwargs['base_url'] = openai_config.base_url
            
            if openai_config.organization:
                client_kwargs['organization'] = openai_config.organization
            
            client = OpenAI(**client_kwargs)
            
            logging.debug(f"{agent_name} OpenAI client initialized successfully")
            
            return client
            
        except Exception as e:
            logging.error(f"Failed to initialize {agent_name} OpenAI client: {str(e)}")
            raise
