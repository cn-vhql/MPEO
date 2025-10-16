"""
System Coordinator - Orchestrates all components of the multi-model system
"""

import asyncio
import uuid
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
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
from ..interfaces import HumanFeedbackInterface


class SystemCoordinator:
    """Main system coordinator that orchestrates all components"""
    
    def _setup_logging(self):
        """Setup logging system with file and console output - only once"""
        # Use the centralized logging utility
        from ..utils.logging import setup_logging
        from ..utils.paths import get_project_paths

        # Ensure project structure exists
        project_paths = get_project_paths()
        project_paths.ensure_directories()

        # Setup logging with ensured directory
        setup_logging(log_dir=str(project_paths.logs_dir))

    def __init__(self, config: Optional[SystemConfig] = None,
                 agent_config: Optional[MultiAgentConfig] = None,
                 agent_config_path: Optional[str] = None):
        # Load environment variables, overriding existing ones
        load_dotenv(override=True)

        # Initialize configuration first
        self.config = config or SystemConfig()

        # Setup logging system only once, after config is available
        self._setup_logging()

        # Load agent configuration using unified configuration loader
        config_loader = get_config_loader()
        if agent_config:
            self.agent_config = agent_config
        elif agent_config_path:
            self.agent_config = config_loader.load_agent_config(agent_config_path)
        else:
            # Try to load from default path, otherwise use default config
            self.agent_config = config_loader.load_agent_config()

        # Always read OPENAI_MODEL from environment and override if present
        openai_model = os.getenv("OPENAI_MODEL")
        if openai_model:
            self.config.openai_model = openai_model
            logging.debug(f"Overriding model from environment: {openai_model}")

        # Initialize OpenAI clients for each agent
        self.openai_clients = self._initialize_openai_clients()

        # Initialize database
        self.database = DatabaseManager(self.config.database_path)

        # Initialize MCP service manager
        self.mcp_manager = UnifiedMCPManager()

        # Initialize components with individual model configurations and clients
        self.planner = PlannerModel(
            self.openai_clients['planner'],
            self.database,
            self.agent_config.planner
        )
        self.executor = TaskExecutor(
            self.openai_clients['executor'],
            self.database,
            self.config,
            self.agent_config.executor
        )
        self.output_model = OutputModel(
            self.openai_clients['output'],
            self.database,
            self.agent_config.output
        )
        self.interface = HumanFeedbackInterface(self.database)

        # Load configuration from database
        self._load_configuration()

        # Note: MCP manager will be initialized when needed (lazy initialization)
        self._mcp_manager_initialized = False

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'mcp_manager') and self.mcp_manager:
            await self.mcp_manager.close()
            self._mcp_manager_initialized = False

    async def _ensure_mcp_manager_initialized(self):
        """Ensure MCP manager is initialized (lazy initialization)"""
        if not self._mcp_manager_initialized:
            await self._initialize_mcp_manager()
            self._mcp_manager_initialized = True

    async def _initialize_mcp_manager(self):
        """Initialize MCP manager and update components"""
        await self.mcp_manager.initialize()

        # Share MCP manager with planner and executor
        await self.planner.set_mcp_manager(self.mcp_manager)
        self.executor.mcp_manager = self.mcp_manager

        # Load MCP services from database
        await self._load_mcp_services_from_database()

        # Load MCP services from configuration file
        await self._load_mcp_services_from_config()

        logging.info("MCP manager initialized and shared with components")

    async def _load_mcp_services_from_database(self):
        """Load MCP services from database and register with manager"""
        try:
            mcp_services = self.database.load_config("mcp_services", {})
            for service_name, service_config in mcp_services.items():
                try:
                    mcp_config = MCPServiceConfig.parse_obj(service_config)
                    await self.mcp_manager.register_from_service_config(mcp_config)
                    logging.info(f"MCP service '{service_name}' loaded from database")
                except Exception as e:
                    logging.error(f"Failed to load MCP service '{service_name}' from database: {str(e)}")
        except Exception as e:
            logging.error(f"Failed to load MCP services from database: {str(e)}")

    def _create_openai_client(self, agent_name: str, agent_config) -> OpenAI:
        """创建单个 OpenAI 客户端的工厂方法"""
        # 获取OpenAI配置
        openai_config = agent_config.openai_config or OpenAIApiConfig()

        # 如果没有配置API密钥，尝试从环境变量获取
        if not openai_config.api_key:
            global_api_key = os.getenv("OPENAI_API_KEY")
            if not global_api_key:
                raise ValueError(f"OPENAI_API_KEY environment variable is required for {agent_name}")
            openai_config.api_key = global_api_key

        # 获取基础URL
        if not openai_config.base_url:
            global_base_url = os.getenv("OPENAI_API_BASE")
            if global_base_url:
                openai_config.base_url = global_base_url

        # 获取组织ID
        if not openai_config.organization:
            global_organization = os.getenv("OPENAI_ORGANIZATION")
            if global_organization:
                openai_config.organization = global_organization

        # 创建客户端
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
            logging.debug(f"  Base URL: {openai_config.base_url or 'Default'}")
            logging.debug(f"  Model: {agent_config.model_name}")

            return client

        except Exception as e:
            logging.error(f"Failed to initialize {agent_name} OpenAI client: {str(e)}")
            raise

    def _initialize_openai_clients(self) -> Dict[str, OpenAI]:
        """为每个智能体初始化独立的OpenAI客户端"""
        clients = {}
        agent_names = ['planner', 'executor', 'output']

        for agent_name in agent_names:
            agent_config = getattr(self.agent_config, agent_name)
            clients[agent_name] = self._create_openai_client(agent_name, agent_config)

        return clients
    
    def _load_configuration(self):
        """Load system configuration from database and config files"""
        # MCP services are now loaded in _initialize_mcp_manager
        pass
    
    async def _load_mcp_services_from_config(self):
        """Load MCP services from configuration file - delegated to unified manager"""
        # The unified MCP manager now handles configuration loading automatically
        logging.info("MCP services configuration loading delegated to UnifiedMCPManager")
    
    def _update_planner_mcp_services(self):
        """Update planner with available MCP services"""
        try:
            # Get list of available MCP service names
            mcp_service_names = list(self.executor.mcp_services.keys())
            
            # Update planner with available services
            self.planner.update_mcp_services(mcp_service_names)
            
            logging.info(f"Updated planner with MCP services: {mcp_service_names}")
            
        except Exception as e:
            logging.error(f"Failed to update planner with MCP services: {str(e)}")
    
    async def process_user_query(self, user_query: str) -> str:
        """
        Process a user query through the complete pipeline
        
        Args:
            user_query: User's original query/question
            
        Returns:
            str: Final answer
        """
        # Create session
        session_id = str(uuid.uuid4())
        session = TaskSession(
            session_id=session_id,
            user_query=user_query,
            status="created"
        )
        
        self.database.log_event(session_id, "coordinator", "session_started", f"Query: {user_query[:100]}...")

        try:
            # Ensure MCP manager is initialized
            await self._ensure_mcp_manager_initialized()

            # Step 1: Planning Phase
            self.database.log_event(session_id, "coordinator", "planning_phase_started")
            task_graph = await self._planning_phase(user_query, session_id)
            if not task_graph:
                return "抱歉，无法为您的需求生成有效的任务计划。请提供更详细的信息。"
            
            session.task_graph = task_graph
            session.status = "planned"
            self.database.save_session(session)
            
            # Step 2: Human Feedback Phase
            self.database.log_event(session_id, "coordinator", "feedback_phase_started")
            confirmed, final_graph = self._feedback_phase(task_graph, user_query, session_id)
            if not confirmed:
                session.status = "cancelled"
                self.database.save_session(session)
                return "操作已取消。"
            
            session.task_graph = final_graph
            session.status = "confirmed"
            self.database.save_session(session)
            
            # Save final task graph
            self.database.save_task_graph(session_id, final_graph, is_final=True)
            
            # Step 3: Execution Phase
            self.database.log_event(session_id, "coordinator", "execution_phase_started")
            execution_results = await self._execution_phase(final_graph, user_query, session_id)
            session.execution_results = execution_results
            session.status = "executed"
            self.database.save_session(session)
            
            # Step 4: Output Generation Phase
            self.database.log_event(session_id, "coordinator", "output_phase_started")
            final_output = self._output_phase(execution_results, user_query, final_graph, session_id)
            session.final_output = final_output
            session.status = "completed"
            self.database.save_session(session)
            
            # Display execution results
            self.interface.display_execution_results(execution_results, session_id)
            
            # Display final output
            self._display_final_output(final_output)
            
            self.database.log_event(session_id, "coordinator", "session_completed", "All phases completed successfully")
            
            return final_output
            
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            self.database.log_event(session_id, "coordinator", "error", error_msg)
            session.status = "error"
            session.final_output = error_msg
            self.database.save_session(session)
            return error_msg
    
    async def _planning_phase(self, user_query: str, session_id: str) -> Optional[TaskGraph]:
        """Execute the planning phase"""
        try:
            logging.debug(f"Coordinator - Starting planning phase for query: {user_query}")
            logging.debug(f"Coordinator - Session ID: {session_id}")
            
            # Generate task graph using planner model
            logging.debug(f"Coordinator - Calling planner.analyze_and_decompose...")
            task_graph = await self.planner.analyze_and_decompose(user_query, session_id)
            
            logging.debug(f"Coordinator - Task graph generated successfully")
            logging.debug(f"Coordinator - Number of tasks: {len(task_graph.nodes)}")
            logging.debug(f"Coordinator - Number of dependencies: {len(task_graph.edges)}")
            
            # Save initial task graph
            self.database.save_task_graph(session_id, task_graph, is_final=False)
            
            self.database.log_event(session_id, "coordinator", "planning_completed", 
                                   f"Generated {len(task_graph.nodes)} tasks")
            
            return task_graph
            
        except Exception as e:
            logging.error(f"Coordinator - Planning phase failed: {str(e)}")
            logging.error(f"Coordinator - Exception type: {type(e).__name__}")
            import traceback
            logging.error(f"Coordinator - Traceback: {traceback.format_exc()}")
            self.database.log_event(session_id, "coordinator", "planning_error", f"Planning failed: {str(e)}")
            return None
    
    def _feedback_phase(self, task_graph: TaskGraph, user_query: str, session_id: str) -> tuple[bool, Optional[TaskGraph]]:
        """Execute the human feedback phase"""
        try:
            # Present task graph to user for confirmation/modification
            confirmed, modified_graph = self.interface.present_task_graph(task_graph, user_query, session_id)
            
            if confirmed and modified_graph:
                # Save modified task graph
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
        """Execute the execution phase"""
        try:
            # Execute all tasks in the graph
            execution_results = await self.executor.execute_task_graph(task_graph, user_query, session_id)
            
            self.database.log_event(session_id, "coordinator", "execution_completed", 
                                   f"Executed {execution_results.success_count}/{len(execution_results.execution_results)} tasks successfully")
            
            return execution_results
            
        except Exception as e:
            self.database.log_event(session_id, "coordinator", "execution_error", f"Execution failed: {str(e)}")
            # Return empty results on error
            return ExecutionResults()
    
    def _output_phase(self, execution_results: ExecutionResults, user_query: str, 
                     task_graph: TaskGraph, session_id: str) -> str:
        """Execute the output generation phase"""
        try:
            # Generate final integrated output
            final_output = self.output_model.generate_final_output(
                execution_results, user_query, task_graph, session_id
            )
            
            self.database.log_event(session_id, "coordinator", "output_completed", "Final output generated")
            
            return final_output
            
        except Exception as e:
            self.database.log_event(session_id, "coordinator", "output_error", f"Output generation failed: {str(e)}")
            return f"生成最终输出时发生错误: {str(e)}"
    
    def _display_final_output(self, final_output: str):
        """Display the final output to user"""
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        console.print("\n")
        console.print(Panel.fit(
            final_output,
            title="🎯 最终答案",
            border_style="green"
        ))
        console.print("\n")
    
    async def register_mcp_service(self, service_config: MCPServiceConfig):
        """Register an MCP service with the system"""
        # Ensure MCP manager is initialized
        await self._ensure_mcp_manager_initialized()

        # Register with MCP manager
        await self.mcp_manager.register_service(service_config)

        # Save to configuration file
        config_loader = get_config_loader()
        config_loader.add_mcp_service(service_config)

        # Refresh planner's MCP tools with force refresh since services changed
        await self.planner.refresh_mcp_tools(force_refresh=True)
    
    def get_session_history(self, limit: int = 50) -> list[TaskSession]:
        """Get session history"""
        return self.database.list_sessions(limit=limit)
    
    def get_session_details(self, session_id: str) -> Optional[TaskSession]:
        """Get detailed session information"""
        return self.database.load_session(session_id)
    
    def get_system_logs(self, limit: int = 100) -> list[Dict[str, Any]]:
        """Get system logs"""
        return self.database.get_logs(limit=limit)
    
    def get_session_logs(self, session_id: str, limit: int = 100) -> list[Dict[str, Any]]:
        """Get logs for a specific session"""
        return self.database.get_logs(session_id, limit)
    
    def update_config(self, **kwargs):
        """Update system configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.database.save_config(key, value)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        recent_sessions = self.database.list_sessions(limit=10)
        recent_logs = self.database.get_logs(limit=20)
        
        # Calculate statistics
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


class CLIInterface:
    """Command Line Interface for the system"""

    def __init__(self):
        self.coordinator = None
        from rich.console import Console
        from rich.panel import Panel
        self.console = Console()

        # Initialize command registry
        from ..interfaces.command_handlers import CommandRegistry
        self.command_registry = CommandRegistry(self.console)

    def initialize(self, config: Optional[SystemConfig] = None):
        """Initialize the system coordinator"""
        try:
            self.coordinator = SystemCoordinator(config)
            self.console.print("[green]✓ 系统初始化成功[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]✗ 系统初始化失败: {str(e)}[/red]")
            return False

    async def run_interactive_mode(self):
        """Run interactive CLI mode"""
        if not self.coordinator:
            self.console.print("[red]系统未初始化[/red]")
            return

        try:
            from rich.panel import Panel
            self.console.print(Panel.fit(
                "[bold blue]多模型协作任务处理系统[/bold blue]\n"
                "[green]交互模式已启动[/green]\n"
                "输入 'help' 查看可用命令，输入 'quit' 退出",
                title="系统状态"
            ))

            while True:
                try:
                    user_input = self.console.input("\n[bold cyan]请输入您的问题或命令:[/bold cyan] ").strip()

                    if not user_input:
                        continue

                    # Prepare context for command handlers
                    context = {"coordinator": self.coordinator}

                    # Try to handle as command
                    result = await self.command_registry.handle_command(user_input, context)

                    # Check if quit signal was received
                    if result is True:
                        break

                    # If command was handled, continue loop
                    if result is not None:
                        continue

                    # Treat as user query
                    self.console.print(f"\n[bold]正在处理: {user_input}[/bold]")
                    query_result = await self.coordinator.process_user_query(user_input)

                    # Display the result to the user
                    if query_result:
                        self.console.print(f"\n[bold green]处理结果:[/bold green]")
                        self.console.print(query_result)

                except KeyboardInterrupt:
                    self.console.print("\n[yellow]操作已中断[/yellow]")
                    # 确保清理资源
                    try:
                        await self.coordinator.cleanup()
                    except:
                        pass
                except Exception as e:
                    self.console.print(f"\n[red]处理错误: {str(e)}[/red]")
                    # 确保清理资源
                    try:
                        await self.coordinator.cleanup()
                    except:
                        pass
        finally:
            # 最终清理
            if self.coordinator:
                try:
                    await self.coordinator.cleanup()
                except:
                    pass