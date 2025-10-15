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

from .models import TaskSession, TaskGraph, ExecutionResults, SystemConfig, MCPServiceConfig
from .database import DatabaseManager
from .planner import PlannerModel
from .executor import TaskExecutor
from .output import OutputModel
from .interface import HumanFeedbackInterface


class SystemCoordinator:
    """Main system coordinator that orchestrates all components"""
    
    def _setup_logging(self):
        """Setup logging system with file and console output"""
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Generate log filename with current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_filename = os.path.join(logs_dir, f"{current_date}.log")
        
        # Configure logging format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        
        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler (for all levels)
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler (for INFO and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Log that logging system is initialized
        logging.info(f"Logging system initialized. Log file: {log_filename}")
    
    def __init__(self, config: Optional[SystemConfig] = None):
        # Load environment variables, overriding existing ones
        load_dotenv(override=True)
        
        # Setup logging system
        self._setup_logging()
        
        # Initialize configuration
        self.config = config or SystemConfig()
        
        # Always read OPENAI_MODEL from environment and override if present
        openai_model = os.getenv("OPENAI_MODEL")
        if openai_model:
            self.config.openai_model = openai_model
            logging.debug(f"Overriding model from environment: {openai_model}")
        
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Get custom API base URL if provided
        openai_api_base = os.getenv("OPENAI_API_BASE")
        
        # Log initialization details
        logging.debug(f"OpenAI API Key: {'***' + openai_api_key[-10:] if openai_api_key else 'None'}")
        logging.debug(f"OpenAI API Base: {openai_api_base or 'Default (https://api.openai.com/v1)'}")
        logging.debug(f"OpenAI Model: {self.config.openai_model}")
        
        # Initialize OpenAI client with custom base URL if provided
        try:
            if openai_api_base:
                self.openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
            else:
                self.openai_client = OpenAI(api_key=openai_api_key)
            logging.debug("OpenAI client initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
        
        # Initialize database
        self.database = DatabaseManager(self.config.database_path)
        
        # Initialize components
        self.planner = PlannerModel(self.openai_client, self.database, self.config.openai_model)
        self.executor = TaskExecutor(self.openai_client, self.database, self.config)
        self.output_model = OutputModel(self.openai_client, self.database, self.config.openai_model)
        self.interface = HumanFeedbackInterface(self.database)
        
        # Load configuration from database
        self._load_configuration()
        
        # Update planner with available MCP services
        self._update_planner_mcp_services()
    
    def _load_configuration(self):
        """Load system configuration from database and config files"""
        # Load MCP services from database
        mcp_services = self.database.load_config("mcp_services", {})
        for service_name, service_config in mcp_services.items():
            mcp_config = MCPServiceConfig.parse_obj(service_config)
            self.executor.register_mcp_service(mcp_config)
        
        # Load MCP services from configuration file
        self._load_mcp_services_from_config()
    
    def _load_mcp_services_from_config(self):
        """Load MCP services from configuration file"""
        try:
            import json
            import os
            
            # Look for mcp_config.json in the project root
            config_file_path = "mcp_config.json"
            if os.path.exists(config_file_path):
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Load MCP services from config
                if "mcpServices" in config_data:
                    for service_name, service_config in config_data["mcpServices"].items():
                        try:
                            # Convert config to MCPServiceConfig
                            mcp_config = MCPServiceConfig(
                                service_name=service_name,
                                service_type=service_config.get("type", "http"),
                                endpoint_url=service_config.get("url", ""),
                                timeout=service_config.get("timeout", 30),
                                headers=service_config.get("headers", {})
                            )
                            
                            # Register the service
                            self.executor.register_mcp_service(mcp_config)
                            logging.info(f"MCP service '{service_name}' loaded from config file")
                            
                        except Exception as e:
                            logging.error(f"Failed to load MCP service '{service_name}' from config: {str(e)}")
                
                logging.info(f"MCP services loaded from config file: {config_file_path}")
            else:
                logging.info("No mcp_config.json file found, skipping MCP service loading from config")
                
        except Exception as e:
            logging.error(f"Failed to load MCP services from config file: {str(e)}")
    
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
            task_graph = self.planner.analyze_and_decompose(user_query, session_id)
            
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
    
    def register_mcp_service(self, service_config: MCPServiceConfig):
        """Register an MCP service with the system"""
        self.executor.register_mcp_service(service_config)
        
        # Save to database
        mcp_services = self.database.load_config("mcp_services", {})
        mcp_services[service_config.service_name] = service_config.dict()
        self.database.save_config("mcp_services", mcp_services)
    
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
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    self.console.print("[yellow]再见！[/yellow]")
                    break
                
                if user_input.lower() in ['help', '帮助']:
                    self._show_help()
                    continue
                
                if user_input.lower() in ['status', '状态']:
                    self._show_status()
                    continue
                
                if user_input.lower() in ['history', '历史']:
                    self._show_history()
                    continue
                
                if user_input.lower() in ['logs', '日志']:
                    self._show_logs()
                    continue
                
                if user_input.startswith('config '):
                    self._handle_config_command(user_input)
                    continue
                
                if user_input.startswith('mcp '):
                    self._handle_mcp_command(user_input)
                    continue
                
                # Treat as user query
                self.console.print(f"\n[bold]正在处理: {user_input}[/bold]")
                result = await self.coordinator.process_user_query(user_input)
                
                # Display the result to the user
                if result:
                    self.console.print(f"\n[bold green]处理结果:[/bold green]")
                    self.console.print(result)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]操作已中断[/yellow]")
            except Exception as e:
                self.console.print(f"\n[red]处理错误: {str(e)}[/red]")
    
    def _show_help(self):
        """Show help information"""
        from rich.panel import Panel
        help_text = """
[bold]可用命令:[/bold]

  [cyan]help/帮助[/cyan]     - 显示此帮助信息
  [cyan]quit/exit/退出[/cyan] - 退出系统
  [cyan]status/状态[/cyan]    - 显示系统状态
  [cyan]history/历史[/cyan]   - 显示会话历史
  [cyan]logs/日志[/cyan]      - 显示系统日志
  [cyan]config set <key> <value>[/cyan] - 设置配置
  [cyan]config get <key>[/cyan]       - 获取配置
  [cyan]mcp register <service_name> <url>[/cyan] - 注册MCP服务
  
  直接输入问题即可开始处理流程
        """
        self.console.print(Panel(help_text, title="帮助"))
    
    def _show_status(self):
        """Show system status"""
        status = self.coordinator.get_system_status()
        
        # Create a simple status display
        self.console.print(f"[bold green]系统状态:[/bold green] {status['system_status']}")
        self.console.print(f"[bold]总会话数:[/bold] {status['total_sessions']}")
        self.console.print(f"[bold]最近成功率:[/bold] {status['recent_success_rate']}")
        self.console.print(f"[bold]注册的MCP服务:[/bold] {status['registered_mcp_services']}")
    
    def _show_history(self):
        """Show session history"""
        sessions = self.coordinator.get_session_history(limit=10)
        
        if not sessions:
            self.console.print("[yellow]暂无会话历史[/yellow]")
            return
        
        self.console.print("[bold]最近的会话:[/bold]")
        for session in sessions:
            status_color = "green" if session.status == "completed" else "red"
            self.console.print(f"  {session.session_id[:8]}... [{status_color}]{session.status}[/{status_color}] - {session.user_query[:50]}...")
    
    def _show_logs(self):
        """Show system logs"""
        logs = self.coordinator.get_system_logs(limit=20)
        
        if not logs:
            self.console.print("[yellow]暂无日志[/yellow]")
            return
        
        self.console.print("[bold]系统日志:[/bold]")
        for log in logs:
            self.console.print(f"  {log['timestamp']} - {log['component']}: {log['operation']}")
    
    def _handle_config_command(self, command: str):
        """Handle configuration commands"""
        parts = command.split()
        if len(parts) < 2:
            self.console.print("[red]用法: config <get|set> <key> [value][/red]")
            return
        
        action = parts[1]
        if action == "get" and len(parts) >= 3:
            key = parts[2]
            value = getattr(self.coordinator.config, key, "未找到")
            self.console.print(f"{key}: {value}")
        elif action == "set" and len(parts) >= 4:
            key = parts[2]
            value = parts[3]
            self.coordinator.update_config(**{key: value})
            self.console.print(f"[green]✓ {key} 已设置为 {value}[/green]")
        else:
            self.console.print("[red]用法错误[/red]")
    
    def _handle_mcp_command(self, command: str):
        """Handle MCP service commands"""
        parts = command.split()
        if len(parts) < 4 or parts[1] != "register":
            self.console.print("[red]用法: mcp register <service_name> <url>[/red]")
            return
        
        service_name = parts[2]
        service_url = parts[3]
        
        mcp_config = MCPServiceConfig(
            service_name=service_name,
            endpoint_url=service_url,
            timeout=30
        )
        
        self.coordinator.register_mcp_service(mcp_config)
        self.console.print(f"[green]✓ MCP服务 '{service_name}' 已注册[/green]")