"""
重构后的CLI界面 - 使用新架构
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from openai import OpenAI

from ..models import SystemConfig
from ..models.agent_config import MultiAgentConfig
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .coordinator import SystemCoordinatorFactory, ISystemCoordinator
from ..services import DatabaseManager
from ..utils.logging import get_logger


class NewCLIInterface:
    """使用新架构的CLI界面"""

    def __init__(self):
        self.coordinator: Optional[ISystemCoordinator] = None
        self.logger = get_logger(__name__)
        
        # 初始化命令注册表
        from rich.console import Console
        self.command_registry = None  # 暂时禁用命令注册表
        self.console = Console()

    async def initialize(self, config: Optional[SystemConfig] = None,
                        openai_client: Optional[OpenAI] = None,
                        agent_config: Optional[MultiAgentConfig] = None,
                        agent_config_path: Optional[str] = None) -> bool:
        """初始化系统"""
        try:
            self.logger.info("🏗️  正在使用新架构初始化系统...")
            
            # 延迟导入避免循环依赖
            from .coordinator import SystemCoordinatorFactory
            
            # 使用新架构工厂创建协调器
            self.coordinator = SystemCoordinatorFactory.create_coordinator(
                config=config,
                agent_config=agent_config,
                agent_config_path=agent_config_path,
                openai_client=openai_client
            )

            # 立即初始化MCP管理器
            self.logger.info("🔄 正在初始化MCP服务...")
            await self.coordinator._ensure_mcp_manager_initialized()
            
            # 显示MCP服务加载状态
            mcp_tools = await self.coordinator.mcp_manager.get_available_tools()
            if mcp_tools:
                total_tools = sum(len(tools) for tools in mcp_tools.values())
                self.logger.info(f"✓ MCP服务初始化成功：{len(mcp_tools)}个服务，{total_tools}个工具")
                for service_name, tools in mcp_tools.items():
                    if tools:
                        self.logger.info(f"  • {service_name}: {len(tools)}个工具")
            else:
                self.logger.warning("⚠ 未加载到MCP工具")

            self.logger.info("✓ 系统初始化成功 (新架构)")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ 系统初始化失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    async def run_interactive_mode(self):
        """运行交互模式"""
        if not self.coordinator:
            self.logger.error("系统未初始化")
            return

        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()

            # 显示详细的系统状态信息
            system_info = await self._get_detailed_system_info()
            console.print(Panel.fit(
                system_info,
                title="系统状态"
            ))

            while True:
                try:
                    user_input = console.input("\n[bold cyan]请输入您的问题或命令:[/bold cyan] ").strip()

                    if not user_input:
                        continue

                    # 简单命令处理
                    if user_input.lower() in ['quit', 'exit', '退出']:
                        console.print("[yellow]再见！[/yellow]")
                        break
                    elif user_input.lower() in ['help', '帮助']:
                        self._show_help(console)
                        continue
                    elif user_input.lower() in ['status', '状态']:
                        status = self.get_system_status()
                        console.print(f"[bold]系统状态:[/bold] {status}")
                        continue

                    # 作为用户查询处理
                    console.print(f"\n[bold]正在处理: {user_input}[/bold]")
                    query_result = await self.coordinator.process_user_query(user_input)

                    # 显示结果
                    if query_result:
                        console.print(f"\n[bold green]处理结果:[/bold green]")
                        console.print(query_result)

                except KeyboardInterrupt:
                    console.print("\n[yellow]操作已中断[/yellow]")
                    break
                except Exception as e:
                    console.print(f"\n[red]处理错误: {str(e)}[/red]")
                    self.logger.error(f"处理错误: {str(e)}")
                    
        finally:
            # 清理资源
            if self.coordinator:
                try:
                    await self.coordinator.cleanup()
                except Exception as e:
                    self.logger.error(f"清理资源时发生错误: {str(e)}")

    async def process_single_query(self, query: str) -> str:
        """处理单个查询"""
        if not self.coordinator:
            raise RuntimeError("系统未初始化")
        
        return await self.coordinator.process_user_query(query)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        if not self.coordinator:
            return {"system_status": "not_initialized"}
        
        return self.coordinator.get_system_status()

    async def register_mcp_service(self, service_config):
        """注册MCP服务"""
        if not self.coordinator:
            raise RuntimeError("系统未初始化")
        
        await self.coordinator.register_mcp_service(service_config)

    def _show_help(self, console):
        """显示帮助信息"""
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        
        # 创建帮助表格
        table = Table(title="可用命令")
        table.add_column("命令", style="cyan", no_wrap=True)
        table.add_column("说明", style="white")
        table.add_column("示例", style="green")
        
        table.add_row("help", "显示帮助信息", "help")
        table.add_row("status", "查看系统状态", "status")
        table.add_row("quit/exit/退出", "退出系统", "quit")
        table.add_row("其他输入", "作为问题处理", "什么是人工智能？")
        
        console.print(table)
        
        # 显示MCP服务信息
        if self.coordinator:
            try:
                # 通过系统状态获取MCP服务信息
                status = self.get_system_status()
                mcp_services = status.get("registered_mcp_services", 0)
                console.print(f"\n[bold]已注册的MCP服务:[/bold] {mcp_services}个")
                
                # 尝试获取详细工具信息
                if hasattr(self.coordinator, 'mcp_manager'):
                    try:
                        mcp_tools = asyncio.run(self.coordinator.mcp_manager.get_available_tools())
                        if mcp_tools:
                            for service_name, tools in mcp_tools.items():
                                console.print(f"  • [cyan]{service_name}[/cyan]: {len(tools)}个工具")
                    except:
                        console.print("[dim]  详细工具信息获取中...[/dim]")
                else:
                    console.print("[dim]  MCP管理器信息不可用[/dim]")
            except Exception as e:
                console.print(f"\n[yellow]无法获取MCP服务信息: {str(e)}[/yellow]")
        
        console.print("\n[dim]提示: 系统使用新架构，支持依赖注入和事件系统[/dim]")

    async def _get_detailed_system_info(self) -> str:
        """获取详细的系统状态信息"""
        import os
        from rich.table import Table
        from rich.text import Text
        
        info_lines = []
        info_lines.append("[bold blue]多模型协作任务处理系统[/bold blue]")
        info_lines.append("[green]新架构模式已启动[/green]\n")
        
        # 模型信息 - 从环境变量获取
        info_lines.append("[bold]🤖 模型配置:[/bold]")
        
        # 获取模型名称和环境变量
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com")
        api_key = os.getenv("OPENAI_API_KEY", "")
        
        # 显示模型信息
        info_lines.append(f"  • 模型名称: {model_name}")
        info_lines.append(f"  • API地址: {base_url}")
        info_lines.append(f"  • API密钥: {'已配置' if api_key else '未配置'}")
        
        # 如果协调器已初始化，显示详细的客户端信息
        if self.coordinator and hasattr(self.coordinator, 'openai_clients'):
            info_lines.append("\n[bold]📋 客户端详情:[/bold]")
            for agent_name, client in self.coordinator.openai_clients.items():
                try:
                    client_model = getattr(client, 'model', model_name)
                    client_base_url = getattr(client, 'base_url', base_url)
                    info_lines.append(f"  • {agent_name}: {client_model}")
                    if client_base_url != base_url:
                        info_lines.append(f"    地址: {client_base_url}")
                except:
                    info_lines.append(f"  • {agent_name}: 配置获取失败")
        
        info_lines.append("")
        
        # MCP服务信息
        if self.coordinator and hasattr(self.coordinator, 'mcp_manager'):
            try:
                mcp_tools = await self.coordinator.mcp_manager.get_available_tools()
                if mcp_tools:
                    info_lines.append("[bold]🔧 MCP服务状态:[/bold]")
                    total_tools = 0
                    for service_name, tools in mcp_tools.items():
                        tool_count = len(tools)
                        total_tools += tool_count
                        status = "🟢 运行中" if tool_count > 0 else "🔴 未加载"
                        info_lines.append(f"  • {service_name}: {tool_count}个工具 ({status})")
                    
                    info_lines.append(f"  📊 总计: {len(mcp_tools)}个服务，{total_tools}个工具")
                else:
                    info_lines.append("[yellow]🔧 MCP服务状态: 未加载服务[/yellow]")
            except Exception as e:
                info_lines.append(f"[red]🔧 MCP服务状态: 获取失败 - {str(e)}[/red]")
        else:
            info_lines.append("[yellow]🔧 MCP服务状态: 未初始化[/yellow]")
        
        info_lines.append("")
        info_lines.append("[dim]输入 'help' 查看可用命令，输入 'quit' 退出[/dim]")
        
        return "\n".join(info_lines)

    async def cleanup(self):
        """清理资源"""
        if self.coordinator:
            try:
                await self.coordinator.cleanup()
            except AttributeError:
                # 如果协调器没有cleanup方法，跳过
                pass
            self.coordinator = None


class HumanFeedbackInterface:
    """人工反馈界面"""
    
    def __init__(self, database):
        self.database = database
    
    def display_execution_results(self, execution_results, session_id):
        """显示执行结果"""
        print(f"执行结果 (会话 {session_id}):")
        print(f"成功: {execution_results.success_count}")
        print(f"失败: {execution_results.failed_count}")
    
    async def present_task_graph(self, task_graph, user_query, session_id, mcp_manager):
        """展示任务图并获取用户反馈"""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.tree import Tree
        from rich.text import Text
        from rich.prompt import Confirm, Prompt
        from rich.layout import Layout
        from rich.columns import Columns
        
        console = Console()
        
        # 显示任务图标题
        console.print(Panel.fit(
            f"[bold blue]任务图展示[/bold blue]\n"
            f"会话ID: {session_id[:8]}...\n"
            f"用户查询: {user_query}",
            title="DAG任务图"
        ))
        
        # 创建任务表格
        table = Table(title="任务列表")
        table.add_column("任务ID", style="cyan", no_wrap=True)
        table.add_column("任务描述", style="white")
        table.add_column("任务类型", style="green")
        table.add_column("优先级", style="yellow")
        table.add_column("预期输出", style="blue")
        
        for node in task_graph.nodes:
            table.add_row(
                node.task_id,
                node.task_desc,
                node.task_type.value,
                str(node.priority),
                node.expected_output[:50] + "..." if len(node.expected_output) > 50 else node.expected_output
            )
        
        console.print(table)
        
        # 显示依赖关系
        if task_graph.edges:
            console.print("\n[bold]任务依赖关系:[/bold]")
            dependency_tree = Tree("依赖关系")
            
            # 构建依赖关系树
            for edge in task_graph.edges:
                from_node = next((n for n in task_graph.nodes if n.task_id == edge.from_task_id), None)
                to_node = next((n for n in task_graph.nodes if n.task_id == edge.to_task_id), None)
                
                if from_node and to_node:
                    branch = dependency_tree.add(f"[cyan]{from_node.task_id}[/cyan] → [green]{to_node.task_id}[/green]")
                    branch.add(f"[dim]类型: {edge.dependency_type.value}[/dim]")
            
            console.print(dependency_tree)
        else:
            console.print("\n[yellow]无任务依赖关系[/yellow]")
        
        # 显示统计信息
        console.print(f"\n[bold]统计信息:[/bold]")
        console.print(f"• 总任务数: {len(task_graph.nodes)}")
        console.print(f"• 依赖关系数: {len(task_graph.edges)}")
        console.print(f"• 有环检测: {'有环' if task_graph.has_cycle() else '无环'}")
        
        # 用户交互选择
        console.print("\n[bold]请选择操作:[/bold]")
        console.print("1. [green]确认执行[/green] - 直接执行当前任务图")
        console.print("2. [yellow]修改任务[/yellow] - 手动修改任务描述或参数")
        console.print("3. [red]取消执行[/red] - 取消本次任务")
        
        while True:
            choice = Prompt.ask(
                "请输入选择",
                choices=["1", "2", "3"],
                default="1"
            )
            
            if choice == "1":
                console.print("[green]✓ 确认执行任务图[/green]")
                return True, task_graph
                
            elif choice == "2":
                console.print("[yellow]进入任务修改模式[/yellow]")
                modified_graph = await self._modify_task_graph(task_graph, console, mcp_manager)
                if modified_graph:
                    console.print("[green]✓ 任务图修改完成[/green]")
                    return True, modified_graph
                else:
                    continue
                    
            elif choice == "3":
                console.print("[red]✗ 取消任务执行[/red]")
                return False, None
    
    async def _modify_task_graph(self, task_graph, console, mcp_manager):
        """修改任务图"""
        from rich.prompt import Prompt, Confirm
        from rich.table import Table
        from rich.panel import Panel
        
        while True:
            # 显示可修改的任务列表
            table = Table(title="选择要修改的任务")
            table.add_column("序号", style="cyan", no_wrap=True)
            table.add_column("任务ID", style="white")
            table.add_column("任务描述", style="green")
            
            for i, node in enumerate(task_graph.nodes, 1):
                table.add_row(str(i), node.task_id, node.task_desc[:30] + "...")
            
            console.print(table)
            
            # 选择任务
            task_choice = Prompt.ask(
                "请输入要修改的任务序号 (输入 'done' 完成修改)",
                default="done"
            )
            
            if task_choice.lower() == 'done':
                break
                
            try:
                task_index = int(task_choice) - 1
                if 0 <= task_index < len(task_graph.nodes):
                    node = task_graph.nodes[task_index]
                    
                    console.print(Panel.fit(
                        f"[bold]当前任务信息:[/bold]\n"
                        f"任务ID: {node.task_id}\n"
                        f"任务描述: {node.task_desc}\n"
                        f"任务类型: {node.task_type.value}\n"
                        f"优先级: {node.priority}\n"
                        f"预期输出: {node.expected_output}",
                        title="任务详情"
                    ))
                    
                    # 选择修改项
                    console.print("\n[bold]选择要修改的项:[/bold]")
                    console.print("1. 任务描述")
                    console.print("2. 优先级")
                    console.print("3. 预期输出")
                    
                    modify_choice = Prompt.ask(
                        "请选择",
                        choices=["1", "2", "3"],
                        default="1"
                    )
                    
                    if modify_choice == "1":
                        new_desc = Prompt.ask("输入新的任务描述", default=node.task_desc)
                        if new_desc.strip():
                            node.task_desc = new_desc.strip()
                            console.print("[green]✓ 任务描述已更新[/green]")
                            
                    elif modify_choice == "2":
                        new_priority = Prompt.ask("输入新的优先级 (1-5)", default=str(node.priority))
                        try:
                            priority = int(new_priority)
                            if 1 <= priority <= 5:
                                node.priority = priority
                                console.print("[green]✓ 优先级已更新[/green]")
                            else:
                                console.print("[red]✗ 优先级必须在1-5之间[/red]")
                        except ValueError:
                            console.print("[red]✗ 请输入有效的数字[/red]")
                            
                    elif modify_choice == "3":
                        new_output = Prompt.ask("输入新的预期输出", default=node.expected_output)
                        if new_output.strip():
                            node.expected_output = new_output.strip()
                            console.print("[green]✓ 预期输出已更新[/green]")
                            
                else:
                    console.print("[red]✗ 无效的任务序号[/red]")
                    
            except ValueError:
                console.print("[red]✗ 请输入有效的数字[/red]")
        
        # 确认修改
        if Confirm.ask("确认完成修改？"):
            return task_graph
        else:
            return None
