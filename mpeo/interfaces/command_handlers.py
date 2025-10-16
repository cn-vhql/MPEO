"""
命令处理器模块 - 处理各种CLI命令
"""

from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..models import TaskSession, MCPServiceConfig


class CommandHandler:
    """基础命令处理器"""

    def __init__(self, console: Console):
        self.console = console

    def can_handle(self, command: str) -> bool:
        """检查是否能处理该命令"""
        raise NotImplementedError

    async def handle(self, command: str, context: Dict[str, Any]) -> Any:
        """处理命令"""
        raise NotImplementedError


class HelpCommandHandler(CommandHandler):
    """帮助命令处理器"""

    def can_handle(self, command: str) -> bool:
        return command.lower() in ['help', '帮助']

    async def handle(self, command: str, context: Dict[str, Any]) -> None:
        """显示帮助信息"""
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


class StatusCommandHandler(CommandHandler):
    """状态命令处理器"""

    def can_handle(self, command: str) -> bool:
        return command.lower() in ['status', '状态']

    async def handle(self, command: str, context: Dict[str, Any]) -> None:
        """显示系统状态"""
        coordinator = context.get('coordinator')
        if not coordinator:
            self.console.print("[red]系统协调器未初始化[/red]")
            return

        status = coordinator.get_system_status()

        # Create status display
        table = Table(title="系统状态", show_header=True, header_style="bold magenta")
        table.add_column("属性", style="cyan", width=20)
        table.add_column("值", style="white")

        table.add_row("系统状态", status['system_status'])
        table.add_row("总会话数", str(status['total_sessions']))
        table.add_row("最近成功率", status['recent_success_rate'])
        table.add_row("注册的MCP服务", str(status['registered_mcp_services']))

        self.console.print(table)


class HistoryCommandHandler(CommandHandler):
    """历史命令处理器"""

    def can_handle(self, command: str) -> bool:
        return command.lower() in ['history', '历史']

    async def handle(self, command: str, context: Dict[str, Any]) -> None:
        """显示会话历史"""
        coordinator = context.get('coordinator')
        if not coordinator:
            self.console.print("[red]系统协调器未初始化[/red]")
            return

        sessions = coordinator.get_session_history(limit=10)

        if not sessions:
            self.console.print("[yellow]暂无会话历史[/yellow]")
            return

        table = Table(title="最近的会话", show_header=True, header_style="bold magenta")
        table.add_column("会话ID", style="cyan", width=10)
        table.add_column("状态", style="white", width=10)
        table.add_column("用户查询", style="white", width=40)

        for session in sessions:
            status_color = "green" if session.status == "completed" else "red"
            status_text = f"[{status_color}]{session.status}[/{status_color}]"

            table.add_row(
                session.session_id[:8] + "...",
                status_text,
                session.user_query[:37] + "..." if len(session.user_query) > 37 else session.user_query
            )

        self.console.print(table)


class LogsCommandHandler(CommandHandler):
    """日志命令处理器"""

    def can_handle(self, command: str) -> bool:
        return command.lower() in ['logs', '日志']

    async def handle(self, command: str, context: Dict[str, Any]) -> None:
        """显示系统日志"""
        coordinator = context.get('coordinator')
        if not coordinator:
            self.console.print("[red]系统协调器未初始化[/red]")
            return

        logs = coordinator.get_system_logs(limit=20)

        if not logs:
            self.console.print("[yellow]暂无日志[/yellow]")
            return

        table = Table(title="系统日志", show_header=True, header_style="bold magenta")
        table.add_column("时间", style="cyan", width=20)
        table.add_column("组件", style="white", width=15)
        table.add_column("操作", style="white", width=20)

        for log in logs[:10]:  # 只显示前10条
            table.add_row(
                log['timestamp'][:19] if len(log['timestamp']) > 19 else log['timestamp'],
                log['component'],
                log['operation'][:18] + "..." if len(log['operation']) > 18 else log['operation']
            )

        self.console.print(table)


class ConfigCommandHandler(CommandHandler):
    """配置命令处理器"""

    def can_handle(self, command: str) -> bool:
        return command.startswith('config ')

    async def handle(self, command: str, context: Dict[str, Any]) -> None:
        """处理配置命令"""
        coordinator = context.get('coordinator')
        if not coordinator:
            self.console.print("[red]系统协调器未初始化[/red]")
            return

        parts = command.split()
        if len(parts) < 2:
            self.console.print("[red]用法: config <get|set> <key> [value][/red]")
            return

        action = parts[1]
        if action == "get" and len(parts) >= 3:
            key = parts[2]
            value = getattr(coordinator.config, key, "未找到")
            self.console.print(f"{key}: {value}")
        elif action == "set" and len(parts) >= 4:
            key = parts[2]
            value = parts[3]
            coordinator.update_config(**{key: value})
            self.console.print(f"[green]✓ {key} 已设置为 {value}[/green]")
        else:
            self.console.print("[red]用法错误[/red]")


class MCPCommandHandler(CommandHandler):
    """MCP命令处理器"""

    def can_handle(self, command: str) -> bool:
        return command.startswith('mcp ')

    async def handle(self, command: str, context: Dict[str, Any]) -> None:
        """处理MCP命令"""
        coordinator = context.get('coordinator')
        if not coordinator:
            self.console.print("[red]系统协调器未初始化[/red]")
            return

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

        try:
            await coordinator.register_mcp_service(mcp_config)
            self.console.print(f"[green]✓ MCP服务 '{service_name}' 已注册[/green]")
        except Exception as e:
            self.console.print(f"[red]✗ MCP服务注册失败: {str(e)}[/red]")


class QuitCommandHandler(CommandHandler):
    """退出命令处理器"""

    def can_handle(self, command: str) -> bool:
        return command.lower() in ['quit', 'exit', '退出']

    async def handle(self, command: str, context: Dict[str, Any]) -> bool:
        """处理退出命令"""
        coordinator = context.get('coordinator')
        if coordinator:
            try:
                await coordinator.cleanup()
            except:
                pass

        self.console.print("[yellow]再见！[/yellow]")
        return True  # Signal to quit


class CommandRegistry:
    """命令注册表"""

    def __init__(self, console: Console):
        self.console = console
        self.handlers: List[CommandHandler] = []
        self._register_default_handlers()

    def _register_default_handlers(self):
        """注册默认命令处理器"""
        self.handlers.extend([
            HelpCommandHandler(self.console),
            StatusCommandHandler(self.console),
            HistoryCommandHandler(self.console),
            LogsCommandHandler(self.console),
            ConfigCommandHandler(self.console),
            MCPCommandHandler(self.console),
            QuitCommandHandler(self.console)
        ])

    def register_handler(self, handler: CommandHandler):
        """注册命令处理器"""
        self.handlers.append(handler)

    async def handle_command(self, command: str, context: Dict[str, Any]) -> Any:
        """处理命令"""
        for handler in self.handlers:
            if handler.can_handle(command):
                return await handler.handle(command, context)

        # No handler found
        return None