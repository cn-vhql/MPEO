"""
Human Feedback Interface - Interactive CLI for DAG confirmation and modification
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.tree import Tree

from ..models import TaskGraph, TaskNode, TaskEdge, TaskType, DependencyType, TaskSession
from ..services.database import DatabaseManager


class HumanFeedbackInterface:
    """Interactive CLI interface for human feedback and DAG modification"""
    
    def __init__(self, database: DatabaseManager):
        self.database = database
        self.console = Console()
    
    def present_task_graph(self, task_graph: TaskGraph, user_query: str, session_id: str) -> Tuple[bool, Optional[TaskGraph]]:
        """
        Present task graph to user for confirmation or modification
        
        Args:
            task_graph: Generated task graph
            user_query: Original user query
            session_id: Session identifier
            
        Returns:
            Tuple[bool, Optional[TaskGraph]]: (confirmed, modified_task_graph)
        """
        self.database.log_event(session_id, "interface", "dag_presentation", "Presenting DAG to user")
        
        while True:
            # Display header
            self.console.print(Panel.fit(
                f"[bold blue]多模型协作任务处理系统[/bold blue]\n"
                f"[green]任务图确认环节[/green]\n\n"
                f"原始问题：{user_query}",
                title="系统提示"
            ))
            
            # Display task graph summary
            self._display_graph_summary(task_graph)
            
            # Display detailed tasks
            self._display_tasks(task_graph)
            
            # Display dependencies
            self._display_dependencies(task_graph)
            
            # Get user action
            action = self._get_user_action()
            
            if action == "confirm":
                self.database.log_event(session_id, "interface", "dag_confirmed", "User confirmed DAG")
                self.console.print("[green]✓ 任务图已确认，开始执行...[/green]")
                return True, task_graph
            
            elif action == "modify":
                modified_graph = self._modify_task_graph(task_graph, session_id)
                if modified_graph:
                    task_graph = modified_graph
                    continue
                else:
                    # User cancelled modification
                    continue
            
            elif action == "save":
                self._save_task_graph(task_graph, session_id)
                continue
            
            elif action == "load":
                loaded_graph = self._load_task_graph(session_id)
                if loaded_graph:
                    task_graph = loaded_graph
                    continue
                else:
                    continue
            
            elif action == "cancel":
                self.database.log_event(session_id, "interface", "dag_cancelled", "User cancelled process")
                self.console.print("[yellow]操作已取消[/yellow]")
                return False, None
    
    def _display_graph_summary(self, task_graph: TaskGraph):
        """Display task graph summary"""
        table = Table(title="任务图概览", show_header=True, header_style="bold magenta")
        table.add_column("属性", style="cyan", width=20)
        table.add_column("值", style="white")
        
        table.add_row("任务总数", str(len(task_graph.nodes)))
        table.add_row("依赖关系数", str(len(task_graph.edges)))
        table.add_row("图类型", "有向无环图 (DAG)")
        
        # Count task types
        task_types = {}
        for task in task_graph.nodes:
            task_type = task.task_type.value
            task_types[task_type] = task_types.get(task_type, 0) + 1
        
        type_summary = ", ".join([f"{t}:{c}" for t, c in task_types.items()])
        table.add_row("任务类型分布", type_summary)
        
        # Check if graph has cycles
        has_cycle = task_graph.has_cycle()
        table.add_row("循环依赖", "是" if has_cycle else "否", style="red" if has_cycle else "green")
        
        self.console.print(table)
        self.console.print()
    
    def _display_tasks(self, task_graph: TaskGraph):
        """Display detailed task information"""
        table = Table(title="任务详情", show_header=True, header_style="bold magenta")
        table.add_column("任务ID", style="cyan", width=8)
        table.add_column("任务描述", style="white", width=30)
        table.add_column("任务类型", style="green", width=12)
        table.add_column("预期输出", style="yellow", width=25)
        table.add_column("优先级", style="red", width=6)
        
        # Sort tasks by priority (high to low)
        sorted_tasks = sorted(task_graph.nodes, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            table.add_row(
                task.task_id,
                task.task_desc[:28] + "..." if len(task.task_desc) > 28 else task.task_desc,
                task.task_type.value,
                task.expected_output[:23] + "..." if len(task.expected_output) > 23 else task.expected_output,
                str(task.priority)
            )
        
        self.console.print(table)
        self.console.print()
    
    def _display_dependencies(self, task_graph: TaskGraph):
        """Display task dependencies"""
        if not task_graph.edges:
            self.console.print("[yellow]无任务依赖关系[/yellow]")
            return
        
        table = Table(title="任务依赖关系", show_header=True, header_style="bold magenta")
        table.add_column("前置任务", style="cyan", width=10)
        table.add_column("→", style="white", width=2)
        table.add_column("依赖任务", style="cyan", width=10)
        table.add_column("依赖类型", style="green", width=12)
        
        for edge in task_graph.edges:
            table.add_row(
                edge.from_task_id,
                "→",
                edge.to_task_id,
                edge.dependency_type.value
            )
        
        self.console.print(table)
        self.console.print()
    
    def _get_user_action(self) -> str:
        """Get user action choice"""
        choices = [
            ("confirm", "确认任务图，开始执行"),
            ("modify", "修改任务图"),
            ("save", "保存当前任务图"),
            ("load", "加载已保存的任务图"),
            ("cancel", "取消操作")
        ]
        
        self.console.print("\n[bold]请选择操作：[/bold]")
        for i, (key, desc) in enumerate(choices, 1):
            self.console.print(f"  {i}. {desc} ({key})")
        
        while True:
            choice = Prompt.ask("\n请输入选择", choices=["1", "2", "3", "4", "5", "confirm", "modify", "save", "load", "cancel"])
            
            if choice in ["1", "confirm"]:
                return "confirm"
            elif choice in ["2", "modify"]:
                return "modify"
            elif choice in ["3", "save"]:
                return "save"
            elif choice in ["4", "load"]:
                return "load"
            elif choice in ["5", "cancel"]:
                return "cancel"
    
    def _modify_task_graph(self, task_graph: TaskGraph, session_id: str) -> Optional[TaskGraph]:
        """Modify task graph based on user input"""
        self.console.print("\n[bold blue]任务图修改模式[/bold blue]")
        
        while True:
            modification_type = self._get_modification_type()
            
            if modification_type == "add_task":
                self._add_task(task_graph)
            elif modification_type == "remove_task":
                self._remove_task(task_graph)
            elif modification_type == "edit_task":
                self._edit_task(task_graph)
            elif modification_type == "add_dependency":
                self._add_dependency(task_graph)
            elif modification_type == "remove_dependency":
                self._remove_dependency(task_graph)
            elif modification_type == "done":
                # Validate the modified graph
                if task_graph.has_cycle():
                    self.console.print("[red]✗ 检测到循环依赖，请重新修改[/red]")
                    continue
                else:
                    self.database.log_event(session_id, "interface", "dag_modified", "Task graph modified successfully")
                    self.console.print("[green]✓ 任务图修改完成[/green]")
                    return task_graph
            elif modification_type == "cancel":
                return None
    
    def _get_modification_type(self) -> str:
        """Get modification type from user"""
        choices = [
            ("add_task", "添加任务"),
            ("remove_task", "删除任务"),
            ("edit_task", "编辑任务"),
            ("add_dependency", "添加依赖关系"),
            ("remove_dependency", "删除依赖关系"),
            ("done", "完成修改"),
            ("cancel", "取消修改")
        ]
        
        self.console.print("\n[bold]请选择修改类型：[/bold]")
        for i, (key, desc) in enumerate(choices, 1):
            self.console.print(f"  {i}. {desc}")
        
        choice = Prompt.ask("\n请输入选择", choices=["1", "2", "3", "4", "5", "6", "7"])
        
        choice_map = {
            "1": "add_task",
            "2": "remove_task", 
            "3": "edit_task",
            "4": "add_dependency",
            "5": "remove_dependency",
            "6": "done",
            "7": "cancel"
        }
        
        return choice_map[choice]
    
    def _add_task(self, task_graph: TaskGraph):
        """Add a new task to the graph"""
        self.console.print("\n[bold]添加新任务[/bold]")
        
        task_id = Prompt.ask("任务ID (如 T4)")
        # Check if task ID already exists
        if any(task.task_id == task_id for task in task_graph.nodes):
            self.console.print(f"[red]任务ID {task_id} 已存在[/red]")
            return
        
        task_desc = Prompt.ask("任务描述")
        
        task_type_choice = Prompt.ask(
            "任务类型",
            choices=["本地计算", "mcp调用", "数据处理"],
            default="本地计算"
        )
        task_type = TaskType(task_type_choice)
        
        expected_output = Prompt.ask("预期输出")
        
        priority = int(Prompt.ask("优先级 (1-5)", default="3"))
        priority = max(1, min(5, priority))  # Ensure within range
        
        new_task = TaskNode(
            task_id=task_id,
            task_desc=task_desc,
            task_type=task_type,
            expected_output=expected_output,
            priority=priority
        )
        
        task_graph.nodes.append(new_task)
        self.console.print(f"[green]✓ 任务 {task_id} 已添加[/green]")
    
    def _remove_task(self, task_graph: TaskGraph):
        """Remove a task from the graph"""
        if not task_graph.nodes:
            self.console.print("[yellow]没有可删除的任务[/yellow]")
            return
        
        task_ids = [task.task_id for task in task_graph.nodes]
        task_id = Prompt.ask("要删除的任务ID", choices=task_ids)
        
        # Remove task
        task_graph.nodes = [task for task in task_graph.nodes if task.task_id != task_id]
        
        # Remove related dependencies
        task_graph.edges = [
            edge for edge in task_graph.edges 
            if edge.from_task_id != task_id and edge.to_task_id != task_id
        ]
        
        self.console.print(f"[green]✓ 任务 {task_id} 及相关依赖已删除[/green]")
    
    def _edit_task(self, task_graph: TaskGraph):
        """Edit an existing task"""
        if not task_graph.nodes:
            self.console.print("[yellow]没有可编辑的任务[/yellow]")
            return
        
        task_ids = [task.task_id for task in task_graph.nodes]
        task_id = Prompt.ask("要编辑的任务ID", choices=task_ids)
        
        # Find the task
        task = next((t for t in task_graph.nodes if t.task_id == task_id), None)
        if not task:
            return
        
        self.console.print(f"\n[bold]编辑任务 {task_id}[/bold]")
        self.console.print(f"当前描述: {task.task_desc}")
        
        new_desc = Prompt.ask("新任务描述 (留空保持不变)", default="")
        if new_desc:
            task.task_desc = new_desc
        
        new_type = Prompt.ask(
            "新任务类型",
            choices=["本地计算", "mcp调用", "数据处理"],
            default=task.task_type.value
        )
        task.task_type = TaskType(new_type)
        
        new_output = Prompt.ask("新预期输出 (留空保持不变)", default="")
        if new_output:
            task.expected_output = new_output
        
        new_priority = Prompt.ask("新优先级 (1-5)", default=str(task.priority))
        try:
            task.priority = max(1, min(5, int(new_priority)))
        except ValueError:
            pass  # Keep original priority
        
        self.console.print(f"[green]✓ 任务 {task_id} 已更新[/green]")
    
    def _add_dependency(self, task_graph: TaskGraph):
        """Add a dependency between tasks"""
        if len(task_graph.nodes) < 2:
            self.console.print("[yellow]需要至少2个任务才能添加依赖关系[/yellow]")
            return
        
        task_ids = [task.task_id for task in task_graph.nodes]
        
        from_task = Prompt.ask("前置任务ID", choices=task_ids)
        to_task = Prompt.ask("依赖任务ID", choices=[tid for tid in task_ids if tid != from_task])
        
        dep_type = Prompt.ask(
            "依赖类型",
            choices=["数据依赖", "结果依赖"],
            default="结果依赖"
        )
        
        # Check if dependency already exists
        for edge in task_graph.edges:
            if edge.from_task_id == from_task and edge.to_task_id == to_task:
                self.console.print("[yellow]该依赖关系已存在[/yellow]")
                return
        
        new_edge = TaskEdge(
            from_task_id=from_task,
            to_task_id=to_task,
            dependency_type=DependencyType(dep_type)
        )
        
        task_graph.edges.append(new_edge)
        
        # Check for cycles
        if task_graph.has_cycle():
            # Remove the edge that caused the cycle
            task_graph.edges.pop()
            self.console.print("[red]✗ 添加此依赖会产生循环依赖，操作已撤销[/red]")
        else:
            self.console.print(f"[green]✓ 依赖关系 {from_task} → {to_task} 已添加[/green]")
    
    def _remove_dependency(self, task_graph: TaskGraph):
        """Remove a dependency"""
        if not task_graph.edges:
            self.console.print("[yellow]没有可删除的依赖关系[/yellow]")
            return
        
        self.console.print("\n[bold]现有依赖关系：[/bold]")
        for i, edge in enumerate(task_graph.edges, 1):
            self.console.print(f"  {i}. {edge.from_task_id} → {edge.to_task_id} ({edge.dependency_type.value})")
        
        choice = Prompt.ask("要删除的依赖关系编号", choices=[str(i) for i in range(1, len(task_graph.edges) + 1)])
        
        index = int(choice) - 1
        if 0 <= index < len(task_graph.edges):
            removed_edge = task_graph.edges.pop(index)
            self.console.print(f"[green]✓ 依赖关系 {removed_edge.from_task_id} → {removed_edge.to_task_id} 已删除[/green]")
    
    def _save_task_graph(self, task_graph: TaskGraph, session_id: str):
        """Save task graph to database"""
        success = self.database.save_task_graph(session_id, task_graph, is_final=False)
        if success:
            self.console.print("[green]✓ 任务图已保存[/green]")
        else:
            self.console.print("[red]✗ 保存失败[/red]")
    
    def _load_task_graph(self, session_id: str) -> Optional[TaskGraph]:
        """Load task graph from database"""
        graphs = self.database.get_task_graphs(session_id)
        
        if not graphs:
            self.console.print("[yellow]没有已保存的任务图[/yellow]")
            return None
        
        self.console.print("\n[bold]已保存的任务图版本：[/bold]")
        for i, graph_info in enumerate(graphs, 1):
            status = "最终版本" if graph_info['is_final'] else f"版本 {graph_info['version']}"
            self.console.print(f"  {i}. {status} ({graph_info['created_at']})")
        
        choice = Prompt.ask("要加载的版本", choices=[str(i) for i in range(1, len(graphs) + 1)])
        
        index = int(choice) - 1
        if 0 <= index < len(graphs):
            graph_data = graphs[index]['graph_data']
            task_graph = TaskGraph.parse_obj(graph_data)
            self.console.print(f"[green]✓ 已加载任务图版本 {graphs[index]['version']}[/green]")
            return task_graph
        
        return None
    
    def display_execution_results(self, execution_results, session_id: str):
        """Display execution results to user"""
        self.console.print(Panel.fit(
            "[bold green]任务执行完成[/bold green]",
            title="执行结果"
        ))
        
        # Display summary
        table = Table(title="执行摘要", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan", width=15)
        table.add_column("值", style="white")
        
        table.add_row("总任务数", str(len(execution_results.execution_results)))
        table.add_row("成功任务数", str(execution_results.success_count))
        table.add_row("失败任务数", str(execution_results.failed_count))
        table.add_row("总执行时间", f"{execution_results.total_execution_time:.2f}秒")
        
        self.console.print(table)
        self.console.print()
        
        # Display detailed results
        table = Table(title="详细执行结果", show_header=True, header_style="bold magenta")
        table.add_column("任务ID", style="cyan", width=8)
        table.add_column("状态", style="white", width=10)
        table.add_column("执行时间", style="green", width=10)
        table.add_column("输出/错误", style="yellow", width=40)
        
        for result in execution_results.execution_results:
            status_text = "✓ 成功" if result.status.value == "成功" else "✗ 失败"
            status_style = "green" if result.status.value == "成功" else "red"
            
            output_text = result.output or result.error_msg or "无输出"
            if len(output_text) > 37:
                output_text = output_text[:37] + "..."
            
            table.add_row(
                result.task_id,
                f"[{status_style}]{status_text}[/{status_style}]",
                f"{result.execution_time:.2f}s",
                output_text
            )
        
        self.console.print(table)
        
        self.database.log_event(session_id, "interface", "results_displayed", 
                               f"Displayed {len(execution_results.execution_results)} results")