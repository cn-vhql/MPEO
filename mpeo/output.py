"""
Output Model - Result integration and final answer generation
"""

import json
from typing import Dict, List, Any, Optional
from openai import OpenAI

from .models import TaskGraph, ExecutionResults, TaskStatus
from .database import DatabaseManager


class OutputModel:
    """Output model for result integration and final answer generation"""
    
    def __init__(self, openai_client: OpenAI, database: DatabaseManager, model_name: str = "gpt-3.5-turbo"):
        self.client = openai_client
        self.database = database
        self.model_name = model_name
    
    def generate_final_output(self, 
                            execution_results: ExecutionResults,
                            user_query: str,
                            task_graph: Optional[TaskGraph] = None,
                            session_id: Optional[str] = None) -> str:
        """
        Generate final output based on execution results
        
        Args:
            execution_results: Results from task execution
            user_query: Original user query
            task_graph: Original task graph (optional)
            session_id: Session identifier for logging
            
        Returns:
            str: Final integrated answer
        """
        if session_id:
            self.database.log_event(session_id, "output", "start_integration", 
                                   f"Integrating {len(execution_results.execution_results)} results")
        
        try:
            # Step 1: Validate and check for conflicts
            validation_result = self._validate_results(execution_results, session_id)
            
            # Step 2: Extract and organize results
            organized_results = self._organize_results(execution_results, task_graph, session_id)
            
            # Step 3: Generate integrated output
            final_output = self._generate_integrated_output(
                organized_results, user_query, validation_result, session_id
            )
            
            # Step 4: Format output based on user preferences
            formatted_output = self._format_output(final_output, user_query, session_id)
            
            if session_id:
                self.database.log_event(session_id, "output", "integration_completed", 
                                       "Final output generated successfully")
            
            return formatted_output
            
        except Exception as e:
            error_msg = f"Failed to generate final output: {str(e)}"
            if session_id:
                self.database.log_event(session_id, "output", "error", error_msg)
            return f"抱歉，生成最终输出时出现错误：{error_msg}"
    
    def _validate_results(self, execution_results: ExecutionResults, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Validate execution results and check for conflicts"""
        validation_result = {
            "is_valid": True,
            "conflicts": [],
            "warnings": [],
            "failed_tasks": [],
            "missing_outputs": []
        }
        
        # Check for failed tasks
        for result in execution_results.execution_results:
            if result.status == TaskStatus.FAILED:
                validation_result["failed_tasks"].append({
                    "task_id": result.task_id,
                    "error_msg": result.error_msg
                })
                validation_result["is_valid"] = False
            elif result.output is None:
                validation_result["missing_outputs"].append(result.task_id)
                validation_result["warnings"].append(f"Task {result.task_id} has no output")
        
        # Check for data conflicts (basic implementation)
        successful_results = [r for r in execution_results.execution_results if r.status == TaskStatus.SUCCESS]
        if len(successful_results) > 1:
            # Look for potential conflicts in text outputs
            outputs_by_type = {}
            for result in successful_results:
                if isinstance(result.output, str):
                    output_type = self._classify_output(result.output)
                    if output_type not in outputs_by_type:
                        outputs_by_type[output_type] = []
                    outputs_by_type[output_type].append({
                        "task_id": result.task_id,
                        "output": result.output
                    })
            
            # Check for conflicts within same output type
            for output_type, outputs in outputs_by_type.items():
                if len(outputs) > 1 and output_type in ["numeric", "categorical"]:
                    # Simple conflict detection for structured data
                    if self._has_conflicts(outputs):
                        validation_result["conflicts"].append({
                            "type": output_type,
                            "tasks": [o["task_id"] for o in outputs],
                            "description": f"Conflicting {output_type} outputs detected"
                        })
        
        if session_id:
            self.database.log_event(session_id, "output", "validation_completed", 
                                   f"Valid: {validation_result['is_valid']}, Conflicts: {len(validation_result['conflicts'])}")
        
        return validation_result
    
    def _organize_results(self, execution_results: ExecutionResults, 
                         task_graph: Optional[TaskGraph] = None,
                         session_id: Optional[str] = None) -> Dict[str, Any]:
        """Organize execution results by tasks and dependencies"""
        organized = {
            "summary": {
                "total_tasks": len(execution_results.execution_results),
                "successful_tasks": execution_results.success_count,
                "failed_tasks": execution_results.failed_count,
                "total_time": execution_results.total_execution_time
            },
            "results_by_task": {},
            "execution_flow": [],
            "key_outputs": []
        }
        
        # Organize results by task
        for result in execution_results.execution_results:
            organized["results_by_task"][result.task_id] = {
                "status": result.status.value,
                "output": result.output,
                "execution_time": result.execution_time,
                "error_msg": result.error_msg
            }
        
        # Build execution flow if task graph is available
        if task_graph:
            organized["execution_flow"] = self._build_execution_flow(task_graph, execution_results)
        
        # Extract key outputs
        for result in execution_results.execution_results:
            if result.status == TaskStatus.SUCCESS and result.output:
                organized["key_outputs"].append({
                    "task_id": result.task_id,
                    "output": result.output,
                    "type": self._classify_output(result.output) if isinstance(result.output, str) else "structured"
                })
        
        if session_id:
            self.database.log_event(session_id, "output", "results_organized", 
                                   f"Organized {len(organized['results_by_task'])} task results")
        
        return organized
    
    def _generate_integrated_output(self, organized_results: Dict[str, Any],
                                  user_query: str,
                                  validation_result: Dict[str, Any],
                                  session_id: Optional[str] = None) -> str:
        """Generate integrated output using OpenAI"""
        
        # Prepare context for AI
        context = {
            "user_query": user_query,
            "execution_summary": organized_results["summary"],
            "task_results": organized_results["results_by_task"],
            "key_outputs": organized_results["key_outputs"],
            "validation": validation_result
        }
        
        prompt = f"""
        请基于以下任务执行结果，为用户的原始问题生成一个完整、准确的答案：

        原始问题：{user_query}

        执行摘要：
        - 总任务数：{context['execution_summary']['total_tasks']}
        - 成功任务数：{context['execution_summary']['successful_tasks']}
        - 失败任务数：{context['execution_summary']['failed_tasks']}
        - 总执行时间：{context['execution_summary']['total_time']:.2f}秒

        任务执行结果：
        {json.dumps(context['task_results'], ensure_ascii=False, indent=2)}

        验证结果：
        {json.dumps(context['validation'], ensure_ascii=False, indent=2)}

        请根据以上信息：
        1. 整合所有成功任务的输出
        2. 处理任何冲突或不一致的地方
        3. 如果有任务失败，说明对最终结果的影响
        4. 生成一个针对原始问题的完整答案
        5. 确保答案逻辑清晰、结构合理

        答案应该：
        - 直接回应用户的原始问题
        - 包含相关的细节和解释
        - 如果有局限性或不确定性，请明确说明
        - 语言简洁明了，易于理解
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的结果整合专家，擅长将多个任务的执行结果整合为完整、准确的答案。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = f"Failed to generate integrated output: {str(e)}"
            if session_id:
                self.database.log_event(session_id, "output", "ai_generation_error", error_msg)
            
            # Fallback: simple concatenation of results
            return self._fallback_integration(organized_results, user_query)
    
    def _format_output(self, output: str, user_query: str, session_id: Optional[str] = None) -> str:
        """Format output based on user preferences and query context"""
        
        # Check if user requested specific format
        format_preferences = self._extract_format_preferences(user_query)
        
        if "表格" in user_query or "table" in user_query.lower():
            # Try to format as table if requested
            formatted = self._format_as_table(output)
        elif "列表" in user_query or "list" in user_query.lower():
            # Try to format as list if requested
            formatted = self._format_as_list(output)
        elif "报告" in user_query or "report" in user_query.lower():
            # Format as report
            formatted = self._format_as_report(output)
        else:
            # Default formatting
            formatted = self._default_formatting(output)
        
        if session_id:
            self.database.log_event(session_id, "output", "formatting_completed", 
                                   f"Format: {format_preferences}")
        
        return formatted
    
    def _classify_output(self, output: str) -> str:
        """Classify the type of output"""
        output_lower = output.lower()
        
        # Check for numeric data
        if any(char.isdigit() for char in output) and len(output.split()) < 50:
            return "numeric"
        
        # Check for categorical data
        if any(keyword in output_lower for keyword in ["是", "否", "true", "false", "通过", "失败"]):
            return "categorical"
        
        # Check for structured data
        if any(char in output for char in ["{", "}", "[", "]", "|", "-"]) and len(output) > 100:
            return "structured"
        
        return "text"
    
    def _has_conflicts(self, outputs: List[Dict[str, Any]]) -> bool:
        """Check if outputs have conflicts"""
        if len(outputs) < 2:
            return False
        
        # Simple conflict detection - check if outputs are significantly different
        first_output = outputs[0]["output"].lower()
        
        for i in range(1, len(outputs)):
            other_output = outputs[i]["output"].lower()
            
            # If outputs are numeric and differ
            if self._is_numeric(first_output) and self._is_numeric(other_output):
                if abs(float(first_output) - float(other_output)) > 0.01:
                    return True
            
            # If outputs are categorical and different
            elif first_output != other_output:
                # Only consider it a conflict if they're clearly contradictory
                contradictory_pairs = [
                    ("是", "否"), ("true", "false"), ("通过", "失败"),
                    ("成功", "失败"), ("yes", "no")
                ]
                for pair in contradictory_pairs:
                    if (pair[0] in first_output and pair[1] in other_output) or \
                       (pair[1] in first_output and pair[0] in other_output):
                        return True
        
        return False
    
    def _is_numeric(self, text: str) -> bool:
        """Check if text represents a number"""
        try:
            float(text)
            return True
        except ValueError:
            return False
    
    def _build_execution_flow(self, task_graph: TaskGraph, execution_results: ExecutionResults) -> List[Dict[str, Any]]:
        """Build execution flow description"""
        flow = []
        
        # Add task nodes in order
        for task in task_graph.nodes:
            result = next((r for r in execution_results.execution_results if r.task_id == task.task_id), None)
            if result:
                flow.append({
                    "task_id": task.task_id,
                    "description": task.task_desc,
                    "status": result.status.value,
                    "execution_time": result.execution_time,
                    "dependencies": [edge.from_task_id for edge in task_graph.edges if edge.to_task_id == task.task_id]
                })
        
        return flow
    
    def _extract_format_preferences(self, user_query: str) -> str:
        """Extract format preferences from user query"""
        query_lower = user_query.lower()
        
        if "表格" in user_query or "table" in query_lower:
            return "table"
        elif "列表" in user_query or "list" in query_lower:
            return "list"
        elif "报告" in user_query or "report" in query_lower:
            return "report"
        elif "简洁" in user_query or "concise" in query_lower:
            return "concise"
        elif "详细" in user_query or "detailed" in query_lower:
            return "detailed"
        
        return "default"
    
    def _format_as_table(self, output: str) -> str:
        """Format output as table"""
        # Simple table formatting - can be enhanced
        lines = output.split('\n')
        if len(lines) < 3:
            return output
        
        # Try to detect table structure
        if any('|' in line for line in lines):
            return output  # Already table-like
        
        # Add simple table formatting
        table_output = "```\n"
        for line in lines[:10]:  # Limit to first 10 lines
            table_output += f"| {line} |\n"
        table_output += "```"
        
        return table_output
    
    def _format_as_list(self, output: str) -> str:
        """Format output as list"""
        lines = output.split('\n')
        list_output = ""
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('-') and not line.startswith('*'):
                list_output += f"- {line}\n"
            else:
                list_output += line + "\n"
        
        return list_output
    
    def _format_as_report(self, output: str) -> str:
        """Format output as report"""
        return f"""
# 处理结果报告

## 概述
{output}

## 总结
以上是基于您的需求生成的处理结果。如果您需要更多详细信息或有其他问题，请随时告知。
        """.strip()
    
    def _default_formatting(self, output: str) -> str:
        """Default output formatting"""
        # Clean up extra whitespace and ensure proper structure
        lines = output.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        return '\n\n'.join(cleaned_lines)
    
    def _fallback_integration(self, organized_results: Dict[str, Any], user_query: str) -> str:
        """Fallback integration method if AI generation fails"""
        
        summary = organized_results["summary"]
        key_outputs = organized_results["key_outputs"]
        
        fallback_output = f"""
基于您的问题："{user_query}"

系统执行了 {summary['total_tasks']} 个任务，其中 {summary['successful_tasks']} 个成功，{summary['failed_tasks']} 个失败。

主要结果：
"""
        
        for output in key_outputs[:5]:  # Limit to top 5 outputs
            fallback_output += f"\n任务 {output['task_id']}: {output['output']}\n"
        
        if summary['failed_tasks'] > 0:
            fallback_output += f"\n注意：有 {summary['failed_tasks']} 个任务执行失败，可能影响结果的完整性。"
        
        fallback_output += "\n\n如需更详细的分析，请重新查询或联系技术支持。"
        
        return fallback_output.strip()