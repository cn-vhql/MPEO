"""
Database layer for task persistence using SQLite
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from .models import TaskSession, TaskGraph, ExecutionResults, SystemConfig


class DatabaseManager:
    """SQLite database manager for task persistence"""
    
    def __init__(self, db_path: str = "mpeo.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_query TEXT NOT NULL,
                    task_graph TEXT,
                    execution_results TEXT,
                    final_output TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT DEFAULT 'created'
                )
            ''')
            
            # Create task_graphs table for versioning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_graphs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    graph_version INTEGER DEFAULT 1,
                    graph_data TEXT NOT NULL,
                    is_final BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Create config table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Create logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    component TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    details TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            conn.commit()
    
    def save_session(self, session: TaskSession) -> bool:
        """Save or update a task session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                session.updated_at = datetime.now()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO sessions 
                    (session_id, user_query, task_graph, execution_results, final_output, created_at, updated_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session.session_id,
                    session.user_query,
                    json.dumps(session.task_graph.dict()) if session.task_graph else None,
                    json.dumps(session.execution_results.dict()) if session.execution_results else None,
                    session.final_output,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    session.status
                ))
                
                conn.commit()
                return True
        except Exception as e:
            self.log_error("database", f"Failed to save session: {str(e)}")
            return False
    
    def load_session(self, session_id: str) -> Optional[TaskSession]:
        """Load a task session by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT session_id, user_query, task_graph, execution_results, final_output, 
                           created_at, updated_at, status
                    FROM sessions WHERE session_id = ?
                ''', (session_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_session(row)
                return None
        except Exception as e:
            self.log_error("database", f"Failed to load session: {str(e)}")
            return None
    
    def list_sessions(self, limit: int = 50, status: Optional[str] = None) -> List[TaskSession]:
        """List recent sessions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT session_id, user_query, task_graph, execution_results, final_output,
                           created_at, updated_at, status
                    FROM sessions
                '''
                params = []
                
                if status:
                    query += ' WHERE status = ?'
                    params.append(status)
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_session(row) for row in rows]
        except Exception as e:
            self.log_error("database", f"Failed to list sessions: {str(e)}")
            return []
    
    def save_task_graph(self, session_id: str, graph: TaskGraph, is_final: bool = False) -> bool:
        """Save a task graph version"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current version
                cursor.execute('''
                    SELECT MAX(graph_version) FROM task_graphs WHERE session_id = ?
                ''', (session_id,))
                result = cursor.fetchone()
                current_version = result[0] or 0
                new_version = current_version + 1
                
                cursor.execute('''
                    INSERT INTO task_graphs 
                    (session_id, graph_version, graph_data, is_final, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    new_version,
                    json.dumps(graph.dict()),
                    is_final,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                return True
        except Exception as e:
            self.log_error("database", f"Failed to save task graph: {str(e)}")
            return False
    
    def get_task_graphs(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all task graph versions for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT graph_version, graph_data, is_final, created_at
                    FROM task_graphs 
                    WHERE session_id = ?
                    ORDER BY graph_version
                ''', (session_id,))
                
                rows = cursor.fetchall()
                return [
                    {
                        'version': row[0],
                        'graph_data': json.loads(row[1]),
                        'is_final': bool(row[2]),
                        'created_at': row[3]
                    }
                    for row in rows
                ]
        except Exception as e:
            self.log_error("database", f"Failed to get task graphs: {str(e)}")
            return []
    
    def save_config(self, key: str, value: Any) -> bool:
        """Save configuration value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO config (key, value, updated_at)
                    VALUES (?, ?, ?)
                ''', (key, json.dumps(value), datetime.now().isoformat()))
                
                conn.commit()
                return True
        except Exception as e:
            self.log_error("database", f"Failed to save config: {str(e)}")
            return False
    
    def load_config(self, key: str, default: Any = None) -> Any:
        """Load configuration value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM config WHERE key = ?', (key,))
                row = cursor.fetchone()
                
                if row:
                    return json.loads(row[0])
                return default
        except Exception as e:
            self.log_error("database", f"Failed to load config: {str(e)}")
            return default
    
    def log_event(self, session_id: Optional[str], component: str, operation: str, details: Optional[str] = None):
        """Log system event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO logs (session_id, component, operation, details, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    component,
                    operation,
                    details,
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            print(f"Failed to log event: {str(e)}")
    
    def log_error(self, component: str, message: str):
        """Log error message"""
        self.log_event(None, component, "ERROR", message)
    
    def get_logs(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get system logs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if session_id:
                    cursor.execute('''
                        SELECT session_id, component, operation, details, timestamp
                        FROM logs 
                        WHERE session_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (session_id, limit))
                else:
                    cursor.execute('''
                        SELECT session_id, component, operation, details, timestamp
                        FROM logs 
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                
                rows = cursor.fetchall()
                return [
                    {
                        'session_id': row[0],
                        'component': row[1],
                        'operation': row[2],
                        'details': row[3],
                        'timestamp': row[4]
                    }
                    for row in rows
                ]
        except Exception as e:
            self.log_error("database", f"Failed to get logs: {str(e)}")
            return []
    
    def _row_to_session(self, row) -> TaskSession:
        """Convert database row to TaskSession object"""
        session_id, user_query, task_graph_json, execution_results_json, final_output, created_at, updated_at, status = row
        
        task_graph = None
        if task_graph_json:
            task_graph = TaskGraph.parse_raw(task_graph_json)
        
        execution_results = None
        if execution_results_json:
            execution_results = ExecutionResults.parse_raw(execution_results_json)
        
        return TaskSession(
            session_id=session_id,
            user_query=user_query,
            task_graph=task_graph,
            execution_results=execution_results,
            final_output=final_output,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            status=status
        )