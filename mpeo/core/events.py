"""
事件系统实现
提供现代化的事件驱动架构支持，包括事件总线、事件处理器、中间件等
"""

import asyncio
from typing import (
    Any, Dict, List, Type, TypeVar, Callable, Optional, 
    Union, Awaitable, Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import uuid
import logging
from functools import wraps
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')
EventDataType = TypeVar('EventDataType')

class EventPriority(Enum):
    """事件优先级"""
    LOWEST = 0
    LOW = 25
    NORMAL = 50
    HIGH = 75
    HIGHEST = 100

@dataclass
class EventMetadata:
    """事件元数据"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    version: str = "1.0"
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Event:
    """基础事件类"""
    data: Any
    metadata: EventMetadata = field(default_factory=EventMetadata)
    event_type: str = field(default="")
    
    def __post_init__(self):
        if not self.event_type:
            self.event_type = self.__class__.__name__

@runtime_checkable
class EventHandler(Protocol):
    """事件处理器协议"""
    
    async def handle(self, event: Event) -> None:
        """处理事件"""
        ...

class EventMiddleware(ABC):
    """事件中间件基类"""
    
    @abstractmethod
    async def process(self, event: Event, next_handler: Callable) -> Any:
        """处理事件中间件"""
        pass

class EventStore(ABC):
    """事件存储抽象基类"""
    
    @abstractmethod
    async def save_event(self, event: Event) -> None:
        """保存事件"""
        pass
    
    @abstractmethod
    async def get_events(self, event_type: Optional[str] = None, 
                        limit: int = 100) -> List[Event]:
        """获取事件"""
        pass

@dataclass
class EventHandlerDescriptor:
    """事件处理器描述符"""
    handler: Callable
    event_type: str
    priority: EventPriority = EventPriority.NORMAL
    async_handler: bool = True
    filter_func: Optional[Callable[[Event], bool]] = None
    weak_ref: bool = False

class InMemoryEventStore(EventStore):
    """内存事件存储实现"""
    
    def __init__(self, max_size: int = 10000):
        self._events: List[Event] = []
        self._max_size = max_size
        
    async def save_event(self, event: Event) -> None:
        """保存事件到内存"""
        self._events.append(event)
        # 保持最大大小限制
        if len(self._events) > self._max_size:
            self._events = self._events[-self._max_size:]
            
    async def get_events(self, event_type: Optional[str] = None, 
                        limit: int = 100) -> List[Event]:
        """从内存获取事件"""
        events = self._events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

class LoggingMiddleware(EventMiddleware):
    """日志中间件"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    async def process(self, event: Event, next_handler: Callable) -> Any:
        """记录事件日志"""
        self.logger.info(
            f"Processing event {event.event_type} with ID {event.metadata.event_id}"
        )
        
        start_time = datetime.now()
        try:
            result = await next_handler(event)
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Event {event.event_type} processed successfully in {duration:.3f}s"
            )
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(
                f"Event {event.event_type} processing failed in {duration:.3f}s: {e}"
            )
            raise

class MetricsMiddleware(EventMiddleware):
    """指标收集中间件"""
    
    def __init__(self):
        self._event_counts: Dict[str, int] = {}
        self._processing_times: Dict[str, List[float]] = {}
        
    async def process(self, event: Event, next_handler: Callable) -> Any:
        """收集事件处理指标"""
        start_time = datetime.now()
        
        # 增加事件计数
        self._event_counts[event.event_type] = self._event_counts.get(event.event_type, 0) + 1
        
        try:
            result = await next_handler(event)
            duration = (datetime.now() - start_time).total_seconds()
            
            # 记录处理时间
            if event.event_type not in self._processing_times:
                self._processing_times[event.event_type] = []
            self._processing_times[event.event_type].append(duration)
            
            return result
        except Exception:
            duration = (datetime.now() - start_time).total_seconds()
            # 即使失败也记录处理时间
            if event.event_type not in self._processing_times:
                self._processing_times[event.event_type] = []
            self._processing_times[event.event_type].append(duration)
            raise
            
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标数据"""
        metrics = {"event_counts": self._event_counts.copy()}
        
        # 计算平均处理时间
        avg_times = {}
        for event_type, times in self._processing_times.items():
            if times:
                avg_times[event_type] = sum(times) / len(times)
        metrics["average_processing_times"] = avg_times
        
        return metrics

class EventBus:
    """事件总线"""
    
    def __init__(self, event_store: Optional[EventStore] = None):
        self._handlers: Dict[str, List[EventHandlerDescriptor]] = {}
        self._middlewares: List[EventMiddleware] = []
        self._event_store = event_store or InMemoryEventStore()
        self._logger = logging.getLogger(__name__)
        
    def subscribe(
        self,
        event_type: str,
        handler: Callable,
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable[[Event], bool]] = None,
        weak_ref: bool = False
    ) -> None:
        """订阅事件"""
        descriptor = EventHandlerDescriptor(
            handler=handler,
            event_type=event_type,
            priority=priority,
            async_handler=asyncio.iscoroutinefunction(handler),
            filter_func=filter_func,
            weak_ref=weak_ref
        )
        
        if event_type not in self._handlers:
            self._handlers[event_type] = []
            
        self._handlers[event_type].append(descriptor)
        
        # 按优先级排序
        self._handlers[event_type].sort(
            key=lambda x: x.priority.value, 
            reverse=True
        )
        
        self._logger.debug(f"Subscribed handler for event {event_type}")
        
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """取消订阅事件"""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] 
                if h.handler != handler and (not h.weak_ref or h.handler() is not None)
            ]
            self._logger.debug(f"Unsubscribed handler for event {event_type}")
            
    def add_middleware(self, middleware: EventMiddleware) -> None:
        """添加中间件"""
        self._middlewares.append(middleware)
        
    def remove_middleware(self, middleware: EventMiddleware) -> None:
        """移除中间件"""
        if middleware in self._middlewares:
            self._middlewares.remove(middleware)
            
    async def publish(self, event: Event) -> None:
        """发布事件"""
        try:
            # 保存事件
            await self._event_store.save_event(event)
            
            # 获取处理器
            handlers = self._handlers.get(event.event_type, [])
            
            if not handlers:
                self._logger.debug(f"No handlers for event {event.event_type}")
                return
                
            # 创建处理链
            handler_chain = self._create_handler_chain(handlers)
            
            # 执行中间件链
            await self._execute_middleware_chain(event, handler_chain)
            
        except Exception as e:
            self._logger.error(f"Error publishing event {event.event_type}: {e}")
            raise
            
    async def publish_batch(self, events: List[Event]) -> None:
        """批量发布事件"""
        tasks = [self.publish(event) for event in events]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    def _create_handler_chain(self, handlers: List[EventHandlerDescriptor]) -> Callable:
        """创建处理器链"""
        async def execute_handlers(event: Event):
            results = []
            
            for descriptor in handlers:
                # 检查弱引用
                if descriptor.weak_ref:
                    handler_ref = descriptor.handler
                    if not callable(handler_ref):
                        continue
                    handler = handler_ref
                else:
                    handler = descriptor.handler
                    
                # 应用过滤器
                if descriptor.filter_func and not descriptor.filter_func(event):
                    continue
                    
                try:
                    if descriptor.async_handler:
                        await handler(event)
                    else:
                        handler(event)
                    results.append(True)
                except Exception as e:
                    self._logger.error(
                        f"Error in handler for event {event.event_type}: {e}"
                    )
                    results.append(False)
                    
            return results
            
        return execute_handlers
        
    async def _execute_middleware_chain(self, event: Event, final_handler: Callable) -> Any:
        """执行中间件链"""
        if not self._middlewares:
            return await final_handler(event)
            
        # 创建中间件链
        middleware_chain = final_handler
        
        for middleware in reversed(self._middlewares):
            current_middleware = middleware
            current_handler = middleware_chain
            
            async def chain_handler(evt: Event, mw=current_middleware, next_handler=current_handler):
                return await mw.process(evt, next_handler)
                
            middleware_chain = chain_handler
            
        return await middleware_chain(event)
        
    def get_handler_count(self, event_type: str) -> int:
        """获取事件处理器数量"""
        return len(self._handlers.get(event_type, []))
        
    def get_subscribed_events(self) -> List[str]:
        """获取已订阅的事件类型"""
        return list(self._handlers.keys())

# 全局事件总线实例
_global_event_bus = EventBus()

def get_event_bus() -> EventBus:
    """获取全局事件总线"""
    return _global_event_bus

def event_handler(event_type: str, priority: EventPriority = EventPriority.NORMAL):
    """事件处理器装饰器"""
    def decorator(func: Callable) -> Callable:
        _global_event_bus.subscribe(event_type, func, priority)
        return func
    return decorator

def weak_event_handler(event_type: str, priority: EventPriority = EventPriority.NORMAL):
    """弱引用事件处理器装饰器"""
    def decorator(func: Callable) -> Callable:
        _global_event_bus.subscribe(event_type, func, priority, weak_ref=True)
        return func
    return decorator

async def publish_event(event: Event) -> None:
    """发布事件的便捷函数"""
    await _global_event_bus.publish(event)

async def publish_events(events: List[Event]) -> None:
    """批量发布事件的便捷函数"""
    await _global_event_bus.publish_batch(events)

# 预定义事件类型
@dataclass
class TaskStartedEvent(Event):
    """任务开始事件"""
    data: Dict[str, Any]
    task_id: str = ""
    task_name: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.task_id:
            self.task_id = self.data.get("task_id", "")
        if not self.task_name:
            self.task_name = self.data.get("task_name", "")

@dataclass
class TaskCompletedEvent(Event):
    """任务完成事件"""
    data: Dict[str, Any]
    task_id: str = ""
    task_name: str = ""
    result: Any = None
    duration: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        if not self.task_id:
            self.task_id = self.data.get("task_id", "")
        if not self.task_name:
            self.task_name = self.data.get("task_name", "")
        if self.result is None:
            self.result = self.data.get("result")
        if self.duration == 0.0:
            self.duration = self.data.get("duration", 0.0)

@dataclass
class TaskFailedEvent(Event):
    """任务失败事件"""
    data: Dict[str, Any]
    task_id: str = ""
    task_name: str = ""
    error: str = ""
    duration: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        if not self.task_id:
            self.task_id = self.data.get("task_id", "")
        if not self.task_name:
            self.task_name = self.data.get("task_name", "")
        if not self.error:
            self.error = self.data.get("error", "")
        if self.duration == 0.0:
            self.duration = self.data.get("duration", 0.0)

@dataclass
class MCPServiceConnectedEvent(Event):
    """MCP服务连接事件"""
    data: Dict[str, Any]
    service_name: str = ""
    connection_type: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.service_name:
            self.service_name = self.data.get("service_name", "")
        if not self.connection_type:
            self.connection_type = self.data.get("connection_type", "")

@dataclass
class MCPServiceDisconnectedEvent(Event):
    """MCP服务断开连接事件"""
    data: Dict[str, Any]
    service_name: str = ""
    reason: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.service_name:
            self.service_name = self.data.get("service_name", "")
        if not self.reason:
            self.reason = self.data.get("reason", "")

@dataclass
class SystemCommandEvent(Event):
    """系统命令事件"""
    data: Dict[str, Any]
    command: str = ""
    parameters: Dict[str, Any] = None
    user_id: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if not self.command:
            self.command = self.data.get("command", "")
        if self.parameters is None:
            self.parameters = self.data.get("parameters", {})
        if self.user_id is None:
            self.user_id = self.data.get("user_id")
