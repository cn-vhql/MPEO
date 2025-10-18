"""
依赖注入容器实现
提供现代化的依赖注入功能，支持单例、瞬态等生命周期管理
"""

import inspect
from typing import (
    Any, Dict, Type, TypeVar, Callable, Optional, 
    Union, List, get_type_hints
)
from dataclasses import dataclass
from enum import Enum
import asyncio
from functools import wraps

T = TypeVar('T')

class LifetimeScope(Enum):
    """服务生命周期范围"""
    SINGLETON = "singleton"  # 单例，整个容器生命周期内只有一个实例
    TRANSIENT = "transient"  # 瞬态，每次请求都创建新实例
    SCOPED = "scoped"       # 作用域，在特定作用域内是单例

@dataclass
class ServiceDescriptor:
    """服务描述符"""
    interface: Type
    implementation: Type
    lifetime: LifetimeScope
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    dependencies: Optional[List[Type]] = None

class DIContainer:
    """依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._building: set[Type] = set()  # 防止循环依赖
        
    def register_singleton(
        self, 
        interface: Type[T], 
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        instance: Optional[T] = None
    ) -> 'DIContainer':
        """注册单例服务"""
        if implementation is None and factory is None and instance is None:
            implementation = interface
            
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation or interface,
            lifetime=LifetimeScope.SINGLETON,
            factory=factory,
            instance=instance
        )
        
        self._services[interface] = descriptor
        if instance is not None:
            self._singletons[interface] = instance
            
        return self
        
    def register_transient(
        self, 
        interface: Type[T], 
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None
    ) -> 'DIContainer':
        """注册瞬态服务"""
        if implementation is None and factory is None:
            implementation = interface
            
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation or interface,
            lifetime=LifetimeScope.TRANSIENT,
            factory=factory
        )
        
        self._services[interface] = descriptor
        return self
        
    def register_scoped(
        self, 
        interface: Type[T], 
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None
    ) -> 'DIContainer':
        """注册作用域服务"""
        if implementation is None and factory is None:
            implementation = interface
            
        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation or interface,
            lifetime=LifetimeScope.SCOPED,
            factory=factory
        )
        
        self._services[interface] = descriptor
        return self
        
    def resolve(self, interface: Type[T]) -> T:
        """解析服务"""
        if interface not in self._services:
            raise ValueError(f"Service {interface} is not registered")
            
        descriptor = self._services[interface]
        
        # 检查循环依赖
        if interface in self._building:
            raise ValueError(f"Circular dependency detected for {interface}")
            
        # 根据生命周期返回实例
        if descriptor.lifetime == LifetimeScope.SINGLETON:
            if interface in self._singletons:
                return self._singletons[interface]
                
            if descriptor.instance is not None:
                return descriptor.instance
                
            instance = self._create_instance(descriptor)
            self._singletons[interface] = instance
            return instance
            
        elif descriptor.lifetime == LifetimeScope.SCOPED:
            if interface in self._scoped_instances:
                return self._scoped_instances[interface]
                
            instance = self._create_instance(descriptor)
            self._scoped_instances[interface] = instance
            return instance
            
        else:  # TRANSIENT
            return self._create_instance(descriptor)
            
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        self._building.add(descriptor.interface)
        
        try:
            if descriptor.factory:
                # 使用工厂方法
                return descriptor.factory()
            else:
                # 使用构造函数注入
                return self._create_with_injection(descriptor.implementation)
        finally:
            self._building.discard(descriptor.interface)
            
    def _create_with_injection(self, cls: Type) -> Any:
        """通过构造函数注入创建实例"""
        # 获取构造函数签名
        sig = inspect.signature(cls.__init__)
        
        # 获取类型提示
        type_hints = get_type_hints(cls.__init__)
        
        # 准备参数
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # 获取参数类型
            param_type = type_hints.get(param_name)
            if param_type is None:
                if param.default == inspect.Parameter.empty:
                    raise ValueError(f"Cannot determine type for parameter {param_name} in {cls}")
                continue
                
            # 解析依赖
            if param_type in self._services:
                kwargs[param_name] = self.resolve(param_type)
            elif param.default != inspect.Parameter.empty:
                # 使用默认值
                kwargs[param_name] = param.default
            else:
                raise ValueError(f"Required dependency {param_type} is not registered for {cls}")
                
        return cls(**kwargs)
        
    def create_scope(self) -> 'DIScope':
        """创建作用域"""
        return DIScope(self)
        
    def clear_scoped(self) -> None:
        """清除作用域实例"""
        self._scoped_instances.clear()
        
    def is_registered(self, interface: Type) -> bool:
        """检查服务是否已注册"""
        return interface in self._services
        
    def get_registered_services(self) -> List[Type]:
        """获取已注册的服务列表"""
        return list(self._services.keys())

class DIScope:
    """依赖注入作用域"""
    
    def __init__(self, container: DIContainer):
        self._container = container
        self._disposed = False
        
    def resolve(self, interface: Type[T]) -> T:
        """在作用域内解析服务"""
        if self._disposed:
            raise RuntimeError("Scope has been disposed")
        return self._container.resolve(interface)
        
    def dispose(self) -> None:
        """释放作用域"""
        if not self._disposed:
            self._container.clear_scoped()
            self._disposed = True
            
    def __enter__(self) -> 'DIScope':
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.dispose()

# 全局容器实例
_global_container = DIContainer()

def get_container() -> DIContainer:
    """获取全局容器"""
    return _global_container

def inject(interface: Type[T]) -> T:
    """依赖注入装饰器"""
    return _global_container.resolve(interface)

def service(
    interface: Optional[Type] = None,
    lifetime: LifetimeScope = LifetimeScope.TRANSIENT
):
    """服务注册装饰器"""
    def decorator(cls: Type[T]) -> Type[T]:
        service_interface = interface or cls
        _global_container.register_transient(service_interface, cls)
        return cls
    return decorator

def singleton_service(interface: Optional[Type] = None):
    """单例服务注册装饰器"""
    def decorator(cls: Type[T]) -> Type[T]:
        service_interface = interface or cls
        _global_container.register_singleton(service_interface, cls)
        return cls
    return decorator

def scoped_service(interface: Optional[Type] = None):
    """作用域服务注册装饰器"""
    def decorator(cls: Type[T]) -> Type[T]:
        service_interface = interface or cls
        _global_container.register_scoped(service_interface, cls)
        return cls
    return decorator

# 异步支持
async def async_inject(interface: Type[T]) -> T:
    """异步依赖注入"""
    instance = _global_container.resolve(interface)
    return instance

def async_service(
    interface: Optional[Type] = None,
    lifetime: LifetimeScope = LifetimeScope.TRANSIENT
):
    """异步服务注册装饰器"""
    def decorator(cls: Type[T]) -> Type[T]:
        service_interface = interface or cls
        
        # 标记为异步服务
        setattr(cls, '_is_async_service', True)
        
        _global_container.register_transient(service_interface, cls)
        return cls
    return decorator
