"""
自定义异常类
"""


class MPEOError(Exception):
    """MPEO系统基础异常"""
    pass


class MCPError(MPEOError):
    """MCP服务相关异常"""
    pass


class TaskExecutionError(MPEOError):
    """任务执行异常"""
    pass


class DatabaseError(MPEOError):
    """数据库操作异常"""
    pass


class ConfigurationError(MPEOError):
    """配置错误异常"""
    pass


class ValidationError(MPEOError):
    """数据验证异常"""
    pass