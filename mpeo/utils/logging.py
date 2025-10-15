"""
日志工具模块
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    设置日志系统（避免重复初始化）

    Args:
        log_level: 日志级别
        log_dir: 日志目录
    """
    # 检查是否已经初始化
    logger = logging.getLogger()
    if logger.handlers:
        # 如果已经有处理器，说明已经初始化过了，直接返回
        return logger

    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # 生成日志文件名
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_filename = log_path / f"{current_date}.log"

    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # 设置根日志器
    logger.setLevel(getattr(logging, log_level.upper()))

    # 文件处理器（所有级别）
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器（INFO及以上级别）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"日志系统初始化完成。日志文件: {log_filename}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志器

    Args:
        name: 日志器名称

    Returns:
        Logger实例
    """
    return logging.getLogger(name)