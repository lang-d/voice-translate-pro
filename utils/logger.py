#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import logging.handlers
from datetime import datetime

class Logger:
    """日志管理类"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.logger = logging.getLogger("vtp")
            self.logger.setLevel(logging.INFO)
            
            # 创建日志目录
            self.log_dir = "logs"
            os.makedirs(self.log_dir, exist_ok=True)
            
            # 设置日志格式
            self.formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # 初始化日志处理器
            self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（按日期）
        date_str = datetime.now().strftime('%Y-%m-%d')
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(self.log_dir, f'app_{date_str}.log'),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        
        # 错误日志处理器
        error_handler = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(self.log_dir, f'error_{date_str}.log'),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.formatter)
        self.logger.addHandler(error_handler)
    
    def set_level(self, level: str):
        """设置日志级别"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        if level.upper() in level_map:
            self.logger.setLevel(level_map[level.upper()])
    
    def get_logger(self) -> logging.Logger:
        """获取日志记录器"""
        return self.logger

# 创建全局日志实例
logger = Logger().get_logger() 