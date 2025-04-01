#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from utils.logger import logger

class BaseTranslator(ABC):
    """翻译引擎基类"""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        self.source_language = None
        self.target_language = None
    
    @abstractmethod
    def configure(self, model: str, source_language: str, target_language: str, gpu: Dict[str, Any], performance: Dict[str, Any]) -> bool:
        """
        配置翻译引擎
        
        Args:
            model: 模型名称
            source_language: 源语言代码
            target_language: 目标语言代码
            gpu: GPU配置，包含：
                - enabled: 是否启用GPU
                - device: 设备名称
                - memory_fraction: 显存使用比例
            performance: 性能配置，包含：
                - use_half_precision: 是否使用半精度
                - use_jit: 是否使用JIT
                - num_workers: 工作线程数
                - batch_size: 批处理大小
        
        Returns:
            bool: 配置是否成功
        """
        pass
    
    @abstractmethod
    def translate(self, text: str) -> Optional[str]:
        """
        执行翻译
        
        Args:
            text: 要翻译的文本
            
        Returns:
            str: 翻译结果，失败返回None
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass
    
    def _validate_text(self, text: str) -> bool:
        """验证文本"""
        try:
            if not isinstance(text, str):
                logger.error("输入必须是字符串")
                return False
            if not text.strip():
                logger.error("输入文本不能为空")
                return False
            return True
        except Exception as e:
            logger.exception(f"文本验证失败: {str(e)}")
            return False 