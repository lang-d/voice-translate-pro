#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any
from utils.logger import logger
from utils.model_manager import model_manager

class TranslationManager:
    """翻译管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化翻译管理器
        
        Args:
            config: 配置项，包含：
            {
                "system": {
                    "gpu": {...},
                    "performance": {...}
                },
                "user": {
                    "translation": {
                        "engine": "nllb",
                        "model": "base",
                        "source_language": "zh",
                        "target_language": "en"
                    }
                }
            }
        """
        self.config = config
        self.engine = None
        
        # 验证配置
        if not self._validate_config():
            raise ValueError("配置无效")
    
    def _validate_config(self) -> bool:
        """验证配置是否有效"""
        try:
            return (
                "system" in self.config 
                and "gpu" in self.config["system"]
                and "performance" in self.config["system"]
                and "user" in self.config
                and "translation" in self.config["user"]
            )
        except Exception as e:
            logger.exception(f"配置验证失败: {str(e)}")
            return False
    
    def pre(self) -> 'TranslationManager':
        """
        预初始化引擎，使用配置文件中的设置
        
        Returns:
            self: 支持链式调用
        """
        try:
            # 如果引擎已存在，先销毁
            if self.engine is not None:
                self.stop()
            
            # 获取用户翻译配置
            trans_config = self.config["user"]["translation"]
            engine_name = trans_config["engine"]
            
            # 根据引擎名称导入对应的引擎类
            if engine_name == "nllb":
                from .nllb_translator import NLLBTranslator
                engine_class = NLLBTranslator
            else:
                raise ValueError(f"不支持的翻译引擎: {engine_name}")
            
            # 创建引擎实例
            self.engine = engine_class()
            
            # 合并系统配置和用户配置
            engine_config = {
                "model": trans_config["model"],
                "source_language": trans_config["source_language"],
                "target_language": trans_config["target_language"],
                "gpu": self.config["system"]["gpu"],
                "performance": self.config["system"]["performance"]
            }
            
            # 配置引擎
            if not self.engine.configure(**engine_config):
                raise RuntimeError(f"引擎 {engine_name} 配置失败")
                
            logger.info(f"翻译引擎 {engine_name} 初始化成功")
            
            return self
            
        except Exception as e:
            logger.exception(f"预初始化失败: {str(e)}")
            self.engine = None
            raise
    
    def translate(self, text: str) -> Optional[str]:
        """执行翻译"""
        try:
            if self.engine is None:
                raise RuntimeError("翻译引擎未初始化")
            return self.engine.translate(text)
        except Exception as e:
            logger.exception(f"翻译失败: {str(e)}")
            return None
    
    def stop(self) -> None:
        """停止并销毁引擎"""
        try:
            if self.engine is not None:
                if hasattr(self.engine, 'cleanup'):
                    self.engine.cleanup()
                self.engine = None
                logger.info("翻译引擎已停止")
        except Exception as e:
            logger.exception(f"停止翻译引擎失败: {str(e)}")

    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动清理"""
        self.stop()

    