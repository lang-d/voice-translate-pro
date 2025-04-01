#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from utils.config import config
from utils.logger import logger
from utils.model_manager import model_manager
import time


class ASRManager:
    """ASR管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化ASR管理器
        
        Args:
            config: 配置项，包含：
            {
                "system": {
                    "gpu": {...},
                    "performance": {...}
                },
                "user": {
                    "asr": {
                        "engine": "whisper",
                        "model": "base",
                        "language": "zh"
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
                and "asr" in self.config["user"]
            )
        except Exception as e:
            logger.exception(f"配置验证失败: {str(e)}")
            return False
    
    def pre(self) -> 'ASRManager':
        """
        预初始化引擎，使用配置文件中的设置
        
        Returns:
            self: 支持链式调用
        """
        try:
            # 如果引擎已存在，先销毁
            if self.engine is not None:
                self.stop()
            
            # 获取用户ASR配置
            asr_config = self.config["user"]["asr"]
            engine_name = asr_config["engine"]
            
            # 根据引擎名称导入对应的引擎类
            if engine_name == "whisper":
                from .whisper_asr import WhisperASR
                engine_class = WhisperASR
            elif engine_name == "faster_whisper":
                from .faster_whisper_asr import FasterWhisperASR
                engine_class = FasterWhisperASR
            elif engine_name == "vosk":
                from .vosk_asr import VoskASR
                engine_class = VoskASR
            else:
                raise ValueError(f"不支持的ASR引擎: {engine_name}")
            
            # 创建引擎实例
            self.engine = engine_class()
            
            # 合并系统配置和用户配置
            engine_config = {
                "model": asr_config["model"],
                "language": asr_config["language"],
                "gpu": self.config["system"]["gpu"],
                "performance": self.config["system"]["performance"]
            }
            
            # 配置引擎
            if not self.engine.configure(**engine_config):
                raise RuntimeError(f"引擎 {engine_name} 配置失败")
                
            logger.info(f"ASR引擎 {engine_name} 初始化成功")
            
            return self
            
        except Exception as e:
            logger.exception(f"预初始化失败: {str(e)}")
            self.engine = None
            raise
    
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """执行语音识别"""
        try:
            if self.engine is None:
                raise RuntimeError("ASR引擎未初始化")
            return self.engine.transcribe(audio)
        except Exception as e:
            logger.exception(f"语音识别失败: {str(e)}")
            return None
    
    def stop(self) -> None:
        """停止并销毁引擎"""
        try:
            if self.engine is not None:
                if hasattr(self.engine, 'cleanup'):
                    self.engine.cleanup()
                self.engine = None
                logger.info("ASR引擎已停止")
        except Exception as e:
            logger.exception(f"停止ASR引擎失败: {str(e)}")

    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动清理"""
        self.stop()

   