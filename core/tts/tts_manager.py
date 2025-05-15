#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core/tts/tts_manager.py
import os
import json
import numpy as np
from typing import Dict, Optional, Any
from utils.logger import logger
from utils.model_manager import model_manager

class TTSManager:
    """TTS管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化TTS管理器
        
        Args:
            config: 配置项，包含：
            {
                "system": {
                    "gpu": {...},
                    "performance": {...}
                },
                "audio": {
                    "input": {...},
                    "output": {...}
                },
                "user": {
                    "tts": {
                        "engine": "edge_tts",
                        "model": "base",
                        "voice": "zh-CN-XiaoxiaoNeural",
                        "language": "zh-CN"
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
                and "audio" in self.config
                and "input" in self.config["audio"]
                and "output" in self.config["audio"]
                and "user" in self.config
                and "tts" in self.config["user"]
            )
        except Exception as e:
            logger.exception(f"配置验证失败: {str(e)}")
            return False
    
    def pre(self) -> 'TTSManager':
        """
        预初始化引擎，使用配置文件中的设置
        
        Returns:
            self: 支持链式调用
        """
        try:
            # 如果引擎已存在，先销毁
            if self.engine is not None:
                self.stop()
            
            # 获取用户TTS配置
            tts_config = self.config["user"]["tts"]
            logger.info(f"""tts config:{json.dumps(tts_config)}""")
            engine_name = tts_config["engine"]
            
            # 根据引擎名称导入对应的引擎类
            if engine_name == "edge_tts":
                from .edge_tts_engine import EdgeTTSEngine
                engine_class = EdgeTTSEngine
            elif engine_name == "xtts":
                from .xtts_engine import XTTSEngine
                engine_class = XTTSEngine
            elif engine_name == "f5_tts":
                from .f5_tts_engine import F5TTSEngine
                engine_class = F5TTSEngine
            elif engine_name == "bark":
                from .bark_tts_engine import BarkTTSEngine
                engine_class = BarkTTSEngine
            else:
                raise ValueError(f"不支持的TTS引擎: {engine_name}")
            
            # 创建引擎实例
            self.engine = engine_class()
            
            # 合并系统配置和用户配置
            engine_config = {
                "model": tts_config["model"],
                "voice": tts_config["voice"],
                "language": tts_config["language"],
                "gpu": self.config["system"]["gpu"],
                "performance": self.config["system"]["performance"],
                "audio": self.config["audio"]
            }
            
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("tts", engine_name, engine_config["model"]):
                logger.warning(f"模型 {engine_name}/{engine_config['model']} 未下载，将在首次使用时自动下载")
            
            # 配置引擎
            if not self.engine.configure(**engine_config):
                raise RuntimeError(f"引擎 {engine_name} 配置失败")
                
            logger.info(f"TTS引擎 {engine_name} 初始化成功")
            
            return self
            
        except Exception as e:
            logger.exception(f"预初始化失败: {str(e)}")
            self.engine = None
            raise
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """执行语音合成"""
        try:
            if self.engine is None:
                raise RuntimeError("TTS引擎未初始化")
            return self.engine.synthesize(text)
        except Exception as e:
            logger.exception(f"语音合成失败: {str(e)}")
            return None
    
    def stop(self) -> None:
        """停止并销毁引擎"""
        try:
            if self.engine is not None:
                if hasattr(self.engine, 'cleanup'):
                    self.engine.cleanup()
                self.engine = None
                logger.info("TTS引擎已停止")
        except Exception as e:
            logger.exception(f"停止TTS引擎失败: {str(e)}")

    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动清理"""
        self.stop()


        """
        获取所有可用的声音特征配置

        Returns:
            配置列表
        """
        profiles = []
        try:
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith(".json"):
                    profile_path = os.path.join(self.profiles_dir, filename)
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        profile = json.load(f)
                        profiles.append({
                            "name": profile.get("name", filename[:-5]),
                            "engine": profile.get("engine", "unknown")
                        })

            return profiles

        except Exception as e:
            logger.exception(f"获取声音特征列表时出错: {e}", exc_info=True)
            return []

   