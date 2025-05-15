#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Any
from pathlib import Path
from utils.logger import logger

class Config:
    """配置管理类"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.base_dir = Path(__file__).parent.parent.absolute()
            
            # 配置文件路径
            self.config_file = self.base_dir / "config.json"

            # 配置数据
            self.config: Dict[str, Any] = {}

            # 加载配置
            self._load_config()
    
    def _load_config(self) -> None:
        """加载所有配置"""
        try:
            # 加载系统配置和用户配置
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                logger.warning("配置文件不存在，使用默认配置")
                self._create_default_config()
            

        except json.JSONDecodeError as e:
            logger.error(f"配置文件格式错误: {str(e)}")
            self._create_default_config()
        except Exception as e:
            logger.exception(f"加载配置失败: {str(e)}")
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """创建默认配置（包含系统配置和用户配置）"""
        self.config = {
            "app": {
                "name": "Voice Translate Pro",
                "version": "1.0.0",
                "debug": False,
                "log_level": "INFO"
            },
            "system": {
                "gpu": {
                    "enabled": True,
                    "device": "auto",
                    "memory_fraction": 0.8
                },
                "performance": {
                    "use_half_precision": True,
                    "use_jit": True,
                    "num_workers": 4,
                    "batch_size": 32
                },
                "cache": {
                    "enabled": True,
                    "base_path": "cache",
                    "max_size": "10GB",
                    "cleanup_threshold": "8GB"
                }
            },
            "audio": {
                "input": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "chunk_duration": 0.5,
                    "noise_suppression": True
                },
                "output": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "volume": 1.0
                }
            },
            "paths": {
                "models": "models",
                "data": "data",
                "output": "output",
                "logs": "logs"
            },
            "user": {
                "language": {
                    "source": "zh",
                    "target": "en"
                },
                "asr": {
                    "engine": "whisper",
                    "model": "base",
                    "language": "zh",
                    "use_cache": True,
                    "cache_path": "cache/asr"
                },
                "translation": {
                    "engine": "nllb",
                    "model": "base",
                    "source_language": "zh",
                    "target_language": "en"
                },
                "tts": {
                    "engine": "f5_tts",
                    "voice": "zh-CN-XiaoxiaoNeural",
                    "language": "zh-CN",
                    "speed": 1.0,
                    "use_enhancements": True,
                    "cache_path": "cache/tts"
                },
                "ui": {
                    "theme": "light",
                    "font_size": 12,
                    "show_waveform": True,
                    "show_spectrogram": False
                }
            }
        }
        self._save_config()
    

    def _save_config(self) -> None:
        """保存配置（包含系统配置和用户配置）"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.exception(f"保存配置失败: {str(e)}")
    

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号访问"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值，支持点号访问"""
        try:
            keys = key.split('.')
            config = self.config
            
            # 设置值
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            config[keys[-1]] = value
            
            # 保存配置
            self._save_config()
                
        except Exception as e:
            logger.exception(f"设置配置失败: {str(e)}")
    

    def validate_config(self) -> bool:
        """验证配置是否有效"""
        try:
            # 验证必要的配置项
            required_keys = [
                "app.name",
                "app.version",
                "system.gpu.enabled",
                "audio.input.sample_rate",
                "audio.output.sample_rate",
                "user.language.source",
                "user.language.target"
            ]
            
            for key in required_keys:
                if self.get(key) is None:
                    logger.error(f"缺少必要的配置项: {key}")
                    return False
            
            # 验证数值范围
            if not 0 <= self.get("system.gpu.memory_fraction", 0) <= 1:
                logger.error("GPU内存比例必须在0-1之间")
                return False
                
            if self.get("audio.input.sample_rate", 0) <= 0:
                logger.error("采样率必须大于0")
                return False
                
            return True
            
        except Exception as e:
            logger.exception(f"配置验证失败: {str(e)}")
            return False

# 创建全局配置实例
config = Config() 