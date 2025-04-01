#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional
import numpy as np
from utils.logger import logger
from abc import ABC, abstractmethod
from core.tts.audio_utils import AudioUtils

class BaseTTSEngine(ABC):
    """TTS引擎基类"""
    
    def __init__(self):
        """初始化"""
        self.model = None
        self.initialized = False
        self.voice = None
        self.language = None
        self.device = "cpu"
        
        # 音频输出配置
        self.sample_rate = AudioUtils.DEFAULT_SAMPLE_RATE
        self.channels = AudioUtils.DEFAULT_CHANNELS
        self.volume = 1.0
        
        # 性能配置
        self.use_half_precision = False
        self.use_jit = False
        self.batch_size = 1
    
    @abstractmethod
    def configure(self, model: str, voice: str, language: str, gpu: Dict[str, Any], 
                 performance: Dict[str, Any], audio: Dict[str, Any]) -> bool:
        """
        配置TTS引擎
        
        Args:
            model: 模型名称
            voice: 语音ID
            language: 语言代码
            gpu: GPU配置，包含：
                - enabled: 是否启用GPU
                - device: 设备名称
                - memory_fraction: 显存使用比例
            performance: 性能配置，包含：
                - use_half_precision: 是否使用半精度
                - use_jit: 是否使用JIT
                - num_workers: 工作线程数
                - batch_size: 批处理大小
            audio: 音频配置，包含：
                - output: 输出音频配置
                    - sample_rate: 采样率
                    - channels: 声道数
                    - volume: 音量
                
        Returns:
            bool: 配置是否成功
        """
        try:
            # 设置基本参数
            self.voice = voice
            self.language = language
            
            # 设置音频输出参数
            self.sample_rate = audio["output"].get("sample_rate", AudioUtils.DEFAULT_SAMPLE_RATE)
            self.channels = audio["output"].get("channels", AudioUtils.DEFAULT_CHANNELS)
            self.volume = audio["output"].get("volume", 1.0)
            
            # 设置性能参数
            self.use_half_precision = performance.get("use_half_precision", False)
            self.use_jit = performance.get("use_jit", False)
            self.batch_size = performance.get("batch_size", 1)
            
            # 设置设备
            if gpu["enabled"]:
                import torch
                if gpu["device"] == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = gpu["device"]
                    
                if self.device.startswith("cuda"):
                    # 设置CUDA内存分配
                    fraction = gpu.get("memory_fraction", 0.8)
                    torch.cuda.set_per_process_memory_fraction(fraction)
            
            logger.info(f"基础配置已更新: 采样率={self.sample_rate}, 声道数={self.channels}, "
                       f"音量={self.volume}, 设备={self.device}")
            return True
            
        except Exception as e:
            logger.exception(f"基础配置失败: {str(e)}")
            return False
    
    @abstractmethod
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """
        执行语音合成
        
        Args:
            text: 要合成的文本
            
        Returns:
            Optional[np.ndarray]: 音频数据,如果失败则返回None
            音频数据格式:
            - shape: (samples,) 一维数组
            - dtype: float32
            - 范围: [-1, 1]
            - 采样率: self.sample_rate
            - 声道: self.channels
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass
    
    def _validate_text(self, text: str) -> bool:
        """验证文本是否有效"""
        try:
            if not text or not isinstance(text, str):
                logger.error("无效的文本输入")
                return False
                
            if len(text.strip()) == 0:
                logger.error("文本为空")
                return False
                
            # 检查文本长度
            if len(text) > 5000:  # 设置合理的最大长度限制
                logger.error(f"文本过长: {len(text)} 字符")
                return False
                
            # 检查是否包含有效字符
            if not any(c.isprintable() for c in text):
                logger.error("文本不包含有效字符")
                return False
                
            return True
            
        except Exception as e:
            logger.exception(f"文本验证失败: {str(e)}")
            return False
    
    def _validate_audio(self, audio: np.ndarray) -> bool:
        """验证音频数据是否有效"""
        return AudioUtils.validate_audio(audio)
    
    def _enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """音频增强处理"""
        try:
            return AudioUtils.enhance_audio(
                audio_data=audio_data,
                sample_rate=self.sample_rate,
                denoise=True,
                normalize=True,
                volume=self.volume
            )
        except Exception as e:
            logger.exception(f"音频增强失败: {str(e)}")
            return audio_data