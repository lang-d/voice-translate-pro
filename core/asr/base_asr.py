
from typing import Any, Dict, Optional

import numpy as np
from utils.logger import logger
from abc import ABC, abstractmethod

class BaseASR(ABC):
    """ASR引擎基类"""
    
    def __init__(self):
        self.model = None
        self.initialized = False
    
    @abstractmethod
    def configure(self, model: str, language: str, gpu: Dict[str, Any], performance: Dict[str, Any]) -> bool:
        """
        配置ASR引擎
        
        Args:
            model: 模型名称
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
        
        Returns:
            bool: 配置是否成功
        """
        pass
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        执行语音识别
        
        Args:
            audio: 音频数据，一维numpy数组，float32类型，范围[-1, 1]
            
        Returns:
            str: 识别结果文本，失败返回None
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass
    
    def _validate_audio(self, audio: np.ndarray) -> bool:
        """验证音频数据"""
        try:
            if not isinstance(audio, np.ndarray):
                logger.exception("音频数据必须是numpy数组")
                return False
            if len(audio.shape) != 1:
                logger.exception("音频数据必须是一维数组")
                return False
            if audio.dtype not in [np.float32, np.float64]:
                logger.exception("音频数据必须是float32或float64类型")
                return False
            return True
        except Exception as e:
            logger.exception(f"音频验证失败: {str(e)}")
            return False
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """标准化音频数据"""
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            return audio
        except Exception as e:
            logger.exception(f"音频标准化失败: {str(e)}")
            return audio