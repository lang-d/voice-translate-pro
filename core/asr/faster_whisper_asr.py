#!/usr/bin/env python
# -*- coding: utf-8 -*-

from faster_whisper import WhisperModel
from typing import Optional, Dict, Any
import numpy as np
import torch
from utils.logger import logger
from utils.model_manager import model_manager
from .base_asr import BaseASR
import os

class FasterWhisperASR(BaseASR):
    """Faster-Whisper ASR引擎"""
    
    def __init__(self):
        super().__init__()
        self.device = None
        self.compute_type = None
    
    def configure(self, model: str, language: str, gpu: Dict[str, Any], performance: Dict[str, Any]) -> bool:
        """配置ASR引擎"""
        try:
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("asr", "faster_whisper", model):
                logger.error(f"Faster-Whisper模型 {model} 未下载")
                return False
            
            # 设置设备和计算类型
            if gpu["enabled"] and torch.cuda.is_available():
                self.device = gpu["device"] if gpu["device"] != "auto" else "cuda"
                self.compute_type = "float16" if performance["use_half_precision"] else "float32"
            else:
                self.device = "cpu"
                self.compute_type = "float32"
            
            # 获取模型信息和路径
            model_info = model_manager.get_model_info("asr", "faster_whisper", model)
            if not model_info:
                logger.error(f"无法获取模型信息: {model}")
                return False
            
            model_path = model_info["save_path"]
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False
            
            # 加载模型
            logger.info(f"正在加载Faster-Whisper模型: {model} (设备: {self.device}, 计算类型: {self.compute_type})")
            try:
                self.model = WhisperModel(
                    model_size_or_path=model_path,
                    device=self.device,
                    compute_type=self.compute_type,
                    num_workers=performance["num_workers"],
                    download_root=None  # 不使用默认下载路径
                )
            except Exception as e:
                logger.exception(f"加载模型失败: {str(e)}")
                return False
            
            # 保存语言设置
            self.language = language
            
            self.initialized = True
            logger.info(f"Faster-Whisper ASR配置完成: model={model}, language={language}")
            return True
            
        except Exception as e:
            logger.exception(f"Faster-Whisper ASR配置失败: {str(e)}")
            return False
    
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """执行语音识别"""
        try:
            if not self.initialized:
                raise RuntimeError("ASR引擎未初始化")
            
            # 验证音频
            if not self._validate_audio(audio):
                return None
            
            # 标准化音频
            audio = self._normalize_audio(audio)
            
            # 执行识别
            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True
            )
            
            # 合并结果
            text = " ".join(segment.text for segment in segments)
            return text.strip()
            
        except Exception as e:
            logger.exception(f"语音识别失败: {str(e)}")
            return None
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = None
            self.initialized = False
            logger.info("Faster-Whisper ASR资源已清理")
        except Exception as e:
            logger.exception(f"清理资源失败: {str(e)}")