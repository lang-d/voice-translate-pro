#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from vosk import Model, KaldiRecognizer
from typing import Optional, Dict, Any
import numpy as np
from utils.logger import logger
from utils.model_manager import model_manager
from .base_asr import BaseASR

class VoskASR(BaseASR):
    """Vosk ASR引擎"""
    
    def __init__(self):
        super().__init__()
        self.recognizer = None
        self.sample_rate = 16000  # Vosk要求16kHz采样率
    
    def configure(self, model: str, language: str, gpu: Dict[str, Any], performance: Dict[str, Any]) -> bool:
        """配置ASR引擎"""
        try:
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("asr", "vosk", model):
                logger.error(f"Vosk模型 {model} 未下载")
                return False
            
            # 获取模型信息和路径
            model_info = model_manager.get_model_info("asr", "vosk", model)
            if not model_info:
                logger.error(f"无法获取模型信息: {model}")
                return False
            
            model_path = model_info["save_path"]
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False
            
            # 加载模型
            logger.info(f"正在加载Vosk模型: {model}")
            try:
                self.model = Model(model_path)
                self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
                self.recognizer.SetWords(True)  # 启用词级时间戳
            except Exception as e:
                logger.exception(f"加载模型失败: {str(e)}")
                return False
            
            # 保存语言设置
            self.language = language
            
            self.initialized = True
            logger.info(f"Vosk ASR配置完成: model={model}, language={language}")
            return True
            
        except Exception as e:
            logger.exception(f"Vosk ASR配置失败: {str(e)}")
            return False
    
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """执行语音识别"""
        try:
            if not self.initialized:
                raise RuntimeError("ASR引擎未初始化")
            
            # 验证音频
            if not self._validate_audio(audio):
                return None
            
            # 标准化音频并转换为bytes
            audio = self._normalize_audio(audio)
            audio_bytes = (audio * 32767).astype('int16').tobytes()
            
            # 执行识别
            if self.recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(self.recognizer.Result())
                if "text" in result:
                    return result["text"].strip()
            
            # 获取最后的部分结果
            final_result = json.loads(self.recognizer.FinalResult())
            return final_result.get("text", "").strip()
            
        except Exception as e:
            logger.exception(f"语音识别失败: {str(e)}")
            return None
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.model is not None:
                del self.recognizer
                del self.model
                self.recognizer = None
                self.model = None
            self.initialized = False
            logger.info("Vosk ASR资源已清理")
        except Exception as e:
            logger.exception(f"清理资源失败: {str(e)}")

   