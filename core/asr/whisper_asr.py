#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import whisper
from typing import Optional, Dict, Any
import numpy as np
from utils.logger import logger
from .base_asr import BaseASR
import os
from utils.model_manager import model_manager

class WhisperASR(BaseASR):
    """Whisper ASR引擎"""
    
    def __init__(self):
        super().__init__()
        self.device = None
    
    def configure(self, model: str, language: str, gpu: Dict[str, Any], performance: Dict[str, Any]) -> bool:
        """配置ASR引擎"""
        try:
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("asr", "whisper", model):
                logger.error(f"Whisper模型 {model} 未下载")
                return False
            
            # 设置设备
            if gpu["enabled"] and torch.cuda.is_available():
                self.device = gpu["device"] if gpu["device"] != "auto" else "cuda"
            else:
                self.device = "cpu"
            
            # 获取模型信息和路径
            model_info = model_manager.get_model_info("asr", "whisper", model)
            if not model_info:
                logger.error(f"无法获取模型信息: {model}")
                return False
            
            model_path = model_info["save_path"]
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False
            
            # 加载模型
            logger.info(f"正在加载Whisper模型: {model} (设备: {self.device})")
            try:
                self.model = whisper.load_model(model_path, device=self.device)
            except Exception as e:
                logger.exception(f"加载模型失败: {str(e)}")
                return False
            
            # 保存语言设置
            self.language = language
            
            # 性能优化
            if gpu["enabled"] and performance["use_half_precision"] and self.device != "cpu":
                self.model = self.model.half()
                logger.info("已启用半精度(FP16)模式")
            
            # JIT优化
            if performance["use_jit"]:
                try:
                    # 对编码器进行JIT优化
                    self.model.encoder.forward = torch.jit.script(self.model.encoder.forward)
                    
                    # 对解码器进行JIT优化
                    # 注意：由于解码器涉及动态控制流，我们只对其中的一些子模块进行优化
                    for block in self.model.decoder.blocks:
                        block.self_attn.forward = torch.jit.script(block.self_attn.forward)
                        block.cross_attn.forward = torch.jit.script(block.cross_attn.forward)
                        block.mlp.forward = torch.jit.script(block.mlp.forward)
                    
                    logger.info("已启用JIT优化")
                except Exception as e:
                    logger.warning(f"JIT优化失败，将使用原始模型: {str(e)}")
            
            self.initialized = True
            logger.info(f"Whisper ASR配置完成: model={model}, language={language}")
            return True
            
        except Exception as e:
            logger.exception(f"Whisper ASR配置失败: {str(e)}")
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
            result = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe"
            )
            
            if not result or "text" not in result:
                logger.exception("识别结果无效")
                return None
            
            return result["text"].strip()
            
        except Exception as e:
            logger.exception(f"语音识别失败: {str(e)}")
            return None
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.model is not None:
                # 将模型移动到CPU
                if self.device != "cpu":
                    self.model = self.model.cpu()
                # 清除CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = None
            self.initialized = False
            logger.info("Whisper ASR资源已清理")
        except Exception as e:
            logger.exception(f"清理资源失败: {str(e)}")