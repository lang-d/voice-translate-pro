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

            # 获取模型信息
            model_info = model_manager.get_model_info("asr", "whisper", model)
            if not model_info:
                logger.error(f"无法获取模型信息: {model}")
                return False

            # 处理模型路径/名称
            # 注意：这里是关键修改 - 我们判断是使用内置模型名称还是本地路径
            model_path = model_info["save_path"]
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False

            # 这里是关键部分 - 判断模型加载方式
            # 检查是否为标准Whisper模型名称
            std_models = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small',
                          'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3',
                          'large', 'large-v3-turbo', 'turbo']

            # 获取文件或目录名部分
            model_name = os.path.basename(model_path)

            logger.info(f"正在加载Whisper模型: {model} (设备: {self.device})")
            try:
                if model_name in std_models:
                    # 如果是标准模型名称，直接使用名称而不是路径
                    self.model = whisper.load_model(model_name, device=self.device)
                    logger.info(f"使用标准Whisper模型: {model_name}")
                else:
                    # 如果是自定义模型，使用完整路径
                    # 确保路径指向正确的模型文件，而不仅仅是目录
                    if os.path.isdir(model_path):
                        # 如果是目录，检查是否有模型文件
                        weight_files = [f for f in os.listdir(model_path)
                                        if f.endswith('.pt') or f.endswith('.bin')]
                        if weight_files:
                            model_file = os.path.join(model_path, weight_files[0])
                            self.model = whisper.load_model(model_file, device=self.device)
                            logger.info(f"使用本地模型文件: {model_file}")
                        else:
                            # 如果目录中没有权重文件，尝试直接使用内置模型
                            self.model = whisper.load_model(model, device=self.device)
                            logger.info(f"未找到本地模型文件，尝试使用内置模型: {model}")
                    else:
                        # 直接使用模型路径
                        self.model = whisper.load_model(model_path, device=self.device)
                        logger.info(f"使用本地模型路径: {model_path}")
            except Exception as e:
                logger.exception(f"加载模型失败: {str(e)}")
                # 如果加载失败，尝试使用内置模型名称
                try:
                    logger.info(f"尝试直接加载内置模型: {model}")
                    self.model = whisper.load_model(model, device=self.device)
                except Exception as e2:
                    logger.exception(f"再次加载失败: {str(e2)}")
                    return False

            # 保留原有的其他配置和JIT优化代码
            # 保存语言设置
            self.language = language
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
            logger.info(f"识别结果: {result}")
            
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