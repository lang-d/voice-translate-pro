#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core/tts/bark_tts_engine.py
import os
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from core.tts.base_tts_engine import BaseTTSEngine
from core.tts.audio_utils import AudioUtils
from utils.model_manager import model_manager
from utils.logger import logger

class BarkTTSEngine(BaseTTSEngine):
    """Bark TTS引擎实现"""

    def __init__(self):
        """初始化Bark TTS引擎"""
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.config = None
        self.initialized = False
        
        # Bark特有配置
        self.temperature = 0.7
        self.waveform_length = 1024
        
        # 声音配置
        self.voice_embeddings = None  # 克隆的声音特征
        self.use_cloned_voice = False  # 是否使用克隆声音
        
        # 预定义的语音列表
        self.voice_presets = {
            "zh": [
                {"id": "v2/zh_speaker_0", "name": "中文女声", "gender": "female"},
                {"id": "v2/zh_speaker_1", "name": "中文男声", "gender": "male"},
                {"id": "v2/zh_speaker_2", "name": "中文女声2", "gender": "female"},
                {"id": "v2/zh_speaker_3", "name": "中文男声2", "gender": "male"}
            ],
            "en": [
                {"id": "v2/en_speaker_0", "name": "英文女声", "gender": "female"},
                {"id": "v2/en_speaker_1", "name": "英文男声", "gender": "male"},
                {"id": "v2/en_speaker_2", "name": "英文女声2", "gender": "female"},
                {"id": "v2/en_speaker_3", "name": "英文男声2", "gender": "male"}
            ]
        }
    
    def configure(self, model: str, voice: str, language: str, gpu: Dict[str, Any], 
                 performance: Dict[str, Any], audio: Dict[str, Any]) -> bool:
        """配置Bark TTS引擎"""
        try:
            # 调用父类配置
            if not super().configure(model, voice, language, gpu, performance, audio):
                return False
            
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("tts", "bark", model):
                logger.warning(f"模型 bark/{model} 未下载，将在首次使用时自动下载")
            
            # 获取模型信息
            model_info = model_manager.get_model_info("tts", "bark", model)
            if not model_info:
                raise ValueError(f"未找到模型信息: bark/{model}")
            
            # 加载模型
            if not self._load_model(model_info["save_path"]):
                return False
            
            # 设置语音
            if not self._set_voice(voice):
                return False
            
            # 设置性能参数
            self._setup_performance(performance)
            
            logger.info(f"Bark TTS引擎配置成功: 模型={model}, 语音={voice}, 语言={language}")
            return True
            
        except Exception as e:
            logger.exception(f"Bark TTS引擎配置失败: {str(e)}")
            return False
    
    def _load_model(self, model_path: str) -> bool:
        """加载Bark模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 加载配置
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            # 加载分词器
            tokenizer_path = os.path.join(model_path, "tokenizer.json")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"分词器文件不存在: {tokenizer_path}")
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.use_half_precision else torch.float32,
                device_map=self.device
            )
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 设置推理模式
            self.model.eval()
            
            self.initialized = True
            logger.info(f"Bark模型加载成功: {model_path}")
            return True
            
        except Exception as e:
            logger.exception(f"加载Bark模型失败: {str(e)}")
            return False
    
    def _setup_performance(self, performance: Dict[str, Any]) -> None:
        """设置性能参数"""
        try:
            # 设置批处理大小
            self.batch_size = performance.get("batch_size", 1)
            
            # 设置温度参数
            self.temperature = performance.get("temperature", 0.7)
            
            # 设置生成长度
            self.waveform_length = performance.get("waveform_length", 1024)
            
            logger.info(f"性能参数已设置: batch_size={self.batch_size}, "
                       f"temperature={self.temperature}, "
                       f"waveform_length={self.waveform_length}")
            
        except Exception as e:
            logger.exception(f"设置性能参数失败: {str(e)}")
    
    def _set_voice(self, voice_id: str) -> bool:
        """设置语音"""
        try:
            # 检查是否是克隆的声音路径
            if os.path.exists(voice_id):
                return self._load_voice_profile(voice_id)
            
            # 验证预设语音ID是否有效
            valid_voices = []
            for voices in self.voice_presets.values():
                valid_voices.extend([v["id"] for v in voices])
            
            if voice_id not in valid_voices:
                logger.warning(f"无效的语音ID: {voice_id}，将使用默认语音")
                voice_id = self.voice_presets[self.language][0]["id"]
            
            self.voice = voice_id
            self.use_cloned_voice = False
            return True
            
        except Exception as e:
            logger.exception(f"设置语音失败: {str(e)}")
            return False
    
    def _load_voice_profile(self, profile_path: str) -> bool:
        """加载声音配置文件"""
        try:
            if not os.path.exists(profile_path):
                logger.error(f"声音配置文件不存在: {profile_path}")
                return False
                
            # 加载声音特征
            self.voice_embeddings = torch.load(profile_path, map_location=self.device)
            self.use_cloned_voice = True
            logger.info(f"已加载克隆声音配置: {profile_path}")
            return True
            
        except Exception as e:
            logger.exception(f"加载声音配置失败: {str(e)}")
            return False
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """执行语音合成"""
        try:
            if not self.initialized:
                raise RuntimeError("Bark引擎未初始化")
            
            if not self._validate_text(text):
                return None
            
            # 分词
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # 添加声音特征
            if self.use_cloned_voice and self.voice_embeddings is not None:
                inputs["voice_embeddings"] = self.voice_embeddings
            else:
                inputs["speaker_id"] = self.voice
            
            # 生成音频
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.waveform_length,
                    do_sample=True,
                    temperature=self.temperature,
                    num_return_sequences=1
                )
            
            # 转换为音频波形
            audio_data = self.model.decoder(outputs[0]).cpu().numpy()
            
            # 音频增强
            audio_data = self._enhance_audio(audio_data)
            
            return audio_data
            
        except Exception as e:
            logger.exception(f"语音合成失败: {str(e)}")
            return None
    
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if self.voice_embeddings is not None:
                del self.voice_embeddings
                self.voice_embeddings = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.initialized = False
            logger.info("Bark TTS引擎资源已清理")
            
        except Exception as e:
            logger.exception(f"清理资源失败: {str(e)}")

    def get_sample_rate(self) -> int:
        """获取采样率"""
        return 24000  # Bark固定采样率为24000

    def _enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """应用音频增强"""
        try:
            # 使用AudioUtils进行降噪
            window_size = int(0.01 * self.get_sample_rate())  # 10ms窗口
            audio_data = AudioUtils.denoise(audio_data, window_size)

            # 应用归一化
            audio_data = AudioUtils.normalize(audio_data)

            return audio_data

        except Exception as e:
            logger.exception(f"音频增强失败: {str(e)}")
            return audio_data
    
  

    