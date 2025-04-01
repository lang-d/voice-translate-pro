#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import edge_tts
import asyncio
from typing import Optional, Dict, List, Any, Tuple
from core.tts.base_tts_engine import BaseTTSEngine
from core.tts.audio_utils import AudioUtils
from utils.logger import logger


class EdgeTTSEngine(BaseTTSEngine):
    """Edge TTS引擎实现"""

    def __init__(self):
        """初始化Edge TTS引擎"""
        super().__init__()
        self.use_cpu = False  # Edge TTS是云服务，不需要CPU设置
        self.rate = "+0%"  # 语速
        self.volume = "+0%"  # 音量
        self.audio_format = "audio-24khz-48kbitrate-mono-mp3"  # 音频格式
        self.max_retries = 3  # 最大重试次数
        self.api_version = "v1"  # API版本
        self.available_voices = {}  # 可用语音列表
        self._load_voices()

    def configure(self, model: str, voice: str, language: str, gpu: Dict[str, Any], 
                 performance: Dict[str, Any], audio: Dict[str, Any]) -> bool:
        """配置TTS引擎"""
        try:
            # 调用父类的配置方法
            if not super().configure(model, voice, language, gpu, performance, audio):
                return False
            
            # 设置API版本
            if "api_version" in performance:
                self.api_version = performance["api_version"]
            
            # 设置语速 (speed: 0.5-2.0 -> rate: -50%~+100%)
            if "speed" in performance:
                speed = performance["speed"]
                if 0.5 <= speed <= 2.0:
                    rate_percent = int((speed - 1.0) * 100)
                    self.rate = f"{rate_percent:+d}%"
                else:
                    logger.warning(f"语速 {speed} 超出范围 [0.5, 2.0]，将使用默认值")
                    self.rate = "+0%"
            
            # 设置音量 (volume: 0-2.0 -> volume: -100%~+100%)
            if "volume" in audio["output"]:
                volume = audio["output"]["volume"]
                if 0.0 <= volume <= 2.0:
                    volume_percent = int((volume - 1.0) * 100)
                    self.volume = f"{volume_percent:+d}%"
                else:
                    logger.warning(f"音量 {volume} 超出范围 [0.0, 2.0]，将使用默认值")
                    self.volume = "+0%"
            
            # 设置音频格式
            if "format" in audio["output"]:
                self.audio_format = audio["output"]["format"]
            
            # 设置最大重试次数
            if "max_retries" in performance:
                self.max_retries = max(1, min(5, performance["max_retries"]))
            
            # 设置声音
            if voice and not self._set_voice(voice):
                return False
            
            self.initialized = True
            logger.info(f"Edge TTS配置完成: 声音={voice}, 语言={language}, 语速={self.rate}, 音量={self.volume}")
            return True
            
        except Exception as e:
            logger.exception(f"Edge TTS配置失败: {str(e)}")
            return False

    def _load_voices(self):
        """加载可用的语音列表"""
        try:
            # 获取所有可用的语音
            voices = asyncio.run(edge_tts.list_voices())

            # 按语言组织语音
            for voice in voices:
                lang = voice["Locale"]
                if lang not in self.available_voices:
                    self.available_voices[lang] = []
                self.available_voices[lang].append({
                    "id": voice["Name"],
                    "name": voice["ShortName"],
                    "gender": voice["Gender"],
                    "language": voice["Locale"],
                    "type": "edge_tts"
                })

            logger.info(f"已加载 {len(voices)} 个Edge TTS语音")

        except Exception as e:
            logger.exception(f"加载Edge TTS语音列表失败: {str(e)}")

    def _set_voice(self, voice_id: str) -> bool:
        """设置语音"""
        try:
            # 验证语音ID是否有效
            for voices in self.available_voices.values():
                for voice in voices:
                    if voice["id"] == voice_id:
                        self.voice = voice_id
                        logger.info(f"设置Edge TTS语音: {voice_id}")
                        return True

            logger.warning(f"未找到Edge TTS语音: {voice_id}")
            return False

        except Exception as e:
            logger.exception(f"设置Edge TTS语音失败: {str(e)}")
            return False


    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """执行语音合成"""
        try:
            if not self.initialized:
                raise RuntimeError("Edge TTS引擎未初始化")
            
            if not self._validate_text(text):
                return None
            
            if not self.voice:
                raise RuntimeError("未设置Edge TTS语音")
            
            # 创建通信实例
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume,
                audio_format=self.audio_format
            )
            
            # 重试机制
            for attempt in range(self.max_retries):
                try:
                    # 获取音频数据
                    audio_data = asyncio.run(communicate.get_audio())
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logger.warning(f"获取音频失败，尝试重试 ({attempt + 1}/{self.max_retries}): {str(e)}")
                    asyncio.sleep(1)
            
            # 转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # 验证音频
            if not self._validate_audio(audio_float):
                return None
            
            # 音频增强
            audio_float = self._enhance_audio(audio_float)
            
            return audio_float
            
        except Exception as e:
            logger.exception(f"Edge TTS合成失败: {str(e)}")
            return None

    def cleanup(self) -> None:
        """清理资源"""
        try:
            self.available_voices.clear()
            logger.info("Edge TTS资源已清理")
        except Exception as e:
            logger.exception(f"清理资源失败: {str(e)}")


    def set_api_version(self, version: str) -> bool:
        """设置API版本"""
        try:
            if version in ["v1", "v2"]:
                self.api_version = version
                logger.info(f"已设置Edge TTS API版本: {version}")
                return True
            else:
                logger.warning(f"不支持的API版本: {version}")
                return False
        except Exception as e:
            logger.exception(f"设置API版本失败: {str(e)}")
            return False


   



