#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from typing import Optional, Dict, Any
import torch
from utils.logger import logger
from utils.model_manager import model_manager
from core.tts.base_tts_engine import BaseTTSEngine

class XTTSEngine(BaseTTSEngine):
    """XTTS语音合成引擎"""
    
    def __init__(self):
        """初始化"""
        super().__init__()
        self.model = None
        self.voice_embeddings = None
        self.use_cloned_voice = False
        self.voice_dir = os.path.join("data", "voice_profiles", "yourtts")
        os.makedirs(self.voice_dir, exist_ok=True)
    
    def configure(self, model: str, voice: str, language: str, gpu: Dict[str, Any], 
                 performance: Dict[str, Any], audio: Dict[str, Any]) -> bool:
        """配置TTS引擎"""
        try:
            # 调用父类的配置方法
            if not super().configure(model, voice, language, gpu, performance, audio):
                return False
            
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("tts", "yourtts", model):
                logger.warning(f"XTTS模型 {model} 未下载，将在首次使用时自动下载")
                if not model_manager.download_model("tts", "yourtts", model):
                    raise RuntimeError(f"下载模型 {model} 失败")
            
            # 加载模型
            if not self._load_model(model):
                return False
            
            # 设置声音
            if voice and not self._set_voice(voice):
                return False
            
            self.initialized = True
            logger.info(f"XTTS配置完成: 模型={model}, 声音={voice}, 语言={language}")
            return True
            
        except Exception as e:
            logger.exception(f"XTTS配置失败: {str(e)}")
            return False


    def _load_model(self, model_name: str) -> bool:
        try:
            # 检查模型是否已下载（可选，按你的 model_manager 逻辑）
            model_info = model_manager.get_model_info("tts", "yourtts", model_name)
            if not model_info:
                logger.error(f"未找到模型信息: yourtts/{model_name}")
                return False

            hf_model_id = model_info["hf_model_id"]
            cache_dir = model_info["save_path"]

            logger.info(f"正在加载YourTTS模型: {hf_model_id} (cache_dir: {cache_dir})")
            from transformers import pipeline

            self.tts_pipe = pipeline(
                "text-to-speech",
                model=hf_model_id,
                device=0 if (self.device != "cpu") else -1,
                cache_dir=cache_dir
            )
            logger.info("YourTTS模型加载成功")
            return True

        except Exception as e:
            logger.exception(f"YourTTS模型加载失败: {str(e)}")
            self.tts_pipe = None
            return False
    
    # def _load_model(self, model_name: str) -> bool:
    #     """加载XTTS模型"""
    #     try:
    #         # 获取模型信息
    #         model_info = model_manager.get_model_info("tts", "xtts", model_name)
    #         if not model_info:
    #             logger.error(f"未找到模型信息: xtts/{model_name}")
    #             return False
    #
    #         files_path = {f["name"]:f["save_path"] for f in model_info["files"]}
    #
    #         # 获取模型文件路径
    #         model_path = model_info["save_path"]
    #         # required_files = {
    #         #     "model.pth": os.path.join(files_path["model.pth"]),
    #         #     "config.json": os.path.join(files_path["config.json"]),
    #         #     "vocab.json": os.path.join(files_path["vocab.json"]),
    #         #     "speakers_xtts.pth": os.path.join(files_path["speakers_xtts.pth"]),
    #         # }
    #         #
    #         #
    #         # # 检查所有必需的文件
    #         # for file_name, file_path in required_files.items():
    #         #     if not os.path.exists(file_path):
    #         #         logger.error(f"缺少必需文件: {file_name}")
    #         #         return False
    #
    #         logger.info(f"正在加载XTTS模型: {model_path}")
    #
    #         try:
    #             base_dir = os.path.join(model_info["save_path"], "models--coqui--XTTS-v2", "snapshots")
    #             snapshot_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    #             if not snapshot_dirs:
    #                 logger.error("未找到XTTS快照目录")
    #                 return False
    #
    #             model_path = snapshot_dirs[-1]
    #             vocoder_path = model_path
    #
    #             config_path = os.path.join(model_path, "config.json")
    #             vocab_path = os.path.join(model_path, "vocab.json")
    #             os.environ["TTS_CACHE_PATH"] = os.path.abspath(model_info["save_path"])
    #             os.environ["COQUI_TOS_AGREED"] = "1"
    #             from TTS.api import TTS
    #             self.model = TTS(
    #                 model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    #                 model_path=model_path,
    #                 config_path=config_path,
    #                 progress_bar=False,
    #                 gpu=(self.device != "cpu")
    #             ).to(self.device)
    #
    #             if self.use_half_precision and self.device != "cpu":
    #                 self.model = self.model.half()
    #
    #             if self.use_jit:
    #                 self.model = torch.jit.script(self.model)
    #
    #             logger.info("XTTS模型加载成功")
    #             return True
    #
    #         except Exception as e:
    #             logger.exception(f"XTTS模型加载失败: {str(e)}")
    #             self.model = None
    #             return False
    #
    #     except Exception as e:
    #         logger.exception(f"加载模型失败: {str(e)}")
    #         return False
        
    def _set_voice(self, voice: str) -> bool:
        """
        设置参考音频（用于YourTTS声音克隆）。
        如果未指定或文件不存在，则使用默认声音。
        """
        import os
        try:
            if not voice or not os.path.isfile(voice):
                logger.warning("未指定参考音频或文件不存在，将使用YourTTS默认声音")
                self.voice = None  # 用None表示用默认声音
                return True
            self.voice = voice
            logger.info(f"已设置YourTTS参考音频: {voice}")
            return True
        except Exception as e:
            logger.exception(f"设置YourTTS声音失败: {str(e)}")
            self.voice = None
            return False
    
    # def _set_voice(self, voice: str) -> bool:
    #     """设置声音"""
    #     try:
    #         # 检查是否是克隆声音配置文件
    #         if voice.endswith('.pth'):
    #             voice_path = os.path.join(self.voice_dir, voice)
    #             if os.path.exists(voice_path):
    #                 self.voice_embeddings = torch.load(voice_path, map_location=self.device)
    #                 self.use_cloned_voice = True
    #                 logger.info(f"已加载克隆声音配置: {voice}")
    #                 return True
    #
    #         # 否则视为参考音频文件
    #         if os.path.exists(voice):
    #             self.voice = voice
    #             self.use_cloned_voice = False
    #             logger.info(f"已设置参考音频: {voice}")
    #             return True
    #
    #         logger.error(f"无效的声音配置: {voice}")
    #         return False
    #
    #     except Exception as e:
    #         logger.exception(f"设置声音失败: {str(e)}")
    #         return False

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        try:
            if not self.model:
                raise RuntimeError("YourTTS模型未初始化")
            kwargs = {}
            if self.voice:
                kwargs["speaker_wav"] = self.voice
            result = self.model(text, **kwargs)
            return result["audio"]
        except Exception as e:
            logger.exception(f"YourTTS语音合成失败: {str(e)}")
            return None

    # def synthesize(self, text: str) -> Optional[np.ndarray]:
    #     """执行语音合成"""
    #     try:
    #         if not self.initialized:
    #             raise RuntimeError("XTTS引擎未初始化")
    #
    #         if not self._validate_text(text):
    #             return None
    #
    #         # 生成音频
    #         with torch.no_grad():
    #             if self.use_cloned_voice:
    #                 # 使用克隆声音
    #                 audio = self.model.synthesize(
    #                     text=text,
    #                     speaker_embeddings=self.voice_embeddings,
    #                     language=self.language
    #                 )
    #             else:
    #                 # 使用参考音频
    #                 audio = self.model.synthesize(
    #                     text=text,
    #                     speaker_wav=self.voice,
    #                     language=self.language
    #                 )
    #
    #         # 转换为numpy数组
    #         audio_data = audio.cpu().numpy().squeeze()
    #
    #         # 验证音频
    #         if not self._validate_audio(audio_data):
    #             return None
    #
    #         # 音频增强
    #         audio_data = self._enhance_audio(audio_data)
    #
    #         return audio_data
    #
    #     except Exception as e:
    #         logger.exception(f"语音合成失败: {str(e)}")
    #         return None
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if hasattr(self, 'voice_embeddings'):
                del self.voice_embeddings
            
            torch.cuda.empty_cache()
            logger.info("XTTS资源已清理")
            
        except Exception as e:
            logger.exception(f"清理资源失败: {str(e)}")
            
