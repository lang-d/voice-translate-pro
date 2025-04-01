# core/tts/resemblyzer_voice_cloning.py
import json
import os
import time

import numpy as np
from typing import Optional,  Dict, Any
from abc import ABC, abstractmethod

import torch

from utils.logger import logger
from utils.model_manager import model_manager
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from core.tts.audio_utils import AudioUtils

class BaseVoiceCloner(ABC):
    """声音克隆基类"""
    
    def __init__(self):
        self.model = None
        self.initialized = False
        self.language = None
        
        # 音频配置
        self.sample_rate = 16000
        self.channels = 1
    
    @abstractmethod
    def configure(self, model: str, language: str, gpu: Dict[str, Any], 
                 performance: Dict[str, Any], audio: Dict[str, Any]) -> bool:
        """配置声音克隆引擎"""
        try:
            self.sample_rate = audio["output"]["sample_rate"]
            self.channels = audio["output"]["channels"]
            logger.info(f"音频配置已更新: 采样率={self.sample_rate}, 声道数={self.channels}")
            return True
        except Exception as e:
            logger.exception(f"配置失败: {str(e)}")
            return False
    
    @abstractmethod
    def extract_features(self, audio: np.ndarray, voice_id: str) -> bool:
        """提取声音特征"""
        pass
    
    @abstractmethod
    def clone_voice(self, source_audio: np.ndarray, target_audio: np.ndarray) -> Optional[np.ndarray]:
        """克隆声音"""
        pass
        
    @abstractmethod
    def save_voice(self, voice_id: str, save_path: str) -> bool:
        """保存声音特征"""
        pass
        
    @abstractmethod
    def load_voice(self, voice_id: str, load_path: str) -> bool:
        """加载声音特征"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理资源"""
        pass

    def _validate_audio(self, audio: np.ndarray) -> bool:
        """验证音频数据是否有效"""
        try:
            if not isinstance(audio, np.ndarray):
                logger.error("音频数据必须是numpy数组")
                return False
                
            if len(audio.shape) != 1:
                logger.error("音频数据必须是一维数组")
                return False
                
            if audio.dtype not in [np.float32, np.float64]:
                logger.error("音频数据必须是float32或float64类型")
                return False
                
            if len(audio) == 0:
                logger.error("音频数据为空")
                return False
                
            if np.isnan(audio).any() or np.isinf(audio).any():
                logger.error("音频数据包含无效值(NaN或Inf)")
                return False
                
            return True
            
        except Exception as e:
            logger.exception(f"音频验证失败: {str(e)}")
            return False
    
    def _enhance_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """音频增强处理"""
        try:
            return AudioUtils.enhance_audio(
                audio_data=audio_data,
                sample_rate=self.sample_rate,
                denoise=True,
                normalize=True
            )
        except Exception as e:
            logger.exception(f"音频增强失败: {str(e)}")
            return audio_data

class VoiceCloneExtractor:
    """基于Resemblyzer的声音特征提取器，用于声音克隆"""

    def __init__(self):
        """初始化声音特征提取器"""
        self._encoder = None
        logger.info("初始化声音特征提取器")

    def _load_encoder(self):
        """加载声音编码器"""
        if self._encoder is not None:
            return

        try:
            # 尝试导入库
            from resemblyzer import VoiceEncoder

            # 创建编码器实例
            self._encoder = VoiceEncoder()
            logger.debug("声音编码器加载完成")

        except ImportError as e:
            logger.exception("无法导入resemblyzer库，请执行: pip install resemblyzer")
            logger.exception("或使用: pip install git+https://github.com/resemble-ai/Resemblyzer.git")
            raise ImportError("请安装所需依赖: resemblyzer")
        except Exception as e:
            logger.exception(f"加载声音编码器失败: {e}", exc_info=True)
            raise

    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        从音频文件提取声音特征嵌入

        Args:
            audio_path: 音频文件路径

        Returns:
            声音特征嵌入向量，如果提取失败则返回None
        """
        try:
            # 加载编码器
            self._load_encoder()

            # 导入音频处理库
            from resemblyzer import preprocess_wav

            # 预处理音频
            wav = preprocess_wav(audio_path)

            # 检查音频长度
            if len(wav) < 3 * 16000:  # 至少3秒
                logger.warning(f"音频过短: {len(wav)/16000:.1f}秒，建议使用至少5秒的音频")

            # 提取嵌入
            embedding = self._encoder.embed_utterance(wav)

            logger.info(f"成功提取声音特征: {embedding.shape}")
            return embedding

        except Exception as e:
            logger.exception(f"提取声音特征失败: {e}", exc_info=True)
            return None

    def convert_embedding_for_tts(self, embedding: np.ndarray, tts_engine: str = "edge_tts") -> Any:
        """
        将通用嵌入转换为特定TTS引擎的格式

        Args:
            embedding: 声音特征嵌入
            tts_engine: TTS引擎名称

        Returns:
            转换后的嵌入
        """
        # 转换为指定引擎的格式
        if tts_engine == "edge_tts":
            # Edge TTS不支持声音克隆，但我们保留接口以保持一致性
            return embedding.tolist()
        elif tts_engine == "bark":
            # Bark需要特定格式的嵌入，但基本的Bark不支持声音克隆
            return embedding.tolist()
        else:
            # 默认返回列表形式
            return embedding.tolist()

class XTTSVoiceCloner(BaseVoiceCloner):
    """XTTS声音克隆实现"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None
        self.device = "cpu"
        self.language_code = "zh-CN"
        
    def configure(self, model: str, language: str, gpu: Dict[str, Any], 
                 performance: Dict[str, Any], audio: Dict[str, Any]) -> bool:
        """配置XTTS声音克隆引擎"""
        try:
            # 调用父类配置
            if not super().configure(model, language, gpu, performance, audio):
                return False
                
            # 设置设备
            if gpu["enabled"]:
                import torch
                if gpu["device"] == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = gpu["device"]
            
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("voice_cloning", "xtts", model):
                logger.warning(f"模型 xtts/{model} 未下载，将在首次使用时自动下载")
            
            # 获取模型信息和路径
            model_info = model_manager.get_model_info("voice_cloning", "xtts", model)
            if not model_info:
                raise ValueError(f"未找到模型信息: xtts/{model}")
            
            # 加载XTTS配置
            config_path = os.path.join(model_info["save_path"], "config.json")
            self.config = XttsConfig()
            self.config.load_json(config_path)
            
            # 初始化XTTS模型
            self.model = Xtts.init_from_config(self.config)
            self.model.load_checkpoint(self.config, 
                                    checkpoint_path=os.path.join(model_info["save_path"], "model.pth"),
                                    vocab_path=os.path.join(model_info["save_path"], "vocab.json"),
                                    speaker_file_path=os.path.join(model_info["save_path"], "speakers_xtts.pth"))
            
            # 设置语言
            self.language_code = language
            
            # 移动到指定设备
            self.model.to(self.device)
            
            # 设置推理模式
            self.model.eval()
            
            if performance["use_half_precision"]:
                self.model.half()
            
            logger.info(f"XTTS声音克隆引擎配置成功: 设备={self.device}, 语言={self.language_code}")
            return True
            
        except Exception as e:
            logger.exception(f"XTTS声音克隆引擎配置失败: {str(e)}")
            return False
    
    def extract_features(self, audio: np.ndarray, voice_id: str) -> bool:
        """提取声音特征"""
        try:
            if self.model is None:
                raise RuntimeError("XTTS模型未初始化")
                
            # 验证音频数据
            if not self._validate_audio(audio):
                return False
                
            # 提取声音特征
            speaker_embedding = self.model.get_speaker_embedding(audio, self.sample_rate)
            
            # 保存声音特征
            save_path = os.path.join("data", "voice_profiles", f"{voice_id}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存特征和元数据
            import torch
            torch.save({
                "speaker_embedding": speaker_embedding,
                "sample_rate": self.sample_rate,
                "language": self.language_code,
                "engine": "xtts"
            }, save_path)
            
            # 保存配置文件
            config_path = os.path.join("data", "voice_profiles", f"{voice_id}.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({
                    "name": voice_id,
                    "engine": "xtts",
                    "language": self.language_code,
                    "sample_rate": self.sample_rate,
                    "created_at": str(int(time.time()))
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"声音特征提取成功: {voice_id}")
            return True
            
        except Exception as e:
            logger.exception(f"声音特征提取失败: {str(e)}")
            return False
    
    def clone_voice(self, source_audio: np.ndarray, target_audio: np.ndarray) -> Optional[np.ndarray]:
        """克隆声音"""
        try:
            if self.model is None:
                raise RuntimeError("XTTS模型未初始化")
                
            # 验证音频数据
            if not self._validate_audio(source_audio) or not self._validate_audio(target_audio):
                return None
                
            # 提取源说话人特征
            source_embedding = self.model.get_speaker_embedding(source_audio, self.sample_rate)
            
            # 使用源说话人特征合成目标音频
            with torch.no_grad():
                cloned_audio = self.model.synthesize(
                    target_audio,
                    source_embedding,
                    language=self.language_code,
                    temperature=0.7
                )
            
            # 转换为numpy数组
            cloned_audio = cloned_audio.cpu().numpy().squeeze()
            
            # 音频增强
            cloned_audio = self._enhance_audio(cloned_audio)
            
            return cloned_audio
            
        except Exception as e:
            logger.exception(f"声音克隆失败: {str(e)}")
            return None
    
    def save_voice(self, voice_id: str, save_path: str) -> bool:
        """保存声音特征"""
        try:
            if not os.path.exists(save_path):
                logger.error(f"保存路径不存在: {save_path}")
                return False
                
            source_path = os.path.join("data", "voice_profiles", f"{voice_id}.pth")
            if not os.path.exists(source_path):
                logger.error(f"声音特征文件不存在: {source_path}")
                return False
                
            import shutil
            shutil.copy2(source_path, save_path)
            
            logger.info(f"声音特征保存成功: {save_path}")
            return True
            
        except Exception as e:
            logger.exception(f"声音特征保存失败: {str(e)}")
            return False
    
    def load_voice(self, voice_id: str, load_path: str) -> bool:
        """加载声音特征"""
        try:
            if not os.path.exists(load_path):
                logger.error(f"加载路径不存在: {load_path}")
                return False
                
            target_path = os.path.join("data", "voice_profiles", f"{voice_id}.pth")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            import shutil
            shutil.copy2(load_path, target_path)
            
            logger.info(f"声音特征加载成功: {voice_id}")
            return True
            
        except Exception as e:
            logger.exception(f"声音特征加载失败: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            logger.info("XTTS声音克隆引擎资源已清理")
        except Exception as e:
            logger.exception(f"资源清理失败: {str(e)}")

class ResemblyzerVoiceCloner(BaseVoiceCloner):
    """基于Resemblyzer的声音克隆实现"""
    
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.device = None
        self.voice_embeddings = {}
    
    def configure(self, model: str, language: str, gpu: Dict[str, Any], 
                 performance: Dict[str, Any], audio: Dict[str, Any]) -> bool:
        """配置声音克隆引擎"""
        try:
            # 调用父类配置
            if not super().configure(model, language, gpu, performance, audio):
                return False
            
            # 检查模型文件
            if not model_manager.is_model_downloaded("voice_cloning", "resemblyzer", model):
                logger.warning(f"模型 resemblyzer/{model} 未下载，将在首次使用时自动下载")
            
            # 设置设备
            self.device = "cuda" if gpu["enabled"] and torch.cuda.is_available() else "cpu"
            
            # 加载编码器模型
            model_info = model_manager.get_model_info("voice_cloning", "resemblyzer", model)
            model_path = os.path.join(model_info["save_path"], "encoder.pt")
            
            try:
                from resemblyzer import VoiceEncoder
                self.encoder = VoiceEncoder(device=self.device)
                self.encoder.load_state_dict(torch.load(model_path))
                self.encoder.eval()
                
                logger.info(f"Resemblyzer声音克隆引擎初始化成功: device={self.device}")
                return True
                
            except Exception as e:
                logger.exception(f"加载Resemblyzer模型失败: {str(e)}")
                return False
                
        except Exception as e:
            logger.exception(f"配置Resemblyzer声音克隆引擎失败: {str(e)}")
            return False
    
    def extract_features(self, audio: np.ndarray, voice_id: str) -> bool:
        """提取声音特征"""
        try:
            if not AudioUtils.validate_audio(audio):
                return False
            
            # 预处理音频
            audio = AudioUtils.enhance_audio(
                audio_data=audio,
                sample_rate=self.sample_rate,
                denoise=True,
                normalize=True
            )
            
            # 提取声音特征
            with torch.no_grad():
                embedding = self.encoder.embed_utterance(audio)
                self.voice_embeddings[voice_id] = embedding
                
            logger.info(f"已提取声音特征: voice_id={voice_id}")
            return True
            
        except Exception as e:
            logger.exception(f"提取声音特征失败: {str(e)}")
            return False
    
    def clone_voice(self, source_audio: np.ndarray, target_audio: np.ndarray) -> Optional[np.ndarray]:
        """克隆声音"""
        try:
            # 注意: Resemblyzer只能提取声音特征，不能直接进行声音转换
            # 这里需要与其他声音转换模型(如RVC)配合使用
            logger.warning("Resemblyzer不支持直接声音转换，请使用其他声音转换模型")
            return None
            
        except Exception as e:
            logger.exception(f"声音克隆失败: {str(e)}")
            return None
    
    def save_voice(self, voice_id: str, save_path: str) -> bool:
        """保存声音特征"""
        try:
            if voice_id not in self.voice_embeddings:
                logger.error(f"未找到声音特征: voice_id={voice_id}")
                return False
            
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存特征
            embedding = self.voice_embeddings[voice_id]
            np.save(save_path, embedding.cpu().numpy())
            
            logger.info(f"已保存声音特征: voice_id={voice_id}, path={save_path}")
            return True
            
        except Exception as e:
            logger.exception(f"保存声音特征失败: {str(e)}")
            return False
    
    def load_voice(self, voice_id: str, load_path: str) -> bool:
        """加载声音特征"""
        try:
            if not os.path.exists(load_path):
                logger.error(f"声音特征文件不存在: {load_path}")
                return False
            
            # 加载特征
            embedding = np.load(load_path)
            self.voice_embeddings[voice_id] = torch.from_numpy(embedding).to(self.device)
            
            logger.info(f"已加载声音特征: voice_id={voice_id}, path={load_path}")
            return True
            
        except Exception as e:
            logger.exception(f"加载声音特征失败: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            self.encoder = None
            self.voice_embeddings.clear()
            torch.cuda.empty_cache()
            logger.info("已清理Resemblyzer资源")
        except Exception as e:
            logger.exception(f"清理资源失败: {str(e)}")

class RVCVoiceCloner(BaseVoiceCloner):
    """基于RVC的声音克隆实现"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None
        self.device = "cpu"
        self.index = None
        self.hubert = None
        
    def configure(self, model: str, language: str, gpu: Dict[str, Any], 
                 performance: Dict[str, Any], audio: Dict[str, Any]) -> bool:
        """配置RVC声音克隆引擎"""
        try:
            # 调用父类配置
            if not super().configure(model, language, gpu, performance, audio):
                return False
                
            # 设置设备
            if gpu["enabled"]:
                import torch
                if gpu["device"] == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = gpu["device"]
            
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("voice_cloning", "rvc", model):
                logger.warning(f"模型 rvc/{model} 未下载，将在首次使用时自动下载")
            
            # 获取模型信息和路径
            model_info = model_manager.get_model_info("voice_cloning", "rvc", model)
            if not model_info:
                raise ValueError(f"未找到模型信息: rvc/{model}")
            
            # 加载RVC配置
            config_path = os.path.join(model_info["save_path"], "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            
            # 加载HuBERT模型
            import fairseq
            self.hubert = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                ["models/hubert/hubert_base.pt"],
                arg_overrides={"data": "models/hubert"}
            )[0][0].to(self.device)
            
            # 加载RVC模型
            self.model = torch.load(os.path.join(model_info["save_path"], "model.pth"), 
                                  map_location=self.device)
            
            # 加载特征索引
            import faiss
            self.index = faiss.read_index(os.path.join(model_info["save_path"], "index.bin"))
            
            # 设置推理模式
            self.model.eval()
            if performance["use_half_precision"]:
                self.model.half()
            
            logger.info(f"RVC声音克隆引擎配置成功: 设备={self.device}")
            return True
            
        except Exception as e:
            logger.exception(f"RVC声音克隆引擎配置失败: {str(e)}")
            return False
    
    def extract_features(self, audio: np.ndarray, voice_id: str) -> bool:
        """提取声音特征"""
        try:
            if self.model is None:
                raise RuntimeError("RVC模型未初始化")
                
            # 验证音频数据
            if not self._validate_audio(audio):
                return False
                
            # 使用HuBERT提取特征
            with torch.no_grad():
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
                features = self.hubert.extract_features(audio_tensor, padding_mask=None)[0]
            
            # 保存特征
            save_path = os.path.join("data", "voice_profiles", f"{voice_id}.npy")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, features.cpu().numpy())
            
            # 保存配置
            config_path = os.path.join("data", "voice_profiles", f"{voice_id}.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({
                    "name": voice_id,
                    "engine": "rvc",
                    "language": self.language,
                    "sample_rate": self.sample_rate,
                    "created_at": str(int(time.time()))
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"声音特征提取成功: {voice_id}")
            return True
            
        except Exception as e:
            logger.exception(f"声音特征提取失败: {str(e)}")
            return False
    
    def clone_voice(self, source_audio: np.ndarray, target_audio: np.ndarray) -> Optional[np.ndarray]:
        """克隆声音"""
        try:
            if self.model is None:
                raise RuntimeError("RVC模型未初始化")
                
            # 验证音频数据
            if not self._validate_audio(source_audio) or not self._validate_audio(target_audio):
                return None
                
            # 提取源说话人特征
            with torch.no_grad():
                source_tensor = torch.FloatTensor(source_audio).unsqueeze(0).to(self.device)
                source_features = self.hubert.extract_features(source_tensor, padding_mask=None)[0]
            
            # 使用特征索引进行检索增强
            source_features_np = source_features.cpu().numpy()
            _, indices = self.index.search(source_features_np, k=3)
            
            # 声音转换
            with torch.no_grad():
                target_tensor = torch.FloatTensor(target_audio).unsqueeze(0).to(self.device)
                converted = self.model.convert(
                    target_tensor,
                    source_features.to(self.device),
                    indices,
                    self.config.get("pitch_factor", 0.7),
                    self.config.get("infer_f0", True),
                    self.config.get("filter_radius", 3),
                    self.config.get("resample_sr", 0),
                    self.config.get("rms_mix_rate", 1),
                    self.config.get("protect", 0.33)
                )
            
            # 转换为numpy数组
            converted_audio = converted.cpu().numpy().squeeze()
            
            # 音频增强
            converted_audio = self._enhance_audio(converted_audio)
            
            return converted_audio
            
        except Exception as e:
            logger.exception(f"声音克隆失败: {str(e)}")
            return None
    
    def save_voice(self, voice_id: str, save_path: str) -> bool:
        """保存声音特征"""
        try:
            source_path = os.path.join("data", "voice_profiles", f"{voice_id}.npy")
            if not os.path.exists(source_path):
                logger.error(f"声音特征文件不存在: {source_path}")
                return False
            
            # 复制特征文件
            import shutil
            shutil.copy2(source_path, save_path)
            
            logger.info(f"声音特征保存成功: {save_path}")
            return True
            
        except Exception as e:
            logger.exception(f"声音特征保存失败: {str(e)}")
            return False
    
    def load_voice(self, voice_id: str, load_path: str) -> bool:
        """加载声音特征"""
        try:
            if not os.path.exists(load_path):
                logger.error(f"声音特征文件不存在: {load_path}")
                return False
            
            target_path = os.path.join("data", "voice_profiles", f"{voice_id}.npy")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # 复制特征文件
            import shutil
            shutil.copy2(load_path, target_path)
            
            logger.info(f"声音特征加载成功: {voice_id}")
            return True
            
        except Exception as e:
            logger.exception(f"声音特征加载失败: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if self.hubert is not None:
                del self.hubert
                self.hubert = None
            if self.index is not None:
                del self.index
                self.index = None
            torch.cuda.empty_cache()
            logger.info("RVC声音克隆引擎资源已清理")
        except Exception as e:
            logger.exception(f"资源清理失败: {str(e)}")