#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import soundfile as sf
from typing import Optional, Tuple, List
from scipy import signal
from utils.logger import logger
from utils.config import config

class AudioUtils:
    """音频处理工具类"""
    
    # 类常量
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg']
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    
    # 单例模式
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioUtils, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化音频处理器"""
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._load_config()
            logger.info("初始化音频处理器")
    
    def _load_config(self):
        """加载音频处理配置"""
        try:
            audio_config = config.get("audio", {})
            self.sample_rate = audio_config.get("output", {}).get("sample_rate", self.DEFAULT_SAMPLE_RATE)
            self.channels = audio_config.get("output", {}).get("channels", self.DEFAULT_CHANNELS)
            self.volume = audio_config.get("output", {}).get("volume", 1.0)
            
            # 音频增强配置
            self.use_denoise = audio_config.get("input", {}).get("noise_suppression", True)
            self.denoise_strength = 0.3  # 默认降噪强度
            self.normalize_target_level = -23.0  # EBU R128标准
            
            logger.info(f"音频处理配置已加载: 采样率={self.sample_rate}, 声道数={self.channels}")
        except Exception as e:
            logger.exception(f"加载音频处理配置失败: {str(e)}")
    
    @staticmethod
    def load_audio(file_path: str, target_sr: Optional[int] = None) -> Optional[Tuple[np.ndarray, int]]:
        """
        加载音频文件，支持自动重采样

        Args:
            file_path: 音频文件路径
            target_sr: 目标采样率，如果指定则自动重采样

        Returns:
            元组 (音频数据, 采样率)，如果加载失败则返回None
        """
        try:
            # 检查文件格式
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in AudioUtils.SUPPORTED_FORMATS:
                logger.error(f"不支持的音频格式: {ext}")
                return None
                
            # 加载音频
            audio_data, sample_rate = sf.read(file_path)
            
            # 数据类型转换
            audio_data = audio_data.astype(np.float32)
            
            # 转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # 重采样(如果需要)
            if target_sr and target_sr != sample_rate:
                audio_data = AudioUtils.resample(audio_data, sample_rate, target_sr)
                sample_rate = target_sr
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.exception(f"加载音频文件失败: {str(e)}")
            return None

    @staticmethod
    def save_audio(audio_data: np.ndarray, sample_rate: int, file_path: str, 
                  normalize: bool = True) -> bool:
        """
        保存音频文件

        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            file_path: 保存路径
            normalize: 是否在保存前归一化

        Returns:
            是否保存成功
        """
        try:
            # 创建目录
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 数据类型转换
            audio_data = audio_data.astype(np.float32)
            
            # 归一化
            if normalize:
                audio_data = AudioUtils.normalize(audio_data)
            
            # 保存音频
            sf.write(file_path, audio_data, sample_rate)
            return True
            
        except Exception as e:
            logger.exception(f"保存音频文件失败: {str(e)}")
            return False

    @staticmethod
    def validate_audio(audio: np.ndarray) -> bool:
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

    @staticmethod
    def resample(audio_data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """
        高质量重采样

        Args:
            audio_data: 音频数据
            src_rate: 源采样率
            dst_rate: 目标采样率

        Returns:
            重采样后的音频数据
        """
        try:
            if src_rate == dst_rate:
                return audio_data
                
            # 使用scipy.signal进行高质量重采样
            number_of_samples = round(len(audio_data) * float(dst_rate) / src_rate)
            resampled_data = signal.resample(audio_data, number_of_samples)
            
            return resampled_data.astype(np.float32)
            
        except Exception as e:
            logger.exception(f"重采样失败: {str(e)}")
            return audio_data

    @staticmethod
    def normalize(audio_data: np.ndarray, target_db: float = -23.0) -> np.ndarray:
        """
        音频响度归一化 (EBU R128标准)

        Args:
            audio_data: 音频数据
            target_db: 目标响度(dB)，默认-23dB (EBU R128标准)

        Returns:
            归一化后的音频数据
        """
        try:
            if len(audio_data) == 0:
                return audio_data
                
            # 计算当前响度
            rms = np.sqrt(np.mean(audio_data**2))
            current_db = 20 * np.log10(rms)
            
            # 计算增益
            gain = np.power(10, (target_db - current_db) / 20)
            
            # 应用增益并使用软裁剪避免过载
            normalized = audio_data * gain
            return np.tanh(normalized)
            
        except Exception as e:
            logger.exception(f"音频归一化失败: {str(e)}")
            return audio_data

    @staticmethod
    def denoise(audio_data: np.ndarray, sample_rate: int, strength: float = 0.3) -> np.ndarray:
        """
        高级降噪处理

        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            strength: 降噪强度 (0-1)

        Returns:
            降噪后的音频数据
        """
        try:
            try:
                import noisereduce as nr
                # 使用noisereduce进行高质量降噪
                denoised = nr.reduce_noise(
                    y=audio_data,
                    sr=sample_rate,
                    prop_decrease=strength,
                    stationary=True
                )
                return denoised
            except ImportError:
                # 降级方案：使用高通滤波器
                nyquist = sample_rate // 2
                cutoff = 100  # Hz
                normalized_cutoff = cutoff / nyquist
                
                # 设计巴特沃斯高通滤波器
                b, a = signal.butter(4, normalized_cutoff, btype='high')
                filtered = signal.filtfilt(b, a, audio_data)
                
                return filtered.astype(np.float32)
                
        except Exception as e:
            logger.exception(f"降噪处理失败: {str(e)}")
            return audio_data

    @staticmethod
    def enhance_audio(audio_data: np.ndarray, sample_rate: int = None, 
                     denoise: bool = True, normalize: bool = True,
                     volume: float = 1.0) -> np.ndarray:
        """应用所有音频增强处理"""
        try:
            if not AudioUtils.validate_audio(audio_data):
                return audio_data
            
            # 1. 降噪
            if denoise:
                audio_data = AudioUtils.denoise(audio_data, sample_rate or AudioUtils.DEFAULT_SAMPLE_RATE)
            
            # 2. 标准化
            if normalize:
                audio_data = AudioUtils.normalize(audio_data)
            
            # 3. 调整音量
            if volume != 1.0:
                audio_data = audio_data * volume
            
            logger.info("已完成音频增强处理")
            return audio_data
            
        except Exception as e:
            logger.exception(f"音频增强处理失败: {str(e)}")
            return audio_data

    @staticmethod
    def adjust_speed(audio_data: np.ndarray, speed: float) -> np.ndarray:
        """
        高质量语音速度调整

        Args:
            audio_data: 音频数据
            speed: 速度因子 (>0)

        Returns:
            调整速度后的音频数据
        """
        try:
            if not 0.1 <= speed <= 3.0:
                logger.warning(f"速度因子 {speed} 超出建议范围 [0.1, 3.0]")
                speed = np.clip(speed, 0.1, 3.0)
                
            try:
                import librosa
                # 使用librosa进行高质量时间拉伸
                return librosa.effects.time_stretch(audio_data, rate=speed)
            except ImportError:
                # 降级方案：使用线性插值
                new_length = int(len(audio_data) / speed)
                indices = np.linspace(0, len(audio_data)-1, new_length)
                return np.interp(indices, np.arange(len(audio_data)), audio_data)
                
        except Exception as e:
            logger.exception(f"调整音频速度失败: {str(e)}")
            return audio_data

    @staticmethod
    def concatenate(audio_list: List[np.ndarray], crossfade_ms: int = 10) -> np.ndarray:
        """
        连接多个音频片段，支持交叉淡入淡出

        Args:
            audio_list: 音频数据列表
            crossfade_ms: 交叉淡化时长(毫秒)

        Returns:
            连接后的音频数据
        """
        try:
            if not audio_list:
                return np.array([], dtype=np.float32)
                
            if len(audio_list) == 1:
                return audio_list[0]
                
            # 计算交叉淡化样本数
            crossfade_samples = int(AudioUtils.DEFAULT_SAMPLE_RATE * crossfade_ms / 1000)
            
            # 创建输出数组
            total_length = sum(len(audio) for audio in audio_list)
            output = np.zeros(total_length, dtype=np.float32)
            
            # 应用交叉淡化
            position = 0
            for i, audio in enumerate(audio_list):
                if i > 0 and len(audio) > crossfade_samples:
                    # 创建淡入淡出曲线
                    fade_in = np.linspace(0, 1, crossfade_samples)
                    fade_out = 1 - fade_in
                    
                    # 应用交叉淡化
                    output[position-crossfade_samples:position] *= fade_out
                    audio[:crossfade_samples] *= fade_in
                
                # 复制音频数据
                output[position:position+len(audio)] += audio
                position += len(audio)
            
            return output
            
        except Exception as e:
            logger.exception(f"连接音频失败: {str(e)}")
            return np.array([], dtype=np.float32)

    @staticmethod
    def split_silence(audio_data: np.ndarray, sample_rate: int, 
                     min_silence_ms: int = 500, silence_thresh_db: float = -40) -> List[np.ndarray]:
        """
        根据静音分割音频

        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            min_silence_ms: 最小静音时长(毫秒)
            silence_thresh_db: 静音阈值(dB)

        Returns:
            分割后的音频片段列表
        """
        try:
            # 计算能量
            frame_length = int(sample_rate * 0.025)  # 25ms帧
            hop_length = int(sample_rate * 0.010)    # 10ms步长
            
            # 计算短时能量
            energy = np.array([
                np.sum(audio_data[i:i+frame_length]**2)
                for i in range(0, len(audio_data)-frame_length, hop_length)
            ])
            
            # 转换为dB
            energy_db = 10 * np.log10(energy + 1e-10)
            
            # 查找静音段
            silence_mask = energy_db < silence_thresh_db
            
            # 合并短静音段
            min_silence_frames = int(min_silence_ms / (1000 * hop_length / sample_rate))
            segments = []
            start = 0
            
            for i in range(1, len(silence_mask)):
                if silence_mask[i] != silence_mask[i-1]:
                    if i - start >= min_silence_frames:
                        frame_pos = start * hop_length
                        segments.append(audio_data[frame_pos:i*hop_length])
                    start = i
            
            # 添加最后一段
            if start * hop_length < len(audio_data):
                segments.append(audio_data[start*hop_length:])
            
            return segments
            
        except Exception as e:
            logger.exception(f"分割音频失败: {str(e)}")
            return [audio_data]