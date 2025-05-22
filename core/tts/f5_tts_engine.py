import os
from typing import Dict, Any, Optional

from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_model,
    load_vocoder,
    get_tokenizer,
    infer_process,
    infer_batch_process,
)
from f5_tts.model import DiT
from f5_tts.model.utils import seed_everything

import numpy as np
import torch

from core.tts import BaseTTSEngine, AudioUtils
from utils.logger import logger
from utils.model_manager import model_manager


class F5TTSEngine(BaseTTSEngine):
    """F5-TTS泰语语音合成引擎"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.vocoder = None
        self.vocab_char_map = None
        self.vocab_size = None
        self.ref_audio_tensor = None
        self.ref_text_proc = None
        self.mel_spec_type = "vocos"

        self.voice = None
        self.ref_tetx = None
    
    def configure(self, model: str, voice: str, language: str, gpu: Dict[str, Any], 
                 performance: Dict[str, Any], audio: Dict[str, Any]) -> bool:
        """配置TTS引擎"""
        try:
            # 调用父类的配置方法
            if not super().configure(model, voice, language, gpu, performance, audio):
                return False
            
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("tts", "f5_tts", model):
                logger.warning(f"F5-TTS模型 {model} 未下载")
                return False


            # 加载模型
            if not self._load_model(model):
                return False

            model_info = model_manager.get_model_info("tts", "f5_tts", model)


            voice = f"""{model_info["save_path"]}/voice/{voice}"""

            # 设置声音(参考音频),todo
            if voice and not self._set_voice(voice):
                return False
            
            self.initialized = True
            logger.info(f"F5-TTS配置完成: 模型={model}, 参考音频={voice}, 语言={language}，设备={self.device}")
            return True
            
        except Exception as e:
            logger.exception(f"F5-TTS配置失败: {str(e)}")
            return False
    
    def _load_model(self, model_name: str) -> bool:
        """加载F5-TTS模型"""
        try:

            
            # 获取模型信息
            model_info = model_manager.get_model_info("tts", "f5_tts", model_name)
            if not model_info:
                logger.error(f"未找到模型信息: f5_tts/{model_name}")
                return False
            
            self.model_path = model_path = os.path.join(model_info["save_path"], "model.pt")
            self.vocab_path = vocab_path = os.path.join(model_info["save_path"], "vocab.txt")
            
            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                logger.error(f"模型文件或词汇表文件不存在")
                return False
            
            # 加载tokenizer
            vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
            
            # 模型参数配置
            model_cfg = dict(
                dim=1024,
                depth=22,
                heads=16,
                ff_mult=2,
                text_dim=512,
                conv_layers=4
            )
            
            # 构建模型
            logger.info(f"正在加载F5-TTS模型: {model_path}")
            self.model = load_model(
                model_cls=DiT,
                model_cfg=model_cfg,
                ckpt_path=model_path,
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                use_ema=True,
                device=self.device
            )
            self.model.eval()
            
            # 加载vocoder
            self.vocoder = load_vocoder(vocoder_name=self.mel_spec_type, is_local=False, device=self.device)
            
            logger.info("F5-TTS模型加载成功")
            return True
            
        except Exception as e:
            logger.exception(f"F5-TTS模型加载失败: {str(e)}")
            self.model = None
            self.vocoder = None
            return False
    
    def _set_voice(self, voice: str) -> bool:
        """设置参考音频"""
        try:
            # 自动补全后缀
            if not os.path.exists(voice):
                # 尝试加.wav
                if os.path.exists(voice + ".wav"):
                    voice = voice + ".wav"
                elif os.path.exists(voice + ".mp3"):
                    voice = voice + ".mp3"
                else:
                    logger.error(f"参考音频文件不存在: {voice}(.wav/.mp3)")
                    return False

            txt_path = f"""{voice.rsplit('.', 1)[0]}.txt"""
            if not os.path.exists(txt_path):
                logger.error(f"参考音频对应的文本文件不存在: {txt_path}")
                return False

            with open(txt_path, "r", encoding="utf-8") as f:
                ref_text = f.read()

            self.ref_tetx = ref_text
            logger.info(f"已加载参考音频文本: {ref_text}")
            self.voice = voice

            # 处理参考音频
            self.ref_audio_tensor, self.ref_text_proc = preprocess_ref_audio_text(voice, ref_text)
            logger.info(f"已设置参考音频: {voice},{self.ref_audio_tensor}")
            return True

        except Exception as e:
            logger.exception(f"设置参考音频失败: {str(e)}")
            return False
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        try:
            if not self.initialized:
                raise RuntimeError("F5-TTS引擎未初始化")

            if not self._validate_text(text):
                return None

            if self.ref_audio_tensor is None or self.ref_text_proc is None:
                logger.error("参考音频未设置")
                return None

            # 设置随机种子确保结果一致性
            seed_everything(42)

            logger.info(f"正在执行F5-TTS语音合成: 文本={text}")

            # 合成参数
            speed = 1.05
            cfg_strength = 2.8
            cross_fade_duration = 0.12
            nfe_step = 48
            sway_sampling_coef = 0.2
            target_rms = -18.0
            fix_duration = None

            audio, sample_rate, _ = infer_process(
                ref_audio=self.ref_audio_tensor,
                ref_text=self.ref_text_proc,
                gen_text=text,
                model_obj=self.model,
                vocoder=self.vocoder,
                mel_spec_type=self.mel_spec_type,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                fix_duration=fix_duration,
                device=torch.device(self.device)
            )

            # 验证音频
            if not self._validate_audio(audio):
                return None

            # 确保音频格式正确
            audio = AudioUtils.ensure_wav_1d_float32(audio)

            # 重采样到目标采样率
            audio = AudioUtils.resample_audio(audio, sample_rate, self.sample_rate)

            # 音频增强
            audio = self._enhance_audio(audio)

            return audio
                

        except Exception as e:
            logger.exception(f"语音合成失败: {str(e)}")
            return None
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.vocoder is not None:
                del self.vocoder
                self.vocoder = None
            
            self.ref_audio_tensor = None
            self.ref_text_proc = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("F5-TTS资源已清理")
            
        except Exception as e:
            logger.exception(f"清理资源失败: {str(e)}")