import os
from typing import Dict, Any, Optional

import numpy as np
import torch

from core.tts import BaseTTSEngine
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

            # 先固定，todo
            voice = "default.wav"

            voice = f"""{model_info["save_path"]}/voice/{voice}"""

            # 设置声音(参考音频),todo
            if voice and not self._set_voice(voice):
                return False
            
            self.initialized = True
            logger.info(f"F5-TTS配置完成: 模型={model}, 参考音频={voice}, 语言={language}")
            return True
            
        except Exception as e:
            logger.exception(f"F5-TTS配置失败: {str(e)}")
            return False
    
    def _load_model(self, model_name: str) -> bool:
        """加载F5-TTS模型"""
        try:
            from f5_tts.infer.utils_infer import load_model, load_vocoder, get_tokenizer
            from f5_tts.model import DiT
            
            # 获取模型信息
            model_info = model_manager.get_model_info("tts", "f5_tts", model_name)
            if not model_info:
                logger.error(f"未找到模型信息: f5_tts/{model_name}")
                return False
            
            model_path = os.path.join(model_info["save_path"], "model.pt")
            vocab_path = os.path.join(model_info["save_path"], "vocab.txt")
            
            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                logger.error(f"模型文件或词汇表文件不存在")
                return False
            
            # 加载tokenizer
            self.vocab_char_map, self.vocab_size = get_tokenizer(vocab_path, "custom")
            
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
            self.vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=self.device)
            
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
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            
            if not os.path.exists(voice):
                logger.error(f"参考音频文件不存在: {voice}")
                return False
            
            # 这里需要提供参考音频对应的文本
            # 实际应用中可能需要从配置中读取或使用默认文本
            # 泰语示例文本,todo 先固定
            ref_text = "ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น."
            
            # 处理参考音频
            self.ref_audio_tensor, self.ref_text_proc = preprocess_ref_audio_text(voice, ref_text)
            logger.info(f"已设置参考音频: {voice}")
            return True
            
        except Exception as e:
            logger.exception(f"设置参考音频失败: {str(e)}")
            return False
    
    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """执行语音合成"""
        try:


            if not self.initialized:
                raise RuntimeError("F5-TTS引擎未初始化")

            if not self._validate_text(text):
                return None

            if self.ref_audio_tensor is None or self.ref_text_proc is None:
                logger.error("参考音频未设置")
                return None

            from f5_tts.infer.utils_infer import infer_process
            from f5_tts.model.utils import seed_everything

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

            # 执行合成
            audio, sample_rate, _ = infer_process(
                ref_audio=self.ref_audio_tensor,
                ref_text=self.ref_text_proc,
                gen_text=text,
                model_obj=self.model,
                vocoder=self.vocoder,
                mel_spec_type="vocos",
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
                fix_duration=fix_duration,
                device=torch.device(self.device)
            )

            # 确保返回适当格式的音频数据
            audio_data = np.array(audio)

            # 验证音频
            if not self._validate_audio(audio_data):
                return None

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