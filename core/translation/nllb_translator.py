#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Optional, Dict, Any
from utils.logger import logger
from utils.model_manager import model_manager
from .base_translator import BaseTranslator

class NLLBTranslator(BaseTranslator):
    """NLLB离线翻译引擎"""
    
    def __init__(self):
        super().__init__()
        self.device = None
        self.tokenizer = None
        self.translation_cache = {}  # 简单的翻译缓存
        
        # NLLB的语言映射（支持200种语言）
        self.lang_map = {
            "en": "eng_Latn",  # 英语
            "zh": "zho_Hans",  # 简体中文
            "zh_tw": "zho_Hant",  # 繁体中文
            "ja": "jpn_Jpan",  # 日语
            "ko": "kor_Hang",  # 韩语
            "es": "spa_Latn",  # 西班牙语
            "fr": "fra_Latn",  # 法语
            "de": "deu_Latn",  # 德语
            "it": "ita_Latn",  # 意大利语
            "ru": "rus_Cyrl",  # 俄语
            "ar": "arb_Arab",  # 阿拉伯语
            "hi": "hin_Deva",  # 印地语
            "bn": "ben_Beng",  # 孟加拉语
            "pt": "por_Latn",  # 葡萄牙语
            "nl": "nld_Latn",  # 荷兰语
            "pl": "pol_Latn",  # 波兰语
            "tr": "tur_Latn",  # 土耳其语
            "vi": "vie_Latn",  # 越南语
            "th": "tha_Thai",  # 泰语
            "id": "ind_Latn",  # 印尼语
            "ms": "zsm_Latn",  # 马来语
            "fa": "fas_Arab",  # 波斯语
            "sw": "swh_Latn",  # 斯瓦希里语
            "ta": "tam_Taml",  # 泰米尔语
            "te": "tel_Telu",  # 泰卢固语
            "ur": "urd_Arab",  # 乌尔都语
            "uk": "ukr_Cyrl",  # 乌克兰语
            "el": "ell_Grek",  # 希腊语
            "he": "heb_Hebr",  # 希伯来语
            "cs": "ces_Latn"   # 捷克语
        }
    
    def configure(self, model: str, source_language: str, target_language: str, gpu: Dict[str, Any], performance: Dict[str, Any]) -> bool:
        """配置翻译引擎"""
        try:
            # 检查模型是否已下载
            if not model_manager.is_model_downloaded("translation", "nllb", model):
                logger.error(f"NLLB模型 {model} 未下载")
                return False
            
            # 验证语言支持
            if source_language not in self.lang_map:
                logger.error(f"不支持的源语言: {source_language}")
                return False
            if target_language not in self.lang_map:
                logger.error(f"不支持的目标语言: {target_language}")
                return False
            
            # 设置设备和计算类型
            if gpu["enabled"] and torch.cuda.is_available():
                self.device = gpu["device"] if gpu["device"] != "auto" else "cuda"
            else:
                self.device = "cpu"
            
            # 获取模型信息和路径
            model_info = model_manager.get_model_info("translation", "nllb", model)
            if not model_info:
                logger.error(f"无法获取模型信息: {model}")
                return False
            
            model_path = model_info["save_path"]
            if not os.path.exists(model_path):
                logger.error(f"模型路径不存在: {model_path}")
                return False
            
            # 加载模型
            logger.info(f"正在加载NLLB模型: {model} (设备: {self.device})")
            try:
                # 加载tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                # 根据设备和性能配置选择最佳配置
                model_kwargs = {
                    "local_files_only": True,
                    "trust_remote_code": True
                }
                
                if self.device == "cuda":
                    model_kwargs.update({
                        "device_map": "auto",
                        "torch_dtype": torch.float16 if performance["use_half_precision"] else torch.float32,
                        "low_cpu_mem_usage": True
                    })
                else:
                    model_kwargs.update({
                        "device_map": None,
                        "torch_dtype": torch.float32
                    })
                    
                # 加载模型
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                
                if self.device == "cpu":
                    self.model = self.model.to("cpu")
                
            except Exception as e:
                logger.exception(f"加载模型失败: {str(e)}")
                return False
            
            # 保存语言设置
            self.source_language = source_language
            self.target_language = target_language
            
            self.initialized = True
            logger.info(f"NLLB翻译配置完成: model={model}, source={source_language}, target={target_language}")
            return True
            
        except Exception as e:
            logger.exception(f"NLLB翻译配置失败: {str(e)}")
            return False
    
    def translate(self, text: str) -> Optional[str]:
        """执行翻译"""
        try:
            if not self.initialized:
                raise RuntimeError("翻译引擎未初始化")
            
            # 验证文本
            if not self._validate_text(text):
                return None
            
            # 检查缓存
            cache_key = f"{text}_{self.source_language}_{self.target_language}"
            if cache_key in self.translation_cache:
                return self.translation_cache[cache_key]
            
            # 获取语言代码
            src_lang_code = self.lang_map.get(self.source_language)
            tgt_lang_code = self.lang_map.get(self.target_language)
            
            try:
                # 设置目标语言
                self.tokenizer.target_lang = tgt_lang_code
                
                # 编码输入
                encoded = self.tokenizer(text, return_tensors="pt", padding=True)
                encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
                
                # 生成翻译
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **encoded,
                        forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang_code),
                        max_length=256,
                        num_beams=5,
                        length_penalty=1.0,
                        early_stopping=True
                    )
                
                # 解码输出
                translation = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0].strip()
                
                # 更新缓存
                self.translation_cache[cache_key] = translation
                
                return translation
                
            except Exception as e:
                logger.exception(f"翻译过程失败: {str(e)}")
                return None
            
        except Exception as e:
            logger.exception(f"翻译失败: {str(e)}")
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.translation_cache.clear()
            self.initialized = False
            logger.info("NLLB翻译资源已清理")
        except Exception as e:
            logger.exception(f"清理资源失败: {str(e)}") 