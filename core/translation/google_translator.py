#!/usr/bin/env python
# -*- coding: utf-8 -*-

from googletrans import Translator
from typing import Optional, Dict, Any
from utils.logger import logger
from .base_translator import BaseTranslator

class GoogleTranslator(BaseTranslator):
    """Google翻译引擎"""
    
    def __init__(self):
        super().__init__()
        self.translator = None
    
    def configure(self, **config) -> bool:
        """配置翻译引擎"""
        try:
            # 验证配置
            if not isinstance(config.get("source_language"), str):
                logger.exception("源语言配置无效")
                return False
            if not isinstance(config.get("target_language"), str):
                logger.exception("目标语言配置无效")
                return False
            
            # 设置配置
            self.source_language = config["source_language"]
            self.target_language = config["target_language"]
            
            # 初始化翻译器
            self.translator = Translator()
            
            self.initialized = True
            logger.info(f"Google翻译配置完成: {self.source_language} -> {self.target_language}")
            return True
            
        except Exception as e:
            logger.exception(f"Google翻译配置失败: {str(e)}")
            return False
    
    def translate(self, text: str) -> Optional[str]:
        """执行翻译"""
        try:
            if not self.initialized:
                logger.exception("翻译引擎未初始化")
                return None
            
            # 验证文本
            if not self._validate_text(text):
                return None
            
            # 执行翻译
            result = self.translator.translate(
                text,
                src=self.source_language,
                dest=self.target_language
            )
            
            if not result or not result.text:
                logger.exception("翻译结果无效")
                return None
            
            return result.text.strip()
            
        except Exception as e:
            logger.exception(f"翻译失败: {str(e)}")
            return None 