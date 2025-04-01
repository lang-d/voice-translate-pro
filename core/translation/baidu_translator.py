#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import random
import requests
from typing import Optional, Dict, Any
from utils.logger import logger
from .base_translator import BaseTranslator

class BaiduTranslator(BaseTranslator):
    """百度翻译引擎"""
    
    def __init__(self):
        super().__init__()
        self.appid = None
        self.secret_key = None
        self.api_url = "http://api.fanyi.baidu.com/api/trans/vip/translate"
    
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
            if not isinstance(config.get("appid"), str):
                logger.exception("APP ID配置无效")
                return False
            if not isinstance(config.get("secret_key"), str):
                logger.exception("密钥配置无效")
                return False
            
            # 设置配置
            self.source_language = config["source_language"]
            self.target_language = config["target_language"]
            self.appid = config["appid"]
            self.secret_key = config["secret_key"]
            
            self.initialized = True
            logger.info(f"百度翻译配置完成: {self.source_language} -> {self.target_language}")
            return True
            
        except Exception as e:
            logger.exception(f"百度翻译配置失败: {str(e)}")
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
            
            # 生成签名
            salt = random.randint(32768, 65536)
            sign = self._generate_sign(text, salt)
            
            # 准备请求参数
            params = {
                "q": text,
                "from": self.source_language,
                "to": self.target_language,
                "appid": self.appid,
                "salt": salt,
                "sign": sign
            }
            
            # 发送请求
            response = requests.get(self.api_url, params=params)
            result = response.json()
            
            if "error_code" in result:
                logger.exception(f"翻译失败: {result.get('error_msg', '未知错误')}")
                return None
            
            if not result.get("trans_result"):
                logger.exception("翻译结果为空")
                return None
            
            return result["trans_result"][0]["dst"].strip()
            
        except Exception as e:
            logger.exception(f"翻译失败: {str(e)}")
            return None
    
    def _generate_sign(self, text: str, salt: int) -> str:
        """生成签名"""
        sign_str = f"{self.appid}{text}{salt}{self.secret_key}"
        return hashlib.md5(sign_str.encode()).hexdigest() 