#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import threading
from typing import Dict, Any, Callable, List, Optional, Union, Set
from utils.logger import logger
from utils.config import Config

class UIEvent:
    """UI事件基类，用于UI组件和业务逻辑之间的通信"""
    def __init__(self, event_type: str, data: Dict[str, Any] = None):
        self.event_type = event_type
        self.data = data or {}
        self.timestamp = time.time()
        
    def __repr__(self):
        return f"UIEvent(type={self.event_type}, data={self.data})"

class UIEventType:
    """UI事件类型常量定义"""
    # 翻译控制事件
    START_TRANSLATION = "start_translation"
    STOP_TRANSLATION = "stop_translation"
    PAUSE_TRANSLATION = "pause_translation"
    RESUME_TRANSLATION = "resume_translation"
    TRANSLATE_FILE = "translate_file"       # 翻译音频文件
    TRANSLATION_RESULT = "translation_result"  # 翻译结果返回
    
    # 状态更新事件
    UPDATE_STATUS = "update_status"
    UPDATE_PROGRESS = "update_progress"
    REQUEST_STATUS_UPDATE = "request_status_update"
    # 配置事件
    CONFIG_CHANGED = "config_changed"
    
    # 系统事件
    ERROR = "error"
    SYSTEM_READY = "system_ready"
    SYSTEM_EXIT = "system_exit"
    
    # 模型事件
    MODEL_DOWNLOAD = "model_download"
    MODEL_DELETE = "model_delete"
    MODEL_LOADED = "model_loaded"
    GET_MODEL_VERSIONS = "get_model_versions"  # 获取模型版本列表
    MODEL_VERSIONS_LOADED = "model_versions_loaded"  # 模型版本列表加载完成
    SET_MODEL = "set_model"  # 设置当前使用的模型
    MODEL_SET_RESULT = "model_set_result"  # 设置模型结果
    
    # 音频设备事件
    DEVICE_CHANGED = "device_changed"

    # 添加声音列表相关事件
    GET_TTS_VOICES = "get_tts_voices"
    TTS_VOICES_LOADED = "tts_voices_loaded"

class UICallbackID:
    """UI回调ID常量定义"""
    # TTS相关回调
    TTS_MODEL_VERSIONS = "tts_model_versions"  # 统一使用此ID获取TTS模型版本
    SET_TTS_MODEL = "set_tts_model"

    # ASR相关回调
    ASR_MODEL_VERSIONS = "asr_model_versions"  # 用于非实时ASR的模型版本
    REALTIME_ASR_MODEL_VERSIONS = "realtime_asr_model_versions"  # 用于实时ASR的模型版本
    SET_ASR_MODEL = "set_asr_model"
    SET_REALTIME_ASR_MODEL = "set_realtime_asr_model"

    # 文件翻译相关回调
    TRANSLATE_AUDIO_FILE = "translate_audio_file"

    # 通用回调
    DEFAULT = "default"
    ERROR = "error"

    # 添加声音列表相关回调ID
    TTS_VOICES = "tts_voices"

class UIConfigManager:
    """UI配置管理器，负责UI相关配置的管理和持久化"""
    
    def __init__(self):
        self._config = {}
        self._observers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._system_config = Config()
        self._load_from_system()

        self._ui_datas_config = {
            "models":[
            # ASR模型
            ("ASR", "whisper-base"),
            ("ASR", "whisper-small"),
            ("ASR", "whisper-medium"),
            ("ASR", "whisper-large-v2"),
            ("ASR", "faster-whisper-base"),
            ("ASR", "faster-whisper-small"),
            ("ASR", "faster-whisper-medium"),
            ("ASR", "faster-whisper-large-v2"),
            ("ASR", "vosk-zh"),
            ("ASR", "vosk-zh-large"),
            
            # TTS模型
            ("TTS", "f5_tts-thai"),

            # 翻译模型
            ("翻译", "nllb-base"),
            ("翻译", "nllb-small"),
            ("翻译", "nllb-medium"),
            ("翻译", "nllb-large")
        ],
        "single_src_lang":[
            "zh (中文)",
            "en (英语)", 
            "th (泰语)",
            "ja (日语)",
            "ko (韩语)"
        ],
        "single_tgt_lang":[
            "zh (中文)",
            "en (英语)", 
            "th (泰语)",
            "ja (日语)",
            "ko (韩语)"
        ],
        "asr_engine":[
            "whisper",
            "faster_whisper",
            "vosk"
        ],
        "tts_engine":[
            "f5_tts"
        ],
        "translation_engine":[
            "nllb"
        ],
        "tts_voices": {
            "f5_tts": [
                "default (默认声音)",
            ]
        }
        }

    def get_ui_datas_config(self, key: str) -> Any:
        """获取UI数据配置"""
        with self._lock:
            return self._ui_datas_config.get(key, None)
        
    def _load_from_system(self):
        """从系统配置加载UI配置"""
        try:
            # 加载用户界面配置
            ui_config = self._system_config.get("user.ui", {})
            if ui_config:
                self._config.update(ui_config)
                
            # 加载音频设备配置
            audio_config = self._system_config.get("audio", {})
            if audio_config:
                self._config["audio"] = audio_config
                
            # 加载语言配置
            lang_config = self._system_config.get("user.language", {})
            if lang_config:
                self._config["language"] = lang_config
                
            # 加载ASR配置
            asr_config = self._system_config.get("user.asr", {})
            if asr_config:
                self._config["asr"] = asr_config
                
            # 加载翻译配置
            translation_config = self._system_config.get("user.translation", {})
            if translation_config:
                self._config["translation"] = translation_config
                
            # 加载TTS配置
            tts_config = self._system_config.get("user.tts", {})
            if tts_config:
                self._config["tts"] = tts_config
            
            logger.info("UI配置从系统配置加载成功")
        except Exception as e:
            logger.exception(f"从系统配置加载UI配置失败: {str(e)}")
    
    def save_to_system(self):
        """保存UI配置到系统配置"""
        try:
            with self._lock:
                # 保存UI配置
                if "ui" in self._config:
                    self._system_config.set("user.ui", self._config["ui"])
                    
                # 保存音频配置
                if "audio" in self._config:
                    self._system_config.set("audio", self._config["audio"])
                    
                # 保存语言配置
                if "language" in self._config:
                    self._system_config.set("user.language", self._config["language"])
                    # 补充ASR语言配置
                    self._system_config.set("user.asr.language", self._config["language"]["source"])
                    # 补充翻译语言配置
                    self._system_config.set("user.asr.translation", self._config["language"]["source"])
                    self._system_config.set("user.asr.translation", self._config["language"]["target"])
                    # 补充TTS语言配置
                    self._system_config.set("user.tts.language",self._config["language"]["target"])
                    
                # 保存ASR配置
                if "asr" in self._config:
                    self._system_config.set("user.asr", self._config["asr"])
                    self._system_config.set("user.asr.language", self._config["language"]["source"])
                    
                # 保存翻译配置
                if "translation" in self._config:
                    self._system_config.set("user.translation", self._config["translation"])
                    self._system_config.set("user.asr.translation", self._config["language"]["source"])
                    self._system_config.set("user.asr.translation", self._config["language"]["target"])
                    
                # 保存TTS配置
                if "tts" in self._config:
                    self._system_config.set("user.tts", self._config["tts"])
                    self._system_config.set("user.tts.language",self._config["language"]["target"])
                
                # 保存到文件
                self._system_config._save_config()
                
                logger.info("UI配置保存到系统配置成功")
        except Exception as e:
            logger.exception(f"保存UI配置到系统配置失败: {str(e)}")
            
    def update_config(self, key: str, value: Any, notify: bool = True):
        """更新配置项"""
        with self._lock:
            # 处理嵌套键，例如 "audio.input.device"
            if "." in key:
                parts = key.split(".")
                config = self._config
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                config[parts[-1]] = value
            else:
                self._config[key] = value
                
            # 通知观察者
            if notify:
                self._notify_observers(key, value)

                # 为关键配置自动保存
            critical_keys = ["asr.engine", "asr.model", "tts.engine", "tts.model","tts.voice",
                             "translation.engine", "translation.model"]
            if key in critical_keys or any(key.startswith(prefix) for prefix in critical_keys):
                self.save_to_system()
                logger.debug(f"关键配置已更改并保存: {key}={value}")
    
    def update_config_dict(self, config_dict: Dict[str, Any], prefix: str = "", notify: bool = True):
        """批量更新配置项"""
        with self._lock:
            for key, value in config_dict.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    self.update_config_dict(value, full_key, False)
                else:
                    self.update_config(full_key, value, False)
            
            # 完成后通知一次
            if notify and prefix:
                self._notify_observers(prefix, self.get_config(prefix))
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        with self._lock:
            # 处理嵌套键
            if "." in key:
                parts = key.split(".")
                config = self._config
                for part in parts[:-1]:
                    if part not in config:
                        return default
                    config = config[part]
                return config.get(parts[-1], default)
            else:
                return self._config.get(key, default)
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            return self._config.copy()
    
    def register_observer(self, key: str, observer: Callable[[str, Any], None]):
        """注册配置观察者"""
        with self._lock:
            if key not in self._observers:
                self._observers[key] = []
            self._observers[key].append(observer)
    
    def unregister_observer(self, key: str, observer: Callable[[str, Any], None]):
        """注销配置观察者"""
        with self._lock:
            if key in self._observers and observer in self._observers[key]:
                self._observers[key].remove(observer)
                if not self._observers[key]:
                    del self._observers[key]
    
    def _notify_observers(self, key: str, value: Any):
        """通知配置观察者"""
        # 直接匹配的观察者
        if key in self._observers:
            for observer in self._observers[key]:
                try:
                    observer(key, value)
                except Exception as e:
                    logger.exception(f"配置观察者通知失败 ({key}): {str(e)}")
        
        # 通配符观察者
        if "*" in self._observers:
            for observer in self._observers["*"]:
                try:
                    observer(key, value)
                except Exception as e:
                    logger.exception(f"配置观察者通知失败 (通配符): {str(e)}")
        
        # 父级键观察者（例如，更新 "audio.input.device" 时通知 "audio.input" 和 "audio" 的观察者）
        parts = key.split(".")
        for i in range(1, len(parts)):
            parent_key = ".".join(parts[:-i])
            if parent_key in self._observers:
                parent_value = self.get_config(parent_key)
                for observer in self._observers[parent_key]:
                    try:
                        observer(parent_key, parent_value)
                    except Exception as e:
                        logger.exception(f"配置观察者通知失败 (父级 {parent_key}): {str(e)}")

class UIManager:
    """UI管理器，负责UI事件处理和流程控制"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(UIManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._event_handlers: Dict[str, List[Callable]] = {}
            self._status_observers: List[Callable] = []
            self._lock = threading.RLock()
            self._config_manager = UIConfigManager()
            self._event_queue = []
            self._processing_events = False
    
    @property
    def config(self) -> UIConfigManager:
        """获取配置管理器"""
        return self._config_manager
    
    def register_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        """注册事件处理器"""
        with self._lock:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(handler)
            
    def unregister_event_handler(self, event_type: str, handler: Callable):
        """注销事件处理器"""
        with self._lock:
            if event_type in self._event_handlers and handler in self._event_handlers[event_type]:
                self._event_handlers[event_type].remove(handler)
                if not self._event_handlers[event_type]:
                    del self._event_handlers[event_type]
    
    def register_status_observer(self, observer: Callable[[Dict[str, Any]], None]):
        """注册状态观察者"""
        with self._lock:
            self._status_observers.append(observer)
            
    def unregister_status_observer(self, observer: Callable):
        """注销状态观察者"""
        with self._lock:
            if observer in self._status_observers:
                self._status_observers.remove(observer)
    
    def handle_event(self, event: UIEvent):
        """处理UI事件"""
        with self._lock:
            self._event_queue.append(event)
            if not self._processing_events:
                self._process_events()
    
    def _process_events(self):
        """处理事件队列"""
        try:
            self._processing_events = True
            while self._event_queue:
                with self._lock:
                    event = self._event_queue.pop(0)
                
                handlers = self._event_handlers.get(event.event_type, [])
                for handler in handlers:
                    try:
                        handler(event.data)
                    except Exception as e:
                        logger.exception(f"事件处理失败 ({event.event_type}): {str(e)}")
                        
                # 对于状态更新事件，通知所有状态观察者
                if event.event_type == UIEventType.UPDATE_STATUS:
                    self.notify_status(event.data)
        finally:
            self._processing_events = False
    
    def notify_status(self, status: Dict[str, Any]):
        """通知状态更新"""
        for observer in self._status_observers:
            try:
                observer(status)
            except Exception as e:
                logger.exception(f"状态通知失败: {str(e)}")
    
    def trigger_event(self, event_type: str, data: Dict[str, Any] = None):
        """触发事件"""
        self.handle_event(UIEvent(event_type, data))
    
    def save_config(self):
        """保存配置"""
        self._config_manager.save_to_system()

    def get_ui_datas_config(self, key: str) -> Any:
        """获取UI数据配置"""
        return self._config_manager.get_ui_datas_config(key)

# 全局UI管理器实例
ui_manager = UIManager()

__all__ = ['UIEvent', 'UIEventType', 'UICallbackID', 'UIConfigManager', 'UIManager', 'ui_manager']