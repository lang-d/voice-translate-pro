#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import threading
import numpy as np
import collections
import sounddevice as sd
import soundfile as sf
from typing import Optional, Dict, Any, Tuple, List, Callable
from core.asr.asr_manager import ASRManager
from core.status_event import StatusEvent, StatusEventType
from core.translation.translation_manager import TranslationManager
from core.tts.tts_manager import TTSManager
from core.status_manager import StatusManager
from utils.logger import logger
from utils.config import config

class AudioCapture:
    """音频捕获类，负责从麦克风捕获音频"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                chunk_duration: float = 0.5, noise_suppression: bool = True):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.noise_suppression = noise_suppression
        self.device_index = None
        
        self.stream = None
        self.is_running = False
        self.callback_fn = None
        
    def set_device(self, device_index: Optional[int] = None) -> bool:
        """设置输入设备"""
        try:
            if device_index is not None:
                # 验证设备
                device_info = sd.query_devices(device_index)
                if device_info.get('max_input_channels', 0) <= 0:
                    logger.warning(f"设备 {device_index} 不支持输入，使用默认设备")
                    self.device_index = None
                    return False
                logger.info(f"设置输入设备: {device_info.get('name')} (索引: {device_index})")
                self.device_index = device_index
            else:
                logger.info("使用默认输入设备")
                self.device_index = None
            return True
        except Exception as e:
            logger.exception(f"设置输入设备失败: {str(e)}")
            self.device_index = None
            return False
    
    def start(self, callback: Callable) -> bool:
        """启动音频捕获"""
        if self.is_running:
            logger.warning("音频捕获已经在运行")
            return False
            
        try:
            self.callback_fn = callback
            self.is_running = True
            
            # 创建并启动流
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device_index,
                callback=self._audio_callback
            )
            self.stream.start()
            logger.info("音频捕获已启动")
            return True
        except Exception as e:
            logger.exception(f"启动音频捕获失败: {str(e)}")
            self.is_running = False
            self.stream = None
            return False
    
    def stop(self):
        """停止音频捕获"""
        if not self.is_running:
            return
            
        try:
            self.is_running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            logger.info("音频捕获已停止")
        except Exception as e:
            logger.exception(f"停止音频捕获失败: {str(e)}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """音频回调函数"""
        if not self.is_running or not self.callback_fn:
            return
            
        if status:
            logger.warning(f"音频回调状态: {status}")
            
        try:
            # 应用降噪（如果启用）
            if self.noise_suppression:
                indata = self._suppress_noise(indata)
                
            # 调用回调函数
            self.callback_fn(indata.copy(), time_info)
        except Exception as e:
            logger.exception(f"音频回调处理失败: {str(e)}")
    
    def _suppress_noise(self, audio: np.ndarray) -> np.ndarray:
        """简单的降噪实现"""
        try:
            # 简单的阈值降噪
            noise_threshold = 0.01
            audio[abs(audio) < noise_threshold] = 0
            return audio
        except Exception as e:
            logger.exception(f"降噪失败: {str(e)}")
            return audio


class AudioOutput:
    """音频输出类，负责将音频输出到扬声器"""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_index = None
        
        self.stream = None
        self.is_running = False
        self.audio_queue = collections.deque(maxlen=10)
        
    def set_device(self, device_index: Optional[int] = None) -> bool:
        """设置输出设备"""
        try:
            if device_index is not None:
                # 验证设备
                device_info = sd.query_devices(device_index)
                if device_info.get('max_output_channels', 0) <= 0:
                    logger.warning(f"设备 {device_index} 不支持输出，使用默认设备")
                    self.device_index = None
                    return False
                logger.info(f"设置输出设备: {device_info.get('name')} (索引: {device_index})")
                self.device_index = device_index
            else:
                logger.info("使用默认输出设备")
                self.device_index = None
            return True
        except Exception as e:
            logger.exception(f"设置输出设备失败: {str(e)}")
            self.device_index = None
            return False
    
    def start(self) -> bool:
        """启动音频输出"""
        if self.is_running:
            logger.warning("音频输出已经在运行")
            return False
            
        try:
            self.is_running = True
            self.audio_queue.clear()
            
            # 创建并启动流
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device_index,
                callback=self._output_callback
            )
            self.stream.start()
            logger.info("音频输出已启动")
            return True
        except Exception as e:
            logger.exception(f"启动音频输出失败: {str(e)}")
            self.is_running = False
            self.stream = None
            return False
    
    def stop(self):
        """停止音频输出"""
        if not self.is_running:
            return
            
        try:
            self.is_running = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.audio_queue.clear()
            logger.info("音频输出已停止")
        except Exception as e:
            logger.exception(f"停止音频输出失败: {str(e)}")
    
    def add_audio(self, audio: np.ndarray):
        """添加音频数据到队列"""
        if self.is_running:
            self.audio_queue.append(audio)
    
    def _output_callback(self, outdata, frames, time_info, status):
        """输出回调函数"""
        if status:
            logger.warning(f"输出回调状态: {status}")
            
        try:
            if self.audio_queue:
                data = self.audio_queue.popleft()
                # 确保数据长度正确
                if len(data) >= frames:
                    outdata[:] = data[:frames]
                else:
                    # 填充不足部分
                    outdata[:len(data)] = data
                    outdata[len(data):].fill(0)
            else:
                outdata.fill(0)
        except Exception as e:
            logger.exception(f"输出回调处理失败: {str(e)}")
            outdata.fill(0)  # 出错时输出静音


class ProcessingPipeline:
    """处理管道，负责ASR->翻译->TTS的处理流程"""
    
    def __init__(self):
        """初始化处理管道"""
        # 注意：这里不直接初始化这些管理器，在configure方法中才初始化
        self.asr_manager = None
        self.translation_manager = None
        self.tts_manager = None
        
        self.source_language = "zh"
        self.target_language = "en"
        
        self.text_results = collections.deque(maxlen=10)
        self._status_observers = []
        self._lock = threading.Lock()
    
    def add_status_observer(self, observer: Callable[[StatusEvent], None]):
        """添加状态观察者"""
        with self._lock:
            self._status_observers.append(observer)
    
    def remove_status_observer(self, observer: Callable[[StatusEvent], None]):
        """移除状态观察者"""
        with self._lock:
            if observer in self._status_observers:
                self._status_observers.remove(observer)
    
    def _notify_status(self, event: StatusEvent):
        """通知所有观察者"""
        with self._lock:
            for observer in self._status_observers:
                try:
                    observer(event)
                except Exception as e:
                    logger.error(f"状态通知失败: {str(e)}")
    
    def configure(self, asr_config: Dict[str, Any], 
                 translation_config: Dict[str, Any], 
                 tts_config: Dict[str, Any]) -> bool:
        """配置处理管道"""
        try:
            # 构建配置
            system_config = config.get("system", {})
            audio_config = config.get("audio", {})
            
            # 合并所有配置
            full_config = {
                "system": system_config,
                "audio": audio_config,
                "user": {
                    "asr": asr_config,
                    "translation": translation_config,
                    "tts": tts_config
                }
            }
            
            # 初始化管理器
            if self.asr_manager is None:
                self.asr_manager = ASRManager(full_config)
                
            if self.translation_manager is None:
                self.translation_manager = TranslationManager(full_config)
                
            if self.tts_manager is None:
                self.tts_manager = TTSManager(full_config)
            
            # 预初始化所有引擎
            self.asr_manager.pre()
            self.translation_manager.pre()
            self.tts_manager.pre()
            
            # 设置语言
            self.source_language = translation_config.get("source_language", "zh")
            self.target_language = translation_config.get("target_language", "en")
            
            logger.info("处理管道配置成功")
            return True
        except Exception as e:
            logger.exception(f"配置处理管道失败: {str(e)}")
            return False
    
    def set_languages(self, source_language: str, target_language: str):
        """设置语言"""
        self.source_language = source_language
        self.target_language = target_language
        logger.info(f"设置语言: 源语言={source_language}, 目标语言={target_language}")
    
    def process_audio(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """处理音频数据，返回合成的音频（如果有）"""
        try:
            # 1. 语音识别
            start_time = time.time()
            text = self.asr_manager.transcribe(audio_data)
            asr_latency = time.time() - start_time
            self._notify_status(StatusEvent(
                StatusEventType.LATENCY,
                {"asr": asr_latency}
            ))
            
            if not text:
                return None
                
            # 2. 翻译
            start_time = time.time()
            translated_text = self.translation_manager.translate(text)
            translation_latency = time.time() - start_time
            self._notify_status(StatusEvent(
                StatusEventType.LATENCY,
                {"mt": translation_latency}
            ))
            
            if not translated_text:
                return None
                
            # 3. 语音合成
            start_time = time.time()
            audio = self.tts_manager.synthesize(translated_text)
            tts_latency = time.time() - start_time
            self._notify_status(StatusEvent(
                StatusEventType.LATENCY,
                {"tts": tts_latency}
            ))
            
            # 存储结果
            self.text_results.append((text, translated_text))
            
            # 通知最新文本
            self._notify_status(StatusEvent(
                StatusEventType.TEXT,
                {
                    "original": text,
                    "translated": translated_text,
                    "timestamp": time.time()
                }
            ))
            
            return audio
        except Exception as e:
            logger.exception(f"处理音频数据失败: {str(e)}")
            self._notify_status(StatusEvent(
                StatusEventType.ERROR,
                {"error": str(e)}
            ))
            return None
    
    def get_latest_results(self, count: int = 1) -> List[Tuple[str, str]]:
        """获取最近的翻译结果"""
        results = list(self.text_results)
        if not results:
            return []
        return results[-count:]
    
    def stop(self):
        """停止所有处理组件"""
        if self.asr_manager:
            self.asr_manager.stop()
        if self.translation_manager:
            self.translation_manager.stop()
        if self.tts_manager:
            self.tts_manager.stop()


class StreamProcessor:
    """流式处理器，负责协调音频捕获、处理和输出"""
    
    def __init__(self):
        # 读取系统配置
        audio_config = config.get("audio", {})
        sample_rate = audio_config.get("input", {}).get("sample_rate", 16000)
        channels = audio_config.get("input", {}).get("channels", 1)
        chunk_duration = audio_config.get("input", {}).get("chunk_duration", 0.5)
        noise_suppression = audio_config.get("input", {}).get("noise_suppression", True)
        
        # 创建组件
        self.audio_capture = AudioCapture(
            sample_rate=sample_rate,
            channels=channels,
            chunk_duration=chunk_duration,
            noise_suppression=noise_suppression
        )
        self.audio_output = AudioOutput(
            sample_rate=audio_config.get("output", {}).get("sample_rate", 16000),
            channels=audio_config.get("output", {}).get("channels", 1)
        )
        self.pipeline = ProcessingPipeline()
        self.status_manager = StatusManager()
        
        # 添加状态观察者
        self.pipeline.add_status_observer(self._handle_status_event)
        
        # 内部状态
        self.is_running = False
        self.is_processing = True  # 是否正在处理音频数据
        self.audio_queue = collections.deque(maxlen=20)
        self.audio_event = threading.Event()
        self.processing_thread = None
        self.error_count = 0
        self.max_errors = 3
        self.error_lock = threading.Lock()
    
    def _handle_status_event(self, event: StatusEvent):
        """处理状态事件"""
        if event.event_type == StatusEventType.LATENCY:
            self.status_manager.update_latency(**event.data)
        elif event.event_type == StatusEventType.ERROR:
            self.status_manager.update_exception(event.data.get("error"))
        elif event.event_type == StatusEventType.TEXT:
            self.status_manager.update_latest_text(event.data)
        elif event.event_type == StatusEventType.PROCESSING:
            self.status_manager.update_running_state(
                event.data.get("is_running", False),
                event.data.get("is_processing", False)
            )
    
    def configure(self, asr: Dict[str, Any], translation: Dict[str, Any], tts: Dict[str, Any]) -> bool:
        """配置处理器"""
        return self.pipeline.configure(asr, translation, tts)
    
    def set_languages(self, source_language: str, target_language: str):
        """设置语言"""
        self.pipeline.set_languages(source_language, target_language)
    
    def set_audio_devices(self, input_device: str, output_device: str):
        """设置音频设备"""
        try:
            # 获取所有音频设备
            devices = sd.query_devices()
            logger.info(f"""系统可用音频设备列表: {[f"{i}: {dev.get('name')}" for i, dev in enumerate(devices)]}""")
            
            # 解析设备索引
            def get_device_index(device_str):
                if device_str == "default":
                    return None
                
                device_data = device_str.split("(")[0].strip()
                for i, dev in enumerate(devices):
                    if dev.get('name').strip() == device_data:
                        return i
                
                logger.warning(f"未找到设备: {device_str}")
                return None
            
            # 设置设备
            input_idx = get_device_index(input_device)
            output_idx = get_device_index(output_device)
            
            self.audio_capture.set_device(input_idx)
            self.audio_output.set_device(output_idx)
            
            logger.info(f"设置音频设备: 输入={input_device}(索引:{input_idx}), 输出={output_device}(索引:{output_idx})")
            return True
        except Exception as e:
            logger.exception(f"设置音频设备失败: {str(e)}")
            return False
    
    def set_tts_engine(self, engine: str, model_version: Optional[str] = None):
        """设置TTS引擎和模型版本 - 此方法需要更复杂的实现"""
        logger.warning("set_tts_engine方法未实现，请在配置阶段配置TTS引擎")
        return False
    
    def set_asr_engine(self, engine: str, model_version: Optional[str] = None):
        """设置ASR引擎和模型版本 - 此方法需要更复杂的实现"""
        logger.warning("set_asr_engine方法未实现，请在配置阶段配置ASR引擎")
        return False
    
    def set_translation_engine(self, engine: str, model_version: Optional[str] = None):
        """设置翻译引擎和模型版本 - 此方法需要更复杂的实现"""
        logger.warning("set_translation_engine方法未实现，请在配置阶段配置翻译引擎")
        return False
    
    def start(self) -> bool:
        """启动流处理"""
        if self.is_running:
            logger.warning("流处理器已经在运行")
            return False
        
        try:
            self.is_running = True
            self.is_processing = True
            self.error_count = 0
            
            # 更新状态
            self.status_manager.update_running_state(True, True)
            self.status_manager.start()
            
            # 启动音频输出
            if not self.audio_output.start():
                raise Exception("启动音频输出失败")
            
            # 启动处理线程
            self.processing_thread = threading.Thread(target=self._process_audio_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # 启动音频捕获（最后启动，以确保其他组件已就绪）
            if not self.audio_capture.start(self._audio_callback):
                raise Exception("启动音频捕获失败")
            
            logger.info("流处理器启动成功")
            return True
        except Exception as e:
            logger.exception(f"启动流处理器失败: {str(e)}")
            self.stop()  # 清理可能已启动的组件
            return False
    
    def stop(self):
        """停止流处理"""
        if not self.is_running:
            return
            
        try:
            self.is_running = False
            self.is_processing = False
            
            # 更新状态
            self.status_manager.update_running_state(False, False)
            self.status_manager.stop()
            
            # 停止音频捕获和输出
            self.audio_capture.stop()
            self.audio_output.stop()
            
            # 通知处理线程退出并等待
            self.audio_event.set()
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            
            # 停止所有处理管道
            self.pipeline.stop()
            
            # 清空队列
            self.audio_queue.clear()
            
            logger.info("流处理器停止成功")
        except Exception as e:
            logger.exception(f"停止流处理器失败: {str(e)}")
    
    def _audio_callback(self, audio_data: np.ndarray, time_info: Dict):
        """音频捕获回调"""
        try:
            # 将音频数据添加到队列
            self.audio_queue.append(audio_data)
            
            # 通知处理线程
            self.audio_event.set()
            
            # 更新状态
            if hasattr(time_info, 'inputBufferAdcTime'):
                self.status_manager.update_latency(asr_latency=time.time() - time_info.inputBufferAdcTime)
        except Exception as e:
            logger.exception(f"音频回调处理失败: {str(e)}")
            self._handle_exception()
    
    def _process_audio_thread(self):
        """音频处理线程"""
        while self.is_running:
            try:
                # 等待新数据
                self.audio_event.wait(timeout=0.1)
                self.audio_event.clear()
                
                # 如果不需要处理，跳过
                if not self.is_processing:
                    continue
                
                # 检查是否有音频数据
                if not self.audio_queue:
                    continue
                
                # 获取音频数据
                audio_data = self.audio_queue.popleft()
                
                # 处理音频
                output_audio = self.pipeline.process_audio(audio_data)
                
                # 如果有输出音频，添加到输出队列
                if output_audio is not None:
                    self.audio_output.add_audio(output_audio)
                
            except Exception as e:
                logger.exception(f"处理音频数据失败: {str(e)}")
                self._handle_exception()
    
    def _handle_exception(self):
        """处理错误"""
        with self.error_lock:
            self.error_count += 1
            if self.error_count >= self.max_errors:
                logger.error(f"错误次数过多 ({self.error_count}/{self.max_errors})，停止处理")
                self.stop()
            else:
                logger.warning(f"发生错误 ({self.error_count}/{self.max_errors})")
                # 更新状态
                self.status_manager.update_error_count(self.error_count)
                self.status_manager.update_exception(f"发生错误 ({self.error_count}/{self.max_errors})")
    
    def translate_audio(self, audio_path: str) -> Tuple[str, np.ndarray]:
        """翻译音频文件"""
        try:
            # 读取音频文件
            audio, _ = sf.read(audio_path)
            
            # 使用处理管道处理音频
            output_audio = self.pipeline.process_audio(audio)
            
            if output_audio is None:
                raise Exception("处理音频失败")
            
            # 获取最新的翻译结果
            results = self.pipeline.get_latest_results(1)
            if not results:
                raise Exception("未能获取翻译结果")
                
            _, translated_text = results[0]
            
            return translated_text, output_audio
        except Exception as e:
            logger.exception(f"翻译音频文件失败: {str(e)}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        return self.status_manager.get_status()