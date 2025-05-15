#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from typing import Dict, Any
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTabWidget, QTextEdit, QLineEdit,
     QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QCheckBox, QMessageBox,
     QSplitter, QSlider, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer

from utils.logger import logger
from app.ui_interface import UIEventType, ui_manager, UICallbackID
from app.ui_adapter import ui_adapter


class MainWindow(QMainWindow):
    def __init__(self):
        """初始化主窗口"""
        try:
            super().__init__()
        
            # 获取管理器实例
            self.ui_manager = ui_manager
            self.ui_adapter = ui_adapter
            
            # 运行时状态
            self.current_download = None
            self.running_duration = 0
            self.available_gpus = []
            
            # 创建UI组件
            self._setup_ui()
            
            # 注册全局事件处理器
            self._register_event_handlers()
            
            # 初始化定时器
            self._setup_timers()
            
            # 初始化GPU设置和更新音频设备列表
            self.init_gpu_settings()
            self.update_audio_devices()
            
            # 更新模型状态
            self.refresh_model_status()
            
            # 触发初始化事件 - 加载默认模型配置
            self.ui_manager.trigger_event(UIEventType.SYSTEM_READY, {})
            
            # 初始化加载各引擎的模型版本
            self._initialize_model_versions()
            
        except Exception as e:
            logger.exception(f"初始化主窗口失败: {str(e)}")
            raise

    def _initialize_model_versions(self):
        """初始化加载所有引擎的模型版本"""
        try:
            logger.info("开始初始化模型版本列表...")
            
            # 加载TTS模型版本
            current_tts_engine = self.tts_engine.currentText()
            if current_tts_engine:
                self.ui_manager.trigger_event(
                    UIEventType.GET_MODEL_VERSIONS,
                    {
                        "engine_type": "tts",
                        "engine_name": current_tts_engine,
                        "callback_id": "tts"
                    }
                )
            
            # 加载ASR模型版本
            current_asr_engine = self.realtime_asr_engine.currentText()
            if current_asr_engine:
                self.ui_manager.trigger_event(
                    UIEventType.GET_MODEL_VERSIONS,
                    {
                        "engine_type": "asr",
                        "engine_name": current_asr_engine,
                        "callback_id": "realtime_asr"
                    }
                )
            
            # 加载单文件处理的ASR模型版本
            if hasattr(self, 'file_asr_engine') and self.file_asr_engine.currentText():
                self.ui_manager.trigger_event(
                    UIEventType.GET_MODEL_VERSIONS,
                    {
                        "engine_type": "asr", 
                        "engine_name": self.file_asr_engine.currentText(),
                        "callback_id": "asr"
                    }
                )
            
            logger.info("模型版本列表初始化请求已发送")
        except Exception as e:
            logger.exception(f"初始化模型版本列表失败: {str(e)}")
    
    def _setup_ui(self):
        """设置UI界面"""
        # 设置窗口标题和大小
        self.setWindowTitle(self.ui_manager.config.get_config("app.name", "Voice Translate Pro"))
        self.setMinimumSize(800, 600)
        self.resize(1200, 800)
        
        # 创建主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 创建标签页
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # 添加各个标签页
        tabs.addTab(self.create_realtime_tab(), "实时翻译")
        tabs.addTab(self.create_single_tab(), "单文件翻译")
        tabs.addTab(self.create_settings_tab(), "设置")
        tabs.addTab(self.create_model_tab(), "模型管理")
    
    def _setup_timers(self):
        """设置定时器"""
        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.setInterval(500)  # 每500毫秒更新一次
        self.status_timer.timeout.connect(self.update_status)
        
        # 运行时长更新定时器
        self.duration_timer = QTimer()
        self.duration_timer.setInterval(1000)  # 每1秒更新一次
        self.duration_timer.timeout.connect(self.update_duration)
        
        # 下载进度更新定时器
        self.progress_timer = QTimer()
        self.progress_timer.setInterval(100)  # 每100毫秒更新一次
        self.progress_timer.timeout.connect(self.update_download_progress)
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 翻译控制事件
        self.ui_manager.register_event_handler(UIEventType.START_TRANSLATION, self._handle_start_translation)
        self.ui_manager.register_event_handler(UIEventType.STOP_TRANSLATION, self._handle_stop_translation)
        
        # 状态和进度事件
        self.ui_manager.register_event_handler(UIEventType.UPDATE_PROGRESS, self._handle_progress_update)
        self.ui_manager.register_status_observer(self._handle_status_update)
        
        # 模型事件
        self.ui_manager.register_event_handler(UIEventType.MODEL_LOADED, self._handle_model_loaded)
        self.ui_manager.register_event_handler(UIEventType.MODEL_VERSIONS_LOADED, self._handle_model_versions_loaded)
        self.ui_manager.register_event_handler(UIEventType.MODEL_SET_RESULT, self._handle_model_set_result)
        
        # 错误事件
        self.ui_manager.register_event_handler(UIEventType.ERROR, self._handle_error)

        # TTS语音事件
        self.ui_manager.register_event_handler(UIEventType.TTS_VOICES_LOADED, self._handle_tts_voices_loaded)
    
    # =============== 事件处理器 ===============
    
    def _handle_start_translation(self, data: Dict[str, Any]):
        """处理开始翻译事件"""
        self._update_ui_state(True)
    
    def _handle_stop_translation(self, data: Dict[str, Any]):
        """处理停止翻译事件"""
        self._update_ui_state(False)
    
    def _handle_status_update(self, status: Dict[str, Any]):
        """处理状态更新"""
        try:
            # 更新CPU和GPU使用率
            cpu_usage = status.get("cpu_usage", 0)
            gpu_usage = status.get("gpu_usage", 0)
            self.resource_text.setText(f"CPU: {cpu_usage:.1f}% | GPU: {gpu_usage:.1f}%")
            
            # 更新延迟信息
            asr_latency = status.get("asr_latency", 0) * 1000  # 转换为毫秒
            mt_latency = status.get("mt_latency", 0) * 1000
            tts_latency = status.get("tts_latency", 0) * 1000
            self.latency_text.setText(
                f"识别: {asr_latency:.0f}ms | 翻译: {mt_latency:.0f}ms | TTS: {tts_latency:.0f}ms"
            )
            
            # 更新最新文本
            latest_text = status.get("latest_text", {})
            if latest_text:
                original = latest_text.get("original", "")
                translated = latest_text.get("translated", "")
                if original and translated:
                    self.current_source.setPlainText(original)
                    self.current_translation.setPlainText(translated)
            
            # 更新错误状态
            error = status.get("error")
            if error:
                self.status_text.setText(f"错误: {error}")
                self.status_text.setStyleSheet("color: red;")
            else:
                self.status_text.setStyleSheet("")
                
            # 更新运行状态
            is_running = status.get("is_running", False)
            is_paused = status.get("is_paused", False)
            
            if is_running and not is_paused:
                self.status_text.setText("正在运行...")
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
            elif is_running and is_paused:
                self.status_text.setText("已暂停")
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
            else:
                self.status_text.setText("已停止")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                
        except Exception as e:
            logger.exception(f"更新状态失败: {str(e)}")
    
    def _handle_progress_update(self, data: Dict[str, Any]):
        """处理进度更新事件"""
        try:
            progress = data.get("progress", 0)
            progress_type = data.get("type", "")
            
            # 处理模型下载进度
            if progress_type == "model_download" and self.current_download:
                # 确保是当前下载的模型
                engine_type = data.get("engine_type")
                engine_name = data.get("engine_name")
                model_name = data.get("model_name")
                
                if (engine_type == self.current_download.get("model_type") and
                    engine_name == self.current_download.get("engine_name") and
                    model_name == self.current_download.get("model_tag")):
                    
                    # 更新进度条
                    new_value = int(progress * 100)
                    self.download_progress.setValue(new_value)
                    QApplication.processEvents()
                    
        except Exception as e:
            logger.exception(f"处理进度更新失败: {str(e)}")
    
    def _handle_model_loaded(self, data: Dict[str, Any]):
        """处理模型加载完成事件"""
        try:
            engine_type = data.get("engine_type")
            engine_name = data.get("engine_name")
            model_name = data.get("model_name")
            success = data.get("success", False)
            is_deleted = data.get("deleted", False)
            
            # 刷新模型表格
            self.refresh_model_status()
            
            # 如果是当前下载的模型
            if self.current_download and (
                engine_type == self.current_download.get("model_type") and
                engine_name == self.current_download.get("engine_name") and
                model_name == self.current_download.get("model_tag")
            ):
                # 停止进度定时器
                self.progress_timer.stop()
                self.download_progress.setValue(100)
                
                # 显示完成消息
                if success:
                    action = "删除" if is_deleted else "下载"
                    QMessageBox.information(self, "成功", f"模型{action}成功")
                else:
                    QMessageBox.critical(self, "失败", f"模型操作失败")
                
                # 清除当前下载
                self.current_download = None
                
            # 更新相关模型版本列表
            if engine_type == "asr":
                self.update_asr_model_versions()
                self.update_realtime_asr_model_versions()
            elif engine_type == "tts":
                self.update_tts_model_versions()
            
        except Exception as e:
            logger.exception(f"处理模型加载完成失败: {str(e)}")


    def _handle_model_versions_loaded(self, data: Dict[str, Any]):
        """处理模型版本列表加载完成事件"""
        try:
            engine_type = data.get("engine_type")
            engine_name = data.get("engine_name")
            callback_id = data.get("callback_id")
            versions = data.get("versions", [])
            current_model = data.get("current_model", "")

            # 确定要更新的UI组件
            model_version_combobox = None
            if callback_id == UICallbackID.TTS_MODEL_VERSIONS:
                model_version_combobox = self.tts_model_version
            elif callback_id == UICallbackID.ASR_MODEL_VERSIONS:
                model_version_combobox = self.realtime_asr_model_version
            elif callback_id == UICallbackID.REALTIME_ASR_MODEL_VERSIONS:
                model_version_combobox = self.realtime_asr_model_version


            # 更新UI
            if model_version_combobox is not None:
                model_version_combobox.clear()

                if versions:
                    model_version_combobox.addItems(versions)
                    model_version_combobox.setEnabled(True)

                    # 设置当前版本
                    if current_model in versions:
                        model_version_combobox.setCurrentText(current_model)
                    else:
                        # 如果当前没有选中的版本，默认选择第一个
                        model_version_combobox.setCurrentText(versions[0])
                else:
                    model_version_combobox.setEnabled(False)
                    model_version_combobox.addItem("无可用版本")

        except Exception as e:
            logger.exception(f"处理模型版本列表加载完成失败: {str(e)}")

    def _handle_model_set_result(self, data: Dict[str, Any]):
        """处理模型设置结果事件"""
        try:
            callback_id = data.get("callback_id")
            engine_type = data.get("engine_type")
            model_version = data.get("model_version")
            success = data.get("success", False)
            error = data.get("error", "")
            
            if success:
                # 更新配置
                if callback_id == UICallbackID.SET_ASR_MODEL or callback_id == UICallbackID.SET_REALTIME_ASR_MODEL:
                    self.ui_manager.config().update_config("asr.engine", engine_type)
                    self.ui_manager.config().update_config("asr.model", model_version)

                elif callback_id == UICallbackID.SET_TTS_MODEL:
                    self.ui_manager.config().update_config("tts.engine", engine_type)
                    self.ui_manager.config().update_config("tts.model", model_version)

                logger.info(f"已切换到{engine_type}模型版本: {model_version}")
            else:
                QMessageBox.warning(self, "警告", f"切换到模型版本失败: {error}")
            
        except Exception as e:
            logger.exception(f"处理模型设置结果失败: {str(e)}")
    
    def _handle_error(self, data: Dict[str, Any]):
        """处理错误事件"""
        try:
            message = data.get("message", "未知错误")
            QMessageBox.critical(self, "错误", message)
        except Exception as e:
            logger.exception(f"处理错误失败: {str(e)}")
    
    # =============== UI操作方法 ===============
    
    def _update_ui_state(self, is_running: bool):
        """更新UI状态"""
        self.start_btn.setEnabled(not is_running)
        self.stop_btn.setEnabled(is_running)
        
        if is_running:
            self.status_text.setText("正在运行...")
            # 启动状态更新定时器
            self.status_timer.start()
            self.duration_timer.start()
            self.running_duration = 0
        else:
            self.status_text.setText("已停止")
            # 停止状态更新定时器
            self.status_timer.stop()
            self.duration_timer.stop()
            # 重置状态显示
            self.resource_text.setText("CPU: 0% | GPU: 0%")
            self.latency_text.setText("识别: 0ms | 翻译: 0ms | TTS: 0ms")
    
    def _extract_lang_code(self, lang_text: str) -> str:
        """从语言选项文本中提取语言代码"""
        return lang_text.split()[0]  # 提取空格前的语言代码
        
    def _extract_voice_name(self, voice_text: str) -> str:
        """从声音选项文本中提取声音名称"""
        return voice_text.split(" (")[0]  # 提取括号前的声音代码
    
    # =============== 用户操作响应 ===============
    
    def start_translation(self):
        """开始翻译"""
        try:
            # 获取界面参数
            input_device = self.input_device.currentText()
            output_device = self.output_device.currentText()
            source_language = self._extract_lang_code(self.single_src_lang.currentText())
            target_language = self._extract_lang_code(self.single_tgt_lang.currentText())
            
            # ASR参数
            asr_engine = self.realtime_asr_engine.currentText()
            asr_model_version = self.realtime_asr_model_version.currentText()
            if asr_model_version in ("无可用版本", "获取失败"):
                asr_model_version = None
                
            # TTS参数
            tts_engine = self.tts_engine.currentText()
            tts_model_version = self.tts_model_version.currentText()
            if tts_model_version in ("无可用版本", "获取失败"):
                tts_model_version = None
            tts_voice = self._extract_voice_name(self.tts_voice.currentText())
            
            # GPU设置
            use_gpu = self.available_gpus and self.enable_gpu.isChecked()
            gpu_device = self.gpu_device.currentText().split(" (")[0] if use_gpu else None
            use_half = self.half_precision.isChecked() if use_gpu else False
            
            # 收集配置参数
            config_data = {
                "input_device": input_device,
                "output_device": output_device,
                "source_language": source_language,
                "target_language": target_language,
                "asr": {
                    "engine": asr_engine,
                    "model_version": asr_model_version
                },
                "tts": {
                    "engine": tts_engine,
                    "model_version": tts_model_version,
                    "voice": tts_voice,
                    "use_cache": self.tts_use_cache.isChecked(),
                    "max_retries": self.tts_max_retries.value()
                },
                "translation": {
                    "engine": "nllb",
                    "model_version": "base"
                },
                "gpu": {
                    "enable": use_gpu,
                    "device": gpu_device,
                    "half_precision": use_half
                }
            }
            
            # 触发开始翻译事件
            self.ui_manager.trigger_event(
                UIEventType.START_TRANSLATION,
                config_data
            )
            
        except Exception as e:
            logger.exception(f"开始翻译失败: {str(e)}")
            self.ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"启动失败: {str(e)}"}
            )
    
    def stop_translation(self):
        """停止翻译"""
        try:
            # 触发停止翻译事件
            self.ui_manager.trigger_event(
                UIEventType.STOP_TRANSLATION,
                {}
            )
        except Exception as e:
            logger.exception(f"停止翻译失败: {str(e)}")
            self.ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"停止失败: {str(e)}"}
            )
    
    def translate_audio(self):
        """翻译音频文件"""
        try:
            audio_path = self.audio_input.text()
            if not audio_path or not os.path.exists(audio_path):
                QMessageBox.warning(self, "错误", "请选择有效的音频文件")
                return
                
            # 准备参数
            source_language = self._extract_lang_code(self.single_src_lang.currentText())
            target_language = self._extract_lang_code(self.single_tgt_lang.currentText())

            # 触发文件翻译事件
            self.ui_manager.trigger_event(
                UIEventType.TRANSLATE_FILE,
                {
                    "audio_path": audio_path,
                    "source_language": source_language,
                    "target_language": target_language,
                    "callback_id": UICallbackID.TRANSLATE_AUDIO_FILE
                }
            )

            # 注册一次性事件处理器
            def handle_translation_result(data):
                if data.get("callback_id") == UICallbackID.TRANSLATE_AUDIO_FILE:
                    # 显示结果
                    translated_text = data.get("translated_text", "")
                    self.translated_text.setText(translated_text)

                    # 注销事件处理器
                    self.ui_manager.unregister_event_handler(UIEventType.TRANSLATION_RESULT, handle_translation_result)
            
            # 注册错误处理器
            def handle_error(data):
                if "translate_file" in data.get("message", ""):
                    # 注销事件处理器
                    self.ui_manager.unregister_event_handler(UIEventType.ERROR, handle_error)
                    self.ui_manager.unregister_event_handler(UIEventType.TRANSLATION_RESULT, handle_translation_result)
                    
                    # 显示错误
                    QMessageBox.critical(self, "错误", data.get("message", "翻译失败"))
            
            # 注册处理器
            self.ui_manager.register_event_handler(UIEventType.TRANSLATION_RESULT, handle_translation_result)
            self.ui_manager.register_event_handler(UIEventType.ERROR, handle_error)
            
        except Exception as e:
            logger.exception(f"翻译音频文件失败: {str(e)}")
            self.ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"翻译失败: {str(e)}"}
            )


    def update_tts_model_versions(self):
        """更新TTS模型版本列表"""
        try:
            engine_name = self.tts_engine.currentText()

            # 触发获取模型列表事件
            self.ui_manager.trigger_event(
                UIEventType.GET_MODEL_VERSIONS,
                {
                    "engine_type": "tts",
                    "engine_name": engine_name,
                    "callback_id": UICallbackID.TTS_MODEL_VERSIONS
                }
            )

        except Exception as e:
            logger.exception(f"更新TTS模型版本列表失败: {str(e)}")
            self.tts_model_version.clear()
            self.tts_model_version.addItem("获取失败")
            self.tts_model_version.setEnabled(False)
    
    def update_tts_model(self):
        """更新TTS模型版本"""
        try:
            engine_name = self.tts_engine.currentText()
            model_version = self.tts_model_version.currentText()

            if model_version and model_version != "无可用版本" and model_version != "获取失败":
                # 触发设置模型事件
                self.ui_manager.trigger_event(
                    UIEventType.SET_MODEL,
                    {
                        "engine_type": "tts",
                        "engine_name": engine_name,
                        "model_version": model_version,
                        "callback_id": UICallbackID.SET_TTS_MODEL
                    }
                )

        except Exception as e:
            logger.exception(f"更新TTS模型版本失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新TTS模型版本失败: {str(e)}")
    
    def update_asr_model_versions(self):
        """更新ASR模型版本列表"""
        try:
            engine_name = self.realtime_asr_engine.currentText()

            # 触发获取模型列表事件
            self.ui_manager.trigger_event(
                UIEventType.GET_MODEL_VERSIONS,
                {
                    "engine_type": "asr",
                    "engine_name": engine_name,
                    "callback_id": UICallbackID.ASR_MODEL_VERSIONS
                }
            )

        except Exception as e:
            logger.exception(f"更新ASR模型版本列表失败: {str(e)}")
            self.realtime_asr_model_version.clear()
            self.realtime_asr_model_version.addItem("获取失败")
            self.realtime_asr_model_version.setEnabled(False)
    

    def update_asr_model(self):
        """更新ASR模型版本"""
        try:
            engine_name = self.realtime_asr_engine.currentText()
            model_version = self.realtime_asr_model_version.currentText()

            if model_version and model_version != "无可用版本" and model_version != "获取失败":
                # 触发设置模型事件
                self.ui_manager.trigger_event(
                    UIEventType.SET_MODEL,
                    {
                        "engine_type": "asr",
                        "engine_name": engine_name,
                        "model_version": model_version,
                        "callback_id": UICallbackID.SET_ASR_MODEL
                    }
                )

        except Exception as e:
            logger.exception(f"更新ASR模型版本失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新ASR模型版本失败: {str(e)}")
    
    def update_realtime_asr_model_versions(self):
        """更新实时ASR模型版本列表"""
        try:
            engine_name = self.realtime_asr_engine.currentText()

            # 触发获取模型列表事件
            self.ui_manager.trigger_event(
                UIEventType.GET_MODEL_VERSIONS,
                {
                    "engine_type": "asr",
                    "engine_name": engine_name,
                    "callback_id": UICallbackID.REALTIME_ASR_MODEL_VERSIONS
                }
            )

        except Exception as e:
            logger.exception(f"更新实时ASR模型版本列表失败: {str(e)}")
            self.realtime_asr_model_version.clear()
            self.realtime_asr_model_version.addItem("获取失败")
            self.realtime_asr_model_version.setEnabled(False)

    def update_realtime_asr_model(self):
        """更新实时ASR模型版本"""
        try:
            engine_name = self.realtime_asr_engine.currentText()
            model_version = self.realtime_asr_model_version.currentText()

            if model_version and model_version != "无可用版本" and model_version != "获取失败":
                # 触发设置模型事件
                self.ui_manager.trigger_event(
                    UIEventType.SET_MODEL,
                    {
                        "engine_type": "asr",
                        "engine_name": engine_name,
                        "model_version": model_version,
                        "callback_id": UICallbackID.SET_REALTIME_ASR_MODEL
                    }
                )

        except Exception as e:
            logger.exception(f"更新实时ASR模型版本失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新实时ASR模型版本失败: {str(e)}")

    # =============== 定时器回调 ===============
    
    def update_status(self):
        """更新状态（定时器调用）"""
        try:
            # 不再直接从processor获取状态，而是触发一个事件来请求状态更新
            self.ui_manager.trigger_event(
                UIEventType.REQUEST_STATUS_UPDATE,
                {"request_update": True}
            )
        except Exception as e:
            logger.exception(f"请求状态更新失败: {str(e)}")
    
    def update_duration(self):
        """更新运行时长"""
        try:
            self.running_duration += 1
            self.duration_text.setText(f"{self.running_duration // 60:02d}:{self.running_duration % 60:02d}")
        except Exception as e:
            logger.exception(f"更新运行时长失败: {str(e)}")
    
    def update_download_progress(self):
        """更新下载进度（定时器调用）"""
        try:
            if hasattr(self, 'current_download') and self.current_download:
                current_value = self.download_progress.value()
                
                # 完成时处理（进度为100%）
                if current_value >= 100:
                    self.progress_timer.stop()
                    self.download_progress.setValue(100)
        except Exception as e:
            self.progress_timer.stop()
            logger.exception(f"更新进度失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"更新进度失败: {str(e)}")
            self.current_download = None
    
    # =============== 其他方法 ===============
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        try:
            # 停止翻译（如果正在进行）
            if hasattr(self, 'stop_btn') and self.stop_btn.isEnabled():
                # 使用UI事件触发停止翻译
                self.ui_manager.trigger_event(UIEventType.STOP_TRANSLATION, {})
            
            # 发送系统退出事件
            self.ui_manager.trigger_event(UIEventType.SYSTEM_EXIT, {})
            
            # 保存配置
            self.ui_manager.save_config()
            
            logger.info("主窗口关闭")
            event.accept()
        except Exception as e:
            logger.exception(f"窗口关闭处理失败: {str(e)}")
            event.accept()  # 始终接受关闭事件，避免窗口无法关闭

    def create_realtime_tab(self):
        """创建实时翻译标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        
        # 创建一个分割器，替换原来的QHBoxLayout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(5)  # 设置分割线宽度
        splitter.setChildrenCollapsible(False)  # 防止子控件被完全折叠
        
        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)  # 设置最小宽度
        left_panel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        
        # 设备选择
        device_group = QGroupBox("设备设置")
        device_layout = QVBoxLayout()
        
        self.input_device = QComboBox()
        device_layout.addWidget(QLabel("输入设备:"))
        device_layout.addWidget(self.input_device)
        
        self.output_device = QComboBox()
        device_layout.addWidget(QLabel("输出设备:"))
        device_layout.addWidget(self.output_device)
        
        device_group.setLayout(device_layout)
        left_layout.addWidget(device_group)
        
        # ASR引擎选择
        asr_group = QGroupBox("ASR设置")
        asr_layout = QVBoxLayout()
        
        self.realtime_asr_engine = QComboBox()
        self.realtime_asr_engine.addItems(ui_manager.get_ui_datas_config("asr_engine"))
        asr_layout.addWidget(QLabel("ASR引擎:"))
        asr_layout.addWidget(self.realtime_asr_engine)
        
        # 添加ASR模型版本选择
        self.realtime_asr_model_version = QComboBox()
        self.realtime_asr_model_version.setEnabled(False)  # 初始禁用
        asr_layout.addWidget(QLabel("模型版本:"))
        asr_layout.addWidget(self.realtime_asr_model_version)
        
        # 当ASR引擎改变时更新模型版本列表
        self.realtime_asr_engine.currentTextChanged.connect(self.update_realtime_asr_model_versions)
        
        asr_group.setLayout(asr_layout)
        left_layout.addWidget(asr_group)
        
        # 语言设置
        lang_group = QGroupBox("语言设置")
        lang_layout = QHBoxLayout()
        
        self.single_src_lang = QComboBox()
        self.single_src_lang.addItems(ui_manager.get_ui_datas_config("single_src_lang"))
        lang_layout.addWidget(QLabel("源语言:"))
        lang_layout.addWidget(self.single_src_lang)
        
        self.single_tgt_lang = QComboBox()
        self.single_tgt_lang.addItems(ui_manager.get_ui_datas_config("single_tgt_lang"))
        lang_layout.addWidget(QLabel("目标语言:"))
        lang_layout.addWidget(self.single_tgt_lang)
        
        lang_group.setLayout(lang_layout)
        left_layout.addWidget(lang_group)

        
        # TTS设置
        tts_group = QGroupBox("TTS设置")
        tts_layout = QVBoxLayout()


        
        self.tts_engine = QComboBox()
        self.tts_engine.addItems(ui_manager.get_ui_datas_config("tts_engine"))
        tts_layout.addWidget(QLabel("TTS引擎:"))
        tts_layout.addWidget(self.tts_engine)
        
        # 添加TTS声音选择
        self.tts_voice = QComboBox()
        tts_layout.addWidget(QLabel("TTS声音:"))
        tts_layout.addWidget(self.tts_voice)
        
        # tts声音要根据引擎的选择而改变
        self.tts_engine.currentTextChanged.connect(self._request_tts_voices )

        
        # 添加TTS模型版本选择
        self.tts_model_version = QComboBox()
        self.tts_model_version.setEnabled(False)  # 初始禁用
        tts_layout.addWidget(QLabel("模型版本:"))
        tts_layout.addWidget(self.tts_model_version)
        
        # 当TTS引擎改变时更新模型版本列表
        self.tts_engine.currentTextChanged.connect(self.update_tts_model_versions)
        
        self.tts_use_cache = QCheckBox("启用音频缓存")
        self.tts_use_cache.setChecked(ui_manager.config.get_config('use_cache', True))
        tts_layout.addWidget(self.tts_use_cache)
        
        self.tts_max_retries = QSlider(Qt.Orientation.Horizontal)
        self.tts_max_retries.setMinimum(1)
        self.tts_max_retries.setMaximum(5)
        self.tts_max_retries.setValue(ui_manager.config.get_config('tts.max_retries', 3))
        self.tts_max_retries.setSingleStep(1)
        tts_layout.addWidget(QLabel("最大重试次数:"))
        tts_layout.addWidget(self.tts_max_retries)
        
        tts_group.setLayout(tts_layout)
        left_layout.addWidget(tts_group)
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout()
        
        # 按钮行
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始翻译")
        self.start_btn.clicked.connect(self.start_translation)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止翻译")
        self.stop_btn.clicked.connect(self.stop_translation)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        control_layout.addLayout(button_layout)
        
       
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        # 添加左侧面板到分割器（不再使用layout.addWidget与stretch factor）
        splitter.addWidget(left_panel)
        
        # 右侧翻译结果面板
        right_panel = QWidget()
        right_panel.setMinimumWidth(500)  # 设置最小宽度
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        
        # 翻译结果表格
        status_group = QGroupBox("状态信息")
         # 状态信息
        status_layout = QVBoxLayout()
        
        # 处理状态
        status_row = QHBoxLayout()
        status_label = QLabel("状态:")
        status_label.setFixedWidth(60)
        status_row.addWidget(status_label)
        self.status_text = QLabel("就绪")
        status_row.addWidget(self.status_text)
        status_layout.addLayout(status_row)
        
        # 处理时长
        duration_row = QHBoxLayout()
        duration_label = QLabel("时长:")
        duration_label.setFixedWidth(60)
        duration_row.addWidget(duration_label)
        self.duration_text = QLabel("00:00")
        duration_row.addWidget(self.duration_text)
        status_layout.addLayout(duration_row)
        
        # 系统资源
        resource_row = QHBoxLayout()
        resource_label = QLabel("资源:")
        resource_label.setFixedWidth(60)
        resource_row.addWidget(resource_label)
        self.resource_text = QLabel("CPU: 0% | GPU: 0%")
        resource_row.addWidget(self.resource_text)
        status_layout.addLayout(resource_row)
        
        # 延迟信息
        latency_row = QHBoxLayout()
        latency_label = QLabel("延迟:")
        latency_label.setFixedWidth(60)
        latency_row.addWidget(latency_label)
        self.latency_text = QLabel("识别: 0ms | 翻译: 0ms | TTS: 0ms")
        latency_row.addWidget(self.latency_text)
        status_layout.addLayout(latency_row)
        
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        
        # 实时监控
        monitor_group = QGroupBox("实时监控")
        monitor_layout = QVBoxLayout()
        
        self.current_source = QTextEdit()
        self.current_source.setReadOnly(True)
        monitor_layout.addWidget(QLabel("当前输入:"))
        monitor_layout.addWidget(self.current_source)
        
        self.current_translation = QTextEdit()
        self.current_translation.setReadOnly(True)
        monitor_layout.addWidget(QLabel("当前翻译:"))
        monitor_layout.addWidget(self.current_translation)
        
        monitor_group.setLayout(monitor_layout)
        right_layout.addWidget(monitor_group)
        
        # 添加右侧面板到分割器（不再使用layout.addWidget与stretch factor）
        splitter.addWidget(right_panel)
        
        # 设置初始大小比例为1:2
        splitter.setSizes([1, 2])
        
        # 将分割器添加到主布局
        layout.addWidget(splitter)

        # 在返回标签页之前触发模型版本加载
        current_asr_engine = self.realtime_asr_engine.currentText()
        if current_asr_engine:
            self.update_realtime_asr_model_versions()
        
        current_tts_engine = self.tts_engine.currentText()
        if current_tts_engine:
            self.update_tts_model_versions()


        # 配置触发更新配置
        # 检查语言选择器的变更事件
        self.single_src_lang.currentIndexChanged.connect(self._on_source_language_changed)
        self.single_tgt_lang.currentIndexChanged.connect(self._on_target_language_changed)

        # 检查ASR引擎选择器的变更事件
        self.realtime_asr_engine.currentIndexChanged.connect(self._on_asr_engine_changed)
        self.realtime_asr_model_version.currentIndexChanged.connect(self._on_asr_model_changed)

        # 检查TTS引擎选择器的变更事件
        self.tts_engine.currentIndexChanged.connect(self._on_tts_engine_changed)
        self.tts_model_version.currentIndexChanged.connect(self._on_tts_model_changed)

        # 检查TTS声音选择器的变更事件
        self.tts_voice.currentIndexChanged.connect(self._on_tts_voice_changed)
        
        return widget
########################################触发配置更新############################################################
    def _on_source_language_changed(self):
        language = self._extract_lang_code(self.single_src_lang.currentText())
        # 检查是否有这行代码
        ui_manager.config.update_config("language.source", language)

    def _on_target_language_changed(self):
        language = self._extract_lang_code(self.single_tgt_lang.currentText())
        # 检查是否有这行代码
        ui_manager.config.update_config("language.target", language)

    def _on_asr_engine_changed(self):
        engine = self.realtime_asr_engine.currentText()
        # 检查是否有这行代码
        ui_manager.config.update_config("asr.engine", engine)

    def _on_asr_model_changed(self):
        model = self.realtime_asr_model_version.currentText()
        # 检查是否有这行代码
        ui_manager.config.update_config("asr.model", model)

    def _on_tts_engine_changed(self):
        engine = self.tts_engine.currentText()
        # 检查是否有这行代码
        ui_manager.config.update_config("tts.engine", engine)

    def _on_tts_model_changed(self):
        model = self.tts_model_version.currentText()
        # 检查是否有这行代码
        ui_manager.config.update_config("tts.model", model)

    def _on_tts_voice_changed(self, index):
        voice = self._extract_voice_name(self.tts_voice.currentText())
        # 检查是否有这行代码
        ui_manager.config.update_config("tts.voice", voice)

    def _on_use_gpu_changed(self, state):
        use_gpu = state == Qt.CheckState.Checked
        # 检查是否有这行代码
        ui_manager.config.update_config("system.use_gpu", use_gpu)

    def _on_input_device_changed(self, index):
        device = self.input_device.currentText()
        # 检查是否有这行代码
        ui_manager.config.update_config("audio.input_device", device)

    def _on_output_device_changed(self, index):
        device = self.output_device.currentText()
        # 检查是否有这行代码
        ui_manager.config.update_config("audio.output_device", device)

####################################################################################################

    def create_single_tab(self):
        """创建单文件翻译标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        
        # 创建一个分割器，替换原来的QHBoxLayout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(5)  # 设置分割线宽度
        splitter.setChildrenCollapsible(False)  # 防止子控件被完全折叠
        
        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setMinimumWidth(300)  # 设置最小宽度
        left_panel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        
        # 音频输入
        audio_group = QGroupBox("音频输入")
        audio_layout = QVBoxLayout()
        
        audio_file_layout = QHBoxLayout()
        self.audio_input = QLineEdit()
        self.audio_input.setPlaceholderText("选择或输入音频文件路径")
        audio_file_layout.addWidget(self.audio_input)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.select_audio_file)
        audio_file_layout.addWidget(browse_btn)
        
        audio_layout.addLayout(audio_file_layout)
        
        audio_group.setLayout(audio_layout)
        left_layout.addWidget(audio_group)
        
        # 语言设置
        lang_group = QGroupBox("语言设置")
        lang_layout = QHBoxLayout()
        
        self.file_src_lang = QComboBox()
        self.file_src_lang.addItems(ui_manager.get_ui_datas_config("single_src_lang"))
        lang_layout.addWidget(QLabel("源语言:"))
        lang_layout.addWidget(self.file_src_lang)
        
        self.file_tgt_lang = QComboBox()
        self.file_tgt_lang.addItems(ui_manager.get_ui_datas_config("single_tgt_lang"))
        lang_layout.addWidget(QLabel("目标语言:"))
        lang_layout.addWidget(self.file_tgt_lang)
        
        lang_group.setLayout(lang_layout)
        left_layout.addWidget(lang_group)
        
        # 翻译按钮
        translate_btn = QPushButton("翻译")
        translate_btn.clicked.connect(self.translate_audio)
        left_layout.addWidget(translate_btn)
        
        # 添加一个弹性空间，让控件靠上对齐
        left_layout.addStretch(1)
        
        # 添加左侧面板到分割器
        splitter.addWidget(left_panel)
        
        # 右侧结果面板
        right_panel = QWidget()
        right_panel.setMinimumWidth(500)  # 设置最小宽度
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        
        # 翻译结果
        result_group = QGroupBox("翻译结果")
        result_layout = QVBoxLayout()
        
        self.translated_text = QTextEdit()
        self.translated_text.setReadOnly(True)
        result_layout.addWidget(self.translated_text)
        
        # 添加操作按钮行
        button_row = QHBoxLayout()
        
        copy_btn = QPushButton("复制结果")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.translated_text.toPlainText()))
        button_row.addWidget(copy_btn)
        
        save_btn = QPushButton("保存结果")
        save_btn.clicked.connect(self._save_translation_result)
        button_row.addWidget(save_btn)
        
        result_layout.addLayout(button_row)
        
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        # 添加右侧面板到分割器
        splitter.addWidget(right_panel)
        
        # 设置初始大小比例为1:2
        splitter.setSizes([1, 2])
        
        # 将分割器添加到主布局
        layout.addWidget(splitter)
        
        return widget

    def select_audio_file(self):
        """选择音频文件"""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.ogg)"
        )
        if file_path:
            self.audio_input.setText(file_path)

    def _save_translation_result(self):
        """保存翻译结果到文件"""
        if not self.translated_text.toPlainText():
            QMessageBox.warning(self, "警告", "没有可保存的翻译结果")
            return
        
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存翻译结果",
            "",
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.translated_text.toPlainText())
                QMessageBox.information(self, "成功", f"翻译结果已保存到 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def create_settings_tab(self):
        """创建设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)  # 设置边距
        
        # GPU设置
        gpu_group = QGroupBox("GPU设置")
        gpu_layout = QVBoxLayout()
        
        self.enable_gpu = QCheckBox("启用GPU加速")
        self.enable_gpu.setChecked(ui_manager.config.get_config('gpu.enable', True))
        gpu_layout.addWidget(self.enable_gpu)
        
        self.gpu_device = QComboBox()
        if self.available_gpus:
            self.gpu_device.addItems(self.available_gpus)
            self.gpu_device.setEnabled(True)
            self.enable_gpu.setEnabled(True)
        else:
            self.gpu_device.addItem("无可用GPU设备")
            self.gpu_device.setEnabled(False)
            self.enable_gpu.setEnabled(False)
        
        gpu_layout.addWidget(QLabel("GPU设备:"))
        gpu_layout.addWidget(self.gpu_device)
        
        self.half_precision = QCheckBox("使用半精度(FP16)加速")
        self.half_precision.setChecked(ui_manager.config.get_config('gpu.half_precision', True))
        if not self.available_gpus:
            self.half_precision.setEnabled(False)
        gpu_layout.addWidget(self.half_precision)
        
        # 启用状态变化回调
        self.enable_gpu.stateChanged.connect(self._update_gpu_settings)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # 音频设置
        audio_group = QGroupBox("音频设置")
        audio_layout = QVBoxLayout()
        
        # 采样率设置
        sample_rate_layout = QVBoxLayout()
        sample_rate_layout.addWidget(QLabel("采样率:"))
        
        self.sample_rate = QSlider(Qt.Orientation.Horizontal)
        self.sample_rate.setMinimum(8000)
        self.sample_rate.setMaximum(48000)
        self.sample_rate.setValue(ui_manager.config.get_config('audio.sample_rate', 16000))
        self.sample_rate.setSingleStep(1000)
        self.sample_rate.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sample_rate.setTickInterval(8000)
        
        sample_rate_value = QLabel(f"{self.sample_rate.value()} Hz")
        self.sample_rate.valueChanged.connect(lambda v: sample_rate_value.setText(f"{v} Hz"))
        
        sample_rate_layout.addWidget(self.sample_rate)
        sample_rate_layout.addWidget(sample_rate_value)
        audio_layout.addLayout(sample_rate_layout)
        
        # 音频块时长设置
        chunk_layout = QVBoxLayout()
        chunk_layout.addWidget(QLabel("音频块时长:"))
        
        self.chunk_duration = QSlider(Qt.Orientation.Horizontal)
        self.chunk_duration.setMinimum(1)
        self.chunk_duration.setMaximum(20)
        self.chunk_duration.setValue(int(ui_manager.config.get_config('audio.chunk_duration', 0.5) * 10))
        self.chunk_duration.setSingleStep(1)
        self.chunk_duration.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.chunk_duration.setTickInterval(5)
        
        chunk_value = QLabel(f"{self.chunk_duration.value() / 10:.1f} 秒")
        self.chunk_duration.valueChanged.connect(lambda v: chunk_value.setText(f"{v / 10:.1f} 秒"))
        
        chunk_layout.addWidget(self.chunk_duration)
        chunk_layout.addWidget(chunk_value)
        audio_layout.addLayout(chunk_layout)
        
        # 降噪设置
        self.noise_suppression = QCheckBox("启用降噪")
        self.noise_suppression.setChecked(ui_manager.config.get_config('audio.noise_suppression', True))
        audio_layout.addWidget(self.noise_suppression)
        
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)


        # 触发更新配置
        # GPU设置
        self.enable_gpu.stateChanged.connect(self._on_use_gpu_changed)

        # 音频设备设置
        self.input_device.currentIndexChanged.connect(self._on_input_device_changed)
        self.output_device.currentIndexChanged.connect(self._on_output_device_changed)

        
        # 配置文件设置
        profile_group = QGroupBox("配置文件")
        profile_layout = QVBoxLayout()
        
        # 配置名称
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("配置名称:"))
        self.profile_name = QLineEdit()
        self.profile_name.setText(ui_manager.config.get_config('profile.current', "默认配置"))
        name_layout.addWidget(self.profile_name)
        profile_layout.addLayout(name_layout)
        
        # 保存配置按钮
        save_profile_btn = QPushButton("保存配置")
        save_profile_btn.clicked.connect(self.save_profile)
        profile_layout.addWidget(save_profile_btn)
        
        profile_group.setLayout(profile_layout)
        layout.addWidget(profile_group)
        
        # 添加弹性空间
        layout.addStretch(1)
        
        return widget

    def save_profile(self):
        """保存配置"""
        try:
            name = self.profile_name.text()
            if not name:
                QMessageBox.warning(self, "警告", "请输入配置名称")
                return
            
            # 收集当前设置
            settings = {
                "audio": {
                    "sample_rate": self.sample_rate.value(),
                    "chunk_duration": self.chunk_duration.value() / 10.0,
                    "noise_suppression": self.noise_suppression.isChecked()
                },
                "gpu": {
                    "enable": self.enable_gpu.isChecked(),
                    "device": self.gpu_device.currentText() if self.enable_gpu.isChecked() else "",
                    "half_precision": self.half_precision.isChecked()
                },
                "language": {
                    "source": self._extract_lang_code(self.single_src_lang.currentText()),
                    "target": self._extract_lang_code(self.single_tgt_lang.currentText())
                },
                "asr": {
                    "engine": self.realtime_asr_engine.currentText(),
                    "model": self.realtime_asr_model_version.currentText()
                },
                "tts": {
                    "engine": self.tts_engine.currentText(),
                    "model": self.tts_model_version.currentText(),
                    "voice": self._extract_voice_name(self.tts_voice.currentText()),
                    "use_cache": self.tts_use_cache.isChecked(),
                    "max_retries": self.tts_max_retries.value()
                },
                "profile": {
                    "current": name
                }
            }
            
            # 更新配置
            for section, values in settings.items():
                self.ui_manager.config.get_config.update_config_dict(values, section)
            
            # 保存配置
            self.ui_manager.save_config()
            
            QMessageBox.information(self, "成功", f"配置 '{name}' 已保存")
            
        except Exception as e:
            logger.exception(f"保存配置失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")

    def _update_gpu_settings(self):
        """更新GPU设置状态"""
        enabled = self.enable_gpu.isChecked() and self.available_gpus
        self.gpu_device.setEnabled(enabled)
        self.half_precision.setEnabled(enabled)

    def create_model_tab(self):
        """创建模型管理标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        
        # 创建表格
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(6)  # 6列：模型类型、模型、状态、下载、删除、刷新
        self.model_table.setHorizontalHeaderLabels(["模型类型", "模型", "状态", "下载", "删除", "刷新"])
        
        # 设置表格样式
        self.model_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # 设置列宽比例
        header = self.model_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)  # 模型类型列 - 15%
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # 模型列 - 30%
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # 状态列 - 15%
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)  # 下载列 - 13%
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)  # 删除列 - 13%
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)  # 刷新列 - 14%
        
        # 设置初始列宽比例
        total_width = self.model_table.width()
        self.model_table.setColumnWidth(0, int(total_width * 0.15))  # 模型类型 15%
        self.model_table.setColumnWidth(1, int(total_width * 0.30))  # 模型 30%
        self.model_table.setColumnWidth(2, int(total_width * 0.15))  # 状态 15%
        self.model_table.setColumnWidth(3, int(total_width * 0.13))  # 下载 13%
        self.model_table.setColumnWidth(4, int(total_width * 0.13))  # 删除 13%
        self.model_table.setColumnWidth(5, int(total_width * 0.14))  # 刷新 14%
        
        # 添加模型数据
        models = ui_manager.get_ui_datas_config("models")
        
        self.model_table.setRowCount(len(models))
        
        for row, (model_type, model_name) in enumerate(models):
            # 模型类型
            self.model_table.setItem(row, 0, QTableWidgetItem(model_type))
            
            # 模型名称
            self.model_table.setItem(row, 1, QTableWidgetItem(model_name))
            
            # 状态（将在refresh_model_status中更新）
            self.model_table.setItem(row, 2, QTableWidgetItem("未知"))
            
            # 下载按钮
            download_btn = QPushButton("下载")
            download_btn.clicked.connect(lambda checked, r=row: self._download_model_from_table(r))
            self.model_table.setCellWidget(row, 3, download_btn)
            
            # 删除按钮
            delete_btn = QPushButton("删除")
            delete_btn.clicked.connect(lambda checked, r=row: self._delete_model_from_table(r))
            self.model_table.setCellWidget(row, 4, delete_btn)
            
            # 刷新按钮
            refresh_btn = QPushButton("刷新")
            refresh_btn.clicked.connect(lambda checked, r=row: self._refresh_model_from_table(r))
            self.model_table.setCellWidget(row, 5, refresh_btn)
        
        layout.addWidget(self.model_table)
        
        # 全局操作按钮
        buttons_layout = QHBoxLayout()
        
        refresh_all_btn = QPushButton("刷新所有")
        refresh_all_btn.clicked.connect(self.refresh_model_status)
        buttons_layout.addWidget(refresh_all_btn)
        
        layout.addLayout(buttons_layout)

        # 进度条
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("下载进度:"))
        
        self.download_progress = QProgressBar()
        progress_layout.addWidget(self.download_progress)
        
        layout.addLayout(progress_layout)
        
        return widget

    def _download_model_from_table(self, row):
        """从表格中下载模型"""
        try:
            engine_name, model_tag, model_type,model_name = self._parese_table_model_info(row)
            
            # 设置进度条的初始状态
            self.download_progress.setVisible(True)
            self.download_progress.setValue(0)
            
            # 保存当前下载的模型信息用于更新进度
            self.current_download = {
                'model_type': model_type,
                'engine_name': engine_name,
                'model_tag': model_tag
            }
            
            # 启动进度更新定时器
            self.progress_timer.start()
            
            # 使用UI事件系统触发模型下载
            self.ui_manager.trigger_event(
                UIEventType.MODEL_DOWNLOAD,
                {
                    "engine_type": model_type,
                    "engine_name": engine_name,
                    "model_name": model_tag,
                    "model_display_name": model_name
                }
            )
            
        except Exception as e:
            if hasattr(self, 'progress_timer'):
                self.progress_timer.stop()
            self.download_progress.setVisible(False)
            logger.exception(f"下载模型失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"下载模型失败: {str(e)}")
            # 清除当前下载信息
            self.current_download = None

    def _parese_table_model_info(self,row):
        model_type = self.model_table.item(row, 0).text().lower()
        model_name = self.model_table.item(row, 1).text()

        # 解析模型信息
        engine_name = None
        model_tag = None

        if model_name.startswith("whisper-"):
            engine_name = "whisper"
            model_tag = model_name.split("-")[1]
        elif model_name.startswith("faster-whisper-"):
            engine_name = "faster_whisper"
            model_tag = model_name.split("-")[2]
        elif model_name.startswith("vosk-"):
            engine_name = "vosk"
            model_tag = model_name.split("-")[1]
        elif model_name.startswith("edge-tts-"):
            engine_name = "edge_tts"
            model_tag = model_name.split("-")[2]
        elif model_name.startswith("xtts-"):
            engine_name = "xtts"
            model_tag = model_name.split("-")[1]
        elif model_name.startswith("yourtts-"):
            engine_name = "yourtts"
            model_tag = model_name.split("-")[1]

        elif model_name.startswith("f5_tts-"):
            engine_name = "f5_tts"
            model_tag = model_name.split("-")[1]

        elif model_name.startswith("bark-"):
            engine_name = "bark"
            model_tag = model_name.split("-")[1]
        elif model_name.startswith("nllb-"):
            engine_name = "nllb"
            model_tag = model_name.split("-")[1]
            model_type = "translation"

        return engine_name, model_tag, model_type,model_name

    def _delete_model_from_table(self, row):
        """从表格中删除模型"""
        try:
            engine_name,  model_tag, model_type,model_name = self._parese_table_model_info(row)
            
            # 确认删除
            reply = QMessageBox.question(self, "确认删除", 
                                        f"确定要删除{model_name}模型吗？",
                                        QMessageBox.StandardButton.Yes | 
                                        QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                # 使用UI事件系统触发模型删除
                self.ui_manager.trigger_event(
                    UIEventType.MODEL_DELETE,
                    {
                        "engine_type": model_type,
                        "engine_name": engine_name,
                        "model_name": model_tag,
                        "model_display_name": model_name
                    }
                )
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"删除模型失败: {str(e)}")

    def _refresh_model_from_table(self, row):
        """刷新表格中的模型状态"""
        try:
            engine_name,  model_tag, model_type,model_name = self._parese_table_model_info(row)
            
            # 检查模型状态
            if all([model_type, engine_name, model_tag]):
                logger.info(f"检查模型状态: 类型={model_type}, 引擎={engine_name}, 标签={model_tag}")
                
                # 导入模型管理器
                from utils.model_manager import model_manager
                
                if model_manager.is_model_downloaded(model_type, engine_name, model_tag):
                    status_item = QTableWidgetItem("已下载")
                    status_item.setBackground(Qt.GlobalColor.green)
                else:
                    status_item = QTableWidgetItem("未下载")
                    status_item.setBackground(Qt.GlobalColor.red)
            else:
                logger.exception(f"无效的模型配置: 类型={model_type}, 名称={model_name}")
                status_item = QTableWidgetItem("配置错误")
                status_item.setBackground(Qt.GlobalColor.yellow)
            
            self.model_table.setItem(row, 2, status_item)
            
        except Exception as e:
            logger.exception(f"刷新模型状态失败: {str(e)}")
            status_item = QTableWidgetItem("错误")
            status_item.setBackground(Qt.GlobalColor.yellow)
            self.model_table.setItem(row, 2, status_item)

    def refresh_model_status(self):
        """刷新所有模型状态"""
        try:
            for row in range(self.model_table.rowCount()):
                self._refresh_model_from_table(row)
        except Exception as e:
            logger.exception(f"刷新模型状态失败: {str(e)}")

    def _handle_tts_voices_loaded(self, data: Dict[str, Any]):
        """处理TTS声音列表加载完成事件"""
        try:
            engine_name = data.get("engine_name")
            voices = data.get("voices", [])
            callback_id = data.get("callback_id")

            if callback_id == UICallbackID.TTS_VOICES:
                # 更新声音列表
                self.tts_voice.clear()

                if voices:
                    self.tts_voice.addItems(voices)
                    self.tts_voice.setEnabled(True)
                else:
                    self.tts_voice.setEnabled(False)
                    self.tts_voice.addItem("无可用声音")
        except Exception as e:
            logger.exception(f"处理TTS声音列表加载事件失败: {str(e)}")
            self.tts_voice.clear()
            self.tts_voice.addItem("default")  # 添加默认选项

    def _request_tts_voices (self):
        """更新TTS声音列表"""
        try:
            # 获取当前选择的引擎
            engine = self.tts_engine.currentText().strip()
            logger.info(f"请求TTS声音列表: 引擎={engine}")

            # 触发获取声音列表事件
            self.ui_manager.trigger_event(
                UIEventType.GET_TTS_VOICES,
                {
                    "engine_name": engine,
                    "callback_id": UICallbackID.TTS_VOICES
                }
            )
        except Exception as e:
            logger.exception(f"请求TTS声音列表失败: {str(e)}")
            self.tts_voice.clear()
            self.tts_voice.addItem("default")  # 添加默认选项
        
    def init_gpu_settings(self):
        """初始化GPU设置"""
        try:
            import torch
            if torch.cuda.is_available():
                # 获取可用的GPU设备
                gpu_count = torch.cuda.device_count()
                gpu_names = [f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(gpu_count)]
                self.available_gpus = gpu_names
                logger.info(f"检测到{gpu_count}个GPU设备: {gpu_names}")
            else:
                self.available_gpus = []
                logger.warning("未检测到可用的GPU设备")
        except Exception as e:
            logger.exception(f"GPU设置初始化失败: {str(e)}")
            self.available_gpus = []

    def update_audio_devices(self):
        """更新音频设备列表，支持虚拟设备过滤"""
        try:
            import sounddevice as sd

            # 获取所有音频设备
            devices = sd.query_devices()

            # 清空当前列表
            self.input_device.clear()
            self.output_device.clear()

            # 过滤和去重用
            input_seen = set()
            output_seen = set()
            # 常见虚拟设备关键词
            virtual_keywords = ["Mapper", "Loopback", "虚拟", "Stereo Mix", "VoiceMeeter", "VB-Audio", "Bluetooth", "Find My"]

            # 是否显示虚拟设备
            show_virtual = True

            # 输入设备
            for i, device in enumerate(devices):
                name = device['name']
                if device['max_input_channels'] > 0:
                    is_virtual = any(keyword.lower() in name.lower() for keyword in virtual_keywords)
                    # 过滤虚拟设备
                    if is_virtual and not show_virtual:
                        continue
                    # 去重
                    key = (name, device['max_input_channels'])
                    if key in input_seen:
                        continue
                    input_seen.add(key)
                    display_name = f"{name} (输入, 索引:{i})"
                    if is_virtual:
                        display_name += " [虚拟]"
                    self.input_device.addItem(display_name, i)

            # 输出设备
            for i, device in enumerate(devices):
                name = device['name']
                if device['max_output_channels'] > 0:
                    is_virtual = any(keyword.lower() in name.lower() for keyword in virtual_keywords)
                    if is_virtual and not show_virtual:
                        continue
                    key = (name, device['max_output_channels'])
                    if key in output_seen:
                        continue
                    output_seen.add(key)
                    display_name = f"{name} (输出, 索引:{i})"
                    if is_virtual:
                        display_name += " [虚拟]"
                    self.output_device.addItem(display_name, i)

            logger.info(f"已更新音频设备列表: {len(input_seen)}个输入, {len(output_seen)}个输出 (虚拟设备{'已显示' if show_virtual else '已隐藏'})")

        except Exception as e:
            logger.exception(f"更新音频设备列表失败: {str(e)}")

def create_ui():
    """创建UI界面"""
    try:
        logger.info("开始创建UI...")
        
        # 获取或创建QApplication实例
        logger.info("获取或创建QApplication实例...")
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            logger.info("创建新的QApplication实例")
        else:
            logger.info("使用现有的QApplication实例")
        
        # 设置Windows特定的高DPI支持
        if sys.platform == "win32":
            logger.info("设置Windows高DPI支持...")
            os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
            os.environ["QT_SCALE_FACTOR"] = "1"
            logger.info("Windows高DPI支持设置完成")
        
        # 创建主窗口
        logger.info("创建主窗口...")
        window = MainWindow()
        logger.info("主窗口创建成功")
        
        # 设置窗口位置
        logger.info("设置窗口位置...")
        screen = app.primaryScreen().geometry()
        logger.info(f"屏幕尺寸: {screen.width()}x{screen.height()}")
        
        # 计算窗口位置使其居中
        window_width = 1200
        window_height = 800
        x = (screen.width() - window_width) // 2
        y = (screen.height() - window_height) // 2
        
        # 设置窗口大小和位置
        window.setGeometry(x, y, window_width, window_height)
        logger.info(f"窗口位置设置为: x={x}, y={y}, width={window_width}, height={window_height}")
        
        # 确保窗口可见
        logger.info("显示主窗口...")
        window.show()
        window.raise_()  # 确保窗口在最前面
        window.activateWindow()  # 激活窗口
        window.setFocus()  # 设置焦点
        

        # 处理待处理的事件
        logger.info("处理待处理的事件...")
        exit_code = app.exec()
        
        logger.info("UI创建完成")
        return exit_code
        
    except Exception as e:
        logger.exception(f"UI创建失败: {str(e)}")
        logger.exception("详细错误信息:", exc_info=True)
        raise
