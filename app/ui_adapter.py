#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import os
from typing import Dict, Any, Optional, List, Callable
from utils.logger import logger
from app.task_manager import (
    TaskManager, TranslationProcessTask, StatusMonitorTask, 
    TaskStatus, TaskCommand, ModelDownloadTask
)
from utils.model_manager import ModelManager, model_manager
from .ui_interface import UIManager, UIEventType, UIEvent, ui_manager
from utils.config import config

class UITaskAdapter:
    """UI与任务管理器的适配器，负责将UI事件转换为任务操作"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(UITaskAdapter, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.task_manager = TaskManager()
            self.active_translation_task_id = None
            self.active_status_task_id = None
            self.task_status_observers = {}
            
            # 初始化任务管理器
            self.task_manager.start()
            
            # 注册UI事件处理器
            self._register_event_handlers()
    
    def _register_event_handlers(self):
        """注册UI事件处理器"""
        ui_manager.register_event_handler(
            UIEventType.START_TRANSLATION, 
            self._handle_start_translation
        )
        
        ui_manager.register_event_handler(
            UIEventType.STOP_TRANSLATION, 
            self._handle_stop_translation
        )
        
        ui_manager.register_event_handler(
            UIEventType.PAUSE_TRANSLATION, 
            self._handle_pause_translation
        )
        
        ui_manager.register_event_handler(
            UIEventType.RESUME_TRANSLATION, 
            self._handle_resume_translation
        )
        
        ui_manager.register_event_handler(
            UIEventType.MODEL_DOWNLOAD, 
            self._handle_model_download
        )
        
        ui_manager.register_event_handler(
            UIEventType.MODEL_DELETE, 
            self._handle_model_delete
        )
        
        ui_manager.register_event_handler(
            UIEventType.GET_MODEL_VERSIONS,
            self._handle_get_model_versions
        )
        
        ui_manager.register_event_handler(
            UIEventType.SET_MODEL,
            self._handle_set_model
        )
        
        ui_manager.register_event_handler(
            UIEventType.TRANSLATE_FILE,
            self._handle_translate_file
        )
        
        ui_manager.register_event_handler(
            UIEventType.SYSTEM_EXIT, 
            self._handle_system_exit
        )
        
        ui_manager.register_event_handler(
            UIEventType.UPDATE_STATUS,
            self._handle_update_status_request
        )
    
    def _handle_start_translation(self, data: Dict[str, Any]):
        """处理开始翻译事件"""
        try:
            # 获取翻译配置
            source_language = data.get("source_language", ui_manager.config.get_config("language.source", "zh"))
            target_language = data.get("target_language", ui_manager.config.get_config("language.target", "en"))
            input_device = data.get("input_device", "default")
            output_device = data.get("output_device", "default")
            
            # 创建翻译任务
            translation_task = TranslationProcessTask(
                source_language=source_language,
                target_language=target_language,
                input_device=input_device,
                output_device=output_device
            )
            
            # 添加状态更新回调
            translation_task.on_progress(self._on_translation_progress)
            translation_task.on_status_change(self._on_translation_status_change)
            translation_task.on_complete(self._on_translation_complete)
            translation_task.on_error(self._on_translation_error)
            
            # 启动翻译任务
            task_id = self.task_manager.submit_task(translation_task)
            self.active_translation_task_id = task_id
            
            # 启动状态监控任务
            self._start_status_monitor()
            
            logger.info(f"已启动翻译任务 (ID: {task_id})")
            
            # 通知UI翻译已开始
            ui_manager.trigger_event(
                UIEventType.UPDATE_STATUS,
                {"is_running": True, "task_id": task_id}
            )
            
            return task_id
        except Exception as e:
            logger.exception(f"启动翻译任务失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"启动翻译失败: {str(e)}"}
            )
            return None
    
    def _handle_stop_translation(self, data: Dict[str, Any]):
        """处理停止翻译事件"""
        try:
            if self.active_translation_task_id:
                # 发送停止命令
                self.task_manager.send_command(
                    self.active_translation_task_id,
                    TaskCommand.STOP
                )
                
                # 停止状态监控
                self._stop_status_monitor()
                
                logger.info(f"已停止翻译任务 (ID: {self.active_translation_task_id})")
                
                # 通知UI翻译已停止
                ui_manager.trigger_event(
                    UIEventType.UPDATE_STATUS,
                    {"is_running": False, "task_id": self.active_translation_task_id}
                )
                
                # 清除活动任务ID
                self.active_translation_task_id = None
                
                return True
            return False
        except Exception as e:
            logger.exception(f"停止翻译任务失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"停止翻译失败: {str(e)}"}
            )
            return False
    
    def _handle_pause_translation(self, data: Dict[str, Any]):
        """处理暂停翻译事件"""
        try:
            if self.active_translation_task_id:
                # 发送暂停命令
                self.task_manager.send_command(
                    self.active_translation_task_id,
                    TaskCommand.PAUSE
                )
                
                logger.info(f"已暂停翻译任务 (ID: {self.active_translation_task_id})")
                
                # 通知UI翻译已暂停
                ui_manager.trigger_event(
                    UIEventType.UPDATE_STATUS,
                    {"is_paused": True, "task_id": self.active_translation_task_id}
                )
                
                return True
            return False
        except Exception as e:
            logger.exception(f"暂停翻译任务失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"暂停翻译失败: {str(e)}"}
            )
            return False
    
    def _handle_resume_translation(self, data: Dict[str, Any]):
        """处理恢复翻译事件"""
        try:
            if self.active_translation_task_id:
                # 发送恢复命令
                self.task_manager.send_command(
                    self.active_translation_task_id,
                    TaskCommand.RESUME
                )
                
                logger.info(f"已恢复翻译任务 (ID: {self.active_translation_task_id})")
                
                # 通知UI翻译已恢复
                ui_manager.trigger_event(
                    UIEventType.UPDATE_STATUS,
                    {"is_paused": False, "task_id": self.active_translation_task_id}
                )
                
                return True
            return False
        except Exception as e:
            logger.exception(f"恢复翻译任务失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"恢复翻译失败: {str(e)}"}
            )
            return False
    
    def _handle_model_download(self, data: Dict[str, Any]):
        """处理模型下载事件"""
        try:
            engine_type = data.get("engine_type")
            engine_name = data.get("engine_name")
            model_name = data.get("model_name")
            
            if not all([engine_type, engine_name, model_name]):
                raise ValueError("缺少必要的模型信息")
            
            # 创建模型下载任务
            download_task = ModelDownloadTask(
                engine_type=engine_type,
                engine_name=engine_name,
                model_name=model_name
            )
            
            # 检查模型可用性
            availability = download_task.check_availability()
            if not availability.get("available", False):
                ui_manager.trigger_event(
                    UIEventType.ERROR,
                    {"message": availability.get("message", "模型不可下载")}
                )
                return None
            
            # 添加进度更新回调
            download_task.on_progress(lambda progress: ui_manager.trigger_event(
                UIEventType.UPDATE_PROGRESS,
                {
                    "progress": progress,
                    "type": "model_download",
                    "engine_type": engine_type,
                    "engine_name": engine_name,
                    "model_name": model_name
                }
            ))
            
            # 添加完成回调
            download_task.on_complete(lambda result: ui_manager.trigger_event(
                UIEventType.MODEL_LOADED,
                {
                    "success": True,
                    "engine_type": engine_type,
                    "engine_name": engine_name,
                    "model_name": model_name
                }
            ))
            
            # 添加错误回调
            download_task.on_error(lambda error: ui_manager.trigger_event(
                UIEventType.ERROR,
                {
                    "message": f"模型下载失败: {error}",
                    "engine_type": engine_type,
                    "engine_name": engine_name,
                    "model_name": model_name
                }
            ))
            
            # 提交任务
            task_id = self.task_manager.submit_task(download_task)
            
            logger.info(f"已启动模型下载任务 (ID: {task_id})")
            
            return task_id
        except Exception as e:
            logger.exception(f"启动模型下载任务失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"启动模型下载失败: {str(e)}"}
            )
            return None
    
    def _handle_model_delete(self, data: Dict[str, Any]):
        """处理模型删除事件"""
        try:
            engine_type = data.get("engine_type")
            engine_name = data.get("engine_name")
            model_name = data.get("model_name")
            
            if not all([engine_type, engine_name, model_name]):
                raise ValueError("缺少必要的模型信息")
            
            # 使用模型管理器删除模型
            from utils.model_manager import model_manager
            success = model_manager.delete_model(engine_type, engine_name, model_name)
            
            if success:
                logger.info(f"已删除模型: {engine_type}/{engine_name}/{model_name}")
                
                # 通知UI更新
                ui_manager.trigger_event(
                    UIEventType.MODEL_LOADED,  # 复用该事件类型，表示模型状态变更
                    {
                        "success": True,
                        "deleted": True,
                        "engine_type": engine_type,
                        "engine_name": engine_name,
                        "model_name": model_name
                    }
                )
                
                return True
            else:
                ui_manager.trigger_event(
                    UIEventType.ERROR,
                    {"message": f"删除模型失败: {engine_type}/{engine_name}/{model_name}"}
                )
                return False
        except Exception as e:
            logger.exception(f"删除模型任务失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"删除模型失败: {str(e)}"}
            )
            return False
    
    def _handle_system_exit(self, data: Dict[str, Any]):
        """处理系统退出事件"""
        try:
            # 停止所有任务
            if self.active_translation_task_id:
                self._handle_stop_translation({})
            
            # 停止状态监控
            self._stop_status_monitor()
            
            # 停止任务管理器
            self.task_manager.stop()
            
            # 保存配置
            ui_manager.save_config()
            
            logger.info("系统正常退出")
            
            return True
        except Exception as e:
            logger.exception(f"系统退出处理失败: {str(e)}")
            return False
    
    def _start_status_monitor(self):
        """启动状态监控任务"""
        try:
            # 如果已有状态监控任务，先停止
            self._stop_status_monitor()
            
            # 创建状态监控任务
            status_task = StatusMonitorTask()
            
            # 添加状态更新回调
            status_task.on_progress(self._on_status_progress)
            
            # 提交任务
            task_id = self.task_manager.submit_task(status_task)
            self.active_status_task_id = task_id
            
            logger.info(f"已启动状态监控任务 (ID: {task_id})")
            
            return task_id
        except Exception as e:
            logger.exception(f"启动状态监控任务失败: {str(e)}")
            return None
    
    def _stop_status_monitor(self):
        """停止状态监控任务"""
        try:
            if self.active_status_task_id:
                # 发送停止命令
                self.task_manager.send_command(
                    self.active_status_task_id,
                    TaskCommand.STOP
                )
                
                logger.info(f"已停止状态监控任务 (ID: {self.active_status_task_id})")
                
                # 清除活动任务ID
                self.active_status_task_id = None
                
                return True
            return False
        except Exception as e:
            logger.exception(f"停止状态监控任务失败: {str(e)}")
            return False
    
    def _on_translation_progress(self, progress: float):
        """翻译任务进度回调"""
        ui_manager.trigger_event(
            UIEventType.UPDATE_PROGRESS,
            {"progress": progress, "type": "translation"}
        )
    
    def _on_translation_status_change(self, status: TaskStatus):
        """翻译任务状态变更回调"""
        ui_manager.trigger_event(
            UIEventType.UPDATE_STATUS,
            {"status": status.value, "type": "translation"}
        )
    
    def _on_translation_complete(self, result: Any):
        """翻译任务完成回调"""
        ui_manager.trigger_event(
            UIEventType.UPDATE_STATUS,
            {"status": "completed", "type": "translation", "result": result}
        )
        
        # 清除活动任务ID
        self.active_translation_task_id = None
        
        # 停止状态监控
        self._stop_status_monitor()
    
    def _on_translation_error(self, error: str):
        """翻译任务错误回调"""
        ui_manager.trigger_event(
            UIEventType.ERROR,
            {"message": f"翻译任务失败: {error}"}
        )
        
        # 清除活动任务ID
        self.active_translation_task_id = None
        
        # 停止状态监控
        self._stop_status_monitor()
    
    def _on_status_progress(self, progress: float):
        """状态监控任务进度回调"""
        # 获取任务对象
        if self.active_status_task_id:
            task = self.task_manager.get_task(self.active_status_task_id)
            if task and hasattr(task, 'result_queue') and not task.result_queue.empty():
                try:
                    result = task.result_queue.get(block=False)
                    # 如果有状态数据，通知UI
                    if "status_data" in result:
                        ui_manager.trigger_event(
                            UIEventType.UPDATE_STATUS,
                            result["status_data"]
                        )
                except:
                    pass
    
    def _handle_get_model_versions(self, data: Dict[str, Any]):
        """处理获取模型版本事件"""
        try:
            engine_type = data.get("engine_type")
            engine_name = data.get("engine_name")
            callback_id = data.get("callback_id")
            
            if not all([engine_type, engine_name]):
                raise ValueError("缺少必要的引擎信息")

            # 获取可用的模型版本
            available_models = []
            current_model = ""

            model_info = model_manager.get_model_info(engine_type,engine_name)
            if model_info:
                available_models = [k for k,v in model_info.items()]
                current_model = available_models[0]


            # 触发返回事件
            ui_manager.trigger_event(
                UIEventType.MODEL_VERSIONS_LOADED,
                {
                    "engine_type": engine_type,
                    "engine_name": engine_name,
                    "versions": available_models,
                    "current_model": current_model,
                    "callback_id": callback_id
                }
            )
            
            logger.info(f"获取模型版本列表: {engine_type}/{engine_name} - 找到{len(available_models)}个版本")
            
            return True
        except Exception as e:
            logger.exception(f"获取模型版本列表失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"获取模型版本失败: {str(e)}"}
            )
            
            # 仍然触发返回事件，但返回空列表
            ui_manager.trigger_event(
                UIEventType.MODEL_VERSIONS_LOADED,
                {
                    "engine_type": data.get("engine_type", ""),
                    "engine_name": data.get("engine_name", ""),
                    "versions": [],
                    "current_model": "",
                    "callback_id": data.get("callback_id", "")
                }
            )
            
            return False
    
    def _handle_set_model(self, data: Dict[str, Any]):
        """处理设置模型事件"""
        try:
            engine_type = data.get("engine_type")
            engine_name = data.get("engine_name")
            model_version = data.get("model_version")
            callback_id = data.get("callback_id")
            
            if not all([engine_type, engine_name, model_version]):
                raise ValueError("缺少必要的模型信息")
            

            # 设置模型
            success = False
            error_msg = ""

            # todo save to config

            
            # 触发返回事件
            ui_manager.trigger_event(
                UIEventType.MODEL_SET_RESULT,
                {
                    "engine_type": engine_type,
                    "engine_name": engine_name,
                    "model_version": model_version,
                    "success": success,
                    "error": error_msg,
                    "callback_id": callback_id
                }
            )
            
            if success:
                logger.info(f"设置模型成功: {engine_type}/{engine_name}/{model_version}")
            else:
                logger.error(f"设置模型失败: {engine_type}/{engine_name}/{model_version} - {error_msg}")
            
            return success
        except Exception as e:
            logger.exception(f"设置模型失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"设置模型失败: {str(e)}"}
            )
            
            # 触发返回事件，表示失败
            ui_manager.trigger_event(
                UIEventType.MODEL_SET_RESULT,
                {
                    "engine_type": data.get("engine_type", ""),
                    "engine_name": data.get("engine_name", ""),
                    "model_version": data.get("model_version", ""),
                    "success": False,
                    "error": str(e),
                    "callback_id": data.get("callback_id", "")
                }
            )
            
            return False
    
    def _handle_translate_file(self, data: Dict[str, Any]):
        """处理翻译音频文件事件"""
        try:
            audio_path = data.get("audio_path")
            source_language = data.get("source_language")
            target_language = data.get("target_language")
            callback_id = data.get("callback_id")
            
            if not all([audio_path, source_language, target_language]):
                raise ValueError("缺少必要的翻译参数")
                
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")
            
            # 获取StreamProcessor实例
            from core.stream_processor import StreamProcessor
            processor = StreamProcessor()
            
            # 设置语言
            processor.set_languages(source_language, target_language)
            
            # 执行翻译
            translated_text, audio_output = processor.translate_audio(audio_path)
            
            # 触发翻译结果事件
            ui_manager.trigger_event(
                UIEventType.TRANSLATION_RESULT,
                {
                    "success": True,
                    "translated_text": translated_text,
                    "audio_output": audio_output.tolist() if hasattr(audio_output, 'tolist') else None,
                    "callback_id": callback_id
                }
            )
            
            logger.info(f"音频文件翻译成功: {audio_path}")
            return True
            
        except Exception as e:
            logger.exception(f"翻译音频文件失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {
                    "message": f"翻译音频文件失败: {str(e)}",
                    "callback_id": data.get("callback_id")
                }
            )
            return False

    def _handle_update_status_request(self, data: Dict[str, Any]):
        """处理更新状态请求事件"""
        try:
            # 获取当前状态
            current_status = self._get_current_status()
            
            # 触发返回事件
            ui_manager.trigger_event(
                UIEventType.UPDATE_STATUS,
                current_status
            )
            
            logger.info("状态更新请求处理成功")
            
            return True
        except Exception as e:
            logger.exception(f"更新状态请求处理失败: {str(e)}")
            ui_manager.trigger_event(
                UIEventType.ERROR,
                {"message": f"更新状态请求处理失败: {str(e)}"}
            )
            return False

    def _get_current_status(self):
        """获取当前状态"""
        status = {
            "is_running": False,
            "is_paused": False,
            "cpu_usage": 0,
            "gpu_usage": 0,
            "asr_latency": 0,
            "mt_latency": 0,
            "tts_latency": 0,
            "latest_text": {},
            "error": None
        }
        
        # 从活动任务中获取状态
        if self.active_translation_task_id:
            task = self.task_manager.get_task(self.active_translation_task_id)
            if task:
                # 任务状态
                status["is_running"] = task.status in [TaskStatus.RUNNING, TaskStatus.PENDING]
                status["is_paused"] = task.status == TaskStatus.PAUSED
                
                # 获取任务内部状态数据
                if hasattr(task, 'result_queue') and not task.result_queue.empty():
                    try:
                        result = task.result_queue.get(block=False)
                        if "status_data" in result:
                            # 合并状态数据
                            status.update(result["status_data"])
                    except Exception as e:
                        logger.exception(f"从任务获取状态数据失败: {str(e)}")
        
        # 从状态监控任务获取系统状态
        if self.active_status_task_id:
            task = self.task_manager.get_task(self.active_status_task_id)
            if task and hasattr(task, 'result_queue') and not task.result_queue.empty():
                try:
                    result = task.result_queue.get(block=False)
                    if "status_data" in result:
                        # 合并CPU和GPU使用率等系统状态
                        system_data = result["status_data"]
                        for key in ["cpu_usage", "gpu_usage", "memory_usage", "gpu_memory_usage"]:
                            if key in system_data:
                                status[key] = system_data[key]
                except Exception as e:
                    logger.exception(f"从状态监控任务获取状态数据失败: {str(e)}")
        
        return status

# 全局UI适配器实例
ui_adapter = UITaskAdapter() 