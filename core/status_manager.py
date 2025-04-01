#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time
from typing import Dict, Any, List, Callable
from utils.logger import logger
from app.task_manager import TaskManager, StatusMonitorTask
from core.status_event import StatusEvent, StatusEventType

class StatusManager:
    """状态管理器"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StatusManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.status = {
                'cpu_usage': 0,
                'gpu_usage': 0,
                'asr_latency': 0,
                'mt_latency': 0,
                'tts_latency': 0,
                'error': None,
                'is_running': False,
                'is_processing': False,
                'error_count': 0,
                'start_time': None,
                'latest_text': None,
                'running_duration': 0
            }
            self._lock = threading.RLock()
            self._monitor_task = None
            self._observers: List[Callable[[StatusEvent], None]] = []
    
    def start(self):
        """启动状态监控"""
        if self._monitor_task is None:
            task_manager = TaskManager()
            self._monitor_task = StatusMonitorTask()
            task_manager.submit_task(self._monitor_task)
            
            # 添加状态更新回调
            self._monitor_task.on_progress(self._update_status)
            
            # 设置初始状态
            self.status['start_time'] = time.time()
            self.status['is_running'] = True
    
    def stop(self):
        """停止状态监控"""
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        
        with self._lock:
            self.status['is_running'] = False
            self.status['is_processing'] = False
    
    def add_observer(self, observer: Callable[[StatusEvent], None]):
        """添加状态观察者"""
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)
    
    def remove_observer(self, observer: Callable[[StatusEvent], None]):
        """移除状态观察者"""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)
    
    def _notify_observers(self, event: StatusEvent):
        """通知所有观察者"""
        with self._lock:
            for observer in self._observers:
                try:
                    observer(event)
                except Exception as e:
                    logger.error(f"状态通知失败: {str(e)}")
    
    def _update_status(self, progress):
        """处理状态更新"""
        if not self._monitor_task:
            return
            
        task = self._monitor_task
        if hasattr(task, 'result_queue') and not task.result_queue.empty():
            try:
                result = task.result_queue.get(block=False)
                if "status_data" in result:
                    with self._lock:
                        self.status.update(result["status_data"])
                        # 创建系统状态事件
                        event = StatusEvent(
                            StatusEventType.SYSTEM,
                            self.status.copy()
                        )
                        self._notify_observers(event)
            except:
                pass
    
    def update_latency(self, asr_latency: float = None, mt_latency: float = None, tts_latency: float = None):
        """更新延迟信息"""
        with self._lock:
            if asr_latency is not None:
                self.status['asr_latency'] = asr_latency
            if mt_latency is not None:
                self.status['mt_latency'] = mt_latency
            if tts_latency is not None:
                self.status['tts_latency'] = tts_latency
            
            # 创建延迟事件
            event = StatusEvent(
                StatusEventType.LATENCY,
                {
                    'asr': self.status['asr_latency'],
                    'mt': self.status['mt_latency'],
                    'tts': self.status['tts_latency']
                }
            )
            self._notify_observers(event)
    
    def update_error_count(self, count: int):
        """更新错误计数"""
        with self._lock:
            self.status['error_count'] = count
            # 创建错误事件
            event = StatusEvent(
                StatusEventType.ERROR,
                {'error_count': count}
            )
            self._notify_observers(event)
    
    def update_exception(self, error: str):
        """更新异常信息"""
        with self._lock:
            self.status['error'] = error
            # 创建错误事件
            event = StatusEvent(
                StatusEventType.ERROR,
                {'error': error}
            )
            self._notify_observers(event)
    
    def update_latest_text(self, text_data: Dict[str, Any]):
        """更新最新文本"""
        with self._lock:
            self.status['latest_text'] = text_data
            # 创建文本事件
            event = StatusEvent(
                StatusEventType.TEXT,
                text_data
            )
            self._notify_observers(event)
    
    def update_running_state(self, is_running: bool, is_processing: bool):
        """更新运行状态"""
        with self._lock:
            self.status['is_running'] = is_running
            self.status['is_processing'] = is_processing
            # 创建处理状态事件
            event = StatusEvent(
                StatusEventType.PROCESSING,
                {
                    'is_running': is_running,
                    'is_processing': is_processing
                }
            )
            self._notify_observers(event)
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        with self._lock:
            return self.status.copy() 