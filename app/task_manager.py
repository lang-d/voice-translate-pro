# utils/task_manager.py

import threading
import multiprocessing as mp
import time
import uuid
import os
import signal
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from utils.logger import logger

# 任务状态
class TaskStatus(Enum):
    PENDING = "pending"     # 等待执行
    RUNNING = "running"     # 正在执行
    COMPLETED = "completed" # 已完成
    FAILED = "failed"       # 失败
    CANCELED = "canceled"   # 已取消
    PAUSED = "paused"       # 已暂停

# 任务优先级
class TaskPriority(Enum):
    LOW = 0     # 低优先级，如下载任务
    NORMAL = 1  # 普通优先级
    HIGH = 2    # 高优先级

# 任务类型
class TaskType(Enum):
    THREAD = "thread"  # 线程任务
    PROCESS = "process"  # 进程任务

# 定义命令类型
class TaskCommand(Enum):
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    UPDATE = "update"

# 任务基类
class Task:
    """任务基类"""
    
    def __init__(self, 
                name: str, 
                task_type: TaskType = TaskType.THREAD, 
                priority: TaskPriority = TaskPriority.NORMAL):
        self.id = str(uuid.uuid4())
        self.name = name
        self.task_type = task_type
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.error = None
        self.created_time = time.time()
        self.start_time = None
        self.end_time = None
        self._canceled = False
        self._paused = False
        self._lock = threading.RLock()
        
        # 进程间通信
        if task_type == TaskType.PROCESS:
            self.command_queue = mp.Queue()
            self.result_queue = mp.Queue()
        
        # 回调函数
        self._progress_callbacks = []
        self._status_callbacks = []
        self._complete_callbacks = []
        self._error_callbacks = []
    
    def execute(self) -> Any:
        """
        执行任务，子类必须实现
        """
        raise NotImplementedError("子类必须实现execute方法")
    
    def run(self):
        """
        运行任务
        """
        with self._lock:
            if self.status != TaskStatus.PENDING:
                return
            self.status = TaskStatus.RUNNING
            self.start_time = time.time()
            self._notify_status_change()
        
        try:
            # 执行任务
            result = self.execute()
            
            with self._lock:
                if self._canceled:
                    self.status = TaskStatus.CANCELED
                    self._notify_status_change()
                    return
                
                self.result = result
                self.progress = 1.0
                self.status = TaskStatus.COMPLETED
                self.end_time = time.time()
                self._notify_status_change()
                
                # 触发完成回调
                self._notify_complete(result)
            
        except Exception as e:
            with self._lock:
                self.error = str(e)
                self.status = TaskStatus.FAILED
                self.end_time = time.time()
                self._notify_status_change()
                
                # 触发错误回调
                self._notify_error(str(e))
                        
            logger.error(f"任务执行失败: {self.name} - {e}")
    
    
    def handle_command(self, command: TaskCommand, data: Any = None) -> bool:
        """
        处理命令，子类可覆盖以实现更复杂的命令处理
        """
        if command == TaskCommand.STOP:
            return self.cancel()
        elif command == TaskCommand.PAUSE:
            return self.pause()
        elif command == TaskCommand.RESUME:
            return self.resume()
        elif command == TaskCommand.UPDATE:
            if isinstance(data, dict) and "progress" in data:
                self.update_progress(data["progress"])
                if "message" in data:
                    self.message = data["message"]
                return True
        return False
    
    def pause(self) -> bool:
        """暂停任务"""
        with self._lock:
            if self.status == TaskStatus.RUNNING:
                self._paused = True
                self.status = TaskStatus.PAUSED
                self._notify_status_change()
                return True
        return False
    
    def resume(self) -> bool:
        """恢复任务"""
        with self._lock:
            if self.status == TaskStatus.PAUSED:
                self._paused = False
                self.status = TaskStatus.RUNNING
                self._notify_status_change()
                return True
        return False
    
    def cancel(self) -> bool:
        """取消任务"""
        with self._lock:
            if self.status in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED]:
                self._canceled = True
                prev_status = self.status
                self.status = TaskStatus.CANCELED
                self.end_time = time.time()
                if prev_status != TaskStatus.CANCELED:
                    self._notify_status_change()
                return True
        return False
    
    def is_canceled(self) -> bool:
        """检查任务是否已取消"""
        return self._canceled
    
    def is_paused(self) -> bool:
        """检查任务是否已暂停"""
        return self._paused
    
    def update_progress(self, progress: float, message: str = ""):
        """更新进度"""
        with self._lock:
            self.progress = min(1.0, max(0.0, progress))
            if message:
                self.message = message
            
            # 触发进度回调
            self._notify_progress(self.progress)
    
    def on_progress(self, callback: Callable[[float], None]):
        """添加进度回调"""
        with self._lock:
            self._progress_callbacks.append(callback)
        return self
    
    def on_status_change(self, callback: Callable[[TaskStatus], None]):
        """添加状态变更回调"""
        with self._lock:
            self._status_callbacks.append(callback)
            # 如果已有状态，立即触发回调
            callback(self.status)
        return self
    
    def on_complete(self, callback: Callable[[Any], None]):
        """添加完成回调"""
        with self._lock:
            self._complete_callbacks.append(callback)
            # 如果已完成，立即触发回调
            if self.status == TaskStatus.COMPLETED:
                callback(self.result)
        return self
    
    def on_error(self, callback: Callable[[str], None]):
        """添加错误回调"""
        with self._lock:
            self._error_callbacks.append(callback)
            # 如果已失败，立即触发回调
            if self.status == TaskStatus.FAILED and self.error:
                callback(self.error)
        return self
    
    def _notify_progress(self, progress: float):
        """通知进度更新"""
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"进度回调执行失败: {e}")
    
    def _notify_status_change(self):
        """通知状态变更"""
        for callback in self._status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                logger.error(f"状态回调执行失败: {e}")
    
    def _notify_complete(self, result: Any):
        """通知任务完成"""
        for callback in self._complete_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"完成回调执行失败: {e}")
    
    def _notify_error(self, error: str):
        """通知任务出错"""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"错误回调执行失败: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """获取任务信息"""
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "type": self.task_type.value,
                "priority": self.priority.value,
                "status": self.status.value,
                "progress": self.progress,
                "message": self.message,
                "created_time": self.created_time,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "error": self.error
            }
    
    def __lt__(self, other):
        """用于任务队列排序"""
        if not isinstance(other, Task):
            return NotImplemented
        if self.priority.value != other.priority.value:
            # 值越小优先级越高
            return self.priority.value < other.priority.value
        # 同等优先级按创建时间排序
        return self.created_time < other.created_time


# 进程任务基类
class ProcessTask(Task):
    """进程任务基类，用于需要独立进程执行的任务"""
    
    def __init__(self, name: str, priority: TaskPriority = TaskPriority.NORMAL):
        super().__init__(name, TaskType.PROCESS, priority)
        self.process = None
        self.stop_event = mp.Event()
        self.command_queue = mp.Queue()
        self.result_queue = mp.Queue()
    
    def execute(self) -> Any:
        """
        进程任务执行逻辑
        """
        try:
            # 创建并启动进程
            self.process = mp.Process(
                target=self._process_main,
                args=(self.command_queue, self.result_queue, self.stop_event)
            )
            self.process.daemon = True
            self.process.start()
            
            # 等待进程就绪
            try:
                result = self.result_queue.get(timeout=10)  # 等待进程初始化
                if result.get("status") != "ready":
                    raise Exception(f"进程初始化失败: {result.get('error', '未知错误')}")
            except Exception as e:
                self._terminate_process()
                raise Exception(f"进程启动失败: {str(e)}")
            
            # 进程就绪，开始监听结果
            while not self.is_canceled() and self.process.is_alive():
                try:
                    # 非阻塞检查结果队列
                    if not self.result_queue.empty():
                        result = self.result_queue.get(block=False)
                        
                        # 处理进度更新
                        if "progress" in result:
                            self.update_progress(result["progress"], result.get("message", ""))
                        
                        # 处理结果
                        if "status" in result:
                            if result["status"] == "completed":
                                return result.get("result")
                            elif result["status"] == "error":
                                raise Exception(result.get("error", "未知错误"))
                            elif result["status"] == "exit":
                                break
                    
                    # 检查暂停状态
                    if self.is_paused():
                        # 发送暂停命令
                        self.command_queue.put({"command": "pause"})
                        # 等待恢复
                        while self.is_paused() and not self.is_canceled():
                            time.sleep(0.1)
                        # 如果恢复了，发送恢复命令
                        if not self.is_canceled():
                            self.command_queue.put({"command": "resume"})
                    
                    time.sleep(0.1)  # 避免CPU过载
                    
                except Exception as e:
                    logger.error(f"进程任务监听错误: {str(e)}")
                    # 继续监听
            
            # 如果是因为取消而退出的，发送停止命令
            if self.is_canceled():
                self.command_queue.put({"command": "stop"})
                # 等待进程退出
                self.process.join(timeout=3)
                # 如果进程没有退出，强制终止
                if self.process.is_alive():
                    self._terminate_process()
            
            # 获取最终结果
            try:
                if not self.result_queue.empty():
                    final_result = self.result_queue.get(block=False)
                    if "result" in final_result:
                        return final_result["result"]
            except:
                pass
                
            return None
            
        except Exception as e:
            # 确保进程被终止
            if self.process and self.process.is_alive():
                self._terminate_process()
            raise
    
    def _process_main(self, command_queue, result_queue, stop_event):
        """
        进程主函数，子类应重写此方法
        """
        try:
            # 初始化完成，通知主进程
            result_queue.put({"status": "ready"})
            
            # 主循环
            while not stop_event.is_set():
                # 检查命令队列
                try:
                    if not command_queue.empty():
                        command = command_queue.get(block=False)
                        
                        # 处理停止命令
                        if command.get("command") == "stop":
                            break
                            
                        # 处理其他命令...
                except:
                    pass
                    
                # 模拟任务进度
                for i in range(10):
                    if stop_event.is_set():
                        break
                    # 更新进度
                    result_queue.put({
                        "progress": (i + 1) / 10,
                        "message": f"处理中... {i+1}/10"
                    })
                    time.sleep(0.5)
                
                # 任务完成，返回结果
                result_queue.put({
                    "status": "completed",
                    "result": {"message": "进程任务完成"}
                })
                break
                
        except Exception as e:
            # 发送错误信息
            result_queue.put({
                "status": "error",
                "error": str(e)
            })
        finally:
            # 通知主进程退出
            result_queue.put({
                "status": "exit"
            })
    
    def cancel(self) -> bool:
        """取消进程任务"""
        if super().cancel():
            # 设置停止事件
            self.stop_event.set()
            # 发送停止命令
            try:
                self.command_queue.put({"command": "stop"})
            except:
                pass
            # 终止进程
            if self.process and self.process.is_alive():
                self._terminate_process()
            return True
        return False
    
    def _terminate_process(self):
        """强制终止进程"""
        try:
            # 先尝试正常终止
            if hasattr(self.process, 'terminate'):
                self.process.terminate()
                self.process.join(timeout=2)
                
            # 如果仍在运行，使用更强力的方法
            if self.process and self.process.is_alive():
                if hasattr(os, 'kill') and hasattr(self.process, 'pid'):
                    try:
                        os.kill(self.process.pid, signal.SIGKILL)
                    except:
                        pass
        except Exception as e:
            logger.error(f"终止进程失败: {str(e)}")


# 模型下载任务
class ModelDownloadTask(Task):
    """模型下载任务"""
    
    def __init__(self, engine_type: str, engine_name: str, model_name: str):
        super().__init__(f"下载模型: {engine_type}/{engine_name}/{model_name}", TaskType.THREAD, TaskPriority.LOW)
        self.engine_type = engine_type
        self.engine_name = engine_name
        self.model_name = model_name
    
    def execute(self) -> Any:
        """执行下载任务"""
        from utils.model_manager import model_manager
        
        # 执行下载
        def progress_callback(progress):
            if self.is_canceled():
                raise Exception("任务已取消")
            if self.is_paused():
                # 暂停时等待
                while self.is_paused() and not self.is_canceled():
                    time.sleep(0.5)
                if self.is_canceled():
                    raise Exception("任务已取消")
            self.update_progress(progress, f"下载中: {progress:.1%}")
        
        # 执行下载
        success = model_manager.download_model(
            self.engine_type,
            self.engine_name,
            self.model_name,
            progress_callback
        )
        
        if not success:
            raise Exception(f"模型下载失败: {self.engine_type}/{self.engine_name}/{self.model_name}")
        
        return {
            "success": True, 
            "engine_type": self.engine_type,
            "engine_name": self.engine_name,
            "model_name": self.model_name
        }
    
    def check_availability(self) -> Dict[str, Any]:
        """检查模型是否可下载"""
        from utils.model_manager import model_manager
        
        # 检查是否已存在
        is_downloaded = model_manager.is_model_downloaded(self.engine_type, self.engine_name, self.model_name)
        
        if is_downloaded:
            return {
                "available": False, 
                "message": "模型已下载",
                "is_downloaded": True
            }
        
        # 检查模型信息是否存在
        model_info = model_manager.get_model_info(self.engine_type, self.engine_name, self.model_name)
        if not model_info:
            return {
                "available": False,
                "message": f"找不到模型信息: {self.engine_type}/{self.engine_name}/{self.model_name}",
                "is_downloaded": False
            }
            
        return {
            "available": True,
            "message": "可以下载",
            "is_downloaded": False,
            "model_info": model_info
        }


# 翻译进程任务示例
class TranslationProcessTask(ProcessTask):

    """翻译进程任务示例"""
    
    def __init__(self, source_language: str, target_language: str, input_device=None, output_device=None):
        super().__init__(f"实时翻译: {source_language} -> {target_language}", TaskPriority.HIGH)
        self.source_language = source_language
        self.target_language = target_language
        self.input_device = input_device
        self.output_device = output_device
    
    def _process_main(self, command_queue, result_queue, stop_event):
        """翻译进程实现"""
        try:
            # 导入所需模块，在进程内部导入避免主进程初始化时的依赖问题
            from core.stream_processor import StreamProcessor
            import time
            
            # 通知主进程就绪
            result_queue.put({"status": "ready"})
            
            # 初始化流处理器
            processor = StreamProcessor()
            processor.set_languages(self.source_language, self.target_language)
            
            if self.input_device or self.output_device:
                processor.set_audio_devices(
                    self.input_device or "default",
                    self.output_device or "default"
                )
            
            # 启动翻译
            if not processor.start():
                result_queue.put({
                    "status": "error",
                    "error": "启动翻译流处理器失败"
                })
                return
            
            # 主循环
            result_queue.put({
                "progress": 0.5,
                "message": "翻译进行中..."
            })
            
            # 监控命令和状态
            paused = False
            while not stop_event.is_set():
                # 检查命令
                try:
                    if not command_queue.empty():
                        command = command_queue.get(block=False)
                        cmd = command.get("command")
                        
                        if cmd == "stop":
                            break
                        elif cmd == "pause" and not paused:
                            # 暂停处理
                            paused = True
                            processor.is_processing = False
                        elif cmd == "resume" and paused:
                            # 恢复处理
                            paused = False
                            processor.is_processing = True
                except:
                    pass
                
                # 获取状态
                status = processor.get_status()
                
                # 发送当前状态
                result_queue.put({
                    "progress": 0.8,  # 始终保持一个运行中的状态
                    "message": f"翻译进行中... " + (
                        f"ASR延迟: {status.get('latency', {}).get('asr', 0):.2f}s, "
                        f"翻译延迟: {status.get('latency', {}).get('translation', 0):.2f}s"
                    ),
                    "translation_status": status
                })
                
                time.sleep(0.5)
            
            # 停止处理器
            processor.stop()
            
            # 返回结果
            result_queue.put({
                "status": "completed",
                "result": {"message": "翻译已完成"}
            })
            
        except Exception as e:
            # 发送错误
            result_queue.put({
                "status": "error",
                "error": str(e)
            })
        finally:
            # 通知主进程退出
            result_queue.put({
                "status": "exit"
            })
    

        """检查翻译任务是否可用"""
        # 检查模型可用性
        from utils.model_manager import model_manager
        
        # 检查ASR模型
        asr_available = model_manager.is_model_downloaded("asr", "whisper", "base")
        
        # 检查翻译模型
        translation_available = model_manager.is_model_downloaded("translation", "nllb", "200M")
        
        # TTS通常不需要检查，因为edge_tts不需要下载模型
        
        if not asr_available:
            return {
                "available": False,
                "message": "ASR模型未下载，请先下载模型",
                "missing_models": [{"type": "asr", "engine": "whisper", "model": "base"}]
            }
            
        if not translation_available:
            return {
                "available": False,
                "message": "翻译模型未下载，请先下载模型",
                "missing_models": [{"type": "translation", "engine": "nllb", "model": "200M"}]
            }
            
        return {
            "available": True,
            "message": "所有必要模型已准备就绪",
            "asr_available": asr_available,
            "translation_available": translation_available
        }

class StatusMonitorTask(ProcessTask):
    """状态监控进程任务"""
    
    def __init__(self):
        super().__init__("系统状态监控", TaskPriority.HIGH)
        self._last_cpu_time = 0
        self._last_gpu_time = 0
        self._update_interval = 1.0  # 更新间隔（秒）
        
    def _process_main(self, command_queue, result_queue, stop_event):
        """状态监控进程实现"""
        try:
            import psutil
            import torch
            import time
            
            # 通知主进程就绪
            result_queue.put({"status": "ready"})
            
            # 初始状态
            status = {
                'cpu_usage': 0,
                'gpu_usage': 0,
                'asr_latency': 0,
                'mt_latency': 0,
                'tts_latency': 0,
                'error': None,
                'is_running': True,
                'is_processing': True,
                'error_count': 0,
                'start_time': time.time(),
                'latest_text': None
            }
            
            # 主循环
            while not stop_event.is_set():
                current_time = time.time()
                
                # 检查命令
                try:
                    if not command_queue.empty():
                        command = command_queue.get(block=False)
                        cmd = command.get("command")
                        
                        # 处理停止命令
                        if cmd == "stop":
                            break
                        # 处理更新延迟命令
                        elif cmd == "update_latency" and "data" in command:
                            data = command["data"]
                            if "asr" in data:
                                status["asr_latency"] = data["asr"]
                            if "mt" in data:
                                status["mt_latency"] = data["mt"]
                            if "tts" in data:
                                status["tts_latency"] = data["tts"]
                        # 处理更新最新文本命令
                        elif cmd == "update_text" and "data" in command:
                            status["latest_text"] = command["data"]
                        # 处理更新错误命令
                        elif cmd == "update_error" and "data" in command:
                            status["error"] = command["data"]
                except:
                    pass
                
                # 更新CPU使用率（每秒更新一次）
                if current_time - self._last_cpu_time >= self._update_interval:
                    status['cpu_usage'] = psutil.cpu_percent(interval=None)
                    self._last_cpu_time = current_time
                
                # 更新GPU使用率（每秒更新一次）
                if current_time - self._last_gpu_time >= self._update_interval:
                    if torch.cuda.is_available():
                        try:
                            status['gpu_usage'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                        except:
                            status['gpu_usage'] = 0
                    else:
                        status['gpu_usage'] = 0
                    self._last_gpu_time = current_time
                
                # 计算运行时长
                if status['start_time']:
                    status['running_duration'] = int(current_time - status['start_time'])
                
                # 发送当前状态
                result_queue.put({
                    "progress": 0.5,  # 保持运行中状态
                    "message": "状态监控中...",
                    "status_data": status.copy()  # 发送副本避免数据竞争
                })
                
                # 使用更精确的睡眠时间
                sleep_time = max(float(0), self._update_interval - (time.time() - current_time))
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
        except Exception as e:
            # 发送错误
            result_queue.put({
                "status": "error",
                "error": str(e)
            })
        finally:
            # 通知主进程退出
            result_queue.put({
                "status": "exit"
            })

# 任务管理器
class TaskManager:
    """任务管理器"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TaskManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.tasks = {}  # 任务ID -> 任务对象
            self.running = False
            self.worker_threads = []
            self.max_workers = 3
            self.task_queue = []  # 任务队列
            self.queue_lock = threading.RLock()
            self.task_available = threading.Event()
            

    def start(self):
        """启动任务管理器"""
        with self._lock:
            if self.running:
                return
                
            self.running = True
            
            # 创建工作线程
            for i in range(self.max_workers):
                worker = threading.Thread(target=self._worker_thread, name=f"TaskWorker-{i}")
                worker.daemon = True
                worker.start()
                self.worker_threads.append(worker)
            
            logger.info(f"任务管理器已启动，工作线程数：{self.max_workers}")
    
    def stop(self):
        """停止任务管理器"""
        with self._lock:
            if not self.running:
                return
                
            self.running = False
            self.task_available.set()  # 唤醒所有等待的线程
            
            # 等待所有线程结束
            for worker in self.worker_threads:
                worker.join(timeout=2.0)
            
            # 取消所有任务
            for task_id in list(self.tasks.keys()):
                task = self.tasks[task_id]
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.PAUSED]:
                    task.cancel()
            
            self.worker_threads = []
            logger.info("任务管理器已停止")
    
    def submit_task(self, task: Task) -> str:
        """提交任务"""
        # 确保任务管理器已启动
        if not self.running:
            self.start()
        
        
        # 添加任务到队列
        with self.queue_lock:
            self.tasks[task.id] = task
            
            # 如果是进程任务，直接执行
            if task.task_type == TaskType.PROCESS:
                # 在新线程中启动进程任务，以免阻塞主线程
                def start_process_task():
                    task.run()
                    
                thread = threading.Thread(target=start_process_task)
                thread.daemon = True
                thread.start()
            else:
                # 线程任务加入队列
                self.task_queue.append(task)
                # 按优先级排序
                self.task_queue.sort(key=lambda t: t.priority.value)
                # 通知有新任务
                self.task_available.set()
            
            logger.info(f"已提交任务: {task.name} (ID: {task.id}, 类型: {task.task_type.value})")
            return task.id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """获取所有任务"""
        return list(self.tasks.values())
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """获取指定状态的任务"""
        return [task for task in self.tasks.values() if task.status == status]
    
    def get_tasks_by_type(self, task_type: TaskType) -> List[Task]:
        """获取指定类型的任务"""
        return [task for task in self.tasks.values() if task.task_type == task_type]
    
    def send_command(self, task_id: str, command: TaskCommand, data: Any = None) -> bool:
        """
        向任务发送命令
        
        Args:
            task_id: 任务ID
            command: 命令类型
            data: 命令附加数据
            
        Returns:
            bool: 命令是否成功发送
        """
        task = self.get_task(task_id)
        if not task:
            return False
            
        return task.handle_command(command, data)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        task = self.get_task(task_id)
        if task:
            return task.cancel()
        return False
    
    def clear_completed_tasks(self):
        """清理已完成的任务"""
        with self.queue_lock:
            for task_id in list(self.tasks.keys()):
                task = self.tasks[task_id]
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED]:
                    del self.tasks[task_id]
    
        
    
    def _worker_thread(self):
        """工作线程函数"""
        while self.running:
            task = None
            
            # 从队列中获取任务
            with self.queue_lock:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                else:
                    self.task_available.clear()
            
            if task:
                # 执行任务
                task.run()
            else:
                # 等待新任务
                self.task_available.wait(timeout=1.0)