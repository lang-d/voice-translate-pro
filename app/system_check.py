#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import sounddevice as sd
from utils.logger import logger

class SystemChecker:
    """系统检查类"""
    
    def __init__(self):
        self.requirements = {
            "python_version": ">=3.8",
            "cuda_version": ">=11.0",
            "memory": ">=8GB",
            "disk_space": ">=10GB"
        }
    
    def check_all(self) -> bool:
        """执行所有检查"""
        checks = [
            self.check_python_version(),
            self.check_cuda(),
            self.check_memory(),
            self.check_disk_space(),
            self.check_audio_devices(),
        ]
        return all(checks)
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        try:
            current_version = sys.version_info
            required_version = tuple(map(int, self.requirements["python_version"].replace(">=", "").split(".")))
            
            if current_version >= required_version:
                logger.info(f"Python版本检查通过: {sys.version}")
                return True
            else:
                logger.exception(f"Python版本不满足要求: 当前={sys.version}, 需要>={self.requirements['python_version']}")
                return False
        except Exception as e:
            logger.exception(f"Python版本检查失败: {str(e)}")
            return False
    
    def check_cuda(self) -> bool:
        """检查CUDA"""
        try:
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                required_version = self.requirements["cuda_version"].replace(">=", "")
                
                if cuda_version >= required_version:
                    logger.info(f"CUDA检查通过: {cuda_version}")
                    return True
                else:
                    logger.exception(f"CUDA版本不满足要求: 当前={cuda_version}, 需要>={required_version}")
                    return False
            else:
                logger.warning("CUDA不可用，将使用CPU")
                return True
        except Exception as e:
            logger.exception(f"CUDA检查失败: {str(e)}")
            return False
    
    def check_memory(self) -> bool:
        """检查内存"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            required_memory = int(self.requirements["memory"].replace(">=", "").replace("GB", "")) * 1024 * 1024 * 1024
            
            if memory.total >= required_memory:
                logger.info(f"内存检查通过: {memory.total / (1024**3):.1f}GB")
                return True
            else:
                logger.exception(f"内存不满足要求: 当前={memory.total / (1024**3):.1f}GB, 需要>={self.requirements['memory']}")
                return False
        except Exception as e:
            logger.exception(f"内存检查失败: {str(e)}")
            return False
    
    def check_disk_space(self) -> bool:
        """检查磁盘空间"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            required_space = int(self.requirements["disk_space"].replace(">=", "").replace("GB", "")) * 1024 * 1024 * 1024
            
            if disk.free >= required_space:
                logger.info(f"磁盘空间检查通过: {disk.free / (1024**3):.1f}GB")
                return True
            else:
                logger.exception(f"磁盘空间不满足要求: 当前={disk.free / (1024**3):.1f}GB, 需要>={self.requirements['disk_space']}")
                return False
        except Exception as e:
            logger.exception(f"磁盘空间检查失败: {str(e)}")
            return False
    
    def check_audio_devices(self) -> bool:
        """检查音频设备"""
        try:
            devices = sd.query_devices()
            if len(devices) > 0:
                logger.info(f"音频设备检查通过: 发现{len(devices)}个设备")
                return True
            else:
                logger.exception("未找到音频设备")
                return False
        except Exception as e:
            logger.exception(f"音频设备检查失败: {str(e)}")
            return False
