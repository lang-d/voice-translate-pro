#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from typing import Dict, Any
from enum import Enum

class StatusEventType(Enum):
    """状态事件类型"""
    LATENCY = "latency"      # 延迟更新
    ERROR = "error"         # 错误更新
    TEXT = "text"           # 文本更新
    PROCESSING = "processing"  # 处理状态更新
    SYSTEM = "system"       # 系统状态更新

class StatusEvent:
    """状态事件类"""
    def __init__(self, event_type: StatusEventType, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        self.timestamp = time.time()
    
    def __str__(self):
        return f"StatusEvent(type={self.event_type.value}, data={self.data}, timestamp={self.timestamp})" 