#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动语音识别 (ASR) 模块
"""

from core.asr.whisper_asr import WhisperASR
from core.asr.faster_whisper_asr import FasterWhisperASR
from core.asr.vosk_asr import VoskASR


__all__ = ['WhisperASR']