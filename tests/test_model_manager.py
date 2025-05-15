#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import shutil
import unittest
from unittest.mock import patch, MagicMock
import tempfile

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_manager import ModelManager
from utils.config import config

class TestModelManager(unittest.TestCase):
    """测试模型管理器"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.original_model_dir = config.get_model_dir
        
        # 修改配置以使用临时目录
        def get_temp_model_dir(engine_type):
            return os.path.join(self.temp_dir, engine_type)
        config.get_model_dir = get_temp_model_dir
        
        # 创建模型管理器实例
        self.model_manager = ModelManager()
        
        # 使用真实的模型信息
        self.test_model_info = {
            "asr": {
                "whisper": {
                    "base": {
                        "size": 1420000000,  # 1.42GB
                        "url": "https://huggingface.co/openai/whisper-base/resolve/main/model.safetensors",
                        "md5": "ed3a0b6b1c0b9b13c0480b4c2967dd72"
                    }
                }
            }
        }
        
        # 保存原始模型信息
        self.original_model_info = self.model_manager.model_info
        
        # 替换模型信息为测试数据
        self.model_manager.model_info = self.test_model_info
        
    def tearDown(self):
        """测试后的清理工作"""
        # 恢复原始配置
        config.get_model_dir = self.original_model_dir
        
        # 恢复原始模型信息
        self.model_manager.model_info = self.original_model_info
        
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_download_model_success(self):
        """测试成功下载模型"""
        # 记录下载进度
        progress_values = []
        def progress_callback(progress):
            progress_values.append(progress)
            print(f"当前下载进度: {progress:.2%}")
        
        # 执行下载
        success = self.model_manager.download_model(
            "asr", "whisper", "base",
            progress_callback=progress_callback
        )
        
        # 验证结果
        self.assertTrue(success)
        
        # 验证文件是否创建
        model_path = os.path.join(self.temp_dir, "asr", "whisper", "base")
        self.assertTrue(os.path.exists(os.path.join(model_path, "model.bin")))
        self.assertTrue(os.path.exists(os.path.join(model_path, "config.json")))
        
        # 验证进度更新
        self.assertTrue(len(progress_values) > 0)
        self.assertTrue(all(0 <= x <= 1 for x in progress_values))
        self.assertTrue(progress_values[-1] >= 0.99)  # 最终进度应该接近1
    
    def test_download_model_resume(self):
        """测试断点续传功能"""
        # 创建部分下载的文件
        model_path = os.path.join(self.temp_dir, "asr", "whisper", "base")
        os.makedirs(model_path, exist_ok=True)
        
        temp_file = os.path.join(model_path, "model.bin.tmp")
        with open(temp_file, 'wb') as f:
            f.write(b"partial_data" * 100)  # 写入部分数据
        
        # 创建进度文件
        progress_file = os.path.join(model_path, "download_progress.json")
        with open(progress_file, 'w') as f:
            json.dump({
                'downloaded_size': 1000,
                'total_size': 1420000000,
                'url': "https://huggingface.co/openai/whisper-base/resolve/main/model.safetensors",
                'timestamp': 0
            }, f)
        
        # 记录下载进度
        progress_values = []
        def progress_callback(progress):
            progress_values.append(progress)
            print(f"当前下载进度: {progress:.2%}")
        
        # 执行下载
        success = self.model_manager.download_model(
            "asr", "whisper", "base",
            progress_callback=progress_callback
        )
        
        # 验证结果
        self.assertTrue(success)
        
        # 验证最终文件大小
        final_file = os.path.join(model_path, "model.bin")
        self.assertTrue(os.path.exists(final_file))
        
        # 验证进度更新
        self.assertTrue(len(progress_values) > 0)
        self.assertTrue(all(0 <= x <= 1 for x in progress_values))
        self.assertTrue(progress_values[-1] >= 0.99)
    
    def test_get_download_progress(self):
        """测试获取下载进度功能"""
        # 记录下载进度
        progress_values = []
        def progress_callback(progress):
            progress_values.append(progress)
            print(f"当前下载进度: {progress:.2%}")
        
        # 开始下载
        self.model_manager.download_model(
            "asr", "whisper", "base",
            progress_callback=progress_callback
        )
        
        # 获取进度
        progress = self.model_manager.get_download_progress("asr", "whisper", "base")
        
        # 验证进度
        self.assertTrue(0 <= progress <= 1)
        self.assertTrue(progress >= 0.99)  # 下载完成后进度应该接近1


if __name__ == '__main__':
    unittest.main() 