#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from utils.logger import logger
from utils.config import config
from app.ui import create_ui


def check_environment():
    """检查运行环境"""
    try:
        import torch
        
        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            logger.info(f"检测到{gpu_count}个GPU设备:")
            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # 转换为GB
                logger.info(f"  GPU {i}: {device_name} (显存: {total_memory:.1f}GB)")
        else:
            logger.warning("未检测到可用的GPU设备，将使用CPU模式运行")
            
        # 检查系统状态
        from app.system_check import SystemChecker
        checker = SystemChecker()
        if not checker.check_all():
            logger.exception("系统检查未通过")
            return False
            
        return True
        
    except Exception as e:
        logger.exception(f"环境检查失败: {str(e)}")
        return False

def main():
    """主函数"""
    try:
        logger.info("启动语音翻译工具")


        # 设置日志级别
        log_level = config.get('app.log_level', 'INFO')
        logger.setLevel(log_level)
        logger.info(f"日志级别设置为: {log_level}")

        # 检查环境
        logger.info("检查系统环境...")
        if not check_environment():
            logger.exception("环境检查失败，程序退出")
            return
        logger.info("环境检查完成")

        # 设置异常钩子
        def exception_hook(exctype, value, traceback):
            logger.exception("未捕获的异常:", exc_info=(exctype, value, traceback))
        sys.excepthook = exception_hook

        # 直接创建UI
        try:
            logger.info("开始创建UI...")
            exit_code = create_ui()
            logger.info("UI结束运行")
            
            # 确保正确退出
            if exit_code != 0:
                logger.warning(f"程序异常退出，退出码: {exit_code}")
            else:
                logger.info("程序正常退出")

            sys.exit(exit_code)
                
        except Exception as e:
            logger.exception(f"UI运行失败: {str(e)}")
            logger.exception("详细错误信息:", exc_info=True)
            return

    except Exception as e:
        logger.exception(f"程序运行失败: {str(e)}")
        logger.exception("详细错误信息:", exc_info=True)
        return

if __name__ == "__main__":
    main()