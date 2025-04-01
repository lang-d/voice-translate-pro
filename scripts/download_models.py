# scripts/download_models.py
import os
import sys
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("voice-translate.download")

def download_whisper_model(model_size="base", models_dir="models/whisper"):
    """下载Whisper模型"""
    logger.info(f"开始下载Whisper模型 ({model_size})...")

    try:
        # 创建模型目录
        os.makedirs(models_dir, exist_ok=True)

        # 尝试导入faster_whisper
        from faster_whisper import WhisperModel

        # 下载模型（会自动缓存到~/.cache/huggingface/hub）
        model = WhisperModel(model_size, download_root=models_dir)

        logger.info(f"Whisper模型 ({model_size}) 下载完成")
        return True

    except ImportError:
        logger.exception("未安装faster_whisper库，请执行 pip install faster-whisper")
        return False
    except Exception as e:
        logger.exception(f"下载Whisper模型时出错: {e}", exc_info=True)
        return False

def download_nllb_model(model_size="distilled-600M", models_dir="models/nllb"):
    """下载NLLB翻译模型"""
    logger.info(f"开始下载NLLB模型 ({model_size})...")

    try:
        # 创建模型目录
        os.makedirs(models_dir, exist_ok=True)

        # 尝试导入transformers
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # 获取模型名称
        if model_size == "distilled-600M":
            model_name = "facebook/nllb-200-distilled-600M"
        elif model_size == "1.3B":
            model_name = "facebook/nllb-200-1.3B"
        elif model_size == "distilled-1.3B":
            model_name = "facebook/nllb-200-distilled-1.3B"
        else:
            model_name = model_size  # 直接使用提供的名称

        # 下载模型
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=models_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=models_dir)

        logger.info(f"NLLB模型 ({model_size}) 下载完成")
        return True

    except ImportError:
        logger.exception("未安装transformers库，请执行 pip install transformers")
        return False
    except Exception as e:
        logger.exception(f"下载NLLB模型时出错: {e}", exc_info=True)
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='下载语音翻译所需模型')
    parser.add_argument('--whisper', type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help='Whisper模型大小 (默认: base)')
    parser.add_argument('--nllb', type=str, default="distilled-600M",
                        choices=["distilled-600M", "1.3B", "distilled-1.3B"],
                        help='NLLB模型大小 (默认: distilled-600M)')
    parser.add_argument('--models_dir', type=str, default="models",
                        help='模型存储目录 (默认: models)')
    parser.add_argument('--skip_whisper', action="store_true",
                        help='跳过下载Whisper模型')
    parser.add_argument('--skip_nllb', action="store_true",
                        help='跳过下载NLLB模型')

    args = parser.parse_args()

    # 创建主模型目录
    os.makedirs(args.models_dir, exist_ok=True)

    success = True

    # 下载Whisper模型
    if not args.skip_whisper:
        whisper_dir = os.path.join(args.models_dir, "whisper")
        if not download_whisper_model(args.whisper, whisper_dir):
            success = False

    # 下载NLLB模型
    if not args.skip_nllb:
        nllb_dir = os.path.join(args.models_dir, "nllb")
        if not download_nllb_model(args.nllb, nllb_dir):
            success = False

    # 检查TTS引擎
    try:
        import edge_tts
        logger.info("Edge TTS已安装")
    except ImportError:
        logger.warning("未安装edge-tts库，请执行 pip install edge-tts")

    if success:
        logger.info("所有模型下载完成")
        return 0
    else:
        logger.exception("部分模型下载失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())