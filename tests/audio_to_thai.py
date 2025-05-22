import os
import soundfile as sf
import numpy as np
from core.stream_processor import StreamProcessor

def split_audio(audio, sample_rate, chunk_duration=30):
    """将音频按chunk_duration（秒）切片"""
    chunk_size = int(sample_rate * chunk_duration)
    total_len = len(audio)
    for start in range(0, total_len, chunk_size):
        yield audio[start:start+chunk_size]

def resample_audio(audio, orig_sr, target_sr):
    """重采样音频到目标采样率"""
    if orig_sr == target_sr:
        return audio
    import librosa
    audio_resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio_resampled


def test_funasr_best_effect():
    """
    测试 FunASR 最佳效果：Paraformer大模型 + VAD + 标点
    """
    try:
        from funasr import AutoModel
        import soundfile as sf

        print("开始 FunASR 最佳效果测试...")
        model_cache_dir = r"D:\github\voice-translate-pro\models\asr\funasr\paraformer-large"

        # 1. 加载模型（自动下载到本地缓存）
        model = AutoModel(
            cache_dir=model_cache_dir,
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",  # 离线大模型
            vad_model="iic/fsmn-vad_zh-cn-16k-common-pytorch",  # 端点检测
            punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"  # 标点恢复
        )

        # 2. 准备音频文件
        audio_path = r"""D:\xhs\vv.WAV"""  # 替换为你的音频文件路径
        sample_rate = 16000  # 目标采样率

        # 读取音频
        audio, sr = sf.read(audio_path)
        # 自动转为单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        # 自动重采样
        if sr != sample_rate:
            print(f"采样率不符，当前: {sr}，需要: {sample_rate}，自动重采样中...")
            audio = resample_audio(audio, sr, sample_rate)
            print("重采样完成！")
            print(f"音频信息 - 采样率: {sample_rate}Hz, 时长: {len(audio)/sample_rate:.2f}秒")

        # 3. 执行识别
        print("正在识别...")
        result = model.generate(input=audio)

        # 4. 输出结果
        print("\n识别结果:")
        print("-" * 50)
        print(result)
        print("-" * 50)

        return True

    except Exception as e:
        print(f"测试过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_funasr_stream():
    """
    测试 FunASR 流式识别：实时语音转写
    """
    try:
        from funasr import AutoModel
        import soundfile as sf
        import numpy as np

        print("开始 FunASR 流式识别测试...")

        # 1. 加载流式模型
        model = AutoModel(
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",  # 流式模型
            vad_model="iic/fsmn-vad_zh-cn-16k-common-pytorch",  # 端点检测
            punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"  # 标点恢复
        )

        # 2. 准备音频
        audio_path = r"D:\github\voice-translate-pro\tests\test.wav"  # 替换为你的音频文件路径
        audio, sample_rate = sf.read(audio_path)

        # 3. 模拟流式输入（每段2秒）
        chunk_size = int(2 * sample_rate)  # 2秒的数据量
        total_chunks = len(audio) // chunk_size + 1

        print(f"\n开始模拟流式识别，将音频分成 {total_chunks} 个片段处理...")
        print("-" * 50)

        # 4. 逐块处理
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(audio))
            chunk = audio[start_idx:end_idx]

            # 是否为最后一块
            is_final = i == total_chunks - 1

            # 识别当前片段
            result = model.generate(
                input=chunk,
                is_final=is_final
            )

            if result.strip():  # 如果有识别结果
                print(f"片段 {i+1}/{total_chunks}: {result}")

        print("-" * 50)
        return True

    except Exception as e:
        print(f"测试过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_soundfile():
    import numpy as np
    import soundfile as sf

    data = np.zeros(16000, dtype=np.float32)
    sf.write(r'D:\github\voice-translate-pro\output\audio\test.wav', data, 16000)
    print("写入成功")

def test_main():
    input_audio_path = r"""D:\xhs\vv.WAV"""  # 输入音频路径
    output_audio_path = r"D:\github\voice-translate-pro\output\audio\output_thai_5.wav"  # 输出泰文语音路径
    temp_dir = r"D:\github\voice-translate-pro\output\audio\temp"
    os.makedirs(temp_dir, exist_ok=True)
    sample_rate = 16000  # 目标采样率

    # 读取音频
    audio, sr = sf.read(input_audio_path)
    # 自动转为单声道
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    # 自动重采样
    if sr != sample_rate:
        print(f"采样率不符，当前: {sr}，需要: {sample_rate}，自动重采样中...")
        audio = resample_audio(audio, sr, sample_rate)
        print("重采样完成！")

    # 初始化处理器
    processor = StreamProcessor()
    asr_config = {
        "engine": "whisper",
        "model": "small",
        "language": "zh"
    }
    translation_config = {
        "engine": "nllb",
        "model": "base",
        "source_language": "zh",
        "target_language": "th"
    }
    tts_config = {
        "engine": "f5_tts",
        "voice": "female_30",
        "language": "th",
        "speed": 1.05,
        "use_enhancements": True,
        "model": "thai"
    }
    ok = processor.configure(asr=asr_config, translation=translation_config, tts=tts_config)
    if not ok:
        print("配置失败")
        return

    # 分段处理
    chunk_duration = 5  # 每段30秒
    # 1. 识别和翻译阶段
    for i, chunk in enumerate(split_audio(audio, sample_rate, chunk_duration)):
        asr_path = os.path.join(temp_dir, f"asr_{i}.txt")
        trans_path = os.path.join(temp_dir, f"trans_{i}.txt")
        if os.path.exists(trans_path):
            print(f"第{i+1}段翻译已存在，跳过识别和翻译")
            continue
        print(f"正在识别第{i+1}段...")
        try:
            # 识别
            asr_text = processor.pipeline.asr_manager.transcribe(chunk)
            print(f"识别结果: {asr_text}")
            with open(asr_path, "w", encoding="utf-8") as f:
                f.write(asr_text or "")
            # 翻译
            if asr_text:
                trans_text = processor.pipeline.translation_manager.translate(asr_text)
            else:
                trans_text = ""
            print(f"翻译结果: {trans_text}")
            with open(trans_path, "w", encoding="utf-8") as f:
                f.write(trans_text or "")
        except Exception as e:
            print(f"第{i+1}段识别/翻译异常: {e}")

    # 2. 合成阶段
    with sf.SoundFile(output_audio_path, 'w', samplerate=sample_rate, channels=1) as f:
        for i in range(len(list(split_audio(audio, sample_rate, chunk_duration)))):
            trans_path = os.path.join(temp_dir, f"trans_{i}.txt")
            audio_path = os.path.join(temp_dir, f"tts_{i}.wav")
            if os.path.exists(audio_path):
                print(f"第{i+1}段音频已存在，跳过合成")
                audio_data, _ = sf.read(audio_path)
                f.write(audio_data)
                continue
            with open(trans_path, "r", encoding="utf-8") as tf:
                trans_text = tf.read().strip()
            if not trans_text:
                print(f"第{i+1}段无翻译文本，跳过合成")
                continue
            print(f"正在合成第{i+1}段...")
            try:
                tts_audio = processor.pipeline.tts_manager.synthesize(trans_text)
                if tts_audio is not None:
                    sf.write(audio_path, tts_audio, sample_rate)
                    f.write(tts_audio)
                else:
                    print(f"第{i+1}段合成失败，跳过")
            except Exception as e:
                print(f"第{i+1}段合成异常: {e}")


    print(f"全部处理完成，输出文件: {output_audio_path}")

def ensure_wav_1d_float32(audio):

    # torch tensor
    if 'torch' in str(type(audio)):
        audio = audio.detach().cpu().numpy()
    # list/tuple包一层
    if isinstance(audio, (list, tuple)):
        # 如果是 [array], 取第一个；如果是多层，flatten
        if len(audio) == 1 and isinstance(audio[0], np.ndarray):
            audio = audio[0]
        else:
            audio = np.array(audio).flatten()
    # 强制一维 float32
    audio = np.array(audio).astype(np.float32).flatten()
    return audio

def test_f5_tts_direct():
    """直接测试F5-TTS引擎"""
    from core.tts.f5_tts_engine import F5TTSEngine
    import soundfile as sf
    import numpy as np
    import torch

    # 配置参数
    gpu_config = {
        "enabled": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    performance_config = {
        "use_jit": False,
        "use_fp16": False
    }
    audio_config = {
        "input": {
            "sample_rate": 16000,
            "channels": 1
        },
        "output": {
            "sample_rate": 16000,
            "channels": 1
        }
    }

    try:
        # 初始化引擎
        engine = F5TTSEngine()

        # 配置引擎
        ok = engine.configure(
            model="thai",
            voice="female_30",
            language="th",
            gpu=gpu_config,
            performance=performance_config,
            audio=audio_config
        )

        if not ok:
            print("引擎配置失败")
            return

        # 测试文本
        test_text = "ถ้าใครอยากดื่มน้ําสักแก้ว ก็สามารถเล่นกับประชาชนของเราได้ในขณะนี้"

        print("开始合成...")
        # 执行合成
        audio = engine.synthesize(test_text)

        if audio is None:
            print("合成失败")
            return

        # 保存音频
        output_path = r"D:\github\voice-translate-pro\output\audio\test_direct_3.wav"
        sf.write(output_path, audio, audio_config["output"]["sample_rate"])
        print(f"合成完成，已保存到: {output_path}")

        # 清理资源
        engine.cleanup()

    except Exception as e:
        print(f"测试过程出错: {e}")
        import traceback
        traceback.print_exc()




