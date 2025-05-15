
def test_yourtts_download_model():
    # Load model directly
    from transformers import AutoModel
    cache_dir = "models/tts/yourtts/base"
    model = AutoModel.from_pretrained("wannaphong/khanomtan-tts-v1.0",cache_dir=cache_dir)

def test_f5_tts():
    import torch
    import numpy as np
    import soundfile as sf
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import (
        preprocess_ref_audio_text,
        load_model,
        load_vocoder,
        get_tokenizer,
        infer_process,
        target_sample_rate,
        n_fft,
        hop_length,
        win_length,
        n_mel_channels,
    )
    from f5_tts.model import DiT
    from f5_tts.model.utils import seed_everything

    # ========== 路径配置 ==========
    MODEL_PATH = r"/models/tts/f5_tts/thai/model.pt"
    VOCAB_PATH = r"D:\github\voice-translate-pro\models\tts\f5_tts\thai\vocab.txt"
    REF_AUDIO_PATH = r"D:\github\voice-translate-pro\tests\ref.wav"
    REF_TEXT = "ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น."  # 替换为你的参考语音的内容
    GEN_TEXT = "สวัสดีค่ะ ยินดีต้อนรับเข้าสู่ร้านของเรา หวังว่าคุณจะมีความสุขกับการเลือกซื้อสินค้าในวันนี้"
    OUTPUT_PATH = f"""output_{2}.wav"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mel_spec_type = "vocos"
    tokenizer = "custom"

    # ========== 加载 tokenizer ==========
    vocab_char_map, vocab_size = get_tokenizer(VOCAB_PATH, tokenizer)

    # ========== 模型参数配置 ==========
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4
    )

    # ========== 构建模型（自动封装为 CFM）==========
    print("🎛 正在构建模型...")
    model = load_model(
        model_cls=DiT,
        model_cfg=model_cfg,
        ckpt_path=MODEL_PATH,
        mel_spec_type=mel_spec_type,
        vocab_file=VOCAB_PATH,
        use_ema=True,
        device=device
    )
    model.eval()

    # ========== 加载 vocoder ==========
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False, device=device)

    # ========== 加载参考音频 ==========
    ref_audio_tensor, ref_text_proc = preprocess_ref_audio_text(REF_AUDIO_PATH, REF_TEXT)

    # ========== 合成语音 ==========
    print("🎙 开始合成语音...")
    seed_everything(42)

    # | 参数名                  | 当前值  | 建议调整范围     | 作用                        |
    # | -------------------- | ---- | ---------- | ------------------------- |
    # | `cfg_strength`       | 2.0  | 1.5 – 2.8  | 控制生成音频与参考风格的相似度（越高越接近参考音） |
    # | `sway_sampling_coef` | -1.0 | -2.0 – 0.0 | 控制情感风格的“摇摆”幅度，提升语气变化感     |
    # | `speed`              | 1.0  | 0.95 – 1.2 | 微调语速，稍快语速在直播带货中更有“吸引力”    |
    # | `nfe_step`           | 32   | 40 – 48    | 增加可以改善合成质量（但可能带来延迟）       |


    speed = 1.05
    cfg_strength = 2.8
    cross_fade_duration = 0.12
    nfe_step = 48
    sway_sampling_coef = 0.2
    target_rms = -18.0
    fix_duration = None


    audio, sample_rate, _ = infer_process(
            ref_audio=ref_audio_tensor,
            ref_text=ref_text_proc,
            gen_text=GEN_TEXT,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type=mel_spec_type,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=torch.device(device)
        )

    # ========== 保存结果 ==========
    sf.write(OUTPUT_PATH, audio, sample_rate)
    print(f"✅ 合成完成，保存为：{OUTPUT_PATH}")

def test_live_f5_tts():
    import torch
    import soundfile as sf
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import (
        preprocess_ref_audio_text,
        load_model,
        load_vocoder,
        infer_process,
        get_tokenizer,
        n_fft, hop_length, win_length,
        n_mel_channels, target_sample_rate,
    )

    from f5_tts.model import UNetT
    from f5_tts.model.utils import seed_everything


    def live_tts(
            ref_audio_path: str,
            ref_text: str,
            gen_text: str,
            model_path: str = "model.pt",
            vocab_path: str = "vocab.txt",
            output_path: str = "live_output.wav",
            use_cuda: bool = True
    ):
        # ===== 设置设备 =====
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        device_torch = torch.device(device)

        # ===== 加载 tokenizer =====
        vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

        # ===== 模型配置（UNetT 更适合情感语调） =====
        model_cfg = dict(
            dim=1024,
            depth=20,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4
        )

        # ===== 加载模型（封装为 CFM） =====
        model = load_model(
            model_cls=UNetT,
            model_cfg=model_cfg,
            ckpt_path=model_path,
            mel_spec_type="bigvgan",
            vocab_file=vocab_path,
            use_ema=True,
            device=device
        )
        model.eval()

        # ===== 加载 vocoder =====
        vocoder = load_vocoder(vocoder_name="bigvgan", is_local=False, device=device)

        # ===== 预处理参考语音 =====
        ref_audio_tensor, ref_text_proc = preprocess_ref_audio_text(ref_audio_path, ref_text)

        # ===== 设置随机种子确保一致性 =====
        seed_everything(42)

        # ===== 合成语音 =====
        audio, sample_rate, _ = infer_process(
            ref_audio=ref_audio_tensor,
            ref_text=ref_text_proc,
            gen_text=gen_text,
            model_obj=model,
            vocoder=vocoder,
            mel_spec_type="bigvgan",
            target_rms=-19.0,
            cross_fade_duration=0.12,
            nfe_step=48,
            cfg_strength=2.8,
            sway_sampling_coef=-1.0,
            speed=1.05,
            fix_duration=None,
            device=device_torch,
        )

        # ===== 保存输出音频 =====
        sf.write(output_path, audio, sample_rate)
        print(f"✅ 合成完成，输出路径: {output_path}")

    MODEL_PATH = r"/models/tts/f5_tts/thai/model.pt"
    VOCAB_PATH = r"D:\github\voice-translate-pro\models\tts\f5_tts\thai\vocab.txt"
    REF_AUDIO_PATH = r"D:\github\voice-translate-pro\tests\ref.wav"
    REF_TEXT = "ได้รับข่าวคราวของเราที่จะหาที่มันเป็นไปที่จะจัดขึ้น."  # 替换为你的参考语音的内容
    GEN_TEXT = "สวัสดีค่ะ ยินดีต้อนรับเข้าสู่ร้านของเรา หวังว่าคุณจะมีความสุขกับการเลือกซื้อสินค้าในวันนี้"
    OUTPUT_PATH = f"""live_output_{2}.wav"""

    live_tts(
        ref_audio_path=REF_AUDIO_PATH,
        ref_text=REF_TEXT,
        gen_text=GEN_TEXT,
        model_path=MODEL_PATH,
        vocab_path=VOCAB_PATH,
        output_path=OUTPUT_PATH,
        use_cuda=True,
    )


