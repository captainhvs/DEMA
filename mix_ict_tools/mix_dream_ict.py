import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard, HighpassFilter, LowpassFilter, Compressor, 
    Reverb, Gain, Limiter, PeakFilter, Chorus, Delay
)

def apply_dream_effect(input_path: str, output_path: str, mode: str = "single_vocal") -> str:
    """
    [混音工具类] 
    项目名称：单轨梦境空间音效模拟器 (Single-Track Dream-like Simulator)
    适用场景：将音频处理成超现实、虚幻且具有漂浮感的“梦境”听感。常用于回忆、幻觉或无意识状态的音频表现。
    
    【重要限制】：本工具仅支持单条音轨处理，无法同时接收或混合两条音轨。如需制作“带背景音的梦境环境”，请分次调用处理。

    功能特性与调用模式：
    1. 'single_vocal' (梦境人声模式)：默认模式。侧重于柔化人声边缘，通过轻微的 Chorus 和大房间混响，使人声听起来疏离且空灵。
    2. 'single_ambient' (梦境环境模式)：通过 Chorus 产生音高微漂移，结合长延时（Delay）和高湿度的混响（Reverb）营造时间模糊感。

    :param input_path: 待处理音频文件的绝对路径。
    :param output_path: 处理后音频文件需要保存的绝对路径（需包含 .wav 后缀）。
    :param mode: 处理模式，请严格从 ['single_vocal', 'single_ambient'] 中选择。
        
    :return: 处理成功或失败的状态文本信息。
    """
    if isinstance(input_path, dict) and 'value' in input_path:
        input_path = input_path['value']
    if isinstance(output_path, dict) and 'value' in output_path:
        output_path = output_path['value']
    try:
        audio, sr = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # --- 梦境环境轨定义 ---
        dream_board1 = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=200),
            LowpassFilter(cutoff_frequency_hz=8000),
            PeakFilter(cutoff_frequency_hz=2500, gain_db=-4, q=1.2),
            Compressor(threshold_db=-35, ratio=2.5, attack_ms=30, release_ms=400),
            Chorus(rate_hz=0.25, depth=0.25, mix=0.35),
            Delay(delay_seconds=0.35, feedback=0.25, mix=0.2),
            Reverb(room_size=0.8, wet_level=0.45, dry_level=0.9),
            Gain(gain_db=1.5),
        ])

        # --- 梦境人声轨定义 ---
        dream_board2 = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=200),
            LowpassFilter(cutoff_frequency_hz=8000),
            Compressor(threshold_db=-30, ratio=2.0, attack_ms=25, release_ms=350),
            Chorus(rate_hz=0.18, depth=0.18, mix=0.25),
            Reverb(room_size=0.75, wet_level=0.5, dry_level=0.95),
        ])

        final_board = Pedalboard([
            Compressor(threshold_db=-20, ratio=1.6, attack_ms=40, release_ms=500),
            Gain(gain_db=-2.0),
            Limiter(threshold_db=-1.0),
        ])

        # 执行处理
        if mode == "single_vocal":
            processed = dream_board2(audio, sr)
        elif mode == "single_ambient":
            processed = dream_board1(audio, sr)
        else:
            return "Error: 不支持该模式。"

        final_audio = final_board(processed, sr)
        sf.write(output_path, final_audio, sr)
        return f"Success: 梦境音效已保存至 {output_path}"
        
    except Exception as e:
        return f"Error: {str(e)}"
    

# ==========================================
# 工具定义 (JSON Schema 格式)
# ==========================================
dream_tool_def = {
    "tool_name": "apply_dream_effect",
    "description": (
        "[混音工具类] 单轨梦境空间音效模拟器。用于将音频处理成超现实、虚幻且具有漂浮感的“梦境”听感，常用于表现回忆、幻觉或无意识状态。"
        "【重要限制】：本工具仅支持单条音轨处理，无法同时接收或混合两条音轨。"
        "包含两种模式：'single_vocal' (梦境人声模式，通过轻微Chorus混响使人声听起来疏离且空灵)；"
        "'single_ambient' (梦境环境模式，营造时间模糊感和音高微漂移，适合处理背景音)。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "待处理单轨音频文件的绝对路径。"
            },
            "output_path": {
                "type": "string",
                "description": "处理后音频文件需要保存的绝对路径（必须包含 .wav 后缀）。"
            },
            "mode": {
                "type": "string",
                "description": "处理模式，必须严格从 ['single_vocal', 'single_ambient'] 中选择。",
                "default": "single_vocal"
            }
        },
        "required": ["input_path", "output_path"]
    }
}