import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard, HighpassFilter, LowpassFilter, Compressor, 
    Reverb, Gain, Limiter, PeakFilter, Delay
)

def apply_cabin_effect(input_path: str, output_path: str, mode: str = "single_vocal") -> str:
    """
    [混音工具类] 
    项目名称：单轨车载空间音效模拟器 (Single-Track Cabin Simulator)
    适用场景：模拟声音在轿车内部密闭空间的听感。通过特定的 EQ 曲线、极短延迟和较小的空间混响，重现狭窄金属/玻璃腔体内的声学特性。
    
    【重要限制】：本工具仅支持单条音轨处理，无法同时接收或混合两条音轨。如需制作“带背景音的车载录音”，请分次调用处理。

    功能特性与调用模式：
    1. 'single_vocal' (人声模式)：默认模式。适用于用户仅提供了一段人声录音，希望听起来像是在车里说话。处理时会强调中频临场感与近距离反射。
    2. 'single_ambient' (环境模式)：适用于一段纯背景音（如音乐或杂音），希望听起来像是车载音响播放或车内底噪。处理时会有低频堆积和高低频切除。

    :param input_path: 待处理音频文件的绝对路径。
    :param output_path: 处理后音频文件需要保存的绝对路径（需包含 .wav 后缀）。
    :param mode: 处理模式，请严格从 ['single_vocal', 'single_ambient'] 中选择。
        
    :return: 处理成功或失败的状态文本信息。
    """
    if isinstance(input_path, dict) and 'value' in input_path:
        input_path = input_path['value']
    if isinstance(output_path, dict) and 'value' in output_path:
        output_path = output_path['value']
        
    # --- 预设定义 (保持你原有的参数) ---
    cabin_board1 = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=90), LowpassFilter(cutoff_frequency_hz=3800),
        PeakFilter(cutoff_frequency_hz=180, gain_db=2.5, q=1.0),
        PeakFilter(cutoff_frequency_hz=4500, gain_db=-6.0, q=0.9),
        Compressor(threshold_db=-32, ratio=3.0, attack_ms=20, release_ms=250),
        Delay(delay_seconds=0.06, feedback=0.15, mix=0.18),
        Reverb(room_size=0.35, wet_level=0.18, dry_level=1.0),
        Gain(gain_db=-1.0),
    ])

    cabin_board2 = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=110), LowpassFilter(cutoff_frequency_hz=4000),
        PeakFilter(cutoff_frequency_hz=1400, gain_db=2.5, q=1.2),
        Compressor(threshold_db=-28, ratio=3.5, attack_ms=10, release_ms=220),
        Delay(delay_seconds=0.045, feedback=0.10, mix=0.12),
        Reverb(room_size=0.30, wet_level=0.15, dry_level=1.0),
        Gain(gain_db=1.0),
    ])

    final_board = Pedalboard([
        Compressor(threshold_db=-18, ratio=2.2, attack_ms=25, release_ms=400),
        Gain(gain_db=-1.5), Limiter(threshold_db=-1.0),
    ])

    try:
        audio, sr = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if mode == "single_vocal":
            processed = cabin_board2(audio, sr)
        elif mode == "single_ambient":
            processed = cabin_board1(audio, sr)
        else:
            return "Error: 不支持的模式。"

        final_audio = final_board(processed, sr)
        sf.write(output_path, final_audio, sr)
        return f"Success: 车载音效应用成功，保存至 {output_path}"
    except Exception as e:
        return f"Error: {str(e)}"
    


# ==========================================
# 工具定义 (JSON Schema 格式)
# ==========================================
cabin_tool_def = {
    "tool_name": "apply_cabin_effect",
    "description": (
        "[混音工具类] 单轨车载空间音效模拟器。用于模拟声音在轿车内部密闭金属/玻璃腔体内的听感。"
        "【重要限制】：本工具仅支持单条音轨处理，无法同时接收或混合两条音轨。"
        "包含两种模式：'single_vocal' (人声模式，强调中频临场感与近距离反射，适合车内说话声)；"
        "'single_ambient' (环境模式，产生低频堆积和高低频切除，适合模拟车载音响播放的音乐或车内底噪)。"
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