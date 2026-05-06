import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, Compressor, Reverb, Gain, Limiter, Distortion



def apply_broadcast_effect(input_path: str, output_path: str, mode: str = "single_vocal") -> str:
    """
    [混音工具类] 
    项目名称：单轨广播级扩声模拟器 (Single-Track Broadcast Simulator)
    适用场景：对单个音频文件模拟广播、扩声系统或老式无线电的播放听感。特点是带有明显的非线性失真、频段限制和较大的空间混响。
    
    【重要限制】：本工具仅支持单条音轨处理，无法同时接收或混合两条音轨。如需制作“带背景音的广播”，请分次调用处理，或由其他方式进行混音。

    功能特性与调用模式：
    1. 'single_vocal' (人声播报模式)：默认模式。模拟公共广播系统（PA System）或车站扩声。通过中频增强与适度失真，突出人声清晰度并附带较大空间混响。适用于纯人声干音。
    2. 'single_ambient' (环境/极度失真模式)：模拟老式电台或故障广播。通过极高失真度营造强烈的颗粒感与破音感。适用于需要特殊效果的声音。

    :param input_path: 待处理音频文件的绝对路径。
    :param output_path: 处理后音频文件的保存绝对路径（需包含 .wav 后缀）。
    :param mode: 处理模式，请严格从 ['single_vocal', 'single_ambient'] 中选择。
    """

    if isinstance(input_path, dict) and 'value' in input_path:
        input_path = input_path['value']
    if isinstance(output_path, dict) and 'value' in output_path:
        output_path = output_path['value']


    # ... 你的 Pedalboard 定义部分 (保持不变) ...
    broadcast_board1 = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=220), LowpassFilter(cutoff_frequency_hz=5200),
        Gain(gain_db=10), Distortion(drive_db=40),
        Compressor(threshold_db=-20, ratio=4.0, attack_ms=8, release_ms=180),
        Reverb(room_size=0.8, wet_level=0.5, dry_level=1.0),
        Gain(gain_db=-32), Limiter(threshold_db=-1.0),
    ])
    # ... 其他 board 定义 ...
    final_board = Pedalboard([
        Compressor(threshold_db=-16, ratio=2.0, attack_ms=20, release_ms=300),
        Limiter(threshold_db=-1.0),
    ])

    try:
        audio, sr = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        if mode == "single_vocal":
            # 这里为了演示简化，假设你用 board1
            final_audio = final_board(broadcast_board1(audio, sr), sr)
        else:
            final_audio = final_board(broadcast_board1(audio, sr), sr)

        sf.write(output_path, final_audio, sr)
        return f"Success: 文件已保存至 {output_path}"
    except Exception as e:
        return f"Error: {str(e)}"



# ==========================================
# 工具定义 (JSON Schema 格式)
# ==========================================
broadcast_tool_def = {
    "tool_name": "apply_broadcast_effect",
    "description": (
        "[混音工具类] 单轨广播级扩声模拟器。用于模拟公共广播、扩声系统或老式无线电的播放听感。"
        "特点是带有明显的非线性失真、频段限制和较大的空间混响。"
        "【重要限制】：本工具仅支持单条音轨处理，无法同时接收或混合两条音轨。"
        "包含两种模式：'single_vocal' (人声播报模式，突出人声清晰度)；'single_ambient' (环境/极度失真模式，营造强烈颗粒感与破音感)。"
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
