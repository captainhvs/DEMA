import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard,       # 效果链容器
    HighpassFilter,   # 高通滤波（电话窄带核心）
    LowpassFilter,    # 低通滤波（窄带核心）
    Compressor,       # 模拟 AGC 自动增益控制
    Gain,             # 失真前后的电平匹配
    Limiter,          # 输出安全阈值
    Distortion        # 核心：产生电子通信的破碎感
)

def apply_telephone_effect(
    input_path: str, 
    output_path: str,
    mode: str = "telephone_vocal"
) -> str:
    """
    [混音工具类] 
    项目名称：单轨电话/无线电通信模拟器 (Single-Track Telephone & Radio Simulator)
    适用场景：模拟通过老式电话、对讲机或窄带无线电进行通信的听感。其核心在于极窄的频带限制（Bandpass）和高强度的非线性失真，产生“电子染色”明显的破碎音质。
    
    【重要限制】：本工具仅支持单条音轨处理，无法同时接收或混合两条音轨。

    功能特性与调用模式：
    1. 'telephone_vocal' (对讲机/电话人声模式)：默认模式。频带保留在 350Hz-3200Hz，在产生强烈失真（50dB Drive）和通信感的同时，尽量保证语音内容的可懂度。适用于模拟主视角的无线电通话或电话交流。
    2. 'radio_ambient' (故障/远端无线电模式)：频带被极度压缩至 300Hz-2800Hz，且整体输出电平极低（适合作为背景音）。产生极度残缺、信号不佳的电子对讲机或战场通讯环境音。

    :param input_path: 待处理单轨音频文件的绝对路径。
    :param output_path: 处理后音频文件需要保存的绝对路径（需包含 .wav 后缀）。
    :param mode: 处理模式，请严格从 ['telephone_vocal', 'radio_ambient'] 中选择。
        
    :return: 处理成功或失败的状态文本信息。
    """
    if isinstance(input_path, dict) and 'value' in input_path:
        input_path = input_path['value']
    if isinstance(output_path, dict) and 'value' in output_path:
        output_path = output_path['value']
    if isinstance(mode, dict) and 'value' in mode:
        mode = mode['value']
    try:
        # 1. 读取音频并统一为单声道
        audio, sr = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 2. 动态构建效果链
        if mode == "telephone_vocal":
            # --- 人声效果链：侧重中频突出与可懂度 ---
            process_board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=350),
                LowpassFilter(cutoff_frequency_hz=3200),
                Gain(gain_db=-10), 
                Distortion(drive_db=50),
                Compressor(
                    threshold_db=-16, ratio=5.0, attack_ms=3, release_ms=120
                ),
                Gain(gain_db=-30), # 补偿 50dB 失真带来的音量爆炸
                Limiter(threshold_db=-1.0),
            ])
            # 输出前的人声补偿系数 (对应原代码中的 1.2 倍 mix 权重)
            mix_weight = 1.2
            
        elif mode == "radio_ambient":
            # --- 环境效果链：侧重极度压缩与带宽剥离 ---
            process_board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=300),
                LowpassFilter(cutoff_frequency_hz=2800),
                Gain(gain_db=-12),
                Distortion(drive_db=50),
                Compressor(
                    threshold_db=-18, ratio=6.0, attack_ms=5, release_ms=150
                ),
                Gain(gain_db=-40), # 极度衰减，模拟远端微弱信号
                Limiter(threshold_db=-1.0),
            ])
            mix_weight = 1.0
            
        else:
            return "Error: 不支持的模式，请选择 'telephone_vocal' 或 'radio_ambient'。"

        # 3. 总线保护
        final_board = Pedalboard([
            Limiter(threshold_db=-1.0)
        ])

        # 4. 执行处理
        processed_audio = process_board(audio, sr)
        
        # 应用权重并过总线保护
        final_audio = final_board(processed_audio * mix_weight, sr)

        # 5. 导出音频
        sf.write(output_path, final_audio, sr)
        return f"Success: 电话/无线电通信效果已成功应用，单轨文件已保存至 {output_path}"
        
    except Exception as e:
        return f"Error occurred during processing: {str(e)}"
    


# ==========================================
# 工具定义 (JSON Schema 格式)
# ==========================================
telephone_tool_def = {
    "tool_name": "apply_telephone_effect",
    "description": (
        "[混音工具类] 单轨电话/无线电通信模拟器。用于模拟通过老式电话、对讲机或窄带无线电进行通信的听感。产生带有极窄频带限制和高强度非线性失真的“电子破碎”音质。"
        "【重要限制】：本工具仅支持单条音轨处理，无法混合音频。"
        "包含两种模式：'telephone_vocal' (对讲机/电话人声模式，在产生强烈失真和通信感的同时，尽量保证语音内容的可懂度)；"
        "'radio_ambient' (故障/远端无线电模式，频带极度压缩且输出电平极低，产生极度残缺、信号不佳的战场通讯环境音或背景音)。"
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
                "description": "处理模式，必须严格从 ['telephone_vocal', 'radio_ambient'] 中选择。",
                "default": "telephone_vocal"
            }
        },
        "required": ["input_path", "output_path"]
    }
}