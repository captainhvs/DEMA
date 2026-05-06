import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard,      # 效果链容器
    HighpassFilter,  # 高通滤波（模拟户外空气特性）
    LowpassFilter,   # 低通滤波（模拟长距离高频损耗）
    Compressor,      # 轻度动态控制
    Reverb,          # 极大型稀疏空间感
    Gain,            # 距离感补偿
    Limiter,         # 输出保护
    Delay,           # 核心：产生旷野物理回声
)

def apply_openfield_effect(
    input_path: str, 
    output_path: str,
    mode: str = "far_mountain"
) -> str:
    """
    [混音工具类] 
    项目名称：单轨户外旷野回声模拟器 (Single-Track Open-Field Echo Simulator)
    适用场景：模拟声音在极度开阔的户外环境（如山谷、旷野、大型操场）中的听感。其核心特征是具有明显的物理回声（Echo）而非密集的室内混响，且伴随随距离产生的高频自然衰减。非常适合需要“喊话”或“远距离感”的场景。
    
    【重要限制】：本工具仅支持单条音轨处理，无法混合音频。

    功能特性与调用模式：
    1. 'far_mountain' (远距离山谷模式)：默认模式。物理回声的延迟时间较长（0.4秒），模拟声波遇到极远处山体或建筑物的清晰反射，空间感极度广阔。
    2. 'close_field' (近距离旷野/操场模式)：物理回声的延迟时间较短（0.15秒），模拟相对开阔但边界较近的户外场地，如大型操场或空旷的街道。

    :param input_path: 待处理单轨音频文件的绝对路径。
    :param output_path: 处理后音频文件需要保存的绝对路径（需包含 .wav 后缀）。
    :param mode: 处理模式，请严格从 ['far_mountain', 'close_field'] 中选择。
        
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

        # 2. 根据模式动态设定物理回声的延迟时间（模拟反射物的距离）
        if mode == "far_mountain":
            delay_time = 0.40  # 400ms，极远的山谷回声
            delay_mix = 0.45   # 回声更明显
        elif mode == "close_field":
            delay_time = 0.15  # 150ms，较近的操场回声
            delay_mix = 0.30
        else:
            return "Error: 不支持的模式，请选择 'far_mountain' 或 'close_field'。"

        # 3. 动态构建效果链
        openfield_board = Pedalboard([
            # 模拟空气吸收：户外低频容易消散，高频随距离衰减
            HighpassFilter(cutoff_frequency_hz=60),    
            LowpassFilter(cutoff_frequency_hz=3500),   
            
            # 自然动态保留：极轻压缩
            Compressor(threshold_db=-30, ratio=1.6, attack_ms=25, release_ms=200),

            # 核心：物理回声 (Discrete Echoes)
            Delay(
                delay_seconds=delay_time,  # 动态延迟时间
                feedback=0.3,              # 适度的反馈，使回声逐渐消失
                mix=delay_mix              # 动态混合比
            ),

            # 极大型空间建模
            Reverb(
                room_size=0.95,            # 模拟极大的物理空间
                wet_level=0.28,            # 低干湿比，防止声音浑浊，保持旷野通透
                dry_level=1.0,  
            ),
        ])

        # 4. 输出保护与二次拉远
        final_board = Pedalboard([
            Gain(gain_db=-2.0),            # 降低总音量，进一步拉开听觉距离
            Limiter(threshold_db=-1.0),
        ])

        # 5. 执行处理
        processed_audio = openfield_board(audio, sr)
        final_audio = final_board(processed_audio, sr)

        # 6. 导出音频
        sf.write(output_path, final_audio, sr)
        return f"Success: 旷野回声效果已成功应用，单轨文件已保存至 {output_path}"
        
    except Exception as e:
        return f"Error occurred during processing: {str(e)}"
    

# ==========================================
# 工具定义 (JSON Schema 格式)
# ==========================================
openfield_tool_def = {
    "tool_name": "apply_openfield_effect",
    "description": (
        "[混音工具类] 单轨户外旷野回声模拟器。模拟声音在极度开阔的户外环境（如山谷、旷野、大型操场）中的听感，特征是明显的物理回声（Echo）与高频自然衰减，极度适合“喊话”或“远距离感”场景。"
        "【重要限制】：本工具仅支持单条音轨处理，无法混合音频。"
        "包含两种模式：'far_mountain' (远距离山谷模式，延迟较长，空间感极度广阔)；"
        "'close_field' (近距离旷野/操场模式，延迟较短，模拟相对开阔但边界较近的户外场地或空旷街道)。"
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
                "description": "处理模式，必须严格从 ['far_mountain', 'close_field'] 中选择。",
                "default": "far_mountain"
            }
        },
        "required": ["input_path", "output_path"]
    }
}