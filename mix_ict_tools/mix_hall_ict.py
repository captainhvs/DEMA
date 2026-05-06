import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard,      # 效果链容器
    HighpassFilter,  # 高通滤波
    LowpassFilter,   # 低通滤波
    Compressor,      # 动态压缩
    Reverb,          # 空间混响
    Gain,            # 增益控制
    Limiter,         # 输出限制保护
)

def apply_hall_effect(
    input_path: str, 
    output_path: str,
    mode: str = "bright_hall"
) -> str:
    """
    [混音工具类] 
    项目名称：单轨大厅空间音效模拟器 (Single-Track Hall Simulator)
    适用场景：模拟声音在宽敞、开阔的室内环境（如音乐厅、车站大厅、大型会议室）中的听感。主要特点是长残响、高空间感。
    
    【重要限制】：本工具仅支持单条音轨处理，无法混合音频。

    功能特性与调用模式：
    1. 'bright_hall' (明亮大厅模式)：默认模式。低通滤波设为 4000Hz，保留较多高频反射，适合模拟音乐厅、空旷的车站大厅等墙面反射较硬的空间，声音明亮、开阔。
    2. 'muffled_hall' (沉闷/隔墙模式)：低通滤波设为 200Hz，切除大量中高频，适合模拟在极度吸音的宽大空间内，或者隔着厚墙听到的大厅回音，声音极度闷响。

    :param input_path: 待处理单轨音频文件的绝对路径。
    :param output_path: 处理后音频文件需要保存的绝对路径（需包含 .wav 后缀）。
    :param mode: 处理模式，请严格从 ['bright_hall', 'muffled_hall'] 中选择。
        
    :return: 处理成功或失败的状态文本信息。
    """
    if isinstance(input_path, dict) and 'value' in input_path:
        input_path = input_path['value']
    if isinstance(output_path, dict) and 'value' in output_path:
        output_path = output_path['value']
    try:
        # 1. 读取音频并转为单声道（若为立体声）
        audio, sr = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 2. 根据大模型传入的模式，动态设置低通滤波的截止频率
        if mode == "bright_hall":
            lp_cutoff = 4000.0
        elif mode == "muffled_hall":
            lp_cutoff = 200.0
        else:
            return "Error: 不支持的模式，请选择 'bright_hall' 或 'muffled_hall'。"

        # 3. 动态构建效果链（放置在函数内部以应用动态参数）
        hall_board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=30),       # 极低切，保留大厅的厚度
            LowpassFilter(cutoff_frequency_hz=lp_cutoff), # 动态低通，控制明暗度
            Compressor(
                threshold_db=-24, 
                ratio=2.0, 
                attack_ms=10, 
                release_ms=120
            ),
            Reverb(
                room_size=0.8,    # 核心参数：0.8 代表大型室内空间（Hall）
                wet_level=0.5,    # 50% 混响比，提供明显的空间深度
                dry_level=1.0, 
            ),
        ])

        # 总线保护效果链
        final_board = Pedalboard([
            Gain(gain_db=-1.5),         # 预留 1.5dB 裕量防止失真
            Limiter(threshold_db=-1.0), # 强制锁定最高电平
        ])

        # 4. 执行单轨处理逻辑
        processed = hall_board(audio, sr)
        final_audio = final_board(processed, sr)

        # 5. 导出处理后的音频
        sf.write(output_path, final_audio, sr)
        return f"Success: 大厅空间音效已成功应用，单轨文件已保存至 {output_path}"
        
    except Exception as e:
        return f"Error occurred during processing: {str(e)}"
    
# ==========================================
# 工具定义 (JSON Schema 格式)
# ==========================================
hall_tool_def = {
    "tool_name": "apply_hall_effect",
    "description": (
        "[混音工具类] 单轨大厅空间音效模拟器。用于模拟声音在宽敞、开阔的室内环境（如音乐厅、车站大厅、大型会议室）中的听感，特点是长残响、高空间感。"
        "【重要限制】：本工具仅支持单条音轨处理，无法混合音频。"
        "包含两种模式：'bright_hall' (明亮大厅模式，保留高频反射，声音明亮开阔，适合音乐厅或空旷大厅)；"
        "'muffled_hall' (沉闷/隔墙模式，切除大量中高频，适合模拟在极度吸音空间内或隔着厚墙听到的大厅回音，声音极度闷响)。"
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
                "description": "处理模式，必须严格从 ['bright_hall', 'muffled_hall'] 中选择。",
                "default": "bright_hall"
            }
        },
        "required": ["input_path", "output_path"]
    }
}