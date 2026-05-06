import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard,      # 效果链容器
    HighpassFilter,  # 空气段滤波
    LowpassFilter,   # 水下段核心滤波：切除所有高频
    Compressor,      # 模拟水压的挤压感
    Reverb,          # 模拟水下短促折射
    Gain,            # 水下由于低频堆积需要的增益补偿
    Limiter,         # 输出保护
    PeakFilter,      # 去除特定的共振点
    Chorus,          # 产生水波纹动的相位感
)

def apply_underwater_effect(
    input_path: str, 
    output_path: str,
    enter_water_time: float = 3.0,
    mode: str = "underwater_vocal"
) -> str:
    """
    [混音效果分析工具] 
    项目名称：单轨动态入水模拟器 (Single-Track Underwater Transition Simulator)
    适用场景：模拟声音从空气中瞬间进入水下的动态听感变化。脚本会自动切割时间线，入水前保持空气中的清晰自然，入水后进行强力高频切除和声学折射模拟。
    
    【重要限制】：本工具仅支持单条音轨处理。

    功能特性与调用模式：
    1. 'underwater_vocal' (水下人声模式)：默认模式。侧重于去除高频细节，模拟水中闷响且带有极小空间的折射感，适合处理人声对白。
    2. 'underwater_ambient' (水下环境模式)：低频更重，并利用 Chorus 模拟水流扰动感和物理压迫感，适合处理背景音或环境音效。

    :param input_path: 待处理单轨音频文件的绝对路径。
    :param output_path: 处理后音频文件需要保存的绝对路径（需包含 .wav 后缀）。
    :param enter_water_time: 入水的时间点（单位：秒）。工具将在此刻执行“空气 -> 水下”的音效交叉渐变切换。
    :param mode: 处理模式，请严格从 ['underwater_vocal', 'underwater_ambient'] 中选择。
        
    :return: 处理成功或失败的状态文本信息。
    """
    if isinstance(input_path, dict) and 'value' in input_path:
        input_path = input_path['value']
    if isinstance(output_path, dict) and 'value' in output_path:
        output_path = output_path['value']
    if isinstance(mode, dict) and 'value' in mode:
        mode = mode['value']
    if isinstance(enter_water_time, dict) and 'value' in enter_water_time:
        enter_water_time = enter_water_time['value']
    try:
        # 1. 读取音频并统一为单声道 1D 数组
        audio, sr = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 2. 动态构建效果链
        # --- 空气段效果链：入水前保持声音清晰自然 ---
        air_board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80),
            Compressor(threshold_db=-28, ratio=2.0, attack_ms=10, release_ms=150),
        ])

        # --- 水下段效果链：根据模式选择 ---
        if mode == "underwater_vocal":
            underwater_board = Pedalboard([
                LowpassFilter(cutoff_frequency_hz=400),
                Compressor(threshold_db=-32, ratio=4.0, attack_ms=3, release_ms=120),
                Reverb(room_size=0.15, wet_level=0.4, dry_level=0.92),
            ])
        elif mode == "underwater_ambient":
            underwater_board = Pedalboard([
                LowpassFilter(cutoff_frequency_hz=400), 
                PeakFilter(cutoff_frequency_hz=42, gain_db=-6, q=1.2),
                Compressor(threshold_db=-32, ratio=10.0, attack_ms=3, release_ms=20),
                Reverb(room_size=0.15, wet_level=0.4, dry_level=0.92),
                Chorus(rate_hz=0.3, depth=0.15, mix=0.2), # 水流扰动
                Gain(gain_db=2.0),
            ])
        else:
            return "Error: 不支持的模式，请选择 'underwater_vocal' 或 'underwater_ambient'。"

        # --- 输出保护链 ---
        final_board = Pedalboard([
            Limiter(threshold_db=-1.0)
        ])

        # 3. 动态时间切割与处理 (Timeline Split)
        total_samples = len(audio)
        enter_sample = int(enter_water_time * sr)
        
        # 默认 0.6 秒的渐变时长
        fade_dur = 0.6 
        fade_samples = int(fade_dur * sr)

        # 边界保护：如果入水时间超过音频总长，则全部当作空气段
        enter_sample = min(enter_sample, total_samples)

        # 分离原始音频
        raw_air = audio[:enter_sample]
        raw_water = audio[enter_sample:]

        # 挂载对应效果器
        air_part = air_board(raw_air, sr)
        water_part = underwater_board(raw_water, sr) if len(raw_water) > 0 else np.array([])

        # 4. 交叉渐变逻辑 (Crossfade) 防止爆音
        if fade_samples > 0 and len(air_part) >= fade_samples and len(water_part) >= fade_samples:
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in  = np.linspace(0, 1, fade_samples)
            air_part[-fade_samples:] *= fade_out
            water_part[:fade_samples] *= fade_in

        # 拼接并执行最终保护限制
        processed_mix = np.concatenate([air_part, water_part])
        final_audio = final_board(processed_mix, sr)

        # 5. 导出音频
        sf.write(output_path, final_audio, sr)
        return f"Success: 动态入水效果已成功应用 (入水点: {enter_water_time}s)，单轨文件已保存至 {output_path}"
        
    except Exception as e:
        return f"Error occurred during processing: {str(e)}"
    


# ==========================================
# 工具定义 (JSON Schema 格式)
# ==========================================
underwater_tool_def = {
    "tool_name": "apply_underwater_effect",
    "description": (
        "[混音工具类] 单轨动态入水模拟器。用于模拟声音从空气中瞬间进入水下的动态听感变化。脚本会自动切割时间线，入水前保持声音清晰，到达指定时间点后，进行强力高频切除和声学折射模拟（带有交叉渐变防爆音）。"
        "【重要限制】：本工具仅支持单条音轨处理。"
        "包含两种模式：'underwater_vocal' (水下人声模式，去除高频细节，模拟水中闷响及折射感，适合人声对白)；"
        "'underwater_ambient' (水下环境模式，低频更重，模拟水流扰动感和物理压迫感，适合处理背景音)。"
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
            "enter_water_time": {
                "type": "number",
                "description": "入水的时间点（单位：秒）。工具将在此刻执行“空气 -> 水下”的音效交叉渐变切换。默认值为 3.0。",
                "default": 3.0
            },
            "mode": {
                "type": "string",
                "description": "处理模式，必须严格从 ['underwater_vocal', 'underwater_ambient'] 中选择。",
                "default": "underwater_vocal"
            }
        },
        "required": ["input_path", "output_path"]
    }
}