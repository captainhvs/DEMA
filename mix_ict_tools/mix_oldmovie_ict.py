import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard,      # 效果链容器
    HighpassFilter,  # 高通滤波（模拟窄带）
    LowpassFilter,   # 低通滤波（滤除高频细节）
    Compressor,      # 强制压扁动态
    Reverb,          # 模拟电影院空间
    Gain,            # 增益与饱和补偿
    Limiter,         # 信号保护
    Distortion,      # 核心：模拟声轨老化失真
)

def apply_oldmovie_effect(
    input_path: str, 
    output_path: str,
    mode: str = "heavy_noise"
) -> str:
    """
    [混音工具类] 
    项目名称：单轨老电影质感模拟器 (Single-Track Old Movie & Film Hiss Simulator)
    适用场景：将现代清晰录音转化为 20 世纪早期电影、光学声轨或老式放映机的听感。包含窄频带处理、高饱和失真以及物理生成的胶片转动底噪（Hiss）。
    
    【重要限制】：本工具仅支持单条音轨输入，但会在内部自动生成放映机白噪声并与输入音轨混合输出。

    功能特性与调用模式：
    1. 'heavy_noise' (重度年代感模式)：默认模式。主体声音被大幅衰减（0.2倍），底噪被极度放大（2.0倍），营造极具年代感的“声音被淹没在噪声中”的残缺效果。适合作为纯环境音效或蒙太奇片段。
    2. 'light_noise' (清晰对白模式)：降低底噪（0.5倍）并保留较多的主体声音（0.8倍）。当用户希望在保持老电影质感的同时，依然能清晰听懂语音内容时使用。

    :param input_path: 待处理音频文件的绝对路径。
    :param output_path: 处理后音频文件需要保存的绝对路径（需包含 .wav 后缀）。
    :param mode: 处理模式，请严格从 ['heavy_noise', 'light_noise'] 中选择。
        
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
        # --- 主体效果链：窄带与高饱和 ---
        old_movie_board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=200),
            LowpassFilter(cutoff_frequency_hz=3200),
            Compressor(threshold_db=-32, ratio=3.5, attack_ms=5, release_ms=150),
            Distortion(drive_db=20),
            Gain(gain_db=-20.0),
            Reverb(room_size=0.25, wet_level=0.18, dry_level=1.0),
        ])

        # --- 底噪处理链：模拟摩擦与颗粒边缘感 ---
        noise_board = Pedalboard([
            HighpassFilter(200),
            LowpassFilter(5000),
            Distortion(drive_db=20), 
            Compressor(threshold_db=-40, ratio=6.0, attack_ms=0.5, release_ms=80),
            Gain(gain_db=-28),
        ])

        # --- 输出保护链 ---
        final_board = Pedalboard([
            Gain(gain_db=-3.0), 
            Limiter(threshold_db=-1.0) # 强制防爆音
        ])

        # 3. 📻 胶片底噪合成 (Film Hiss Generation)
        # 生成与音频等长的基础白噪声
        raw_noise = np.random.normal(0, 0.1, len(audio)).astype(np.float32)
        
        # 构建 1Hz 的低频正弦波包络，模拟老式放映机胶盘旋转的物理周期感
        t = np.arange(len(audio)) / sr
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 1.0 * t) 
        modulated_noise = raw_noise * envelope

        # 4. 执行独立处理
        processed_audio = old_movie_board(audio, sr)
        processed_noise = noise_board(modulated_noise, sr)

        # 5. 根据模式执行混合逻辑
        if mode == "heavy_noise":
            mix = 0.2 * processed_audio + 2.0 * processed_noise
        elif mode == "light_noise":
            mix = 0.8 * processed_audio + 0.5 * processed_noise
        else:
            return "Error: 不支持的模式，请选择 'heavy_noise' 或 'light_noise'。"

        # 6. 总线保护与导出
        final_audio = final_board(mix, sr)
        sf.write(output_path, final_audio, sr)
        return f"Success: 老电影质感已成功应用，带底噪的单轨文件已保存至 {output_path}"
        
    except Exception as e:
        return f"Error occurred during processing: {str(e)}"
    

# ==========================================
# 工具定义 (JSON Schema 格式)
# ==========================================
oldmovie_tool_def = {
    "tool_name": "apply_oldmovie_effect",
    "description": (
        "[混音工具类] 单轨老电影质感模拟器。将现代清晰录音转化为20世纪早期电影、光学声轨或老式放映机的听感。包含窄带处理、高饱和失真及物理生成的胶片转动底噪。"
        "【重要限制】：本工具仅支持单条音轨输入，但会在内部自动生成放映机白噪声并与输入音轨混合输出。"
        "包含两种模式：'heavy_noise' (重度年代感模式，声音被大幅衰减，底噪被极度放大，营造声音被淹没在噪声中的残缺效果)；"
        "'light_noise' (清晰对白模式，降低底噪并保留较多主体声音，保持老电影质感的同时能听懂语音内容)。"
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
                "description": "处理模式，必须严格从 ['heavy_noise', 'light_noise'] 中选择。",
                "default": "heavy_noise"
            }
        },
        "required": ["input_path", "output_path"]
    }
}