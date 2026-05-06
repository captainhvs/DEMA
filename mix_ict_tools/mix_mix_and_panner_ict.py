import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Limiter
from typing import List, Union


def apply_two_track_mix(
    track1_path: str,
    track2_path: str,
    output_path: str,
    track1_gain_db: float = 0.0,
    track2_gain_db: float = 0.0,
    track1_pan_mode: str = "center",
    track2_pan_mode: str = "center"
) -> str:
    """
    [混音工具类] 
    项目名称：双轨混音与声像控制器 (Two-Track Mixer & Panner)
    适用场景：将两条独立的音轨合并为一条立体声音轨。支持独立调节每条音轨的音量增益（Gain）。以及针对每一轨调节极其灵活的立体声声像（Pan）定位，包括中央、左声道、右声道
    左右摇摆、环绕声。
    
    【重要限制】：两轨音频的采样率必须一致。输出固定为立体声（Stereo）格式。

    声像模式 (Pan Mode) 详解：
    1. 'center' (居中)：默认模式。声音在正前方，双耳音量均等。适合主干人声或核心对白。
    2. 'left' (全左)：声音仅在左声道出现。适合特殊的方位暗示。
    3. 'right' (全右)：声音仅在右声道出现。
    4. 'swing' (左右摇摆)：声音在左耳和右耳之间以 0.5Hz 的频率来回游走，产生强烈的眩晕感或游离感（常用于梦境、醉酒或特殊效果）。
    5. 'surround' (动态环绕/加宽)：利用哈斯效应（Haas Effect）和相位翻转，将单调的声音强制拉宽到耳朵两侧，产生包围感。极其适合处理环境音、底噪或雨声，将其作为宽广的背景垫在人声后面。

    :param track1_path: str 第一条音轨的绝对路径。
    :param track2_path: str 第二条音轨的绝对路径。
    :param output_path: str 混音后文件保存的绝对路径（需包含 .wav 后缀）。
    :param track1_gain_db: float 第一轨的音量增益（单位：dB）。正数放大，负数衰减，控制第一轨声音的大小。
    :param track2_gain_db: float 第二轨的音量增益（单位：dB）。正数放大，负数衰减，控制第二轨声音的大小。
    :param track1_pan_mode: str 第一轨声像，从 ['center', 'left', 'right', 'swing', 'surround'] 中选择。
    :param track2_pan_mode: str 第二轨声像，从 ['center', 'left', 'right', 'swing', 'surround'] 中选择。
        
    :return: 处理成功或失败的状态文本信息。
    """
    if isinstance(track1_path, dict) and 'value' in track1_path:
        track1_path = track1_path['value']
    if isinstance(track2_path, dict) and 'value' in track2_path:
        track2_path = track2_path['value']
    if isinstance(output_path, dict) and 'value' in output_path:
        output_path = output_path['value']
    if isinstance(track1_gain_db, dict) and 'value' in track1_gain_db:
        track1_gain_db = track1_gain_db['value']
    if isinstance(track2_gain_db, dict) and 'value' in track2_gain_db:
        track2_gain_db = track2_gain_db['value']
    if isinstance(track1_pan_mode, dict) and 'value' in track1_pan_mode:
        track1_pan_mode = track1_pan_mode['value']  
    if isinstance(track2_pan_mode, dict) and 'value' in track2_pan_mode:
        track2_pan_mode = track2_pan_mode['value']  


    try:
        # 1. 读取音频文件
        audio1, sr1 = sf.read(track1_path)
        audio2, sr2 = sf.read(track2_path)

        if sr1 != sr2:
            return f"Error: 采样率不匹配 (Track1: {sr1}Hz, Track2: {sr2}Hz)，无法直接混音。"

        # 2. 统一化处理：将输入音频全部转为立体声格式 (N, 2)
        def to_stereo(audio):
            if len(audio.shape) == 1:
                return np.column_stack((audio, audio))
            elif audio.shape[1] > 2:
                return audio[:, :2]
            return audio

        audio1 = to_stereo(audio1)
        audio2 = to_stereo(audio2)

        # 3. 对齐时间轴（以最长的轨道为基准，短的补静音）
        max_len = max(len(audio1), len(audio2))
        audio1 = np.pad(audio1, ((0, max_len - len(audio1)), (0, 0)), mode='constant')
        audio2 = np.pad(audio2, ((0, max_len - len(audio2)), (0, 0)), mode='constant')

        # 4. 应用音量增益 (Gain)
        audio1 *= (10 ** (track1_gain_db / 20.0))
        audio2 *= (10 ** (track2_gain_db / 20.0))

        # 5. 定义声像处理算法 (Constant Power Panning)
        def apply_pan(audio, mode, sr):
            N = len(audio)
            t = np.arange(N) / sr
            
            # 为了实现纯粹的声像控制，先将轨道降混为单声道作为控制源
            mono_source = np.mean(audio, axis=1)
            
            if mode == "surround":
                # 环绕模式：利用 Haas 效应延迟右声道 15ms，并进行部分反相，产生强烈的两侧包围感
                delay_samples = int(0.015 * sr)
                surround_audio = np.zeros((N, 2), dtype=np.float32)
                surround_audio[:, 0] = mono_source # 左耳正常
                # 右耳延迟并轻微反相
                surround_audio[delay_samples:, 1] = mono_source[:-delay_samples] * -0.8
                return surround_audio

            # 计算恒定功率的分配角度 (Theta: 0 为全左, Pi/2 为全右, Pi/4 为正中)
            if mode == "left":
                theta = np.full(N, 0.0)
            elif mode == "right":
                theta = np.full(N, np.pi / 2.0)
            elif mode == "swing":
                # 左右摇摆：使用 0.5Hz 的正弦波 LFO 控制角度
                # np.sin 范围是 -1 到 1，映射到 0 到 Pi/2
                theta = (np.pi / 4.0) + (np.pi / 4.0) * np.sin(2 * np.pi * 0.5 * t)
            else: # center 默认
                theta = np.full(N, np.pi / 4.0)

            # 通过 sin/cos 曲线分配左右声道电平，确保总功率在移动时不塌陷
            L_mult = np.cos(theta)
            R_mult = np.sin(theta)
            
            return np.column_stack((mono_source * L_mult, mono_source * R_mult))

        # 6. 对两轨分别应用声像
        audio1_panned = apply_pan(audio1, track1_pan_mode, sr1)
        audio2_panned = apply_pan(audio2, track2_pan_mode, sr1)

        # 7. 执行物理相加混音
        mixed_audio = audio1_panned + audio2_panned

        # 8. 挂载总线保护限制器（防爆音）
        board = Pedalboard([Limiter(threshold_db=-0.5)])
        final_audio = board(mixed_audio, sr1)

        # 9. 导出结果
        sf.write(output_path, final_audio, sr1)
        return f"Success: 双轨混音成功！Track1({track1_pan_mode}), Track2({track2_pan_mode})，文件保存至 {output_path}"

    except Exception as e:
        return f"Error occurred during mixing: {str(e)}"
    


def apply_multi_track_mix(
    track_paths: List[str],
    output_path: str,
    track_gains_db: List[float] = None,
    track_pan_modes: List[str] = None
) -> str:
    """
    [混音工具类] 
    项目名称：多轨混音与声像控制器 (Multi-Track Mixer & Panner)
    适用场景：将任意数量的独立音轨合并为一条立体声音轨。支持独立调节每条音轨的音量增益（Gain）。
    以及针对每一轨调节极其灵活的立体声声像（Pan）定位，包括中央、左声道、右声道、左右摇摆、环绕声。
    
    【重要限制】：所有音轨的采样率必须与第一条音轨一致。输出固定为立体声（Stereo）格式。

    声像模式 (Pan Mode) 详解：
    1. 'center' (居中)：默认模式。声音在正前方，双耳音量均等。适合主干人声或核心对白。
    2. 'left' (全左)：声音仅在左声道出现。适合特殊的方位暗示。
    3. 'right' (全右)：声音仅在右声道出现。
    4. 'swing' (左右摇摆)：声音在左耳和右耳之间以 0.5Hz 的频率来回游走，产生强烈的眩晕感或游离感（常用于梦境、醉酒或特殊效果）。
    5. 'surround' (动态环绕/加宽)：利用哈斯效应（Haas Effect）和相位翻转，将单调的声音强制拉宽到耳朵两侧，产生包围感。极其适合处理环境音、底噪或雨声，将其作为宽广的背景垫在人声后面。

    :param track_paths: List[str] 所有音轨绝对路径的列表。
    :param output_path: str 混音后文件保存的绝对路径（需包含 .wav 后缀）。
    :param track_gains_db: List[float] 每一轨对应的音量增益（单位：dB）。正数放大，负数衰减。
    :param track_pan_modes: List[str] 每一轨对应的声像模式，选填 ['center', 'left', 'right', 'swing', 'surround']。
        
    :return: 处理成功或失败的状态文本信息。
    """
    # 智能处理默认参数
    if track_gains_db is None:
        track_gains_db = []
    if track_pan_modes is None:
        track_pan_modes = []

    # 兼容 Agent 可能传进来的 dict 包装格式
    if isinstance(track_paths, dict) and 'value' in track_paths:
        track_paths = track_paths['value']
    if isinstance(output_path, dict) and 'value' in output_path:
        output_path = output_path['value']
    if isinstance(track_gains_db, dict) and 'value' in track_gains_db:
        track_gains_db = track_gains_db['value']
    if isinstance(track_pan_modes, dict) and 'value' in track_pan_modes:
        track_pan_modes = track_pan_modes['value']

    # 如果列表内部元素是 dict 包装的，进行解包
    track_paths = [p['value'] if isinstance(p, dict) and 'value' in p else p for p in track_paths]
    track_gains_db = [g['value'] if isinstance(g, dict) and 'value' in g else g for g in track_gains_db]
    track_pan_modes = [m['value'] if isinstance(m, dict) and 'value' in m else m for m in track_pan_modes]

    # 自动补全不足的参数列表，防止越界
    while len(track_gains_db) < len(track_paths):
        track_gains_db.append(0.0)
    while len(track_pan_modes) < len(track_paths):
        track_pan_modes.append("center")

    if len(track_paths) == 0:
        return "Error: 传入的音轨列表为空，无法进行混音。"

    try:
        loaded_audios = []
        base_sr = None
        max_len = 0

        # 1. 循环读取所有音频文件，并检查采样率
        for path in track_paths:
            audio, sr = sf.read(path)
            if base_sr is None:
                base_sr = sr  # 以第一轨的采样率作为基准
            elif sr != base_sr:
                return f"Error: 采样率不匹配。基准为 {base_sr}Hz，但 {path} 为 {sr}Hz，无法直接混音。"

            loaded_audios.append(audio)
            if len(audio) > max_len:
                max_len = len(audio)

        # 2. 统一化处理：将输入音频全部转为立体声格式 (N, 2)
        def to_stereo(audio):
            if len(audio.shape) == 1:
                return np.column_stack((audio, audio))
            elif audio.shape[1] > 2:
                return audio[:, :2]
            return audio

        # 3. 定义声像处理算法 (Constant Power Panning)
        def apply_pan(audio, mode, sr):
            N = len(audio)
            t = np.arange(N) / sr
            
            # 为了实现纯粹的声像控制，先将轨道降混为单声道作为控制源
            mono_source = np.mean(audio, axis=1)
            
            if mode == "surround":
                # 环绕模式：利用 Haas 效应延迟右声道 15ms，并进行部分反相，产生强烈的两侧包围感
                delay_samples = int(0.015 * sr)
                surround_audio = np.zeros((N, 2), dtype=np.float32)
                surround_audio[:, 0] = mono_source # 左耳正常
                # 右耳延迟并轻微反相
                surround_audio[delay_samples:, 1] = mono_source[:-delay_samples] * -0.8
                return surround_audio

            # 计算恒定功率的分配角度 (Theta: 0 为全左, Pi/2 为全右, Pi/4 为正中)
            if mode == "left":
                theta = np.full(N, 0.0)
            elif mode == "right":
                theta = np.full(N, np.pi / 2.0)
            elif mode == "swing":
                # 左右摇摆：使用 0.5Hz 的正弦波 LFO 控制角度
                theta = (np.pi / 4.0) + (np.pi / 4.0) * np.sin(2 * np.pi * 0.5 * t)
            else: # center 默认
                theta = np.full(N, np.pi / 4.0)

            # 通过 sin/cos 曲线分配左右声道电平，确保总功率在移动时不塌陷
            L_mult = np.cos(theta)
            R_mult = np.sin(theta)
            
            return np.column_stack((mono_source * L_mult, mono_source * R_mult))

        # 初始化最终混音容器
        mixed_audio = np.zeros((max_len, 2), dtype=np.float32)

        # 4. 循环处理每一条轨道：立体声转换 -> 补齐长度 -> 增益 -> 声像 -> 累加混音
        for i, audio in enumerate(loaded_audios):
            stereo_audio = to_stereo(audio)
            
            # 对齐时间轴（不足最长长度的补静音）
            padded_audio = np.pad(stereo_audio, ((0, max_len - len(stereo_audio)), (0, 0)), mode='constant')
            
            # 应用音量增益 (Gain)
            padded_audio *= (10 ** (track_gains_db[i] / 20.0))
            
            # 应用声像模式
            panned_audio = apply_pan(padded_audio, track_pan_modes[i], base_sr)
            
            # 执行物理相加混音
            mixed_audio += panned_audio

        # 5. 挂载总线保护限制器（防爆音）
        board = Pedalboard([Limiter(threshold_db=-0.5)])
        final_audio = board(mixed_audio, base_sr)

        # 6. 导出结果
        sf.write(output_path, final_audio, base_sr)
        
        # 组装成功信息
        pan_info = ", ".join([f"Track{i+1}({track_pan_modes[i]})" for i in range(len(track_paths))])
        return f"Success: 多轨混音成功！共 {len(track_paths)} 轨，状态：[{pan_info}]，文件保存至 {output_path}"

    except Exception as e:
        return f"Error occurred during mixing: {str(e)}"


# ==========================================
# 工具定义 (JSON Schema 格式)
# ==========================================
two_track_mix_tool_def = {
    "tool_name": "apply_two_track_mix",
    "description": (
        "[混音工具类] 双轨混音与声像控制器。这是处理多轨音频的最终合成工具，用于将两条独立的音轨合并为一条立体声音轨。"
        "支持独立调节每条音轨的音量增益（Gain）和极度灵活的立体声声像（Pan）定位。"
        "【重要限制】：两轨音频的采样率必须一致。输出固定为立体声（Stereo）格式。"
        "声像模式(pan_mode)包括：'center'(居中，适合主干人声/对白)；'left'(全左)；'right'(全右)；"
        "'swing'(左右摇摆，产生眩晕/游离感)；'surround'(动态环绕加宽，极度适合处理环境音、底噪将其作为宽广背景)。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "track1_path": {
                "type": "string",
                "description": "第一条音轨的绝对路径。"
            },
            "track2_path": {
                "type": "string",
                "description": "第二条音轨的绝对路径。"
            },
            "output_path": {
                "type": "string",
                "description": "混音后文件保存的绝对路径（必须包含 .wav 后缀）。"
            },
            "track1_gain_db": {
                "type": "number",
                "description": "第一轨的音量增益（单位：dB）。正数放大，负数衰减。默认值为 0.0。",
                "default": 0.0
            },
            "track2_gain_db": {
                "type": "number",
                "description": "第二轨的音量增益（单位：dB）。正数放大，负数衰减。默认值为 0.0。",
                "default": 0.0
            },
            "track1_pan_mode": {
                "type": "string",
                "description": "第一轨声像模式，必须严格从 ['center', 'left', 'right', 'swing', 'surround'] 中选择。",
                "default": "center"
            },
            "track2_pan_mode": {
                "type": "string",
                "description": "第二轨声像模式，必须严格从 ['center', 'left', 'right', 'swing', 'surround'] 中选择。",
                "default": "center"
            }
        },
        "required": ["track1_path", "track2_path", "output_path"]
    }
}