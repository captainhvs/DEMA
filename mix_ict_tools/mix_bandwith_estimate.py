import numpy as np
import soundfile as sf
from typing import Tuple, Union

def estimate_bandwidth(audio_path: Union[str, dict], low_pct: Union[float, dict] = 0.05, high_pct: Union[float, dict] = 0.95) -> Tuple[float, float]:
    """
    估算音频信号的有效频率带宽 (Estimate the effective frequency bandwidth of an audio signal).
    
    通过计算音频的快速傅里叶变换 (FFT) 和累积功率谱，找到占据总能量特定分位点（百分比）的频率上下限。
    
    参数 (Parameters):
    :param audio_path (str): 输入的音频文件路径。
    :param low_pct (float): 功率谱的低频能量分位点（范围 0.0 - 1.0）。例如 0.05 表示能量累计达到总能量 5% 时的截断频率。默认值 0.05。
    :param high_pct (float): 功率谱的高频能量分位点（范围 0.0 - 1.0）。例如 0.95 表示能量累计达到总能量 95% 时的截断频率。默认值 0.95。
    
    【🤖 给大语言模型的特别提示 / Note for LLM】: 
    `low_pct` 和 `high_pct` 是以小数表示的百分比分位点。
    注意：你在调用此工具时，这两个参数**不需要是对称的**！
    你可以根据用户的具体音频分析需求，自由、不对称地设置这两个分位点。
    例如：若用户想严格排除极低频底噪，但希望保留尽可能多的高频泛音，你可以将其设置为非对称的 `low_pct=0.01` 和 `high_pct=0.98`。
    
    返回 (Returns):
    - Tuple[float, float]: 返回一个元组 `(low_cut, high_cut)`，分别代表估算的有效最低频率和最高频率（单位：Hz）。
    """
    # ---------------------------------------------------------
    # 1. 参数字典包裹校验 (防御 LLM 工具调用幻觉)
    # ---------------------------------------------------------
    if isinstance(audio_path, dict) and 'value' in audio_path:
        audio_path = audio_path['value']
    if isinstance(low_pct, dict) and 'value' in low_pct:
        low_pct = low_pct['value']
    if isinstance(high_pct, dict) and 'value' in high_pct:
        high_pct = high_pct['value']

    # 强制类型转换，防止 LLM 传入字符串形式的数字
    low_pct = float(low_pct)
    high_pct = float(high_pct)

    # ---------------------------------------------------------
    # 2. 读取音频文件
    # ---------------------------------------------------------
    audio, sr = sf.read(audio_path)
    
    # 转单声道处理以计算整体能量
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # ---------------------------------------------------------
    # 3. FFT 与累计能量计算
    # ---------------------------------------------------------
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)

    power = spec ** 2
    cumsum = np.cumsum(power)
    
    # 极低电平保护 (防止全静音音频导致的除以零或索引错误)
    if cumsum[-1] == 0:
        return 0.0, float(sr / 2.0)
        
    total = cumsum[-1]

    # ---------------------------------------------------------
    # 4. 查找分位点对应的真实频率
    # ---------------------------------------------------------
    low_cut = freqs[np.searchsorted(cumsum, total * low_pct)]
    high_cut = freqs[np.searchsorted(cumsum, total * high_pct)]

    return float(low_cut), float(high_cut)