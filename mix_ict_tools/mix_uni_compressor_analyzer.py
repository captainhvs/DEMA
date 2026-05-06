import numpy as np
import soundfile as sf
import librosa

def analyze_adaptive_compression_params(input_path):
    """
    分析音频的物理特征，返回音频的听感描述、压缩器触发条件评估，
    以及推荐的 Pedalboard Compressor 参数。
    本函数仅作数据分析，不进行音频渲染和导出
    。
    :param input_path (str): 输入的待分析音频路径。
    """

    if isinstance(input_path, dict) and 'value' in input_path:
        input_path = input_path['value']

    # 1. 读取音频与降轨
    audio, samplerate = sf.read(input_path)
    mono_audio = np.mean(audio, axis=1) if len(audio.shape) > 1 else audio
        
    # 2. 统计特征计算
    rms_db = 20 * np.log10(np.sqrt(np.mean(mono_audio ** 2)) + 1e-8)
    peak_db = 20 * np.log10(np.max(np.abs(mono_audio)) + 1e-8)
    crest_factor = peak_db - rms_db
    
    centroids = librosa.feature.spectral_centroid(y=mono_audio, sr=samplerate)
    mean_centroid = np.mean(centroids)
    
    # 3. 生成音频听感特征与压缩器触发条件
    characteristics = []
    triggers = []
    need_compression = True

    # 评估动态与瞬态
    if crest_factor > 20:
        characteristics.append("极高动态范围，存在非常强的瞬态冲击（如剧烈鼓点、爆破音、突兀的尖叫或大喊）。")
        triggers.append("满足条件：瞬态过强、动态范围过大。极度需要使用压缩器来控制削波风险并压制突兀的峰值能量。")
        ratio, attack = 8.0, 3.0
    elif crest_factor > 12:
        characteristics.append("中等动态范围，有明显的音量起伏、节奏跳动或对白远近变化。")
        triggers.append("满足条件：需要提高声音稳定性、增强声音密度与存在感。适合进行标准动态控制。")
        ratio, attack = 4.0, 10.0
    else:
        characteristics.append("动态相对平稳，能量分布均匀（如铺底环境声、持续的噪音或已处理过的音频）。")
        triggers.append("触发条件较弱：动态已自然稳定。除非用于多轨总线胶合(Glue)或微调空间密度，否则建议不使用压缩器。")
        ratio, attack = 2.0, 20.0
        need_compression = False

    # 评估频域听感
    if mean_centroid > 3000:
        characteristics.append("频谱重心偏高，声音听感明亮，或包含较多高频细节/刺耳成分。")
        release = 40.0
    elif mean_centroid > 1500:
        characteristics.append("频段分布均衡。")
        release = 80.0
    else:
        characteristics.append("频谱重心偏低，声音偏沉闷、厚重，或低频能量密集。")
        release = 150.0

    # 4. 返回分析报告与参数
    return {
        "analysis_report": {
            "audio_characteristics": " ".join(characteristics),
            "compression_conditions_met": " ".join(triggers),
            "action_suggestion": "RECOMMENDED" if need_compression else "OPTIONAL_OR_SKIP"
        },
        "recommended_pedalboard_params": {
            "threshold_db": round(rms_db - 8, 2),
            "ratio": ratio,
            "attack_ms": attack,
            "release_ms": release
        }
    }