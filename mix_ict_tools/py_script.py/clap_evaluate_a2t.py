import os
import sys
import numpy as np
import librosa
import torch
import laion_clap

# ==========================================
# 1. 强行接管网络环境
# ==========================================
os.environ['http_proxy'] = 'socks5h://127.0.0.1:45818'
os.environ['https_proxy'] = 'socks5h://127.0.0.1:45818'
os.environ['no_proxy'] = '*'

# ==========================================
# 2. 辅助函数与核心类
# ==========================================
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

class CLAPSimilarityEvaluator:
    def __init__(self, ckpt_path):
        self.model = laion_clap.CLAP_Module(enable_fusion=True)
        self.model.load_ckpt(ckpt_path)

    def _get_cosine_similarity(self, audio_emb, text_emb):
        audio_emb_norm = audio_emb / np.linalg.norm(audio_emb, axis=1, keepdims=True)
        text_emb_norm = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        return np.dot(audio_emb_norm, text_emb_norm.T)[0][0]

    def evaluate_single(self, audio_path, text):
        audio_data, _ = librosa.load(audio_path, sr=48000)
        audio_data = audio_data.reshape(1, -1)
        processed_audio_data = int16_to_float32(float32_to_int16(audio_data))
        
        audio_emb = self.model.get_audio_embedding_from_data(x=processed_audio_data, use_tensor=False)
        text_emb = self.model.get_text_embedding([text], use_tensor=False)
        
        return self._get_cosine_similarity(audio_emb, text_emb)

# ==========================================
# 3. 主程序入口：解析参数并执行
# ==========================================
if __name__ == "__main__":
    try:
        # 获取外部通过 subprocess 传进来的参数
        if len(sys.argv) < 5:
            raise ValueError("参数不足。需要提供: 原始音频路径, 原始文本, 混音音频路径, 混音文本")

        orig_audio_path = sys.argv[1]
        orig_text_desc = sys.argv[2]
        mixed_audio_path = sys.argv[3]
        mixed_text_desc = sys.argv[4]

        CKPT_PATH = './ckpt/630k-audioset-fusion-best.pt'
        
        # 初始化模型并计算
        evaluator = CLAPSimilarityEvaluator(ckpt_path=CKPT_PATH)
        stan_score = evaluator.evaluate_single(orig_audio_path, orig_text_desc)
        mixed_score = evaluator.evaluate_single(mixed_audio_path, mixed_text_desc)
        orig_score = evaluator.evaluate_single(orig_audio_path, mixed_text_desc)


        # 组织输出字符串
        result_text = (
            f"原始音频与原始描述文本匹配度 ({stan_score:.4f})\n"
            f"原始音频与混音描述文本匹配度 ({orig_score:.4f})\n"
            f"混音音频与混音描述文本匹配度 ({mixed_score:.4f})\n"
        )

        # 使用严格的分隔符输出，供上层 Wrapper 解析
        print("===CLAP_RESULT_START===")
        print(result_text)
        print("===CLAP_RESULT_END===")

    except Exception as e:
        # 捕获错误并用错误分隔符输出
        print("===CLAP_ERROR_START===")
        print(str(e))
        print("===CLAP_ERROR_END===")