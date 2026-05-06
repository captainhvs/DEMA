import os
import sys
import numpy as np
import torch
import laion_clap

# ==========================================
# 1. 强行接管网络环境
# ==========================================
os.environ['http_proxy'] = 'socks5h://127.0.0.1:45818'
os.environ['https_proxy'] = 'socks5h://127.0.0.1:45818'
os.environ['no_proxy'] = '*'

# ==========================================
# 2. 核心类：CLAP 文本相似度评估器
# ==========================================
class CLAPTextSimilarityEvaluator:
    def __init__(self, ckpt_path):
        # 即使只用文本，模型结构依然需要完整初始化
        self.model = laion_clap.CLAP_Module(enable_fusion=True)
        self.model.load_ckpt(ckpt_path)

    def _get_cosine_similarity(self, emb1, emb2):
        """内部计算余弦相似度"""
        emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
        emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
        return np.dot(emb1_norm, emb2_norm.T)[0][0]

    def evaluate_text_pair(self, text1, text2):
        """提取两段文本的特征并计算相似度"""
        # CLAP 支持批量提取，这里分别提取两段文本
        emb1 = self.model.get_text_embedding([text1], use_tensor=False)
        emb2 = self.model.get_text_embedding([text2], use_tensor=False)
        
        return self._get_cosine_similarity(emb1, emb2)

# ==========================================
# 3. 主程序入口：解析参数并执行
# ==========================================
if __name__ == "__main__":
    try:
        # 接收外部通过 subprocess 传进来的两个文本参数
        if len(sys.argv) < 4:
            raise ValueError("参数不足。需要提供: 目标文本描述, Qwen评价文本")
        origin_text = sys.argv[1]
        target_text = sys.argv[2]
        qwen_eval_text = sys.argv[3]

        CKPT_PATH = './ckpt/630k-audioset-fusion-best.pt'
        
        # 初始化模型并计算
        evaluator = CLAPTextSimilarityEvaluator(ckpt_path=CKPT_PATH)
        org_sim_score = evaluator.evaluate_text_pair(origin_text, target_text)
        qwen_sim_score = evaluator.evaluate_text_pair(target_text, qwen_eval_text)

        # 组织输出字符串
        result_text = (
            f"混音客观描述与原始音频描述相似度: （{org_sim_score :.4f}）\n"
            f"混音客观描述与 Qwen描述相似度: （{qwen_sim_score:.4f}）\n"
            f"🎯 原始音频描述: '{origin_text}'\n"
            f"🎯 混音客观描述: '{target_text}'\n"
            f"🤖 Qwen评价: '{qwen_eval_text}'"
        )

        # 使用严格的分隔符输出，供上层 Wrapper 解析
        print("===CLAP_TEXT_SIM_START===")
        print(result_text)
        print("===CLAP_TEXT_SIM_END===")

    except Exception as e:
        # 捕获错误并用错误分隔符输出
        print("===CLAP_TEXT_SIM_ERROR_START===")
        print(str(e))
        print("===CLAP_TEXT_SIM_ERROR_END===")