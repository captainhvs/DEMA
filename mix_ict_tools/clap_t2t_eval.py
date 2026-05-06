import os
import subprocess

def evaluate_qwen_text_similarity(origin_text: str,target_text: str, qwen_eval_text: str) -> str:
    """
    [文本语义相似度分析工具 (基于 CLAP 空间)]
    调用本地 CLAP 模型，计算“混音音频的目标文本描述”与“Qwen-Audio 对该音频的评价文本”之间的余弦相似度。
    （已强制在独立的 clap 虚拟环境中执行）

    输入参数:
    :param target_text: 设定的目标混音文本描述。
    :param qwen_eval_text: Qwen-Audio 生成的评价文本。

    输出：
    final_response: 包含两者相似度得分的字符串。
    """
    # 强制指定 Conda 环境的 Python 解释器
    CLAP_PYTHON_ENV = "/opt/anaconda3/envs/clap/bin/python"
    
    # ⚠️ 请将此路径修改为你实际存放下方 worker 脚本的绝对路径
    WORKER_SCRIPT = "./mix_ict_tools/py_script.py/clap_evaluate_t2t.py" 
    
    # 兼容字典解包 (防止大模型传参时套了一层 dict)
    if isinstance(origin_text, dict): origin_text = origin_text.get('value', origin_text)
    if isinstance(target_text, dict): target_text = target_text.get('value', target_text)
    if isinstance(qwen_eval_text, dict): qwen_eval_text = qwen_eval_text.get('value', qwen_eval_text)

    # 环境检查
    if not os.path.exists(CLAP_PYTHON_ENV):
        return f"Error: 找不到 Python 解释器: {CLAP_PYTHON_ENV}"
    if not os.path.exists(WORKER_SCRIPT):
        return f"Error: 找不到 CLAP 文本相似度执行脚本: {WORKER_SCRIPT}"

    print(f"\n[🔧 工具执行] 正在跨环境唤醒 CLAP 计算文本相似度...")
    
    try:
        # ==========================================
        # 核心传参：将两段文本通过 CLI 参数传入
        # ==========================================
        result = subprocess.run(
            [CLAP_PYTHON_ENV, WORKER_SCRIPT,origin_text, target_text, qwen_eval_text], 
            capture_output=True,
            text=True,
            check=False
        )
        
        stdout = result.stdout
        
        # 解析标准格式输出
        if "===CLAP_TEXT_SIM_START===" in stdout:
            return stdout.split("===CLAP_TEXT_SIM_START===")[1].split("===CLAP_TEXT_SIM_END===")[0].strip()
        elif "===CLAP_TEXT_SIM_ERROR_START===" in stdout:
            error_msg = stdout.split("===CLAP_TEXT_SIM_ERROR_START===")[1].split("===CLAP_TEXT_SIM_ERROR_END===")[0].strip()
            return f"CLAP 内部报错: {error_msg}"
        else:
            return f"分析失败，未知输出。\n标准错误: {result.stderr}\n标准输出: {stdout}"

    except Exception as e:
        return f"跨环境调用失败: {str(e)}"