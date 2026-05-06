import os
import subprocess

def evaluate_audio_text_similarity(orig_audio: str, orig_text: str, mixed_audio: str, mixed_text: str) -> str:
    """
    [音频-文本匹配度分析工具]
    调用本地 CLAP 模型分析原始音频与混音音频，分别计算它们与对应文本描述的余弦相似度（匹配度）。
    （已强制在独立的 clap 虚拟环境中执行）

    输入参数:
    :param orig_audio: 原始音频文件的绝对路径。
    :param orig_text: 原始音频对应的文本描述。
    :param mixed_audio: 混音音频文件的绝对路径。
    :param mixed_text: 给定混音音频对应的文本描述。

    输出：
    final_response: 包含原始音频和混音音频匹配度的字符串。
    """
    # 强制指定 Conda 环境的 Python 解释器
    CLAP_PYTHON_ENV = "/opt/anaconda3/envs/clap/bin/python"
    
    # ⚠️ 请将此路径修改为你实际存放下方 worker 脚本的绝对路径
    WORKER_SCRIPT = "./mix_ict_tools/py_script.py/clap_evaluate_a2t.py" 
    
    # 兼容字典解包 (防止大模型传参时套了一层 dict)
    if isinstance(orig_audio, dict): orig_audio = orig_audio.get('value', orig_audio)
    if isinstance(orig_text, dict): orig_text = orig_text.get('value', orig_text)
    if isinstance(mixed_audio, dict): mixed_audio = mixed_audio.get('value', mixed_audio)
    if isinstance(mixed_text, dict): mixed_text = mixed_text.get('value', mixed_text)
    
    # 路径与环境检查
    if not os.path.exists(orig_audio) or not os.path.exists(mixed_audio):
        return "Error: 找不到输入的音频文件，请检查路径。"
    if not os.path.exists(CLAP_PYTHON_ENV):
        return f"Error: 找不到 Python 解释器: {CLAP_PYTHON_ENV}"
    if not os.path.exists(WORKER_SCRIPT):
        return f"Error: 找不到 CLAP 执行脚本: {WORKER_SCRIPT}"

    print(f"\n[🔧 工具执行] 正在跨环境唤醒 CLAP 模型计算匹配度...")
    
    try:
        # ==========================================
        # 核心传参：把路径和文本通过 CLI 参数依次传入
        # ==========================================
        result = subprocess.run(
            [CLAP_PYTHON_ENV, WORKER_SCRIPT, orig_audio, orig_text, mixed_audio, mixed_text], 
            capture_output=True,
            text=True,
            check=False
        )
        
        stdout = result.stdout
        
        # 解析标准格式输出
        if "===CLAP_RESULT_START===" in stdout:
            return stdout.split("===CLAP_RESULT_START===")[1].split("===CLAP_RESULT_END===")[0].strip()
        elif "===CLAP_ERROR_START===" in stdout:
            error_msg = stdout.split("===CLAP_ERROR_START===")[1].split("===CLAP_ERROR_END===")[0].strip()
            return f"CLAP 内部报错: {error_msg}"
        else:
            return f"分析失败，未知输出。\n标准错误: {result.stderr}\n标准输出: {stdout}"

    except Exception as e:
        return f"跨环境调用失败: {str(e)}"