import os
import subprocess
import textwrap

def analyze_audio_scene_contrast(audio_path1: str,audio_path2: str) -> str:
    """
    [多轨混音比较分析工具]
    调用 Qwen 视觉/听觉大模型分析音频文件，获得对输入路径下两个音频的比较描述，其中第一个为原始音频，第二个为混音音频。
    （已强制在独立的 QwenAudio 虚拟环境中执行）

    输入参数:
    :param audio_path1: 原始音频文件的绝对路径。
    :param audio_path2: 待比较的混音音频文件的绝对路径。

    输出：
    final_response:对音频的描述
    """
    QWEN_PYTHON_ENV = "/opt/anaconda3/envs/QwenAudio/bin/python"
    WORKER_SCRIPT = "/data/liangzhechun/llama-stack/mix_ict_tools/py_script.py/qwen_audio_eval_script.py" 
    custom_env = os.environ.copy()
    custom_env["CUDA_VISIBLE_DEVICES"] = "2,3"

    
    # 兼容字典解包
    if isinstance(audio_path1, dict): audio_path1 = audio_path1.get('value', audio_path1)
    if isinstance(audio_path2, dict): audio_path2 = audio_path2.get('value', audio_path2)

    if not os.path.exists(audio_path1) or not os.path.exists(audio_path2):
        return "Error: 找不到输入的音频文件，请检查路径。"
    if not os.path.exists(QWEN_PYTHON_ENV):
        return f"Error: 找不到 Python 解释器: {QWEN_PYTHON_ENV}"
    if not os.path.exists(WORKER_SCRIPT):
        return f"Error: 找不到 Qwen 执行脚本: {WORKER_SCRIPT}"

    print(f"\n[🔧 工具执行] 正在跨环境唤醒 Qwen-Audio 分析...")
    
    try:
        # ==========================================
        # 🌟 核心传参魔法：直接把参数放在列表中依次传入
        # 相当于在终端输入: python qwen_worker.py path1 path2
        # ==========================================
        result = subprocess.run(
            [QWEN_PYTHON_ENV, WORKER_SCRIPT, audio_path1, audio_path2], 
            capture_output=True,
            text=True,
            check=False,
            env=custom_env
        )
        
        stdout = result.stdout
        
        # 解析输出
        if "===QWEN_RESULT_START===" in stdout:
            return stdout.split("===QWEN_RESULT_START===")[1].split("===QWEN_RESULT_END===")[0].strip()
        elif "===QWEN_ERROR_START===" in stdout:
            error_msg = stdout.split("===QWEN_ERROR_START===")[1].split("===QWEN_ERROR_END===")[0].strip()
            return f"Qwen 内部报错: {error_msg}"
        else:
            return f"分析失败，未知输出。\n标准错误: {result.stderr}\n标准输出: {stdout}"

    except Exception as e:
        return f"跨环境调用失败: {str(e)}"