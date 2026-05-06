import os
import subprocess
import textwrap

def analyze_audio_scene(audio_path: str) -> str:
    """
    [混音分析工具]
    调用 Qwen 视觉/听觉大模型分析音频文件，获得对输入路径下音频的描述。
    （已强制在独立的 QwenAudio 虚拟环境中执行）

    输入参数:
    :param audio_path: 待描述音频文件的绝对路径。

    输出：
    final_response:对音频的描述
    """
    # ==========================================
    # 🌟 核心：强制指定 Qwen 环境的 Python 绝对路径
    # ==========================================
    QWEN_PYTHON_ENV = "/opt/anaconda3/envs/QwenAudio/bin/python"
    if isinstance(audio_path, dict) and 'value' in audio_path:
        audio_path = audio_path['value']

    # 1. 检查文件和环境
    if not os.path.exists(audio_path):
        return f"Error: 找不到音频文件，路径不正确: {audio_path}"
    if not os.path.exists(QWEN_PYTHON_ENV):
        return f"Error: 找不到指定的 Python 解释器: {QWEN_PYTHON_ENV}"

    print(f"\n[🔧 工具执行] 正在跨环境唤醒 Qwen-Audio 分析: {audio_path}")
    print(f"            使用的解释器: {QWEN_PYTHON_ENV}")

    # ==========================================
    # 2. 将要在 Qwen 环境中执行的代码写成字符串
    # ==========================================
    # 注意：这里的代码是在另一个 Python 进程里跑的
    script_code = textwrap.dedent(f"""
            import os
            # 铁律：设定显卡必须在所有深度学习库之前！
            os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
            
            import sys
            import torch
            import warnings
            from transformers import AutoModelForCausalLM, AutoTokenizer

            warnings.filterwarnings("ignore")

            audio_path = r"{audio_path}"
            model_path = "/data/liangzhechun/Qwen-Audio/Qwen_model/Qwen-Audio-Chat"

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    device_map="cuda", 
                    trust_remote_code=True
                ).eval()

                # 注意内部所有缩进必须严格对齐
                prompt_text = (
                    "你现在是一位敏锐的音效体验专家。请尽量忽略音频里的具体说话内容或旋律，"
                    "把注意力集中在【声音自带的特效质感和环境氛围】上，用直白、通俗的语言描述以下3点：\\n"
                    "1. 【空间与回声】：声音听起来是在什么样的地方发出的？（例如：开阔的户外、空旷的大厅、山洞、还是狭小封闭的房间？）有没有明显的回音或空灵感？\\n"
                    "2. 【距离与清晰度】：声音感觉是贴着耳朵说的，还是离得很远？音质是清晰明亮的，还是听起来发闷、隔着东西、或者有些单薄？\\n"
                    "3. 【特殊设备与滤镜感】：声音有没有带上某种“特殊设备”的感觉？（例如：听起来像不像是在通电话的嘈杂、老电影的底噪、或者像在水下？）有没有明显的杂音或失真？\\n"
                    "最后，综合以上听感，用一句话总结这段音频最像是在什么样的【具体物理场景】中录制的。"
                )
                
                query = tokenizer.from_list_format([
                    {{'audio': audio_path}},
                    {{'text': prompt_text}},
                ])
                
                response, _ = model.chat(tokenizer, query=query, history=None)
                
                print("===QWEN_RESULT_START===")
                print(response)
                print("===QWEN_RESULT_END===")

            except Exception as e:
                print("===QWEN_ERROR_START===")
                print(str(e))
                print("===QWEN_ERROR_END===")
        """).strip()

    # ==========================================
    # 3. 强制使用指定的 Python 执行这段代码
    # ==========================================
    try:
        # 执行命令，并捕获输出
        result = subprocess.run(
            [QWEN_PYTHON_ENV, "-c", script_code],
            capture_output=True,
            text=True,
            check=False
        )
        
        stdout = result.stdout
        
        # 4. 精准解析输出结果
        if "===QWEN_RESULT_START===" in stdout and "===QWEN_RESULT_END===" in stdout:
            # 提取成功的结果
            final_response = stdout.split("===QWEN_RESULT_START===")[1].split("===QWEN_RESULT_END===")[0].strip()
            print("[🔧 工具执行] ✅ 音频分析完成！")
            return final_response
            
        elif "===QWEN_ERROR_START===" in stdout:
            # 提取报错信息
            error_msg = stdout.split("===QWEN_ERROR_START===")[1].split("===QWEN_ERROR_END===")[0].strip()
            return f"Qwen 模型执行内部报错: {error_msg}"
            
        else:
            # 防止模型输出了预料之外的格式
            return f"Qwen 分析失败，未知输出。\\n标准错误: {result.stderr}\\n标准输出: {stdout}"

    except Exception as e:
        return f"跨环境调用失败: {str(e)}"