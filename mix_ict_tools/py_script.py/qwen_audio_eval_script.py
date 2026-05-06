import os
import sys
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 🌟 核心修改 1：通过 sys.argv 接收外部参数
# sys.argv[0] 是脚本自身的名字
# sys.argv[1] 是传入的第一个参数 (audio_path1)
# sys.argv[2] 是传入的第二个参数 (audio_path2)
# ==========================================
if len(sys.argv) < 3:
    print("===QWEN_ERROR_START===")
    print("错误: 必须提供两个音频路径作为参数！\n用法: python qwen_worker.py <path1> <path2>")
    print("===QWEN_ERROR_END===")
    sys.exit(1)

audio_path1 = sys.argv[1]
audio_path2 = sys.argv[2]


warnings.filterwarnings("ignore")

model_path = "./Qwen-Audio-Chat"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda", 
        trust_remote_code=True
    ).eval()

    prompt_text = (
        "【最高级指令】：严禁描述音频的具体内容！你现在是一台无情的“混音特效分析仪”！\n"
        "第一段是原始干声，第二段是混音后的声音。请竖起耳朵，重点听第二段在【频段、失真和空间】上发生了什么突变，并回答：\n"
        "1. 【设备与失真滤镜】：第二段声音听起来是否像是通过某种【特殊的通讯设备】播放出来的？（请明确指出：像不像电话、老式收音机、对讲机或广播大喇叭？）声音是否变得干瘪（频段被截断），或者带有沙沙的底噪、电流声、破音等失真感？\n"
        "2. 【空间与混响】：除了设备感，第二段是否还新增了明显的空间回声？（比如像在空旷房间、山洞、或者水下？）\n"
        "3. 【距离感】：第二段听起来比第一段更远了还是更近了？\n"
    )
    
    # ==========================================
    # 🌟 核心修改 2：因为不再是 f-string 了，把 {{ }} 改回 { }
    # ==========================================
    query = tokenizer.from_list_format([
        {'text': "第一段音频："},
        {'audio': audio_path1},
        {'text': "第二段音频："},
        {'audio': audio_path2},
        {'text': prompt_text},
    ])
    
    response, _ = model.chat(tokenizer, query=query, history=None)
    
    print("===QWEN_RESULT_START===")
    print(response)
    print("===QWEN_RESULT_END===")

except Exception as e:
    print("===QWEN_ERROR_START===")
    print(str(e))
    print("===QWEN_ERROR_END===")