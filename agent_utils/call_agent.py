import os
import re
import subprocess
import textwrap
import traceback
import shutil
import datetime
from mix_ict_tools.qwen_audio_ict import analyze_audio_scene
from mix_ict_tools.qwen_audio_eval_ict import analyze_audio_scene_contrast
from mix_ict_tools.clap_a2t_eval import evaluate_audio_text_similarity
from mix_ict_tools.clap_t2t_eval import evaluate_qwen_text_similarity
from agent_utils.summary_utils import summarize_execution_case
from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient

def generate_base_mixing_script(script_path, input_audio_path, output_audio_path):
    """
    在指定路径下生成一份基础的混音脚本。
    包含固定的库导入、写死的输入输出路径，以及留给 Agent 编排效果器的区域。
    """
    
    # 构造要写入的 Python 脚本内容
    # 注意：使用 textwrap.dedent 可以保持代码美观，内部的路径变量会被格式化替换进去
    script_content = textwrap.dedent(f"""
        import numpy as np
        import soundfile as sf
        from pedalboard import (
            Pedalboard, HighpassFilter, LowpassFilter, 
            Compressor, Distortion, Gain, Delay, Reverb
        )

        # 1. 设置路径 (Agent 自动填入)
        input_path = r"{input_audio_path}"
        output_path = r"{output_audio_path}"

        print("-> 1. 读取原始音频并统一为单声道...")
        audio, samplerate = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 2. 初始化效果链
        plugins = []
        print("-> 2. 装载效果器参数...")

        # ==========================================
        # <<<AGENT_INJECT_START>>>
        # 此处由 Agent 顶格写入逻辑，拒绝任何前导空格
        need_highpass = False
        if need_highpass:
            plugins.append(HighpassFilter(cutoff_frequency_hz=100.0))

        need_lowpass = False
        if need_lowpass:
            plugins.append(LowpassFilter(cutoff_frequency_hz=10000.0))

        need_compressor = False
        if need_compressor:
            plugins.append(Compressor(threshold_db=-15.0, ratio=3.0, attack_ms=5.0, release_ms=50.0))

        need_distortion = False
        if need_distortion:
            plugins.append(Distortion(drive_db=0.0))
            plugins.append(Gain(gain_db=0.0)) 

        need_delay = False
        if need_delay:
            plugins.append(Delay(delay_seconds=0.5, feedback=0.2, mix=0.2))

        need_reverb = False
        if need_reverb:
            plugins.append(Reverb(room_size=0.5, wet_level=0.3, dry_level=0.8))
        # <<<AGENT_INJECT_END>>>
        # ==========================================

        print("-> 3. 渲染处理音频...")
        board = Pedalboard(plugins)
        effected_audio = board(audio, samplerate)

        print("-> 4. 导出并写入指定路径...")
        sf.write(output_path, effected_audio, samplerate)
        print("✅ 混音处理完成！")
    """).strip()

    # 确保保存路径的文件夹存在
    os.makedirs(os.path.dirname(os.path.abspath(script_path)), exist_ok=True)
    
    # 写入文件
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        print(f"✅ 基础模板脚本已成功生成至: {script_path}")
        return True
    except Exception as e:
        print(f"❌ 生成基础模板脚本失败: {e}")
        return False


def call_agent(agent,prompt,session_id,use_stream=True):
    
    print("🤖 Agent 正在思考与执行...\n")


    print(f"--- 打印执行日志 ---")
    print(f"输入{prompt}")
    # 使用同一个 session_id 请求大模型
    response = agent.create_turn(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id, 
        stream=use_stream,
    )

    # Only call `AgentEventLogger().log(response)` for streaming responses.
    if use_stream:
        for log in AgentEventLogger().log(response):
            if hasattr(log, 'print'):
                log.print()
            else:
                # Print text chunks inline without newlines
                print(log, end='', flush=True)
        print()  # Final newline at the end
    else:
        print(response.output_text)
    
    print(f"\n--- 轮次结束 ---")

    return response





def extract_python_code(text: str) -> str:
    """
    从大模型输出的文本中提取 Python 代码段。
    
    参数:
        text (str): 包含 Markdown 格式代码块的原始字符串
        
    返回:
        str: 提取出的纯净 Python 代码。如果未找到代码块，则返回去除首尾空白的原文本。
    """
    if not text:
        return ""

    # 优先匹配带 python 标识的代码块，re.DOTALL 使得 '.' 可以匹配换行符
    pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)
    match = pattern.search(text)
    
    if match:
        return match.group(1).strip()
    
    # 容错处理：大模型有时会漏掉 'python' 标识，退化匹配普通的 ``` ... ```
    fallback_pattern = re.compile(r'```\s*(.*?)\s*```', re.DOTALL)
    fallback_match = fallback_pattern.search(text)
    
    if fallback_match:
        return fallback_match.group(1).strip()
        
    # 如果没有找到任何 Markdown 代码块标记，假设输入文本直接是代码
    return text.strip()
  



def call_agent_with_evolution(
    agent,
    client,
    summary_agent,
    summary_session_id,
    debug_vector_store_uuid,
    file_folder,
    initial_prompt,
    prompt_analysis,
    session_id,
    exec_python_script_path,
    input_audio_path,
    output_audio_path,
    use_stream=True,
    max_retries=5,
    timeout=120,
):
    """
    调用 Agent + 本地脚本执行 + 自动纠错（写入.py再执行）+ 执行后案例总结
    """
    current_prompt = initial_prompt
    accumulated_errors = "" 
    accumulated_errors_analysis = ""
    code_str = "" # 初始化防止提取代码失败时变量未定义
    
    for attempt in range(max_retries):
        print(f"\n🤖 Agent 正在思考与执行... (第 {attempt + 1}/{max_retries} 次尝试)")
        print(f"--- 打印执行日志 ---")
        print(f"输入: {current_prompt}")
        generate_base_mixing_script(exec_python_script_path, input_audio_path, output_audio_path)

        # 1️⃣ 调 Agent (假设外部环境已实现 agent.create_turn)
        response = agent.create_turn(
            messages=[{"role": "user", "content": current_prompt}],
            session_id=session_id,
            stream=use_stream,
        )

        # 2️⃣ 收集输出
        output_text = ""
        if use_stream:
            # 假设外部环境已实现 AgentEventLogger
            for log in AgentEventLogger().log(response):
                if hasattr(log, 'print'):
                    log.print()
                else:
                    output_text += str(log)
                    print(log, end='', flush=True)
            print()
        else:
            output_text = response.output_text
            print(output_text)

        print(f"--- Agent 回复结束 ---")

        # 3️⃣ 提取代码 (假设外部环境已实现 extract_python_code)
        extracted = extract_python_code(output_text)
        if extracted:
            code_str = extracted

        if not extracted:
            print("❌ 未检测到有效 Python 代码")
            current_prompt = "请严格输出 ```python ... ``` 代码块，不要包含解释。"
            error_traceback = "未检测到有效 Python 代码，流程中断。"
            accumulated_errors += f"\n【第 {attempt + 1} 次尝试报错】:\n{error_traceback}\n"
            continue



        # 4️⃣ 注入脚本文件 
        print(f"\n📝 正在将 Agent 编排代码注入到脚本: {exec_python_script_path}")
        try:
            # 1. 确保模板脚本已存在
            if not os.path.exists(exec_python_script_path):
                print(f"❌ 找不到基础模板文件: {exec_python_script_path}，请先执行生成模板函数！")
                return False
                
            # 2. 读取原始基础脚本
            with open(exec_python_script_path, "r", encoding="utf-8") as f:
                original_script = f.read()
                
            # 3. 对 Agent 生成的代码进行缩进 (匹配函数内部的 12 个空格缩进)
            # code_str 是你在步骤 3 提取出来的纯代码字符串
            indented_agent_code = code_str.strip()
            
            # 4. 使用正则精确替换标记块之间的内容
            # \1 捕获开始标记行，\2 捕获结束标记行。中间的内容将被彻底替换为 Agent 的新代码
            pattern = r"(# <<<AGENT_INJECT_START>>>\n).*?(\n\s*# <<<AGENT_INJECT_END>>>)"
            replacement = r"\1" + indented_agent_code + r"\2"
            
            new_script, count = re.subn(pattern, replacement, original_script, flags=re.DOTALL)
            
            if count == 0:
                print("❌ 脚本注入失败：未在目标脚本中找到 <<<AGENT_INJECT_START>>> 等标记符。")
                return False
            
            # 5. 写回文件
            with open(exec_python_script_path, "w", encoding="utf-8") as f:
                f.write(new_script)
                
            print("✅ 代码片段注入成功！")
            
        except Exception as e:
            print(f"❌ 脚本读取/写入异常: {e}")
            return False




 



        # 5️⃣ 执行脚本
        print("\n▶️ 开始执行脚本...")
        try:
            result = subprocess.run(
                ["python", exec_python_script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            stdout = result.stdout
            stderr = result.stderr
            returncode = result.returncode

            print("----- STDOUT -----")
            print(stdout)

            if returncode == 0:
                print("✅ 代码执行成功！")
                
                # ==========================================
                # 根据尝试次数，进入不同的成功总结分支
                # ==========================================
                if attempt == 0:
                    print("🟢 一次性成功")

                    


                else:
                    print("🟡 触发 [Debug成功] 总结逻辑...")
                    summarize_execution_case(
                        summary_agent=summary_agent,
                        summary_session_id=summary_session_id,
                        file_folder=file_folder,
                        initial_prompt=initial_prompt,
                        is_success=True,
                        attempt_count=attempt + 1,
                        query_for_rag=query_for_rag,
                        error_history=accumulated_errors,
                        error_analysis=accumulated_errors_analysis
                    )
                return True

            print("----- STDERR -----")
            print(stderr)

            clean_stderr = re.sub(r'[\U00010000-\U0010ffff]', '', stderr) if stderr else ""
            clean_stdout = re.sub(r'[\U00010000-\U0010ffff]', '', stdout) if stdout else ""

            error_traceback = textwrap.dedent(f"""
                Return Code: {returncode}

                STDERR:
                {clean_stderr}

                STDOUT:
                {clean_stdout}
            """).strip()

        except subprocess.TimeoutExpired as e:
            print("⏰ 执行超时")
            error_traceback = f"TimeoutExpired: {str(e)}"

        except Exception as e:
            print(f"⚠️ 执行异常: {e}")
            error_traceback = traceback.format_exc()

        # 记录报错历史
        accumulated_errors += f"\n【第 {attempt + 1} 次尝试报错】:\n{error_traceback}\n"


        print("\n📚 正在使用结果分析")
        dynamic_rag_prompt = textwrap.dedent(f"""
            【任务：诊断代码错误并检索修复方案】
            刚才执行的 Python 脚本发生了如下报错，导致流程中断：
            ```text
            {error_traceback}
            ```
            按照以下步骤执行错误分析：
            {prompt_analysis}
        """).strip()

        analysis = call_agent(agent, dynamic_rag_prompt, session_id, use_stream=False).output_text 
        accumulated_errors_analysis += f"\n【第 {attempt + 1} 次尝试错误分析】:\n{analysis}\n"

        prompt_for_query = f"根据以下错误分析，检索相关的修复方案和参考文档：\n{analysis},只保留报错提示中提到的错误类型（如参数单位不匹配错误、 功能配置不当错误、数据格式或类型不匹配等）和错误关键词，去掉无关的描述性语言,最长不超过20个词。"
        prompt_for_query = re.sub(r'[\U00010000-\U0010ffff]', '',  prompt_for_query)  # 清理可能的特殊字符
        query_for_rag = call_agent(summary_agent, prompt_for_query, session_id, use_stream=False).output_text 
        # ==========================================
        # 使用 analysis 手动检索 RAG 库
        # ==========================================
        
        print("\n🔍 触发 RAG 查询和错误分析...")
        try:
            search_results = client.vector_stores.search(
                vector_store_id=debug_vector_store_uuid,  # 假设 vs (Vector Store) 和 client 已在外部定义
                query=query_for_rag,         # 用错误分析结果去检索最相关的文档
                max_num_results=5
            )


            # 提取上下文内容
            parsed_texts = []
            # 1. 遍历每个检索结果 (Data 对象)
            for r in search_results.data:
                if not getattr(r, 'content', None):
                    continue
                    
                # 2. 遍历 content 列表里的每个块 (DataContent 对象)
                for chunk in r.content:
                    if getattr(chunk, 'text', None):
                        parsed_texts.append(chunk.text)
            
            # 3. 干净利落地拼接纯文本
            rag_context = "\n\n".join(parsed_texts)


            query = f"根据以下错误分析，结合检索到的 RAG 参考文档，提供针对性的代码修复建议：\n错误分析:\n{analysis}\nRAG 参考文档:\n{rag_context}"
            completion = client.chat.completions.create(
                model="ollama/llama3.3:latest",
                messages=[
                    {"role": "system", "content": "Use the provided context to answer questions."},
                    {"role": "user", "content": query}
                ]
            )
            rag_completion=completion.choices[0].message.content

        except Exception as e:
            print(f"⚠️ RAG 检索失败: {e}")
            rag_context = "未能获取到有效的参考文档。"



        # 7️⃣ 构造下一轮 prompt (将 RAG 上下文加入提示词)
        current_prompt = textwrap.dedent(f"""
            你刚刚生成的 Python 代码运行失败。
            请再次参考最初的要求，根据报错信息,使用下方提供的参考文档和错误分析来针对性修复代码，并重新输出代码块。

            【特别注意】严禁调用任何工具！
            
            原始要求如下：
            {initial_prompt}


            # AI 错误分析结果：
            # {analysis}

            RAG 检索到的参考文档 (Context)：
            {rag_completion}
        """).strip()

    
    return False



def call_agent_with_script_refinement(
    client,
    agent,
    session_id,
    refine_vector_store_uuid,
    input_script_path,
    output_script_path,
    modification_instructions="",
    use_stream=True,
    timeout=120
):
    """
    单次调用 Agent，仅对标记区域做小侵入式提取和修改。
    【已新增 RAG 机制】：执行前会根据 instruction 检索历史黄金调参经验。
    """
    
    # 1️⃣ 读取输入脚本，并进行【小侵入式提取】
    if not os.path.exists(input_script_path):
        print(f"❌ 找不到输入文件: {input_script_path}")
        return False
        
    try:
        with open(input_script_path, "r", encoding="utf-8") as f:
            original_script = f.read()
            
        # 仅匹配标记块内部的代码
        match = re.search(r"(# <<<AGENT_INJECT_START>>>\n)(.*?)(\n\s*# <<<AGENT_INJECT_END>>>)", original_script, flags=re.DOTALL)
        if not match:
            print(f"❌ 在 {input_script_path} 中找不到 <<<AGENT_INJECT_START>>> 标记，无法进行局部微调！")
            return False
            
        start_marker = match.group(1)
        target_snippet = match.group(2)
        end_marker = match.group(3)
        
    except Exception as e:
        print(f"❌ 读取或解析输入文件失败: {e}")
        return False

    # ==========================================
    # 2️⃣ 触发 RAG 查询：拿着 Instruction 去“藏经阁”翻找经验
    # ==========================================
    rag_completion = ""
    if modification_instructions and refine_vector_store_uuid:
        print(f"\n🔍 触发 RAG 检索，正在匹配历史调参经验...")
        try:
            search_results = client.vector_stores.search(
                vector_store_id=refine_vector_store_uuid, 
                query=modification_instructions, # 直接使用微调指令作为检索词
                max_num_results=5
            )

            # 提取上下文内容（坚固的解包逻辑）
            parsed_texts = []
            for r in search_results.data:
                if not getattr(r, 'content', None):
                    continue
                for chunk in r.content:
                    if getattr(chunk, 'text', None):
                        parsed_texts.append(chunk.text)
            
            rag_context = "\n\n".join(parsed_texts)

            if rag_context.strip():
                # 用轻量级 LLM 对查出来的历史记录进行“经验提炼”
                query = f"根据当前的调参指令，结合检索到的 RAG 历史黄金案例，提取出最相关的参数修改建议和数值参考：\n当前指令:\n{modification_instructions}\n历史 RAG 案例:\n{rag_context}"
                completion = client.chat.completions.create(
                    model="ollama/llama3.3:latest",
                    messages=[
                        {"role": "system", "content": "You are an expert audio mixing consultant. Extract useful parameter tuning advice from the context."},
                        {"role": "user", "content": query}
                    ]
                )
                rag_completion = completion.choices[0].message.content
                print("    📚 成功提取 RAG 调参经验，已注入外脑！")
            else:
                print("    ⚠️ RAG 检索结果为空，无相关经验。")

        except Exception as e:
            print(f"    ⚠️ RAG 检索失败 (退回无经验模式): {e}")


    # 3️⃣ 构造严格的 Prompt（喂入局部代码 + RAG 经验）
    current_prompt = textwrap.dedent(f"""
        【任务目标】
        请审查下方提供的 Python 局部代码片段，并{f'根据以下要求："{modification_instructions}"，' if modification_instructions else ''}对代码进行修改。
        
        【RAG 历史参考经验】（非常重要，请优先参考其中的数值设定）：
        {rag_completion if rag_completion else "无历史经验参考，请严格依据你的声学和算法知识进行盲调。"}
        
        【待处理的局部代码区域】：
        ```python
        {target_snippet.strip()}
        ```

        【严格红线要求】
        1. 你**只允许修改**上述代码中函数的参数数字（数值）或变量赋的值。
        2. 绝对**不允许**修改原有的逻辑架构、增删函数、改变变量名或添加新的处理模块。
        3. 仅输出修改后的完整局部代码，必须包裹在 ```python ... ``` 代码块中。
        4. 不要包含任何解释性文字、Markdown 废话，也不要输出 <<<AGENT_INJECT_START>>> 等标记。
    """).strip()

    print(f"\n🤖 Agent 正在思考微调参数...")

    # 4️⃣ 调 Agent (单次请求)
    try:
        response = agent.create_turn(
            messages=[{"role": "user", "content": current_prompt}],
            session_id=session_id,
            stream=use_stream,
        )
    except Exception as e:
        print(f"❌ Agent 请求失败: {e}")
        return False

    # 5️⃣ 收集输出日志
    output_text = ""
    if use_stream:
        # 假设外部环境已实现 AgentEventLogger
        for log in AgentEventLogger().log(response):
            if hasattr(log, 'print'):
                log.print()
            else:
                output_text += str(log)
                print(log, end='', flush=True)
        print()
    else:
        output_text = response.output_text
        print(output_text)

    print(f"--- Agent 回复结束 ---")

    # 6️⃣ 提取代码 (假设外部环境已实现 extract_python_code)
    code_str = extract_python_code(output_text)

    if not code_str:
        print("❌ 未在 Agent 回复中检测到有效 Python 代码，流程终止。")
        return False

    # 7️⃣ 【小侵入式写回】局部替换并写入输出脚本路径
    print(f"\n📝 正在将修改后的局部代码写回并保存至: {output_script_path}")
    try:
        # 剥离多余换行，拒绝一切额外的缩进操作
        clean_agent_code = code_str.strip()
        
        # 将原始脚本拆解，仅替换中间部分
        new_script = original_script[:match.start()] + start_marker + clean_agent_code + end_marker + original_script[match.end():]
        
        os.makedirs(os.path.dirname(output_script_path), exist_ok=True)
        with open(output_script_path, "w", encoding="utf-8") as f:
            f.write(new_script)
        print("✅ 局部修改成功并已存入新文件！")
    except Exception as e:
        print(f"❌ 写入输出文件失败: {e}")
        return False

    # 8️⃣ 执行刚刚写入的脚本
    print(f"\n▶️ 开始执行微调后的脚本: {output_script_path} ...")
    try:
        result = subprocess.run(
            ["python", output_script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode

        print("----- STDOUT -----")
        print(stdout)

        if returncode == 0:
            print("✅ 脚本执行成功！")
            return True
        else:
            print("❌ 脚本执行报错！")
            print("----- STDERR -----")
            print(stderr)
            return False

    except subprocess.TimeoutExpired as e:
        print(f"⏰ 执行超时 ({timeout}s): {e}")
        return False
    except Exception as e:
        print(f"⚠️ 执行异常: {e}")
        return False
    



def run_multitrack_mixing(agent, session_id, track_save_paths: list, final_save_path: str, mix_requirements: list):
    
    mix_requirements = "\n\n".join(mix_requirements)
    track_list_str = "\n".join([f"            - Track {i+1}: '{path}'" for i, path in enumerate(track_save_paths)])
    num_tracks = len(track_save_paths)

    prompt5 = f"""
            For the {num_tracks} temporary audio tracks processed just now:
{track_list_str}

            Refer to the following mixing requirements and, combined with the [Mapping Rules], 
            analyze how to invoke the `apply_multi_track_mix` mixing tool to merge the above {num_tracks} tracks:
            '{mix_requirements}'

            - Output path for the mixed track (output_path): '{final_save_path}' 

            # Core Parameter Description
            You must provide parameters as ordered lists corresponding exactly to the {num_tracks} tracks in the order they are listed above.
            1. **Gain (track_gains_db)**: A list of floats representing the volume gain for each track (e.g., [0.0, -3.0, 1.5]).
            2. **Spatial (track_pan_modes)**: A list of strings representing the sound phase and spatial positioning mode for each track (e.g., ["center", "surround", "left"]).

            [Mapping Rules]
            1. Volume Gain Mapping Rules (Gain Mode)
            Based on the degree quantifiers in the user's description, select the corresponding dB range (positive values for enhancement, negative values for reduction):
            *   **[0 dB] Normal/Baseline**
                *   Trigger words: normal, maintain, baseline, original volume, clear subject.
            *   **[±1 dB ~ ±2 dB] Slight adjustment**
                *   Trigger words: slightly, gently, a little bit.
            *   **[±3 dB ~±4 dB] Moderate adjustment**
                *   Trigger words: moderate, noticeable, general level of reduction/enhancement.
            *   **[±5 dB ~ ±6 dB] Significant/Extreme adjustment**
                *   Trigger words: significant, extremely low, maximum, extreme, dominant, masked.

            2. Spatial and Panning Mapping Rules (Spatial Mode)
            Based on the user's description of spatial feel, immersion, or sound image distribution, choose strictly matching English instructions:
            *   **[center] Center Focus**
                *   Trigger words: center, straight ahead, focus on subject, normal dialogue.
            *   **[left / right] Unilateral or Separation**
                *   Trigger words: left channel/pan left (left), right channel/pan right (right), unilateral interference, left-right separation, dual-source structure.
            *   **[surround] Space and Immersion**
                *   Trigger words: surround, environmental enclosure, widen space, immersive, drowned by environment, broad space.
            *   **[swing] Dynamic and Abnormal**
                *   Trigger words: left-right swinging, dizzy, unstable consciousness, dynamic changes, unstable atmosphere.
            """

    call_agent(agent, prompt5, session_id, use_stream=True)




def run_mixing(agent, session_id, temp_save_path1, temp_save_path2, final_save_path, mix_requirements):

    prompt5 = f"""
            For the two temporary audio tracks processed just now:
            - Track 1 (Vocal/Main): '{temp_save_path1}'
            - Track 2 (Background): '{temp_save_path2}'

            Refer to the following mixing requirements and, combined with the [Mapping Rules], analyze how to invoke the `apply_two_track_mix` mixing tool to merge the above two tracks:
            '{mix_requirements}'

            - Output path for the mixed track (output_path): '{final_save_path}' 

            # Core Parameter Description
            1. **Gain (Gain/Volume)**: Divided into `track1_gain_db` (Vocal/Main) and `track2_gain_db` (Background/Environment).
            2. **Spatial (Panning/Space)**: Controls the sound phase and spatial positioning mode.

            [Mapping Rules]
            1. Volume Gain Mapping Rules (Gain Mode)
            Based on the degree quantifiers in the user's description, select the corresponding dB range (positive values for enhancement, negative values for reduction):
            *   **[0 dB] Normal/Baseline**
                *   Trigger words: normal, maintain, baseline, original volume, clear subject.
            *   **[±1 dB ~ ±2 dB] Slight adjustment**
                *   Trigger words: slightly, gently, a little bit.
            *   **[±3 dB ~±4 dB] Moderate adjustment**
                *   Trigger words: moderate, noticeable, general level of reduction/enhancement.
            *   **[±5 dB ~ ±6 dB] Significant/Extreme adjustment**
                *   Trigger words: significant, extremely low, maximum, extreme, dominant, masked.

            2. Spatial and Panning Mapping Rules (Spatial Mode)
            Based on the user's description of spatial feel, immersion, or sound image distribution, choose strictly matching English instructions:
            *   **[center] Center Focus**
                *   Trigger words: center, straight ahead, focus on subject, normal dialogue.
            *   **[left / right] Unilateral or Separation**
                *   Trigger words: left channel/pan left (left), right channel/pan right (right), unilateral interference, left-right separation, dual-source structure.
            *   **[surround] Space and Immersion**
                *   Trigger words: surround, environmental enclosure, widen space, immersive, drowned by environment, broad space.
            *   **[swing] Dynamic and Abnormal**
                *   Trigger words: left-right swinging, dizzy, unstable consciousness, dynamic changes, unstable atmosphere.
            """

    call_agent(agent, prompt5, session_id, use_stream=True)



def extract_scores(sim_a2t, sim_t2t):
    pattern = r"[\(（](.*?)[\)）]"
    # Extract and convert to a list of floats
    values_a2t = [float(num) for num in re.findall(pattern, sim_a2t)]
    values_t2t = [float(num) for num in re.findall(pattern, sim_t2t)]

    all_values = values_a2t + values_t2t
    if len(all_values) < 5:
        return False

    else:
        org_sim = all_values[0]
        mix_mix_sim = all_values[1]
        org_mix_sim = all_values[2]
        txt_sim = all_values[3]
        qwen_sim = all_values[4]

        score = (mix_mix_sim + qwen_sim) / 2
        return score



def evolutionary_parameter_optimization(
    client,
    agent, 
    refine_agent,
    session_id, 
    base_script_a, 
    base_script_b, 
    track1_save_path,
    track2_save_path,
    mixed_save_path,
    result_path,
    original_audio_path,
    Refine_success_md,
    refine_vector_store_uuid,
    origin_description,
    mixed_description,
    initial_score,
    modification_instructions,
    mix_prompt,
    tar_dir,
    num_iterations=3,      # Total number of evolutionary iterations
    variations_per_iter=3, # Number of variants generated per iteration for the shootout
):
    """
    Agent-based script parameter optimization main process, archiving parameter records to Markdown upon breaking the highest score.
    """
    os.makedirs(os.path.join(tar_dir, "refinement_scripts"), exist_ok=True)
    temp_dir = os.path.join(tar_dir, "refinement_scripts") # Ensure temp_dir is defined
    
    prompt5 = f"""Given the Qwen audio description text {qwen_description_contrast}, please shorten it and translate it into English, keeping the main subject of the event and the description of the timbre effect.
            Output an overall summary of the mixed audio content and effect description text in one sentence as briefly as possible (no more than 20 words).
            It must be an English text.
            You only need to reply with this text, do not reply with anything else."""
            
    # Record the current best base script paths and score
    current_best_a = base_script_a
    current_best_b = base_script_b
    
    current_best_score = initial_score
    
    print(f"🚀 Starting parameter optimization process | Total iterations: {num_iterations} | Variants per iter: {variations_per_iter}")

    for iteration in range(num_iterations):
        print(f"\n" + "="*50)
        print(f"🔄 Starting evolutionary search for iteration {iteration + 1}/{num_iterations}")
        print(f"="*50)
        
        # Record all results for this iteration: list storing dicts: {"a": path, "b": path, "score": float}
        candidates = [] 
        
        # 1. Unfold inner search: generate multiple variants
        for var_idx in range(variations_per_iter):
            print(f"\n🧪 Generating variant set [{var_idx + 1}/{variations_per_iter}] for iteration {iteration + 1}...")
            
            # Set temporary file paths
            temp_a_path = os.path.join(temp_dir, f"iter{iteration}_var{var_idx}_a.py")
            temp_b_path = os.path.join(temp_dir, f"iter{iteration}_var{var_idx}_b.py")
            
            # Use our previous function to call the Agent to generate and execute Track A
            success_a = call_agent_with_script_refinement(client=client,
                agent=agent, session_id=session_id, refine_vector_store_uuid=refine_vector_store_uuid,
                input_script_path=current_best_a, output_script_path=temp_a_path,
                modification_instructions=modification_instructions,
                use_stream=False # Suggest turning off streaming for batch generation to avoid spamming the console
            )
            
            # Generate and execute Track B
            success_b = call_agent_with_script_refinement(client=client,
                agent=agent, session_id=session_id, refine_vector_store_uuid=refine_vector_store_uuid,
                input_script_path=current_best_b, output_script_path=temp_b_path,
                modification_instructions=modification_instructions,
                use_stream=False
            )
            
            # If both tracks are modified and executed successfully, proceed to mixing and scoring
            if success_a and success_b:
                run_mixing(agent, session_id, track1_save_path, track2_save_path, mixed_save_path, mix_prompt)
                qwen_description_contrast = analyze_audio_scene_contrast(original_audio_path, mixed_save_path) 

                prompt6 = f"""Given the Qwen audio description text {qwen_description_contrast} and mixing requirements {mix_prompt}, please analyze whether this mixing result meets the original mixing requirements, and output what improvements still need to be made to the audio."""
                modification_instructions = call_agent(agent, prompt6, session_id, use_stream=False).output_text

                qwen_description_contrast_en = call_agent(agent, prompt5.format(qwen_description_contrast=qwen_description_contrast), session_id, use_stream=False).output_text
                sim_a2t = evaluate_audio_text_similarity(original_audio_path, origin_description, mixed_save_path, mixed_description)
                sim_t2t = evaluate_qwen_text_similarity(origin_description, mixed_description, qwen_description_contrast_en)

                score = extract_scores(sim_a2t, sim_t2t)
                print(f"    🎯 Final score for this variant set: {score:.4f}")
                candidates.append({
                    "a": temp_a_path,
                    "b": temp_b_path,
                    "score": score
                })
            else:
                print(f"    ⚠️ This variant set encountered a single-track execution error and was skipped for mix scoring.")
        
        # 2. Evaluate results for this iteration
        if not candidates:
            print(f"❌ All variants in iteration {iteration + 1} failed. Moving to the next iteration keeping it as is.")
            continue
            
        # Find the highest-scoring variant in this iteration
        best_candidate = max(candidates, key=lambda x: x["score"])
        print(f"\n🏆 Search for this iteration complete! Best variant score: {best_candidate['score']:.4f}")
        
        # 3. Survival of the fittest: if historical high score is broken, overwrite and re-verify
        if best_candidate["score"] > current_best_score:
            print(f"📈 Broke historical high score ({current_best_score:.4f} -> {best_candidate['score']:.4f}), executing base replacement!")
            delta_score = best_candidate["score"] - current_best_score
            current_best_score = best_candidate["score"]
            
            # Overwrite original base scripts
            shutil.copy2(best_candidate["a"], current_best_a)
            shutil.copy2(best_candidate["b"], current_best_b)
            
            print(f"    📝 Overwritten base files: {current_best_a} & {current_best_b}")
            
            # According to requirements: Re-execute single-track scripts, mix, and score operations
            print(f"    🔄 Running a full re-verification on the new base scripts...")
            subprocess.run(["python", current_best_a], capture_output=True, text=True, timeout=120)
            subprocess.run(["python", current_best_b], capture_output=True, text=True, timeout=120)

            run_mixing(agent, session_id, track1_save_path, track2_save_path, mixed_save_path, mix_prompt)
            qwen_description_contrast = analyze_audio_scene_contrast(original_audio_path, mixed_save_path) 
            qwen_description_contrast_en = call_agent(agent, prompt5.format(qwen_description_contrast=qwen_description_contrast), session_id, use_stream=False).output_text
            sim_a2t = evaluate_audio_text_similarity(original_audio_path, origin_description, mixed_save_path, mixed_description)
            sim_t2t = evaluate_qwen_text_similarity(origin_description, mixed_description, qwen_description_contrast_en)

            final_verification_score = extract_scores(sim_a2t, sim_t2t)

            # === Check if all three audio files exist ===
            files_exist = os.path.exists(track1_save_path) and os.path.exists(track2_save_path) and os.path.exists(mixed_save_path)

            results = sim_a2t + "\n" + sim_t2t + "\n" + str(files_exist)
            print(results)
            
            with open(result_path, "w") as f:
                f.write(results)

            print(f"    ✅ Re-verification score: {final_verification_score:.4f}")
            
            # ==========================================
            # 🌟 New addition: Archive this successful Instruction and code to MD
            # ==========================================
            print("    📚 Archiving this successful evolutionary action...")
            try:
                with open(current_best_a, "r", encoding="utf-8") as fa:
                    code_a_content = fa.read()
                with open(current_best_b, "r", encoding="utf-8") as fb:
                    code_b_content = fb.read()
                    
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                md_record = f"\n\n## [{timestamp}] Successful evolutionary variant record, score improved by: {delta_score:.4f}\n"
                md_record += f"**Instructions guiding the refinement:**\n{modification_instructions}\n\n"
                md_record += f"**Refined Track A code:**\n```python\n{code_a_content}\n```\n\n"
                md_record += f"**Refined Track B code:**\n```python\n{code_b_content}\n```\n"
                md_record += "---\n"
                
                # Append to file, creating directories if necessary
                if Refine_success_md:
                    os.makedirs(os.path.dirname(Refine_success_md), exist_ok=True)
                    with open(Refine_success_md, "a", encoding="utf-8") as f_md:
                        f_md.write(md_record)
                    print(f"    💾 Successfully archived to: {Refine_success_md}")
            except Exception as e:
                print(f"    ⚠️ Failed to write to Markdown archive: {e}")
            # ==========================================

        else:
            print(f"📉 Best score this iteration ({best_candidate['score']:.4f}) failed to surpass historical best ({current_best_score:.4f}), maintaining original baseline.")
            
        # 4. Clean up environment: delete all temporary variant scripts generated this iteration
        print("🧹 Cleaning up temporary files...")
        for c in candidates:
            if os.path.exists(c["a"]): os.remove(c["a"])
            if os.path.exists(c["b"]): os.remove(c["b"])

    print(f"\n🎉 All {num_iterations} optimization iterations finished! Final retained best score: {current_best_score:.4f}")
    return current_best_a, current_best_b









def evolutionary_parameter_optimization_multitrack(
    client,
    agent, 
    refine_agent,
    session_id, 
    base_scripts,       # [修改点] 接收任意轨的 Python 脚本路径列表 (List[str])
    track_save_paths,   # [修改点] 接收任意轨的 音频保存路径列表 (List[str])
    mixed_save_path,
    result_path,
    original_audio_path,
    Refine_success_md,
    refine_vector_store_uuid,
    origin_description,
    mixed_description,
    initial_score,
    modification_instructions,
    mix_prompt,
    tar_dir,
    num_iterations=3,      # Total number of evolutionary iterations
    variations_per_iter=3, # Number of variants generated per iteration for the shootout
):
    """
    Agent-based script parameter optimization main process, archiving parameter records to Markdown upon breaking the highest score.
    Supports an arbitrary number of tracks.
    """
    os.makedirs(os.path.join(tar_dir, "refinement_scripts"), exist_ok=True)
    temp_dir = os.path.join(tar_dir, "refinement_scripts") # Ensure temp_dir is defined
    
    num_tracks = len(base_scripts)
    
    # [修改点] 修正为普通字符串，以便后续使用 .format() 进行动态格式化
    prompt5 = """Given the Qwen audio description text {qwen_description_contrast}, please shorten it and translate it into English, keeping the main subject of the event and the description of the timbre effect.
            Output an overall summary of the mixed audio content and effect description text in one sentence as briefly as possible (no more than 20 words).
            It must be an English text.
            You only need to reply with this text, do not reply with anything else."""
            
    # Record the current best base script paths and score
    current_best_scripts = list(base_scripts)  # 复制一份当前最优脚本列表
    current_best_score = initial_score
    
    print(f"🚀 Starting parameter optimization process | Total iterations: {num_iterations} | Variants per iter: {variations_per_iter} | Tracks: {num_tracks}")

    for iteration in range(num_iterations):
        print(f"\n" + "="*50)
        print(f"🔄 Starting evolutionary search for iteration {iteration + 1}/{num_iterations}")
        print(f"="*50)
        
        # Record all results for this iteration: list storing dicts: {"scripts": list_of_paths, "score": float}
        candidates = [] 
        
        # 1. Unfold inner search: generate multiple variants
        for var_idx in range(variations_per_iter):
            print(f"\n🧪 Generating variant set [{var_idx + 1}/{variations_per_iter}] for iteration {iteration + 1}...")
            
            temp_variant_scripts = []
            all_success = True
            
            
            for track_idx in range(num_tracks):
                temp_script_path = os.path.join(temp_dir, f"iter{iteration}_var{var_idx}_track{track_idx}.py")
                temp_variant_scripts.append(temp_script_path)
                
                success = call_agent_with_script_refinement(
                    client=client,
                    agent=agent, 
                    session_id=session_id, 
                    refine_vector_store_uuid=refine_vector_store_uuid,
                    input_script_path=current_best_scripts[track_idx], 
                    output_script_path=temp_script_path,
                    modification_instructions=modification_instructions,
                    use_stream=False # Suggest turning off streaming for batch generation to avoid spamming the console
                )
                
                if not success:
                    all_success = False
            
            # If all tracks are modified and executed successfully, proceed to mixing and scoring
            if all_success:
                run_multitrack_mixing(agent, session_id, track_save_paths, mixed_save_path, mix_prompt)
                qwen_description_contrast = analyze_audio_scene_contrast(original_audio_path, mixed_save_path) 

                prompt6 = f"""Given the Qwen audio description text {qwen_description_contrast} and mixing requirements {mixed_description},
                please analyze whether this mixing result meets the original mixing requirements, and output what improvements still need to be made to the audio."""
                modification_instructions = call_agent(agent, prompt6, session_id, use_stream=False).output_text

                qwen_description_contrast_en = call_agent(agent, prompt5.format(qwen_description_contrast=qwen_description_contrast), session_id, use_stream=False).output_text
                sim_a2t = evaluate_audio_text_similarity(original_audio_path, origin_description, mixed_save_path, mixed_description)
                sim_t2t = evaluate_qwen_text_similarity(origin_description, mixed_description, qwen_description_contrast_en)

                score = extract_scores(sim_a2t, sim_t2t)
                print(f"    🎯 Final score for this variant set: {score:.4f}")
                candidates.append({
                    "scripts": temp_variant_scripts,
                    "score": score
                })
            else:
                print(f"    ⚠️ This variant set encountered a single-track execution error and was skipped for mix scoring.")
        
        # 2. Evaluate results for this iteration
        if not candidates:
            print(f"❌ All variants in iteration {iteration + 1} failed. Moving to the next iteration keeping it as is.")
            continue
            
        # Find the highest-scoring variant in this iteration
        best_candidate = max(candidates, key=lambda x: x["score"])
        print(f"\n🏆 Search for this iteration complete! Best variant score: {best_candidate['score']:.4f}")
        
        # 3. Survival of the fittest: if historical high score is broken, overwrite and re-verify
        if best_candidate["score"] > current_best_score:
            print(f"📈 Broke historical high score ({current_best_score:.4f} -> {best_candidate['score']:.4f}), executing base replacement!")
            delta_score = best_candidate["score"] - current_best_score
            current_best_score = best_candidate["score"]
            
            # Overwrite original base scripts for all tracks
            for idx in range(num_tracks):
                shutil.copy2(best_candidate["scripts"][idx], current_best_scripts[idx])
                print(f"    📝 Overwritten base file: {current_best_scripts[idx]}")
            
            # According to requirements: Re-execute single-track scripts, mix, and score operations
            print(f"    🔄 Running a full re-verification on the new base scripts...")
            
            # [修改点] 循环执行所有被更新的最优脚本
            for script_path in current_best_scripts:
                subprocess.run(["python", script_path], capture_output=True, text=True, timeout=120)

            run_multitrack_mixing(agent, session_id, track_save_paths, mixed_save_path, mix_prompt)
            qwen_description_contrast = analyze_audio_scene_contrast(original_audio_path, mixed_save_path) 
            qwen_description_contrast_en = call_agent(agent, prompt5.format(qwen_description_contrast=qwen_description_contrast), session_id, use_stream=False).output_text
            sim_a2t = evaluate_audio_text_similarity(original_audio_path, origin_description, mixed_save_path, mixed_description)
            sim_t2t = evaluate_qwen_text_similarity(origin_description, mixed_description, qwen_description_contrast_en)

            final_verification_score = extract_scores(sim_a2t, sim_t2t)

            # === Check if all audio files exist ===
            files_exist = all(os.path.exists(path) for path in track_save_paths) and os.path.exists(mixed_save_path)

            results = sim_a2t + "\n" + sim_t2t + "\n" + str(files_exist)
            print(results)
            
            with open(result_path, "w") as f:
                f.write(results)

            print(f"    ✅ Re-verification score: {final_verification_score:.4f}")
            
            # ==========================================
            # 🌟 New addition: Archive this successful Instruction and code to MD
            # ==========================================
            print("    📚 Archiving this successful evolutionary action...")
            try:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                md_record = f"\n\n## [{timestamp}] Successful evolutionary variant record, score improved by: {delta_score:.4f}\n"
                md_record += f"**Instructions guiding the refinement:**\n{modification_instructions}\n\n"
                
                # [修改点] 动态读取和写入每一轨的代码内容
                for idx, script_path in enumerate(current_best_scripts):
                    with open(script_path, "r", encoding="utf-8") as f_script:
                        code_content = f_script.read()
                        md_record += f"**Refined Track {idx + 1} code:**\n```python\n{code_content}\n```\n\n"
                        
                md_record += "---\n"
                
                # Append to file, creating directories if necessary
                if Refine_success_md:
                    os.makedirs(os.path.dirname(Refine_success_md), exist_ok=True)
                    with open(Refine_success_md, "a", encoding="utf-8") as f_md:
                        f_md.write(md_record)
                    print(f"    💾 Successfully archived to: {Refine_success_md}")
            except Exception as e:
                print(f"    ⚠️ Failed to write to Markdown archive: {e}")
            # ==========================================

        else:
            print(f"📉 Best score this iteration ({best_candidate['score']:.4f}) failed to surpass historical best ({current_best_score:.4f}), maintaining original baseline.")
            
        # 4. Clean up environment: delete all temporary variant scripts generated this iteration
        print("🧹 Cleaning up temporary files...")
        for c in candidates:
            for temp_path in c["scripts"]:
                if os.path.exists(temp_path): 
                    os.remove(temp_path)

    print(f"\n🎉 All {num_iterations} optimization iterations finished! Final retained best score: {current_best_score:.4f}")
    return current_best_scripts