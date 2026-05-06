import os
import time
import uuid
import textwrap
import json

def _write_to_md(filepath: str, content: str):
    """辅助函数：以追加模式写入 Markdown 文件"""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or '.', exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(content + "\n")


def summarize_execution_case(
    summary_agent,
    summary_session_id,
    file_folder: str,
    initial_prompt: str,
    is_success: bool,
    attempt_count: int,
    query_for_rag: str,
    error_history: str = "",
    error_analysis: str = ""
):
    """
    使用无记忆的 Agent 对执行结果进行复盘总结，并分别写入对应的 md 文件。
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    case_id = f"CASE_{uuid.uuid4().hex[:8]}"
    
    print(f"\n📝 正在生成案例总结... (状态: {'成功' if is_success else '失败'}, 尝试次数: {attempt_count})")

    if is_success and attempt_count > 1:
        # ==========================================
        # 分支 2: 经过 Debug 后成功
        # ==========================================
        file_path = os.path.join(file_folder, "debug_success_cases_temp.md")
        
        analyze_prompt = textwrap.dedent(f"""
            【任务：分析代码 Debug 过程】
            你是一个资深的 Python 专家，以下的代码经过多轮debug最终成功运行，请分析以下由于代码报错引发的 Debug 过程。

            [原始任务]: 
            {initial_prompt}

            [曾遇到的报错信息]: 
            ```text
            {error_history}
            ```

            [错误分析与总结]:
            ```text
            {error_analysis}
            ```
            [查询关键词]
            ```text
            {query_for_rag}
            ```

            请参考每一轮报错，对照最终成功运行的代码，严格按照以下 4 个模块输出结构化的总结，不要输出多余的废话：
            1. **查询关键词**: 每一轮针对报错查询的关键词是什么？将查询关键词的文本原封不动地放上来。
            3. **失败报错概括**: 每一轮遇到了什么核心错误？
            4. **成功原因**: 什么修改解决了问题，为什么这样的修改解决了问题？
            【注意】严禁把整个脚本的代码放上去，只需要针对每一轮报错提炼出核心的错误信息和修复动作即可。
        """).strip()
        
        analysis = summary_agent.create_turn(
            messages=[{"role": "user", "content": analyze_prompt}],
            session_id=summary_session_id,
            stream=False
        ).output_text.strip()

        content = textwrap.dedent(f"""
            ### [{timestamp}] 🟡 经过 {attempt_count} 次尝试后 Debug 成功

            **Debug 总结与反思:**
            {analysis}

            """).strip()
        
        _write_to_md(file_path, content)
        print(f"✅ 已记录 [Debug 成功] 案例至 {file_path}")





def aggregate_and_summarize_cases(summary_agent,agent_prompt,summary_session_id, temp_file_path: str, output_dir:str, output_prefix: str):
    """
    读取 temp.md 文件，交由 Agent 进行错误分类与合并，并输出到新的带有时间戳的文件中。
    
    :param summary_agent: 无记忆的总结 Agent
    :param temp_file_path: 原始 temp.md 的绝对路径
    :param output_prefix: 输出文件的前缀 (如 'debug_success_cases')
    :param case_type: 'success' 或 'failure'，用于区分提示词策略
    """
    if not os.path.exists(temp_file_path):
        print(f"⚠️ 文件不存在，创建新 md 文件: {temp_file_path}")

        # 确保父目录存在
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        # 创建新的 md 文件
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write("")  # 可以写入默认内容，比如 "# 新文档\n"



    with open(temp_file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read().strip()

    if not raw_content:
        print(f"⚠️ 文件为空，跳过处理: {temp_file_path}")
        return

    print(f"\n📂 正在读取并分析: {temp_file_path}")
    
    agent_prompt = f"{agent_prompt}\n```markdown\n{raw_content}\n```"

    print("🤖 Agent 正在进行错误类型聚合与概括...")
    
    
    
    summary_result = summary_agent.create_turn(
        messages=[{"role": "user", "content": agent_prompt}],
        session_id=summary_session_id,
        stream=False
    ).output_text.strip()

    # 生成带时间戳的目标文件路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_prefix}_{timestamp}.md"
    output_path = os.path.join(output_dir, output_filename)

    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# 归档时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(summary_result)
        
    print(f"✅ 提炼概括完成！已保存至: {output_path}")
    
    # 可选：归档完成后清空或备份原始的 temp.md 文件
    # open(temp_file_path, 'w').close() 


