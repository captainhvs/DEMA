import traceback

def execute_custom_code(python_code: str) -> str:
    """
    [自定义音频处理脚本执行工具]
    当现有的预设混音工具无法满足特殊或全新的音频处理需求时，调用此工具执行自定义的 Python 脚本。
    
    输入参数:
    :param python_code: 完整的 Python 代码。代码中必须包含读取输入音频、使用 pedalboard/soundfile/numpy/scipy 等库进行处理，并将结果保存到指定输出路径的完整逻辑。
    
    输出：
    final_response: 包含代码执行成功与否的状态或报错信息的字符串。
    """
    print(f"\n[系统拦截] ⚠️ Agent 正在尝试执行动态生成的代码:\n{python_code}\n")
    try:
        # 定义一个干净的命名空间执行代码
        exec_globals = {}
        exec(python_code, exec_globals)
        return "✅ 自定义音频处理代码执行成功，文件已保存。"
    except Exception as e:
        error_msg = f"❌ 代码执行报错，请根据报错信息修改你的代码:\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg