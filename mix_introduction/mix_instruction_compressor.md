# 🎵 智能混音助手知识库：压缩器 (Compressor) 决策与执行指南

## 1. 核心决策逻辑：何时以及如何决定参数？
Agent 在决定压缩器处理策略时，必须严格遵循以下优先级：

* **🥇 第一优先级：优先满足用户明确要求**
  如果用户在对话中明确指定了音频类型或处理目标（例如：“让人声对白更稳定”、“控制一下鼓点的动态”、“做一下总线胶合”），Agent 应**直接参考**下方的【常见场景参数参考】设定参数，无需调用分析工具。
* **🥈 第二优先级：未知场景调用工具分析**
  如果用户仅提供音频而没有具体要求（例如：“帮我处理一下这个声音”、“听听看要不要加压缩”），Agent **必须调用** `analyze_adaptive_compression_params(input_path)` 工具。通过阅读诊断报告来决定是否需要压缩，并直接提取工具返回的 `recommended_pedalboard_params` 作为参数。

## 2. 常见场景参数参考 (用于响应用户明确需求)
当用户明确场景时，请基于以下经验值设定参数（`Threshold` 通常设为音频平均能量 RMS 减去 8dB）：
* **🎙️ 对白/人声 (Vocals/Dialogue)**：Ratio 3.0 - 5.0 | Attack 3.0 - 10.0ms | Release 50.0 - 150.0ms
* **🥁 鼓点/打击乐 (Drums)**：Ratio 4.0 - 8.0 | Attack 15.0 - 30.0ms (较慢起音，保留敲击瞬态) | Release 20.0 - 50.0ms
* **🌊 环境声/铺底 (Ambient)**：Ratio 2.0 - 3.0 | Attack 30.0 - 50.0ms | Release 150.0 - 300.0ms (较慢释放，防止呼吸效应)
* **🎛️ 总线/胶合 (Master Bus)**：Ratio 1.5 - 2.0 | Attack 30.0ms+ | Release 100.0 - 300.0ms

## 3. Pedalboard 代码实现指南与示例
当你需要生成代码时，只需使用 Python 的 `pedalboard` 库实例化压缩器模块。
* **核心参数**：向 `Compressor` 传入 `threshold_db`, `ratio`, `attack_ms`, `release_ms`。
* **模块化原则**：**请勿包含任何音频文件的读取、写入或导出代码。** 仅需提供构建 `Pedalboard` 效果链或配置参数的函数即可。

**💡 代码示例（单独的压缩器模块写法）：**

```python
from pedalboard import Pedalboard, Compressor

    """
    根据给定的参数，构建并返回一个纯粹的压缩器效果链模块。
    该模块后续可直接被调用并作用于音频数组，例如：effected_audio = board(audio_array, samplerate)
    """

    Compressor(
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms
    )

    