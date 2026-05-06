# 🎛️ 智能混音助手知识库：全链路混音与效果器串联 (Chain) 执行指南

## 1. 核心工程概念：信号流向与串联顺序 (Signal Flow)
在 `pedalboard` 中，音频信号是**从左到右、按顺序**流经各个效果器的。前一个效果器的输出，就是后一个效果器的输入。因此，效果器的摆放顺序会决定最终的音色。

**Agent 必须遵守的行业标准串联顺序（从先到后）：**
1. **🔪 滤波器 (Filter)**：排在最前。先清理底噪、去浑浊、去高频毛刺，确保后续效果器处理的是“干净”的信号。
2. **🗜️ 压缩器 (Compressor)**：排在第二。对清理后的信号进行动态控制，让音量变得平稳、紧凑。
3. **🎸 失真与补偿 (Distortion + Gain)**：排在中间。为平稳的信号添加谐波和粗糙质感。（⚠️ 谨记：失真后紧跟 Gain 补偿削波）。
4. **⏱️ 延迟 (Delay)**：排在倒数第二。在进入最终的声场空间前，先制造清晰的节奏拖尾或回声。
5. **⛪ 混响 (Reverb)**：排在最后。空间效果必须放在最末尾，确保前面所有的声音（包括延迟的回声）都能自然地融入同一个环境空间中，避免浑浊。

*(💡 提示：并非每次混音都需要用到所有模块，但只要使用，就必须遵循上述相对顺序。)*

## 2. 完整混音执行标准流程 (Execution Workflow)
当 Agent 收集完用户意图或调用完相关分析工具后，完整的处理生命周期如下：

* **Step 1: 读取音频** -> 使用 `soundfile` (`sf.read`) 读取原始音频，并通过 `numpy` 统一降混为单声道，防止多通道冲突。
* **Step 2: 声明效果链** -> 创建一个空的 Python 列表 `plugins = []`。
* **Step 3: 获取参数并按顺序组装** -> 严格查阅各个模块的专项文档获取正确的参数，将实例化的插件通过 `.append()` 压入列表。
* **Step 4: 构建 Pedalboard** -> 执行 `board = Pedalboard(plugins)`。
* **Step 5: 渲染音频** -> 执行 `effected_audio = board(audio, samplerate)`。
* **Step 6: 导出文件** -> 使用 `soundfile` (`sf.write`) 将处理后的数组安全写入目标路径。

## 3. Pedalboard 全链路代码实现模板
请使用以下代码结构作为组装完整混音处理脚本的参考标准。**注意：不要直接复制占位符，务必根据场景动态替换。**

```python
import numpy as np
import soundfile as sf
from pedalboard import (
    Pedalboard, HighpassFilter, LowpassFilter, 
    Compressor, Distortion, Gain, Delay, Reverb
)

# 1. 读取原始音频并统一为单声道
audio, samplerate = sf.read(input_path)
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)


# 2. 声明效果链列表
plugins = []

# 3. 按标准顺序组装效果链 (Signal Flow)
# ⚠️ Agent 警告：此处严禁捏造具体参数值！
# 必须严格查阅各模块专属指南，或调用工具（如 estimate_bandwidth, analyze_adaptive_compression_params）获取真实参数。

# [第一级] 滤波器 (Filters)
# -> 查阅 mix_uni_freq 获取如何设置 cutoff_frequency_hz
if need_highpass:
    plugins.append(HighpassFilter(cutoff_frequency_hz=...))
if need_lowpass:
    plugins.append(LowpassFilter(cutoff_frequency_hz=...))

# [第二级] 压缩器 (Compressor)
# -> 查阅 mix_uni_compressor 获取 threshold_db, ratio, attack_ms, release_ms
if need_compressor:
    plugins.append(Compressor(
        threshold_db=...,
        ratio=...,
        attack_ms=...,
        release_ms=...
    ))

# [第三级] 失真与增益补偿 (Distortion & Gain)
# -> 查阅 mix_uni_distortion 获取 drive_db，并严格执行 -0.5 * drive_db 的补偿原则
if need_distortion:
    plugins.append(Distortion(drive_db=...))
    plugins.append(Gain(gain_db=...)) 

# [第四级] 延迟 (Delay) 
# -> 查阅 mix_uni_delay 获取 delay_seconds, feedback, mix
if need_delay:
    plugins.append(Delay(
        delay_seconds=...,
        feedback=...,
        mix=...
    ))

# [第五级] 混响 (Reverb) - 必须放在最后
# -> 查阅 mix_uni_reverb 获取 room_size, wet_level, dry_level
if need_reverb:
    plugins.append(Reverb(
        room_size=...,
        wet_level=...,
        dry_level=...
    ))

# 4. 实例化 Pedalboard
board = Pedalboard(plugins)

# 5. 渲染处理音频
effected_audio = board(audio, samplerate)

# 6. 导出并写入指定路径
sf.write(output_path, effected_audio, samplerate)