# 🎛️ 智能混音助手知识库：自适应滤波器 (Filter) 决策与执行指南

## 1. 核心决策逻辑：何时使用滤波器？
滤波器用于在频域上修剪音频信号，切除不需要的频段。当用户表达以下意图时，Agent 应当决定使用高通 (Highpass) 或低通 (Lowpass) 滤波：

* **🔪 高通滤波 (Highpass Filter) - 切除低频，保留高频**
  * **触发场景**：用户要求“去除底噪/轰隆声”、“消除风噪/麦克风震动”、“提升声音清晰度”、“让声音变薄”。
* **🧱 低通滤波 (Lowpass Filter) - 切除高频，保留低频**
  * **触发场景**：用户要求“去除嘶嘶声/高频刺耳声”、“模拟隔着墙/门听声音”、“制造水下沉闷效果”、“拉远声音距离”。
* **📻 带通滤波 (Bandpass / 同时使用高通+低通)**
  * **触发场景**：用户要求“模拟电话音效”、“老式收音机效果”、“对讲机声音”。

## 2. 参数获取策略：如何调用工具计算截止频率？
**Agent 严禁自行猜测滤波器的截止频率（Hz）。** 不同的音频频谱结构完全不同，固定频率极易破坏音频主体。
当你决定使用滤波器时，**必须调用** `estimate_bandwidth(audio, sr, low_pct, high_pct)` 工具来获取自适应频率。

### 📌 分位点 (Percentile) 参数映射指南
你需要根据用户的意图强烈程度，灵活且不对称地设置工具的 `low_pct` 和 `high_pct` 参数：

* **针对 Highpass (对应工具返回的 `low_cut` 频率)：**
  * **轻度去底噪**：设置 `low_pct = 0.02` ~ `0.05` (仅切除极低频能量)。
  * **提升清晰度/略微变薄**：设置 `low_pct = 0.10` ~ `0.15`。
  * **电话音/对讲机 (重度)**：设置 `low_pct = 0.20` ~ `0.25`。
* **针对 Lowpass (对应工具返回的 `high_cut` 频率)：**
  * **轻度去毛刺/高频去噪**：设置 `high_pct = 0.95` ~ `0.98` (仅切除极高频)。
  * **中度距离感/隔挡感**：设置 `high_pct = 0.85` ~ `0.90`。
  * **水下沉闷效果/极度压抑**：设置 `high_pct = 0.70` ~ `0.80` (切除大量中高频)。

*(💡 例如：如果用户想做“电话音效”，你需要调用工具并传入 `low_pct=0.20, high_pct=0.85`，然后将返回的两个频率分别交给 Highpass 和 Lowpass。)*

## 3. 滤波器核心参数说明
在使用 `pedalboard` 构建滤波模块时，只需关注以下核心参数：
* **`cutoff_frequency_hz` (截止频率)**：
  * 在 `HighpassFilter` 中：代表低于此频率的声音将被切除。**传入工具返回的 `low_cut`**。
  * 在 `LowpassFilter` 中：代表高于此频率的声音将被切除。**传入工具返回的 `high_cut`**。

## 4. Pedalboard 代码实现指南与示例
当你需要生成代码时，请使用 `pedalboard` 库实例化滤波模块。
* **依赖项**：导入 `Pedalboard`, `HighpassFilter`, `LowpassFilter`。
* **模块化原则**：**请勿包含任何音频文件的读取、写入或导出代码。** 仅需提供构建效果链的函数。

**💡 代码示例（包含高通与低通的模块化写法）：**

```python
from pedalboard import Pedalboard, HighpassFilter, LowpassFilter

# 假设此前已调用 estimate_bandwidth 获取了 low_cut_hz 和 high_cut_hz
# 例如：low_cut_hz = 300.0, high_cut_hz = None (仅高通)

plugins = []

# 如果存在低频截断值，则实例化 HighpassFilter
if low_cut_hz is not None:
    plugins.append(HighpassFilter(cutoff_frequency_hz=low_cut_hz))
    
# 如果存在高频截断值，则实例化 LowpassFilter
if high_cut_hz is not None:
    plugins.append(LowpassFilter(cutoff_frequency_hz=high_cut_hz))
    
# 将插件列表传入 Pedalboard 构建最终的效果链模块
board = Pedalboard(plugins)