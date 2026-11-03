# 电表识别 ONNX 模型包

当前已测试版本：`2026-10-20-current-v3`。包含 5 个 ONNX，共 **40,062,298 字节（约 38.2 MiB）**；从实验项目原样复制，保留当前权重、裁剪和解码行为。

完整流程：**原图 → 整图分类 digital/wheel + 单类 area 检测 → 按类型识别**。digital 使用原 LCD 数字检测器；wheel 使用第三版完整轮位检测器和 VisionState 识别器。

## 模型清单

所有输入输出均为 `float32`；图像张量采用 **NCHW**。下表中高在宽前，`N` 是当前照片实际检测到的轮位数。

| 文件 | 用途 | 输入名称及形状 | 输出名称及形状 | ONNX opset |
| --- | --- | --- | --- | --- |
| [meter_type.onnx](meter_type.onnx) | 整图分类 | `images [1,3,256,256]` | `logits [1,2]` | 12 |
| [meter_area.onnx](meter_area.onnx) | 整条读数区定位，单类 area | `images [1,3,640,640]` | `output0 [1,5,8400]` | 12 |
| [lcd_digits.onnx](lcd_digits.onnx) | LCD 数字及符号检测 | `images [1,3,640,640]` | `output0 [1,16,8400]` | 12 |
| [wheel_slots.onnx](wheel_slots.onnx) | 完整轮位窗口检测，v3 local | `images [1,3,640,640]` | `output0 [1,5,8400]` | 12 |
| [wheel_reader.onnx](wheel_reader.onnx) | 单个滚轮状态识别，可批量输入 | `cells [N,1,48,32]` | `logits [N,100]` | 17 |

前后处理在应用代码中执行，检测模型没有内置 NMS。分类、检测、滚轮识别的归一化方式各不相同，不能共用一套图像预处理。

## 文档和参考实现

- [PIPELINE.md](PIPELINE.md)：输入图像要求、每个模型的预处理/后处理、选框和裁剪规则、坐标系、输出字段。
- [SOURCES.md](SOURCES.md)：权重来源、训练版本、许可记录和评测边界。
- [manifest.json](manifest.json)：文件 SHA256、精确大小、输入输出、ONNX 元数据和当前参数。
- [reference/pipeline.py](reference/pipeline.py)：只依赖本目录的完整 Python ONNX Runtime 参考流程。
- [verification.json](verification.json)：独立参考流程与原项目 104 张照片的对齐结果。
- [examples/](examples/)：LCD、滚轮、漏 area、已知 floor 问题的真实输出示例，省略大体积 logits。

## 直接运行

以下命令在项目根目录执行，使用当前已安装依赖的环境：

```powershell
# 检查全部模型哈希及 ONNX Runtime 输入输出
.\test-prj\.venv\Scripts\python.exe models\reference\pipeline.py --verify-only

# 单图；输出 JSON 文件必须不存在
.\test-prj\.venv\Scripts\python.exe models\reference\pipeline.py --input datas\lcd\1792403667245.jpg --output models\runs\single.json

# 混合目录，递归读取支持的图片文件
.\test-prj\.venv\Scripts\python.exe models\reference\pipeline.py --input datas\lcd datas\wheel datas\test_wheel --output models\runs\all.json
```

独立使用时复制整个 `models` 目录，安装 [requirements.txt](requirements.txt)，再运行 `python reference/pipeline.py --input <图片或目录> --output <新JSON文件>`。推理无需 PyTorch、Ultralytics 或训练数据；模型路径默认按脚本位置解析，不依赖工作目录。

```powershell
python -m pip install -r models\requirements.txt
```

Python 调用示例（项目根目录）：

```python
from models.reference.pipeline import MeterPipeline

pipeline = MeterPipeline()  # 创建一次，重复使用；默认桌面 CPU
result = pipeline.predict_file("datas/lcd/1792403667245.jpg")
print(result["meter_type"], result["raw_digits"], result["reading"])
# digital 1013964 10139.64
```

## 当前结果与已知问题

104 张本地照片：类型 104/104，检测到 area 99/104；98 张有可评分参考，完整读数一致 **79/98（80.6%）**，其中 LCD 27/28（96.4%）、wheel 52/70（74.3%）。另 6 张参考不确定，单列不计分。包含训练照片和同一块表的重复拍摄，属于本地回测。详细结果见 [原项目报告](../test-prj/ALL_LOCAL_RESULTS.md)。

**`003733 → 003732` 问题仍在**：部分图最后一轮独立预测为 3.0，机械联合解码加共同偏移后成为 2.9，当前 floor 规则得到 2。参考实现同时保存作者四舍五入候选；本次整理保持原主读数规则。还存在漏 area、漏轮位、背景误框和 LCD 漏末位等问题。

所有读数均是候选值；分类 softmax、检测分数不等于读数正确率。当前小数位是应用配置约定：LCD 两位，wheel 一位，并非模型自动推断。

当前验证环境为桌面 ONNX Runtime CPU。微信小程序端的模型加载、算子支持、内存、前后处理数值一致性和耗时尚未实测。移植时按 [PIPELINE.md](PIPELINE.md) 对齐各阶段；可使用示例中的 `source_sha256` 和坐标与本地原图复核。
