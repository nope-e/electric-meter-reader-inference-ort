# 模型来源与版本记录

本包权重按 `manifest.json` 中 SHA256 固定。以下来源、训练及许可描述来自项目已有下载记录、训练记录和 ONNX 元数据，本次未重新下载或改动权重。

## 当前模型

| 部署文件 | 原项目文件（相对项目根） | 来源/版本 |
| --- | --- | --- |
| meter_type.onnx | `test-prj/models/meter_type_cnn/meter_type.onnx` | torchvision MobileNetV3-small，ImageNet 预训练后使用本地 86 张照片（33 LCD、53 wheel）训练 20 epochs |
| meter_area.onnx | `test-prj/models/meter_area/area.onnx` | YOLO11n 单类 area；601 图/609 框，train/valid/test=499/60/42；70 epochs，best epoch 50 |
| lcd_digits.onnx | `test-prj/models/digits.onnx` | adbartx/multimeter-ocr 的 digits.pt 导出；LCD 数字识别权重保持不变 |
| wheel_slots.onnx | `test-prj/models/wheel_slots/wheel_slots.onnx` | YOLO11n 第三版 v3 local；完整轮位、背景负样本和本地适配数据；57 epochs，best epoch 27 |
| wheel_reader.onnx | `test-prj/models/visionstate/wheels-v1.onnx` | oleost/VisionState 发布的 wheels-v1；模型保持不变，采用原作者预处理与联合解码，主字符串使用项目既有 floor 约定 |

原目录是实验/训练使用的位置，本目录是部署副本；原文件继续保留，既有实验脚本仍可运行。

### Area 与轮位训练

area 从已有读数区框统一成单类，digital/wheel 类型不作为 area 类别。基础权重来自项目已有的 [adbartx/multimeter-ocr](https://github.com/adbartx/multimeter-ocr) display.pt，原文件 SHA256 `26cac8eebb331fb10d89ab7eb79c391e1954ab784a3e97d971cc12bb5080a74b`。部署权重是本地训练后的 area，哈希与基础 PT 不同。

轮位第三版继承 v2 negatives，训练链为 area → 初版轮位 → v2 negatives → v3 local。导出数据 142 图/689 轮位框，train 127 图/609 框、valid 7 图/39 框、test 8 图/41 框；包含 18 张零框负样本，也包含重复和构造样本，这个数量不等于独立实拍表数。目标为完整轮位窗口，支持 5/6 位。7 张新增实拍照片已用于训练，另有验证/留出照片，完整 104 张回测也包含训练照片。

### LCD 原始权重

来源：[digits.pt](https://github.com/adbartx/multimeter-ocr/blob/main/models/digits.pt)，项目下载记录中的 Git blob 为 `555e491e5ba132cdc9f8d5f29acb1032b0cb82f0`，原 PT SHA256 `c841f72bfd65ef89150af1b3cc2546189fc856d2479206a403d9a7ef444206d6`。本包保存 ONNX，不要将 PT 哈希用来校验 ONNX。

### VisionState

- 模型：[oleost/visionstate-wheel-reader](https://huggingface.co/oleost/visionstate-wheel-reader)，revision `42f32402973b3bd3de0258766404232b11e8216f`，文件 `wheels-v1.onnx`。
- 作者代码：[oleost/VisionState](https://github.com/oleost/VisionState)，commit `9e5e4f54b309adbed159be244d3ec747f72efe61`。
- `common.py` Git blob `670c973e9680c2bdaf3f6156809ca1f4f2689e26`；随包保存为 [reference/visionstate_common.py](reference/visionstate_common.py)。
- 保留 [VISIONSTATE_SOURCE.json](reference/VISIONSTATE_SOURCE.json) 和 [Apache-2.0 LICENSE](reference/VISIONSTATE_LICENSE)。随包 common 中含训练辅助函数，当前推理入口只使用 cell_input、decode_shifted 及其依赖。

## 许可记录

VisionState 已有来源记录声明 Apache-2.0，其作者代码许可证随包保留。multimeter-ocr 已有来源记录声明仓库为 AGPL-3.0；area/LCD/轮位 YOLO 的 ONNX 元数据也声明 Ultralytics AGPL-3.0。分类器基于 torchvision MobileNetV3-small。这里只记录已有来源声明，本次没有重新审查各数据集、预训练权重或完整部署应用的许可适用范围，也没有把整套模型统一重新授权。

## 核验与评测边界

[verification.json](verification.json) 记录本包独立 Python 流程与既有 104 张回测的逐阶段核验；模型输入输出也由 ONNX Runtime 实际加载检查。核验脚本不依赖原推理模块，但需要本地原图和既有报告：

```powershell
# 项目根目录运行；将验证报告写到新的路径
.\test-prj\.venv\Scripts\python.exe models\reference\verify_local.py --output models\runs\recheck\verification.json
```

比较图像 SHA256、分类/读数、area 框、LCD 数字框、轮位裁剪坐标、VisionState logits 和全部轮位输入灰度图。box 允许 0.0001 像素绝对误差，分类分数和 logits 允许 0.000001 绝对误差，轮位输入图要求 uint8 完全相同。

识别准确率引用本地 104 张评测：98 张有参考，79 张整串一致；包括训练样本和同表重复图，参考主要来自人工目视复核，并非独立实地真值。当前已验证桌面 CPU，未验证手机或微信小程序。模型 score、area 检出数量、包装一致性都不等于独立泛化准确率。
