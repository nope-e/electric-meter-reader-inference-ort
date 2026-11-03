# 输入输出与前后处理规范

本文对应 `manifest.json` 的当前模型版本。Python 中精确的图像插值、舍入、NMS 和解码实现见 [reference/ops.py](reference/ops.py)、[reference/visionstate_common.py](reference/visionstate_common.py)、[reference/pipeline.py](reference/pipeline.py)。manifest 中参数描述当前流程；修改参数或解码代码后需要重新核对输出。

## 1. 原图与坐标

输入为完整照片，包含完整读数区域；读数位数可以是 5 位或 6 位。推理只读取图像像素，文件名和目录类别不参与分类或识别。参考 CLI 支持 JPG/JPEG/PNG/WebP/BMP。

先解码，再按 EXIF 旋转/翻转，转成 RGB、`uint8`、HWC。所有分支复用同一份方向校正后的原图。不要先将原图裁成屏幕再送给整图分类器，也不要使用中间保存的有损 JPEG 裁图做识别。

坐标约定：`xyxy = [x1,y1,x2,y2]`，原点左上，x 向右、y 向下；图像宽 W、高 H。浮点检测框的裁剪方式为左/上 `floor`、右/下 `ceil`，限制在 `[0,W]×[0,H]`，右/下边界不包含在像素切片中。参考输出里的原图坐标均指 EXIF 校正后图像；LCD 数字框和 `xyxy_in_area` 指对应 area 裁图。

## 2. 整图分类 meter_type.onnx

输入 `images [1,3,256,256] float32`，输出 `logits [1,2] float32`。

1. 整图 RGB 按比例缩放并居中补到 256×256，Pillow `ImageOps.pad` + 双线性插值，补边 RGB `(128,128,128)`。
2. 转 float32，除以 255。
3. 每通道执行 `(v-mean)/std`，mean=`[0.485,0.456,0.406]`，std=`[0.229,0.224,0.225]`。
4. HWC → CHW → 加 batch 维，保证连续存储。
5. 对两个 logits 做稳定 softmax：`exp(logits-max)/sum(exp(logits-max))`，取 argmax。

输出索引 0=`lcd`，索引 1=`wheel`。组合接口将 `lcd` 映射为 `digital`。这是两类封闭分类，没有 unknown 类、没有内置可读取性判断。

## 3. 三个 YOLO 检测模型共用的规则

area、LCD digits、wheel slots 输入均为 `images [1,3,640,640] float32`，范围 `[0,1]`。

### Letterbox

对于当前待检测图片宽 w、高 h：

```text
r = min(640/w, 640/h)
rw = round(w*r); rh = round(h*r)
left = round((640-rw)/2 - 0.1)
top  = round((640-rh)/2 - 0.1)
sx = rw/w; sy = rh/h
```

采用 Python `round`（恰好半数时取偶数），OpenCV `INTER_LINEAR` 缩放到 `(rw,rh)`；放在 640×640、RGB 114 的画布上。RGB 转 NCHW、float32、除 255，不做 ImageNet 标准化。参考函数 `preprocess()` 接收 BGR 并在内部转 RGB；直接从 RGB 实现时不要再交换两次通道。

反算坐标时使用实际 `sx`、`sy`，两者可能因尺寸取整略有不同；不能一律使用未取整的 r。

### 原始输出、阈值和 NMS

输出布局为 **`[1,4+C,8400]`**。每个候选前四个值为网络输入坐标中的 `cx,cy,width,height`，后 C 个值为各类得分。该导出没有单独 objectness；类别得分无需再做 sigmoid/softmax，也无需乘 objectness。

取最大类别得分和对应 class_id，按阈值 0.25 过滤，转换左上角 xywh 后做 NMS，IoU 阈值 0.45。area 和 LCD 用 class-agnostic NMS；轮位只有一个类别。精确边界处理以 OpenCV `cv2.dnn.NMSBoxes` 为准，分数恰为阈值时可能被其过滤。

保留框转换到当前待检测图坐标：

```text
x1 = (cx-width/2-left)/sx; x2 = (cx+width/2-left)/sx
y1 = (cy-height/2-top)/sy; y2 = (cy+height/2-top)/sy
```

裁到图片边界，舍弃空框，按得分降序输出。应用代码需实现 NMS，ONNX 图内不包含该步骤。

## 4. 整条读数区 meter_area.onnx

输入为完整原图。输出 `output0 [1,5,8400]`，只有 class 0=`area`。area 代表整条读数区域；数字屏和滚轮表共用这个定位模型，类型由整图分类决定。

在 NMS 保留的所有 area 中选择**置信度最高**的框；分数完全相同时选框面积最大的。没有框时输出 `status=no_area`，读数为 null，结束本张图片。当前规则已不是早期的最大面积优先。

- digital：按所选框 floor/ceil 直接从原图裁图，不额外扩大 area。
- wheel：为轮位检测加入上下文。设所选框宽 bw、高 bh，每侧 x 扩 `0.10*bw`，每侧 y 扩 `0.10*bh*7/6`，再 floor/ceil、限制在原图内。得到 `area_crop_xyxy`。

## 5. LCD 数字 lcd_digits.onnx

输入为 digital 分支的 area 裁图，使用第 3 节的预处理。输出 `output0 [1,16,8400]`，12 个类别：0～9 对应同名数字，10=`dot`，11=`minus`。

先对全部 12 类做 class-agnostic NMS，随后只保留 0～9 数字框，按框中心 x 从左到右排列，拼成字符串 `raw_digits`，保留前导零。当前读取不使用 dot/minus 框，不支持根据符号自动处理负值或推断小数位。

显示值固定最后两位为小数：`1013964 → 10139.64`。数字少于 3 位时 `reading=null`，添加 `too_few_digits` 原因。area 接触原图边缘时添加 `window_touches_image_edge`。无原因时 `status=candidate`，有原因时 `status=review`；candidate 只代表生成了候选。

数字框坐标相对实际整数 area 裁图；换到原图时 x 加 `area_crop_xyxy[0]`，y 加 `[1]`。

## 6. 滚轮轮位 wheel_slots.onnx

输入为包含上下文的 wheel area 裁图。输出 `output0 [1,5,8400]`，class 0=`wheel_slot`，检测目标为**完整轮位窗口**。一个轮位里的上下两半数字仍属于同一个框。

采用第 3 节预处理/NMS，舍弃空框后，按中心 x、再中心 y 排序。保留实际检测到的所有轮位，不固定数量，不补缺失位，不按颜色或文件名套布局。通常应为 5 或 6 个框，数量异常时添加 `unusual_slot_count_review_required`，仍保留候选供检查。没有轮位则 `status=no_slots`。

### 轮位实际识别裁剪

1. 将轮位浮点框从 area 坐标平移到原图：加 area 整数裁剪起点。
2. 每侧额外留**该轮位框宽的 8%、高的 10%**。
3. 左/上 floor、右/下 ceil，限制在原图边界。
4. 从原始 RGB 照片裁像素，不从已经裁好的 area 取最终识别图。这样留边可以超出上游 area 范围。

输出中 `xyxy_in_area` 是轮位原始检测框，`xyxy_in_image` 是平移后的原始框，`crop_xyxy_in_image` 是加留边后实际识别裁图的整数原图坐标。

## 7. 滚轮识别 wheel_reader.onnx

VisionState 输入 `cells [N,1,48,32] float32`；宽 32、高 48，N 为从左到右排序后的轮位数。五位和六位可分别以 N=5/6 一次批量推理；该模型只有识别能力，不做定位。

### 每个轮位的输入

1. RGB 每像素取三个通道最小值：`gray=min(R,G,B)`。不能替换为常规亮度加权灰度。
2. Pillow 双线性插值直接缩放到宽 32、高 48，拉伸、不补边。
3. 对缩放后的单张灰度图做 `ImageOps.autocontrast(cutoff=1)`：独立计算每轮整数直方图，分别裁去两端约 1% 像素，按 Pillow 查表/取整拉伸对比度。常量图保持其规则；不能对全部轮位混算直方图。
4. float32 除 255。按轮位顺序组成 `[N,1,48,32]`，范围 `[0,1]`。

输出 `logits [N,100]`，第 b 类表示轮位状态 `b/10`，即 0.0～9.9；不是 100 个不同数字。

### 当前联合解码

对每轮 logits 转 float64、稳定 softmax 得到 p。对 100 个周期位置做相邻容差：

```text
near[b] = p[(b-1) mod 100] + p[b] + p[(b+1) mod 100]
logp[b] = log(max(near[b], 1e-12))
```

按照随包作者代码做机械进位约束动态规划：从右到左，右轮状态 bin 为 b 时，左轮允许状态为 `digit*10 + max(0,b-90)`。最右轮自由变化。尝试共同偏移 s=-2…2，以 `roll(logp,-s)` 解码，总分扣 `0.3*abs(s)` 后选择最高分；相同分数保留先遍历到的结果。该步骤会联合调整各轮状态，不能用逐轮 argmax 拼接替代。

得到 `joint_value_in_last_wheel_units`（以最右轮的一个数字单位计，含该轮转动小数）。当前主字符串为：

```text
raw_digits = zero_pad_N(floor(joint_value) mod 10**N)
```

同时输出独立 argmax 状态 `independent_states`、共同偏移 `common_shift_bins`，以及作者规则 `floor(joint_value+0.5)` 的 `author_rounded_raw_digits`。主读数仍用 floor。显示值固定最后一位为小数，例如 `003733 → 00373.3`。这里的显示小数与轮子转动状态的小数属于不同层次。

所有 wheel 识别结果为 `status=review`。当 N≤1 时保留原始数字但显示值为 null，添加 `too_few_slots_for_decimal_format`。

**已知边界案例**：`1792416424435.jpg` 等图片的参考串为 `003733`，末轮独立状态 3.0，经共同偏移 +1 后联合值 3732.9，floor 得到 `003732`（显示 `00373.2`）；作者 rounded 候选为 `003733`。本包保留原行为，示例见 [wheel_known_floor_issue.json](examples/wheel_known_floor_issue.json)。

## 8. 参考接口 JSON

| 字段 | 含义 |
| --- | --- |
| `image` / `source_sha256` | 文件路径与原始文件哈希；仅 predict_file/CLI 添加 |
| `width`, `height` | 方向校正后的原图尺寸 |
| `meter_type`, `type_scores` | digital/wheel 分类和两类相对分数 |
| `area_candidates`, `area`, `area_best_raw_score` | 全部 NMS area 框、选中的框或 null、阈值过滤前最大分数 |
| `area_crop_xyxy` | 实际整数裁图，原图坐标；无 area 时无此字段 |
| `raw_digits` | 保留前导零的数字字符串；无 area/无轮位时 null，LCD 无数字时空串 |
| `reading` | 加固定小数位后的字符串；不能格式化时 null |
| `status`, `reasons` | no_area/no_slots/candidate/review，以及可观测的检查原因 |
| `digit_detections`, `all_symbols` | LCD 数字及全部符号，坐标相对 area 裁图 |
| `slots` | 从左到右轮位原框、实际裁图坐标、分数、独立状态 |
| `prediction` | wheel 的联合值、主/rounded 字符串、独立状态、共同偏移、logits/probabilities |

完整数组输出由参考 CLI 保存；示例 JSON 为阅读方便省略 prediction 中 logits/probabilities。空 `reasons` 不表示读数一定正确；当前没有可读取性分析模型。

## 9. 微信小程序移植对齐

模型文件可作为 ONNX 资产使用；本包提供的可执行参考为 Python。小程序端尚未接入验证，具体加载方式和执行后端需在实际目标环境验证。

移植后先逐项比较：方向校正后的尺寸与 RGB 通道、分类输入、letterbox 的整数尺寸与 padding、NCHW 索引、area 选框、轮位原图裁剪坐标、48×32 的轮位灰度输入、logits、联合解码结果。避免把 RGBA 直接当 RGB，避免 JavaScript `Math.round` 与 Python `round` 的半数行为差异；Pillow/OpenCV 的插值和 autocontrast 也要核对。

分类、area、LCD 和轮位检测固定 batch=1；只有 wheel_reader 的 N 为动态维度。会话应缓存复用，按分类结果执行对应识别分支；手机内存与耗时需要实机测量。本次桌面核验不代表小程序端性能。
