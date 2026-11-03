# 电表图片推理服务器

Rust + `ort = 2.0.0-rc.15` + `actix-web = 4.17.0`，使用仓库内的五个 ONNX 模型完成整图类型分类、读数区域定位及液晶 / 滚轮识别。纯 Rust 推理，不需要 Python 或 Node.js 运行环境。

## 启动

在项目根目录运行，需要 Rust 1.92 或更新版本和对应平台的 C/C++ 链接工具（Windows 使用 MSVC 工具链）：

```powershell
cargo run --release
```

打开 **http://127.0.0.1:8080**。首次构建会下载 Rust 依赖和 CPU 版 ONNX Runtime；Windows 发布时需同时携带构建输出目录中的 ONNX Runtime DLL，以及整个 `models` 目录。网页资源编译进可执行文件，修改 `web/` 后需重新构建。

可选环境变量：

```powershell
$env:BIND_ADDR = "127.0.0.1:8080"
$env:MODEL_DIR = "models"
$env:ORT_THREADS = "12"
$env:ORT_SPINNING = "true"
$env:RUST_LOG = "info"
cargo run --release
```

`MODEL_DIR` 默认相对于运行时工作目录。若需要局域网访问，可将 `BIND_ADDR` 改为 `0.0.0.0:8080`。当前接口没有登录鉴权。

模型在启动时加载一次并复用，启动时核对 manifest 中的大小、SHA256、输入输出名称、数据类型与形状，任一不符则启动失败。推理在 Actix 阻塞任务中执行，共享一组会话顺序处理；最多接纳四个上传 / 排队请求，额外请求返回 429。

五个模型共享一个 ORT 线程池，避免每个模型单独创建工作线程。`ORT_THREADS` 默认取系统报告的可用逻辑处理器数，最多 12；`ORT_SPINNING` 默认为 `true`，适合连续推理，可设为 `false` 降低空闲等待时的 CPU 消耗。线程数越多不一定越快，建议用下面的基准命令比较 4、8、12 线程。

## DirectML 候选后端（Windows）

CPU 仍为默认后端。DirectML 通过可选 feature 构建，依赖版本、原始模型和 manifest 不变。建议使用独立输出目录：

```powershell
cargo build --release --features directml --target-dir target/directml
$env:ORT_BACKEND = "directml-hybrid"
$env:ORT_THREADS = "1"
$env:ORT_SPINNING = "false"
& .\target\directml\release\electric-meter-reader-inference-ort.exe
```

打开同一个网页 http://127.0.0.1:8080。如果旧服务仍在运行，可用 `BIND_ADDR` 换一个端口。发布时携带该输出目录中的实际 `DirectML.dll` 文件（构建目录内可能是指向缓存的符号链接，分发时应复制其目标内容），以及 `models` 目录。

| `ORT_BACKEND` | 区域 / 液晶 / 轮位检测 | 表型分类 / 滚轮识别 |
| --- | --- | --- |
| `cpu`（默认） | CPU | CPU |
| `directml` | DirectML | DirectML |
| `directml-hybrid` | DirectML | CPU |

混合模式用于比较小模型留在 CPU 是否更快。DirectML 模式默认 `ORT_THREADS=1`、`ORT_SPINNING=false`；显式设置的环境变量优先。未设置 `ORT_DML_DEVICE_ID` 时请求高性能适配器；可用非负 DXGI 适配器索引显式选择设备，该索引不等同于 `nvidia-smi` 的 GPU 编号。非 Windows、未启用 feature 或提供器注册失败时明确报错。

按照 [ONNX Runtime DirectML 要求](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)，DML 会话禁用 memory pattern，并采用顺序执行。共享会话的访问仍由互斥锁串行保护。DML 内不支持的节点允许回退 CPU；启动日志和健康接口的 `runtime.model_providers` 仅表示会话注册策略。

用执行记录验证实际节点的提供器：

```powershell
$env:ORT_PROFILE_DIR = ".work/dml-profile"
& .\target\directml\release\electric-meter-reader-inference-ort.exe --benchmark datas/lcd/1792403667245.jpg 3
Remove-Item Env:ORT_PROFILE_DIR
# 正式测速关闭 profiling，预热后测量 30 次
& .\target\directml\release\electric-meter-reader-inference-ort.exe --benchmark datas/lcd/1792403667245.jpg 30
```

基准 JSON 的 `execution_profiles[].provider_execution_events` 汇总实际的 `DmlExecutionProvider` / `CPUExecutionProvider` 事件；未执行的分支为空。逐模型 Chrome trace JSON 保存在指定目录。Profiling 仅供 CLI 使用，会影响计时，不应开启后进行性能比较或启动 HTTP 服务。首次模型加载和特定输入形状的首次运行可能较慢，基准先预热五次。

本机 i7-14650HX、RTX 4060 Laptop GPU，2026-10-20 用同一个 release 候选程序串行比较，原始 3072×4096 JPEG 样本各预热 5 次、测量 30 次，关闭 profiling。CPU 使用 12 线程且 spinning=true，DML 两种模式使用 1 线程且 spinning=false：

| 后端 | 液晶推理平均 / P95 | 滚轮推理平均 / P95 | 液晶 / 滚轮推理 FPS | 液晶 / 滚轮含解码平均 |
| --- | --- | --- | --- | --- |
| CPU | 52.15 / 62.46 ms | 51.94 / 69.04 ms | 19.17 / 19.25 | 109.65 / 134.83 ms |
| DirectML | 22.79 / 25.43 ms | 21.97 / 24.52 ms | 43.87 / 45.52 | 75.16 / 88.51 ms |
| DirectML hybrid | 20.66 / 23.57 ms | 21.44 / 24.14 ms | 48.40 / 46.63 | 68.43 / 88.35 ms |

两张样本读数仍为 `10139.64`、`00373.2`。独立开启 profiling 的单图检查覆盖两种分支，五个模型的实际执行事件均出现 `DmlExecutionProvider`，未出现 CPU 节点事件；这不保证其他输入也完全在 GPU 执行。推理部分超过 30 FPS，**包含原始 JPEG 解码仍未达到 30 FPS**；不含 HTTP、上传或前端显示，也不是连续相机帧的实测帧率。每台机器、负载和输入都需要重新测试。

候选验证：`directml` 与 `directml-hybrid` 分别回放本地 104 张图片，`raw_digits`、`meter_type`、`status` 均与此前 FP32 CPU 对照一致；浮点分数和框坐标不保证逐位一致，也不是独立标注的准确率评估。混合模式的健康接口及液晶、滚轮两图批量 HTTP 请求均返回成功，读数一致。默认 CPU 构建和 DirectML feature 构建均通过 Clippy（所有 target、禁止 warnings）及九个单元测试；未进行网页自动化测试。

## 网页

- 多选或拖拽 JPG / JPEG / PNG / WebP / BMP；液晶、滚轮照片可以混合上传。
- 自动按每组四张请求批量接口，显示进度与每张照片的结果。
- 显示读数区域和数字 / 轮位框，可关闭标注。
- 显示保留前导零的原始数字、格式化读数、类型、耗时和复核状态。
- 照片 Canvas 内绘制文件名、读数、类型、状态、原始数字、数字 / 轮位数量、尺寸、阶段耗时、复核原因及滚轮备选；信息栏位于照片下方，不遮挡读数区域。
- 点击卡片或详情中的图片放大，支持上一张 / 下一张、键盘左右方向键、放大缩小、适应窗口和 Esc 关闭；按当前筛选范围翻页。
- 查看各阶段耗时、完整 JSON、滚轮四舍五入备选；筛选结果、移除照片、失败重试及导出 JSON。
- 上传文件仅在内存中处理，不保存到服务端磁盘。页面刷新后清空当前队列。

单张文件上限 **16 MiB**、**2400 万像素**，宽高各不超过 **12000**；单次批量请求最多 **20 张 / 64 MiB 文件内容**。前端通过 `URL.createObjectURL(file)` 显示本地图片，浏览器处理图片方向，并用 Canvas 叠加检测框；预览绘制到最长边 1280。后端只返回识别结果与坐标，不生成或回传预览，模型仍使用方向校正后的原始图像。

## HTTP 接口

### 单张图片

`POST /api/predict`，`multipart/form-data` 文件字段 `image`。响应仅包含结构化识别结果，不包含图片。

```powershell
curl.exe -F "image=@datas/lcd/1792403667245.jpg" http://127.0.0.1:8080/api/predict
```

返回结构（节选）：

```json
{
  "bundle_version": "2026-10-20-current-v3",
  "filename": "1792403667245.jpg",
  "result": {
    "meter_type": "digital",
    "raw_digits": "1013964",
    "reading": "10139.64",
    "status": "candidate",
    "reasons": []
  }
}
```

结果还包含原始文件 SHA256、方向校正后的宽高、分类分数、所有 area 候选及选中框、实际裁剪坐标、数字 / 轮位框、滚轮独立状态与联合预测、logits / probabilities 和阶段耗时。具体字段与坐标系见 [模型流程说明](models/PIPELINE.md)。`elapsed_ms` 从解码完成后开始，包含预处理、推理、后处理，不包含上传、排队、图像解码或前端预览绘制。

`model_timings_ms` 单独记录各模型调用耗时（含张量包装、输出检查和输出复制），已包含在 `timings_ms` 对应阶段内，不能重复相加。分类预处理与源文件 SHA256 在一个辅助线程中和区域检测重叠执行；阶段耗时之和不等于总墙钟耗时。

### 批量图片

`POST /api/batch`，重复添加 `images` 文件字段（也接受 `files` 或 `image`）：

```powershell
curl.exe -F "images=@datas/lcd/1792403667245.jpg" -F "images=@datas/wheel/1792403667565.jpg" http://127.0.0.1:8080/api/batch
```

```json
{
  "bundle_version": "2026-10-20-current-v3",
  "total": 2,
  "succeeded": 1,
  "failed": 1,
  "elapsed_ms": 1234.5,
  "results": [
    {"filename": "a.jpg", "result": {"meter_type": "digital", "reading": "10139.64"}},
    {"filename": "b.jpg", "error": "图片无效: ..."}
  ]
}
```

上面为响应结构示意，非该命令的实测结果。结果顺序与上传顺序一致；批量内单张损坏图片或单张推理失败会记录到该项 `error`，继续处理其余图片。无 area / 无轮位是正常推理结果，计入 `succeeded`，不表示读数正确。整个上传格式错误、文件超限等会拒绝该请求。

主要 HTTP 状态：200 处理完成，400 文件字段错误 / 空上传 / 单图解码失败，413 文件或请求内容超限，429 服务繁忙，500 推理异常。批量逐项失败通过 `results[].error` 表达。

### 健康状态

`GET /api/health` 返回模型加载状态、版本、后端、`runtime` 配置与上传限制。该接口只在启动加载模型成功后可用。

## 命令行

```powershell
# 仅校验并加载五个模型
cargo run --release -- --check-models

# 直接识别单张图片，向标准输出打印 JSON
cargo run --release -- --predict datas/lcd/1792403667245.jpg

# 预热 5 次后，串行测量 30 次；每次重新解码并执行全部推理阶段
cargo run --release -- --benchmark datas/lcd/1792403667245.jpg 30
```

基准 JSON 的 `inference` 为解码后的完整流水线，`decode_and_inference` 包含逐次图片解码；均报告平均值、P50、P95、最小值、最大值及按平均耗时换算的 FPS。`stage_mean_ms`、`model_mean_ms` 用于定位瓶颈。文件仅在开始时从磁盘读取一次；模型初始化、磁盘读取、上传、HTTP、JSON 序列化和页面绘制不计入这些指标。重复同一张照片用于测量运行速度，不能替代连续相机帧或完整数据集验收。

## 识别行为与验证边界

- EXIF 方向校正后，从完整 RGB 图像开始分类与 area 检测，再按类型进入对应识别分支；YOLO 输出执行应用层 NMS。
- 图像解码、EXIF 校正与裁剪使用 `image`，缩放使用 `fast_image_resize` 自动选择 CPU SIMD 指令。分类与轮位采用抗锯齿双线性卷积，检测采用固定核双线性插值与 letterbox，保留独立 x/y 实际缩放比例；轮位缩放直接处理单通道灰度图。
- 与此前手写缩放对比，20 组人工图像缩放及两张实图的分类 / 整图检测预处理最大像素差为 1（0–255）；两张样本最终读数一致。这不代表与 Python 逐像素一致，尚未完成全数据集精度回归。
- 液晶表最后两位、滚轮表最后一位为配置约定的小数位，符号框不会自动决定小数位或负号。
- 滚轮使用机械进位约束动态规划及共同偏移搜索；主读数保持原流程的 **floor** 规则，另提供 `author_rounded_raw_digits`。已知 `003733 → 003732` 问题保留，所有滚轮读数均标记 `review`。异常轮位数量会提示复核；超过 32 个轮位则拒绝本张推理以限制异常输出的资源使用。
- `candidate` 表示已生成候选，分类分数和检测分数不等于读数正确率。没有自动判断照片是否可读的模型。

源码结构：

```text
src/main.rs        启动、配置与单图 CLI
src/benchmark.rs   串行基准、解码与推理分项统计
src/runtime.rs     CPU / DirectML 选择、设备配置与 profiling
src/server.rs      上传、资源限制、批量处理和 HTTP 路由
src/pipeline.rs    模型校验、会话复用、检测解码和分支流程
src/preprocess.rs  解码、EXIF、缩放、归一化、裁图
src/wheel.rs       滚轮联合解码与读数格式化
web/               内嵌网页，无前端构建依赖
models/            原模型、manifest 与 Python 参考流程
```

代码检查命令：

```powershell
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
# Windows DirectML 候选也单独检查
cargo clippy --features directml --target-dir target/directml --all-targets -- -D warnings
cargo test --features directml --target-dir target/directml
```

已通过九个单元测试、五个模型的加载与契约检查；液晶、滚轮样本分别得到 `10139.64`、`00373.2`。后续独立标注的精度验证、批量 HTTP 与网页交互验收留给你执行。没有引入 Playwright。

本机 release、CPU 四线程下的有限性能对比：两张样本的分类预处理由手写实现的约 68–74 ms 降到约 3.4–3.6 ms；换库后每张运行三次，液晶完整推理约 78–87 ms，滚轮约 76–81 ms，读数分别为 `10139.64`、`00373.2`。计时不含上传、排队、图像解码与前端绘制，不代表其他机器或整个数据集的性能。

后续 Rust 优化保持原有五个 FP32 模型、640×640 检测输入及后处理逻辑：共享 ORT 线程池；复用 SIMD 缩放临时区、RGB 图像和 NCHW 输入缓冲；融合填充、通道转换与归一化；借用输入缓冲创建 ORT 张量；分类预处理和文件哈希与区域检测重叠执行。10 组人工图像预处理对照与优化前 SIMD 版本的输入张量逐项一致。

本机 i7-14650HX、12 线程、spinning=true，重复已解码样本各预热 5 次、测量 30 次，最后一轮液晶平均 56.67 ms（P95 89.25 ms），滚轮平均 47.76 ms（P95 53.91 ms），约 17.65 / 20.94 FPS，**尚未达到 30 FPS**。两次检测模型调用合计约 40–44 ms。此前四线程版本在本轮调查中约 72–73 ms；不同轮次、整机负载及线程配置会影响比较，不能把热测换算帧率当作稳定实时帧率。

逐次解码原始 3072×4096 JPEG 的另一次 30 次测试波动较大：液晶解码加推理平均约 124 ms，滚轮约 295 ms；其中推理平均约 60 / 153 ms。该结果包含整机调度和图片解码的影响，不能用热测约 50 ms 代替实际端到端延迟。请用 `--benchmark` 在部署机器上测量并观察 P95。

还在项目临时目录中试验了独立的 Conv INT8 S8S8 QDQ 模型（opset 13，逐通道权重，每个检测模型 24 张本地图片校准，输出头保留部分 FP32）。本机 12 线程热测约 83–87 ms，104 张回放中有 11 张的原始数字、状态或类型与 FP32 对照不同，未采用；原始模型及 manifest 保持不变。这些变化不是独立标注的准确率评估。
