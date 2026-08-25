# Addendum 证据：fps_cache 档（现役 FPS ×1 + PrefixFPS ×3）

- 时间: 2026-08-19
- 环境: python=/home/wangpeng/miniforge3/envs/hpenet/bin/python | torch=2.2.2+cu118 | TRT=8.6.1.6 | onnx=1.17.0 | nsys 2022.4.2 | GPU=NVIDIA L20（测量时全部 8 卡空闲，util 0%）
- checkpoint: log/radar/radar-train-hpenet-ll-ngpus1-20260812-201051-gHBQ4DMy5jP2fZfkGkFcue/checkpoint/*_ckpt_best.pth
- cfg: cfgs/radar/hpenet-ll.yaml | strides=[1,2,2,2,2] | 采样节点 4 个均 stride=2
- 前置：task-4b/6 的 cache-only（FlashFPS k1.0）已实证前缀等价性（ti10 acc 0.9741 逐文件一致）；fps_cache 换现役 FPS kernel 为 carrier

## 1. 代码改动（2 处，零插件/kernel 改动）

### (a) `deploy/onnx_backend.py` `patch_model_for_onnx` 新增 `'fps_cache'` 档
- 校验分支（onnx_backend.py:218-223）：`fps_factory=None`，`fps_tag='make_fps_op (stage 1) + make_prefixfps_op (stage 2-4)'`
- sample_fn 替换循环（onnx_backend.py:269-295）：**与 flashfps 档同判据**（模块名末两段 `<stage>.<subidx>`，subidx==0，如 `encoder.encoder.1.0`）：
  - `stage==1` → `make_fps_op(stride)`（**现役 FPS 插件原样**，种子 0 + 按选择序输出，作为 cache carrier）
  - `stage>=2` → `make_prefixfps_op(stride)`（PrefixFPS，arange 前缀）
  - named_modules() 深度优先按注册序 → stage 索引升序，判据与遍历顺序一致（注释已写明）
- 实测 patch 计数：`Patched 1 SA.sample_fn → FPS + 3 SA.sample_fn → PrefixFPS`
- **fps 默认路径零变化**：'fps'/'samplefps' 档走原 make_fps_op/make_samplefps_op 分支，代码路径未动（不需要重验）

### (b) `deploy/onnx_export.py` `--fps-algo` choices 扩为 `fps/samplefps/flashfps/fps_cache`
- choices=['fps', 'samplefps', 'flashfps', 'fps_cache']，默认 'fps'（零行为变化），透传 patch_model_for_onnx

## 2. 导出与图验证

导出命令（照 task-3 evidence，--no_simplify）：
```
python deploy/onnx_export.py --checkpoint <ckpt> --cfg cfgs/radar/hpenet-ll.yaml \
  --output deploy/fps_algo_fps_cache.onnx --fps-algo fps_cache --no_simplify
```

### 图断言（MUST DO，机械比对 op_type+domain）
```
fps nodes=628  fps_cache nodes=628
raw op_type diffs: 3
  [125] fps=FPS  fps_cache=PrefixFPS
  [331] fps=FPS  fps_cache=PrefixFPS
  [455] fps=FPS  fps_cache=PrefixFPS
fps graph: FPS=4 | fps_cache graph: FPS=1 PrefixFPS=3
non-sampler node count identical: 625
```
- **1×`hpenet::FPS`{stride:2}（node[1] = encoder.1 首级）+ 3×`hpenet::PrefixFPS`{stride:2}（node[125]/[331]/[455] = encoder.2-4）**
- 非 hpenet 部分 625 节点与 fps 图 **逐节点一致**（op_type + domain）
- TRT parse：插件层 `/model/encoder.1/encoder.1.0/FPS` + `/model/encoder.2/encoder.2.0/PrefixFPS` + `/model/encoder.3/encoder.3.0/PrefixFPS` + `/model/encoder.4/encoder.4.0/PrefixFPS`，每个 PrefixFPS 输入 = 上一级 GatherElements 输出（前缀语义拓扑正确）

## 3. Engine 构建

同一 profile（照 task-6）：`--min_n 1024 --opt_n 4096 --max_n 10000 --workspace 4 --num_input_features 5`

| engine | 精度 | 大小 | 结果 |
|---|---|---|---|
| fps_algo_fps_cache_fp32.engine | fp32 | 15.0 MB | exit=0 |
| fps_algo_fps_cache_fp16.engine | fp16 | 10.1 MB | exit=0 |

deserialize + 端到端推理冒烟：N=1024/4096/6500 输出 shape=(1,2,N)、值域有限，全部 PASS。

## 4. (c) acc 测试（ti10 协议，10 文件 0000068..0000077）

### 4.1 per-file acc（对比基准：现役新 build fps engine，task-6 产物）

| 文件 | fps fp32 | fps_cache fp32 | fps_cache fp16 |
|---|---|---|---|
| 0000068 | 0.9854 | 0.9854 | 0.9854 |
| 0000069 | 0.9446 | 0.9446 | 0.9446 |
| 0000070 | 0.9862 | 0.9862 | 0.9865 |
| 0000071 | 0.9536 | 0.9536 | 0.9533 |
| 0000072 | 0.9826 | 0.9826 | 0.9826 |
| 0000073 | 0.9596 | 0.9596 | 0.9596 |
| 0000074 | 0.9749 | 0.9749 | 0.9749 |
| 0000075 | 0.9891 | 0.9891 | 0.9891 |
| 0000076 | 0.9801 | 0.9801 | 0.9797 |
| 0000077 | 0.9843 | 0.9843 | 0.9843 |
| **mean** | **0.9741** | **0.9741** | 0.9740 |

- **fps_cache fp32 per-file acc 与现役逐文件一致（mean 0.9741）** —— 前缀等价性实证（见 §4.2）
- fps_cache fp16 = 0.9740（0000070/71/76 有 ±0.0003-0.0004 fp16 数值噪声，与 task-6 fps fp16 同模式）

### 4.2 前缀等价性（单算子级，MUST DO 归因）

ti10 全部 10 文件的**部署子云**（preprocess_test+preprocess_subcloud 真实输入）上，链式现役 FPS（L1→gather→L2→gather→L3→gather→L4，与 encoder 结构一致）与 PrefixFPS 语义对拍：
```
subcloud-level checks: 147, prefix mismatch: 0, genuine diff: 0
L2-4 chained FPS output == arange 前缀: 147/147 精确相等
L1 FPS 种子 = 0（首点索引）: 全部 True
```
- **L2-4 现役 FPS 输出与 arange 前缀 bit 级相等**（0 mismatch / 0 genuine diff）——fps_cache 的 PrefixFPS 与现役 FPS 深层采样索引完全一致，acc 一致性是精确等价而非平局豁免
- 端到端 logits 对拍（ti10 逐文件汇总 logits）：fps_cache fp32 vs fps fp32 非逐位（max_abs 4e-3..1e-2）——TRT tactic 选择漂移（同 task-6 fps-vs-incumbent 1.5e-2 归因量级），**索引本身 bit 级一致（§4.2），acc 逐文件一致（§4.1）**

## 5. (d) 延迟

### 5.1 端到端纯 engine 延迟（ti10，49 subcloud runs，含 H2D/D2H，空闲 GPU）

| engine | median (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|
| fps fp32（现役等价） | 7.222 | 8.216 | 7.269 |
| fps fp16 | 7.220 | 8.218 | 7.260 |
| **fps_cache fp32** | **5.486** | **6.100** | **5.506** |
| fps_cache fp16 | 5.506 | 6.682 | 5.606 |

- **fps → fps_cache：median -24.0%（7.22 → 5.49ms）**，相对降幅与 task-6 预期（28.8 → ~22ms）一致
- ⚠️ 绝对量级说明：task-6 的 28.8ms 基线是在 6 个并行 engine build 占卡时测得（GPU 争用）；本次在 8 卡全部空闲下重测 fps=7.2ms。**相对对比（同条件同脚本）为准**，task-6 表中其余算法数据（samplefps 134.5 / flashfps 43.7ms）同为争用环境，端到端并列对比见 §6

### 5.2 nsys 复测 FPS 段（-t cuda，2 文件 0000001+0000006，13 subclouds，无 warmup）

| engine | FPS 段 kernel | FPS 总耗时 | GPU 总耗时 | FPS 占比 | per-subcloud FPS |
|---|---|---|---|---|---|
| fps fp32 | furthest_point_sampling_kernel ×52（39×<1024> + 13×<512>） | 83.16ms | 124.39ms | **66.9%** | 6.40ms |
| **fps_cache fp32** | furthest_point_sampling_kernel ×13 + prefix_fill_kernel ×39（≈0.08ms） | **50.97ms** | 92.32ms | **55.2%** | **3.92ms** |

- **FPS kernel launch 次数 52 → 13（MUST DO，13 subclouds × 4 级 → 每 subcloud 仅第 1 级）**
- PrefixFPS fill ×39（≈0.08ms，可忽略）——前缀等价性的代价为零距离计算
- FPS 段 -38.6%（83.16 → 50.97ms）；per-subcloud FPS 3.92ms **达到 task-6 的 <5ms 目标**（现役 8.1ms/subcloud 未达）
- 注：本工作负载 subcloud N≈4872-5308，小于 task-6 nsys 负载（~6500），故绝对量级低于 task-6 的 105.7ms；同负载两 engine 相对对比为准

## 6. 与三算法并列对比（task-6 表扩列）

### 端到端 median（task-6 争用环境数字 + 本次空闲 GPU fps/fps_cache 同条件对）

| engine | median (ms) | p99 (ms) |
|---|---|---|
| samplefps fp32（task-6） | 134.5 | 352.8 |
| flashfps fp32（task-6） | 43.7 | 159.5 |
| fps fp32（task-6 争用） | 28.8 | 29.4 |
| fps fp32（本次空闲） | 7.22 | 8.22 |
| **fps_cache fp32（本次空闲）** | **5.49** | **6.10** |

### nsys FPS 段（13 subclouds，负载 subcloud N 量级有差异，relative 为准）

| engine | FPS 段 kernel | FPS 总耗时 | GPU 总耗时 | FPS 占比 |
|---|---|---|---|---|
| samplefps fp32（task-6） | samplefps_iterate ×52 | 1108.4ms | 1142.6ms | 97.0% |
| flashfps fp32（task-6） | samplefps_iterate ×13 | 299.4ms | 362.9ms | 82.5% |
| fps fp32（task-6） | furthest_point_sampling ×52 | 105.7ms | 173.3ms | 61.0% |
| fps fp32（本次同负载） | furthest_point_sampling ×52 | 83.16ms | 124.39ms | 66.9% |
| **fps_cache fp32（本次同负载）** | furthest_point_sampling ×13 + prefix_fill ×39 | **50.97ms** | 92.32ms | **55.2%** |

- fps_cache 在 **保留现役 FPS 精度的前提下**（acc == 现役 0.9741 逐文件一致），FPS 段 kernel launch 减少 75%（52→13），FPS 段耗时较同负载 fps 降 38.6%，端到端 median 降 24.0%
- 相对 samplefps（1108.4ms）FPS 段 -95.4%；相对 flashfps（299.4ms）FPS 段 -83.0%，且 **无精度损失**（flashfps 0.9707 vs 现役 0.9741）

## 7. 结论

1. **实现**：fps_cache 档（stage 1 → 现役 FPS，stage 2-4 → PrefixFPS），判据与 flashfps 档一致；fps/samplefps/flashfps 档零改动
2. **图**：628 节点，1×FPS + 3×PrefixFPS，非 hpenet 部分与 fps 图逐节点一致；TRT 插件层落位正确（encoder.1 FPS + encoder.2-4 PrefixFPS）
3. **engine**：fp32/fp16 同 profile 构建成功，冒烟全过
4. **acc**：fps_cache fp32 ti10 **逐文件一致 0.9741**；前缀等价性 147/147 subcloud 精确成立（L2-4 现役 FPS == arange，0 genuine diff）——非平局豁免而是精确等价
5. **延迟**：端到端 median 7.22→5.49ms（-24%，同条件）；FPS 段 83.16→50.97ms（-38.6%），launch 52→13，PrefixFPS fill 39≈0；per-subcloud FPS 3.92ms **达 <5ms 目标**（现役 8.1ms 未达）
6. **定位**：fps_cache = 「现役精度 + cache 速度」——在四算法中唯一同时满足 acc==现役 0.9741 且 FPS 段 <5ms/subcloud 的档位

## 8. 未做（遵守 MUST NOT DO）
- 未改任何插件/kernel（fps/samplefps/flashfps/prefixfps 全部原样）；未改 checkpoint/dataset
- 未做 git 操作
- nsys 仅用 `-t cuda`（未加 cudnn/cublas）
