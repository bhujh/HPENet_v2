# fps_cache_prune 档（现役 FPS kernel + Prune + Cache）端到端实测证据

- 时间: 2026-08-19 | 阶段 B
- 环境: python=/home/wangpeng/miniforge3/envs/hpenet/bin/python | torch=2.2.2+cu118 | TRT=8.6.1.6 | onnx=1.17.0 | nsys 2022.4.2 | GPU=NVIDIA L20 ×8（测量时全部空闲，util 0%）
- checkpoint: `log/radar/radar-train-hpenet-ll-ngpus1-20260812-201051-gHBQ4DMy5jP2fZfkGkFcue/checkpoint/*_ckpt_best.pth`
- cfg: `cfgs/radar/hpenet-ll.yaml`（strides=[1,2,2,2,2] 已确认）| 采样节点 4 个均 stride=2
- 前置锚点: 阶段 A torch 模拟（现役 FPS + Prune + Cache，`samplers.py::prune_fill` ascending）：ti10 k0.75 acc=0.9707、全量 k0.75 acc=0.9558

## 1. 实现（新增 5 文件 + 注册/CMake/ONNX 接线，现役文件零改动）

### (a) TRT 插件 `FPSPrune`（独立类，type="FPSPrune" version="1"）
- `deploy/trt_plugins/src/fpsprune_plugin.cpp` + `include/fpsprune_plugin.h`：属性 `stride`(int) + `keep_rate`(float)，独立 `FPSPrunePlugin` 类（不复用 FPS/FlashFPS 类）
- `deploy/trt_plugins/src/fpsprune_kernel.cu` + `include/fpsprune_kernel.h`：新写 `fill_unselected_kernel`（升序未选中尾部填充）
- 语义（enqueue，照 FPS_Prune / prune_fill）：
  ```
  M = N/stride;  N_points = int(keep_rate*N);  sample_rate = N/M;  num_points = N_points/sample_rate
  if num_points >= M:  # exact 路径（keep_rate≈1），逐字复用现役 FPS enqueue
      fill_kernel(temp, B*N, 1e10); fps_launcher_with_stream(B, N, M, xyz, temp, idx, stream)
  else:  # prune 路径
      fill_kernel(temp, B*N_points, 1e10)
      fps_launcher_with_stream(B, N_points, num_points, xyz, temp, idx, stream)
      fill_unselected_kernel(idx, num_points, M, N_points, sel, ...)  # 升序填充 idx[num_points..M)
  ```
- `getOutputDimensions`: `kFLOOR_DIV(N, stride)` → [B, M]（与 keep_rate 无关）
- `supportsFormatCombination`: pos0 FLOAT / pos1 INT32（照现役 fps_plugin.cpp）
- `getWorkspaceSize`: `maxB*maxN*float`（temp）+ `maxB*maxN` bytes（selected 位图）
- `serialize`: 存 `stride`(int) + `keep_rate`(float)，deserialize 按序读回
- keep_rate 是 **attribute**（不是 runtime input）
- fill_unselected 语义与 samplefps_kernel.cu 尾部填充**严格一致**：位图标记 idx[0..num_points) 已选 → 扫描 [0,M) 升序收集未选中 → 容量截断 T=M-num_points；num_points==0 → arange(M)；选中点落在 [M,N_points) 不占 [0,M) 名额（仅需 ≥T 个未选中即满）

### (b) 注册与构建
- `plugin_registry.cpp` 追加 `REGISTER_TENSORRT_PLUGIN(FPSPrunePluginCreator)`（第 9 个，末尾）
- `CMakeLists.txt` 加 `src/fpsprune_kernel.cu` + `src/fpsprune_plugin.cpp`
- `cd deploy/trt_plugins/build && make -j` → libhpenet_plugins.so 编译零错误；nm 确认 `FPSPrunePluginCreator` + `fill_unselected_kernel` 存在，FPS/BallQuery/…/PrefixFPS 原 8 插件仍在

### (c) ONNX 接线
- 新建 `deploy/onnx_ops/fpsprune_op.py`：`symbolic` 发 `hpenet::FPSPrune`，attrs `stride_i` + `keep_rate_f`，`make_fpsprune_op(stride, keep_rate=0.75)`
- `deploy/onnx_backend.py` 加 `'fps_cache_prune'` 档：`patch_model_for_onnx(model, fps_algo, keep_rate=0.75)` —— stage 1 → `make_fpsprune_op(stride, keep_rate)`，stage 2-4 → `make_prefixfps_op(stride)`（与 fps_cache 档同判据 `encoder.encoder.{1..4}.0`）
- `deploy/onnx_export.py`：`--fps-algo` choices 加 `fps_cache_prune`，新增 `--keep-rate`（默认 0.75）透传
- 导出日志：`Patched 1 SA.sample_fn → FPSPrune(keep_rate=0.75) + 3 SA.sample_fn → PrefixFPS`

## 2. 单元验证（新 kernel 独立数值测试，5 用例全过）

独立 CUDA 测试 `/tmp/opencode/fpsprune_verify/test_fill_unselected.cu`（含参考语义 host 实现对拍）：

| case | 场景 | 结果 |
|---|---|---|
| 1 | 基础 prune 路径（选中全在 [0,M)） | PASS |
| 2 | 选中落在 [M,N_points)（截断到 750） | PASS |
| 3 | num_points==0 → arange(M) | PASS |
| 4 | B=2 逐 batch 偏移 | PASS |
| 5 | 选中>M 混合、小 M | PASS |

插件级端到端（`/tmp/opencode/fpsprune_verify/verify_plugin_trt.py`：小 ONNX → TRT engine → 与 torch `prune_fill` 逐位对拍）：keep_rate∈{0.75,1.0} × N∈{6000,4096,5000} 全部 **bit-exact 相等**（含 keep_rate=1.0 exact 路径 == 现役 FPS）。

## 3. 图断言（MUST DO，机械比对 op_type+domain）

```
fps_cache nodes=628  fps_cache_prune nodes=628
prune op counts: ... FPSPrune:1, PrefixFPS:3, BallQueryGroup:4, BallQueryDP:4, ThreeInterp:5 ...
non-hpenet node count: 611 / 611  identical_in_order=True
fps_cache hpenet: [FPS@1, PrefixFPS@125/331/455, ...]
prune hpenet:     [FPSPrune@1, PrefixFPS@125/331/455, ...]   # node[1] FPS → FPSPrune
FPSPrune attrs: {'keep_rate': 0.75, 'stride': 2}
```
- **1×`hpenet::FPSPrune`{stride:2, keep_rate:0.75}（node[1]=encoder.1 首级）+ 3×`hpenet::PrefixFPS`{stride:2}（node[125]/[331]/[455]=encoder.2-4）**
- 非 hpenet 611 节点与 fps_cache 图**逐节点一致**（op_type+domain 顺序全同）；fps_cache 图 FPS@1 换成 FPSPrune@1，其余不变

## 4. Engine 构建

同一 profile：`--min_n 1024 --opt_n 4096 --max_n 10000 --workspace 4 --num_input_features 5`

| engine | 精度 | 大小 | 结果 |
|---|---|---|---|
| fps_algo_fps_cache_prune_fp32.engine | fp32 | 13.8 MB | exit=0 |
| fps_algo_fps_cache_prune_fp16.engine | fp16 | 10.5 MB | exit=0 |

TRT parse 插件层：`/model/encoder.1/encoder.1.0/FPSPrune` + 3×`/model/encoder.{2,3,4}/encoder.{2,3,4}.0/PrefixFPS`，每个 PrefixFPS 输入 = 上一级 GatherElements 输出（前缀语义拓扑正确）。

## 5. acc（双口径 × 双精度，与阶段 A 锚点交叉验证）

### 5.1 ti10（10 文件 0000068..0000077，test 20% 尾部）

| 引擎 | mean acc | 阶段 A 锚点 | 判定 |
|---|---|---|---|
| fps_cache_prune fp32 | **0.970691** | 0.9707 | ✓ |
| fps_cache_prune fp16 | 0.970550 | 0.9707 | ✓（fp16 数值噪声模式同 fps_cache） |

### 5.2 全量 339 文件（sorted，含训练/验证文件）

| 引擎 | mean | median | min | 阶段 A 锚点 | 判定 |
|---|---|---|---|---|---|
| fps_cache_prune fp32 | **0.955761** | 0.958652 | 0.881982 | 0.9558 (±0.001) | ✓ |
| fps_cache_prune fp16 | 0.955765 | 0.958639 | 0.881982 | 0.9558 (±0.001) | ✓ |

### 5.3 逐文件对拍阶段 A 锚点（fp32，抽样 10 文件）

max dev = 0.000398；7/10 文件 bit 精确相等（0000001/57/68/75/77/106/295 全等），0000048 dev=0.000208、0000239 dev=0.000398、0000335 dev=0.000000。TRT 实现**逐文件复现** torch prune-fill 语义。

## 6. 延迟

### 6.1 nsys FPS 段（-t cuda，2 文件 0000001+0000006，13 subclouds，无 warmup，同负载同会话三引擎并列）

| engine | FPS 段 kernel | FPS 段总耗时 | GPU 总耗时 | FPS 占比 | per-subcloud FPS |
|---|---|---|---|---|---|
| fps fp32 | furthest_point_sampling ×52 | 91.45ms | 136.00ms | **67.2%** | 7.03ms |
| fps_cache fp32 | furthest_point_sampling ×13 + prefix_fill ×39 | 50.91ms | 92.47ms | **55.1%** | 3.92ms |
| **fps_cache_prune fp32** | furthest_point_sampling ×13 + fill_unselected ×13(≈0.07ms) + prefix_fill ×39 | **35.64ms** | 80.06ms | **44.5%** | **2.74ms** |

- FPS kernel launch 52 → **13**（同 fps_cache；每 subcloud 仅第 1 级 FPS，且 n=N_points、m=num_points 更小）
- **fps_cache_prune vs fps：FPS 段 -61.0%、GPU 总 -41.1%；vs fps_cache：FPS 段 -30.0%**
- 注：任务预期锚点「~9.1ms×subcloud 数」来自阶段 A torch 的 **L1 SA 整段 forward**（FPS+BQ+MLP=9.11ms），本 nsys 只统计 FPS kernel 本身，实测 2.74ms/subcloud，快于该粗略估计

### 6.2 端到端 median（ti10，49 subcloud runs，纯 engine H2D+compute+D2H，CUDA event，同会话空闲 GPU）

| engine | fp32 per-subcloud median | vs fps | vs fps_cache |
|---|---|---|---|
| fps fp32 | 7.413ms | — | — |
| fps_cache fp32 | 5.751ms | -22.4% | — |
| **fps_cache_prune fp32** | **4.602ms** | **-37.9%** | **-20.0%** |
| fps_cache_prune fp16 | 4.531ms | — | — |

每文件 pipe median：fps 46.82ms → fps_cache 38.21ms → fps_cache_prune 31.26ms（fp32）。

### 6.3 全量 per-subcloud（engine_ms/n_subcloud 每文件比值口径，fp32）

| engine | mean | median | p99 |
|---|---|---|---|
| fps（fps-cache-fullset 证据） | 10.107 | 10.475 | 14.312 |
| fps_cache（fps-cache-fullset 证据） | 7.575 | 7.822 | 10.636 |
| **fps_cache_prune（本次，空闲 GPU）** | **6.125** | **6.365** | **8.369** |

全量每文件 pipe median：fp32 44.73ms / fp16 46.11ms（prune 全量含 27 个大 subcloud 文件，绝对量级高于 ti10）。fp16 数值：全配置 0 个 NaN/Inf logit subcloud。

## 7. 结论

1. **实现**：FPSPrune 插件（独立类，stride+keep_rate 属性）+ fill_unselected_kernel（升序/截断/num_points==0/越界选中 语义与 samplefps 尾部一致）+ 注册（第 9 个）+ CMake + ONNX 接线（fpsprune_op.py / onnx_backend.py fps_cache_prune 档 / onnx_export.py --keep-rate）
2. **单元**：kernel 5 边界用例全过；插件端到端与 torch prune_fill **bit-exact**（k0.75 prune 路径 + k1.0 exact 路径）
3. **图**：628 节点，1×FPSPrune{stride:2,keep_rate:0.75} + 3×PrefixFPS，非 hpenet 611 节点与 fps_cache 逐节点一致
4. **engine**：fp32/fp16 同 profile 构建成功
5. **acc**：ti10 fp32=0.970691 / 全量 fp32=0.955761，**双口径均落在阶段 A 锚点（0.9707 / 0.9558±0.001）内**；逐文件抽样对拍 max dev 0.0004
6. **延迟**：FPS 段 91.45→50.91→**35.64ms**（vs fps -61%、vs fps_cache -30%，launch 52→13）；端到端 ti10 per-subcloud median 7.41→5.75→**4.60ms**（vs fps -37.9%、vs fps_cache -20.0%）
7. **定位**：fps_cache_prune = 「现役 kernel + prune + cache」——比 fps_cache 再省 FPS 段 30%、端到端 20%，代价是 acc 掉 0.12pp（-0.034pp ti10 / -0.12pp 全量，与阶段 A 预测一致）

## 8. 未做（遵守 MUST NOT DO）

- 未改现役 fps_kernel.cu / fps_plugin.cpp（mtime 未动，回退通道完整）；未改 flashfps/samplefps/prefixfps 既有插件
- 未做 git 操作；未改 checkpoint/dataset；未引入 thrust/cub
- nsys 仅用 `-t cuda`（未加 cudnn/cublas）

## 附：测试产物位置

- kernel 单测: `/tmp/opencode/fpsprune_verify/test_fill_unselected.cu`
- 插件端到端验证: `/tmp/opencode/fpsprune_verify/verify_plugin_trt.py`
- 全量/ti10 bench: `/tmp/opencode/fpsprune_fullset/bench_prune.py` + `out/*.csv` + `preds/`
- nsys trace: `/tmp/opencode/fpsprune_fullset/nsys_{fps_cache_prune,fps,fps_cache}.nsys-rep` + `kern_*.csv_gpukernsum.csv`
