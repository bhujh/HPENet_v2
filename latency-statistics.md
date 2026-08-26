# HPENet V2 推理延迟统计与优化方向

> 生成：2026-08-26 | 数据来源：本 session 问答 + `.omo/plans/`、`.omo/notepads/`、`.omo/evidence/` 全部 latency 相关归档 + 任务 18 nsys 实测（`/tmp/opencode/latency_v002/`）
> 部署口径：stride-4 + FPSPrune(keep_rate=0.75) + **voxel_size=0.02**（训练 run `20260825-161134` 标定，cfg.yaml:21），ti10 acc 基线 **0.9578**（fp32，10 文件 mean）
> 引擎：`deploy/hpenet_v2_fp32.engine`（**14,446,844 B**，2026-08-26 从 `20260825-161134` pth 转出），profile min_n=2024/**opt_n=4096**/max_n=10000，in_channels=5

---

## 一、两平台环境对照

| 项 | 本机（开发验证） | Orin AGX（部署目标） |
|---|---|---|
| GPU | L20 (sm_89) | Jetson AGX Orin 64GB (sm_87) |
| CUDA | 11.8 | 11.4 |
| TensorRT | 8.6.1.6 | 8.5.2（JetPack 5.1.6-b5） |
| cuDNN | 8.9.7 | — |
| 单子云 GPU 推理 T1（N≈3523） | **1.977 ms** | **7.49 ms**（≈3.8× L20） |
| host enqueue / 子云 | **~2 ms**（2091µs） | **~8 ms**（8007µs，第一瓶颈） |

---

## 二、本机（L20 / x86）延迟现状

### 2.1 端到端（两口径，voxel_size=0.02 现状）

| 口径 | 端到端 | 说明 |
|---|---|---|
| **部署口径**（点云内存直通，无 PLY IO） | **≈6.4 ms/帧** | GPU kernel 4.79ms（**75%**）+ host 纯串行 ~1.6ms |
| **benchmark 口径**（含磁盘 ASCII PLY 读取） | **16 ms/文件**（per-file 15~17ms） | PLY_LOAD 9.64ms 是 IO 假象，真实部署不存在 |

> 来源：任务 18 nsys 实测（`/tmp/opencode/latency_v002/cpp_v002.nsys-rep`，ti10，voxel 0.02，新引擎 20260825-161134）
> 对比历史（voxel 0.3 口径，旧引擎 20260824）：端到端 23.5ms → **16ms（−32%）**；部署口径 14ms → **6.4ms（−55%）**；GPU kernel 12.05ms → **4.79ms（−60%）**；子云数 5→2（总点次 20900→~8800）

### 2.2 benchmark 口径分段（NVTX，voxel_size=0.02 现状，任务 18 实测）

| 阶段 | ms/文件 | 占比 | 备注 |
|---|---|---|---|
| PLY_LOAD | 9.64 | 60% | 纯 host 文本解析，**GPU 完全空闲**（假象） |
| SUBCLOUD_LOOP | 3.54 | 22% | 2 子云 ×（ENQUEUE 1.64 + PREPROCESS 0.11） |
| TAIL | 2.39 | 15% | scatter + D2H + 末次 sync |
| VOXELIZE | 0.51 | 3% | CPU 体素化 |

### 2.3 GPU kernel 内部构成（4.79 ms/文件，voxel 0.02 现状）

| kernel 段 | 时间/文件 | 占比 | 处置状态 |
|---|---|---|---|
| **FPS warp**（furthest_point_sampling_kernel_warp<1024>） | 1.65 ms | **34.5%** | warp 归约已落地 |
| **ball_query**（4 stage） | 1.38 ms | **28.8%** | 已证到头（GridBallQuery / 融合1 均负） |
| bq_dp（4 stage） | 0.26 ms | 5.3% | — |
| TRT 卷积 / GEMM | ~0.55 ms | 11.4% | — |
| Myelin 融合 + 其他（gather/reduce/sort） | ~0.95 ms | 20% | — |

---

## 三、已使用的优化方法（全量清单）

按「采用状态」分三类：**已落地/已采用**（生产路径在用）、**已验证未落地**（留档备启用）、**已尝试并否决**（负结果，勿重试）。

### 3.1 已落地 / 已采用（生产路径在用）

**模型结构级（精度换延迟，均为有意配置）**

| 方法 | 机理 | 实测收益 | 来源 |
|---|---|---|---|
| stride-4（strides `[1,4,4,4,4]`） | 提高下采样倍率、降低 FPS/ball_query 采样密度 | 与 stride-2 相比延迟大幅下降，acc 由 0.9707→0.9636（旧引擎 20260824 口径） | 用户决策（ballquery learnings） |
| FPSPrune（keep_rate=0.75） | 剪枝 FPS 采样点数（`num_points=N/stride×0.75`） | 配合 stride-4 降延迟；acc 0.9636 为预期精度（旧引擎 20260824 口径） | 同上 |
| FPS 算法档演进（精确 fps → fps_cache → fps_cache_prune） | fps_cache 缓存前缀 FPS 结果；prune 档再加剪枝 | fps_cache 档延迟 **−24~25%**（全量 339 acc 0.9569，**stride-2 口径**）；部署用 stride-4 fps_cache_prune | fps-samplefps learnings |
| **voxel_size=0.02**（训练标定，20260825-161134 run） | 缩小体素使子云数 5→2、总点次 20900→~8800 | 端到端 23.5→**16ms（−32%）**，GPU kernel 12.05→**4.79ms（−60%）**，部署口径 14→**6.4ms（−55%）** | 任务 18 实测 + 训练 cfg.yaml:21 |

**kernel 级（bit 级等价 / 正确性无损）**

| 方法 | 端到端 | GPU kernel | 关键数据 | 状态 |
|---|---|---|---|---|
| **项目一：CPP 批量化**（方案 A，GPU 累积，`trim_transpose_kernel`） | **−2.0%**（25.0→24.5ms） | 持平（+0.018%） | sync 98→39 次（−60%）、总时 −54%；pred 10 文件 bit 级 0 mismatch | ✅ 落地 |
| **项目二：FPS warp 归约**（`__shfl_down_sync` 替代树归约） | 端到端增量 0（host 吸收） | **−7.32%**（129.99→120.48ms） | FPS 段 −20.0%（48.14→38.51ms）；15/15 用例 idx bit 级 0 mismatch | ✅ 用户决策落地为默认（宏 1） |
| d_output_ 预分配（删按需 realloc） | — | — | 消除去 sync 后 cudaFree 竞态（正确性防御） | ✅ 落地 |

**工程级**

| 方法 | 内容 | 状态 |
|---|---|---|
| GPU 显存监控 | `cuda_utils.h` 加 `get_gpu_memory_info()` + main/pipeline 关键点打印水位 | ✅ 落地（gpu-memory-monitor 计划） |

> 数字链（nsys，10 文件累计，voxel 0.3 旧口径）：S1 25.0ms/129.99ms → S2 24.5ms/130.01ms → S3② 24.5ms/120.48ms（GPU）
> 落地后 per-file median = **23ms**，ti10 acc 0.9636 全程零回退（**旧引擎 20260824 口径**）。

### 3.2 已验证未落地（留档备启用）

| 方法 | 机理 | 实测锚点 | 状态 |
|---|---|---|---|
| CUDA Graph 捕获 | 固化全部 kernel launch 为单次 `cudaGraphLaunch` | L20 enqueue 2091µs→2.2µs（~970×）；Orin 8007µs→42.4µs（188.8×）；replay bit 级一致 | spike③ PASS，产品未接 |
| 多 context 多流并发 | 每 context 独占 profile + 多 stream 真并发 | L20 双 ctx 1.86×（N=4608 pad）/ 1.53×（kit 复测 N=3523）、4 路 3.10×；Orin 双 ctx 1.20× | spike B PASS，产品未接（需重建多 profile engine） |

> 注：原「voxel_size 调小（0.3→0.05/0.08）」方向已落地（voxel **0.02** 成为训练/部署标定并实测），移入 §3.1。

### 3.3 已尝试并否决（负结果，勿重试）

| 方法 | 结果 | 结论 |
|---|---|---|
| **GridBallQuery** | 搜索段慢 8.9~13.5×，整网 GPU-only 慢 2.4~3.5× | 否决（gridballquery learnings） |
| **融合1 ball_query+dp** | 净收益 **−0.49%**（dp strided 写 1.10×T_dp） | 否决（ballquery verdict） |
| **FPS 多 block**（cooperative groups） | grid.sync 每轮 1.94µs > 每轮计算 1.81µs | 结构性必亏，CANCELLED |
| **桶化 pad-4608 静态 shape** | pred 一致率仅 98.5%（算法性差异：FPSPrune M/num_points 随 N 缩放） | 否决（spike①） |
| **SampleFPS**（精确桶化 FPS） | 慢 **~4×**（70.74 vs 17.93ms） | 已证不适配 |
| **FlashFPS**（近似 FPS） | FPS 段 −72%，但端到端 43.7ms **反慢于同负载现役 fps 28.8ms**，未达 26.53ms 目标 | 实验档，未达标，未落地 |

---

## 四、Orin AGX 延迟现状

> 来源：`.omo/plans/latency-graphpool-multistream.md` §五（Orin 端验收，2026-08-24 方案 B 代跑）

### 4.1 三冒烟实测

| 冒烟 | 结果 | 关键数字 |
|---|---|---|
| 0a graph capture | **PASS**（bit 级一致，三组 memcmp 全等） | enqueue 8007µs → graphLaunch 42.4µs（**188.8×**） |
| 0b 双 context 并发 | **1.20×**（<1.3 机械阈值） | T1=7.49ms / serial=15.51ms / concurrent=12.93ms |
| 0c workspace/ctx | **PASS** | +8.6MB RSS / +7.4MB 显存（远低于 700MB 预算） |

### 4.2 端到端画像

- **现役 ≈48 ms/帧**，且 **host 主导**（host 路径 48ms > GPU 45ms）——与 L20「GPU 主导」相反。
- 根因：Orin 单子云 host enqueue ~8ms（L20 仅 ~2ms），6 子云/文件 → host enqueue 成为第一瓶颈。
- GPU ~45ms 为 N=3523 单 shape smoke 外推下界（全集 N 至 7467，真实 GPU 路径可能更长）。

---

## 五、延迟瓶颈梳理（排序）

### 本机（L20，voxel 0.02 现状）
1. **（benchmark 口径）PLY_LOAD 9.64ms（60%）** — 纯 ASCII 文本解析，真实部署无此，属测量假象。
2. **（部署口径）GPU kernel 4.79ms/帧（75%）** — FPS warp（1.65ms）+ ball_query（1.38ms）居首，两者仍有优化空间。
3. host ENQUEUE ~1.64ms/子云 ×2 — TRT enqueueV3 host 开销，CUDA Graph 可打。

### Orin AGX
1. **host enqueue ~8ms/子云** — 第一瓶颈（host 48ms > GPU 45ms）。
2. GPU kernel ~45ms/帧 — 单子云 7.49ms × 6，SM 利用率低（FPS warp 单 block 只占 1/16 SM）。

### 共同瓶颈本质
- **GPU 侧**：FPS（贪心串行 M 轮 + 归约/同步开销）+ ball_query（内存带宽型，已到头）+ SM 空闲（单流单 block）。
- **host 侧**：TRT `enqueueV3` 提交开销（每子云一次），Orin 上尤甚（~8ms vs L20 ~1.6ms）。

---

## 六、可能的优化方向

> 排序按「预期收益 × 可行性」。标注的实测锚点来自 L20/Orin 冒烟。

### 高优先级（有实测锚点支撑）

| 方向 | 机理 | 实测锚点 | 预期收益 |
|---|---|---|---|
| **① CUDA Graph 捕获**（host enqueue） | 固化全部 kernel launch 为单次 `cudaGraphLaunch` | L20 2091µs→2.2µs（~970×）；Orin 8007µs→42.4µs（188.8×） | Orin ~−6%（48→45ms，纯 enqueue 外推上界，预处理等 CPU 未计入）；L20 流水化后受益 |
| **② 多 context 多流并发**（打 SM 空闲） | 每 context 独占 optimization profile + 多 stream 真并发 | L20 双 ctx 1.86×、4 路 3.10×；Orin 双 ctx 1.20× | Orin 叠加 graph 后 ~−22%（→37.5ms）；L20 上界 ~6-7ms |
| **③ 逐 shape GraphPool**（配合 ①②） | 对每个真实 N（全集 2803–7467、328 唯一值、mean 5286）各捕一张图 | graph 输出 bit 级一致（spike③） | 解 graph「shape 烧死」约束；per-ctx LRU 384 全集不淘汰 |

> 注：原方向 ④「voxel_size 调小」已落地（voxel 0.02 成为训练/部署标定），见 §2.1/§3.1，不再列为待办方向。

### 中优先级

| 方向 | 机理 | 预期收益 |
|---|---|---|
| **⑤ 跨文件流水线化**（PLY 预取，benchmark 口径） | 生产者线程 load+voxelize 与上一文件 GPU 推理重叠 | 16→~6.4ms（−60%，PLY_LOAD 9.64ms 全消）；真实部署已无 PLY，仅 benchmark 受益 |
| **⑥ VOXELIZE GPU 化** | CPU 排序 → GPU radix/hash | ~0.8ms/帧 |
| **⑦ TAIL 融合** | scatter 并入 trim_transpose / argmax 移 GPU | ~0.5-1ms/帧 |
| **⑧ 融合1.5（dp 布局交错化 float3）** | 消除 dp strided 写 | ~4% GPU kernel（暂缓：需改下游 HPE 消费方式，高风险） |

### 关键实现约束（立项前须知）

- **TRT 8.5/8.6 限制**：同一 optimization profile 同时只能被一个 context 占用；N 路并发需建 engine 时加 N 个相同 profile（`trt_build.py` 需加 `--num_profiles`）。多 profile engine 为独立新文件，现役 engine 不动。
- **多 context 并发前提**：需重建 4-profile engine（`hpenet_v2_fp32_mp.engine`）。
- **voxel_size 调小**已连带暴露并修复一个测试崩溃 bug（`main.py:651` nearest_neighbor 误判，见 `.omo/plans/scatter-mean-always.md`，已注释 L651 落地）。
- **CPP `main.cpp` 硬编码默认 `voxel_size=0.2f`**（与训练标定 0.02 差 10×）——CPP 推理须显式传 `--voxel_size=0.02`，否则子云结构偏移（延迟 16→19ms、精度偏移）。建议修正 main.cpp 默认值。
- **精度红线**：任何优化须保持 acc **0.9578**（新引擎 ti10 基线，逐文件 4 位小数逐字符）或 pred bit 级一致；logits 不做 bit 级判据（scatter_mean 浮点原子加顺序本征非确定）。

---

## 七、Orin 落地路线建议（综合评估）

按「GraphPool + 多流」计划（`.omo/plans/latency-graphpool-multistream.md`，状态：定稿待执行）：

1. **阶段 2 先行**：逐 shape GraphPool（单流）→ 验证 → 预期 Orin ~−6%。
2. **阶段 3 叠加**：多 context 多流（K=2/4）→ 组合预期 Orin **48→37.5ms（−22%）**。
3. 阶段 4 回填测量 + Orin 端到端验收（三冒烟已在 L20/Orin 逻辑层 PASS，Orin 端需复测端到端）。

**未决前提**：Orin 0b 双 ctx 实测 1.20×（低于 1.3 机械阈值）——已改判为「graph 落地后的第二阶段」，叠加后若 Orin 帧时无改善则 `num_streams` 保持 1 回退。

---

## 八、数据来源索引

| 来源文件 | 内容 |
|---|---|
| `.omo/plans/latency-cppbatch-fpswarp.md` | 项目一/二 完整计划、S1/S2/S3 实测表、host profiling、FPS warp 机理 |
| `.omo/notepads/latency-cppbatch-fpswarp/learnings.md` | 逐任务实测记录、NVTX 分段、spike 结果、voxel_scan |
| `.omo/plans/latency-graphpool-multistream.md` | Orin 三冒烟、GraphPool+多流计划、收益修正 |
| `.omo/notepads/ballquery-dp-fusion/{verdict,learnings}.md` | 融合1 否决、FPS 微基准分解、CPP 批量化调研 |
| `.omo/notepads/gridballquery-trt-plugin/learnings.md` | GridBallQuery nsys 负结论 |
| `.omo/notepads/fps-samplefps-flashfps/learnings.md` | FPS 四算法端到端对比、fps_cache 档、已关闭路线 |
| `.omo/plans/gpu-memory-monitor.md` | GPU 显存监控落地（get_gpu_memory_info） |
| `.omo/plans/scatter-mean-always.md` | voxel_size=0.02 崩溃根因与 L651 修复 |
| `.omo/boulder.json` | 6 个 work 状态（5 completed / 1 paused） |
| `/tmp/opencode/latency_v002/cpp_v002.nsys-rep` | 任务 18 nsys 实测（voxel 0.02，新引擎 20260825-161134，e2e 16ms / GPU 4.79ms / 部署 6.4ms） |
