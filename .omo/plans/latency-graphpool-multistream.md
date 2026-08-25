# Graph 池（逐 shape 优先）+ 多 Context 多流 延迟优化计划（Orin 冒烟前置）

> 创建：2026-08-24 | R1 双评审修订（主路径改逐 shape 池、5a/5b 拆分、ctx 切片、决策树统一、0c 默认值、基线钉死）；R2 双评审修订（per-ctx LRU 防 thrash、miss 按 ctx 计、median 双报、0c 含池 ≤700MB）；R3 双评审修订（R2 数字同步任务/TODO 层）；**Orin 冒烟实测回填（2026-08-24 方案 B 代跑：0a PASS 188.8×/0b FAIL 1.20×/0c PASS +8.6MB → 0b 实测 1.20× 改判第二阶段；收益修正 graph ~-6%/组合 ~-22%（原 ~2× 证伪）——host enqueue 8ms/子云是 Orin 第一瓶颈）** | 状态：定稿待执行
> 前置计划：latency-cppbatch-fpswarp（已落地：CPP 批量化 + FPS warp，部署口径 ~14ms/帧，GPU kernel 12.05ms 占 86%）
> 部署环境：**Jetson Orin / CUDA 11.4 / TensorRT 8.5**（开发验证机：L20 / CUDA 11.8 / TRT 8.6.1）
> 部署口径：stride-4 + FPSPrune(keep_rate=0.75)，真实部署无 PLY 读取（点云内存直通）

## 一、项目动机

### 现状与实证链（全部来自前置计划实测）

1. 部署口径端到端 ~14ms/帧，GPU kernel 12.05ms（86%）、host 纯串行 ~2ms——GPU 是主战场。
2. GPU kernel SM 占用极低（FPS warp 单 block 1/60 SM）：Spike B 实证（L20/TRT8.6）双 context 并发 1.86×、4 路 3.10×、kernel 重叠 ~60%。
3. Host enqueue ~2ms/子云：Spike ③ 实证 CUDA Graph replay bit 级一致、2091µs→2.2µs（~970×），全部自定义插件 graph-safe。
4. 全集 N 分布：2803–7467、**328 唯一值（同文件内 6 子云同 shape）**、mean 5286。
5. **R1 关键修正（桶化证伪）**：Spike① pad-4608 pred 一致率 98.5% 且两模式各自 bit 确定 → 差异是**算法性**的，源码机理 = `fpsprune_plugin.cpp` 的 `M=N/stride`、`num_points=N_points/sample_rate` 随 N 缩放 → 桶化改变 FPSPrune 采样点集合 → 邻域聚合漂移。**桶化不可能 pred==100%，故弃桶化，主路径 = 逐 shape 池**（无 pad、输入零改动，spike③ 已证 graph 输出 bit 级一致）。

### 目标

部署口径端到端 **14ms → 8~10ms/帧**（原预估；**Orin 冒烟实测后修正为：graph 单独 ~-6%（48→45ms）、graph+多流组合 ~-22%（→37.5ms），见 §五**）（Orin 保守；L20 理论上限 ~6-7ms），acc 验收 = 逐 shape 池 graph 与现役**逐文件 bit 级一致**（logits 本征非确定除外，pred/acc 必须完全一致）。

## 二、优化机理

### 机理 1：多 context 多流——打 SM 空闲
单流 6 子云 kernel 串行，GPU 大部分 SM 空闲。多 context（每 context 独占一个 optimization profile——TRT 限制，Spike B 实证）+ 多 stream 真并发（L20 4 路 3.10×）。

### 机理 2：CUDA Graph——打 host enqueue
Graph 固化全部 kernel launch 为单次 `cudaGraphLaunch`（2.2µs vs 2091µs）。多流场景 host 需同时喂 K 条 stream，graph 是配套前提。

### 机理 3：逐 shape 池——解 graph 的"shape 烧死"约束
Graph 固化捕获时的 shape 与地址。**逐 shape 池**：对每个出现过的真实 N 各捕一张图，池 `{(N, ctx): graphExec}`。全集 328 个唯一 N（同文件 6 子云同 N）；**注意 miss 代价按 ctx 计：每文件 1 个新 N × K 个 ctx = 每文件 K 次新图捕获（~4–5ms/次）**——同文件内 6 子云在同 ctx 串行复用同一张图（拷入→launch→读出，stream-FIFO 保序）。**池为 per-ctx 独立 LRU，每 ctx 容量 = 384（>328 唯一 N，全集不淘汰；总量 ≤328×K 图，估算 65–260MB，并入 0c 显存口径监控）**。全局统一池 + 小容量 LRU 已在 R2 评审中被证伪（K=4 循环扫描 1312>512 → thrash → 每帧 miss 开销 +18ms 反超 host 节省，负收益）。**输入零改动 → 输出与现役 bit 级一致（spike③ 实证），无精度争议**。

### 复制 pad 点无害性（不再作为主路径依据；保留为 bucket 方案的记录）
原论证（FPS tie 保低、ball_query 凑满 break）在理论层成立，但 **spike① 实测证伪**：`fpsprune_plugin.cpp` 的 M/num_points 随 N 缩放导致采样集合变化，属于算法性差异，与 tactic 无关。**结论：pad 到非真实 N 的任何方案（含桶化）都不可保证 pred==100%，弃用**；如未来确需桶化（显存/图数约束），按"逐 shape 池不可行再试桶化 + 重新定义验收（接受 ~1.5% pred 差异）"。

### Graph 与 context 的绑定关系（实现约束，R1 钉死）
- **共享 buffer 按 ctx 切段**：维度 = `[ctx_id][max_n]`，`(N, ctx)` 图的输入/输出地址 = **ctx 段内固定偏移**（同 ctx 不同 N 的图共享该 ctx 的输入/输出段——**同 ctx 内不并发**，安全；不同 ctx 各用各段，并发安全）。**切片 key = ctx，不是 N，不是 (N,ctx)**。
- **输出 buffer 单一所有权**：GraphPool 持有全部输入/输出共享 buffer；`trt_inference` 返回 ctx 段输出指针（不再有独立的 d_output_ 双头管理）。
- 同一 graphExec **禁止并发 launch**（地址烧死）→ 多流下每 ctx 各 instantiate 一份（同一捕获结果 per-ctx 实例化）。
- capture 用**非默认 stream**（L20 实证 legacy default stream 不能 capture）。
- miss 路径 = 正常 enqueue 一遍（拿结果）+ capture 一遍（录制，GPU 不执行）+ instantiate，每 shape 一次性 ~4–5ms。

## 三、具体计划

### 阶段 0：Orin 三冒烟前置（R1 统一决策树：见下）

**冒烟决策树（R1 统一，替代原三处矛盾表述）**：
- 0a FAIL（graph capture）→ **限时修复尝试（1 次）**；仍 FAIL 则 **graph 路线在 Orin 停用**，多流继续但不带 graph（host enqueue 成瓶颈，预期收益降为 ~0–5%，由 0b/端到端实测决定是否保留多流）；**不再"全计划中止"**。
- 0b FAIL（并发加速比 <1.3×）→ **改判（见 §五 Orin 验收修正版）**：多流为 graph 落地后第二阶段（graph 后 host→1-3ms，其 -16.6% GPU 节省全额兑现 45→37.5ms）；TODO 8/9 在 GraphPool 验证通过后执行。
- 0c FAIL（超预算）→ 减 context 4→2（engine 4 profile 不变，只用前 2 个）。

- 0a. **graph capture 冒烟**（Orin/TRT8.5）：`/tmp/opencode/spike3_cudagraph.py` 移植，包住完整推理（全部自定义插件）→ 判据：捕获无 violation + replay bit 级一致。
- 0b. **双 context 并发冒烟**（Orin）：`dual_ctx_bench.py` + 多 profile engine → 判据：2 路并发加速比 ≥1.3×。
- 0c. **workspace + graph 池实测**（Orin）：每 context 显存增量 + 预估 graph 池（≤328×K×~0.2MB）→ **默认判据：4 context workspace + graph 池总增量 ≤700MB**（L20 锚点 82MB/ctx + 池 65–260MB 预估；超限降 2 context 或减池容量）；用户可覆盖，但默认值已写入不阻塞执行。
- 依赖：Orin 机器访问权（用户提供或代跑）；L20 侧并行推进阶段 1-2。

### 阶段 1：多 profile engine 重建（build 配置，不动 ONNX/模型/插件）
- 1. `trt_build.py` 加 `--num_profiles`（默认 1 零变化；本次 4），每 profile 同参数（min 2024/opt 5500/max 10000），产出 `deploy/hpenet_v2_fp32_mp.engine`（新文件，现役不动）；fp16 可选。

### 阶段 2：逐 shape GraphPool 实现（CPP_trt3，C++，运行时开关）
- 2. **5a pipeline.cpp**：桶 pad 不再需要（无桶化）——但需把"当前推理 N"与 min_n pad 逻辑与 GraphPool 对接（`padded.N_padded` = 真实 N 或 min_n 兜底，传入 inference）；上传尺寸 = N_padded（现役语义不变）。
- 3. **5b trt_inference.cpp/.h**：GraphPool——①构造期分配共享 buffer `[ctx_id][max_n]`（输入 pos/x + 输出，各 ctx 独立段）；②推理接口 `use_graph` 开关（默认关=现役路径零变化）：N→查池 `{(N, ctx)}`，命中 → 拷数据进该 ctx 输入段 → `cudaGraphLaunch`；miss → 现役 enqueue（拿结果）+ capture + per-ctx instantiate 入池；③**per-ctx LRU 容量 384**（>328 唯一 N 全集不淘汰；总量 ≤328×K 图并入 0c 监控）；④capture 用非默认 stream；⑤输出 buffer 单一所有权（GraphPool 持有，infer 返回 ctx 段指针，删除旧 d_output_ 独立管理）。
- 4. **trim_transpose 零改动确认**（写注释）：N_true=chunk_N（host 侧）、N_padded=图输出 shape（=真实 N 或 min_n）仅作布局 stride，现状语义已兼容，**禁止 worker 自行"修复"**。
- 5. **逐 shape 精度验证（L20）**：对照基线 = **同一 `hpenet_v2_fp32_mp.engine` + use_graph 关 + num_streams=1（即 G-S1 配置）**——判据：graph 开 vs 关，**pred 逐点 100% 一致 + acc 逐文件 4 位小数逐字符相同**（logits 允许 scatter ULP）。**ti10 逐点 + 全量 339 文件 acc 逐字符**（写死，非"抽测"）。

### 阶段 3：多 context 多流并发（CPP_trt3）
- 6. `pipeline.cpp` 多流版：K（=0b/0c 决定，2 或 4）× K stream，6 子云 round-robin；trim_transpose 聚积：每流写该 ctx 的 d_src 私有段，末尾统一 scatter 前用 event 等全部流完成；`num_streams` 开关默认 1（单流零变化）；d_output_ 并入 GraphPool 输出段（无独立分配）。
- 7. 组合验证（L20）：多流+graph 全开 vs 现役——pred/acc 判据同上；**miss 率 = 全量 339 文件单遍的 miss 计数**（预期 = 328×K 全量首遍捕获、**第二遍起应严格为 0**——若不为 0 说明池容量/thrash，按 §六.3 处置）；**median 不剔除 miss 帧，另报剔除后口径**（双报防美化——R2 评审修正）。

### 阶段 4：测量与收尾
- 8. §四测量全轮次 + §五回填（含冷启动项）。
- 9. Orin 端验收（阶段 0 未先行的情形）：三冒烟 + 端到端复测。

## TODOs

- [ ] 1. Orin 冒烟 0a：graph capture（TRT8.5 插件 graph-safe）——PASS 判据见 §三阶段 0 决策树
- [ ] 2. Orin 冒烟 0b：双 context 并发加速比 ≥1.3×
- [ ] 3. Orin 冒烟 0c：workspace+graph 池实测（默认判据 4ctx+池 ≤700MB，用户可覆盖）
- [ ] 4. 多 profile engine 重建（--num_profiles 4 → hpenet_v2_fp32_mp.engine 新文件，现役不动）
- [ ] 5. pipeline.cpp 对接（5a：N 传递/min_n pad 与 GraphPool 接口；trim_transpose 零改动注释）
- [ ] 6. GraphPool 实现（5b：共享 buffer [ctx][max_n] 按 ctx 切片 + 逐 shape 池 + use_graph 默认关 + **per-ctx LRU 384** + 非默认 stream 捕获 + 输出单一所有权）
- [ ] 7. 逐 shape 精度验证（L20，对照 G-S1 配置：ti10 逐点 + 全量 339 acc 逐字符，pred 100%）
- [ ] 8. 多 context 多流实现（K×K round-robin + 跨 stream event 聚积 + num_streams 默认 1）
- [ ] 9. 组合验证（全开 vs 现役：pred/acc + miss 率统计口径写死 + **median 双报：主口径不剔除 miss 帧，另报剔除后口径（防美化）**）
- [ ] 10. 测量全轮次（G-S1/S2/S3/combo nsys 留档）+ §五回填
- [ ] 11. Orin 端验收（若 1-3 未先行：三冒烟 + 端到端复测）

## 四、nsys 延迟收益测量方式

### 统一口径（含前置教训内置）
- **部署口径为主**：bench 模式 `--loop N`（内存直通重复推理，跳过文件读取）或 NVTX 分离 PLY_LOAD 后仅统计 LOOP+TAIL+VOXELIZE；两口径都记录，归因只用部署口径。
- engine：`hpenet_v2_fp32_mp.engine`（**G-S1 基线 = 同一 mp engine + use_graph 关 + num_streams=1**——消除 engine 差异，R1 钉死）。
- 数据 ti10 + 全量 339；GPU 独占、测量串行；nsys-rep 按 `G-S1/G-S2/G-S3/G-combo` 留档；warmup ≥10 + ≥30 帧取 median；**冷启动单独记录**（首 X 帧 max 延迟 + 填满池所需帧数）。
- 判据口径：pred bit 级 + acc 逐文件 4 位小数逐字符（logits 本征非确定已知，不比）。

### 测量时序（增量归因链）
| 轮 | 状态 | 归因 |
|---|---|---|
| G-S1 基线 | mp engine + 现役（单流/动态/无 graph） | 基线 |
| G-S2 | +graph 池（逐 shape，单流） | graph host 收益（单流下小）+ 池正确性 |
| G-S3 | +多流（2/4 路，无 graph） | 并发 GPU 段归因（host 主导下端到端预期 ~0%，仅验证 kernel 并发） |
| G-combo | graph 池 + 多流全开 | 终态（预期 48→37.5ms，-22%；见 §五 Orin 修正） |

## 五、延迟收益测量结果统计（执行后回填）

| 指标 | G-S1 | G-S2 | G-S3 | G-combo |
|---|---|---|---|---|
| 端到端 per-frame median（部署口径，ms） | ___ | ___ | ___（2路/4路） | ___ |
| 端到端 p99 / 冷启动 max | ___ | ___ | ___ | ___ |
| GPU kernel 总时间（ms/帧） | ___ | ___ | ___ | ___ |
| host enqueue 总时间（ms/帧） | ___ | ___ | ___ | ___ |
| graph 池命中率（全量单遍 miss 计数） | — | ___ | — | ___ |
| pred 一致率 / acc 逐文件 | 基线 | ___ | ___ | ___ |

### Orin 端验收（2026-08-24 实测回填，AGX Orin 64GB/JetPack 5.1.6-b5/TRT8.5.2，方案 B 代跑）

| 冒烟 | 结果 | 关键数字 |
|---|---|---|
| graph capture | **PASS**（bit 级一致，三组 memcmp 全等） | enqueue 8007µs → graphLaunch 42.4µs（**188.8×**） |
| 双 context | 1.20×（<1.3 机械阈值，**2026-08-24 改判为"graph 后第二阶段"**——host 瓶颈解除后 -16.6% GPU 全额兑现） | T1=7.49ms / serial=15.51ms / concurrent=12.93ms（串行/T1=2.07 正常） |
| workspace/ctx | **PASS（+8.6MB RSS / +7.4MB 显存）** | 远低于 700MB |
| build 链 | PASS（插件 sm_87 编译 + 18.2MB 多 profile engine） | RTC 未同步无碍 |

**§三决策树裁定（0b FAIL 分支，2026-08-24 修正版）**：~~多流停用~~ → **多流改为"graph 落地后的第二阶段"**——重新核算：1.20× 并发在 host 主导时（现役 host 路径 48ms > GPU 45ms）确无端到端收益，但 graph 落地后（host→1-3ms）其 GPU 节省（-16.6%）**全额兑现**：组合帧时 48→45（graph 单独，-6%）→**37.5ms（叠加多流，-22%）**。原 ≥1.3× 阈值制定于"Orin 预期 1.5-2.5×"的预估，对 1.20× 判停用过于机械。执行顺序调整：**阶段 2（GraphPool）先行 → 验证 → 阶段 3（多流）作为第二阶段叠加验证**（Orin 实测 1.20× 为锚，L20 1.86× 为上限参考；若第二阶段 Orin 实测帧时无改善则 num_streams 保持 1 回退）。Orin 端完整收益修正：**graph 单独 ~6%（上界取决于完整 host 路径实测——纯 enqueue 外推，预处理等 CPU 部分未计入，可能更高），graph+多流组合 ~-22%**（修正前误估 ~2×——基于"host+GPU 串行"错误假设，实际批量化流水已存在，帧时=max 两条路径）。

## 六、收益、代价与回退条款

### 收益预期（诚实区间）
- **Orin 实测修正（2026-08-24）**：graph 单独 ~-6%（host 48ms 外推，上界待完整管线实测）、graph+多流组合 ~-22%（48→37.5ms）；原 1.5~2.5×/8-10ms 预估已证伪（基于 host+GPU 串行错误假设——批量化流水已存在，帧时=max 两路径）。注意：GPU 45ms 为 N=3523 单 shape smoke 外推下界，全集 N 至 7467、FPS 段随 N 增长，真实 GPU 路径可能更长（相应 -6%/-22% 比率会变）。
- **若 graph 在 Orin FAIL（0a）**：多流不带 graph，host enqueue（2ms×6/帧）成瓶颈 → 收益降 ~0–5%，由实测决定保留与否。

### 代价
- 显存：每 context workspace ~50–100MB（Orin 实测；L20 锚点 82MB）× K + **graph 池 ≤328×K 图 ≈ 65–260MB（per-ctx LRU 384，并入 0c 监控）** + 共享 buffer ~2MB；0c 判据扩展为 "4ctx workspace + graph 池总增量 ≤700MB"（默认，用户可覆盖）。
- 实现复杂度：池管理（LRU）/ 跨 stream 聚积 / per-ctx 实例化 / 输出单一所有权重构。

### 回退条款（逐级独立，零成本回退）
1. **graph 池**：`use_graph` 默认关；miss 路径 = 现役 enqueue（不差于现状）——全 FAIL 无回退损失。
2. **多流**：`num_streams=1` 即单流；净负（第二阶段叠加后 Orin 实测帧时无改善/劣化）则 num_streams 保持 1。
3. **逐 shape 池**（主路径）若显存/图数不可行（LRU 仍溢出）→ 桶化需**重新定义验收**（接受 ~1.5% pred 差异 + 精度重验），不作为无精度代价方案；或放弃 graph 仅多流。
4. **engine**：mp engine 独立新文件，现役 engine 全程不动。
5. 任何阶段 FAIL：证据记入 §五与 ledger，负结果即交付。

## 七、风险权衡

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| **TRT 8.5 插件 graph-safe 未知**（8.6 已证） | 中 | 高 | 0a 冒烟前置 + 决策树（FAIL → 限时修复 → 仅多流降级，**不再全计划中止**） |
| Orin 16 SM 并发打折（<1.3×） | 中 | 高 | 0b 冒烟前置；**实测 1.20×——改判第二阶段（graph 后兑现）；叠加实测无改善则回退单流** |
| 逐 shape 池图数 328×K 显存/miss thrash | 低（per-ctx LRU 384>328 全集不淘汰；若实际图显存超估则升中） | 中 | **per-ctx 池（R2 修正，弃全局小容量 LRU——循环扫描 thrash 负收益）** + 0c 扩展口径 + miss 双报 |
| workspace 超预算 | 低 | 中 | 0c 默认 **≤700MB（4ctx workspace + graph 池总增量）** + 减 context |
| 多流实现复杂度（跨 stream 聚积/event 同步） | 中 | 中 | bit 级 pred 对拍 + 分阶段开关验证 |
| Orin 环境不可得 | — | 流程 | L20 先行开发验证，Orin 冒烟为部署验收门 |
| **桶化精度（已弃主路径）** | 高（必 FAIL） | — | 已从主路径移除；若启用须重新定义验收（见 §六.3） |

## Final Verification Wave

- [ ] F1. Plan compliance：pred/acc 判据全过（逐 shape 池 100%）、Must NOT 零违反（不动 ONNX/模型/插件算法/现役 engine；零 git；不动已关闭路线）、测量时序合规（G-* nsys-rep）、Orin 冒烟结果回填（或注明环境未得，但 §五 Orin 表须如实标 N/A）
- [ ] F2. 代码质量：共享 buffer 按 ctx 切片、同 graphExec 无并发 launch（每 ctx 实例化）、capture 非默认 stream、跨 stream 聚积 event 正确、输出单一所有权（无 d_output_ 双头）、trim_transpose 零改动、开关默认值 = 现役行为零变化
- [ ] F3. 性能结论审查：§五完整回填（含冷启动/p99）、nsys 可复现、归因链正确、收益为正才默认开启（净负按 §六回退）
- [ ] F4. **计划 vs 代码一致性**（独立 reviewer 用 git diff 逐项对照本文档 §三改动清单——文件、改动点、边界；抓遗漏/不符/scope creep，输出 CONSISTENT/INCONSISTENT）
