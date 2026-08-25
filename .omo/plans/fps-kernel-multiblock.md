# fps-kernel-multiblock - Work Plan

> **状态：CANCELLED（2026-08-19）** —— 高精度评审（Oracle+Momus）后经数据核算，多 block 并行化在 N≈7200/M=3600 画像下必亏（3600 轮 × 2 次 grid.sync ≈ 7ms > 要优化的 6.5ms），用户决定放弃，仅保留 warp 归约（2a）为可选后续。未执行任何 todo。

## TL;DR (For humans)

**What you'll get:** 一个新的多 block 并行 FPS kernel（`FPSMB` 插件），与现役 FPS 算法语义完全相同（朴素精确 FPS），只是把「单 block 串行 3600 轮」改成「G 个 block 协作并行」，把 fps_cache 里那个仍占 71% GPU 时间的第 1 级采样加速。**现役 FPS kernel 和插件一个字都不动**，作为回退通道保留。

**Why this approach:** 现役 kernel 慢在单 block 串行（每轮全量扫 7200 点 + 10 次 `__syncthreads` 树归约，全堆在 1 个 SM 上）。多 block 把点分片给 G 个 block 并行扫描，只剩跨 block 归约这一处要同步。但归约是 block 局部的、串行轮是贪心本质，所以必须**先微基准测出「扫描 vs 归约 vs grid.sync」占比**，确认多 block 真能回本、并选出最优 G——避免重蹈 SampleFPS「方向对载体错」的覆辙。

**What it will NOT do:** 不改现役 fps_kernel.cu / fps_plugin.cpp 任何一行；不改算法语义（无桶、无剪枝、无近似）；不改模型权重；不做 SampleFPS 桶结构；不追求 <5ms 硬凑（以实测为准）；不做 git commit。

**Effort:** Medium（1 个新 kernel 文件 + 1 个新插件 + workspace 扩展 + 1 个新 fps_algo 档 + 端到端回归）
**Risk:** Medium - 两个硬验收点：(1) tie-free 输入下新 kernel idx 与现役 bit 级一致（acc 零回退）；(2) 多 block 的 grid.sync 开销可能吃掉并行收益（须测量门裁决）
**Decisions to sanity-check:** 新 kernel 命名 FPSMB（可改）；并行度 G 的选取；grid.sync 机制（cooperative groups）；若测量显示归约主导则主攻 warp 归约（新 kernel 内部）而非堆 block 数；tie-break 语义改为「全局最小索引」

Your next move: 审阅后可用 `$start-work` 开始执行。Full execution detail follows below.

---

> TL;DR (machine): Medium effort, Medium risk; 新建 FPSMB 多 block 并行 kernel + 插件 + fps_cache_mb 档，现役 FPS 零改动，测量优先，acc 零回退守门。

## Scope
### Must have
1. **现役 FPS kernel 与插件零改动**（fps_kernel.cu / fps_plugin.cpp / fps_kernel.h 一个字节不动，回退通道）
2. 新建 `deploy/trt_plugins/src/fps_kernel_multiblock.cu` + `include/fps_kernel_multiblock.h`：朴素精确 FPS 的**多 block 并行**实现（算法语义与现役一致）
3. 新建 `FPSMBPlugin`（type="FPSMB"，IPluginV2DynamicExt，独立类不共类）+ `plugin_registry.cpp` 注册 + `CMakeLists.txt` 追加
4. 新 fps_algo 档 `fps_cache_mb`：第 1 级（stage 1）→ FPSMB，第 2-4 级 → PrefixFPS（Cache），原 `fps_cache` 档保留作对比
5. 测量驱动的分阶段实施：先微基准分解（扫描/归约/grid.sync 占比 + G 扫描），再定杠杆
6. tie-free 输入下新 kernel idx 与现役 bit 级一致（acc 零回退是硬门）
7. 端到端回归：acc（ti10 0.9741 / 全量 0.9569 逐文件一致）+ nsys FPS 段复测
8. plugin.md §14.12 更新

### Must NOT have (guardrails, anti-slop, scope boundaries)
- **不改现役 fps_kernel.cu / fps_plugin.cpp / fps_kernel.h**（回退通道，F1 会 diff 验证零改动）
- 不改算法语义：仍是朴素精确 FPS（无桶、无剪枝、无近似）
- 不改模型权重 / dataset / 其它算子（BallQuery/ThreeInterp 等）
- 不改 fps_cache 档的图结构（原档保留；新档 fps_cache_mb 是新增，不是替换）
- 不做 SampleFPS 桶结构（已证此画像不划算）
- 不引入 thrust/cub/参考仓库代码
- 不做 git commit

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after + 单算子对拍（复用 `deploy/tests_fps_algos.py` 的 save_model/build/run 模式）+ 微基准分解脚本
- Evidence: `.omo/evidence/fps-kernel-multiblock/task-<N>.<log|md>`
- 判据：tie-free 输入 idx 逐索引一致（vs 现役 fps_launcher）；每步全局 argmax（numpy float32 参考，复用 tests_fps_algos 的 seq_exact 逻辑）；acc 守门 0.9741（ti10 逐文件）
- 平局口径：雷达平局率实测 0/14341，tie-break 语义从「min(k mod block_size)→min(k)」改为「全局最小索引」，tie-free 下与现役 bit 级一致
- 距离表达式锁死：`((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))+(z2-z1)*(z2-z1)` float32 左结合（与现役 L41 完全一致，保 bit 级）

## Execution strategy
### Dependency matrix
| Todo | Depends on | Blocks |
| --- | --- | --- |
| 1 (微基准分解) | - | 2,3 |
| 2 (新 kernel fps_kernel_multiblock.cu) | 1 | 3,4 |
| 3 (FPSMBPlugin 插件+注册+workspace) | 1 | 4 |
| 4 (fps_cache_mb 档接线+对拍) | 2,3 | 5 |
| 5 (端到端回归) | 4 | 6 |
| 6 (文档) | 5 | F |

### 关键设计（多 block kernel）
- **并行度 G**：gridDim.x = B×G，每 batch 一维 block g 负责连续点分片 [g·chunk, (g+1)·chunk)
- **temp 无竞争**：temp 按点分片，各 block 只读写自己的 slice，无需原子
- **跨 block 归约**：cooperative groups `grid.sync()`（**每轮 2 次**，位置：sync#1 在各 block 写完局部 (val,idx) 到 reduce_scratch 之后、reducer 读之前；sync#2 在 reducer 写全局 old 广播缓冲之后、下轮读之前）+ 全局 reduce_scratch（B×G 的 val+idx）。**B>1 归约由每 batch 组的 g==0 block 执行（`b=blockIdx.x/G, g=blockIdx.x%G`，写 old_buf[b]），不是单一全局 block 0**。备选 1 次 sync 方案：`atomicMax` 于 `__float_as_uint(d2)`（非负 float 位序==数值序）+ `atomicCAS`/uint64 pack 处理 argmax
- **warp 归约**：intra-warp 用 `__shfl_down_sync`（新 kernel 内部，替代现役的共享内存树归约，减少同步次数）
- **tie-break**：全局最小索引；tie-free 下与现役 bit 级一致（雷达平局率实测 0/14341，须在真实 339 文件全量复验，不能只引用历史数字）
- **workspace 扩展**：FPSMB 的 `getWorkspaceSize` = temp（maxB×maxN×float）+ reduce_scratch + old 缓冲
- **launcher**：`cudaLaunchCooperativeKernel`（cooperative launch，无需 -rdc，已验证）。**M<=0/B<=0/N<=0 在 launcher 层 guard 早退**（cooperative kernel 内零早退，防 grid.sync 死锁）；cooperative launch 失败（`cudaErrorCooperativeLaunchTooLarge`）时回退：G 减半重试 → 仍失败则回退现役单 block `fps_launcher_with_stream`

## Todos
- [ ] 1. 微基准：分解现役 FPS 每轮耗时 + grid.sync 延迟 + G 扫描
  What to do：写 /tmp/opencode/fps-kernel-multiblock/bench_decompose.cu，对 N=7200 M=3600（及 N=5500）测：①完整现役 kernel；②去掉树归约只扫描（写 temp 不 reduce）；③空转归约。量化「扫描 vs 归约」占比。同时测 cooperative groups 空 kernel × grid.sync 单次延迟，G∈{4,8,16,32}。据此估算多 block 的净收益（并行收益 vs grid.sync 开销）并定 G。**另测「单 block warp 归约变体」完整耗时**（它是首要候选方案，见 todo 2a）。「是否做多 block」的判据 = `M×(scan_us/G + 2×sync_us) < 现役 6.5ms`，其中 scan_us/sync_us 为本次实测，写入 evidence 作门。
  Must NOT do：不改任何仓库文件。
  Acceptance：产出分解表（扫描/归约/grid.sync 占比 + G 扫描表），明确「多 block 是否回本 + 最优 G」结论写入 evidence。若估算为负收益，仍继续（G 取最小可行值，task-2 产物作为正确性基线），并在 evidence 记录。
  Evidence：`.omo/evidence/fps-kernel-multiblock/task-1.md`

- [ ] 2a. 单 block warp 归约 kernel（首要方案，无 cooperative）
  What to do：新建 `deploy/trt_plugins/src/fps_kernel_warp.cu` + `include/fps_kernel_warp.h`。算法 = 朴素精确 FPS（与现役逐字符一致），实现 = 单 block（gridDim=B，blockDim=opt_n_threads(n)）+ **intra-warp `__shfl_down_sync` 归约 + 跨 32 warps shared-mem 归约（~2 次 `__syncthreads`）**替代现役 10 次 `__syncthreads` 树归约。零 cooperative、零 grid.sync、零 workspace 扩展。**距离表达式逐字符照抄 fps_kernel.cu L41（裸 `*`/`+`，禁 `__fmul_rn/__fadd_rn`——那会与现役可能存在的 FMA 收缩产生 1 ULP 差）**；fill 值锁死 `1e10f`（禁 +inf）。
  Must NOT do：不改现役 fps_kernel.cu；不引入桶/剪枝/近似/多 block。
  Acceptance：①先做 **d2/temp 逐 bit 对拍**（随机 N=1024/7200，现役 vs 新 kernel 逐点 dump temp/d2 raw bits，diff 必须全零——比 idx 对拍更早暴露 FMA 差异）+ nvdisasm 抽查两 kernel 距离指令序列一致；②tie-free idx 与现役 bit 级一致（N 扫 1024/2750/5500/6500/7200）；③每步全局 argmax 100%；④nsys 单 kernel 耗时 vs 现役 6.5ms 下降（记录实测）。
  Evidence：`.omo/evidence/fps-kernel-multiblock/task-2a.md`

- [ ] 2b. 多 block cooperative kernel（仅在 task-1 判正收益后做）
  What to do：新建 `deploy/trt_plugins/src/fps_kernel_multiblock.cu` + `include/fps_kernel_multiblock.h`。算法 = 朴素精确 FPS，实现 = G block 点分片 + cooperative grid.sync（每轮 2 次，位置见设计段）+ 跨 block 归约（每 batch g==0 block 归约）+ warp 归约。接口 `fps_multiblock_launcher(B, N, M, xyz, temp, reduce_scratch, idx, stream)`。G 按 task-1 结论。**若 task-1 判负收益，本任务跳过，evidence 记录裁决（不算失败，测量结论即交付）。**
  Must NOT do：不改现役；不引入桶/剪枝/近似。
  Acceptance：tie-free idx 与现役 bit 级一致 + d2/temp 逐 bit 对拍；每步全局 argmax 100%；M<=0/B<=0/N<=0 launcher 层 guard；cooperative launch 失败回退路径验证。
  Evidence：`.omo/evidence/fps-kernel-multiblock/task-2b.md`

- [ ] 3. FPSMBPlugin 插件 + 注册 + workspace 扩展
  What to do：新建 `deploy/trt_plugins/src/fpsmb_plugin.cpp` + `include/fpsmb_plugin.h`（照 samplefps_plugin.cpp 骨架）：type="FPSMB" version="1"，属性仅 stride，getOutputDimensions kFLOOR_DIV(N,stride)，supportsFormatCombination FLOAT/INT32，workspace = maxB×maxN×float + reduce_scratch + old 缓冲（getWorkspaceSize 扩），enqueue 调 fps_multiblock_launcher（temp=workspace 起始，reduce_scratch/old 按偏移切分）。plugin_registry.cpp 追加注册（第 9 个），CMakeLists.txt 加源文件。**enqueue 内：运行时 cudaOccupancyMaxActiveBlocksPerMultiprocessor 校验 B×G 共驻上限，cooperative launch 失败→G 减半→仍败→回退现役 fps_launcher_with_stream**。重编 libhpenet_plugins.so。
  Must NOT do：不改现役插件；不共类双 creator（独立类）。
  Acceptance：nm 有 FPSMBPluginCreator 符号；tests_fps_algos.py 能 build+run 含 FPSMB 节点的最小图（idx 与现役 bit 级一致）。
  Evidence：`.omo/evidence/fps-kernel-multiblock/task-3.md`

- [ ] 4. fps_cache_mb 档接线 + 单算子对拍
  What to do：onnx_backend.py 加 `'fps_cache_mb'` 档（stage 1 → make_fpsmb_op，stage 2-4 → make_prefixfps_op；新增 fpsmb_op.py symbolic 发 `hpenet::FPSMB`）；onnx_export.py choices 加 fps_cache_mb。导出图验证：1×FPSMB + 3×PrefixFPS，628 节点，非 hpenet 部分与 fps_cache 图一致。tests_fps_algos.py 加 fpsmb 配置（或独立对拍脚本）验证 FPSMB idx 与现役 bit 级一致。
  Must NOT do：不改 fps_cache 档；不删现役档。
  Acceptance：fps_cache_mb 图 628 节点（1×FPSMB+3×PrefixFPS）；FPSMB 单算子 idx 与现役 bit 级一致。
  Evidence：`.omo/evidence/fps-kernel-multiblock/task-4.md`

- [ ] 5. 端到端回归：重导 fps_cache_mb engine + acc + nsys
  What to do：重导 fps_cache_mb 图 engine（fp32+fp16，同 profile min1024/opt4096/max10000），跑 acc + nsys。**acc 命令**：ti10 用 `python deploy/trt_inference.py --engine deploy/fps_algo_fps_cache_mb_{fp32,fp16}.engine ...`（10 文件 0000068-77 产出 per-file acc+mean，须 == 0.9741 逐文件）；全量 339 文件复用 fullset 脚本（`/tmp/opencode/fps_cache_fullset/` 的同款，本次落盘到 evidence 目录并提交复刻脚本），逐文件 acc == 0.9569 对照 fps_cache。「逐文件一致」断言 = per-file acc 对 fps_cache 逐文件 diff==0（或 argmax pred 逐点比对，照 fps-cache-fullset.md 口径）。nsys FPS 段复测（对照 fps_cache 的 per-subcloud 3.92ms、N=7200 单 kernel 6.5ms、现役 6.40ms/subcloud）。**fp16 engine 下 FPS 节点输入仍为 fp32（TRT 插 cast），FPSMB supportsFormatCombination 拒绝 kHALF**。
  Must NOT do：不改图结构；不动其它算子。
  Acceptance：acc 逐文件一致（ti10 0.9741，全量 0.9569 四配置四位小数一致）；**性能条件**——若 task-1 判正收益 → nsys FPS 段 / 端到端 median 较 fps_cache 下降；若判负收益 → 记录实测值 + 保留 FPSMB 作正确性基线即 PASS。tie-free 判定在真实 339 文件全量复验。
  Evidence：`.omo/evidence/fps-kernel-multiblock/task-5.md`

- [ ] 6. plugin.md §14.12 更新
  What to do：更新 §14.12，追加 FPSMB 多 block kernel（设计、G 选取、实测降幅、与现役/多 block 取舍），修订历史 v15.2。
  Must NOT do：不改既有章节历史数据。
  Acceptance：数据来自 task-2/3/5 evidence，无悬空引用。
  Evidence：`.omo/evidence/fps-kernel-multiblock/task-6.md`

## Final verification wave
- [ ] F1. Plan compliance：逐 todo 核对 acceptance、Must NOT 零违反（重点：现役 fps_kernel.cu/fps_plugin.cpp/fps_kernel.h **diff 为零**、算法语义未变、无 git commit）
- [ ] F2. Code quality：CUDA 规范（grid.sync 前不可早退、cooperative launch 正确、shared/scratch 边界、tie-break 确定性、距离表达式锁死）
- [ ] F3. Real manual QA：fps_cache_mb engine 完整推理 + acc 复测 + nsys 抽查一份 + 与现役 FPS 单算子对拍复跑
- [ ] F4. Scope fidelity：无 SampleFPS 桶结构混入、无算法语义改动、现役零 diff、无 batch>1 硬编码

## Commit strategy
不执行任何 git 操作（AGENTS.md 约束）。改动留工作区供用户审查。

## Success criteria
1. FPSMB/warp kernel（N=7200 单 kernel）耗时较现役 6.5ms 下降（实测降幅，不预设硬目标；**若 task-1 判负收益，本条改为「记录实测值」，以 Success #5 为准**）
2. tie-free 输入 idx 与现役 bit 级一致；acc ti10 0.9741 / 全量 0.9569 逐文件一致（零回退）
3. fps_cache_mb 端到端 median 较 fps_cache（45.06ms engine 延迟全量 median）下降（**若 task-1 判负收益则本条豁免，以 #5 为准**）
4. 现役 FPS kernel/插件 diff 为零（F1 验证）
5. 若多 block 被测量裁决为不划算（M×(scan_us/G+2×sync_us) ≥ 6.5ms），如实记录并保留 warp kernel/FPSMB 作正确性基线，任务仍算完成（测量结论即交付）；此时 #1/#3 的性能要求豁免
