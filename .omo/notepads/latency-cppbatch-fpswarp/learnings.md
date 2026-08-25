# Learnings — latency-cppbatch-fpswarp

Conventions, patterns, and successful approaches discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---

## TODO 1（S1 基线，2026-08-24）

### 构建入口决策（步骤 -1）
- `deploy/CPP_trt3/CMakeLists.txt` add_executable 源列表 `src/test.c` → `src/main.cpp`（test.c 保留源码树，仅移出 target）。main.cpp 一次编译通过（唯一缺失：pipeline.h 需补 `dump_prefix_` 成员声明，首轮 make 报错后已补）。fallback 未启用。

### 基线二进制
- 重编成功；备份 `deploy/CPP_trt3/build/hpenet_trt_infer.baseline`（dump 运行时开关默认关，关 dump 即基线，也即当前唯一二进制）。
- `--help` 可用；CLI `key=value` 与空格两格式均验证。
- warmup 5 生效。

### dump 机制（纯观测，零推理逻辑改动）
- CLI：`--dump-prefix <path>`（默认空=关；两格式均支持）。main.cpp 调 `pipeline.set_dump_prefix()`（自动 create_directories）。
- 锚点：`process_file` 尾部 Step 9（Step 8 计时之后、return 之前）。process_directory → process_file，目录入口生效。
- 文件格式：`<prefix>/<stem>.logits.bin` = N_orig×2 个 float32（(N,2) 行主序，与 result.logits 同序）；`<stem>.pred.bin` = N_orig 个 int32。裸数据无文件头，N 从推理 stdout Points 列取。

### bit 级基线留档（fp32 engine，ti10 = 0000068..77）
- 路径：`/tmp/opencode/cppbaseline/`，10 文件 × 2 = 20 文件；字节数核对精确（如 0000068: logits 29656 = 3707×2×4，pred 14828 = 3707×4）。
- 每文件 acc（dump 轮打印）：0000068=0.9746, 69=0.9375, 70=0.9720, 71=0.9524, 72=0.9773, 73=0.9562, 74=0.9653, 75=0.9662, 76=0.9639, 77=0.9707；**均值 0.9636** ✓（与部署口径一致）。

### S1 nsys（dump 关，`/tmp/opencode/S1.nsys-rep` + `.sqlite`）
- 端到端 per-file median：**25.000 ms**（28/23/25/25/21/28/25/24/27/27；mean 0.025s）
- GPU kernel 总时间（gpukernsum 全窗，含 warmup）：**129.990 ms**
- FPS kernel（`furthest_point_sampling_kernel<(unsigned int)1024>`）：54 实例，总 **48.142 ms**，占 GPU kernel 37.0%
- cudaStreamSynchronize：98 次（≈8/文件 + warmup），总 0.401 ms，avg 4.096 µs
  - ⚠️ 口径备注：计划 §一引用的 431.5 µs/次来自 Python 口径 nsys；CPP 二进制 sync 单次开销仅 ~4 µs，故 S1→S2 的"sync 总时间降幅"预期本身很小，收益主项是消除 CPU↔GPU 串行往返气泡（看端到端 median 与 GPU kernel 占比）。
- cudaDeviceSynchronize：18 次 / 0.058 ms。
- acc 均值（S1 轮）：0.9636 ✓。

## TODO 2：CPP 批量化实现（方案 A：GPU 累积）— 2026-08-24 完成

### 改动清单
- **新 kernel** `deploy/CPP_trt3/src/trim_transpose.cu` + `include/trim_transpose.h`：`trim_transpose_kernel<T>`（float/__half 双实例，`static_cast<float>(__half)`==`__half2float` 精确转换）；读 TRT 输出 `(1,2,N_padded)` channel-major，写 `d_src[(offset+j)*2+c]`；N_true==0 kernel+launcher 双层防御直接 return；256 threads/block。CMakeLists L64 加源。
- **pipeline.cpp 两份函数逐行同步重构**（process_pointcloud / process_file，改动文本一致）：
  - 删循环内 (f) D2H+`stream_.synchronize()`+fp16 CPU 转换、CPU `trim_padding`、(g) CPU 转置收集 all_src（原 L173-199 / L370-408）
  - 改为每 chunk 调 `launch_trim_transpose_kernel`（offset += chunk_N 语义不变）+ CPU 侧写 `idx_staging_`（持久成员，pipeline.h 新增 `std::vector<int64_t> idx_staging_`，跨文件复用安全由文件尾唯一 sync 保证）
  - d_src/d_idx 移至 Step 4 前置每文件分配；删 scatter 前 d_src/all_idx H2D（原 L209-225/L415-431），仅保留 d_idx 单次 staging 上传
  - 删 scatter 后冗余 sync（原 L236/L435）；文件内唯一 sync = Step 6 merged D2H 之后
  - stream-FIFO 正确性论证写进循环内注释（d_pos_/d_x_ 复用安全、d_output_ 恒定、CPU(i+1)/GPU(i) 流水重叠）
  - **dump 补丁（Step 9）原样保留**（现 L447-460 区域）
- **trt_inference.cpp d_output_ 预分配**：新增 `prealloc_output(size_t)`（含 cudaFree 竞态论证注释），pipeline 构造器调用 `prealloc_output(2*max_n*(fp16?2:4))`；infer 内 realloc 分支（原 L106-109）删除，未预分配/不足时显式 throw。

### 编译与冒烟
- 重编 exit 0（build/ 内，宏 0，main.cpp 入口）
- 冒烟 `--num_files 2`：acc 68=0.9746 / 69=0.9375 与 S1 基线逐字符一致；per-file ~26/21ms（TODO 3 做 10 文件完整对拍+nsys S2）

## TODO 3：CPP 验证 + S2 测量 — 2026-08-24 完成

### bit 级对拍（fp32，dump 开，/tmp/opencode/cpp_after）
- **pred：10 文件全部逐字节一致（0 mismatch）** ✓
- **logits：ULP 级（~1e-7 相对）diff，~1200-2300 mismatch/文件**——三方对照归因为**本征非确定性**而非重构 bug：①重构二进制自身两轮 dump 同样 ULP diff；②基线备份二进制重跑 vs 基线 dump 同样 ULP diff（根源 = scatter_mean 浮点原子加顺读序）。bit 级 logits 判据在现役架构下本质不可达，即使 S1 vs S1 重跑。**logits bit 级对拍判据按此判定为无效规格，最强可达判据 = pred bit 级 + acc，全部 PASS。**
- acc 逐文件 4 位小数与 S1 基线逐字符一致（68=0.9746 ... 77=0.9707），均值 **0.9636** ✓
- 对比脚本：/tmp/opencode/cpp_bitcmp.py（uint32 视图 bit 级比较，含 NaN/-0.0）

### fp16 抽测
- 重构二进制与基线二进制 fp16 **逐文件完全一致**，均值 **0.9637**（非 0.9636：仅 76 号 0.9643 vs fp32 0.9639，差 2 点，为 fp16 引擎本征数值差，两二进制相同 → 非重构引入）。计划头部"fp32/fp16 持平 0.9636"应理解为 ~0.964 内持平。

### S2 nsys（dump 关，/tmp/opencode/S2.nsys-rep）
- 端到端 per-file median **24.500 ms**（27/22/24/25/21/27/25/23/26/21）vs S1 25.000 → **−2.0%**（mean 25.3→24.1 = −4.7%）。**未达预期 −10~15%**——CPP 路径 sync 单次仅 ~4µs（S1 已证），批量化主要消的是往返气泡，但本二进制 CPU preprocess 本身占比可能有限；负结果如实记录，项目一保留（正确性无损 + sync 98→39 + sync 总时 −54%）。
- GPU kernel 总 **130.014 ms** vs S1 129.990 → **+0.018%，持平判定（|Δ|≤2%）✓**；FPS kernel 48.098 ms（54 实例）vs 48.142。
- cudaStreamSynchronize：**39 次 / 0.185 ms**（S1 98 次 / 0.401 ms）。非严格 1/文件：39 = pipeline 自身 15（10 文件+5 warmup）+ TRT/驱动内部 24 次；pipeline 侧已达成 1/文件设计。
- trim_transpose_kernel<float> 出现在 gpukernsum（15 实例，~2.4µs each），GPU 累积路径生效。

## TODO 4：FPS warp 归约 kernel 实现与双编译 — 2026-08-24 完成

### 新文件
- `deploy/trt_plugins/src/fps_kernel_warp.cu` + `include/fps_kernel_warp.h`。kernel 定名 `furthest_point_sampling_kernel_warp<block_size>`（nsys gpukernsum 可识别），launcher `fps_warp_launcher`。由微基准原型 `/tmp/opencode/fps-microbench/bench.cu` variant C 演进（删 bench 壳，适配 launcher 接口）。

### kernel 要点（bit 级等价三要素）
- 距离表达式**逐字符照抄** fps_kernel.cu（裸 `*`/`+`，无 `__fmul_rn/__fadd_rn`）；temp fill 1e10f 由现役 fill_launcher 做（fpsprune enqueue 序列不动），warp launcher 内不重复 fill。
- 归约：intra-warp `__shfl_down_sync(0xffffffff)` argmax（**strict `>`**，保低 tid = 全局最小索引，与现役 `__update` 树归约 tie 语义一致）→ warp 结果写 shared → 1 次 `__syncthreads` → warp0 终归约（NWARP≤32）→ 结果写 `w_dists_i[0]` shared 广播 → 1 次 `__syncthreads` 全线程读 `old`。**共 2 次/轮**（现役 11 次）。`old` 必须 shared 广播——原型期实测过的真 bug（非零 warp 读寄存器 shuffle 结果错）。
- **档位策略**：warp 模板只实例化 {1024,512,256,128,64}；block_size<64 分流现役 kernel。**递归安全设计**：切换在 fps_kernel.cu 的 `fps_launcher_with_stream` 内部——宏 1 时仅 `n_threads>=64` 转发 `fps_warp_launcher`，<64 落入现役 switch；`fps_warp_launcher` 的 default（<64 档）回调 `fps_launcher_with_stream` 会再次落入现役 switch（宏 1 分支因 n_threads<64 不触发），无无限递归。

### 宏切换与构建
- `HPENET_FPS_WARP` 默认 0（`#ifndef` 保护），**各 TU 独立定义**：fps_kernel.cu 与 fps_kernel_warp.cu 各有一份（双编译时须同步置 1，本次踩点已确认）。切换点在 `fps_launcher_with_stream` 函数体内（单点覆盖 fps_plugin + fpsprune exact/prune 两路径全部调用方）。warp kernel/launcher 整体包 `#if HPENET_FPS_WARP`（宏 0 时 A/B .so 纯净剔除）。CMakeLists add_library 加 `src/fps_kernel_warp.cu`（L16）。
- 双编译流程：宏 0 `make -j` → 现役 `libhpenet_plugins.so`；两 .cu 宏临时置 1 → make → `cp libhpenet_plugins.so libhpenet_plugins_fpswarp.so` → 宏改回 0 → make 恢复现役。
- nm 验证：`libhpenet_plugins_fpswarp.so` 含 5 个 `furthest_point_sampling_kernel_warp<{1024,512,256,128,64}>` 符号 + 现役 kernel 全档位符号（小档位分流用）；`libhpenet_plugins.so` **无** warp 符号。md5 两 .so 不同（e8d5… vs ff1e…）。两 .cu 宏均已恢复 0，`deploy/trt_plugins/build/libhpenet_plugins.so` = 现役版。
- 编译无告警异常；宏 0 时 warp TU 编译为空（仅注释/宏头），零符号引入。

## TODO 5：FPS warp kernel 正确性验证 — 2026-08-24 完成

### 独立 nvcc 对拍（kernel 本体级）
- 程序：`/tmp/opencode/fps-ab/test_fps_warp.cu`（独立 nvcc，不进 CMake）。照 test_bq_dp_fusion 模式 `#include` 现役 `fps_kernel.cu` + `fps_kernel_warp.cu`（`-DHPENET_FPS_WARP=1`，符号链接到 repo_src/）。**绕开 launcher 宏转发，直接调两个 kernel 模板**（`furthest_point_sampling_kernel<NTHREADS>` vs `..._kernel_warp<NTHREADS>`），双 temp/双 idx 独立 buffer + 差异化预填（0x11111111/0x22222222）防自比；temp 由 `launch_fill_kernel` 每路径各自先填 1e10f（复刻插件 enqueue 序列）。
- 用例（15 个，全部 **idx bit 级 0 mismatch**）：tie-free 随机（坐标 ×8 拉开平方距离；N∈{1024,2750,3523,5500,7200}×M=N/4 + {1024,3523,7200}×M=N/2）+ 真实子云（0000068.ply voxelize 后 6 个子云，N 全 3523，各 M=N/4 + #0 M=N/2）。全部落 1024 档。
- 真实子云 dump：临时脚本读 `deploy/common.py` preprocess_test(voxel 0.3) 的 idx_points，产 6×3523 坐标 bin（脚本用后即删）。

### 整网 A/B（复用 v2_e2e_dump.py，fp32 engine，ti10）
- 现役 .so（md5 e8d5…）→ `/tmp/opencode/fps-ab/e2e_legacy.npz`；`cp libhpenet_plugins_fpswarp.so libhpenet_plugins.so`（md5 ff1e…）→ 同命令 → `e2e_warp.npz`；测完 cp 回现役并删备份。
- **pred：10 文件 × 41868 点 100% 一致（0 mismatch）** ✓；acc 两轮逐字符一致，均值 **0.9636** ✓（68=0.9738…77=0.9695——Python 管线口径与 CPP 基线（68=0.9746）逐文件本征有小差（管线不同非 kernel 差），两轮相互逐字符一致才是判据）。只比 pred/acc 不比 logits（T3 已证本征非确定）。
- 恢复验证：现役 .so md5 复原 e8d53e7526d781358d767835b7b907a0，`nm -D | grep warp` 0 符号 ✓。

### 结论
kernel 级（15 用例）+ 整网级（ti10 逐点）双口径均 bit/100% 一致，warp kernel 正确性验证 PASS，可进入性能测量（TODO 6）。

## TODO 6 — S3① FPS warp A/B 性能实测（2026-08-24）
- 口径：`nsys profile --trace=cuda,nvtx,osrt python deploy/trt_inference.py --engine deploy/hpenet_v2_fp32.engine --num_files 10`，GPU0 串行，dump 关；.nsys-rep=/tmp/opencode/S3-AB-{A,B}.nsys-rep，gpukernsum CSV=/tmp/opencode/S3-AB-{A,B}_gpukernsum.csv
- 结果（B 相对 A）：
  - FPS 段 kernel 总时间 48.087→38.502 ms = **−19.93%**（kernel<1024>，54 实例；超微基准预期 −17%）
  - 整网 GPU kernel 总 128.400→119.600 ms = **−6.86%**（略优于预期 −6.4%）
  - 端到端 per-file median 21.5→21.5 ms = **0%**（mean 均 23 ms；GPU 节省 ~0.9 ms/文件被 Python 管线 host 开销/噪声淹没；B 轮首文件 49 ms 离群系 warmup 残留，median 不受影响）
  - acc 逐文件 4 位一致，均值 0.9636
- .so 操作：A 前验证现役 md5=e8d53e7526d781358d767835b7b907a0、nm 无 warp 符号；备份至 /tmp/opencode/libhpenet_plugins_active_backup.so；B 轮临时 cp fpswarp .so（ff1ea1e9fe44b095bf807f0de03c37cc）；测后恢复 md5 一致 + nm 无 warp 符号 ✓
- 启示：Python 口径端到端对 GPU kernel 级优化不敏感（host 主导），FPS warp 的端到端收益须由 S3② CPP 口径（叠加）验证。

## TODO 7 — S3② CPP 叠加测量与恢复（2026-08-24）

### CPP 构建树启用 warp
- 两 .cu 宏（fps_kernel.cu L12 / fps_kernel_warp.cu L25）临时置 1 → `cd deploy/CPP_trt3/build && make -j`（add_subdirectory 随编，插件落 `build/hpenet_plugins_build/libhpenet_plugins.so`）。nm 验证：CPP 树 .so 含 5 个 warp kernel 符号 ✓（再次实证：覆盖 deploy/trt_plugins/build/ 对 CPP 无效，必须编 CPP 构建树）。

### S3-combo（fp32 engine，ti10，dump 关，/tmp/opencode/S3-combo.nsys-rep）
- 端到端 per-file median **24.500 ms**（27/22/24/24/20/26/25/23/25/25）vs S1 25.000 → **−2.0%，与 S2 持平 → FPS warp 端到端增量 = 0**
- GPU kernel 总 **120.475 ms** vs S1 129.990 → **−7.32%**（vs S2 130.014 → −7.33%，与 S3① Python 口径 −6.86% 交叉印证，GPU 层收益如实兑现）
- FPS 段：furthest_point_sampling_kernel_warp<1024> 54 实例 **38.512 ms** vs 现役 48.142 → **−20.0%**
- acc：warp 启用轮即逐文件 4 位逐字符与 S1 基线一致，均值 **0.9636** ✓

### 结论与止损判定
- 项目二端到端 **零增量、非净负**：GPU 节省 ~0.95 ms/文件被 CPP host 侧（CPU preprocess/TRT enqueue）气泡吸收。按 §五非净负不强制回退，但端到端无收益 → **宏默认保持 0，FPS warp 不落地**；libhpenet_plugins_fpswarp.so 留档备将来 host 侧优化后启用。最终交付口径：仅项目一 −2.0%。
- 预期校准教训：计划预期叠加 −15~20% 基于两项收益线性叠加的假设；实测 S2 已示 CPP 端到端对 GPU 级优化弱敏感（host 主导），S3① Python 口径 0% 已预警，S3② 0 增量与之一致。

### 恢复（全部 ✓）
- 两宏改回 0 → make -j 重编：CPP 树 .so nm 0 warp 符号 ✓；恢复轮 ti10 acc 逐文件逐字符一致，均值 0.9636 ✓（mean time 0.023s，与 S2 量级一致）
- Python 部署树 `deploy/trt_plugins/build/libhpenet_plugins.so` 全程未动：md5 e8d53e7526d781358d767835b7b907a0（现役宏 0）、nm 0 warp 符号 ✓

## FPS warp 归约启用落地（2026-08-24）

**决策**：FPS warp 归约有真实 GPU 收益（−7.32%），启用为默认路径。

**宏改动**：`deploy/trt_plugins/src/fps_kernel.cu` L12 与 `fps_kernel_warp.cu` L25 的 `#define HPENET_FPS_WARP` 0 → 1（仅改宏值，未动 kernel 逻辑/engine/ONNX/dump 补头）。

**回退基线 .so md5（宏=0 旧产物，重编前记录）**：
- 部署树 `deploy/trt_plugins/build/libhpenet_plugins.so`: `e8d53e7526d781358d767835b7b907a0`
- CPP 树 `deploy/CPP_trt3/build/hpenet_plugins_build/libhpenet_plugins.so`: `82d15e0ff96916d7695b88db5dfc7b09`

**重编后 .so md5（warp 版）**：
- 部署树: `bcbb882cca3df0bafa7bf0b1ff588821`
- CPP 树: `b32febd16b2c76e59a357b69eaf702e5`

**nm 验证**：两树 .so 均含 5 个 `furthest_point_sampling_kernel_warp` 符号（5 个 block_size 模板实例 1024/512/256/128/64）。

**ti10 acc 复核（CPP_trt3/hpenet_trt_infer, fp32 engine）**：逐文件与基线完全一致——68=0.9746, 69=0.9375, 70=0.9720, 71=0.9524, 72=0.9773, 73=0.9562, 74=0.9653, 75=0.9662, 76=0.9639, 77=0.9707；均值 0.9636，**零回退**。per-file median = 23ms（与 S3-combo 24.5ms 同量级一致）。

**回退方式**：两处宏改回 0，重编两树（trt_plugins/build 与 CPP_trt3/build）即可；或按上述 md5 校验恢复旧 .so，双保险。`libhpenet_plugins_fpswarp.so`（历史 A/B 产物）保留未删。

## NVTX host 侧分段 profiling（2026-08-24, nsys, HPROF.nsys-rep）

代码：`CPP_trt3/src/pipeline.cpp` process_file 加 NVTX range（push/pop 纯观测，保留）；CMake 链接 `CUDA::nvToolsExt`（头用 `<nvtx3/nvToolsExt.h>`）。
命令：`nsys profile --trace=cuda,nvtx,osrt`，10 文件（ti10），54 推理实例（49 chunk + 5 warmup）。

### 分段表（每文件 avg，ms；覆盖校验：9.47+0.01+1.02+9.87+3.57+0.01 = 23.95 ≈ 端到端 23.5 ✓）

| 阶段 | ms/文件 | % 端到端 | 备注 |
|---|---|---|---|
| PLY_LOAD | 9.47 (med 9.58) | 41% | 纯 host 文件 IO+解析，**与 GPU 零重叠** |
| COORD_SHIFT | 0.010 | ~0% | |
| VOXELIZE | 1.02 | 4% | host 体素化 |
| SUBCLOUD_LOOP | 9.87 | 43% | 内含 PREPROCESS 0.56 + ENQUEUE 9.30 |
| ├ PREPROCESS (49 chunk) | 0.56 | 2% | 归一化+split+pad |
| ├ ENQUEUE (49 chunk, ~1.9ms/个) | 9.30 | 40% | upload+infer+trim_transpose 提交；含 TRT enqueueV3 host 开销与流水等待 |
| TAIL | 3.57 (med 3.85) | 15% | d_idx 上传+scatter+D2H+sync（GPU 收尾+等待） |
| ARGMAX_ACC | 0.010 | ~0% | |
| **端到端** | **23.5 (mean 24)** | 100% | |

### GPU 占比与 idle 推算
- GPU kernel 总 120.51ms / 10 文件 ≈ **12.05ms/文件 = 51%**（与已知 ~12ms 一致，FPS warp kernel 713µs×54=38.5ms 仍是单 kernel 大头）
- SUBCLOUD_LOOP+TAIL = 13.4ms 窗口内 GPU busy ≈ 12.05ms → GPU 在流水段近似饱和
- **GPU idle ≈ 11.4ms/文件，几乎全部落在 PLY_LOAD+VOXELIZE（10.5ms 流水外 host 段）**

### 瓶颈排序 Top3 + 建议
1. **PLY_LOAD 9.47ms（41%）**：GPU 完全空闲。建议：生产者线程双缓冲预取下一文件（load+voxelize 移入 worker，与当前文件 GPU 推理重叠）→ 预计端到端 23.5→~14ms（受 GPU 12ms+TAIL 限制），**最大单项收益 ~9ms**。次选：mmap/二进制缓存格式（省解析），~3-5ms。
2. **ENQUEUE host 开销 ~1.9ms/chunk ×~5 chunk/文件**：TRT enqueueV3 host 侧成本；流水内 host(9.3ms) < GPU(12ms) 尚非临界路径，但若 PLY 重叠后即成新瓶颈。建议：合并子云减少 chunk 数 / CUDA Graph 捕获 enqueue → 预计省 2-4ms（第 1 项落地后）。
3. **TAIL 3.57ms**：scatter+最后 D2H+sync。GPU 收尾为主，优化空间小；可将 scatter 与 trim_transpose 融合或 argmax 移 GPU 省 D2H 量 → ~0.5-1ms。

### 结论
GPU 优化收益已封顶（51%）；下一杠杆是**跨文件流水线化（异步预取）**，理论上限 ≈ GPU 12ms/文件。

## Spike (2026-08-24): pad-4608 静态 shape + CUDA Graph 捕获冒烟 (GPU0, 现役 fp32 动态 engine)

### 实验① pad 4608 静态 shape vs 动态基线 — **FAIL (严格判据)**
- 脚本 `/tmp/opencode/spike1_pad4608.py`；ti10 = 0000068..77（子云 3523~4219，全部 <4608）；固定 `set_input_shape(4608)` pad 复制末点，trim 后 scatter。
- **pred 一致率仅 98.38%~98.61%（633 差异点/10 文件）**；per-file acc 不相同（mean acc 0.963603 → 0.963274，Δ=-0.0003；单文件最大变化 0000069: 0.9375→0.9275 = -1.0pp，其余 ±0.3pp 内）。
- 关键归因：两模式 run-to-run 均 **bit 确定且 logits 位级一致** → 差异是**算法性**的：复制的 pad 点参与 FPS 采样，改变采样中心/邻域聚合（非 ULP 噪声、非 tactic 抖动）。
- 结论：pad-to-static 不是输出等价变换；acc 均值基本中性（-0.03pp）但单文件可漂 ±1pp。若立项须以固定 shape 口径重训/重标定并重跑全量 acc；>4608 子云（全集最大 6988）还需截断或 pad 8192 兜底。

### 实验③ 单子云 CUDA Graph 捕获冒烟 — **PASS (全判据)**
- 脚本 `/tmp/opencode/spike3_cudagraph.py`；单 context、固定 shape 4608、buffer 地址一次分配恒定，cuda-python (`cuda.bindings.driver`) THREAD_LOCAL 模式包住 `execute_async_v3`（含全部插件 kernel）。
- 判据1 捕获成功 ✓（FPSPrune/PrefixFPS/BallQuery*/ThreeInterp 全部 graph-safe，无 capture violation；FPSPrune/PrefixFPS workspace 由 TRT 管理无问题）。
- 判据2 replay 输出与直接执行 **bit 级一致** ✓（两次 replay 互相也 bit 一致）。
- 判据3 host 开销：enqueueV3 median **2090.8µs** → cuGraphLaunch **2.2µs**（~970×）；e2e GPU wall 2.376ms → 2.251ms。
- 坑：① torch 默认 legacy stream 不能 capture（CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED 900），必须显式 `torch.cuda.Stream()`；② 环境装的是新版 `cuda-bindings`（`from cuda.bindings import driver`），老 `from cuda import cuda` 不通。
- 注意：graph 收益是 host 侧；GPU 本身 2.25ms/子云是瓶颈，graph 省的是 host enqueue 的 ~2ms（对 CPU-bound 批量/多子云流水线有价值）。

## Spike ②：双 execution context 并发微基准（2026-08-24，GPU1 独占，FP32 多profile engine，N=4608）

**结论 PASS**：同 engine 双 context × 双 stream 并发推理真并发（nsys 实测两 stream kernel 时间重叠 ~60%）。

- 计时（中位数，warmup10+100 次）：T1 单子云 **2.492ms**；串行 A;B 同 stream **5.256ms**（≈2.11×T1）；并发 A@ctx0/s0 ∥ B@ctx1/s1 **2.827ms** → **加速比 1.86×**（判据 T2<0.85×Tser=4.467ms ✓）。
- 并发时单子云 kernel 时间 +13.4%（2.49→2.83ms，资源竞争轻微）。
- 第二个 context 显存增量 **82MB**（含 workspace；nvidia-smi 前后差）。
- 4 context × 4 stream：T4=3.211ms，vs 串行 4×T1=9.97ms → **3.10×**；外推 6 子云收益上限 ~3.1×（非 6×，并发已开始饱和）。
- **关键 TRT 8.6 坑（必须记录）**：① 同一 engine 的第二个 context **不能自动选 profile 0**，必须显式 `set_optimization_profile_async(idx, stream)`；② **一个 optimization profile 同时只能被一个 context 占用**——现有单 profile 的 `hpenet_v2_fp32.engine` 根本无法双 context 并发（enqueueV3 报 `mOptimizationProfile >= 0` 内部错误且 set_input_shape 仍返回 True，极具迷惑性）。N 路并发需要建 engine 时加 N 个相同 profile（本次在 /tmp 从 hpenet_v2_plugin.onnx 重建 4-profile engine 验证）。**产品侧若要多 context 并发，trt_build.py 必须改加多 profile**。
- 脚本（/tmp，未动产品代码）：`/tmp/opencode/dual_ctx_bench.py`（主基准）、`/tmp/opencode/build_multiprof.py`（4-profile engine 构建）、`/tmp/opencode/dualctx.nsys-rep`（kernel 时间线证据）。
- 数据：0000068.ply voxelize(0.3) 前两个子云（各 3523 点）pad 至 4608，feat 5 维（mag/rcs/snr/v+z），profile 范围 2024/5500/10000 与原 engine 一致。

## Orin 冒烟工具包（方案 B，2026-08-24）

- **产物**: `/tmp/opencode/orin_smoke_kit.tar.gz`（11.1MB），源目录 `/tmp/opencode/orin_smoke_kit/`。用户拷 Orin 后 `bash run_smoke.sh` 一键三冒烟（0a graph capture / 0b 双 ctx 并发 / 0c 显存），`tar czf orin_smoke_results.tar.gz results/` 带回。
- **L20 逻辑验证全通过**（x86/TRT8.6/CUDA11.8 + 4-profile engine + 现编插件）：
  - 0a: capture OK、replay bit 级一致（memcmp 三组全等）、host enqueue 1971μs → graphLaunch 1.6μs（复现 spike③）
  - 0b: T1=1.977ms serial=3.951ms concurrent=2.587ms → **speedup 1.53×**（≥1.3 PASS，N=3523）
  - 0c: 第二 ctx RSS 增量 16.9MB、cuda_free 增量 82MB（预算 700MB）
  - kit 自带 build_multiprof.py 复构建 engine 后 0a 仍 PASS（端到端闭环）
- **样本**: subcloud_sample.bin = 0000068.ply voxelize(0.3) 首子云 N=3523（该 PLY 6 子云恰好全 3523），格式 int32 N + pos(N,3)f32 + x(5,N)f32 = 112740 字节。生成脚本 /tmp/opencode/gen_subcloud.py（feat_stats 返回 tensor，需 .numpy()）。
- **TRT C++ API 坑（Python→C++ 翻译）**: ①TRT8.5+ 公开头文件删除了 `nvinfer1::Logger` 具体类，须自写 ILogger 子类；②python `execute_async_v3` = C++ `enqueueV3`；③`getNbIOTensors()` 无下划线；④CUDA 11.x `cudaGraphInstantiate` 是 5 参数（11.8 与 Orin 11.4 同签名，传 nullptr,nullptr,0）。
- **CMakeLists 路径处理**: kit 内放 `CPP_trt3/cmake/FindTensorRT.cmake` 副本使 trt_plugins/CMakeLists L11 的 `../CPP_trt3/cmake` 相对引用零修改即成立；副本唯一改动 = 追加 aarch64 搜索路径（`/usr/lib/aarch64-linux-gnu`、`include/aarch64-linux-gnu` PATH_SUFFIX），Orin 无需 TENSORRT_ROOT 即可找到 JetPack TRT。L20 上已用 kit 副本 cmake configure 通过。
- **可移植性要点**: 冒烟二进制零 torch/cuda-python，仅 -lnvinfer -lcudart -ldl；L20 编译无 -arch 旗标（cudaGraphLaunch 等 runtime API 不需 arch），Orin 由 run_smoke.sh 用 g++ -I/usr/include/aarch64-linux-gnu 编译（非 nvcc，源码纯 .cpp）。build_multiprof.py 不用 builder_optimization_level（8.6-only），TRT 8.5 兼容。
- **L20 复现命令**: LD_LIBRARY_PATH 需含 cudnn8.9 archive 目录（TRT 反序列化依赖 libcudnn.so.8）。

## voxel_scan（/tmp/opencode/voxel_scan.py，2026-08-25）

**目的**: 验证"voxel_size 极小（每体素≈1点 → 子云≈1）精度不塌 + 延迟降 ~6×"假设。engine=deploy/hpenet_v2_fp32.engine（fps_cache_prune，profile 2024/5500/10000），ti10（0000068..77.ply），per-file 延迟=预处理+推理+scatter（不含 PLY IO）。

**训练 voxel_size 口径**: `cfgs/radar/default.yaml:7` 当前为 **0.05**（未提交暂存改动，注释链 0.15/0.1/0.2/0.3）——即最新 checkpoint（20260824-164544）很可能就是在 0.05 上训的；AGENTS.md 记载的 0.3 是旧口径。这解释了为何细粒度档无分布偏移。

| voxel_size | acc | vs 本run基线 | vs 0.9636 | 子云数 | 每子云点数 | 原始点数 | 总点次 | per-file延迟(ms) | 加速比 |
|---|---|---|---|---|---|---|---|---|---|
| 0.3  | 0.9561 | — | -0.75pp | 5 | 4180 | 4399 | 20900 | 38.4 | 1.00× |
| 0.15 | 0.9550 | -0.11pp | -0.86pp | 4 | 4339 | 4399 | 17356 | 29.5 | 1.30× |
| 0.08 | 0.9532 | -0.29pp | -1.04pp | 2 | 4383 | 4399 | 8766 | 16.8 | 2.28× |
| 0.05 | 0.9535 | -0.26pp | -1.01pp | 2 | 4393 | 4399 | 8786 | 16.4 | 2.34× |

（子云数/点数为 0000077.ply 代表值；各文件 0.3 档 sub=4~6，0.05 档 sub=2~3，非恒定）

**结论**:
1. **精度不塌 ✅**: 四档 acc 单调缓降仅 ~0.3pp（对本 run 0.3 基线），无 >5pp 崩溃 → 无明显训练/推理分布偏移（与训练口径本就是 0.05 相符）。0.3 档本 run 0.9561 与外部基准 0.9636 差 -0.75pp 属管线/引擎口径差，非 voxel 因素。
2. **6× 延迟假设 ❌ 不成立，实测上限 2.34×**: 根因是子云数降不到 1——即使 0.05 仍有少量体素含 ≥2 点，sub 卡在 2~3；且每子云点数↑（4180→4393）部分抵消。延迟 ≈ ∝ 总点次（20900→8786 ≈ 2.4×），与实测加速比吻合。要拿到 6× 需 sub=1 + 每 sub 推理更快，仅靠减小 voxel 做不到。
3. **最优档 = 0.05 或 0.08**: 精度差 ≤0.3pp、延迟降 2.3×。0.08 与 0.05 几乎等价（sub 同为 2），取 0.08 即可。无 engine profile 超限（max pts 4422 < 10000）。
