
## T1 实现记录 (2026-08-23)
- 改动仅 `deploy/trt_plugins/src/ballquerygroup_kernel.cu`：顶部 `#ifndef HPENET_BQ_DP_FUSION / #define ... 0`（L14-16）、融合 kernel `ball_query_dp_kernel_fast`（L18-63，包进 `#if HPENET_BQ_DP_FUSION`，匿名 namespace）、两个 launcher 宏切换（L129 / L166）。`bq_dp_kernel` 与 `#include "ballquery_kernel.h"` 原样保留。
- 编译验证：宏 0 `make -j` 通过（无 warning）；宏 1 通过，仅 `warning #177-D: bq_dp_kernel was declared but never referenced`（预期——保留作回退/对照路径，宏 1 时自然未引用）。验证后已恢复宏 0 并重编，build/ 下 .so 为现役路径。
- 环境注意：容器内无 conda（`source activate hpenet` 不可用），系统 nvcc 11.8 直接可用，build/ 缓存完好，make 无需 conda env。
- 宏 1 时 dp 基址 `dp += (bs_idx*3+0)*m*nsample + pt_idx*nsample` 与 bq_dp_kernel 写法对齐（计划伪代码写 `bs_idx*3*m*nsample`，等价）；d² 用原始 dx（先判据后乘 inv_r），padding 槽位 idx+dp 3 channel 同填。
- 未触碰：ballquery_kernel.cu/.h、ballquery_plugin.cpp、ballquerygroup_kernel.h、gridballquery_*、git。

## V1 单元对拍记录 (2026-08-23)
- 新建 `deploy/trt_plugins/tests/test_bq_dp_fusion.cu`（独立 nvcc 对拍程序，未进 CMake；include 顺序：先 ballquery_kernel.cu 再 ballquerygroup_kernel.cu，无重定义）。未触碰 src/ 任何文件，未做 git 操作。
- 编译命令（tests/ 目录下）：
  `nvcc -O2 -arch=sm_89 -I../include -DHPENET_BQ_DP_FUSION=1 test_bq_dp_fusion.cu -o test_bq_dp_fusion`
  （系统 nvcc 11.8，无 conda；include guard 齐全，两个 .cu 同翻译单元可行；首次编译报 GEN_DUP 枚举名笔误，改名 GEN_DUP_COORDS 后通过）
- 对拍机制：两路径写独立 buffer（不同 sentinel 预填防自比/防未写检测），idx 与 dp 均 memcmp 位级比较，失败打印首个 diff 位置与值。inv_radius 与现役 launcher 一致。空球 sentinel 验证融合 kernel cnt==0 时 idx+dp 完全不写。
- 用例矩阵：norm{0,1} × radius{0.1,10} × nsample{1,32,256} × shape{(2,4),(32,16),(2048,1024)}（N<S/N≈S/N≫S, B=1）+ 对抗 B=2（same_voxel/dup_coords/exact_sphere d²==r² 严格</N=1）。查询点均取自 xyz 既有坐标（d=0<r²）保证非空球。共 68 非空 + 1 空球。
- 结果：**69/69 PASS**，非空球 idx bit 100% 一致 + dp max-abs-diff==0（含精确球面点被严格 `<` 排除、首邻居 padding 槽位一致）。
- 空球 sanitizer：`compute-sanitizer --tool memcheck --error-exitcode 1 ./test_bq_dp_fusion --empty` → ERROR SUMMARY: 0 errors，exit=0（--empty 模式仅跑融合路径，规避现役 bq_dp 读垃圾 idx 的已知 §15.7 行为）。
- 结论：融合 kernel 与现役路径 bit 级等价，V1 通过，可进 T3（双编译 .so）/V2（整网 E2E）。

## T3 双编译产出两个 .so（2026-08-23）

**双编译命令**（build 目录 `deploy/trt_plugins/build`，临时改 `src/ballquerygroup_kernel.cu` L15 宏值，不动 CMakeLists）：
1. 宏=0 `make -j` → 现役 `libhpenet_plugins.so`
2. 宏=1 `make -j` → `cp libhpenet_plugins.so libhpenet_plugins_fusion.so`（出现预期 warning #177-D: bq_dp_kernel declared but never referenced，回退路径保留，非错误）
3. 宏改回 0 `make -j` 恢复现役 .so

**产物**：
- `build/libhpenet_plugins.so` 1,910,544 B（宏0，现役默认）
- `build/libhpenet_plugins_fusion.so` 1,931,328 B（宏1，融合路径）

**nm -D 符号验证**：
- fusion .so 含 `ball_query_dp_kernel_fast` ×2（stub+host）；现役 .so = 0 ✓
- 两者均含 `bq_dp_kernel`（gridballquery + ballquerygroup 两个 TU）、`ball_query_launcher_with_stream` ✓
- 注意：fusion .so 同时也含 `bq_dp_kernel` 符号（匿名命名空间内未被调用的回退 kernel 仍会被 nvcc 生成 host stub），故区分标志只看 `ball_query_dp_kernel_fast`。

`.cu` 宏已恢复 `#define HPENET_BQ_DP_FUSION 0`（grep 确认）。

## V2 整网 E2E（2026-08-23）✅ 通过

**方法**：新建 `deploy/v2_e2e_dump.py`（复刻 trt_inference.py 完整管线：preprocess_test seed100 voxel voting + preprocess_subcloud feat5 归一化 + pad + scatter-mean，dump 每文件 pred 为 npz）。同一 engine 文件、同一数据/预处理，仅换 `trt_plugins/build/libhpenet_plugins.so`（现役宏0 ↔ fusion 复制覆盖，测完恢复）。min_n=1024（engine profile 口径；实测子云均 >2024，与 2024 无差异）。ti10 = sorted 后 20% 尾部前 10 文件 0000068..77（41868 点）。

**逐点对比结果（bit 级 pred 100% 一致）**：
| engine | 现役 acc | fusion acc | 逐点一致率 |
|---|---|---|---|
| deploy/hpenet_v2_fp32.engine | 0.9636 | 0.9636 | **100.0000%** (0/41868 mismatch) |
| deploy/hpenet_v2_fp16.engine | 0.9635 | 0.9635 | **100.0000%** |
| fps_algo_fps_cache_prune_fp32.engine | 0.9707 | 0.9707 | **100.0000%** |
| fps_algo_fps_cache_prune_fp16.engine | 0.9705 | 0.9705 | **100.0000%** |

**acc 兜底**：fusion .so 全量 339（全目录 sorted，fps-prune-fullset 口径）prune fp32 = **0.9558** == 锚点 0.9558 ✓；ti10 prune fp32/fp16 = 0.9707/0.9705 == 锚点 ✓（prune 档锚点为 0.9707/0.9558，见 plugin.md v15.2；任务书写的 0.9741/0.9569 是 fps_cache/fps 档锚点，prune 档本就 -0.03/-0.12pp）。

**⚠️ 独立发现（非融合问题）**：`deploy/hpenet_v2_fp32/fp16.engine`（Aug 21 10:27/10:33 重导）ti10 acc=0.9636/0.9635，比同档 Aug 19 prune engine（0.9707/0.9705）低 ~0.7pp——engine 产物本身 acc 偏离锚点，来源待查（疑重导时配置/checkpoint 与 Aug 19 不同）。两 .so 在该 engine 上仍逐点 100% 一致，不影响 V2 结论，但建议上报核对 hpenet_v2_*.engine 的重导口径。

**.so 恢复确认**：`cp` 备份恢复，md5 == 宏0 现役版（2e6bb441…），`nm -D | grep -c ball_query_dp_kernel_fast` = 0，大小 1,910,544 B ✓。未改 src/、未 build engine、未做 git 操作。dump 产物在 /tmp/opencode/v2/。

## V3 nsys A/B 性能对比（2026-08-23）❗净收益为负，触发融合 1.5 立项条件

**方法**：复用 `/tmp/opencode/profile_trt.py`（单文件端到端：voxelize → 6 子云 pad→H2D→TRT enqueue → GPU scatter_mean，nvtx 包循环）。engine = `deploy/fps_algo_fps_cache_prune_fp32.engine`（Aug 19 锚点档），数据 = 真实 `data/RadarClassi/radarfullwl/raw/0000068.ply`（3707 点 → 6 子云 × 3523），iters=30 / warmup=5，CUDA_VISIBLE_DEVICES=0，仅换 .so：
- ref：`libhpenet_plugins.so`（宏0，md5 2e6bb441…，nm fast 符号=0）→ `/tmp/nsys_bq_ref.nsys-rep`（wall 30.321 ms/file）
- fused：cp `libhpenet_plugins_fusion.so` 覆盖（nm fast 符号=2）→ `/tmp/nsys_bq_fused.nsys-rep`（wall 30.494 ms/file）

**gpukernsum 结果（420 = 30 iter × 6 子云 × 2 stage；另有 grid=7/4/2/1 四形状，求和）**：

| 量 | 数值 (ns) | 占 ref GPU kernel 总时间 850,618,714 ns |
|---|---|---|
| T_bq（ball_query_kernel_fast ×4 形状和） | 208,970,493 | 24.57% |
| T_dp（bq_dp_kernel ×4 形状和） | 41,663,711 | 4.90% |
| T_fused（ball_query_dp_kernel_fast ×4 形状和） | 254,818,429 | —（29.93% 口径） |

（基线参考 plugin.md 的 26.7%/5.4%，本次口径 24.6%/4.9%，量级吻合。）

**结论**：
- 净收益 = (T_bq+T_dp) − T_fused = 250,634,204 − 254,818,429 = **−4,184,225 ns（≈ −0.49% GPU kernel 总时间，−139 µs/file，30 iter 口径）→ 净收益 < 0，融合 1 整体更慢**
- dp 写增量 = T_fused − T_bq = **45,847,936 ns = 1.10 × T_dp ≫ 0.5×T_dp（阈值 20,831,856 ns）→ 严重触发融合 1.5（dp 布局交错化）条件**
- 融合 kernel 逐形状都比纯 ball_query 慢 8~29%（grid=7: 280µs vs 258µs；grid=1: 66µs vs 37.8µs，小 grid 惩罚更大），strided dp 写（3 段 stride=m·nsample）主导瓶颈实锤，把 bq_dp 的收益全部吃掉还倒贴
- **T6 裁决依据**：净收益≤0 → 保持默认宏 0（现役）；增量>0.5×T_dp → 立项融合 1.5（交错 float3，目标 dp 写增量 ≤0.3×T_dp）。两条同时命中：**宏保持 0，fusion .so 留作对照档，立项融合 1.5**
- 稳定性佐证：两次采集 furthest_point_sampling_kernel = 342,859,235 / 342,852,938 ns（差 0.002%），A/B 环境一致

**.so 恢复确认**：`cp /tmp/opencode/libhpenet_plugins_ref_backup.so` 恢复，md5 = 2e6bb4416d10db38edda564d71b20a54 == 宏0 现役版，大小 1,910,544 B，`nm -D | grep -c ball_query_dp_kernel_fast` = 0 ✓。未改 src/、未 build engine、未做 git 操作。nsys 产物在 /tmp/nsys_bq_{ref,fused}.nsys-rep，csv 在 /tmp/opencode/kern_{ref,fused}.csv。

## Engine 重导（修正 profile opt_n=5500）——acc 未恢复，profile 假设被否定 (2026-08-23)

**重导命令**（工作目录 HPENet_v2-main，LD_LIBRARY_PATH=cuDNN 8.9.7 + TRT 8.6.1.6 + cuda-11.8，CUDA_VISIBLE_DEVICES=0）：
```
python deploy/trt_build.py --onnx deploy/hpenet_v2_plugin.onnx --output deploy/hpenet_v2_fp32.engine --min_n 2024 --opt_n 5500 --max_n 10000 --num_input_features 5   # EXIT=0
python deploy/trt_build.py --onnx deploy/hpenet_v2_plugin.onnx --output deploy/hpenet_v2_fp16.engine --fp16 --min_n 2024 --opt_n 5500 --max_n 10000 --num_input_features 5  # EXIT=0
```
build 日志确认 `Profile: min_n=2024, opt_n=5500, max_n=10000` 真实生效（非默认 4096），加载/更新 timing.cache，"Engine saved" + "Done!"，无段错误。

**产物**：
- `deploy/hpenet_v2_fp32.engine` 15,238,884 B（2026-08-23 19:01:56；旧 14,446,460 B Aug 21）
- `deploy/hpenet_v2_fp16.engine` 10,330,540 B（2026-08-23 19:05:22；旧 8,701,148 B Aug 21）

**重导后 ti10 acc（trt_inference.py --num_files 10，口径与锚点一致——同管线跑 fps_algo_fps_cache_prune_fp32.engine 复现 0.9707 验证过）**：
- fp32 = 0.9636（锚点 0.9707，差 -0.71pp）❌
- fp16 = 0.9636（锚点 0.9705，差 -0.69pp）❌

**结论 / 根因修正**：opt_n=4096→5500 对 acc 无影响（两 engine 均仍 0.9636）——"profile 偏离导致 conv tactic 次优 → 掉 0.7pp" 的假设被实测否定。真正的差异源是 **ONNX 本身**：`hpenet_v2_plugin.onnx`（Aug 21 10:31 导出，md5 ae4e0536…）与锚点 `fps_algo_fps_cache_prune.onnx`（Aug 19 17:16，md5 1122f4e7…）大小同为 11,899,013 B 但内容不同——Aug 21 重导出 ONNX 时模型权重/导出配置已变（与 V2 notepad §52 的怀疑一致）。要恢复 0.9707 需用 Aug 19 口径的 ONNX（或其导出时的 checkpoint）重导，而非调 profile。本次按任务约束未换 ONNX、未改 trt_build.py/src/、未做 git 操作。

## 用户决策：stride-4 + prune 为有意配置，0.9636 是预期精度（2026-08-23）

- 用户目标：**降低延迟**。`strides [1,4,4,4,4]` 与 `FPSPrune keep_rate=0.75` 均为有意选择（精度换延迟），保持现状。
- `deploy/hpenet_v2_plugin.onnx`（= 20260819-142944 stride-4 run 权重 + fps_cache_prune 档 k0.75）即目标部署模型。
- **ti10 acc 0.9636（fp32/fp16）是该组合的预期精度水平，不是回归**。此前三个假设已逐一被实测否定：①TRT profile（opt_n=5500 重导后仍 0.9636）；②prune 剪枝率（两 ONNX keep_rate 均 0.75，等价）；③融合 .so（4 组 engine 逐点 100% 一致）。真因：与 stride-2 锚点（20260812 run，0.9707）相比是不同训练配置，属设计选择差异。
- 目标部署 engine：`deploy/hpenet_v2_fp32.engine`（15,238,884 B）与 `hpenet_v2_fp16.engine`（10,330,540 B），均 2026-08-23 用 profile `min_n=2024/opt_n=5500/max_n=10000` 构建。
- ⚠️ 注意：AGENTS.md 中"BN 死层修复 = strides 改 [1,2,2,2,2]"的表述已被本决策覆盖——stride-4 是当前有意配置，未来勿按 AGENTS.md 旧描述把它"修复"回 stride-2。
- 延迟优化已关闭的路线（勿重复尝试）：GridBallQuery（搜索段慢 8.9~13.5×）、融合1 ball_query+dp（净收益 -0.49%）、FPS 多 block（grid.sync 每轮开销 > 每轮计算，结构性必亏）。

## FPS 微基准分解：扫描 vs 归约 + warp 归约原型 (2026-08-23)

**bench 程序**: `/tmp/opencode/fps-microbench/bench.cu`（独立 nvcc，未改仓库任何文件；系统 nvcc 11.8，L20 sm_89，`nvcc -O3 -arch=sm_89`；cudaEvent warmup 10 + 100 次取 mean/med，B=1，block=1024=opt_n_threads，temp 预填 1e10 与现役 fill 一致；xyz 随机 [-1,1]³）。

**三变体**（距离表达式逐字符照抄 `fps_kernel.cu`）：
- A = 现役 kernel 原样复制
- B = 只扫描（去树归约；`if (temp[0] > 1e30f)` 假读防优化消除，实测 0.49~1.70ms 非零，扫描未被消掉）
- C = warp 归约原型：`__shfl_down_sync` intra-warp argmax（strict `>` 保持低 lane，与 `__update` 语义一致）→ lane0 写 shared[32] → 1 次 `__syncthreads` → warp0 归约 32 项 → lane0 写 `w_dists_i[0]` shared 广播 → 第 2 次 `__syncthreads` → 全线程读。**每轮 2 次 `__syncthreads`（现役 11 次）**。

**耗时表**：

| 画像 | A full | B scan-only | C warp-red | 扫描占比 | 归约占比 | C 加速比 |
|---|---|---|---|---|---|---|
| N=3523, M=880 | 1.182 ms | 0.486 ms | 0.976 ms | 41.1% | **58.9%** | **1.211x (−17.4%)** |
| N=7200, M=1800 | 3.265 ms | 1.697 ms | 2.756 ms | 52.0% | 48.0% | 1.185x (−15.6%) |

两次运行结果一致（±0.3%）。正确性：A vs C idx 序列 **0/880、0/1800 mismatch（bit 级一致）**。⚠️ 实现要点：C 中 `old` 必须经 shared 广播给全 block（首版只 lane0 更新 → 878/880 mismatch 的非 tie-order bug）。

**裁决：归约占大头（48~59%）→ warp 归约值得立项（2a）**。
- N=3523（当前 stride-4 部署画像）归约占 58.9%，warp 归约实测整 kernel 提速 17.4%（≈1.21x）。按 nsys FPS 占 GPU kernel 36.6% 折算，整网 GPU kernel 时间可省约 **6.4%**。
- N=7200 大帧归约占 48%，提速 15.6%，仍显著。
- 上限注记：即使归约开销清零（B），也只有 41~52% 的 scan 本底——warp 归约已拿到理论收益的 ~1/3（17.4/58.9），属典型递减曲线，是 FPS 单 block 内"最后一口可吃的肉"。后续若还想动 FPS，只剩算法级（如近似 FPS/prune 已另立档）。

## CPP_trt3 逐子云同步开销调研（只读，未改代码/未做 git 操作，2026-08-23）

背景：stride-4 + prune，子云 N≈3523（min_n=2024 → 恒需 pad），每文件 6 子云。源码逐行读完：pipeline.cpp / trt_inference.cpp / trt_engine.cpp / subcloud_utils.cpp / preprocessor.cpp / scatter_mean.cu / cuda_utils.cu / pipeline.h / main.cpp。

### 1. 当前 sync 模式（每处调用点）

grep -n 结果（src/*.cpp）：
- `cudaMemcpyAsync`：pipeline.cpp L175/L182（子云 D2H，fp16/fp32 分支）、L240（最终合并 D2H）、L353/L357（子云 pos/x H2D）、L416/L420（d_src/d_idx H2D）
- `stream_.synchronize()`：pipeline.cpp L176、L183（子云循环内，紧跟 D2H）、L235（scatter 后）、L246（最终 D2H 后）、L553（warmup）

**每子云 1 次显式 sync**（L375-376 fp16 / L381-382 fp32），每文件 6 子云 = 6 次 + scatter 后 1 次（L434）+ 最终 D2H 后 1 次（L445）= **8 次/文件**；每子云 3 次 cudaMemcpyAsync（2 H2D + 1 D2H），每文件共 6×3 + 3（d_src/d_idx H2D + 最终 D2H）= 21 次，其中 D2H 7 次。

**为什么必须 sync**：CPU 依赖 GPU 数据做 trim（subcloud_utils.cpp L148-173 `trim_padding` 的 memmove）+ 转置收集（pipeline.cpp L392-397 写 all_src/all_idx）。这是纯 CPU 数据依赖，不 sync 无法继续；且下一子云的 H2D 写同一 d_pos_/d_x_（pipeline.h L88-89）+ TRT 输出 buffer 复用（trt_inference.cpp L106-110）形成 buffer 覆盖 hazard（该 hazard 靠 stream 顺序解决，但 CPU 读取必须 sync）。本质：**全流水线 CPU→GPU→CPU→GPU 严格串行，零重叠**。

### 2. 开销量化

- nsys 实测（/tmp/opencode/nsys_fp32.sqlite 复核）：cudaStreamSynchronize 264 次，**avg 431.5µs / max 1.066ms**（任务书引用 467µs/1.07ms，量级吻合；nsys_file_fp32 口径 avg 745µs/269 次）。注意 avg 值 ≈ 单子云 GPU 耗时——因为 CPU 在 enqueue 后立即等 sync，sync 等待的是"整个子云 GPU 推理 + D2H 完成"，非纯空等。
- memcpy avg 5.5µs/次（nsys_file_fp32），D2H/H2D 本身开销可忽略；真正的浪费是**结构性串行气泡**：GPU 完成子云 i 后空转等 CPU 做 i+1 的 preprocess+pad+trim+transpose（preprocessor.cpp 全量 gather/归一化 ~5 遍 N=2024 内存遍历），CPU 处理完 GPU 才继续。引用任务书口径：单文件 GPU kernel ~80% / sync+CPU ~20%。每文件 8 次 sync × ~467µs ≈ 3.7ms 泡在 sync 上，其中真正可省的是 GPU 空转气泡 + CPU trim/transpose + 冗余 sync。

### 3. 批量化方案评估

**方案 A：GPU 累积（首选）**——子云 logits 不 D2H。
- 改法：循环内去掉 L375-383 的 D2H+sync 与 L385-397 的 CPU trim/transpose/collect。infer 输出 (1,2,N_padded) 留在 GPU，用新写一个 ~20 行 CUDA kernel 做 trim+transpose（(2,N_padded)→(N_true,2) 行式）直接写入预分配的 GPU d_src（total_src×2×float）+ idx 切片 H2D 进 d_idx；跑完循环后仍走现役 scatter_add/div（scatter_mean.cu L15-52），最后**单次** D2H + 单次 sync。
- 收益：每文件 sync 8→2 次（scatter 后 + 最终 D2H，且 L434 那次可合并进最终 D2H，实际 8→1）；消除 6 次 D2H + 6 次 CPU trim/transpose/collect 往返；移除循环内 sync 后 CPU preprocess 天然与 GPU 推理软件流水重叠（stream 顺序保证 buffer 安全）。预估端到端省 **~10-15%**。
- 风险：低-中。需新 kernel 且与现役 scatter 路径 bit 级对拍；显存增量 = d_src(3707×2×4≈30KB)+d_idx(3707×8≈30KB) ≈ 0.06MB（雷达稀疏点云），可忽略；engine/context 零改动；TRT 输出 buffer 复用由 stream 顺序保证安全（无需双 buffer）。唯一注意：total_src 上限需按 max_n×最大子云数预算（当前 ~3707，安全）。

**方案 B：双缓冲（CPU/GPU 重叠）**
- 改法：双份 d_pos_/d_x_，CPU preprocess i+1 与 GPU infer i 显式并行。但 A 方案去掉 sync 后已自然获得同等效果（CPU 无阻塞即可跑 ahead），双缓冲只额外隐藏 ~5.5µs 的 H2D。收益小（**~2-5%**），且需双 context 或复杂 buffer 轮换。风险：中。**不推荐单独做**，只作为 A 的后续叠加项。

**方案 C：多 stream（各子云独立 stream）**
- 改法：6 子云各绑一 stream 全异步。但 TRT 单 context（pipeline.h L87）下多 stream enqueue 内部仍串行（共享 workspace），并行性需 6 个 context——每个 context 独立 workspace（引擎级数十 MB+），显存峰值 ×6，且 d_output_ 覆盖 hazard 需每 stream 独立输出 buffer。GPU 已 ~80% 占用，并发填充空间小。收益 **~0-5%**，风险高。**不推荐**。

### 4. 结论

首选方案 A。预估端到端收益 **~10-15%**（主要来自消除 6 次/文件的子云 CPU 往返 + 循环内 sync 移除带来的 CPU/GPU 软件流水重叠 + 尾部冗余 sync 合并），风险低-中（一个新 kernel + 两个 ~60KB GPU buffer + bit 级对拍验证）。附赠零成本优化：pipeline.cpp L434 的 sync 冗余（同 stream 的最终 D2H L439 已隐式排序于 scatter 后），可直接删。

CPP-BATCH VERDICT: 方案 A（GPU 累积，去循环内 D2H+sync，GPU 统一 scatter + 单次 D2H），预估端到端收益 ~10-15%，风险低-中。
