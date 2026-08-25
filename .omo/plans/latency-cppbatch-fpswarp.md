# CPP 批量化 + FPS warp 归约 延迟优化计划

> 创建：2026-08-23（R1 任务重排/测量时序/叠加测量/源码锚点/基线留档/stream-FIFO；R2 CPP 自 dump 基线、双构建树实证、迭代口径；R3 一致性残留/dump 开关/CMakeLists 源列表/fp16 抽测/kernel 定名；**R4 双评审（Metis+Momus）：构建入口决策（hpenet_trt_infer 实编 test.c 非 main.cpp——CLI/dump 锚点全部依赖此决策）、d_output_ cudaFree 竞态防御、warp kernel 小档位分流、acc 逐文件口径量化、持平容差、对拍止损条款、__syncthreads 11 次事实修正、宏切换点定位于 launcher 内部、staging 生命周期、deploy 树 .so 先重编**）| 状态：待执行
> 部署口径：stride-4 + FPSPrune(keep_rate=0.75)（用户有意配置），engine = `deploy/hpenet_v2_fp32.engine` / `hpenet_v2_fp16.engine`（2026-08-23 构建，profile min=2024/opt=5500/max=10000），**ti10 acc 均值基线 = 0.9636（fp32/fp16 持平；"逐文件一致"定义见 §五验收口径）**

## 一、项目动机

### 共同背景

当前部署单文件端到端 ~15-19ms，其中 GPU kernel ~80%、CPU/sync ~20%。GPU kernel 内部：FPS 36.6%（最大瓶颈）、ball_query 26.7%（已到头：GridBallQuery 负、融合 1 负，见 `.omo/notepads/ballquery-dp-fusion/verdict.md`）。本计划打两块剩余空间：**CPU/sync 气泡**（项目一）与 **FPS 归约开销**（项目二）。

### 项目一：CPP 批量化（方案 A：GPU 累积）

- **现状**：`CPP_trt3/src/pipeline.cpp` 的 `process_file`（L350-459）对每个子云做 `infer → cudaMemcpyAsync(D2H) → stream_.synchronize() → CPU trim → CPU 转置收集`，每文件 6 子云 = **8 次 sync/文件**（6 子云各 1 + 尾部 2：L434/L445；前一份 `process_pointcloud` L173-198/L235 为对应副本），CPU↔GPU 全串行零重叠。
- **根源**：每子云 logits D2H 到 CPU 做 trim+转置写 `all_src`/`all_idx`（L385-397），最后又 H2D 上传（L416-423）做 GPU scatter——纯 CPU↔GPU 往返；且 scatter 后 sync（L434）冗余（同 stream 后续 D2H 已隐式排序）。
- **nsys 实测**：`cudaStreamSynchronize` avg **431.5µs**/max 1.066ms（264 次/30 文件，Python 口径同构参照）。
- **预估收益**：**~10-15% 端到端**（消 6 次 D2H+sync+CPU 往返，循环内 sync 移除后 CPU preprocess 与 GPU 推理天然软件流水重叠）。

### 项目二：FPS warp 归约（2a）

- **微基准实证（2026-08-23，/tmp/opencode/fps-microbench/）**：FPS 慢在**归约**而非扫描——归约占比 **58.9%**（N=3523）/ 48.0%（N=7200）；现役每轮 **11 次 `__syncthreads`**（fps_kernel.cu L49 预同步 1 次 + L51-60 树归约 10 次；R4 事实修正：此前误写 10 次）是主因。
- **warp 归约原型实测提速 1.211×（N=3523，−17.4%）/ 1.185×（N=7200）**，采样 idx 与现役 **bit 级一致**（0 mismatch，原型代码可直接演进）。
- **已关闭的 FPS 路线**（勿重试）：多 block cooperative（grid.sync 每轮 1.94µs > 每轮计算 1.81µs，结构性必亏）；GridBallQuery / SampleFPS / FlashFPS（见 plugin.md §14/§15）。
- **预估收益**：FPS 占 GPU kernel 36.6% × 17.4% ≈ **−6.4% GPU kernel ≈ −5% 端到端**。
- **注**：扫描本底占 41~52%，这是 FPS 单 block 内最后一块可吃的肉。

## 二、具体计划

**测量时序（全计划强制）**：`S1 基线 → S2 项目一（CPP 改后，warp 仍宏 0）→ S3 项目二（①FPS A/B = Python 管线换 deploy 构建树 .so；②叠加 = CPP_trt3 构建树内启用 warp 重编 CPP）`。增量归因：项目一 = S2−S1；项目二 = S3① 内部 A/B；叠加 = S3②−S1。**所有 nsys 测量串行**（GPU 0 独占）；**所有延迟测量轮不开 dump**（dump 仅对拍轮开启）；**各轮 .nsys-rep 按 S1/S2/S3-AB/S3-combo 命名留档，作为 F1 时序合规证据**。

### 步骤 -1（R4 新增，先于一切）：构建入口决策

**事实**：`hpenet_trt_infer` 的 CMake `add_executable`（CMakeLists L48-64）源列表是 `src/test.c`，**`src/main.cpp` 未编译进 target**。test.c 的 main() 硬编码 engine/stats/数据目录、无任何 CLI、`readdir` 无序遍历全目录、不调 warmup、走 wrapper→`process_pointcloud`——**`process_file` 在该二进制中不执行**。计划全部 CPP 测量/dump 机制依赖 main.cpp 的 CLI（--num_files/--engine/warmup/process_directory/per-file 统计）。

**决策**：把 `add_executable` 的 `src/test.c` 替换为 `src/main.cpp`（test.c 文件保留在源码树不删，仅移出 target——它含另一个 main() 符号，不能同时链接）。改后入口 = main.cpp：CLI 完整、warmup（main.cpp:171）、process_directory→**process_file**（dump 锚点成立）、per-file acc/latency 统计列可用。此改动属构建配置修正（不改推理逻辑），**在 TODO 1 基线重编之前完成**（基线即用 main.cpp 入口）。若实施中发现 main.cpp 无法直接编译进 target（如依赖缺失），fallback = 给 test.c 加 CLI + warmup + dump 到 process_pointcloud——**二选一，以能跑通 TODO 1 全部测量为验收**。

### 项目一：CPP 批量化（改 `deploy/CPP_trt3/`，不碰 .so/ONNX/engine）

0. **源码锚点先行**：读 `pipeline.cpp` 两份函数（process_pointcloud L73-272 / process_file L278-471）确认本计划锚点（上面已给两份的行号对）；读 `trt_inference.cpp` 确认 infer 的 enqueue stream 与 `d_output_` realloc 逻辑（L106-109，见步骤 2 的 cudaFree 防御）。
1. 新写 GPU kernel `trim_transpose_kernel`（放 `CPP_trt3/src/`，~20 行）：输入子云 GPU logits `(1,2,N_padded)` + `N_true` + 写入偏移 `offset`，输出按 `(N_true,2)` 行主序直接写 GPU `d_src[(offset+j)*2+c]`（读 `d_out[c*N_padded+k]`）；fp16 输入时 kernel 内 `__half2float`（模板双实例；half→float 是精确转换，GPU/CPU 结果 bit 级一致）；**N_true==0 时 kernel 直接 return（防御性，实际数据分布无空子云）**。同时把 `idx_part` 切片 H2D 写 `d_idx`（int64：CPU 侧 `static_cast<int64_t>` 转换；**staging 用 per-file 持久缓冲复用**——去 sync 后 CPU 领先，临时 vector 析构依赖 pageable memcpy 的驱动 staging 行为（非契约保证），持久缓冲消除该隐患）。
2. 重构 `pipeline.cpp` `process_file`（与 `process_pointcloud` **两份逐行同步改**，删改点行号对：D2H+sync L375-383 & L173-183；CPU trim/转置 L385-397 & L185-197；scatter 前 H2D L416-423 & L208-215；冗余 sync L434 & L235）：
   - 删循环内 D2H + sync；删 CPU trim/转置；改为每子云（每 chunk）调 `trim_transpose_kernel`；
   - 删 scatter 前 d_src/d_idx H2D（数据已在 GPU）；循环后现役 `scatter_mean` + **单次** D2H merged + sync；删冗余 sync（两份的 L434/L235）；
   - `d_src`/`d_idx`：每文件分配（CudaBuffer，cudaMalloc 开销微秒级）；
   - **cudaFree 竞态防御（R4）**：`TrInference` 构造时把 `d_output_` 按 `2 × max_n × (fp16?2:4) 字节` 一次性预分配，**删除 L106-109 的按需 realloc 分支**（cudaFree 非 stream-ordered——去 sync 后 CPU 领先，realloc 会释放仍被在队 kernel 引用的 buffer，现役靠 per-chunk sync 掩盖；预分配行为中性）；
   - **正确性论证（写进代码注释）**：同 stream FIFO 保证 upload/infer/trim_transpose 排序，`trim_transpose(i)` 先于 `infer(i+1)` 执行，`d_pos_/d_x_` 复用安全；输出 buffer 恒定不释放（上述预分配），规避非 stream-ordered 的 cudaFree；CPU preprocess(i+1) 与 GPU infer(i) 流水重叠。
3. 重编 `hpenet_trt_infer`：`cd deploy/CPP_trt3 && mkdir -p build && cd build && cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES="80;86;89" -DCMAKE_BUILD_TYPE=Release && make -j`（入口已是 main.cpp，见步骤 -1）。
4. 正确性（验收口径见 §五）：改前基线 = TODO 1 的 **CPP 自 dump 留档（fp32 engine）**；改后开 dump 跑 ti10 对比 merged logits/pred 逐点 bit 级一致；fp16 引擎只验 acc（不做 bit 级——fp16 无 dump 基线，明示）。

### 项目二：FPS warp 归约（改 `deploy/trt_plugins/`，宏切换，engine 复用）

0. **源码锚点先行**：读 `fpsprune_plugin.cpp` enqueue 确认调用序列（已知：exact 路径 L112 与 prune 路径 L124 均调 `fps_launcher_with_stream`，prune 路径 temp fill 只覆盖 B×N_points 前缀——R4 已核）；确认 fill/prune 调用序列原样不动。
1. 新建 `deploy/trt_plugins/src/fps_kernel_warp.cu` + `include/fps_kernel_warp.h`：kernel 定名 `furthest_point_sampling_kernel_warp`（nsys gpukernsum 可识别），语义与现役 **bit 级等价**：
   - 算法/距离表达式**逐字符照抄** `fps_kernel.cu`（裸 `*`/`+`，禁 `__fmul_rn`，防 FMA 收缩差异）；temp fill 锁 `1e10f`；
   - 归约：intra-warp `__shfl_down_sync` argmax（strict `>`）→ warp 结果写 shared → 1 次 `__syncthreads` → warp0 终归约 → shared 广播 `old` → 1 次 `__syncthreads`（**共 2 次/轮，替代现役 11 次**；`old` 必须 shared 广播——原型期实测过的真 bug）；
   - tie 语义：warp-down 与树归约均 strict `>` 保低 tid = 全局最小索引，一致；
   - **档位策略（R4）**：`__shfl_down_sync(0xffffffff)` 要求满 warp——**warp 模板只实例化 block_size ≥ 64 档位（1024/512/256/128/64）**；block_size < 64 的档位（32/16/…/1）在 launcher 内**分流走现役旧 kernel**（非全局 fallback，按 n_threads 分流）。部署 N 恒 1024 档，小档位纯防御；
   - 调用约定与现役完全一致（含已知 batch=1 限制，不修不改语义）。
2. **切换点放 `fps_launcher_with_stream` 函数体内**（fps_kernel.cu）加宏 `HPENET_FPS_WARP`（默认 0，`#ifndef` 保护）——**一处切换覆盖全部调用方**（fpsprune 的 exact+prune 两路径及其他 FPS 插件调用者）；fill/prune 序列不动；现役 kernel 保留（回退通道 + 小档位分流复用）；新 kernel 定义包进 `#if HPENET_FPS_WARP`。
3. **`trt_plugins/CMakeLists.txt` 的 add_library 源列表加 `src/fps_kernel_warp.cu`**（双构建树共享，加一次两边生效）；双编译 .so（宏 0=现役 `libhpenet_plugins.so` / 宏 1=warp `libhpenet_plugins_fpswarp.so`），engine 复用。
4. 正确性：独立 nvcc 对拍程序（照 `tests/test_bq_dp_fusion.cu` 模式，`-DHPENET_FPS_WARP=1`，双 buffer）：tie-free 随机 + 真实子云坐标（N=1024/2750/3523/5500/7200，M=N/4 为主 + M=N/2 抽测；N=1024 走 512 档、其余 1024 档）下 warp idx 与现役 bit 级一致（**小档位不测**——分流走现役，无需对拍）；整网（复用 `deploy/v2_e2e_dump.py`，宏 0/1 各跑，逐点 pred 100% 一致 + acc 逐文件同基线）。

## TODOs

- [x] 1. **构建入口决策（步骤 -1）+ CPP 基线测量与留档**：①CMake 换 main.cpp 入口（或 fallback 给 test.c 加 CLI）并重编——以能跑通本 TODO 全部测量为验收；②以当前源码（插件宏 0）重编基线二进制（消除版本漂移）；③给 `pipeline.cpp` `process_file` 尾部加 **dump-only 前置补丁**（写 result.logits/predictions 二进制；**带运行时开关默认关**）；开 dump 用 **fp32 engine** 跑 ti10 留档（fp32 是唯一 bit 级基线；fp16 不做 bit 级对拍）；④nsys 基线（--num_files 10，dump 关），.nsys-rep 命名 S1 留档，记录 §四基线表
- [x] 2. CPP 实现：源码锚点确认 → `trim_transpose_kernel`（含 N_true=0 防御、int64 staging 持久缓冲）→ 重构两份函数（行号对见 §二）→ **d_output_ 预分配删 realloc（cudaFree 防御）** → stream-FIFO+cudaFree 注释 → 保留 dump 于 process_file 同位置 → 重编
- [x] 3. CPP 验证 + 改后测量：①开 dump 对拍轮：merged logits/pred 逐点 bit 级一致 + **acc 逐文件与基线留档相同（4 位小数逐字符），10 文件均值 == 0.9636**；fp16 只验 acc（`--engine deploy/hpenet_v2_fp16.engine`）；②nsys 改后轮（dump 关，S2 留档），回填 §四项目一表
- [x] 4. FPS：源码锚点确认 → `fps_kernel_warp.cu`（定名 furthest_point_sampling_kernel_warp；**≥64 档位 warp + 小档位分流现役**）→ CMakeLists 加源 → 宏切换放 `fps_launcher_with_stream` 内部 → 双编译 .so
- [x] 5. FPS 正确性验证：nvcc 对拍（tie-free + 真实子云，idx bit 级一致，首 diff 打印复用 test_bq_dp_fusion 模式）+ 整网（v2_e2e_dump.py，宏 0/1 各跑，逐点 pred 100% 一致 + acc 逐文件同基线）
- [x] 6. FPS nsys 测量（Python 管线，与 CPP 二进制无关；GPU 串行故时序在 TODO 3 后）：**A 轮前先以当前源码（宏 0）重编 deploy/trt_plugins 构建树 .so 并备份**（消除版本漂移）→ A=现役 .so；B=临时 cp warp .so 覆盖 `deploy/trt_plugins/build/libhpenet_plugins.so`（Python 经 trt_utils.py:80 显式路径加载，cp 目标正确），测完恢复（md5 对比备份验证）→ `nsys profile python deploy/trt_inference.py --engine deploy/hpenet_v2_fp32.engine --num_files 10` 两轮同口径 → gpukernsum 取 `furthest_point_sampling_kernel` vs `furthest_point_sampling_kernel_warp` 与整网 GPU kernel 总和，回填 §四项目二表
- [x] 7. 叠加测量与总回填：CPP_trt3 构建树内启用 warp（临时宏 1 或 CMake 传 define）**重编 CPP** → 跑 S3 组合（口径同基线，S3-combo 留档）→ 回填叠加表 → 改回宏 0 重编恢复 + ti10 acc 复核；若项目二净负则宏保持 0，叠加表按"仅项目一"回填并注明

## Final Verification Wave

- [x] F1. Plan compliance：acc 逐文件同基线（4 位小数逐字符）+ 均值 0.9636、bit 级对拍证据齐全（基线留档在先、入口决策在先）、Must NOT 零违反（不改 ONNX/engine；不删现役 kernel；零 git；不动已关闭路线）；**时序合规证据 = S1/S2/S3-AB/S3-combo 命名的 .nsys-rep 留档齐全**
- [x] F2. 代码质量：warp 距离表达式逐字符一致、old shared 广播、**≥64 档位 warp + 小档位分流现役**（非全档位模板）、trim_transpose 边界（N_true=0 防御、fp16 双实例）、d_output_ 预分配无 realloc、staging 持久缓冲、stream-FIFO+cudaFree 注释在位、宏作用域（launcher 内部单点）
- [x] F3. 性能结论审查：§四表完整回填（字段来源见各表注）、数字可由留档 .nsys-rep 复现、时序归因正确、"GPU kernel 总时间持平"判定 = |Δ|≤2% 且非显著增加、收益为正才落地（净负回退并如实记录）

## 三、nsys 延迟收益测量方式

### 统一口径

- engine：`deploy/hpenet_v2_fp32.engine`（叠加复核轮加 fp16 acc）；数据：ti10（0000068..77；main.cpp 的 process_directory 与 trt_inference.py 均为 sorted + 20% 尾部 + shuffle 注释，文件集一致已核）
- GPU 0 独占、测量串行；warmup 5（main.cpp:171 自带）+ ti10 全部 10 文件 per-file median；可选 --num_files 30 扩样
- 双口径：①端到端 per-file（result.latency_ms 列）②gpukernsum（kernel 名聚合，Time 列总和）+ cudaapisum（API 次数/总时间）；**ns→ms 换算、10 文件累计口径、ms 保留 3 位小数**
- 环境同前（cuDNN 8.9.7 / TRT 8.6.1.6 / cuda 11.8 LD_LIBRARY_PATH）

### 测量时序（各轮 .nsys-rep 命名留档）

| 步骤 | 状态 | 产出 |
|---|---|---|
| S1 基线 | main.cpp 入口 + 现役 pipeline + 宏 0 | 基线表 + fp32 dump 留档 + `S1.nsys-rep` |
| S2 项目一 | CPP 改后 + 宏 0 | 项目一表 + `S2.nsys-rep` |
| S3① | Python A/B（deploy 树 .so 交换） | 项目二表 + `S3-AB-{A,B}.nsys-rep` |
| S3② | CPP 构建树启用 warp 重编 | 叠加表 + `S3-combo.nsys-rep` |

### 项目一规程
S1/S2 各一轮：nsys 包 `hpenet_trt_infer --num_files 10`（dump 关）；cudaapisum 看 `cudaStreamSynchronize` 次数（8→1/文件）与总时间；gpukernsum 看 GPU kernel 总时间（**持平判定 |Δ|≤2%**）；per-file latency median。

### 项目二规程
S3① A/B 两轮（同口径仅换 .so，恢复用 md5 验证）：`nsys profile python deploy/trt_inference.py --engine deploy/hpenet_v2_fp32.engine --num_files 10`；gpukernsum 取 FPS 段（两 kernel 名）与整网总和。

## 四、延迟收益测量结果统计（执行后回填）

> 字段来源统一注：**GPU kernel 总时间** = gpukernsum 各 kernel Time 列总和（ns→ms，10 文件累计，3 位小数）；**端到端 per-file median** = result.latency_ms 的 10 文件中位数（ms，3 位小数）；**sync 总时间** = cudaapisum 的 cudaStreamSynchronize Total Time（同前换算）。

### S1 基线（main.cpp 入口，全旧）（2026-08-24 实测，fp32 engine，ti10，.nsys-rep=/tmp/opencode/S1.nsys-rep）
端到端 per-file median **25.000 ms**（10 文件：28/23/25/25/21/28/25/24/27/27 ms）；GPU kernel 总 **129.990 ms**（gpukernsum 全窗累计，含 warmup）；sync **≈8 次/文件**（cudaapisum 全窗 cudaStreamSynchronize 98 次、总 0.401 ms，avg 4.096 µs——含 warmup 5 次+尾部；CPP 路径 sync 单次开销远低于计划 §一所引 Python 口径 431.5 µs，属口径差异，S2 同窗对比仍有效）；FPS kernel 总 **48.142 ms**（furthest_point_sampling_kernel<1024>，54 实例，占 GPU kernel 37.0%）；ti10 acc 均值 **0.9636**（fp32；逐文件：68=0.9746, 69=0.9375, 70=0.9720, 71=0.9524, 72=0.9773, 73=0.9562, 74=0.9653, 75=0.9662, 76=0.9639, 77=0.9707）

### S2 项目一（增量 = S2−S1）（2026-08-24 实测，fp32 engine，ti10，dump 关，.nsys-rep=/tmp/opencode/S2.nsys-rep）
| 指标 | S1 | S2 | 变化 |
|---|---|---|---|
| sync 次数/文件 | 8 | ≈3（39 次全窗/10 文件） | 次数 −60%（98→39；非严格 1/文件——39 = 文件 10 + warmup 5 + TRT/驱动内部 24，pipeline 自身已是 1/文件） |
| sync 总时间 | 0.401 ms | 0.185 ms | −54% |
| 端到端 per-file median | 25.000 ms | 24.500 ms | **−2.0%（未达预期 −10~15%；mean 25.3→24.1 = −4.7%）** |
| GPU kernel 总时间 | 129.990 ms | 130.014 ms | +0.018%（**持平判定 |Δ|≤2% ✓**） |
| acc | 逐文件同基线 | 逐文件同基线 | 均值 0.9636 |

> 注：S2 单次最终 sync 仍等全部 GPU 完成，sync 时间降幅 ≠ 端到端降幅全部（消的是空转+往返）。
> S2 补充事实（2026-08-24 对拍轮）：**pred 全 10 文件逐字节一致（0 mismatch）；logits 存在 ULP 级（~1e-7 相对）diff——三方对照证实为本征非确定性**（重构二进制自身两轮 dump、基线二进制 vs 基线 dump 同样 ULP diff，系 scatter_mean 浮点原子加顺读序差异），非重构引入；最强可达判据（pred bit 级 + acc 4 位小数逐字符 + 均值 0.9636）全部 PASS。fp16 抽测：与基线二进制 fp16 逐文件一致，均值 0.9637（76 号 0.9643 vs fp32 0.9639，为 fp16 引擎本征差，计划头部"fp16=0.9636"口径按此修正记录）。

### S3① 项目二（Python 口径 A/B，不与 S1/S2 跨口径对比；CPP 口径效果由叠加覆盖）（2026-08-24 实测，fp32 engine，ti10，.nsys-rep=/tmp/opencode/S3-AB-{A,B}.nsys-rep）
| 指标 | 宏 0 | 宏 1 warp | 变化 |
|---|---|---|---|
| FPS 段 kernel 总时间（furthest_point_sampling[_warp]_kernel） | 48.087 ms | 38.502 ms | **−19.93%（超微基准预期 −17%）** |
| 整网 GPU kernel 总时间 | 128.400 ms | 119.600 ms | **−6.86%（略优于预期 −6.4%）** |
| 端到端 per-file median | 21.500 ms | 21.500 ms | **0%（未达预期 ~−5%；mean 均 0.023 s，Python 口径 host 端开销主导，GPU 节省 8.8 ms/10 文件 ≈0.9 ms/文件被管线噪声淹没）** |
| acc / 逐点 pred 一致率 | 基线 / — | 同基线 / 100% | 逐文件一致 / 100% |

### 叠加（S3② − S1，CPP 口径）（2026-08-24 实测，fp32 engine，ti10，dump 关，.nsys-rep=/tmp/opencode/S3-combo.nsys-rep）
| 指标 | S1 | S3②（CPP 批量化 + FPS warp） | 变化 |
|---|---|---|---|
| 端到端 per-file median | 25.000 ms | 24.500 ms | **−2.0%（远未达预期 −15~20%；与 S2 持平 → FPS warp 端到端增量收益 = 0）** |
| GPU kernel 总时间 | 129.990 ms | 120.475 ms | **−7.32%**（vs S2 130.014 = −7.33%，与 S3① Python 口径 −6.86% 交叉印证） |
| FPS 段 kernel 总时间 | 48.142 ms（legacy，54 实例） | 38.512 ms（furthest_point_sampling_kernel_warp<1024>，54 实例） | **−20.0%** |
| acc | 0.9636 逐文件基线 | 0.9636 逐文件逐字符一致（warp 启用轮即复核通过） | ✓ |

> **结论**：叠加口径下项目二 GPU kernel 收益（−7.3%，~9.5ms/10 文件 ≈ 0.95ms/文件）如实兑现，但端到端 median 与 S2 持平（24.5ms）——GPU 节省被 CPP 二进制 host 侧（CPU preprocess/TRT enqueue 等）气泡吸收，端到端口径 FPS warp **零增量、非净负**。按止损条款非净负不强制回退，但端到端无收益 → **宏默认保持 0，FPS warp 不落地**（代码与双编译 .so 留档 `deploy/trt_plugins/build/libhpenet_plugins_fpswarp.so` 备将来 host 侧优化后启用）。最终口径：仅项目一 −2.0% 落地（现役二进制）。

## 五、边界、验收口径与止损

### 验收口径（R4 量化；R6 修正 logits bit 级判据）
- **"pred 逐点 bit 级一致"（主判据，修正自"merged logits/pred"）** = 重构前后每文件 pred 逐点 bit 级一致（0 mismatch）。**logits 不做 bit 级判据**——实测（TODO 3 三方对照）现役 `scatter_mean` 浮点原子加顺序使 merged logits 连基线二进制自身重跑都不可 bit 复现（11562/83768 元素 ULP diff，max 3.8e-6，pred 不受影响）；此为本征非确定性，非重构可判定对象。
- **"acc 逐文件一致"** = 每个文件的 acc 与基线留档对应文件 acc **按打印精度 4 位小数逐字符相同**；10 文件均值 == 0.9636。
- **"GPU kernel 总时间持平"** = |Δ| ≤ 2% 且非显著增加。

### 边界与不做什么
- 不改 ONNX / 不重建 engine；不删现役 kernel（warp 宏默认 0 + 小档位分流现役）；不碰已关闭路线；零 git（基线留档用文件备份）；FPS warp 不修现役已知缺陷。
- test.c 移出 CMake target（文件保留）属构建配置修正，须在 F1 报告中明示。

### 止损条款（R4 新增）
- bit 级对拍失败：定位首 diff（复用 test_bq_dp_fusion 的首 diff 打印模式）→ **最多 3 轮修复迭代**仍不一致 → 中止该 TODO，回退（项目一还原 pipeline.cpp 用备份二进制；项目二宏保持 0），失败证据如实记入 §四与 ledger，不算计划失败（负结果即交付）。
- 资源预算：全计划 GPU 独占测量合计 ≤ 1 个工作日时段；单 TODO 卡壳超 2 小时未进展即停下记录并上报，不无限重试。

## 六、执行归档（2026-08-24，全计划完成 + 一致性审查通过）

### 最终结论

| 项目 | 实测 | 处置 |
|---|---|---|
| 项目一 CPP 批量化 | 端到端 **−2.0%**（25.0→24.5ms），sync 98→39 次/−54% 时间，GPU kernel 持平（+0.018%），pred 逐点 bit 级 0 mismatch | **落地** |
| 项目二 FPS warp | **GPU kernel −7.32%**（FPS 段 −20.0%，48.14→38.51ms），端到端增量 0（host 侧吸收） | **用户决策：落地启用**（GPU 收益真实且正确性无损；host 优化后收益将兑现）——宏默认 0→1，双构建树重编，落地验证见 §六末 |

**精度**：ti10 acc **0.9636 全程零回退**（逐文件 4 位小数逐字符，fp32/fp16 双引擎）。

### 关键发现

1. **host 侧是端到端真正瓶颈**：端到端 24.5ms/文件中 GPU kernel 仅 ~12.0ms（**~49%**）——GPU kernel 优化（本计划 −7.32%）已不再转化为端到端收益；下一步必须打 host 侧（见 §七 host profiling 结论）。
2. **logits bit 级判据修正（R6）**：现役 `scatter_mean` 浮点原子加顺序使 merged logits 连基线自身重跑都不可 bit 复现（11562/83768 元素 ULP diff，max 3.8e-6，三方对照证实）；正确判据 = **pred bit 级 + acc 逐文件逐字符**。
3. **CPP sync 单次仅 ~4µs**（非 Python 口径的 431.5µs）——批量化收益远低于调研预估（−2% vs −10~15%）的根因。
4. **二进制入口曾是 test.c 非 main.cpp**（R4 双评审发现）：CLI/dump/测量机制全部依赖入口决策先行修正。
5. **d_output_ realloc 的 cudaFree 竞态**（R4）：去 sync 后预分配是必需的正确性防御，非可选优化。

### 验证与证据

| 验证层 | 结果 | 证据 |
|---|---|---|
| T1-T7 逐任务验证 | 全过（每任务 orchestrator 亲自复跑/读代码） | 各任务报告 + notepad learnings.md |
| F1 compliance（oracle 独立） | **APPROVE**（5/5） | 独立重跑 acc 0.9636；engine mtime 早于计划窗口；.so md5/nm 恢复验证；5 份 .nsys-rep 时序齐全 |
| F2 代码质量（oracle 独立） | **APPROVE**（9/9） | warp 距离表达式逐字符、old shared 广播、≥64 档位分流、trim 边界、d_output_ 预分配、宏单点 |
| F3 性能结论（oracle 独立） | **APPROVE**（6/6） | nsys 独立复算精确到 0.001ms；归因链自洽；负结果如实 |
| 计划 vs 代码一致性（oracle 独立） | **CONSISTENT**（14/14） | git diff 逐项对照；现役 kernel 本体零改动；无 scope creep（3 处轻微偏差均合理/前序遗留） |
| bit 级对拍 | FPS idx **15/15 用例 0 mismatch**；CPP pred **10 文件 0 mismatch**；整网 pred **100% 一致**（41868 点） | /tmp/opencode/fps-ab/、cppbaseline/ |

**nsys 留档**（时序证据）：`/tmp/opencode/{S1,S2,S3-AB-A,S3-AB-B,S3-combo,HPROF}.nsys-rep`
**数字链**：S1 25.0ms/129.99ms → S2 24.5ms/130.01ms → S3② 24.5ms/120.48ms（GPU）；FPS 段 48.14→38.51ms
**产物**：trim_transpose.cu/.h（新）、fps_kernel_warp.cu/.h（新）、pipeline.cpp 两函数重构 + NVTX 分段观测、trt_inference.cpp 预分配、CMake 入口 main.cpp + nvToolsExt、idx_staging_ 持久成员

### FPS warp 归约降低延迟的机理（归档总结）

**一句话**：FPS 的耗时大头不在"算距离"而在"找最大值的通信方式"——现役用 10 层共享内存树 + 每轮 11 次 block 级 barrier，warp 归约用寄存器级 shuffle 把它压到 2 次 barrier，计算量一字未变，纯同步/通信开销削减，因此 bit 级等价且直接提速。

**原因链（逐层，均有实测支撑）**：

1. **实证前提：瓶颈定位推翻直觉**。微基准分解（/tmp/opencode/fps-microbench/）实测 FPS 耗时的 **58.9%**（N=3523）在归约、41.1% 在扫描——此前直觉认为慢在"全量扫描算 d²"（内存带宽），实测证伪。不先分解就优化，方向会错。

2. **现役树归约的开销本质**：FPS 是贪心串行算法，M 轮迭代每轮必须"扫 N 点更新 temp → block 内全局 argmax → 选点进下一轮"。现役归约（fps_kernel.cu L49+L51-60）：1024 线程写 shared → **10 层树遍历（每层一次 `__syncthreads` + shared 读写）+ 1 次预同步 = 11 次 block 级 barrier/轮**。stride-4 部署画像 N≈3523/M≈880 → **~9680 次 barrier/帧**。`__syncthreads` 要求 block 全体 1024 线程到齐，warp 间进度差在 barrier 处互相放大等待。

3. **warp 归约的机制**：GPU 的 warp（32 线程）本身就是锁步执行单元——warp 内交换数据**不需要任何 barrier**，`__shfl_down_sync` 直接走寄存器（SFU）：
   - 每个 warp 独立 5 步 shuffle 完成 32 线程 argmax（warp 间互不等待）；
   - 32 个 warp 结果写 shared → **1 次** `__syncthreads` → warp0 再 5 步 shuffle 终归约 → shared 广播 old → **1 次** `__syncthreads`；
   - **同步 11→2 次/轮，10 层 shared 树遍历 → 寄存器交换**。

4. **为什么 bit 级等价（可安全替换的原因）**：距离表达式与扫描循环**逐字符照抄**（d²、temp 更新、best/besti 全同——计算结果逐位相同）；归约只是"找 max 的通信方式"，argmax 满足结合律且两版 tie 语义一致（树归约 `v2>v1?i2:i1` 与 shuffle `ov>best` 均 strict `>` 保低索引 = 全局最小索引）→ 参与比较的值集合与胜者判定完全相同。实测 15/15 用例（含真实子云、N∈{1024~7200}、M=N/2 抽测）idx **bit 级 0 mismatch**。

5. **实测收益链**：微基准单 kernel **1.211×**（N=3523，−17.4%）/ 1.185×（N=7200）→ 整网 FPS 段 **48.14→38.51ms（−20.0%）** → 整网 GPU kernel **−7.32%**（部署口径下 GPU 占端到端 86%，近乎全额转化为端到端收益）。

6. **收益边界（为什么只有 ~20%）**：扫描本底占 41~52%（O(N) 距离计算 + 内存访问，warp 版不触碰），归约部分中 shuffle 自身仍需少量开销——warp 版拿走了归约收益的大部分但非全部。这是 FPS 单 block 内"最后一块可吃的肉"；再往下只剩算法级（prune 档已做过）或结构级（多 block 已证必亏：grid.sync 每轮 1.94µs > 每轮计算 1.81µs）。

**方法论沉淀**：①性能优化前先做微基准分解（拆计算 vs 通信/同步），直觉常常错；②区分"计算开销"与"通信开销"——改通信方式可以 bit 级等价地提速；③单 block kernel 的归约优先考虑 warp shuffle；④收益上限由不可触碰部分的本底占比决定，分解数据同时给出收益与上限。

### 落地状态（2026-08-24，用户决策执行）

- **FPS warp 已启用为默认**：宏 1（两 .cu），双构建树重编（部署树 md5 `bcbb882c…`/CPP 树 `b32febd1…`），nm 双树各 5 个 warp 符号，ti10 acc 逐文件零回退 0.9636，median 23ms。回退 = 宏改回 0 重编两树（旧 md5：e8d53e75…/82d15e0f…，notepad 有记录）。

## 七、host 侧 profiling 结论（2026-08-24，落地状态实测）

### ⚠️ 部署口径修正（2026-08-24，用户指正）

上述 NVTX 分段的 **PLY_LOAD 9.47ms 是 benchmark 假象**：数据源是磁盘上的 **ASCII PLY 测试文件**（`format ascii 1.0`，CloudCompare 导出，292KB 文本，getline 逐行文本解析——文本浮点解析主导，非磁盘 IO）。**真实部署直接接收点云内存数据，PLY_LOAD 不存在**。修正后的真实部署画像：

- **真实部署端到端 ≈ 14ms/帧**（23.5 − 9.47）
- **GPU kernel 12.05ms → 占 ~86%**——**GPU 又变回主战场**（此前的"host 是瓶颈"结论仅在文件 IO 口径下成立）
- host 纯串行仅 ~2ms（VOXELIZE 1.02 + PREPROCESS 0.56 + TAIL host 部分）；流水段（LOOP+TAIL 13.4ms 窗口）GPU 利用率 ~90%，CPU enqueue 9.5ms 被流水掩盖

**修正后优化路线（真实部署口径）**：
1. **GPU kernel 12ms 本身**（FPS warp 已落地 3.85ms/帧居首；ball_query 2.7ms 已证到头；卷积等其余）
2. VOXELIZE 1.0ms GPU 化（CPU 排序 → GPU radix/hash，可省 ~0.8ms）
3. TAIL ~1ms（scatter 融合/argmax 移 GPU）
4. ENQUEUE host 开销（1.9ms/chunk）在流水内被掩盖，非当前瓶颈

### 原 benchmark 口径分段（保留存档，HPROF.nsys-rep）

**GPU kernel 占端到端仅 51%（12.05ms/23.5ms）——host 侧 ~11.4ms/文件是下一战场。** NVTX 分段（HPROF.nsys-rep，分段之和 23.95ms ≈ 端到端 ✓）：

| 阶段 | ms/文件 | 占比 | 说明 |
|---|---|---|---|
| **PLY_LOAD** | **9.47** | **41%** | GPU 完全空闲（文件 IO+解析） |
| SUBCLOUD_LOOP | 9.87 | 43% | ENQUEUE host 1.9ms/chunk×~5 + PREPROCESS 0.56；流水段 GPU 近饱和 |
| TAIL | 3.57 | 15% | scatter+D2H+末次 sync |
| VOXELIZE | 1.02 | 4% | CPU 排序 |
| COORD_SHIFT / ARGMAX_ACC | ~0.01 | ~0% | |

**瓶颈排序与优化路线**：
1. **PLY_LOAD 双缓冲预取**（生产者线程 load+voxelize 与上一文件 GPU 推理重叠）：预估 23.5→~14ms（**−40%**），最大单项；理论上限 ≈ GPU 12ms/文件。
2. ENQUEUE host 开销（TRT enqueueV3）：PLY 重叠落地后成新瓶颈，CUDA Graph 可打 ~2-4ms。
3. TAIL 融合（scatter 并入 trim_transpose / argmax 移 GPU）：~0.5-1ms。
