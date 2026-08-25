# gridballquery-trt-plugin - Work Plan

## TL;DR (For humans)

- **What you'll get**：GridBallQuery（体素哈希网格球查询）被改造为 deploy 管线的新可选档 `--bq_algo gridballquery`——两个新 TRT 插件 `GridBallQueryGroup`/`GridBallQueryDP`（ONNX 签名与现役完全一致，hash 构建内置插件，ONNX 节点数不增），加一份 nsys 实测的 grid vs 现役暴力 ball query 性能对比报告（FP16+FP32）。
- **Why this approach**：FPS 仓库原版不能照搬（hash 建在 query 点且每 voxel 单 query、依赖不存在的 HAV 前置算子、atomicAdd 非确定），故重设计为 source-hash + 确定性稳定计数排序 + "半径内（严格 <）索引最小 nsample 个"选择——后者使 grid 档输出有望与现役 bit 级一致。
- **What it will NOT do**：不动现役插件行为（默认档导出 bit 不变）、不新增 ONNX 节点（628 基线 ±0）、不做任何 git 操作、不重训、不接受照搬 FPS 原文。
- **Effort**：7 个实现任务（kernel → 插件/导出并行 → 单元对拍 → 整网 → nsys → 文档）+ 4 个终验。
- **Risk**：①性能前提未验证——O(N²) 稳定排序与现役暴力 O(M·N) 同量级，grid 可能不快反慢（nsys 分桶报告如实量化，负结论合法交付）；②整网精度受邻居集合差异影响，以绝对 mIoU 差 ≤0.5 个百分点作门槛；③现役空球 cnt==0 输出未初始化内存系既有 bug，仅记录不修（零 diff 守护）。
- **Decisions（已确认）**：新增可选档（现役保留回退）/ 容差+mIoU 验收 / nsys 系统已装 / FP16+FP32 双测。

## Scope

将 FPS 仓库（`FPS/pointcloud-3d-detector-tensorrt/`）的 GridBallQuery 算法思想改造为 HPENet V2 deploy 管线可用的 TensorRT 自定义插件，作为**新增可选档**（不动现役路径），并用 nsys 对比性能。

**In scope**：
1. 新建 `GridBallQueryGroup` / `GridBallQueryDP` 两个 TRT 插件（ONNX 签名与现役完全一致，voxel hash 构建内置 enqueue，不新增 ONNX 节点）
2. 导出侧接线：`onnx_backend.py` + `onnx_export.py` 新增 `--bq_algo {ballquery,gridballquery}`
3. 正确性验证：逐调用点对拍 + 整网对比 + radar 测试集 mIoU（绝对差 ≤0.5 个百分点）
4. 性能验证：nsys 对比 grid vs 现役暴力 ball query（FP16+FP32 双精度）
5. 文档：plugin.md 新章节 + AGENTS.md 增补

**Out of scope / Must-NOT-Have**：
- ❌ 不修改现役 `ballquery_kernel.cu` / `ballquerygroup_kernel.cu` / `ballquery_plugin.cpp` / `ballquerygroup_plugin.cpp` 的任何行为（回退通道）
- ❌ 不新增任何 ONNX 图节点（节点总数维持 628 基线；grid 档只是把 8 个节点的 op_type 换名）
- ❌ 不执行任何 git 操作（AGENTS.md 约束，含 deploy 子模块）
- ❌ 不改训练配置 / 不重训 / 不动 `cfgs/`
- ❌ 不照搬 FPS 仓库 kernel 原文（其语义与 HPENet 不兼容，见 Background）

## Verification strategy

- **单元级**：静态 shape 单插件 engine（复用 `tests_fps_algos.py` 的 ~0.1s build 模式）对拍 grid vs 现役 kernel，N∈{1024,3500,6000,10000}，fp32/fp16，含 7 类对抗用例（同 voxel 多质心/重复点/空球/精确球面点/N<nsample/N=1/大 radius 全覆盖）
- **整网级**：grid 档 ONNX 节点数检查（=628±0）→ FP32+FP16 engine → 与现役 engine 输出 max-abs-diff + radar 测试集绝对 mIoU 差 ≤0.5 个百分点
- **性能级**：nsys profile（系统已装 Nsight Systems），kernel 级耗时对比 + 整网 GPU 时间 + e2e 延迟，FP16/FP32 双精度
- 所有验证 agent 可执行，evidence 落盘 `deploy/evidence/`

## Execution strategy

**依赖矩阵**：

| 任务 | 依赖 | 可并行 |
|---|---|---|
| 1 kernel | — | 与 2 插件壳可并行开发（.h 接口先定） |
| 2 插件+CMake+注册 | 1 的头文件 | 与 3 导出接线并行 |
| 3 导出接线 (python) | 无（symbolic 只发节点，不依赖 .so） | 与 1/2 并行 |
| 4 单元对拍 | 1+2+3 | — |
| 5 整网验证 | 4 | — |
| 6 nsys 对比 | 5 | — |
| 7 文档 | 6 的数据 | — |

批间串行（1+2+3 一批内并行）→ 4 → 5 → 6 → 7。测试策略：tests-after（kernel/插件无法 TDD，先实现后用对拍矩阵锁行为）。

## Todos

- [x] 1. 实现 grid 邻居搜索 kernel（`deploy/trt_plugins/src/gridballquery_kernel.cu` + `include/gridballquery_kernel.h`）
  - **设计规格（零判断空间）**：
    - 数据结构：**哈希桶表（bucketed count-sort，无探测）**，`T = next_pow2(2 * maxN_)`（**公式**，从 configurePlugin 缓存的 profile maxN 计算；maxN=10000 时实例值为 32768——单元对拍静态 shape engine 的 maxN=1024/3500/6000/10000 各自成立，用户 `--max_n` 改建 engine 亦自动适配；**禁止字面写死 32768**；`getWorkspaceSize` 看不到运行时 dims，动态 profile 下 build 期 `dims.d[1]` 可能为 -1 —— 现役 `ballquerygroup_plugin.cpp:77` 注释明证），`bucket = coord_hash_32(cell_x,cell_y,cell_z) & (T-1)`；voxel 尺寸 `v = voxel_size > 0 ? voxel_size : radius`，**构造函数内 clamp `v = fminf(v, radius)`**（防误配）；查询扫描范围**公式化** `R = (int)ceilf(radius / v) + 1`，扫 `[-R,R]³` 格（默认 v=radius → R=2 → 125 格；覆盖证明：roundf 每轴误差 ≤0.5 ⇒ `|c_p−c_q| ≤ |p−q|/v + 1 < radius/v + 1 ≤ ceilf(radius/v)+1 = R`，故扫 [-R,R]³ 充分；d2 判据的严格 `<` 吸收 radius/v 在 ceilf 边界的 fp32 舍入；clamp 使 R≥2 恒成立，公式单独亦覆盖 v>radius 情形——双保险）；enqueue 内用实际 N（≤ maxN）
    - 构建三步（全在 enqueue 内，workspace 分配）：
      - `build_count_kernel`：每 source 点算 cell（`roundf(coord/v)`，参照 `FPS/.../gridBallQueryPlugin.cu:209-211`）→ bucket → 计数
      - 前缀和：**寄存器驻留单 block 扫描 over T，输出排他（exclusive）前缀**（count-sort 桶起点需要 exclusive offsets）：1024 线程，每线程 `ceil(T/1024)` 个寄存器元素（T=32768→32、16384→16、8192→8、2048→2，**按 T 公式推导，不写死 32**）→ 线程内顺序前缀 → warp shuffle 归约 → 32 个 warp 部分和入 smem（128B，**总 smem <1KB，无 cudaFuncAttribute opt-in**）→ warp 间扫描写回；**禁止把 T 个元素整体装入 smem**——32768×4B=128KB 超静态 48KB 上限也超 sm_86/87/89 动态上限（~100KB 级），naive 写法编译即失败；不引入 thrust/cub 依赖）
      - `build_place_kernel`：**稳定放置**（同一 bucket 内按 source 索引升序；实现：每点 rank = 前驱同 bucket 计数，O(N²) 上限 N=10000；**⚠️ 性能前提风险**：该量级 ≈ 现役暴力 O(M·N)，grid 档可能不快反慢——已列为风险，nsys 任务按 N 分桶报告验证；备选方案若需提速：atomicAdd 捕获 + (bucket,idx) 对做确定性排序；禁止 atomicAdd 顺序依赖）
    - 查询 `grid_query_kernel`：每线程一个 query 点；dx,dy,dz ∈ [-R,R] 升序扫格 → 桶内点按升序索引遍历 → **每候选点重算其 cell 并验证 ∈ 目标格集**（不同 cell key 哈希碰撞到同 bucket 时必须消歧，否则跨格点混入遍历序破坏剪枝与一致性论断）→ `d2 < radius2`（**严格小于**；d2/radius2 表达式从现役 `ballquery_kernel.cu:25-26` **逐字复制普通运算符**（含 `radius2 = radius*radius`），**禁止 `__fmul_rn`/`__fadd_rn` intrinsic**——nvcc 默认 `--fmad=true` 很可能已将现役收缩为 FMA，单边禁收缩反而制造 ulp 级错位；同表达式 + 同 nvcc 参数 + 同一 .so ⇒ 构造上 bit 级一致，无需任何收缩假设）者为候选 → **取索引最小的 nsample 个**（局部最大堆/插排，容量为插件属性 nsample 的参数值，32 仅为当前实例）→ **写入前将选中候选按 source 索引升序插入排序，槽 0..cnt-1 按升序写入**（现役 k 升序扫描天然升序；max-heap 弹出序为降序，不排序则 bit 级一致诊断项必挂）→ 空槽填首个找到的邻居（= 最小索引候选，**逐字复刻现役 padding 规则** `ball_query_gpu.cu:41-44` / plugin.md §13.8 警告）
    - **空球（cnt==0）行为定义**：enqueue 开始先 `cudaMemsetAsync` 将 idx 清 0，再正常填充——grid 档对 cnt==0 的行输出全 0（**确定性**）；现役对 cnt==0 整行不写（stale memory + 下游 gather 越界读的潜在 bug，记入 plugin.md 风险表）；**单测对空球只验证 grid 自身确定性与不越界，不与现役对拍该行**
    - **语义关键**：现役 kernel 也是按 source 索引升序扫、严格 <、取前 nsample → grid 档取"半径内（严格 <）索引最小的 nsample 个"应与现役 **idx bit 级一致**（诊断性检查项，非验收门槛；对抗集须含"精确球面点"用例验证 < 边界）
    - launcher：`gridballquerygroup_launcher(...)` = memset + build_count + scan + place + grid_query（→idx_ws）+ **将 `bq_dp_kernel` / `bq_gather_kernel` 复制进本文件**（它们在 `ballquerygroup_kernel.cu:13-66` 的 anonymous namespace 内部链接，**跨编译单元不可链**，且现役文件零 diff 守护禁止改其可见性——复制时注明来源行号与复刻义务）；`gridballquerydp_launcher` 同理（dp+idx 输出）
    - 交付物含 `deploy/trt_plugins/include/hash.h`：从 `FPS/pointcloud-3d-detector-tensorrt/plugins/src/common/hash.h`（**完整路径**，:16 `kEmpty` / :51 `coord_hash_32`）复制（Apache-2.0，保留版权头）
    - workspace 布局（fp32 计，**各段起始 256B 对齐**）：`counts+prefix uint32[T]*B + start uint32[T+1]*B + sorted int32[N_max]*B + idx_ws int32[B_max*M_max*S]`，全部按 profile 极值（maxB/maxN/maxM）定尺寸，`configurePlugin` 缓存极值（参照 `ballquerygroup_plugin.cpp:146`；B*M*S 索引用 size_t）
  - References: `deploy/trt_plugins/src/ballquerygroup_kernel.cu`（整体结构/dp/gather 复用）、`deploy/trt_plugins/src/ballquery_kernel.cu`（padding 语义）、`FPS/pointcloud-3d-detector-tensorrt/plugins/src/gridBallQueryPlugin/gridBallQueryPlugin.cu:186-273`（hash/格扫描思路，**不照搬方向与原子散射**）、`FPS/.../common/hash.h:51`（coord_hash_32）
  - Acceptance: `nvcc -c` 编译零 error 零 warning（`-Xcompiler -Wall`）；头文件接口与 launcher 签名如上锁定
  - QA happy: 单元对拍脚本（任务 4）全绿后回填 evidence；QA failure: N=1 / 空球（全 padding=idx[0] 规则触发）用例在任务 4 覆盖
  - Commit: 不提交（项目禁止 git 操作，改动留工作区）

- [x] 2. 插件类 + 注册 + 构建（`deploy/trt_plugins/src/gridballquerygroup_plugin.cpp` + `include/gridballquerygroup_plugin.h`）
  - 单文件实现两个插件类（`GridBallQueryGroupPlugin` / `GridBallQueryDPPlugin`，模式照抄 `ballquerygroup_plugin.cpp` 单文件双类），IPluginV2DynamicExt + **协变返回**（沿 Covariant return fix 的 17 文件规范：`supportsFormatCombination`/`getOutputDimensions` 等 override 声明协变类型）
  - ONNX 签名与现役完全一致：`GridBallQueryGroup(xyz[B,N,3], new_xyz[B,M,3], features[B,C,N]) → (grouped[B,C,M,S], dp[B,3,M,S])`；`GridBallQueryDP(xyz, new_xyz) → (dp, idx[B,M,S])`；plugin 名分别为 `"GridBallQueryGroup"` / `"GridBallQueryDP"`（版本 "1"）
  - attributes: `radius`(f32), `nsample`(i32), `normalize_dp`(i32), `voxel_size`(f32, 默认 -1.0 → =radius)；**两个构造路径（createPlugin 属性构建 与 deserializePlugin 反序列化）都执行 clamp `v=min(v,radius)`**（序列化存原始属性值，deserialize 后再 clamp——否则带 v>radius 的旧 engine 会重获坏 R；clamp 一次性记 log；扫描范围已公式化 R=ceil(radius/v)+1，clamp 为双保险）；序列化含全部 4 个属性
  - `getOutputDataType`：float 输出一律 kFLOAT、idx 返回 kINT32（plugin.md §13.5 v14.1/v14.2 陷阱规范）
  - `plugin_registry.cpp` 追加 2 个 `REGISTER_TENSORRT_PLUGIN`；`deploy/trt_plugins/CMakeLists.txt` 把新 .cu/.cpp **追加进现有 `add_library(hpenet_plugins SHARED ...)` 共享目标**（勿新建 target——自动继承 `CMAKE_CUDA_ARCHITECTURES` 默认 "87;89"（Orin+L20，:6-8）、TensorRT 链接（:33）与 include 路径；aarch64/Windows 重编走 plugin.md §7.2/§7.3 同一 CMakeLists + 对应 arch 覆盖）
  - Acceptance: `cd deploy/trt_plugins/build && cmake .. && make -j` 成功（开发机 L20=sm_89，arch 用默认 "87;89" 即可，plugin.md §7.1）；`nm -D libhpenet_plugins.so | grep GridBallQuery` 出现 2 个 Creator；现役 9 个插件注册数不变（共 11）
  - QA happy: 重新加载 .so 后 `trt_builder` 无重复注册警告；QA failure: 删掉 .so 时 trt_build.py 行为与改动前一致（CDLL 缺省路径守护，`trt_build.py:63-65`）
  - Commit: 不提交

- [x] 3. 导出侧接线（`deploy/onnx_ops/gridballquerygroup_op.py` + `gridballquerydp_op.py` + `onnx_backend.py` + `onnx_export.py`）
  - 两个 op 文件仿 `ballquerygroup_op.py` / `ballquerydp_op.py`（§13.5 模式）：symbolic 发 `hpenet::GridBallQueryGroup` / `hpenet::GridBallQueryDP` 节点（attributes radius/nsample/normalize_dp/voxel_size）；forward CPU/CUDA 均返回 shape/dtype 正确占位张量（导出在 CPU 跑 `onnx_export.py:87`；grid kernel 无 python 绑定，验证全部在 TRT 层做——与 fps 档不同但合规，在 op 文件 docstring 说明）
  - `onnx_backend.py::patch_model_for_onnx` 加 `bq_algo='ballquery'` 参数：`'ballquery'`（默认，现役工厂，行为 bit 不变）| `'gridballquery'`（`_make_sa_group_forward` 的 `group_fn` 与 `_make_invresmlp_group_forward` 的 `dp_fn` 换 grid 工厂，voxel_size 透传 -1）；未知值 raise ValueError
  - `onnx_export.py` 加 `--bq_algo` CLI（choices/默认/help，仿 `--fps-algo` L238-248），透传给 patch
  - Acceptance: `python deploy/onnx_export.py --bq_algo gridballquery`（CPU）导出成功；ONNX 图检查：`GridBallQueryGroup`×4 + `GridBallQueryDP`×4，BallQueryGroup/DP×0，总节点数与默认档相差 0（仅 op_type 改名）；默认档（不传参）导出产物与改动前 bit 级一致（回归守护）
  - QA happy: 两档各导出一次，节点计数脚本（仿 plugin.md L526-529 片段）输出符合期望；QA failure: `--bq_algo bogus` → SystemExit/ValueError 非零退出
  - Commit: 不提交

- [x] 4. 单元对拍（新文件 `deploy/tests_gridballquery.py`，仿 `tests_fps_algos.py` 的静态 shape engine ~0.1s build + harness）
  - 对拍矩阵：插件 {GridBallQueryGroup vs BallQueryGroup, GridBallQueryDP vs BallQueryDP} × N ∈ {1024, 3500, 6000, **10000**} × 精度 {fp32, fp16} × 数据 {随机均匀, 簇状(模拟雷达), 对抗集}
  - 对抗集（每个都必须有）：同 voxel 多质心（FPS 语义核心差异点）、坐标完全重复点、空球（radius 内 0 点，**只验 grid 自身确定性与不越界，不与现役对拍**）、**精确球面点（d2==radius²，验证严格 < 边界与现役一致）**、N=1、N<nsample、大 radius 全覆盖（全部点为邻居，验证 nsample 截断）
  - 判据（验收门槛 = 用户已确认的容差制）：①每 query 邻居**集合**在 min(cnt,nsample) 内一致（cnt≤nsample 时 bit 级；cnt>nsample 时允许集合差异但所有 idx 必须在 radius 内且互不重复）②padding 规则：空槽 == 首个邻居（cnt≥1 时）③dp/grouped max-abs-diff fp32 ≤1e-5 / fp16 ≤1e-2 ④同输入重复推理 3 次输出 bit 级一致（确定性守护，FPS 原版 atomicAdd 方案在此会挂——这是重设计的核心动机）
  - **诊断项（非门槛）**：grid idx 与现役 bit 级一致（升序索引 + 严格 < 选择使然）；若不一致需解释原因记入 evidence
  - Acceptance: 全矩阵 PASS，结果写 `deploy/evidence/gridballquery-unit.md`（含判据③实测最大值）
  - QA happy: 全绿；QA failure: 任何对抗用例 fail 即阻断任务 5
  - Commit: 不提交

- [x] 5. 整网验证（导出 → 双精度 build → 输出对比 → mIoU）
  - `python deploy/onnx_export.py --bq_algo gridballquery --output deploy/hpenet_v2_gridbq.onnx`（同 checkpoint 同 cfg，默认档产物 `hpenet_v2_plugin.onnx` 为对照）
  - `python deploy/trt_build.py --onnx deploy/hpenet_v2_gridbq.onnx --output deploy/hpenet_v2_gridbq_fp32.engine` 与 `--fp16` 版；现役 engine 已有（`hpenet_v2_fp32.engine` 等），缺则同法补建
  - 输出对比：同输入（真实雷达帧 N∈{3817,5727} + 合成 N∈{2024,4096,10000}）跑两套 engine，logits max-abs-diff fp32 ≤1e-2 / fp16 ≤3e-2，预测标签一致率 ≥99.5%；**若 diff 超标：先按 query 分桶定位 cnt==0（空球）行——现役对空球输出未初始化内存（既有 bug），其行 diff 无意义，单独报告并剔除后再判；mIoU 为主仲裁判据**
  - mIoU：radar 测试集（`RadarClassi` test split）经 engine 推理计算 mIoU/OA——**新增脚本 `deploy/eval_gridballquery_miou.py`**（**入口先 `ctypes.CDLL(deploy/trt_plugins/build/libhpenet_plugins.so, RTLD_GLOBAL)` 再反序列化 engine——plugin.md §6/L1908 硬前提，否则 `Could not find plugin: hpenet::GridBallQueryGroup`**，模式照抄 `trt_utils.py:80-82`；复用 `trt_utils.py` 会话管理与 `RadarClassi` test 预处理：voxelize→engine 单次前向→argmax→与 label 累计混淆矩阵→mIoU/OA；**不做 voxel voting**，两档 engine 用完全相同预处理保证公平对比），grid vs 现役**绝对 mIoU 差 ≤0.5 个百分点**（用户已确认门槛）
  - Acceptance: evidence 写 `deploy/evidence/gridballquery-e2e.md`（diff 表 + mIoU 表）
  - QA happy: 全部门槛达标；QA failure: mIoU 降幅 >0.5% → 定位（单元对拍回溯）并修复 kernel 后重跑 4→5
  - Commit: 不提交

- [x] 6. nsys 性能对比（FP16+FP32 双精度，用户已确认）
  - 前置：`which nsys || ls /opt/nvidia/nsys-system/*/bin/nsys`（用户确认系统已装）定位可执行文件
  - 基准脚本：新文件 `deploy/bench_gridballquery.py`——**入口先 `ctypes.CDLL(libhpenet_plugins.so, RTLD_GLOBAL)` 再加载 engine**（同 `tests_fps_algos.py:59-63` 模式），warmup 20 + 计时 100 次（cuda.Event 交叉验证），输入 N=opt 4096（主）+ 2024/10000（动态区间两端）
  - nsys 采集：`nsys profile -o deploy/evidence/gridbq_{prec} --trace=cuda,nvtx python deploy/bench_gridballquery.py --engine ...`；对 4 个 engine（grid/现役 × fp32/fp16）各采一次；NVTX range 包裹推理循环
  - 对比指标（`nsys stats --report cuda_gpu_kern_sum` 提取）：①grid 侧 kernel 分解（build_count / scan / place / query / dp / gather 各自耗时）vs 现役 `ball_query` kernel 单核耗时 ②两组件（Group/DP）插件总耗时 ③整网 GPU 时间 ④e2e 延迟（cuda.Event 交叉验证）；**按 N∈{2024,4096,10000} 分桶报告**（O(N²) rank 在小 N 可能被 launch 开销主导、大 N 与暴力法同量级——分桶才能暴露退化区间）
  - Acceptance: `deploy/evidence/gridballquery-nsys.md` 含完整对比表 + 结论（grid 更快/更慢/持平，量化倍数）+ `.nsys-rep` 文件留存；**结论允许为负**（grid 更慢也是合法交付物，如实记录）
  - QA happy: 4 份 profile 采集成功且 kernel 名可分辨；QA failure: nsys 不可用/版本过旧 → 记录错误，仅交付 cuda.Event 数据并明确标注降级
  - Commit: 不提交

- [x] 7. 文档（plugin.md + AGENTS.md）
  - plugin.md 新增 §xx「GridBallQuery 档（gridballquery）」：设计（source-hash/稳定计数排序/公式化范围 R=ceil(radius/v)+1 格扫描+桶内 cell 消歧/索引最小 nsample 个 + 严格 < 判据 + 升序槽位写入）、与 FPS 原版的三处语义差异（query-hash→source-hash、原子散射→确定序、HAV 前置依赖→内置 enqueue）、**性能前提风险**（O(N²) rank ≈ 现役 O(M·N)，附 nsys 分桶实测结论）、验收数据回填（单元/e2e/nsys 三份 evidence 摘要）、`--bq_algo` 用法、风险表追加两行（①voxel_size 误配 → 对策：构造函数 clamp v=min(v,radius)（任务 2）+ 公式化扫描范围双保险 ②**现役空球 cnt==0 输出未初始化内存的既有潜在 bug**——grid 档已用 memset 规避，现役行为原样保留并记录）
  - AGENTS.md DEPLOY 节：插件清单 9→11、`--bq_algo` 一句话
  - **跨平台声明（plugin.md §7/§11 平台矩阵对照）**：GridBallQueryGroup/DP **不使用 `kFLOOR_DIV`**（输出 dim = inputs[1].d[1] + constant(nsample)，无整除表达式）→ 不引入 PrefixFPS 那类 TRT 8.5/Orin 兼容风险（plugin.md L1156）；kernel 原语（`__shfl_up_sync`/`roundf`/`ceilf`/`fminf`/寄存器 scan）全部 CUDA 11.4（JetPack 5.x）兼容；无新增 python/C++ 依赖；协变返回系 C++ 语言特性，TRT 8.5 Orin 重编同样兼容（L1159 口径）。Orin(aarch64)/Windows 重编+engine 重建属**部署期步骤**（plugin.md §7.2/§7.3、与 v14 遗留口径一致，L1700），不入本计划任务，写入 plugin.md 风险表备查
  - Acceptance: 两文件更新，plugin.md 章节含实测数字（非占位）
  - QA happy: `grep -c 'GridBallQuery' plugin.md` ≥5；QA failure: 无（纯文档）
  - Commit: 不提交

## Final verification wave

- [x] F1. 计划符合性审计：逐任务核对 1-7 的 Acceptance 全部满足、evidence 文件（gridballquery-unit.md / gridballquery-e2e.md / gridballquery-nsys.md）存在且含实测数字；现役插件行为零改动验证（默认档导出 bit 一致 + 现役 engine 推理输出与改造前一致）
- [x] F2. 代码质量审查：新 kernel/插件代码过一遍协变返回/序列化对称/getOutputDataType 规范（对照 plugin.md §13.5 陷阱清单）；无内存泄漏（workspace 全部经 TRT 分配）；N=10000 边界无 int 溢出（B*M*S 用 size_t 索引）
- [x] F3. 真实手工 QA：亲手跑一次完整链路 `export --bq_algo gridballquery → trt_build fp16 → bench`，确认可复现；nsys GUI/CLI 打开 .nsys-rep 确认 kernel 时间线可见
- [x] F4. 范围忠实性：确认 Must-NOT-Have 全部守住（git status 显示无 git 操作痕迹、现役 4 文件零 diff、ONNX 节点数 628 不变、cfgs/ 零改动）

## Commit strategy

**不执行任何 git 操作**（AGENTS.md 明确约束，用户未授权）。所有改动留在工作区（根仓库 + deploy 子模块），由用户自行审查提交。

## Success criteria

1. `--bq_algo gridballquery` 一键导出可复现，节点数 628±0
2. 单元对拍矩阵全绿（含 7 类对抗用例 + 确定性 3 次复跑 bit 级一致）
3. 整网 logits diff 达标 + radar **绝对 mIoU 差 ≤0.5 个百分点**
4. nsys 对比报告（FP16+FP32）落盘，结论量化
5. 现役路径 bit 级不受影响（默认档导出 + 现役 engine 回归）
6. plugin.md/AGENTS.md 文档同步

## Background（可行性结论，探索已证实）

- 现役链路：`onnx_backend.py:406-513` monkeypatch → `hpenet::BallQueryGroup`×4（SA）+ `hpenet::BallQueryDP`×4（InvResMLP）；kernel 在 `ballquerygroup_kernel.cu`（暴力 O(M·N) + dp + gather）
- FPS GridBallQuery 不能照搬的三处硬伤：①hash 建在 query 点且每 voxel 单 query（slots2queries 单映射，配合 HAV 体素采样设计；HPENet 质心来自 FPS，同 voxel 多质心会静默丢邻居）②依赖 HAVSForQuery 前置插件构建 hash（deploy 无此算子，外置会新增 ONNX 节点违背碎片化约束）③source 驱动 atomicAdd → 非确定
- 改造核心：hash 改建 source 点（稳定计数排序解析多源点/桶）、query 驱动确定性别扫描、构建内置 enqueue（workspace）、padding 复刻现役规则、"半径内索引最小 nsample 个"选择使 idx 有望与现役 bit 级一致
- hpenet-ll 参数：radius=10（encoder 内 `_to_full_list(radius, radius_scaling=2)` 逐级放大）、nsample=32、N 动态 2024~10000（`trt_build.py` argparse 默认 ~L177-191）
- 用户决策（2026-08-22 已确认）：新增可选档 / 容差+mIoU≤0.5% / nsys 系统已装 / FP16+FP32 都测
