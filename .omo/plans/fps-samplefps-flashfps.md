# fps-samplefps-flashfps - Work Plan

## TL;DR (For humans)
<!-- Fill LAST -->

**What you'll get:** 现在最远点采样插件保留不动，另加两个新插件：一个是我们自己写的高速精确算法（SampleFPS），一个是借鉴 FlashFPS 论文的"少算"方案（剪枝+跨层复用）。三个算法可以用一个开关随意切换，并附一套一键对比脚本，分别验证精度和速度。

**Why this approach:** 采样占了推理耗时的八成以上，是最大的提速目标；先让三个算法并存对比，拿到实测数据后再决定主力方案，避免一步换死。新算法里"精确版"逐点结果与现役完全一致（零精度风险），"FlashFPS 版"以极小精度代价换最大速度。

**What it will NOT do:** 不动模型权重、不重新训练；不改球查询等其它算子；不删除现有插件；不执行任何 git 提交；不支持多 batch 和跨平台编译。

**Effort:** Medium-Large（7 个 todo，4 个执行波次）
**Risk:** Medium - SampleFPS kernel 的 bit 级等价（tie-break 一致性）与 FlashFPS 近似路径的 acc 守门是两个硬验收点
**Decisions to sanity-check:** keep_rate=保留比例（与 FlashFPS PruneRate 对齐，0.75=留 75% 精确采样）；FlashFPS 填充语义取仓库复现的"升序（顺序）填充"（非论文随机填充，对扫描序雷达有空间偏置风险，由 per-frame acc 分布监控）；FlashFPS 与 SampleFPS 为两独立插件类（共享 kernel 头文件）；Cache 用 PrefixFPS 动态形状插件而非常量 Slice；"bit 级一致"验收已放宽为两级判据（现役 tie-break 为 min(k mod block_size)→min(k)，不复刻）

Your next move: 审阅后可用 `$start-work` 开始执行；或先要求一轮高精度评审（momus+Oracle 双审）。Full execution detail follows below.

---

> TL;DR (machine): Medium-Large effort, Medium risk; deliverables = 三选项 FPS 插件族（FPS/SampleFPS/FlashFPS）+ 图级 Cache + 对比测试报告，逐索引等价与精度守门。

## Scope
### Must have
1. **三选项 FPS 算法插件族**（同图可切换，便于 A/B/C 对比）：
   - 选项 A：**hpenet::FPS**（现役插件，零改动保留）
   - 选项 B：**hpenet::SampleFPS**（自研精确 GPU kernel，参考 `FPS/fpsample/src/_ext` KDLineTree 桶结构）
   - 选项 C：**hpenet::FlashFPS**（SampleFPS kernel + FlashFPS Prune 剪枝语义；端到端方案另含图级 Cache）
2. SampleFPS/FlashFPS TRT 插件（IPluginV2DynamicExt，对齐现役插件代次与张量约定）
3. FlashFPS **Prune**（候选剪枝+迭代剪枝+升序（顺序）填充，`keep_rate` 属性=保留比例，与 FlashFPS PruneRate 对齐）与 **Cache**（第 2-4 级 FPS 节点 → PrefixFPS 轻量插件（动态形状安全的前缀取点），图级改造 `onnx_backend.py`）
4. 导出开关：`onnx_backend.patch_model_for_onnx` 增加 `fps_algo ∈ {fps, samplefps, flashfps}`（默认 `fps` 保持现状）
5. 三算法对比测试：单算子对拍（idx bit 级 vs 现役）、端到端 acc、nsys 延迟
6. plugin.md 文档更新（新插件设计 + 实测记录）

### Must NOT have (guardrails, anti-slop, scope boundaries)
- 不删/不改现役 FPS 插件任何行为（回退通道）
- 不重训/微调模型，不动权重；acc 守门 0.9741
- 不改 BallQuery/ThreeInterp 等其它算子
- 不引入 FPS/ 各参考仓库的任何构建依赖（仅算法语义移植，代码自写）
- 不做 batch>1、不做 Orin/Windows 交叉编译（沿用既有部署阶段）
- 不做 git commit（仓库约定：未经用户允许禁止 git 操作）

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after + 单算子对拍脚本（照搬 `/tmp/opencode/v1_plugins.py` 的 save_model/build/run 模式）+ gtest 可选；无 pytest 框架（仓库无单测基建）
- Evidence: `.omo/evidence/task-<N>-fps-samplefps-flashfps.{log,md}`（非 ulw-loop 会话用 `.omo/evidence/`）
- 判定标准（沿用 plugin.md:1374 V1 标准，**FPS 的 idx 一致性按两级判据放宽，见 task 1/5**）：fp32 idx 无平局输入逐索引一致；fp16 max_rel_err ≤ 1e-3；N 扫 1024/2750/5500/6500 + 真实雷达帧
- FPS 序列精确性判定（照搬 `/tmp/opencode/verify_fps.py`：逐步验证每步选点为全局 argmax）

## Execution strategy
### Parallel execution waves

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 (SampleFPS kernel) | - | 2,3,4,6 | 5 |
| 2 (插件类+注册) | 1 | 3,4(c),6 | 5 |
| 3 (ONNX op + backend 开关) | 2 | 6 | 4 |
| 4 (Prune kernel 路径 + FlashFPS 独立插件类) | 1, 2, 3（wiring 复用 task 3 的 PrefixFPS 算子，仅 wiring+验收需 3；(a)(b)(c) 仅需 1,2） | 6 | 5 |
| 5 (对比测试脚本) | - | 6 | 1-4 |
| 6 (三算子对拍+acc+nsys) | 1-5 | 7 | - |
| 7 (plugin.md 文档) | 6 | F | - |

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

- [x] 1. 编写 SampleFPS 精确 GPU kernel（samplefps_kernel.cu/.h）
  What to do：在 `deploy/trt_plugins/src/` 新写 `samplefps_kernel.cu` + `include/samplefps_kernel.h`。算法=kdline 桶结构 GPU 化（蓝本 `FPS/fpsample/src/_ext/KDLineTree.h`、`KDTreeBase.h`，语义参考本会话审查结论），**两阶段结构**：(1) 建桶 kernel（单次、coalesced 空间哈希、CSR 布局 bucket_offsets+point_idx）；(2) 单 block 迭代 kernel（共享内存缓存 best/桶上界，每轮读选中点→剪枝→**幸存桶紧凑（compaction：共享内存 alive 标志 + 块内 scan/prefix-sum 生成紧凑桶表，或 atomicAdd 工作表——禁止靠逐桶分支退化回 strided 访存）**→幸存桶内合并访存 min-dist 更新→树归约 argmax）。**共享内存容量约束：桶数=占用栅格单元数，最坏 O(N)（profile max N=10000 时上界数组 40KB + dists 8KB 逼近 48KB 上限）——建桶网格分辨率须限定桶数上限（如 ≤6144）或上界数组驻 global + 分块缓存，禁止无界 shared 数组**。其余约束：(a) 迭代 kernel 单 kernel 内完成全部 K 轮（**禁止逐迭代 kernel launch、禁止 host 循环/同步、禁止 multi-block+grid sync**——K=2750 次 grid 同步 ≈3-5.5ms 纯开销会吃掉剪枝收益，B=1 下单 block 是唯一安全起点）；(b) 禁用 QuickFPS 的 devide 归并排序建树；(c) 确定性 tie-break=小索引优先（SampleFPS 自身语义，**不复刻现役 kernel 的 min(k mod block_size)→min(k) 归约序**——见验收标准）；(d) 固定种子：首点=索引 0；(e) **距离表达式锁死**为 `(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)` float32 左结合、禁 fmaf/double 中间量（否则非平局也可能 1 ULP 分叉）；(f) 接口 `samplefps_launcher(B, N, M, keep_rate, xyz_fp32, temp, idx_int32, stream)`，B=1 优化但不硬编码 batch。
  Must NOT do：不使用 thrust；不改 `fps_kernel.cu`；不引入参考仓库头文件（自写，注释标注蓝本出处）。
  Parallelization: Wave 1 | Blocked by: 无 | Blocks: 2,4,6
  References：
  - 蓝本：`FPS/fpsample/src/_ext/KDLineTree.h`、`KDTreeBase.h`、`Interval.h`（CPU 桶 FPS，已证 L1=2.39ms 精确；注意其距离表达式未经验证，必须按上式重写）
  - 反面教材：`FPS/bucket-based_farthest-point-sampling_GPU/src/kdtree.cu:296-314`（checkBucket 剪枝不等式可借鉴，devide 建树禁用）
  - 现役 kernel（对拍基准）：`deploy/trt_plugins/src/fps_kernel.cu:13-65`；launcher 模式 `fps_kernel.cu:67-86`；注意现役 tie-break 实际为 `min(k mod block_size)→min(k)`（每线程 k=tid;k+=stride，树归约小 tid 胜），block_size 经 opt_n_threads 随 N 变化（N=5500/2750/1375→1024，687→512，343→256）
  - torch-fps 正确性设计参考：`FPS/torch-fps/torch_fps/fps_cuda.cu:44-229`（单 kernel 迭代、确定性归约写法；性能反面教材：256线程/int64/多屏障）
  - 张量约定：xyz FLOAT (B,N,3) 连续、idx INT32 (B,M)、temp=workspace (B,N) float（`deploy/trt_plugins/src/fps_plugin.cpp:36-54`）
  Acceptance criteria：独立 .cu 编译通过；**两级判据**——(a) 无平局输入（随机浮点数据）上 idx 序列与现役 fps_launcher 逐索引一致（N=5500,M=2750 及全 N 扫描）；(b) 主判据=verify_fps 逻辑"每步选点为全局 argmax 且 tie-break 最小原索引"在合成 4096 点 + 真实雷达帧上 100% 通过（**不要求复刻现役的平局选择**）。退化输入（全同点）只要求不死锁不越界（现役对全同点返回全 0 序列，SampleFPS 允许不同）。
  QA scenarios：happy=两级判据通过；failure=乱序输入/全同点（退化输入 CUDA error check 通过）。附：写 10 行脚本构造精确平局输入（立方体 8 顶点+原点/规则栅格），跑现役 vs openpoints kernel 对拍确认二者平局行为，并在真实雷达帧上统计"逐索引不一致但每步仍全局 argmax"的平局出现率，量化结果记入 evidence。Evidence `.omo/evidence/task-1-fps-samplefps-flashfps.log`
  Commit: N（仓库约定禁 git 操作）

- [x] 2. SampleFPS TRT 插件类 + 注册
  What to do：新写 `deploy/trt_plugins/src/samplefps_plugin.cpp` + `include/samplefps_plugin.h`，照抄现役 `fps_plugin.cpp`（133 行）骨架改：type="SampleFPS"、version="1"；属性仅 `stride`(int32)（**SampleFPS 恒精确不带 keep_rate**；keep_rate 属性归 FlashFPSPlugin，防两类型语义混用）；`getOutputDataType`=INT32、`supportsFormatCombination` 输入 FLOAT/输出 INT32、workspace=maxB×maxN×sizeof(float)；enqueue 调 samplefps_launcher。改 `plugin_registry.cpp`（include + `REGISTER_TENSORRT_PLUGIN(SampleFPSPluginCreator)`）与 `CMakeLists.txt:14-24`（加 samplefps_kernel.cu/samplefps_plugin.cpp）。重编 `libhpenet_plugins.so`。
  Must NOT do：不动 fps_plugin.* 与其它插件行；注册顺序追加在现有 5 个之后。
  Parallelization: Wave 2 | Blocked by: 1 | Blocks: 3,6
  References：骨架蓝本 `deploy/trt_plugins/src/fps_plugin.cpp` 全文；注册 `deploy/trt_plugins/src/plugin_registry.cpp:8-16`；构建 `deploy/trt_plugins/CMakeLists.txt:14-24`；构建命令 `plugin.md` §7.1（cmake .. && make，plugin.md:911-920）；加载机制=静态初始化自动注册（trt_build.py:52-56 ctypes.CDLL，无需改加载侧）。
  Acceptance criteria：`nm build/libhpenet_plugins.so | grep SampleFPSPluginCreator` 有符号；TRT builder 能解析含 SampleFPS 节点的最小 ONNX（用 v1_plugins.py 的 save_model 模式）。
  QA scenarios：happy=最小图 build+run 成功；failure=stride 越界（<1）序列化/反序列化后属性保持。Evidence `.omo/evidence/task-2-fps-samplefps-flashfps.log`
  Commit: N

- [x] 3. ONNX 算子 hpenet::SampleFPS + onnx_backend 三选项开关 + PrefixFPS 轻量插件（Cache 载体）
  What to do：(a) 新写 `deploy/onnx_ops/samplefps_op.py`（照 `fps_op.py` 模式：FPSOp→SampleFPSOp，symbolic 发 `hpenet::SampleFPS`，attrs 仅 `stride_i`（keep_rate_f 归 flashfps_op），forward 回落 openpoints furthest_point_sample 保证 torch 侧语义）；(b) **新写 PrefixFPS 轻量插件**（Cache 的动态形状安全载体）：`deploy/trt_plugins/src/prefixfps_plugin.{cpp,h}` + ONNX 算子 `hpenet::PrefixFPS`（attr `stride_i`）——getOutputDimensions 用 `kFLOOR_DIV(输入 N, stride)` 动态推导 Mℓ（照抄现役 fps_plugin.cpp:23-34 的写法），enqueue 仅执行 `idx[b*M+i]=i`（arange 填充，零距离计算，支持 B 循环）；**torch 侧 forward 必须返回 arange(M, dtype=int32)（不是真 FPS——Cache 语义就是取前缀）**；**workspace=0**（不申请距离缓冲）；supportsFormatCombination 显式 FLOAT 输入/INT32 输出（照 fps_plugin.cpp:36-43）；**不得用 ONNX Slice/arange 常量实现 Cache**（torch.jit.trace 会把导出时的 Mℓ=N/2^ℓ 烘焙成常量，而部署管线 N 动态——trt_inference.py 每帧 N 可变、profile min1024/opt4096/max10000，常量切片对 N≠导出值直接形状错误）；(c) `onnx_backend.py` patch_model_for_onnx 增加参数 `fps_algo='fps'`：'fps'→现状 make_fps_op（**默认，零行为变化**）；'samplefps'→各级 make_samplefps_op(stride, keep_rate=1.0)；'flashfps' 档的 wiring **由 task 4 拥有**（首节点是 task 4 的 hpenet::FlashFPS 算子；task 3 不实现 flashfps 档，避免两任务并行冲突）（前缀等价性：**除浮点平局外 bit 级成立**——各级种子=同一几何点（首点索引 0）、下级输入=上级输出原序（hpenetv2.py:160-161,202 gather 保序）、min 距离增量更新在 float32 下精确；平局时 tie-break 索引空间不同（原索引 vs FPS 序位置）可能断裂，雷达 range-azimuth 准栅格结构需实测量化，见 task 1 平局实验。两个性质须区分：『第 1 级按选择序输出』是 kernel 构造保证（可 enforce，逐位写入 idxs[j]）；『无跨前缀平局』是数据依赖（不可 enforce，只能统计验证））。注意：hpenet-ll 的 use_res=False 且 feature_type='dp_fj'，onnx_backend.py:322-324 的 fi gather 分支是死代码，**无需为其做切片适配**；(d) PrefixFPS 的 CMakeLists.txt:14-24 源清单与 plugin_registry.cpp 注册（REGISTER_TENSORRT_PLUGIN(PrefixFPSPluginCreator)）随本任务一并完成；**注意 plugin_registry.cpp 与 CMakeLists.txt 由 tasks 2/3/4 三方追加行，须按 2→3→4 顺序串行合并（或由单执行者统一追加三行），禁止并行编辑同文件**；(e) 导出脚本 `onnx_export.py` 透传 fps_algo（argparse 新增）。
  Must NOT do：默认路径（fps_algo='fps'）导出的 ONNX 与现状逐节点一致（回归保证）；不删 traceable_random_fps；Cache 禁止常量 Slice 实现。
  Parallelization: Wave 2-3 | Blocked by: 2 | Blocks: 6
  References：`deploy/onnx_ops/fps_op.py` 全文（25 行）；`deploy/onnx_backend.py:210-213`（sample_fn 替换点）、`L313-320`（idx emit 与 gather 消费）；动态形状依据：`deploy/onnx_export.py:98-102`（npoints 动态标记）、`deploy/trt_build.py:84-94`（动态 profile）、`deploy/trt_inference.py:40-52`（每帧 N 可变）；SA stride 流转 `openpoints/models/backbone/hpenetv2.py:107,160`；fi 死代码依据：`cfgs/radar/hpenet-ll.yaml`（use_res=False、feature_type='dp_fj'）+ `onnx_backend.py:322` 条件。
  Acceptance criteria：**两种 fps_algo（'fps'/'samplefps'）各导出并验收一个 onnx**（'flashfps' 档图归 task 4 导出+验收，task 3 不导出 flashfps）：'fps' 图与现存 `hpenet_v2_plugin.onnx` 节点数/类型一致——**用 onnx 库机械比对**：`python -c "import onnx; a=onnx.load('x.onnx'); b=onnx.load('y.onnx'); assert [n.op_type for n in a.graph.node]==[n.op_type for n in b.graph.node] and len(a.graph.node)==len(b.graph.node)"`（onnx 1.17 已装）；'samplefps' 图含 4 个 hpenet::SampleFPS 节点（恒精确）；'flashfps' 档图（task 4 验收）含 1 个 hpenet::FlashFPS（keep_rate_f=0.75）+ 3 个 hpenet::PrefixFPS 节点；**PrefixFPS 动态形状运行时断言**（构建期 getOutputDimensions 只做符号推导，须实际跑）：照 v1_plugins.py 的 save_model/build/run 模式建 PrefixFPS 最小引擎，N=1024 与 6500 两档各跑一次，断言输出 shape==(B, N//stride)（链式 floor 恒等式保证与真 FPS 链形状一致，二审已证）。
  QA scenarios：happy=三图 build 成 engine 且 PrefixFPS 动态形状两档全过；failure=keep_rate 传非法值报错清晰。Evidence `.omo/evidence/task-3-fps-samplefps-flashfps.md`（图 diff 摘要）
  Commit: N

- [x] 4. FlashFPS Prune kernel 路径 + hpenet::FlashFPS 独立插件类
  What to do：(a) samplefps_kernel.cu 内实现 prune 路径（keep_rate<1 时，**仅 FlashFPSPlugin 调用此路径；SampleFPSPlugin 不带 keep_rate 属性，恒精确**）：**keep_rate 语义=保留比例（与 FlashFPS PruneRate 对齐，keep_rate=0.75 ⇔ 参考代码 PruneRage=0.75，注意参考参数名拼写即 PruneRage）**——候选剪枝=仅对前 int(keep_rate·N) 个点建桶与参与距离更新；迭代剪枝=只跑 **num_points = int(int(N·keep_rate)/stride)**（**照抄参考公式，勿简化为 int(keep_rate·M)——奇数 N 时二者差 1**，部署动态 N 可为奇数）轮精确 FPS；填充=剩余 (1−keep_rate)·M 个位置按候选前缀 [0, keep_rate·N) 内未被选中索引**升序（顺序）填充**——kernel 实现=selected 位图 + 块内 scan 紧凑出未选中序列，可在迭代 kernel 尾声完成，不引入第三段 kernel（语义=FlashFPS-Openpoints `openpoints/models/layers/subsample.py:83-144` standard_fps+rearrange_indices+FPS_Prune，本会话已完整审读；**升序填充≠随机填充**——FlashFPS 论文用随机，仓库复现用顺序，取仓库语义；在扫描序雷达数据上升序填充=取最早扫描位置的空间簇，75% 填充时 FPS 覆盖基本丧失，此风险由 task 6 的 per-frame acc 分布监控，keep_rate 0.5/0.75 为计划内可调档位）。(b) 新写 `deploy/onnx_ops/flashfps_op.py`：symbolic 发 `hpenet::FlashFPS`（attrs stride_i + keep_rate_f，默认 0.75）；(c) **新写独立插件类 FlashFPSPlugin**（`deploy/trt_plugins/src/flashfps_plugin.{cpp,h}`，type="FlashFPS"，**不得与 SampleFPSPlugin 共类双 creator**——getPluginType 返回固定串，共类会导致 FlashFPS 节点以 "SampleFPS" 类型反序列化失败；仓库先例 ballquerygroup_plugin.cpp 即两独立类 BallQueryGroupPlugin/BallQueryDPPlugin，可共享 kernel 头文件）；plugin_registry 追加 FlashFPSPluginCreator 注册。task 3 的 'flashfps' 档 wiring 由本任务实现：首节点发 hpenet::FlashFPS（make_flashfps_op），第 2-4 节点替换为 hpenet::PrefixFPS 节点（复用 task 3 的 PrefixFPS 算子）。**cache 前缀假设『前 Mℓ 点为精确 FPS』要求 keep_rate·M ≥ M₂ 即 keep_rate ≥ 0.5——低于 0.5 时第 2 级前缀落入升序填充区，cache 精度进一步退化（近似语义下允许，文档注明）**。验收补充：'flashfps' 档导出图含 1 个 FlashFPS + 3 个 PrefixFPS 节点。
  Must NOT do：prune 不改变输出 dtype/shape 约定；填充必须确定性顺序；PluginField 名用剥后缀裸名 `stride`(kINT32)/`keep_rate`(kFLOAT32)，serialize/deserialize 字节序严格对称。
  Parallelization: Wave 2（(a)(b)(c) 可与 3 并行）→ Wave 2-3 收尾（wiring+验收，需 task 3 完成） | Blocked by: 1, 2, 3（wiring 需 3） | Blocks: 6
  References：语义蓝本 `FPS/FlashFPS/FlashFPS-Openpoints/openpoints/models/layers/subsample.py:83-144`（standard_fps/rearrange_indices@:98/FPS_Prune@:127-144；注意 `FPScache.py` 无实际 cache 实现（仅 `cache = None` 单行占位），cache 语义以论文 Algorithm 1 为准）；FlashFPS 论文 Algorithm 1（arXiv:2604.17720 §3.2-3.3，本会话已取全文）；两独立类先例 `deploy/trt_plugins/src/ballquerygroup_plugin.cpp`；符号映射：`keep_rate = PruneRage`（参考代码 N_points=N*PruneRage 即保留比例）。
  Acceptance criteria：keep_rate=1.0 时 FlashFPS 输出与 SampleFPS 逐索引一致；keep_rate=0.75 时输出前 keep_rate·M 个索引与精确 FPS 序逐索引一致、尾部为候选前缀内升序未选中索引；verify_fps 判定对前缀段 100% 通过。
  QA scenarios：happy=keep_rate=1.0/0.75/0.5/0.25 四档全过；failure=keep_rate 极小（候选数<目标 M 的退化，输出仍完整 M 个索引）。Evidence `.omo/evidence/task-4-fps-samplefps-flashfps.log`
  Commit: N

- [x] 5. 三算法对比测试脚本（单算子级）
  What to do：新写 `deploy/tests_fps_algos.py`（仓库内持久化，脱离 /tmp）：(a) 照 `/tmp/opencode/v1_plugins.py` 的 save_model/build/run 模式，对 FPS/SampleFPS/FlashFPS(keep_rate=1.0/0.75)/PrefixFPS 五配置各生成最小 ONNX→engine→执行；(b) 对拍矩阵（两级判据）：①无平局随机输入上 idx 逐索引一致（vs 现役插件、vs openpoints CUDA 参考实现；**随机输入先坐标去重（unique rows）；①的失败一律转②归因平局而非直接判败，tie 用 float32 精确 == 判定**）；②真实雷达帧上 verify_fps 序列精确性（每步全局 argmax）+ 平局出现率统计；**参考实现约束：numpy 用 float32 且与 kernel 同表达式（(dx*dx+dy*dy)+dz*dz 左结合、非融合、逐步 float32），比较 eps=0 bit-exact——float64 参考会产生 1-ULP 伪不一致**；(b2) **PrefixFPS 语义对拍**（不只验形状）：cache-only 图（flashfps+keep_rate=1.0）第 2-4 级输出与全 SampleFPS 图对应级逐索引一致（前缀等价性直接实证）；(c) fp32/fp16 双精度；(d) kernel 级计时（CUDA event，warmup 20 + median）；(e) PrefixFPS 动态 N 两档（1024/6500）形状验证；(f) 结果输出 markdown 表。
  Must NOT do：不依赖 GPU 之外的临时文件；脚本可重复执行（固定种子）。
  Parallelization: Wave 1（脚本编写）→ 全绿验收**延后**（deferred acceptance：FPS 配置可立即验收，SampleFPS/FlashFPS/PrefixFPS 配置须待 tasks 1-4 完成，task 6 前补跑全量） | Blocked by: 无（全绿验收实质依赖 1-4） | Blocks: 6
  References：`/tmp/opencode/v1_plugins.py`（框架+三个已知坑：属性名无 _f/_i 后缀、engine 绑定 dtype、force_half_bindings）；`/tmp/opencode/verify_fps.py`（序列判定）；V1 判定标准 `plugin.md:1374`（fp32≤1e-5/idx 一致/fp16≤1e-3，其中"idx 一致"按本计划两级判据执行）；真实数据路径与加载 `openpoints/dataset/radar/s3disRadar.py`（RadarClassi，PLY 字段 x,y,z,mag,rcs,snr,v,label）。
  Acceptance criteria：脚本一键跑完输出对比表；SampleFPS 无平局输入全 N 全精度 idx 逐索引一致；FlashFPS keep_rate=1.0 等价 SampleFPS；平局出现率统计入表。
  QA scenarios：happy=全绿表；failure=故意注入 stride 错配时脚本能报出不一致（自检能力）。Evidence `.omo/evidence/task-5-fps-samplefps-flashfps.md`
  Commit: N

- [x] 6. 端到端三算法对比：acc + 延迟 + nsys
  What to do：(a) 三图（task 3 的 fps/samplefps 图 + task 4 的 flashfps 图）各 build engine（fp32+fp16，照 trt_build.py 流程，动态 profile 保持现状）；(b) 端到端推理对拍：与现役 engine 在 10 个测试文件上 logits 对比——'fps'（回归：必须与现役逐位一致）/'samplefps'（精确等价：真实帧上逐位一致为**主判据**，若出现不一致须用 task 5 平局统计归因）/'flashfps'（近似：报告 max_abs_err 与分布 + **per-frame acc 分布（min/分位数）+ 与 samplefps 的混淆矩阵差异**——监控升序填充对扫描序雷达的空间偏置，keep_rate 0.5/0.75 为计划内可调档位，未达门槛时允许下调并复测，不视为失败）；(c) acc 测试：`mode=test` 全测试集，门槛 'samplefps' 与 Cache 路径 acc == 现役（0.9741 守门），'flashfps' keep_rate=0.75 ≥ 0.9712；(d) 延迟：trt_inference 计时（median/P99）+ nsys 复测 FPS 段占比，产出三算法延迟对比表。注意：单 block 桶结构在 B=1 小 N 下 GPU 未必赢 CPU 串行 kdline（CPU 2.39ms 不保证 GPU 同速），<5ms 目标以实测为准，若未达记录瓶颈分析而非硬凑。
  Must NOT do：不修 checkpoint；不改 dataset 管线；nsys 用 `-t cuda`（推理期不载 cuDNN/cuBLASLt 的已知结论）；不为凑延迟门槛改变算法语义。
  Parallelization: Wave 3 | Blocked by: 1,2,3,4,5 | Blocks: 7
  References：engine build `deploy/trt_build.py`（dlopen L52-56、动态 profile L84-94）；推理 `deploy/trt_inference.py`/`trt_utils.py:79-82`；acc 测试入口 `script_me/main_segmentation_test.sh`（现指向 hpenet-ll.yaml）；基线数据：acc 0.9741、26.53ms/iter（plugin.md §9）；FPS 段 22.05ms/82.4% 出自 nsys trace `/tmp/opencode/hpenet_full.nsys-rep`（**非 plugin.md**，task 6(d) 复测重建立基线）。
  Acceptance criteria：产出对比表（acc 含 per-frame 分布/延迟/FPS占比 三算法 × fp32/fp16）；'samplefps' 真实帧 logits 逐位=现役（或平局归因成立）；'flashfps' acc≥0.9712 且 per-frame 无显著尾部（min ≥ 现役 min−0.02）；FPS 合计目标 <5ms（未达则记录瓶颈分析）；端到端 'flashfps' ≤ 现役 26.53ms。
  QA scenarios：happy=表全达标；failure=精度门槛未达 → keep_rate 下调复测或平局归因，均不达标才判定失败。Evidence `.omo/evidence/task-6-fps-samplefps-flashfps.md`
  Commit: N

- [x] 7. plugin.md 文档更新
  What to do：plugin.md 追加新章节（沿用现有版本号风格 v15.x）：三算法插件设计（接口/属性/kernel 结构）、前缀等价性论证引用、FlashFPS 语义移植说明（含与上游仓库差异：升序填充 vs 随机）、对比实测数据（task 5/6 结果回填）、修订历史。
  Must NOT do：不改既有章节的历史记录；数据必须来自 task 5/6 实测，不引论文数字充当实测。
  Parallelization: Wave 4 | Blocked by: 6 | Blocks: F
  References：`plugin.md` 结构（§3 架构、§9 实测、§13 实施记录、修订历史行 5）。
  Acceptance criteria：章节含三算法对比表（实测数据）；grep 验证无悬空引用。
  QA scenarios：happy=文档与产物一致；failure=数据与 evidence 不符即回改。Evidence `.omo/evidence/task-7-fps-samplefps-flashfps.md`
  Commit: N

## Final verification wave
- [x] F1. Plan compliance audit：逐 todo 核对 acceptance 全过、evidence 齐全、Must NOT have 零违反（重点：现役 FPS 插件 diff 为零、默认 fps_algo='fps' 导出图与现状逐节点一致）
- [x] F2. Code quality review：CUDA 规范（__syncthreads 前不可早退、batch 偏移显式、float/double 比较约定——沿用 plugin.md 既有 CUDA 审查清单）；无 thrust/参考仓库代码拷贝
- [x] F3. Real manual QA：三算法各跑一次完整推理 + 重跑 tests_fps_algos.py 全绿；nsys 抽查一份 trace 确认 FPS 段耗时与报告一致
- [x] F4. Scope fidelity：无重训、无其它算子改动、无 git 操作、无 batch>1/交叉编译越界

## Commit strategy
不执行任何 git 操作（AGENTS.md 约束）。所有改动留在工作区，由用户自行审查提交。

## Success criteria
1. 三选项并存：hpenet::FPS（原样）/ hpenet::SampleFPS（精确自研）/ hpenet::FlashFPS（Prune+Cache），fps_algo 一键切换
2. SampleFPS 两级判据通过：无平局输入 idx 逐索引一致（全 N 扫描 + fp32/fp16）+ 真实雷达帧每步全局 argmax 100% + 平局出现率量化入表
3. samplefps 路径与 cache-only 路径（flashfps+keep_rate=1.0，纯 Cache 无剪枝）端到端真实帧 logits 与 acc 完全等于现役（0.9741，平局归因豁免）；flashfps keep_rate=0.75 acc ≥ 0.9712 且 per-frame 无显著尾部
4. FPS 合计耗时 <5ms（基线 22.05ms）；flashfps 端到端 ≤26.53ms
5. 对比测试可重复（deploy/tests_fps_algos.py 一键）
