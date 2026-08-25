---
slug: fps-samplefps-flashfps
status: drafting
intent: clear
review_required: false
pending-action: collect deploy anchors (bg_151f53c6) → approval brief
approach: 双组件替换现有 hpenet::FPS 插件：(A) SampleFPS 自研精确 GPU kernel（kdline/桶结构，参考 fpsample/_ext，单 kernel 内迭代、确定性小索引 tie-break、B=1、fp32 xyz→int32 idx）；(B) FlashFPS 机制移植 = 插件属性 prune_rate（默认 0=精确，0.75=近似剪枝+顺序填充）+ 图级 FPS-Cache（第 2-4 级 FPS 节点替换为前缀切片，基于前缀等价性：深层 FPS 输出=第 1 级序的前缀）
---

# Draft: fps-samplefps-flashfps

## Components (topology ledger)
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
| C1 | SampleFPS ONNX 算子 + TRT 插件（精确 kernel，逐索引等价现役） | active | 待 bg_151f53c6 锚点 |
| C2 | FlashFPS Prune 选项（prune_rate 属性，候选/迭代双重剪枝+填充） | active | FlashFPS subsample.py:127-144 语义 |
| C3 | FPS-Cache 图级替换（第 2-4 级 hpenet::FPS → 前缀切片/arange） | active | 前缀等价性已验证（FlashFPS paper §3.3 + 本会话核对张量流） |
| C4 | 单算子对拍 + 端到端验收（acc 一致性、nsys 延迟、fp16） | active | 沿用 V1/v2_network.py 模式 |

## Open assumptions (announced defaults)
<!-- assumption | adopted default | rationale | reversible? -->
| Prune 默认值 | prune_rate=0（精确模式默认；0.75 作为可选开关） | 精度风险自选，等价替换优先 | 可逆（插件属性） |
| tie-break | 确定性小索引优先（与现役 pointnet2 kernel 一致） | 保证与现役插件逐索引对拍可过 | 可逆 |
| batch | B=1（与现管线一致） | 部署画像固定 batch=1 | 后续可扩 |
| SampleFPS kernel 形态 | 单 kernel 完成全部 K 轮迭代（无逐迭代 launch），桶/桶表驻 shared+global，host 零循环零同步 | torch-fps 与 QuickFPS-GPU 的教训：launch 开销是主要坑 | 可逆 |
| 精度 | xyz 锁 fp32、idx int32（对齐现有插件约定） | xyz 精度锁死 fp32 已有结论（plugin.md 附录C.2） | 不可逆约束 |
| 旧 hpenet::FPS 插件 | 保留注册不删（回退通道） | A/B 对拍需要 | 可逆 |

## Findings (cited - path:lines)
- 现画像：4 级 FPS（5500→2750→1375→687→343，各级采 50%），FPS 占端到端 82%（22.05ms/26.77ms，nsys）
- 前缀性质成立条件核验：hpenetv2.py:160 idx=sample_fn(p, p.shape[1]//stride)，pointnet2 kernel 种子 old=0，各级输入=上级 FPS 输出原序 → 第 ℓ 级输出 = 第 1 级序的前 ℓ 级截断（本会话核对）
- fpsample kdline CPU L1=2.39ms 证明桶结构在小 N 高效；GPU QuickFPS 慢因 3-launch/迭代+建树 31ms（本会话实测）
- FlashFPS 论文：Prune 3.47×/Cache 1.35×，p=0.75 端到端 5.16×，精度损失<0.3%
- fpsample bucket 系 start_idx 不生效的坑（首点随机）→ SampleFPS 必须显式支持固定种子
- **deploy 锚点（bg_151f53c6 已核实）**：
  - 现役插件：trt_plugins/src/fps_plugin.cpp（FPSPlugin:IPluginV2DynamicExt L12，type "FPS" L81，属性仅 stride L106；enqueue L56-71=fill_kernel+fps_launcher_with_stream）+ src/fps_kernel.cu（经典单 block 串行 FPS L13-65）
  - 新增插件 3 处改动：CMakeLists.txt:14-24 源清单、plugin_registry.cpp include+REGISTER_TENSORRT_PLUGIN（静态初始化自动注册，Python ctypes.CDLL trt_build.py:52-56/trt_utils.py:79-82、C++ initLibHPENetPlugins trt_engine.cpp:52 加载侧零改动）
  - ONNX 侧：onnx_ops/fps_op.py（hpenet::FPS, stride_i）；onnx_backend.py:212 sample_fn 替换点、L317 emit 点；stride 终源 cfgs/radar/hpenet-ll.yaml:11 [1,2,2,2,2]→图中 4 个 FPS 节点均 stride=2
  - idx 下游仅 2 处 gather（onnx_backend.py:318 pos、L323-324 feature）；group_fn(p,new_p,f) 消费 new_p 不消费 idx → Cache 替换第 2-4 级 FPS 节点为 Slice/arange 时适配面极小
  - 验证资产：/tmp/opencode/v1_plugins.py（V1 单算子框架，含属性名无 _f/_i、engine dtype 绑定等坑的规避写法）；/tmp/opencode/verify_fps.py（FPS 序列精确性判定）；deploy/CPP_trt/tests gtest 模式；plugin.md:1374 V1 判定标准（fp32≤1e-5/idx bit 一致/fp16≤1e-3）
  - **风险澄清**：plugin.md §1.2[C] acc 0.93→0.66 是"原始输入取前N、完全不做FPS"；FlashFPS Cache 是"第1级精确FPS后取其输出序前缀"，前缀等价性保证精确，二者不同，风险不可照搬。计划中 Cache 验收=与 SampleFPS-only 精确路径 logits 一致
  - FlashFPS 仓库 FPScache.py 为空文件，不可直接搬代码；Prune 语义以 FlashFPS-Openpoints/openpoints/models/layers/subsample.py:127-144（FPS_Prune+rearrange_indices，本会话已完整审读）为准做语义移植

## Decisions (with rationale)
- 架构：新算子 hpenet::SampleFPS（attrs: npoint/stride, prune_rate, seed）+ 保留 hpenet::FPS；Cache 以图改造实现（onnx_backend patch 层），不写有状态插件（TRT 插件跨层共享 buffer 需 workspace hack，图级 slice/arange 更干净）
- SampleFPS kernel 算法 = kdline 桶结构 GPU 化（fpsample/_ext/KDLineTree.h 为伪代码级蓝本），不用 QuickFPS-GPU 的 devide 归并排序建树（重、31ms），改单 kernel 轻量建桶

## Scope IN
- SampleFPS CUDA kernel + TRT 插件（IPluginV2DynamicExt 对齐现有插件代次）+ ONNX 算子 + onnx_backend patch 开关
- FlashFPS Prune（候选剪枝=前缀切片子集+迭代剪枝+顺序填充）作为 kernel 内路径
- FPS-Cache 图级替换（第 2-4 级）
- 单算子对拍脚本（vs 现役 FPS 插件 + vs torch 参考实现）+ 端到端验收（acc/延迟/fp16）
- nsys 延迟剖析对比

## Scope OUT (Must NOT have)
- 不重训/微调模型（权重不动，acc 必须守门）
- 不改 BallQuery/ThreeInterp 等其它算子
- 不删旧 hpenet::FPS 插件与旧 engine 的兼容性
- 不做 batch>1、不做 Orin/Windows 交叉编译（沿用后续部署阶段）
- 不引入 QuickFPS/HAVSampler/FlashFPS 仓库的任何构建依赖（仅算法语义移植）

## Open questions
（无 owner 级分叉——prune 默认关、等价优先等默认已记录于上表，用户可在审批时否决）

## Approval gate
status: approved-with-amendment; plan written
approach: 见 front-matter。四个组件：C1 SampleFPS 插件（逐索引等价现役）→ C2 Prune 属性（默认关）→ C3 Cache 图级替换（第2-4级FPS→前缀切片，等价性验收=logits与C1路径一致）→ C4 验收（单算子bit级对拍+acc守门0.9741-0.005+nsys延迟）。
验收目标：FPS合计 22ms→<5ms；Cache 路径 logits 与精确路径一致；Prune(p=0.75) acc≥0.9712；端到端 ≤26.53ms 基线不劣化（目标 ~7-8ms）。
下一步：用户 okay → 写 .omo/plans/fps-samplefps-flashfps.md（含分批 todos）→ 询问是否要高精度评审（review_required=false）

## High-accuracy review receipts (2026-08-18)
- Momus (bg_30a3b7b0): APPROVE-WITH-FIXES — 必修1项（prune_rate 语义与 FlashFPS PruneRate 互为补数）+ 3 轻微（依赖矩阵、FPScache.py 不存在而非空文件、22.05ms/82.4% 出处应为 nsys trace）
- Oracle (bg_9abd4867): APPROVE-WITH-FIXES — 阻断级3项：①Cache 常量 Slice 在动态 N 部署下形状错误→PrefixFPS 插件；②bit 级一致与提速目标结构互斥（现役 tie-break=min(k mod block_size)→min(k)，桶结构无法复刻）→两级判据；③双 creator 共类反序列化失败→两独立类
- 全部 6 项修正已落入计划 v2（keep_rate 语义、PrefixFPS、两级判据+平局量化实验、独立插件类、依赖矩阵、引用修正）
- 轻微项同样已修：FPScache.py 描述、rearrange_indices 行号(:98)、基线数字出处、fi 死代码说明、kernel 两阶段 CSR 设计、per-frame acc 监控、<5ms 目标改实测口径

## Round-2 review receipts (2026-08-18)
- Momus二审 (bg_fb90c2ec): APPROVE-WITH-FIXES — B1: flashfps 首节点类型矛盾（task3 SampleFPS vs task4/scope FlashFPS）+ 并行冲突；C1: Verification strategy line46 残留"bit 级一致"。6 项一审修订中 5 项干净落地、引用全部有效。
- Oracle二审 (bg_e16dbdce): APPROVE-WITH-FIXES — PrefixFPS 数学闭环验证通过（链式 floor 恒等式/arange→gather/keep_rate≥0.5 前缀落精确段）；必修：PrefixFPS forward=arange+workspace=0、flashfps 首节点=FlashFPS、参考实现 float32 同表达式 eps=0+坐标去重、num_points 取整整照参考公式（奇数 N 差1）、cache 语义对拍；文档级：keep_rate<0.5 下限、有序vs无平局措辞、SampleFPS 去 keep_rate。
- 全部落入计划 v3（13 处编辑）；二审双审均确认架构无需改动。

## Round-3 review receipts (2026-08-18)
- Momus三审 (bg_c70c1206): APPROVE-WITH-FIXES（3 非阻塞文案）：task 6 "三图（task 3 产物）"旧指针、升序/顺序术语分叉、"FPScache.py 不存在（非空文件）"自相矛盾、§13.6→§7.1 章节号。v3 13 处编辑全部正确落地（num_points 公式经独立数学复核确认）。
- Oracle三审 (bg_19bf8028): APPROVE-WITH-FIXES：(1) 依赖矩阵漏标 3→4（flashfps wiring 复用 PrefixFPS；task 4 Wave2 完成早于 task 3 Wave2-3 的矛盾；task 3 验收"三图"与"不实现 flashfps"冲突）；(2) onnx 节点 diff 与 PrefixFPS 动态形状两条验收缺机械工具/运行时断言；(3) task 1 缺幸存桶 compaction 机制与共享内存桶上界容量约束（O(N) 桶 40KB 逼近 48KB）；(4) task 5 Wave1 验收不可判定（deferred）；registry/CMake 三方共编需串行。
- 全部落入计划 v4（11 处编辑，1 处 MISS 为已被前一处替换覆盖的重复表述，无影响）。
