# GridBallQuery → deploy TRT 插件改造计划（草稿）

status: awaiting-approval
intent: clear
review_required: false
slug: gridballquery-trt-plugin

## Decisions（用户已确认 2026-08-22）
1. 落地形态：**新增可选档** `--bq_algo {ballquery,gridballquery}`（onnx_export.py + onnx_backend.patch_model_for_onnx），现役 BallQueryGroup/DP 保留为默认/回退
2. 数值验收：**容差+mIoU 门槛** —— 逐调用点 idx/dp 容差对拍 + 整网 max-abs-diff + radar 测试集 mIoU 降幅 ≤0.5%
3. nsys：**系统已装 Nsight Systems**（用户确认），无需安装；worker 首步用 `which nsys || ls /opt/nvidia/nsys-system` 定位
4. 基准精度：**FP16 + FP32 都测**

## 下一步
- ~~用户批准后写计划~~ ✅ 计划已写入 `.omo/plans/gridballquery-trt-plugin.md`（含 Metis 修订：B1 kernel 复制而非链接 / M1 严格< / M2 桶内消歧 / M3 profile 极值定尺寸 / M4 空球行为 / M5 性能前提风险 + 7 处 minor）

## Review rounds

### r1-20260822-gridballquery（已终结：plan 变更作废）
- momus-l1 (bg_80fe4e67): **APPROVED**（4 minor 观察：hash.h 路径缩写/ONNX 序列化确定性/F4 git status 措辞/mIoU 口径注明）
- oracle-l1 (bg_b7284dcc): **CHANGES_REQUESTED** — MAJOR×2（①"voxel_size≤radius 校验已在 plugin 层"为不实陈述且 27 格硬编码有 fp 边界隐患 → 改公式化扫描范围 R=ceil(radius/v)+1 + 构造期 clamp；②T=32768 单块 scan 的 smem 放不下（128KB>99KB），naive 实现编译即败 → 锁定寄存器驻留方案）+ MINOR×5（heap 弹出序需升序插入排序/d2 需 __fmul_rn 禁 FMA/T 须公式非字面/e2e diff 需剔除空球行并以 mIoU 仲裁/路径订正）
- 核心论断全部经 Oracle 实码核实为真（现役 kernel 语义/匿名命名空间/工厂换名不增节点/profile 极值）
- **全部 findings 已修入计划**（2026-08-22）

### r2-20260822-gridballquery（已终结：plan 变更作废）
- momus-l2 (bg_d5fc74c6, sha 613bde94…): **APPROVED**（minor：成功准则 6/7 对抗用例计数口径——已修）
- oracle-l2 (bg_473e9efb): **CHANGES_REQUESTED** — MAJOR×1（**`__fmul_rn` 禁 FMA 的"bit 级对齐保证"不成立且方向可能反**：现役 :25 用普通运算符 + nvcc 默认 --fmad=true，大概率为 FMA 收缩 SASS；单边禁收缩制造 ulp 级错位。正解 = 逐字复制表达式，同参数同 .so 构造上 bit 级一致）+ MINOR×3（覆盖证明句内部错误 |c_p−c_q|<2R→≤R-1 不成立，正解 ≤|p−q|/v+1<R / clamp 须两个构造路径都执行且序列化存原值 / scan 须写明 exclusive 且 elems-per-thread 按 T 推导）
- r1 全部 6 项修复被两 lane 确认正确落地、无新矛盾
- **全部 findings 已修入计划**（2026-08-22）

### r3-20260822-gridballquery（已终结：双双无条件批准 ✅）
- momus-l3 (bg_f7f2da05): **APPROVED**（sha256 `3a1c5dbf48af1888e6426e9eb4f665c313d90e059e1799d40753c113620bc419`；minor：ball_query_gpu.cu:41-44 系上游出处引用、radius2 行号 :15 微偏——内容均可定位，不阻断）
- oracle-l3 (bg_9350742a): **APPROVED**（同 sha `3a1c5dbf…`，两 lane 对同一活文件实算一致；minor×2：①"构造上 bit 级一致"措辞偏强——FMA 收缩是启发式，但计划自身已消化（诊断项非门槛 + 精确球面点对抗用例实证 + 容差验收），无需改计划 ②双处 clamp 幂等冗余属有意双保险）
- r2 全部 5 项修复经两 lane 实码核验确认正确落地；覆盖证明数学、exclusive scan 规格、双构造 clamp 均成立；零判断终扫通过
- **双评审通过，计划交付完毕（2026-08-22）**

### post-r3 平台/依赖审计增补（用户要求，2026-08-22）
用户要求对照 plugin.md 平台/依赖矩阵审计计划。发现并修入 3 处：
1. **（真实漏洞）** 任务 5 `eval_gridballquery_miou.py` / 任务 6 `bench_gridballquery.py` 未写明须先 `ctypes.CDLL(libhpenet_plugins.so, RTLD_GLOBAL)` 再反序列化 engine（plugin.md §6/L151/161/L1908 硬前提，否则 "Could not find plugin"）——两任务已补，模式引用 `trt_utils.py:80-82` / `tests_fps_algos.py:59-63`
2. **（构建接线明确化）** 任务 2 补明新文件追加进**现有共享 `add_library` 目标**（继承 arch 默认 "87;89"/TensorRT 链接），acceptance 注明 L20=sm_89 用默认 arch
3. **（文档/引用）** 任务 7 补跨平台声明（grid 插件不用 kFLOOR_DIV→无 Orin 8.5 新风险；CUDA 11.4 原语兼容；Orin/Win 重编为部署期步骤与 v14 遗留口径一致）；任务 1 sm_86 引用改为 sm_86/87/89 ~100KB 级
核对通过项：TRT 8.6 IPluginV2DynamicExt 签名/协变（8.5 兼容）、动态 shape 走 optimization profile 非 dim_param（与 workspace 极值方案一致）、cuDNN 仅 C++ 侧无关、无新 python 依赖、CMake 三平台流程沿用既有 CMakeLists。
增补均为文档级/接线级文字（不改架构与验收门槛）。

### r4-20260822-gridballquery（已终结：双双无条件批准 ✅ 增补确认完毕）
- momus-l4 (bg_ff24fa6b): **APPROVED**（sha256 `994a5227182102036934bd16ae568b9d74652d5e0ec028e22ec96cacd1ba1103`；三增补全部在场、引用行号实测吻合、无回归；minor：§6/L1908 引用并置略松散、plugin.md 行号漂移提示）
- oracle-l4 (bg_a62452b4): **APPROVED**（同 sha `994a5227…`；逐项对抗核验：①单次 CDLL 注册全部 11 creator（plugin_registry.cpp:12-20 单 .so）覆盖两 engine 对照 ✓ RTLD_GLOBAL 合惯例 ✓ 双加载无害（dlopen 语义）②add_library 追加完整（include/ 已在 :32 覆盖，头文件无需列源）③无 kFLOOR_DIV 属实（镜像现役 :50-53/:177-184 模式，plugin.md:1266 佐证）原语 CUDA 11.4 兼容 ✓ smem 99KB 表述准确 ✓ 部署期重编口径有据（plugin.md:1700/:1409）④r3 核心内容无回归）
- **r4 通过 = post-r3 三增补确认完毕，计划最终交付（2026-08-22）**

## 探索结论（事实）

### 现役 ballquery 链路
- ONNX 导出（`deploy/onnx_backend.py::_patch_group_plugins`）：
  - 4 个 SA 调用点 → `hpenet::BallQueryGroup(xyz,new_xyz,features)→(grouped,dp)`
  - 4 个 InvResMLP 调用点 → `hpenet::BallQueryDP(xyz,new_xyz)→(dp,idx)`
  - 独立 `hpenet::BallQuery` 仅作兜底（group_mod.ball_query=ballquery_op，无害保留）
- 插件侧 `deploy/trt_plugins/src/`：
  - `ballquery_kernel.cu` = 现役 O(M·N) 暴力 ball query（复刻 ball_query_gpu.cu，**空槽填第一个邻居** :41-44）
  - `ballquerygroup_kernel.cu` = ball_query_launcher + bq_dp_kernel(dp=(xyz[k]-q)/radius) + bq_gather_kernel(特征 gather, fp16/fp32 模板)
  - `plugin_registry.cpp` 注册 9 个 plugin（FPS/BallQuery/BallQueryGroup/BallQueryDP/ThreeInterp/SampleFPS/FlashFPS/PrefixFPS/FPSPrune）
- engine 构建：`trt_build.py` 动态 profile（min2024/opt4096/max10000），CDLL 加载 libhpenet_plugins.so
- 测试基建：`deploy/tests_fps_algos.py` 已有 静态shape逐N建engine(~0.1s) + cuda.Event 计时（time_engine）harness
- 精度/构建现状：v14 后 628 节点；图节点数验收 ≤1100

### GridBallQuery（FPS 仓库）事实
- `FPS/.../gridBallQueryPlugin/gridBallQueryPlugin.cu`：IPluginV2DynamicExt，plugin 名 "GridBallQuery"
- **5 输入**：source_xyz[B,N,3], query_xyz[B,M,3], voxel_hash_table int32[B,T], subset_ind_table(slots2queries) int32[B,T], voxel尺寸 float32[B,3*3]；**2 输出**：cnt[B,M], indices[B,M,S]
- hash 表由 **HAVSForQuery 插件**（havSamplingForQueryPlugin）在前置算子构建——deploy 无此算子
- 复杂度：对每个 source 点搜 (2·step+1)³ 体素格，step=ceil(radius/voxel)-1
- 语义差异（关键）：
  1. hash 建在 **query 点**上、每 voxel 单 query（slots2queries 单映射）——HPENet 的质心来自 FPS，同 voxel 多质心会丢邻居 → **不能直接用**
  2. 迭代方向：source 驱动 + atomicAdd → 邻居顺序非确定
  3. pad_indices 用重复首邻居填充（与现役 kernel 语义巧合一致，但顺序不同）
- hpenet-ll 参数：radius=10（scalar，encoder 内 `_to_full_list(radius, radius_scaling=2)` 逐级展开），nsample=32，N 动态 2024~10000

### 可行性判定：✅ 可行，但不能照搬
- GridBallQuery 本身就是 IPluginV2DynamicExt TRT 插件（TRT 8.6 兼容），授权 Apache-2.0 可复用 kernel 思路
- 正确改造 = 新建 **GridBallQueryGroup / GridBallQueryDP**（与现役插件 ONNX 签名完全一致），只把邻居搜索 kernel 换成 grid 法；hash 表构建**内置在 enqueue**（plugin workspace），不引入任何新 ONNX 节点 → 算子碎片化不增反与现役持平
- 语义修正：hash 建在 **source 点**上（多源点/格需 chaining 或 counting-sort），query 驱动确定性格扫描；空槽填首个找到的邻居（复刻现役 padding 规则）
- 邻居集合与顺序必然与现役不同 → 数值验收必须容差制 + mIoU，不可能 bit 级一致
- voxel 尺寸策略：默认 voxel=radius（27 格/查询），做成插件属性可调
- nsys：常见路径未找到，需 worker 验证/安装（pip nsys-system）

## 待问分叉（owner 决策）
1. 落地形态：新增可选档（--bq_algo，保留现役回退）vs 直接替换现役
2. 数值验收标准：容差 + mIoU 门槛（建议降幅 ≤0.5%）
3. nsys 缺失时是否允许安装（pip 装进 hpenet env）
4. 基准精度：FP16（现役工作流）/ FP32 / 两者

## Decisions
- (待用户回答后填)
