# 去掉体素化 — 调查与改动方案

> **目标**：radar 训练 / 验证 / 测试全链路去掉体素化机制（去重 + 子云投票），并删除部署侧 `deploy/CPP_trt4` 的 C++ 体素化，降低前处理冗余与推理延迟。
> **原则**：**Python 侧尽量保持代码不动**（A2 用参数绕过）；C++ 侧 `Voxelizer` 是无条件调用、`float voxel_size_` 无 `None` 语义 → **无法绕过，必须删代码**。
> **状态**：方案已定稿。⚠️ **执行前必读 §4 顶部的 2 个 blocker。**

---

## 1. 背景与结论

### 1.1 参数与当前值

| 参数 | 位置 | 作用 | 当前值 |
|---|---|---|---|
| `voxel_size` | `dataset.common` | 体素边长。控制去重粒度与测试子云数；`voxelize()` = floor + FNV hash + argsort + unique 取随机点 | **0.0001**（`cfgs/radar/default.yaml:7`） |
| `voxel_max` | `train`/`val`/`test` | 裁剪/补齐到固定点数，供 batch 等长。**与体素化机制无关**（在 `crop_pc` 的不同 `if` 块，L154 vs L149） | train/val **8000**（`:10`/`:15`）、test `null`（`:19`） |
| `presample` | 三段 | 是否预建体素化缓存 pkl | 全 `False` |
| `test_mode` | cfg 顶层 | 测试投票策略 | 未设（默认 `multi_voxel`） |

### 1.2 为什么 radar 不需要体素化

| 事实 | 数值 |
|---|---|
| 单文件点数 | 3005–7837（中位 5845；p25 4557 / p75 6737 / max 7837） |
| 339 文件精确重复坐标 | **0** |
| 全库最小点对 Chebyshev 距离 | 3.2e-4 m（0.32 mm） |
| 当前 0.0001 口径 | **0 合并 / 单子云** |
| 引擎容量 | profile `max_n=10000` > 7837，单次前向绰绰有余 |

雷达点云稀疏、点数少 → 去重近乎无效、投票纯属多余。原论文的 `multi_voxel` 投票是为**数十万点大场景**设计的折中，radar 不适用。

### 1.3 为什么选 `0.0001`（`null` / `0` 均不可行）

| 值 | train/val | test | init f-string | 结论 |
|---|---|---|---|---|
| `null` | `crop_pc` 正常跳过 | 支持（全量单次推理） | 🔴 **崩** `TypeError: {None:.3f}` —— **仅 A1 修复前**；A1 落地后即可用，**A2 采用的就是 `null`** | ✅（A1 后） |
| `0` | `crop_pc` 跳过 | 🔴 **不崩但静默灾难** | 正常 `"0.000"` | ❌ |
| **`0.0001`** | 走 `voxelize` 但零合并 | `count.max()==1` 单次推理 | 正常 `"0.000"` | ✅ |

- **`null` 崩点**：`RadarClassi.__init__`（`radar/s3disRadar.py:73-74`）无条件构造 `f'radar_{split}_area{test_area}_{voxel_size:.3f}_{str(voxel_max)}.pkl'` → `f"{None:.3f}"` 抛 `TypeError`。`presample=False` 也躲不过（filename 构造在 `if presample` 之前），且 `main.py:191` 在 mode 分支前无条件建 `val_loader` → **train/val/test 三模式都会撞**。
- **`0` 不崩但更危险**：numpy 除零只给 `inf` + RuntimeWarning、**不抛异常**；`coord/0 → inf → floor → astype(uint64)` 全部塌缩（`astype` 对 `inf` 的行为是实现相关的）。实测 `count.max()` 与 N **同量级**（5119 点文件实测 2510~5116）→ `multi_voxel` **静默跑 O(N) 次前向**。
- **为什么是 1e-4**：实测 339 文件逐档验证——

| voxel_size | 有多子云的文件 | 最大子云数 | 总合并点数 |
|---|---|---|---|
| 0.02 | 250 | 3 | 523（0.027%） |
| 0.008 | 45 | 3 | 55（0.003%） |
| 0.001 | 1 | 2 | 1 |
| **0.0001** | **0** | **1** | **0** |
| 1e-5 / 1e-6 / 1e-7 | 0 | 1 | 0 |

  更小（1e-5/1e-6）无额外收益且有 float32 风险：`coord/v` 在 float32 计算，`v=1e-6` 时商最大 6.08e8、`ul=64`，极远处可能因舍入误合并。`v=1e-4` 时商最大 6.08e6 < 2²³=8.39e6 → 每个商都是 float32 精确整数，floor 无抖动。

> **`voxel_max` 已同步落地**：`8000 > max N 7837` → **0 裁剪**（旧 4608 口径会裁掉 252/339 文件、中位丢 21%）。补齐是随机重复采样 + shuffle，不改变分布、不偏置模型。
> 备选：未来若出现 > 8000 点的文件，调大 `voxel_max` 即可（它只是上限）。

### 1.4 BN：保留，不换 IN

| 归一化 | TRT 中的表现 |
|---|---|
| **BN**（当前 `norm: 'bn'`） | 被 **conv+BN fusion** 吸收进 conv 权重 → **engine 中完全消失，零开销** |
| IN | `InstanceNormalization` 无法融入 conv，作为独立算子保留 → 40–50 层约 +<0.3ms |

**结论**：保留 BN。部署侧 BN folding 是"免费"优化，不值得为训练端噪声去付出部署代价。

---

## 2. 关键代码位置

### 2.1 改动目标（本次会动的）

| 文件 | 行号 | 内容 |
|---|---|---|
| `cfgs/radar/default.yaml` | 7 | `voxel_size: 0.0001 → null`（A2） |
| `openpoints/dataset/radar/s3disRadar.py` | 73-74 | init f-string（`voxel_size=None` 崩溃点，A1） |
| `examples/segmentation/main.py` | 139-140 | `load_data` 的 `else: arange(N)` 死分支（A3，补 shuffle） |
| `deploy/common.py` | 71 | `preprocess_test` 的 `else: arange(N)` 死分支（A3，补 shuffle） |
| `deploy/onnx_inference.py` | 267 | `float(cfg.dataset.common.voxel_size)`（A3，改为 `.get()`） |
| `examples/segmentation/main.py` | 703 | `test()` 的 scatter（B1） |
| `deploy/trt_inference.py` | 88 / 113 / 141 | 3 个 scatter 站点（B2） |
| `deploy/onnx_inference.py` | 120 / 162 | 2 个 scatter 站点（B2） |
| `deploy/CPP_trt4/src/pipeline.cpp` | 见 §4 方案 C | 体素调用点 + `voxel_size` 全链路 |
| `deploy/CPP_trt4/src/main.cpp` | 31 / 46 / 73-74 / 123 / 143 / 185 | `voxel_size` 默认值 / 注释 / help / 解析 / 构造调用 |
| `deploy/CPP_trt4/src/trt_inference_wrapper.cpp` | 192-194 | C-API 硬编码 `0.02f`（C6，**blocker**） |
| `deploy/CPP_trt4/include/pipeline.h` | 43 / 51 / 106 | `voxel_size` 注释 / 形参 / 成员 |
| `deploy/CPP_trt4/include/types.h` | 37 | 死代码 `Config::voxel_size` |
| `deploy/CPP_trt4/CMakeLists.txt` | 64-65 | 待移除编译项 `src/voxelizer.cu`、`src/fnv_hash.cu` |
| `deploy/CPP_trt4/src/voxelizer.cu`、`include/voxelizer.h`、`src/fnv_hash.cu`、`include/fnv_hash.h` | — | 待删文件 |
| `deploy/measure_orin.sh` | 19-20 | 仍建议 `--voxel_size=0.02` → C5 后该 flag 不存在，须同步改注释 |

### 2.2 只读参考（本次**不动**）

| 位置 | 说明 |
|---|---|
| `openpoints/dataset/data_util.py:127-143` | `voxelize()` —— **保留**（`crop_pc` 共享，见 §4 方案 D） |
| `openpoints/dataset/data_util.py:149-153` | `crop_pc` 的体素分支 —— **保留** |
| `openpoints/dataset/data_util.py:154-173` | `crop_pc` 的 `voxel_max` 块（含 :169-171 shuffle）—— 保留 |
| `deploy/CPP_trt4/include/random_util.h`（`:21` explicit ctor、`:27` `shuffle(int*,int)`）+ `deploy/CPP_trt4/src/random_util.cpp` | `NumpyMT19937`，**必须保留**（新路径 shuffle 依赖）。⚠️ **头文件在 `include/` 而不是 `src/`** |
| `deploy/CPP_trt4/include/preprocessor.h:25` | `preprocess_subcloud(const int* idx_part)` —— 决定索引类型 |
| `deploy/CPP_trt4/include/pipeline.h:114` | `std::vector<int64_t> idx_staging_` |
| `deploy/CPP_trt4/include/voxelizer.h:30` ⚠️ | `VoxelizeResult.idx_points` 为 `std::vector<std::vector<int>>` —— **该文件本身在 §4 C4 删除**；此行仅用于查证 `idx_points` 的类型，**不是"保留不动"** |
| `openpoints/cpp/pointnet2_batch/src/sampling_gpu.cu:120-122` | FPS 首中心点（Python 侧扩展，非 `CPP_trt4`） |
| `openpoints/models/layers/norm.py:74-115` | norm 注册表（bn/in） |

### 2.3 独立工程 / 范围外（显式排除）

| 位置 | 说明 |
|---|---|
| `deploy/CPP_trt1`、`CPP_trt2`、**`CPP_trt3`** | 同含完整 `voxelizer`/`fnv_hash`/`scatter_mean`/`random_util`（`CPP_trt3` 是 `CPP_trt4` 的复制蓝本，**保持原样不动**）—— **独立工程**，本方案只动 `CPP_trt4` |
| `deploy/CPP_onnx`、`CPP_onnx1`（C++ 侧） | 各含完整 **CPU** 体素化（`voxelize_cpu` + `fnv_hash_coord`；`CPP_onnx/onnx_inference.cpp:186-187`、`CPP_onnx1/onnx_inference.cpp:222-223`） |
| `deploy/CPP_trt/tests/{test_voxelize,test_fnv_hash,test_scatter_mean}.cu`、`scripts/gen_golden_data.py:26` | 引用被删符号，但属 `CPP_trt`（另一工程），不影响 `CPP_trt4` 构建 |
| `deploy/trt_manual/{trt,onnx}_inference.py` | scatter 站点在 `trt_inference.py:65/90`、`onnx_inference.py:120/162`；另有同病行 `onnx_inference.py:247` 的 `float(...)` |
| `deploy/CPP_onnx/verify.py:94` | scatter 站点 |
| `examples/segmentation/main_debug.py`、`debug_compare_val_test.py` | `main.py` 同构副本（`main_debug.py:108` arange 死代码、`:490`/`:622` scatter；`debug_compare_val_test.py:122` import、`:125` `def load_data_radar`） |
| `FPS/FlashFPS*` | 仓库内 vendored openpoints 副本，含 `voxelize` import；A2 只改 radar 配置，不影响它 |
| `openpoints/dataset/radar/s3disRadar_sphere.py`（`S3DISSphereRadar`，由 `openpoints/dataset/radar/__init__.py:1` 注册） | 含**同型** `{voxel_size:.3f}`（`:93` / `:151` / `:205` / `:257`）→ 理论上 A2 后也会崩；但 **`cfgs/` 对它零引用**（radar 只写 `NAME: RadarClassi`）→ **不在 radar train/val/test 路径**，故 A1 不覆盖它 |
| `openpoints/dataset/datalist.py`（`:38-49`，`else: arange` 在 `:48`）、`deploy/eval_gridballquery_miou.py`、`deploy/onnx_backend.py:467/536`（+ `onnx_export.py:257`、`tests_gridballquery.py:94`） | 良性未覆盖项：① `datalist.py` 是**零 import 的死文件**（第 5 处可参数绕过分支，但无人调用）；② `eval_gridballquery_miou.py` 是自带 `--voxel_size 0.3` 的独立实验脚本；③ `onnx_backend` / `onnx_export` / `tests_gridballquery` 的 `voxel_size=-1.0` 是 **GridBallQuery 网格参数**，与数据体素化无关 |
| `openpoints/models/backbone/pointnetv2.py:39`、`cfgs/s3dis_pix4point/default.yaml`、`deploy/analyze_latency.py:79/88` | 另 3 处 `voxel_size` 出现点，**均良性**：① `pointnetv2.py` 的是**模型超参**（VoxelNet 式体素编码），非数据体素化；② `s3dis_pix4point` 属另一任务配置（A2 只改 radar）；③ `analyze_latency.py` 的 `'VOXELIZE'` 只是 NVTX 名单，`text IN (...)` 与 `if name in seg` 对缺失段**容错**（§5.2 已把它列为延迟验收工具） |
| `deploy/CPP_trt4/{src/test.c,src/test_rpc.c}` | 未纳入 CMake 编译；`test.c:132` 仅一处 `voxel_size` 注释 |
| `deploy/CPP_trt4/include/trt_inference_wrapper.h` | **无 `voxel_size` 形参** → C6 只需改 `.cpp` |
| `trt_plugins` 的 ball_query 网格参数 | 与数据体素化无关，**不要动** |

---

## 3. 去掉体素化的依据与硬约束

### 3.1 收益来源（实测，L20 / 40 文件 / N mean 5516）

**`crop_pc` 内部拆解**：`np.argsort` 0.225（**47%**）、gather×3 0.099、`np.unique` 0.066、`np.random.randint`+cumsum 0.049、`fnv_hash_vec` 0.032、`np.floor` 0.006 → **voxelize 合计 0.475 ms**

| 路径 | 可去除 |
|---|---|
| train/val `crop_pc` 端到端 | `0.0001` = 1.214 ms → `None` = 0.454 ms → **−0.760 ms（−63%）** |
| test（voxelize + 子云构建 + fancy-index） | ≈ **0.62 ms** |
| test 后处理 `scatter_mean`（GPU N=5845） | 0.098 ms（替代写法 0.020 ms） |
| 部署 VOXELIZE（引自 `latency-statistics.md` NVTX） | L20 **0.51 ms**；**Orin 1.85 ms** |

> ⚠️ **口径**：上表 `0.51 / 1.85 ms` 与 `6.4 / 25.66 ms` 取自 **voxel 0.02** 口径（`latency-statistics.md:4`；0.02 下 250/339 文件多子云），而 `main.cpp:31` 已是 `0.0001f`。绝对量 −1.85ms **可迁移**（hash+sort 是 O(N log N)），但百分比分母偏大 → 0.0001 基线下实际 **≥ 7.2%**；若在 0.02 口径下删体素还会额外坍缩 2→1 子云（省整次 enqueue，Orin ≈8ms）→ 收益远大于 1.85ms。

叠加 TAIL 内 scatter 段（估 0.5–1ms）→ **Orin 合计 −2.4~2.9 ms ≈ −9~11%**（0.0001 口径）。

### 3.2 无体素分支已存在，但**不能直接复用**（必须补 shuffle）

上游 PointNeXt 本就按「可选体素化」设计，框架里已有 4 处可被参数绕过的分支：

| 位置 | 代码 | `voxel_size=None` 时 |
|---|---|---|
| `data_util.py:149` | `if voxel_size and downsample:` | 整块跳过 ✓ |
| `s3disRadar.py:87` | `if voxel_size:`（presample 分支） | 跳过 ✓ |
| `main.py:139-140` | `else: idx_points.append(np.arange(label.shape[0]))` | 命中 ⚠️ 死代码 |
| `deploy/common.py:71` | `else: idx_points.append(np.arange(coord.shape[0]))` | 命中 ⚠️ 死代码 |

> 🔴 **后两处的 `arange(N)` 不能直接复用** —— 它产出**恒等排列、无 shuffle**，会让 FPS 永远从第 0 个点起步（见 §3.3）→ **必须补一次均匀随机 shuffle**。

### 3.3 为什么必须保留 shuffle

**FPS 的起始中心点 = 输入点序的第 0 个点**：`sampling_gpu.cu:120-122` `int old = 0; if (threadIdx.x == 0) idxs[0] = old;`。（注意 `:126 int besti = 0` 是**每轮搜索的初值**，会被 `:143` 覆盖，**不是**首中心点。）→ **点序决定 FPS 中心序列 → 决定输出。**

现状（`voxel_size=0.0001`，测试路径）：

```
voxelize → idx_sort（按 FNV hash 排序，确定性）
         → idx_part = idx_sort[idx_select]（每体素一点 = 全部点）
         → np.random.shuffle(idx_part)      ← main.py:137，均匀随机排列
```

最终排列 = **均匀随机排列**。若改成恒等排列：点集不变，但 FPS 永远从 PLY 第 0 点起步；PLY 按雷达扫描线序写入 → 第 0 点固定同一方向 → **FPS 子采样出现系统性偏向（非随机噪声）**。

**正确做法**：在无体素路径保留一次均匀随机 shuffle（沿用同一 seed 流）→ 点集相同、**排列分布相同** → 结果分布与现状一致，acc 对拍才是同口径。

**C++ 侧对应实现**（`voxelizer.cu:161-171` 用单个 `NumpyMT19937 rng(seed)` 做 Fisher-Yates；RNG 只被 shuffle 消耗，`idx_select` 是确定性 cumsum）：

```cpp
std::vector<int> idx(N);          // N = 点数：process_pointcloud 用 num_points，process_file 用 pc.num_points
std::iota(idx.begin(), idx.end(), 0);
NumpyMT19937 rng(seed_);
rng.shuffle(idx.data(), N);       // 与现状的 RNG 抽样序列一致
std::vector<std::vector<int>> idx_points{ std::move(idx) };
```

> ⚠️ **现役 shuffle 的长度是 `M`（体素数）不是 `N`** —— `voxelizer.cu:165` 定义 `idx_part` 尺寸为 `M`、`:171` 调 `shuffle(idx_part.data(), M)`。仅当 `count.max()==1`（无合并）时 `M == N`，即本方案的目标配置。

### 3.4 非 bit 等价 → 必须按 acc 验收

去掉 voxelize 会改变 RNG 消耗序列与点序 → FPS 起点变 → 输出变。但**点集完全相同**（1e-4 实测 0 合并），故**统计等价**。

`latency-statistics.md` §六「精度红线」要求 acc 0.9578 **或** pred bit 级一致 —— 本方案只能走前者，判据沿用 **Δacc < 0.3pp**。

**另两处必须知道的口径**：

- **索引类型**：`idx_points` / `idx_part` 是 **`int`**（`voxelizer.h:30`；消费端 `preprocessor.h:25` 为 `const int*`）；只有 `idx_staging_` / `d_idx` 是 **`int64`**（`pipeline.h:114`，转换在 `pipeline.cpp:200` / `:401`）。**不要把 `idx_points` 改成 int64** —— 会与 `preprocess_subcloud(const int*)` 不兼容。
- **gate 口径**：方案 C 落地后单子云由构造保证（`idx_points` 恒 1 元素）→ **无需运行期 gate**；仅当保留「未来回到 voxel 0.02 多子云」的回退路径时，才需写 `idx_points.size()==1` 并为 else 分支保留 mean 归约。本方案按「C 先落地 + B3 选 (b)」执行，**故不加 gate**。

### 3.5 seed 口径

| 路径 | shuffle seed |
|---|---|
| `main.py` `test()`（`:606 set_random_seed(0)`） | **0** |
| `deploy/common.py:62`（`np.random.seed(100)`） | **100** |
| `CPP_trt4/src/main.cpp:36`（`int seed = 100`） | **100** |
| `CPP_trt4/src/trt_inference_wrapper.cpp:192-194`（C-API 未传 seed，走默认） | **100** |

- **核心 Δacc（`null` vs `0.0001`）不需要统一 seed** —— 只需两次运行 seed 恒定。
- seed 统一只是**部署侧 acc 级对拍的卫生项**（acc 级本就不要求 bit 一致）。
- 但须知：统一 seed 会**改变被测数值**（shuffle → FPS 起点 → acc），这正是 §5 前置条件 1「重测 baseline」存在的理由。

---

## 4. 改动方案

> **部署侧改动只在 `deploy/CPP_trt4`**（已从 `CPP_trt3` 复制）。**不加 `--no_voxel` 开关，直接删除体素相关代码**；`CPP_trt3` 保持原样。

> 🔴 **执行前必读 —— 2 个 blocker**（编号 **BLK-1 / BLK-2**，以免与"方案 B 的步骤 B1/B2/B3"混淆）
> | # | Blocker | 后果 | 对策 |
> |---|---|---|---|
> | **BLK-1** | `CPP_trt4/src/trt_inference_wrapper.cpp:192-194` 硬编码 `0.02f` | C5 删掉构造函数 `voxel_size` 形参后，`0.02f` **静默落到 `seed`（int 截断 → 0）**，不报错不崩溃 | 步骤 **C6** |
> | **BLK-2** | `deploy/onnx_inference.py:267` 的 `float(cfg.dataset.common.voxel_size)` | A2 后变成 `float(None)` → **TypeError 硬崩**（已实测） | A3 第三处 |

### 方案 A —— Python 侧：参数绕过 + 3 处阻塞点修复

| # | 文件:行 | 改动 |
|---|---|---|
| **A1** | `openpoints/dataset/radar/s3disRadar.py:73-74` | f-string 条件化：`voxel_size is None` 时用**固定字面量 tag `novx`** 替代 `{:.3f}`。可直接写 `tag = 'novx' if voxel_size is None else f'{voxel_size:.3f}'`，再拼进 filename。（`presample: False` 顺带消除 `{:.3f}` 撞名隐患——`0.0001`~`0.0009` 都渲染成 `"0.000"`，若将来开 presample 会静默加载错缓存。） |
| **A2** | `cfgs/radar/default.yaml:7` | `0.0001 → null` |
| **A3①** | `examples/segmentation/main.py:139-140` | 该 `else: arange(N)` 分支**补一次 `np.random.shuffle`**（§3.3） |
| **A3②** | `deploy/common.py:71` | 同上 |
| **A3③** | `deploy/onnx_inference.py:267` | 🔴 `voxel_size=float(cfg.dataset.common.voxel_size)` → `voxel_size=cfg.dataset.common.get('voxel_size', None)`。该脚本 `:178` 默认 `--cfg cfgs/radar/hpenet-ll.yaml`、`:221` `recursive=True` → 级联到 `default.yaml:7`，**必然触发**。`trt_inference.py` / `v2_e2e_dump.py` 是直接透传（不崩），只有本文件多套了一层 `float()`。 |

**规范**：
- ✅ **train/val 无需任何改动** —— `crop_pc:169-171` 的 shuffle 位于 `if voxel_max is not None`（`data_util.py:154`）块内，`voxel_max: 8000` 下照常执行。
- ❌ **不要**在 `s3disRadar.py.__getitem__` 补 shuffle —— 会多一次冗余 shuffle、改变 RNG 流与排列 → 训练结果偏离基线。

**收益**：train/val −0.76 ms/file CPU；test −0.62 ms/file；部署 Python −0.6 ms/file
**不改**：模型 / BN / collate / `batch_size`（`hpenet-ll.yaml:40 = 8`）/ `voxel_max: 8000` / `CPP_trt3`

### 方案 B —— Python 侧：去掉 scatter 的 **mean 归约**（保留重排）

> 依赖 §3.3 的「保留 shuffle」决策（推荐保留）与方案 C 先落地；A + C 通过后再做，做完再验一次 acc。

🔴 **核心原则：不是「跳过 scatter」，而是「保留重排、只去 mean 归约」。**
保留 shuffle ⇒ `idx_points` 是**非恒等排列** ⇒ 该 scatter 不只求均值，**更是把「子云内顺序」重排回「原始点序」**。整体跳过会让 logits 停在置换序、而 `label` 是原始序 → 逐点错配 → **静默低 acc（不报错）**。
`count.max()==1` 时等价于 `out[idx] = logits`。

| # | 文件:行 | 改动 |
|---|---|---|
| **B1** | `examples/segmentation/main.py:703`（`test()` 内） | `all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')` → `out = torch.empty_like(all_logits); out[idx_points] = all_logits; **all_logits = out**`（🔴 **必须回绑 `all_logits`** —— `:707` 的 `pred = all_logits.argmax(dim=1)` 消费的是 `all_logits`；只建 `out` 不回绑 → 预测仍停在置换序 → **静默低 acc**） |
| **B2** | `deploy/trt_inference.py:88`（`infer_one_cloud_trt`）/ `:113`（`_onnx`）/ `:141`（`_pytorch`）；`deploy/onnx_inference.py:120`（`_onnx`）/ `:162`（`_pytorch`） | 5 个站点统一：`merged = torch.empty_like(all_logits_cat); merged[idx_flat] = all_logits_cat` |
| **B3** | `CPP_trt4/src/pipeline.cpp:208-229`（`process_pointcloud`，内核调用在 `:221-229`）；`:411-433`（`process_file`，内核调用在 `:425-433`） | 见下 |

**B3 具体做法（选 (b)）**：**D2H 后在 CPU 按 `idx_staging_` 的逆排列重排**

```cpp
// —— 用以下代码替换原 Step 6（原 pipeline.cpp:236-243 / :440-447）；循环落位见下方 NVTX 条目 ——
result.logits.resize(static_cast<size_t>(N_orig) * 2);         // ⚠️ 保留原 resize，勿删！

std::vector<float> shuffled(static_cast<size_t>(N_orig) * 2);  // 临时缓冲：承接 d_src 的乱序 logits
CUDA_CHECK_THROW(cudaMemcpyAsync(
    shuffled.data(), d_src.data(),
    static_cast<size_t>(N_orig) * 2 * sizeof(float),
    cudaMemcpyDeviceToHost, stream_.native()));
stream_.synchronize();                       // 取代原 :243 / :447 的 sync（是"取代"，不是"叠加"）

for (int j = 0; j < N_orig; ++j)             // 逆排列：乱序位置 j → 原始点 idx_staging_[j]
    for (int c = 0; c < 2; ++c)
        result.logits[idx_staging_[j] * 2 + c] = shuffled[j * 2 + c];
```

- 🔴 **`result.logits.resize(...)` 必须保留** —— 上面这段是 Step 6 的**替代实现**；若按"整块替换"而漏掉 resize，`result.logits` 为空 → **越界写**
- 点数一律用 **`N_orig`**（`:209` / `:412`），**不要引入未定义的 `N`**。C 落地后 `total_src ≡ N_orig`（单子云、`idx` 为全体 N 点），故 D2H 尺寸与索引范围都安全
- 现有的 memcpy 外层包 `CUDA_CHECK_THROW`（`:237` / `:441`）、且用 `stream_.native()`（`:242` / `:446`）—— 上例已对齐，勿退化成裸 `cudaMemcpyAsync(..., stream)`
- **必须经临时缓冲 `shuffled`** —— 现状 D2H 直写 `result.logits`，就地重排会造成读写别名
- 重排循环**插在 D2H 与 Step 7 argmax 之间**；随后原有 argmax / 精度统计 / dump 逻辑不动（Step 7.5 精度用由 logits 派生的 predictions，`:459-466`；dump `:479-489` 本就期望原始点序）
- ⚠️ **NVTX 归因**：`process_file` 在 D2H/sync（`:447`）与 Step 7（`:450`）之间还夹着 `nvtxRangePop(); // TAIL`（`:448`）。请把重排循环**放在 `:448` TAIL pop 之后**（或明确置于 TAIL 内），以免 §3.1「TAIL 内 scatter 段 0.5–1ms」的计时口径漂移
- 随之**不再需要** GPU 侧 `d_out`（分配在 `pipeline.cpp:212` / `:416`）与 `d_cnt`（`:214` / `:418`），以及 `d_idx` 的**分配**（`:127` / `:323`）与 `d_idx.upload()`（`:217-220` / `:421-424`）—— (b) 的 CPU 重排只用 CPU 侧 `idx_staging_`，`d_idx` 已无消费者
- 删掉该次 `launch_scatter_mean_kernel`

> **(a) 不推荐**：保留内核只去 mean 归约时，`count==1` 下数值与现状逐位相同，而 memset / 两段内核 / `d_cnt` 耗时**全部保留** → 兑现不了本方案的收益（等价于不改）。

**注意**：
- 措辞用「**scatter 去掉 mean 归约**」，不要用 "gather"（字面 gather 需逆排列）
- 索引类型见 §3.4（`idx_points`=`int`，`idx_staging_`/`d_idx`=`int64`）
- 若**不** shuffle（`idx_points=arange`）→ 可完全去回填，但引入系统性 FPS 偏向，须重新验证 acc

**收益**：−0.08 ms(L20) / Orin TAIL 内 scatter 段约 0.5–1ms

### 方案 C —— C++ 侧：彻底删除体素化（只在 `CPP_trt4`）

> **C 是纯 C++、独立于方案 A**，A 与 C 可并行。

| # | 文件:行 | 改动 |
|---|---|---|
| **C1** | `pipeline.cpp:111-112`（`process_pointcloud`）；同函数 `:117` / `:132` 的 `vox.idx_points` | 删 `Voxelizer::voxelize` 调用，改由本地构造单子云（§3.3 片段） |
| **C2** | `pipeline.cpp:305-308`（`process_file`，含 `nvtxRangePushA("VOXELIZE")` @305）；同函数 `:313` / `:327` / `:330` | 同上，NVTX 区间一并删；`:327 size_t count_subcloud = vox.idx_points.size();` 改写（去体素后恒为 1）。下游 `:474 result.count_subcloud = count_subcloud;` 无需单改 |
| **C3** | `CMakeLists.txt:64-65` | 移除 `src/voxelizer.cu`、`src/fnv_hash.cu` |
| **C4** | 删除 `src/voxelizer.cu`、`src/fnv_hash.cu`、`include/voxelizer.h`、`include/fnv_hash.h` | 同时**必须删掉 `pipeline.cpp:31` 的 `#include "voxelizer.h"`**（否则报 "No such file or directory"）。另建议一并清理 `include/pipeline.h:13` 已注释的 `//#include "voxelizer.h"`。⚠️ 另有四处**过期注释**（不影响编译，可选清理）：`pipeline.cpp:6`（`// 3. Voxelizer::voxelize → idx_points`）、`pipeline.cpp:15`（`// 5. launch_scatter_mean_kernel → GPU 合并`）、`pipeline.cpp:234`（`// ---- Step 6: 下载合并后的 logits ----` —— B3(b) 后已无 GPU 合并；其"文件内唯一 sync"部分仍正确）、`pipeline.h:44`（`/// @param seed 体素化随机种子` —— 属 C5"保留"行，文案会变陈旧） |
| **C5** | `include/pipeline.h:43/51/106`；`src/pipeline.cpp:46`（形参 `float voxel_size,`）、`:50`（初始化列表 `, voxel_size_(voxel_size)`）；`src/main.cpp:31/46/73-74/123/143/185`；`include/types.h:37` | 移除 `voxel_size` **全链路**。**`seed` 保留**（`pipeline.h:44/52/107`、`main.cpp:36/126/146`）。⚠️ `:73-74` 是 help 的**两行**（`:73` 的 `--voxel_size=<float>` + `:74` 的 `(default: 0.02)`），删 `:73` 必须连 `:74`。⚠️ 同步改 `deploy/measure_orin.sh:19-20` 的**注释**（那里仍写"如需覆盖再显式加 `--voxel_size=0.02`"）—— 该 flag 在 C5 后不存在，**若照注释执行**会命中 `main.cpp:129-131` 的 `Unknown argument` → exit 1（仅保留注释则不会报错） |
| **C6** | 🔴 `src/trt_inference_wrapper.cpp:192-194` | 删掉硬编码的 `0.02f` 实参；若原意是 `seed=100` 则显式补 `100`（不补则默认也是 100）。**必须在 C5 之后做** |

**C1/C2 必须同时补齐（否则编译失败）**：

1. `VoxelizeResult` 随 `include/voxelizer.h` 消失 → 替换为 `std::vector<std::vector<int>>`
2. 新增两个 include：
   - `#include "random_util.h"` —— `pipeline.cpp` 的 include 块（20-31 行）没有它，`NumpyMT19937` 只在 `voxelizer.cu:8` 可见
   - `#include <numeric>` —— 片段用 `std::iota`，现有 include（含 `<random>`）都不提供（`main.cpp:10` 有，`pipeline.cpp` 没有）；或改用显式 `for` 循环
3. 单子云构造：用 §3.3 的片段（`NumpyMT19937(uint32_t seed=100)` 是 `explicit`，此处为直接初始化不受影响；`seed_` 是 `int` → 隐式转 `uint32_t` 无碍；`shuffle(int*, int)` 签名匹配）
4. `build/CMakeFiles/hpenet_trt_infer.dir/link.txt` 仍引用 `src/voxelizer.cu`（stale 缓存）→ **必须 clean rebuild**，不能只 `make`

**⚠️ C-API 路径语义跳变**：`trt_inference_wrapper.cpp:194` 当前传 `0.02f`（**真体素化**，250/339 文件多子云），而 CLI 路径 `main.cpp:31` 是 `0.0001f`（no-op）。删体素对 C-API 路径是「**多子云 + scatter_mean**」→「单子云」的跳变，**行为变化远大于 CLI 路径** → §5 的 C-API 对拍须用 **acc 级**，不能用 pred。

**不要误删**：

| 项 | 说明 |
|---|---|
| `random_util.cpp/.h` | **必须保留** —— 新路径 shuffle 依赖它 |
| `scatter_mean.cu/.h` | **保留文件**。⚠️ B3 选 (b) 后 `launch_scatter_mean_kernel` 不再被调用 → `scatter_mean.cu/.h`、`pipeline.cpp:27` 的 `#include "scatter_mean.h"`、`CMakeLists.txt:66` 均成**死代码**（保留不影响编译/正确性）。注意 `scatter_mean.cu` 同时是另一工程 `CPP_trt/tests/test_scatter_mean.cu` 的被测对象 |
| `subcloud_utils` | 保留（pad/split 对应 `min_n`/`max_n`，与体素无关） |
| `trt_plugins` 的 ball_query 网格参数 | **不要动**（与数据体素化无关） |

**收益**：**Orin −1.85 ms/帧（≥7.2%，口径见 §3.1）**；叠加方案 B → **−9~11%**

### 方案 D —— **已撤销（不执行）**

> **决策（用户指令）**：`crop_pc` 的体素分支与 `voxelize()` 函数**一律保留、不删任何代码**。

D1（删 `crop_pc:149-153`）与 D3（删 `voxelize()`）**均不执行**：

1. **参数已能绕过** —— A2 的 `null` 使 `crop_pc:149` 的 `if voxel_size and downsample:` 为假，该分支本就不会执行。
2. 🔴 **`crop_pc` 是共享函数，删分支会跨任务破坏** —— 被 6 个 loader 调用，其中 S3DIS / ScanNet 仍用**有效** voxel_size：

| 调用者 | voxel_size 来源 |
|---|---|
| `radar/s3disRadar.py:188`、`radar/s3disRadar_sphere.py:158` | radar cfg（A2 后 `null` → 绕过） |
| **`s3dis/s3dis.py:130`**、`s3dis/s3dis_sphere.py:158` | **`cfgs/s3dis/default.yaml:7 = 0.04`（有效）** |
| **`scannetv2/scannet.py:159` / `:207`**（另 `:197` 在 `"""debug…"""` 注释块内，非活调用点） | **`cfgs/scannet/default.yaml:6 = 0.02`（有效）** |
| `semantic_kitti/semantickitti.py:174 / 221` | KITTI cfg |

   → 删掉会让 **S3DIS / ScanNet / SemanticKITTI 的体素降采样静默失效**。**该分支只对 radar 是死代码，对其它任务是活代码。**
3. **`voxelize()` 同样不能删** —— 仍被 `s3dis/s3dis.py:8`、`s3dis_sphere.py:8`、`scannetv2/scannet.py:7`、`semantic_kitti` 等多个 loader import。这符合「尽量保持代码不动」：Python 侧只改 **1 行 yaml（A2）+ 3 处阻塞点（A1/A3）**。

**结论**：`openpoints/dataset/data_util.py`（`voxelize()` 127-143、体素分支 149-153、`:173`）**全部原样保留**，**不在改动清单中**（§2.2 仅作参考）。

---

## 5. 验收

### 5.1 前置条件

1. 🔴 **钉死 baseline 并重测（硬性）**：**指定用 `log/radar/...20260907-170521-...` 的 `_ckpt_best.pth`**（也是 `trt_inference.py:156` 的默认 checkpoint）—— 扫描全部 **16 个** hpenet-ll run，它是**唯一**以 `voxel_size: 0.0001` + `voxel_max: 8000` 训练的（`cfg.yaml:20` / `:31,:36`）。
   - 🔴 **必须用该 ckpt 重测 `ti10`（10 文件 mean）作为 Δacc 的基准；不要沿用 `0.9578`** —— `latency-statistics.md:4` 把 `0.9578` 归给 **`20260825-161134` + voxel `0.02`** 的部署口径，而被钉 run 自身 CSV 是 `OA 92.06 / mACC 88.50 / mIoU 77.03`，**两者不是同一个量**。
   - ❌ **不要用 `20260825-161134`** —— 训练口径是 `voxel_size: 0.02` / `voxel_max: 4608`（`cfg.yaml:21` / `:32,:37`），会以「训练口径差异」污染 Δacc。
   - 🔴 **两个「自带默认值」与此冲突，必须显式覆盖**：
     - `script_me/main_segmentation_test.sh:43` 的 `--pretrained_path` **正是被禁用的 `20260825-161134`**（`mode=test` 经 `main.py:225` 的 `load_checkpoint(..., pretrained_path=cfg.pretrained_path)` 消费该参数）
     - `deploy/onnx_inference.py:175` 默认 checkpoint 是 `20260812-201051`（训练口径 voxel 0.3）
2. **seed（卫生项）**：核心 Δacc **只需两次运行 seed 恒定**，不需要与 C++ 统一（§3.5）。
3. **engine 无需重建（确认项）**：profile `max_n=10000` > 7837 → 单子云仍在覆盖范围内；但 `opt_n=4096` 低于中位 5845，profile 非最优点、性能可能略偏。（本项**无法静态核实**，需 TRT 运行时内省。）

### 5.2 验收矩阵

| 阶段 | 方式 | 判据 |
|---|---|---|
| train | 同 seed 跑 1–2 epoch | loss 无发散/无 NaN；**train acc 与现基线之差 < 1.0pp** |
| val / test | `ti10 acc` 对拍（`voxel_size=null` vs `0.0001`），**同 seed 同 checkpoint** | **Δacc < 0.3pp** |
| 部署（Python 脚本） | `deploy/trt_inference.py --num_files 10`、`deploy/onnx_inference.py --num_files 10` | 与 Python 参考 acc 差 **< 0.3pp** —— 覆盖 **B2 的 5 个站点**（改错会**静默低 acc**，其他行捕获不到）；且 `onnx_inference.py` **不再抛 TypeError** —— 覆盖 **A3③** |
| 部署（CLI） | L20 + Orin 端到端，`deploy/analyze_latency.py` | VOXELIZE 段消失；**端到端延迟下降 ≥ 5%**（分母用 0.0001 基线，非 25.66ms） |
| 部署（C-API） | `CPP_trt4` vs `CPP_trt3` | **同工程前后 acc 级对拍**（⚠️ **本行不设「<0.3pp」阈值**：该路径原本用 `0.02f` 真体素化，**本身即提供 2 路集成增益**，去掉必然产生差值，应**量化并交用户裁定取舍**，而非套用阈值 —— 用户已裁定接受）。
> 📌 **判据缺陷说明（2026-09-16，实施后复查）**：本行原写「与去体素化后 Python 参考 acc 之差 < 0.3pp」，**该判据无判别力** —— 用**未改动的 pristine before 镜像**验算亦不通过（CLI before `93.58` vs 参考 `93.20` = **+0.38pp**；C-API before `94.62` = **+1.42pp**）。根因：把两个**不同模型变体**放进同一判据 —— TRT 引擎经 `deploy/onnx_export.py:238` 默认 `fps_cache_prune` + `deploy/onnx_backend.py:338-368` 打补丁为 **`hpenet::FPSPrune(keep_rate=0.75)` 变体**，而 Python 参考是无 prune 的 PyTorch 模型。**正确口径为「同变体对拍」**，实测全部通过：C-API vs `_trt`（同引擎）= **0.0000pp**；`_pytorch` vs Python 参考 = **−0.21pp**；去体素化本身（val/test 行）= **−0.07pp**。据此**本行与「部署（Python 脚本）」行均已改为同变体口径**。<br>⚠️ **测量机制须自建**：C-API 各入口（`trt_inference_wrapper.cpp:216/254/302/381`）**只返回 `latency_ms`、不返回 accuracy**；且 `trt_ai_infer_and_update` / `trt_ai_infer_all_radars` 经 `update_predictions_to_cdi` **原地覆盖 `valid` 字段、销毁 ground truth** → 必须在调用**前**备份标签，再用外部脚本对拍。 |
| 构建 | `CPP_trt4` **clean rebuild**，跑通 CLI + C-API 两个入口 | exit 0；两入口结果自洽 |

---

## 6. 否决项

| 方案 | 原因 |
|---|---|
| `voxel_size: 0` | 静默 2510× 前向（§1.3），灾难 |
| `voxel_size: null` + `voxel_max: null` | 模型不支持 batch>1（`SetAbstraction.forward` 只读 `p.shape[1]//stride`，完全不读 `batch`/`o`）→ 只能 batch_size=1，收益不抵复杂度；且会丢 `crop_pc:169-171` 的 shuffle |
| 用 IN 替换 BN | BN 在 TRT 可 conv-folding 归零；IN 不可 → 纯负收益（§1.4） |
| 删掉 `NumpyMT19937` / `random_util` | 无体素路径仍需它做 shuffle（§3.3） |
| 动 `trt_plugins` 的 ball_query 网格参数 | 与数据体素化无关 |
| 直接复用现有 `else: arange(N)` 死代码 | 产出恒等排列 → FPS 系统性偏向（§3.2 / §3.3） |
| 用字面 "gather" 替换 `scatter_mean` | 字面 gather 需**逆排列**；应表述为「scatter 去掉 mean 归约」（§4 方案 B） |
| 在 `s3disRadar.py.__getitem__` 补 shuffle | **冗余 shuffle** → 改变 RNG 流与排列 → 训练结果偏离基线；train/val 已有 `crop_pc:169-171`（§4 方案 A） |
| 把 `idx_points` / `idx_part` 改成 `int64` | 与 `preprocess_subcloud(const int*)` 不兼容（§3.4） |
| 只加 `#include "random_util.h"` 而不加 `<numeric>` | `std::iota` 报「未声明」→ 编译失败（§4 方案 C item 2） |
| 在「保留多子云回退路径」的前提下不加 gate 就去回填 | 多子云（索引重叠）会**静默丢掉均值投票**（§3.4）。**注意**：本方案 C 落地后单子云由构造保证 → **B3 选 (b) 时不需要 gate**；仅在保留回退路径时才写 `idx_points.size()==1` |
| 只 `make` 不做 clean rebuild | `build/.../link.txt` 仍引用已删的 `voxelizer.cu`（§4 方案 C item 4） |
| **删除 `crop_pc` 的体素分支或 `voxelize()` 函数** | **方案 D 已撤销**：`crop_pc` 被 6 个 loader 共享，S3DIS（0.04）/ ScanNet（0.02）仍用有效 voxel_size → 降采样**静默失效**（跨平台破坏）（§4 方案 D） |

---

## 7. 执行顺序

1. **方案 A**（Python 侧）：A1（f-string 条件化）→ A2（yaml `0.0001 → null`）→ **A3 三处**（`main.py:139-140` 与 `common.py:71` 补 shuffle；`onnx_inference.py:267` 改 `.get()`）→ 跑 train 1 epoch + `ti10` 验 acc。
2. **方案 C**（纯 C++，**与 A 并行、不依赖 A**）：
   **C1/C2**（含类型替换 + `random_util.h` + `<numeric>` 两个 include）→ **C5**（移除 `voxel_size` 全链路）→ **C6**（删 `trt_inference_wrapper.cpp:194` 的 `0.02f`）→ **C3/C4**（删编译项与文件、清 `pipeline.cpp:31` include）→ **clean rebuild** → L20 / Orin 对比 `CPP_trt3`。
   > 🔴 **C5 不可省、且必须在 C6 之前**：C5 是唯一移除 `--voxel_size` 开关与 `voxel_size_` 成员的步骤（漏做即违反"不加 `--no_voxel` 开关"的硬约束）；顺序颠倒时 C6 补的 `100` 会绑到仍是 `float` 的 `voxel_size` → **voxel_size=100m，全部点并成一格**。同步改 `deploy/measure_orin.sh:19-20` 的注释。
3. **方案 B**（Python + CPP_trt4）：A + C 都通过后落地，再验一次 acc。
4. **方案 D —— 已撤销，不执行**（用户指令）。

> ⚠️ 方案 B 跨三层：B1（`main.py`）、B2（`deploy/*.py`）是 **Python 侧**，只有 B3 在 **`CPP_trt4`**。

---

## 8. 修订历史

| 轮 | 审查方式 | 结果 |
|---|---|---|
| 1 | Momus + Oracle | 1 blocker + 6 should-fix + 3 nit（原稿缺陷） |
| 2 | Momus + Oracle | 修第 1 轮 patch **自引入**的 4 处 + 2 should-fix + 10 nit |
| 3 | Momus + Oracle | 12/12 断言 CONFIRMED；宣称收敛（**后被证伪**） |
| 4 | Momus + Oracle（**实际同一模型**） | 证伪收敛：2 should-fix + 1 nit |
| 5 | **V4.1 Flash × V4 Pro**（首次真双模型） | 结论分歧：V4 Pro 报 0 SF / 2 nit；**V4.1 Flash 报 1 blocker + 3 SF + 4 nit**（复核全部成立） |
| 6 | V4.1 Flash × V4 Pro | 两模型均判无 blocker；Momus 另报 4 SF + 8 nit（复核全部成立，**Oracle 未发现**） |
| 7 | V4.1 Flash × V4 Pro | 重写回归核验：发现重写**引入** 6 处（R1–R6）+ 9 latent nit；**Oracle 报的 `onnx_inference.py:267→268` 经复核为误判** |
| 8 | V4.1 Flash × V4 Pro | P1–P8 两模型独立核验**全部正确**；**双方一致判「可执行、第九轮不必要」**（唯一 SF 为 F1：B3(b) 片段保真度） |
| 9 | V4.1 Flash × V4 Pro | 核验第 8 轮后补丁：**F1–F7 全部正确**，B3(b) 代码块经**逐符号双向追踪**判定可直接照抄；唯一 SF 为 **B1 缺 `all_logits` 回绑**（第 5 轮引入、潜伏 4 轮）；**双方一致判「已收敛、第十轮不必要」** |
| — | 用户指令 | **方案 D 撤销**（保留代码不动） |
| — | **实施后复盘**（用户指令） | ① **判据更正**（用户指令「做 G」）：「部署 Python / C-API」两行的「vs Python 参考 <0.3pp」经证为**非判别**（pristine before 亦不通过）→ 改为**同变体对拍**，见 §5.2 注；② **用户裁定「F 接受」**：接受 C-API 路径 **−0.90pp** 精度代价；③ **代码审查 + 加固**：发现并修复 2 项（`main.py:703` 失实注释；`pipeline.cpp` ×2 加 `total_src == N_orig` 运行期不变量检查 —— 用 `std::runtime_error` 而非 `assert`，因 Release 带 `-DNDEBUG` 会编译掉裸 assert），并据此**重跑受影响的 F1/F2/F4/F5**；④ **F4 延迟以 ABBA 反向配平 + n=20/侧 复测**（用户指令「做 A」，修正此前「顺序恒为 B→A」的方法学缺陷）：部署口径 **−11.56%**，块内对比 95% CI 下界 **+5.43% ≥ 5%** → 「真值 ≥5%」的强断言成立 |

**被驳回/更正的关键误判**（勿重犯）：`voxel_size=0` **不崩溃**（静默 2510× 前向，更危险）；`main.py:552` 属 `validate_sphere`，**不在 radar 路径**；FPS 首中心点证据是 **`:120-122`**（`:126` 是每轮搜索初值）；删 `voxel_size` **能编过**（float→int 静默 misbind `seed=0`）；A3 **不需**在 `s3disRadar.py.__getitem__` 补 shuffle；`idx_points`/`idx_part` 是 **`int`** 不是 `int64`；B1/B3 **不得**「跳过 scatter」（静默错误）。

**已独立复核为正确的关键推理**：RNG 只被 shuffle 消耗（`idx_select` 是确定性 cumsum+modulo）；`count.max()==1` 时 `idx_sort[idx_select]` 是全体 N 点的完整排列 → **非 bit 等价、但统计等价**；**坐标范围不变**（`__getitem__:185` 已 min-subtract、`crop_pc:151` 本为 no-op）→ `radius: 5` 无新风险；`novx` tag **不干扰** `feat_stats_area5.pth`；`voxel_size: null` **不改** `main.py:677-684` 的 `variable` 分支；VOXELIZE **确属部署口径**（PLY_LOAD 是唯一被排除项）；`batch_size=8` 正确（`hpenet-ll.yaml:40` 覆盖 `default.yaml:24=16`）。
