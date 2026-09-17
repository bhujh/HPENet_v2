# Padding 对 Test 结果的影响 —— 分析与实测记录

> 分析日期：2026-09-17
> 对象：HPENet V2 / radar 二分类分割（`cfgs/radar/hpenet-ll.yaml`）
> 仓库状态：`voxel_size: null`（已去体素化）、`dataset.train/val.voxel_max: 8000`、`dataset.test.voxel_max: null`
> 数据：`data/RadarClassi/radarfullwl`，339 个 PLY（train 281 / val 58）
> **本文只记录分析与实测，未修改任何仓库代码。** 分析脚本全部位于 `/tmp/opencode/vm/`。

---

## 摘要（结论速览）

1. **三个阶段的 padding 策略不一致**：训练 ✅补齐到 8000、验证 ✅补齐到 8000、测试 不补齐（原始 N）；**部署路径名义上 pad 到 `min_n`，但实测从不触发，等价于不补齐**。
2. **val 与 test 用的是同一批 58 个文件**，所以同一个 ckpt、同一批文件会报出两个不同口径的指标。
3. padding 影响结果的通道有 **5 条**，其中 **2 条主通道**（FPS、ball_query 截断）与 **1 条独立通道**（`PointCloudXYZAlign` 的均值平移，因 stage-1 HPE 吃绝对坐标而**不被抵消**）。
4. **关键分界线：`max` 对重复点幂等，`mean` 不是。** 实测未截断时 `max` 结果 0.0% 改变，而 `mean` 仍 58.2~69.5% 改变。
5. **口径切换的代价实测 ≈ 2.4pp**（健康模型 B：补齐 77.91 vs 不补齐 75.51），且偏差不均匀 —— 文件越小偏差越大。
6. padding 的**冗余计算**：中位 **+37%**（重复点占比中位 26.9%，最大 59.6%）。
7. **部署侧因不补齐而无法静态化**：TRT 只能用 dynamic optimization profile、每次 `setInputShape`、CUDA Graph 只能**逐 shape** 捕获；但「**桶化 pad 到静态 shape**」已因 **FPS 采样点数随 N 缩放** 被否决（pred 一致率仅 98.5%）——与本分析 §4.1 是同一机理。

---

## 1 背景：为什么用 padding

- **雷达点云点数动态**：全 339 个文件 `N_raw ∈ [3005, 7837]`，中位 5845，无固定值。
- **模型要求同 batch 内等长**：`RadarClassi` 没有 `collate_fn`（`openpoints/dataset/build.py:73` → `None`），走 PyTorch `default_collate`，变长张量无法 stack。
- **因此 `variable=False`（= 补齐到 `voxel_max`）是 `batch_size > 1` 的前提**。实测：

  ```
  variable=True,  batch_size=8  → RuntimeError in DataLoader worker（无法 stack 变长张量）
  variable=True,  batch_size=1  → OK, pos.shape=(1, 6717, 3)
  variable=False, batch_size=8  → OK, pos.shape=(8, 8000, 3)
  ```

- **补齐的收益**：
  - 允许 `batch_size=8` → 训练吞吐高（实测 1.5 min/epoch vs `bs=1` 的 1.8 min/epoch）
  - **BatchNorm 健康**：`bs=8` 的 batch 统计量已接近群体统计（`running_var` 与真值 39.7 vs 38.4，低估仅 4.6%）；而 `bs=1` 会让 BN 退化为逐样本归一化，`running_var` 低估群体方差 50.8~67.1%，`validate()` 用 running stats 前向时逐层放大 → val 崩溃（另见独立分析）
- **补齐的代价**：① 冗余计算（§6）② 训练/测试口径不一致（§5、§7）③ 阻碍静态 shape / 计算图捕获（§8）

---

## 2 三个阶段（含部署）的 padding 现状

| 阶段 | 入口 | 是否经过 `crop_pc` | 目标点数 | 是否 padding |
|---|---|---|---|---|
| **训练** | `train_loader`（`examples/segmentation/main.py:262`）→ `RadarClassi(train)` → `crop_pc`（`openpoints/dataset/radar/s3disRadar.py:189-191`） | ✅ | `dataset.train.voxel_max = 8000` | ✅ **补齐到 8000** + shuffle |
| **验证** | `val_loader`（`main.py:193`），同一路径 | ✅ | `dataset.val.voxel_max = 8000` | ✅ **补齐到 8000** + shuffle |
| **测试**（`mode=test`） | `main.py:226-242` → `generate_data_list` + `test()` | ❌ **完全不经过 dataset** | `dataset.test.voxel_max = null` | ❌ **不补齐**，原始 N |
| **部署**（C++ TRT） | `deploy/CPP_trt4/src/pipeline.cpp` | ❌ | C++ 侧 `min_n`（默认 2024） | ⚠️ **名义上 pad 到 `min_n`，实测从不触发** |

### 2.1 代码证据

- **全仓 `voxel_max` 的消费点**只有：`openpoints/dataset/{data_util.py, radar/s3disRadar.py, s3dis/s3dis.py, scannetv2/scannet.py, semantic_kitti/semantickitti.py}`、`deploy/{onnx_export.py, eval_gridballquery_miou.py}`、`debug_compare_val_test.py`。
  **`examples/segmentation/main.py` 根本不在列表里** → test 路径不可能 padding。
- `mode=test` 在 `main.py:242` 提前 `return`，早于 `:262` 建 `train_loader`；`val_loader` 虽在 `:193` 无条件构造，但只用于取 `num_classes / classes / cmap` 元数据、**从不迭代** → `__getitem__` → `crop_pc` 不会被调到。
- 实测 C 运行的 test 输出每文件点数：`3234 / 4641 / 5014 / 5174 / 5439 / 5623 / 6504 / 6776` = 原始 N ✅

### 2.2 部署侧的实际情况（重要修正）

`deploy/CPP_trt4/src/pipeline.cpp:9-13` 注释描述的流程：

```
//      b. 若 N > max_n: SubcloudUtils::split_oversized 拆分为 chunks
//      c. SubcloudUtils::pad_subcloud → 填充至 min_n
//      e. TrInference::infer → GPU logits (1,2,N_padded)
//      f. cudaMemcpyAsync → CPU, trim_padding → (1,2,N_true)
```

即部署侧**确实有 pad 逻辑**（`pad_subcloud` → `min_n`，推理后 `trim_padding` 回 `N_true`）：

- `CPP_trt4/src/main.cpp:29-30`：默认 `min_n = 2024`、`max_n = 10000`
- `CPP_trt4/include/pipeline.h:47-48`：默认 `min_n = 1024`、`max_n = 10000`
- `pipeline.h:40-41` 注释：`min_n` 子云最小点数（不足则填充）；`max_n` 子云最大点数（超出则拆分）

**但对雷达数据实测从不触发**：

```
全部 339 个文件 N_raw: min=3005  p5=3595  median=5845  max=7837
N < min_n(2024) 的文件数 = 0     → pad_subcloud 从不执行
N > max_n(10000) 的文件数 = 0     → split_oversized 从不执行
```

→ **部署路径实际是「单子云 + 不定长 + 不补齐」**，与 Python 的 test 口径一致，与训练口径（8000）不一致。

---

## 3 关键前提：val 与 test 是同一批文件

- `RadarClassi`（`s3disRadar.py:62-68`）：`np.random.seed(100)` → `shuffle` → `data_list[int(n*0.83):]`
- `generate_data_list`（`main.py:59-65`）：`np.random.seed(100)` → `shuffle` → `data_list[int(n*0.83):]`

**种子与切片完全相同 → val ≡ test（同一批 58 个文件）。**

因此同一个 ckpt、同一批文件会得到两个不同口径的指标：**val = 补齐 8000、test = 原始 N**。这是 §5 实测口径差的直接来源。

---

## 4 Padding 影响结果的通道（逐算子）

| # | 算子 | 归约 | padding 影响 | 实测（6 个 val 文件） |
|---|---|---|---|---|
| 1 | **FPS** | — | ✅ **点数 + 选点** | 补齐 8000 → 固定 **2000**；不补 → `N//4 ∈ [808, 1925]`，中位 1462（**1.04~2.48×**） |
| 2 | **ball_query** | 前 `nsample` 截断 | ✅ 仅当 `cnt ≥ nsample` | 截断率 **pad 17.4~21.7% / raw 9.1~19.1%** |
| 3 | **max pooling** | `max(dim=-1)` | ️ **只经 ball_query 截断**（对重复点幂等） | 未截断时 **0.0% 不同**；整体 11.3~18.1%（全部来自截断） |
| 4 | **mean over `nsample`** | `mean(dim=-1)` | ✅ **未截断也受影响** | 整体 65.5~76.0% 不同；**未截断者仍 58.2~69.5% 不同**；相对偏差 **1.76~2.16%** |
| 5 | **PointCloudXYZAlign** | `pos -= mean(pos)`（点维） | ✅ **不被抵消** | xyz 均值相对偏移 **0.44~0.72%** |

次级通道：**BAFM 的点维 `mean`**、**HPE 的 BN 统计**、**逐点指标的随机重加权**（§4.5）。

### 4.1 FPS

`openpoints/cpp/pointnet2_batch/src/sampling_gpu.cu:120-145`：

```cpp
int old = 0;
if (threadIdx.x == 0)
idxs[0] = old;        // 种子点恒为 index 0
for (int j = 1; j < m; j++) { ... besti = d2 > best ? k : besti; ... }   // 贪心 max-min
```

- **点数随 N 缩放**：调用处 `hpenetv2.py:160` → `self.sample_fn(p, p.shape[1] // self.stride)` → `m = N // 4`（stride=4）。
  - 补齐 8000 → **m 恒为 2000**
  - 不补齐 → `m = N_raw // 4 ∈ [808, 1925]`，中位 1462
  - **密度比 1.04× ~ 2.48×，文件越小差越大**
- **选点不同**：种子恒为 index 0，且贪心平局按索引序决定 → 点序/点数变了，选出的子集就变了。
- **重复点永远不会被 FPS 选中**（与原件距离恒为 0，`argmax` 选最远点）→ 补齐的作用是把「同一批几何位置上选 2000 个」vs「选 `N//4` 个」，即**纯粹抬高采样密度**。

### 4.2 ball_query

`openpoints/cpp/pointnet2_batch/src/ball_query_gpu.cu:16-49` 的语义：**按索引序扫描，凑满 `nsample` 就 `break`**，取的是「索引序里前 `nsample` 个」而**不是按距离挑**；收集不满时用**第一个命中点重复填充**；一个都没命中时 `idx` 全 0。

```cpp
int cnt = 0;
for (int k = 0; k < n; ++k) {
    ... d2 ...
    if (d2 < radius2){
        if (cnt == 0){ for (int l = 0; l < nsample; ++l) idx[l] = k; }
        idx[cnt] = k;
        ++cnt;
        if (cnt >= nsample) break;      // ← 截断
    }
}
```

所以在**同一套 query 点**上比较 补齐 support vs 原始 support（radius=5、nsample=32，numpy 逐点复刻 kernel 语义）：

| 情况 | 邻居集合是否不同 | 说明 |
|---|---|---|
| `cnt < nsample`（球内点数不足） | **0.0%** | 所有球内点都被收进来，重复点只是同名副本 → **集合完全相同** |
| `cnt ≥ nsample`（截断） | **89.7~97.2%** | 截断取「索引序前 32 个」，padding 加进去的重复项会**挤掉别的不同点** |
| **整体** | **16.7~20.8%** | —— |

**padding 会放大截断率**：补齐后球内条目数被重复点灌水（`cnt` 计数含重复项），所以截断率 **pad 21.7% vs raw 19.1%**；小文件更明显（`N=4244` 时 **19.2% vs 9.1%**）。

**对池化结果的影响**：实测 **13.4%** 的 query 点，`max` 池化后的中位数坐标位移平均 **0.19~0.47**（坐标单位）。

### 4.3 关键分界线：`max` 幂等 vs `mean` 非幂等

- **`max`**：对一个含重复项的 multiset 取最大 = 对去重后的集合取最大 → **重复点永远不改变 max**。
- **`mean`**：重复项直接改变均值 → **非幂等**。
- 而且 `mean` **不需要截断就已经被影响** —— 因为 kernel 在球内点数不足 `nsample` 时会用**第一个命中点重复填充**，所以 `mean(-1)` 本来就是在含重复项的 multiset 上求均值。

**对照实测（同一套 query、同半径）**：

```
                    未截断时不同     整体不同      相对偏差
max over nsample       0.0%        11.3~18.1%       —
mean over nsample     58.2~69.5%    65.5~76.0%    1.76~2.16%
```

**归约位置与本配置的实际生效情况**：

| 位置 | 代码 | 是否为活代码 |
|---|---|---|
| `dh = x.mean(-1)` / `aj = x.max(-1)` / `fj = x.max(-1)` | `hpenetv2.py:82-83` 所在函数 | ❌ **未被使用** —— `HPENetV2Encoder` 内 `LocalAggregation.forward` 在 `hpenetv2.py:61-83` 被重写，只做 `grouping_operation` + `self.pool`，不调用原 `get_aggregation_feautres` 路径。注：`aj` 是 `max`，**幂等**，即使生效也不受重复点影响 |
| `self.pool(fj + pe)` | `hpenetv2.py:73`，`self.pool = torch.max`（`hpenetv2.py:141`） | ✅ 活代码，**max → 幂等**，仅经 ball_query 截断受影响 |
| `FeaturePropogation.pool = mean(dim=-1)` | `hpenetv2.py:241`，用于 `pf2 is None` 分支 | ❌ **死代码** —— decoder 的 4 个 FP **全部传了 `pf2`**（`hpenetv2.py:590-598`），该分支永不执行 |

→ **对 `hpenet-ll`（`feature_type: 'dp_fj'`、reduction=`max`）而言，`mean` 通道在当前配置下不生效**，但它对 `feature_type: 'dp'` 变体是实打实的通道，且是理解「padding 为什么比直觉更敏感」的关键。

### 4.4 PointCloudXYZAlign —— 唯一「不被抵消」的独立通道

`openpoints/transforms/point_transformer_gpu.py:87-90`：

```python
data['pos'] -= torch.mean(data['pos'], axis=0, keepdims=True)   # ← 点维均值，补齐会变
data['pos'][:, self.gravity_dim] -= torch.min(data['pos'][:, self.gravity_dim])   # gravity_dim=2 (z)
```

- **z 方向**：`pos−mean` 造成的偏移被随后的 `−min` **完全抵消** ✅
- **x/y 方向不被抵消** → 整个点云被平移了一个**不同的量**（实测 xyz 均值相对偏移 **0.44~0.72%**，坐标量级 ~100 m 时约 0.5 m）
- **而这个平移会被模型"看见"**：`SetAbstraction` 中 `stride == 1` 的 stage 走 `all_aggr`（`hpenetv2.py:110`）→ `group_args.nsample = None`（`:138`）→ `create_grouper` 返回 **`GroupAll`**（`group.py:345-352`，`GroupAll()` 在 `:351`）→ `GroupAll.forward` 返回的是 `xyz.transpose(1,2).unsqueeze(2)` = **绝对坐标**（不是相对坐标！）→ `pe = self.rel_pos(dp)`（`hpenetv2.py:196`）吃的是**绝对位置** → xy 平移直接改变 `pe` → 改变输出。

→ **val（补齐）与 test（不补齐）会把同一个文件平移到不同中心**，这是独立于 FPS 的额外分布差。

### 4.5 次级通道

| 通道 | 位置 | 机制 |
|---|---|---|
| **BAFM 的点维 mean** | `openpoints/models/layers/channel_attention.py:35` `y_mean = torch.mean(x2, -1, keepdim=True)` | 在**点维**取均值 → 重复点直接改变均值（随机重加权 → 期望无偏、方差增大）；活代码（`hpenetv2.py:601` `self.ddfc(f2_new, f[1])`） |
| **HPE 的 BN 统计维度** | `position_encoding.py:18` `create_convblock2d(3, in_channels//4, norm_args=norm_args)` → `conv.py:30` 传 `dimension='2d'` → `BatchNorm2d` | 统计维度是 **(B, npoint, nsample)** → 训练期统计受 DP 值变化影响（这也正是历史上 `rel_pos.conv.0.1` 出现 `running_var=5.61e-45` 下溢的位置）；eval 用 running stats → **无独立通道** |
| **逐点指标的随机重加权** | `openpoints/utils/metrics.py:51` `ConfusionMatrix` | 重复点与原件同坐标同特征 → 逐点算子下**预测完全相同** → 逐点指标＝随机重加权平均（**期望无偏、方差增大**）。重复点占比 **3.8% ~ 59.6%**，中位 **26.9%** |
| **训练期 BN 整体口径** | `models/layers/norm.py` | padding 让**每个文件等权**（都是 8000 点）；不补齐则按文件大小加权 → 统计口径不同 |

### 4.6 不受 padding 影响的算子

- 所有逐点卷积（`create_convblock1d/2d`）—— 逐点算子，重复点产生完全相同的重复输出
- `SegHead`（`base_seg.py:92`）—— 无池化，逐点
- feat_stats 归一化（`s3disRadar.py:208-209`，标量统计）
- `PointsToTensor`、`PointCloudRotation`、`PointCloudJitter`、`PointCloudScaling`（逐点）
- `three_interpolation`（`upsampling.py:92`）—— 3-NN + 反距离加权；**源点集 = FPS 选点，本身不含重复点**（FPS 永不选重复点）→ 只经 FPS 间接受影响，无独立重复点通道

---

## 5 实测：口径切换 ≈ 2.4pp

同一 ckpt、同一批 58 个文件、同一 eval 模式，**唯一差别 = 补齐 + `variable`**：

| ckpt | 不补齐口径 | 补齐 8000 口径 | 差 |
|---|---|---|---|
| **B E29**（padded 训练，健康模型） | 75.51 | **77.91** | **−2.40 pp** |
| B E32 | 74.91 | 77.45 | −2.54 pp |
| A E1 | 34.75 | 35.96 | −1.21 pp |
| A E67 | 17.74 | 18.77 | −1.03 pp |

→ **对健康模型，口径切换值 ≈ 2.4pp**，方向是「不补齐（test/部署口径）偏低」。
且偏差**不是常数**：由 §4.1 的密度比 `1.04~2.48×`（文件越小差越大）决定 → **test 的逐文件指标分布也失真**。

---

## 6 冗余计算量化

- **重复点占比** `(8000 − N_raw) / 8000`：**min 3.8% / median 26.9% / max 59.6%**（58 个 val 文件）
- **计算量放大** `8000 / N_raw − 1`：`N=3005` → **+166%**；中位 `N=5845` → **+37%**；`N=7837` → +2%
- **FPS 逐层放大**：stage-2 采样点数从 `N//4`（中位 1462）抬到固定 **2000** → stage-2 及以下每层的点数放大 **1.04~2.48×**
- 部署侧：因 `min_n=2024` 从不触发（§2.2），**部署不存在由 padding 带来的冗余计算**；冗余只存在于 Python 训练/验证侧

---

## 7 口径不一致的后果

1. **同 ckpt、同文件、两个口径**：val（补齐 8000）与 test（不补齐）相差 ≈ 2.4pp（§5），且逐文件偏差不均匀。
2. **`train_miou` 不能用于判断泛化**：它在 train-mode 下计算（`main.py:411` `model.train()` + `cm.update(logits...)`），即 BN 用**逐样本统计量**，被系统性抬高；`bs=1` 时会放大成灾难（train 84.9 / val 17.7 的假象）。
3. **部署 acc 与 Python val 不可直接比较**：`latency-statistics.md` 记录的部署口径 ti10 基线（fp32，10 文件 mean）是在**不补齐**口径下测的；与 Python 侧 `val`（补齐 8000）不同口径。
4. **同一模型不同文件的偏差不同**：小点云文件在不补齐口径下被采样得更稀疏 → 逐文件指标的可比性被削弱。

---

## 8 部署侧：padding 与静态 shape / 计算图捕获

### 8.1 现状：dynamic shape，逐次设置

- TRT 侧用 **dynamic optimization profile**（`deploy/trt_build.py:90-103`）：

  ```python
  profile.set_shape("pos", (1, min_n, 3), (1, opt_n, 3), (1, max_n, 3))
  profile.set_shape("x",   (1, num_features, min_n), (1, num_features, opt_n), (1, num_features, max_n))
  ```

  脚本 CLI 默认：`--min_n 2024 / --opt_n 5096 / --max_n 10000`（`trt_build.py:184-188`）。`latency-statistics.md` 记录现役 engine 的 profile 为 `min_n=2024 / opt_n=4096 / max_n=10000`。
- 每次推理前都要 `context_->setInputShape(name, dims)`（`deploy/CPP_trt4/src/trt_inference.cpp:41`），随后 `setTensorAddress`（`:53`）+ `enqueueV3`（`:63`）。
- 引擎内部注释也明确：`trt_inference.h:12` "封装 IExecutionContext + 动态形状 + enqueueV3"。

### 8.2 后果

| 受限手段 | 原因 |
|---|---|
| **静态 shape 优化**（静态 kernel 选择、常量折叠、固定 workspace） | 输入 N 每次不同，只能用 dynamic profile（min/opt/max 三档） |
| **一次捕获、复用所有输入的 CUDA Graph** | Graph 要求 shape 与地址固定；dynamic shape 下只能**逐 shape 捕获** |
| **无 `setInputShape` 的零开销提交** | 每次 `enqueueV3` 前必须按本文件 N 设 shape |

### 8.3 已经做过的验证与已否决的路线（来自 `latency-statistics.md`）

**CUDA Graph 技术可行性已 PASS**（§4.3）：`enqueue 8007µs → graphLaunch 42.4µs（188.8×）`，graph 输出 bit 级一致。但**全仓目前没有任何 CUDA Graph 落地**（grep `cudaGraph | graph_capture | cudaStreamBeginCapture` 无命中）。

**「桶化 pad 到静态 shape」已经被试过并否决**（`latency-statistics.md` 优化方向表）：

| 方法 | 结果 | 结论 |
|---|---|---|
| **桶化 pad-4608 静态 shape** | pred 一致率仅 **98.5%**（算法性差异：**FPSPrune 的 M/num_points 随 N 缩放**） | 否决（spike①） |

**这与本分析 §4.1 是同一个机理**：FPS 的采样点数 `m = N // stride` **随 N 缩放**。把 N 桶化/补齐到固定值就改变了采样密度 → 预测与变长口径不一致（pred 一致率 98.5%）。

**当前选定的路线是「逐 shape GraphPool」**（`latency-statistics.md` 方向③）：对每个真实 N（记录为「全集 2803–7467、328 唯一值、mean 5286」）各捕一张 CUDA Graph，per-ctx LRU 384 个 context 不淘汰 —— 借此绕开 graph 的「shape 烧死」约束，而**不**引入 pad。

### 8.4 三方权衡（结论）

| 方案 | 冗余计算 | 静态化能力 | 口径一致性 |
|---|---|---|---|
| **变长（当前）** | 无 | ❌ 只能动态 profile + 逐 shape GraphPool | train 补齐 8000 vs test/部署 不补齐 → **不一致（≈2.4pp）** |
| **全链路固定 N（如 8000）** | **+37%（中位），最大 +166%** | ✅ 静态 shape + 单张 CUDA Graph + 无 `setInputShape` | 需把训练/验证/测试/部署**全部**统一到同一 N，否则重复 §4.1 的密度不一致；且 `pred 一致率 98.5%` 的教训说明**改 N 会改预测** |
| **变长 + 逐 shape GraphPool（推荐）** | 无 | ⚠️ 逐 shape 捕图（已验证 bit 级一致），能打掉 host enqueue | 仍需单独解决 train/test 口径差 |

→ **核心矛盾**：`FPS 的采样点数 = N // stride` 使「预测结果」与「输入点数」强耦合，所以**任何为了让计算图静态化而固定 N 的做法，都会改变模型行为**。这也是为什么当前路线选择「保留变长、逐 shape 捕图」而不是「补齐到固定 shape」。

---

## 9 结论与建议

### 结论

1. **现状是三方口径**：训练/验证 = 补齐 8000；测试 = 不补齐；部署 = 事实上不补齐（`min_n=2024` 从不触发）。**测试与部署已自然对齐，训练侧孤立。**
2. padding 影响结果的通道明确为 5 条，其中 FPS（点数/密度）与 ball_query（`nsample` 截断）是主通道，`PointCloudXYZAlign` 的 xy 平移是唯一「不被抵消」的独立通道；**`max` 幂等、`mean` 非幂等**是理解敏感性的关键。
3. 口径切换实测 ≈ **2.4pp**，且偏差随文件大小变化（1.04~2.48×），不是常数偏移。
4. padding 的收益（`batch_size>1` + BN 健康）与代价（冗余 +37% 中位、口径不一致、阻碍静态化）都很实在。

### 建议（按代价排序）

1. **零改动：统一到「不补齐」评估** —— `test()` 不读 `voxel_max`，纯配置无法让 test 补齐。等价替代是用 **`mode=val`** 作为与训练同口径的评估；同时**在报告中标注口径**，不要用 val 与 test/部署指标互比。
2. **零改动：量化自己数据上的口径差** —— 同 ckpt 跑一次 `mode=val`（补齐）与一次 `variable=True` 的 val（不补齐）即可，本文 §5 已给出示范（77.91 vs 75.51）。
3. **若要统一到「不补齐」训练**：需 `variable=True` → 强制 `bs=1` → **必须先解决 BN 病理**（换成与 batch 无关的归一化，如 `norm: in`），否则 val 会崩。这是重训级改动。
4. **若要统一到「补齐」**：需改 `main.py` 让 `test()` 也补齐（目前它完全不读 `voxel_max`）；代价是 +37% 中位冗余，且**不会**带来静态化收益 —— 因为 §8.3 已证「固定 N 会改变预测」（pred 一致率 98.5%），除非全链路（含部署 engine）统一。
5. **延迟优化优先走「逐 shape GraphPool」而非「桶化 pad」**（`latency-statistics.md` 方向③），这是已被验证 bit 级一致且不引入冗余计算的路径。

---

## 附录 A 代码锚点

| 主题 | 位置 |
|---|---|
| padding / 裁剪实现 | `openpoints/dataset/data_util.py:146-174`（`:154` `if voxel_max is not None`、`:161` `elif not variable`、`:165-167` 补齐、`:169-171` shuffle） |
| shuffle 与补齐共享同一 `if` 块 | `data_util.py:154` vs `:149`（两个独立 `if`） |
| dataset 侧参数透传 | `openpoints/utils/registry.py:287-291`（`obj_cfg.update(split_cfg)`） |
| `RadarClassi` 划分与 `crop_pc` 调用 | `openpoints/dataset/radar/s3disRadar.py:62-68`、`:189-191` |
| val/test 文件列表构造 | `s3disRadar.py:62-68`、`examples/segmentation/main.py:59-65` |
| `mode` 分支与 test 提前返回 | `main.py:193`（val_loader）、`:226-242`（test）、`:262`（train_loader） |
| `train_one_epoch` / `validate` | `main.py:411`、`main.py:475` |
| FPS kernel | `openpoints/cpp/pointnet2_batch/src/sampling_gpu.cu:120-145` |
| FPS 调用点 | `openpoints/models/backbone/hpenetv2.py:143`、`:160` |
| ball_query kernel | `openpoints/cpp/pointnet2_batch/src/ball_query_gpu.cu:16-49` |
| grouper 分派（`nsample=None` → `GroupAll`） | `openpoints/models/layers/group.py:338`（`create_grouper`）、`:345-352`（分派）、`:351`（`GroupAll()`）、`:258`（`class GroupAll`） |
| max 池化 | `hpenetv2.py:141`、`:199`、`:346` |
| `dh/aj/fj`（未被使用） | `hpenetv2.py:61-83` |
| `FeaturePropogation.pool = mean`（死代码） | `hpenetv2.py:241`、decoder 调用 `:590-598` |
| BAFM 点维均值 | `openpoints/models/layers/channel_attention.py:34-35` |
| HPE 的 BN | `openpoints/models/layers/position_encoding.py:18`、`conv.py:24-37`（`:30` `dimension='2d'`）、`norm.py:74-89` |
| `PointCloudXYZAlign` | `openpoints/transforms/point_transformer_gpu.py:76-90` |
| 部署侧 pad / split / trim | `deploy/CPP_trt4/src/pipeline.cpp:9-13`、`include/pipeline.h:40-41,47-48`、`src/main.cpp:29-30` |
| TRT dynamic profile | `deploy/trt_build.py:90-103`、`:184-188` |
| 逐次 setInputShape | `deploy/CPP_trt4/src/trt_inference.cpp:41`、`:53`、`:63`、`include/trt_inference.h:12` |
| ONNX 导出（dummy N + dynamic_axes） | `deploy/onnx_export.py:78-113`、`:234-235` |
| 指标 | `openpoints/utils/metrics.py:51`（`ConfusionMatrix`）、`:176`（`get_mious`） |

## 附录 B 数据来源与复现

- **数据集**：`data/RadarClassi/radarfullwl/raw`，339 个 PLY（train 281 / val-test 58）
- **点云点数**：全 339 文件 `N_raw ∈ [3005, 7837]`，中位 5845，mean 5716（读 PLY header `element vertex`）
- **val/test 共享性**：两边都是 `np.random.seed(100)` + `[int(n*0.83):]`（§3）
- **分辨率/截断实验**：`/tmp/opencode/vm/bq.py`、`bq2.py`、`bq3.py`（numpy 逐点复刻 ball_query kernel 语义）
- **口径切换实测**：`/tmp/opencode/vm/exp2.py`（同 ckpt、同 loader、切 `variable`，BN=running）
- **BN 对照实验**：`/tmp/opencode/vm/exp.py`、`expC.py`（切 BN 用 running stats / batch stats）
- **分离实验运行**：`log/radar/radar-train-hpenet-ll-ngpus1-20260917-105906-nqjLSX4SfyYg4SMU6PNANT`（补齐 8000 + bs=1 + seed 2218，3 epoch）
- **延迟与 CUDA Graph 参考**：`latency-statistics.md`（§4.3 冒烟、§六 方向③、优化方向表中的「桶化 pad-4608」否决记录）
- **未修改任何仓库代码**：本轮分析只读日志/ckpt/源码，脚本全部写在 `/tmp/opencode/vm/`