# 去掉体素化 — 调查与结论

## 1. 体素化的参数及其作用

### 直接相关参数

| 参数 | 位置 | 作用 | 当前值 |
|---|---|---|---|
| `voxel_size` | `dataset.common` | 体素边长，控制去重粒度和测试投票子云数。`voxelize(coord, v)` = floor + fnv hash + unique 取随机点 | 0.008 |
| `test_mode` | cfg 顶层 | 测试投票策略：`multi_voxel`（默认，每个体素选一个点做多次推理）或 `nearest_neighbor`（单次推理 + 最近邻插值） | 未设（默认 multi_voxel） |
| `presample` | `dataset.train/val/test` | 是否预建体素化缓存 pkl | False |

### 看似相关但独立于体素化的参数

| 参数 | 作用 | 与体素化的关系 |
|---|---|---|
| `voxel_max` | 裁剪/补齐到固定点数，用于 batch 等长 | 独立机制，在 `crop_pc`（`data_util.py`）的不同 `if` 块中（L154 vs L149），互不依赖 |

### 体素化在各阶段的作用

**训练/验证**（`crop_pc` L149-153）：同一体素内多个点只保留一个随机点（`mode=0`），本质是去重。

**测试**（`load_data` L118-140）：`mode=1` 返回每个体素的点数 `count`，`multi_voxel` 按 `count.max()` 产生子云，每个子云 = 每个体素选一个点，做多次前向推理后 logits 散播平均回原点。这是原论文为处理**数十万点大场景点云**设计的折中。

### 毫米波雷达是否需要体素化？

**结论：不需要。**

| 事实 | 数值 |
|---|---|
| 单文件点数 | 3005–7837（中位 5845） |
| 单次前向 7837 点 GPU 内存 | 完全够（引擎 profile max_n=10000） |
| 当前 0.008 去重点数 | 55 个点（0.003%），可忽略 |
| 当前 0.008 多子云文件数 | 45/339，做 2-3 倍多余推理 |
| 无精确重复坐标 | 339 文件全零 |
| 全库最小点对 Chebyshev 距离 | 3.2e-4 m（0.32mm） |

雷达点云稀疏，点数少，去重近乎无效，投票纯属多余。

---

## 2. 能否将 voxel_size 设为空？

**不能直接设 `null` 或 `0`，但设一个极小值 `0.0001`（1e-4 m）即可等效于"不去体素化"，零代码改动。**

| 值 | train/val | test | init f-string | 结论 |
|---|---|---|---|---|
| `null` (None) | `crop_pc` 正常跳过（`if voxel_size` 为假） | `load_data` 显式支持 None（全量单次推理） | **崩** `TypeError: {None:.3f}` | ❌ |
| `0` | `crop_pc` 跳过（falsy） | **崩**：`coord/0` 除零 | 正常 `"0.000"` | ❌ |
| `0.0001` | 走 `voxelize` 但零合并 | `count.max()==1` 单次推理 | 正常 `"0.000"` | ✅ |

### 崩溃原因

- **`null` 崩在 `RadarClassi.__init__`**（`s3disRadar.py:73-74`）：`f'radar_{split}_area{test_area}_{voxel_size:.3f}_{str(voxel_max)}.pkl'` 无条件执行，`f"{None:.3f}"` → `TypeError`。`presample=False` 也躲不过——filename 构造在 `if presample` 之前。
- **`0` 崩在测试路径**（`main.py:123`）：`voxelize(coord, 0)` → `np.floor(coord / 0)` 除零。

### 为什么选 0.0001

用真实 `voxelize` 逻辑（floor + fnv hash）逐档验证全部 339 个文件：

| voxel_size | 有多子云的文件数 | 最大子云数 | 总合并点数 |
|---|---|---|---|
| 0.02 | 250 | 3 | 523（0.027%） |
| 0.008（当前） | 45 | 3 | 55（0.003%） |
| 0.001 | 1 | 2 | 1 |
| **0.0001** | **0** | **1** | **0** |
| 1e-5 / 1e-6 / 1e-7 | 0 | 1 | 0 |

再小（1e-5/1e-6）也能过，但无额外收益且有 float32 精度风险：`coord/v` 在 float32 中计算，`v=1e-6` 时商最大 6.08e8，ul=64，极远处可能因舍入误合并。`v=1e-4` 时商最大 6.08e6 < 2²³=8.39e6，每个商都是 float32 精确整数，floor 无抖动。

### 效果

- 训练/验证：不丢任何点（0.003% → 0%）
- 测试：单子云全量推理（45 个多余文件 → 0 个），无 voxel 投票开销

---

## 3. voxel_max 的作用

### 在各阶段的行为

**训练（train）**：

1. N ≥ 4608 时：空间锚点裁剪——随机选一个点作锚点，保留距其最近的 4608 点，远处区域整片丢弃。
2. N < 4608 且 `variable=False`（当前配置）：随机重复采样补齐到 4608，再 shuffle。
3. 净效果：每个样本恒定 4608 点 → 默认 stack collate 组成 (B, 4608, C) 的 batch。

**验证（val）**：同 train，但锚点固定为中心点（`N//2`）→ 确定性裁剪。补齐和 shuffle 仍是随机的。

**测试（test）**：**无作用**。测试不走 `__getitem__`/`crop_pc`，`load_data` 全程未引用 `voxel_max`。test 的 `voxel_max: null` 只出现在 presample 缓存的 pkl 文件名中（`presample=False`，仅构造不使用）。

### 数据分布

339 文件 raw 点数：min=3005, p25=4557, median=5845, p75=6737, p95=7235, max=7837。

| voxel_max | 被裁剪的文件 | 被补齐的文件 | 中位信息损失 |
|---|---|---|---|
| 4608（当前） | 252/339（74%） | 87/339（26%） | 丢 21%（5845→4608） |
| 8000 | 0/339 | 339/339（100%） | 0 裁剪，中位补齐 2155 点（27%） |
| null | 0/339 | 0/339 | 0 裁剪 0 补齐 |

---

## 4. 去掉 voxel_max 对 BN 的影响

### 当前配置

- `norm: 'bn'` → `BatchNorm2d`（`norm.py:77`），作用于 `(B, C, 1, N)` 形状
- BN 在**每个 channel 上跨 batch 维度和空间维度**计算均值和方差
- 当前 `batch_size=8`：BN 统计量来自 8 个样本的所有点（约 8×4608=36864 个点）

### 不同方案对 BN 的影响

| 方案 | voxel_max | batch_size | BN 统计量来源 | BN 影响 |
|---|---|---|---|---|
| A（当前） | 4608 | 8 | 8×4608 点 | 正常 |
| B（推荐） | 8000 | 8 | 8×8000 点 | **正常**（补齐不改变分布，BN 统计量一致） |
| C | null | 1 | 1×N 点 | 能跑，单 batch 方差估计噪声略大，收敛稍慢，但 eval 用累积 running stats 不受影响 |

### 补齐对训练的影响

`crop_pc`（`data_util.py:163-167`）的补齐是**随机重复采样 + shuffle**。补齐点与原始点坐标/特征完全相同 → 分布不变 → BN 统计量不变。补齐点在 FPS 中浪费步骤但选出的唯一坐标与原始一致；在 ball query 中产生重复邻域但特征值不变；在梯度反向传播中相当于给原始点随机加权（每个 epoch 不同），不偏置模型。

### BN vs IN 和部署影响

- **BN 在 TRT 中**：被 conv+BN fusion 吸收进 conv 权重，**最终 engine 中完全消失，零额外开销**
- **IN 在 TRT 中**：`InstanceNormalization` 算子无法融入 conv，作为独立算子保留在图中
- 40-50 个 IN 层额外开销约 < 0.3ms，不是瓶颈，但确实有部署代价
- **结论：保留 BN，不换 IN。** 部署 BN folding 是"免费"优化，无需为它的训练噪声去付出部署代价

---

## 5. 推荐方案

### 最小改动：仅去体素化

```yaml
# cfgs/radar/default.yaml — 改一行
dataset:
  common:
    voxel_size: 0.0001   # 原 0.008
```

效果：训练不丢点（0.003% → 0%），测试单子云全量推理。voxel_max=4608 裁剪保持不变。

### 推荐方案：去体素化 + 全点输入

```yaml
# cfgs/radar/default.yaml — 改一行
dataset:
  common:
    voxel_size: 0.0001
  train:
    voxel_max: 8000       # 原 4608 → 覆盖所有文件（max N=7837）
  val:
    voxel_max: 8000       # 原 4608
```

| 维度 | 改动 | 影响 |
|---|---|---|
| yaml | 3 行 | 0 代码改动 |
| 训练 | 全点输入（0 裁剪） | 信息完整，补齐 27% 冗余但不偏置 |
| 验证 | 全点输入（0 裁剪） | 确定性评估（固定锚点 N//2，N 均 < 8000 不触发裁剪） |
| 测试 | 单子云全量推理 | 无 voxel 投票开销 |
| BN | 正常 | batch_size=8 不变 |
| 部署 | 零影响 | BN 仍被 conv folding 吸收 |

### 备选（如果未来有 > 8000 点的文件）

把 `voxel_max` 调大即可。`voxel_max` 只是一个上限，小于它的不会被裁。

---

## 6. 涉及的关键代码位置

| 文件 | 行号 | 内容 |
|---|---|---|
| `cfgs/radar/default.yaml` | 7, 10, 15 | voxel_size, voxel_max（train/val） |
| `openpoints/dataset/radar/s3disRadar.py` | 72-74 | init 的 f-string（voxel_size=None 崩溃点） |
| `openpoints/dataset/radar/s3disRadar.py` | 171-210 | `__getitem__` → `crop_pc` |
| `openpoints/dataset/data_util.py` | 127-143 | `voxelize()`：floor + fnv hash + unique |
| `openpoints/dataset/data_util.py` | 146-174 | `crop_pc()`：体素去重(L149) + 裁剪/补齐(L154-172) |
| `examples/segmentation/main.py` | 118-140 | 测试 `load_data`：voxel_size 驱动子云划分 |
| `examples/segmentation/main.py` | 641-707 | 测试投票：multi_voxel 散播平均 / nearest_neighbor 插值 |
| `openpoints/dataset/build.py` | 13-27 | `concat_collate_fn`（变长 batch 方案，需 batch_size=1） |
| `openpoints/dataset/build.py` | 73-75 | collate_fn 选择逻辑 |
| `openpoints/models/layers/norm.py` | 74-115 | norm 注册表：bn/bn2d/in/in1d 等 |
| `deploy/onnx_export.py` | 96, 133 | IN 导出注意事项，onnx-simplifier 折叠 IN 常量 |
| `deploy/trt_utils.py` | 85 | `NATIVE_INSTANCENORM` 插件（注释掉，IN 备选） |
| `deploy/onnx_backend.py` | 19-44, 378-381 | `ONNXInstanceNorm1d` 和 IN 导出说明 |