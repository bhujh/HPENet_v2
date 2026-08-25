# FlashFPS Prune 填充语义实验：升序 vs 随机

**日期**: 2026-08-19
**目标**: 验证 FlashFPS prune 的 2pp acc 掉幅是否由「升序填充」导致（vs 论文「随机填充」）。
**口径**: ti10（sorted 后 20% 尾部前 10 文件 0000068..0000077），复刻 `deploy/trt_inference.py` 的 preprocess/subcloud/voxel-voting（scatter-mean）协议，仅把 TRTSession 换成 torch 模型 forward。
**模型**: `cfgs/radar/hpenet-ll.yaml` + `radar-train-hpenet-ll-ngpus1-20260812-201051-gHBQ4DMy5jP2fZfkGkFcue_ckpt_best.pth`（BN 修复后），eval + CUDA L20。

## 变体定义

| 变体 | L1 (encoder.encoder.1.0) | L2-4 (encoder.encoder.{2,3,4}.0) |
|------|--------------------------|----------------------------------|
| baseline | 原样 `furthest_point_sample`（精确 FPS 4 级） | 原样 |
| ascending k | prune + **升序填充**（严格复现 kernel 语义） | arange 前缀 |
| random k | prune + **随机填充**（固定 seed=42 的 generator） | arange 前缀 |

L1 prune 语义与 `deploy/trt_plugins/src/samplefps_kernel.cu`（launcher + iterate kernel 尾声段）逐行对应：

```
N = xyz.shape[1]; M = npoint (= N//stride = N//2)
N_points = int(keep_rate * N);  sample_rate = N // M
num_points = N_points // sample_rate
idx_fps = furthest_point_sample(xyz[:, :N_points], num_points)   # openpoints CUDA，seed=idx0
尾部: [0, M) 内未选中 → ascending 取升序前 (M-num_points) 个；random 从 [0,N_points) 内未选中
      随机抽 (M-num_points) 个（前缀不够补 [N_points, M)）；num_points==0 → arange(M)
输出 = cat([idx_fps, 尾部])
```

ascending 填充输出与 kernel `bidx[num_points+pos]=i (i∈[0,M) 未选中升序, pos<T)` 完全一致。random 变体每文件重置 seed=42（重复运行逐文件 acc 完全一致，已验证可复现）。

## 结果

| 变体 | 0000068 | 0000069 | 0000070 | 0000071 | 0000072 | 0000073 | 0000074 | 0000075 | 0000076 | 0000077 | **mean** | TRT 锚点 | 偏差 |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|----------|------|
| baseline | 0.9854 | 0.9446 | 0.9862 | 0.9536 | 0.9826 | 0.9596 | 0.9749 | 0.9891 | 0.9801 | 0.9843 | **0.9741** | 0.9741 | 0.0000 |
| ascending k0.75 | 0.9814 | 0.9425 | 0.9798 | 0.9522 | 0.9803 | 0.9569 | 0.9752 | 0.9823 | 0.9743 | 0.9818 | **0.9707** | 0.9707 | 0.0000 |
| ascending k0.5 | 0.9736 | 0.9373 | 0.9680 | 0.9467 | 0.9702 | 0.9572 | 0.9632 | 0.9673 | 0.9609 | 0.9668 | **0.9611** | 0.9612 | -0.0001 |
| ascending k0.25 | 0.9674 | 0.9349 | 0.9572 | 0.9439 | 0.9573 | 0.9519 | 0.9618 | 0.9537 | 0.9483 | 0.9586 | **0.9535** | 0.9535 | 0.0000 |
| random k0.75 | 0.9817 | 0.9425 | 0.9801 | 0.9517 | 0.9798 | 0.9579 | 0.9761 | 0.9821 | 0.9752 | 0.9820 | **0.9709** | — | — |
| random k0.5 | 0.9741 | 0.9375 | 0.9680 | 0.9467 | 0.9700 | 0.9574 | 0.9632 | 0.9671 | 0.9614 | 0.9664 | **0.9612** | — | — |
| random k0.25 | 0.9687 | 0.9373 | 0.9614 | 0.9449 | 0.9581 | 0.9519 | 0.9606 | 0.9526 | 0.9492 | 0.9586 | **0.9543** | — | — |

## 交叉验证锚点（必须满足）

- **baseline mean = 0.9741** vs 锚点 0.9741 —— 完全一致（torch 侧管线与 task 6 TRT 口径逐文件级吻合）。
- **ascending k0.25 mean = 0.9535** vs TRT flashfps k0.25 = 0.9535 —— 完全一致（±0.0001 内）。k0.75=0.9707(锚0.9707)、k0.5=0.9611(锚0.9612)。**torch 模拟逐字复现了 kernel 的升序填充语义**，交叉验证通过。

## 结论

**掉幅不是由「升序填充」导致。** 随机填充在三个 keep_rate 档位上均无实质改善：

| keep_rate | ascending mean | random mean | Δ(random−ascending) |
|-----------|----------------|-------------|----------------------|
| 0.75 | 0.9707 | 0.9709 | +0.0002 |
| 0.5 | 0.9611 | 0.9612 | +0.0001 |
| 0.25 | 0.9535 | 0.9543 | **+0.0008** |

- **随机填充无法把 k0.25 拉回 ≥0.97**（0.9543 仍在 0.95 量级，距 baseline 0.9741 差 −1.98pp）。
- 掉幅主体来自 prune 本身：候选集被截断到前缀 [0, N_points) + FPS 只跑 num_points(<M) 轮 + L2-4 降级为 arange 前缀（cache 图）。填充顺序（升序 vs 随机）只贡献 ≤0.001 的差异。
- 论文「p=0.75（留 25%）损失 <0.3%」在扫描序雷达数据上不成立：k0.25 时掉幅约 2.0pp，远超论文声称的 0.3%。这一差距的根因是候选剪枝与 cache 图本身，而非尾部填充的排列顺序。

## 附注

- 实现细节：实验脚本全部位于 `/tmp/opencode/fill_exp/`（`samplers.py` + `run_exp.py`），未改动仓库任何现役文件（kernel/插件/onnx_backend/模型代码），无 git 操作。
- 每个 variant 独立重新加载模型 + checkpoint，patch 只改 4 个 `SetAbstraction.sample_fn`（`encoder.encoder.{1..4}.0`），权重/其它模块全不动。
- random 变体 seed：torch.Generator(device=cuda) 每文件 manual_seed(42)，重复运行逐文件 acc 逐位一致（可复现）。
- 交叉验证数值细节：ascending 模拟与 TRT 在 ti10 上的 mean 偏差 ≤ 0.0001（k0.5 差 0.0001，其余完全相等），说明 kernel 语义复刻与 ti10 管线两处均正确。
