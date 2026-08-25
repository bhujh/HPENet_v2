# FPSPrune B>1 batch-stride 修复 — 证据

**日期**: 2026-08-20 | **对象**: `deploy/trt_plugins/src/fpsprune_plugin.cpp` enqueue（prune 路径）

## 背景

`fps_launcher_with_stream` 的 `n` 参数同时充当「点数」与「每 batch 的 dataset stride」。kernel 内 `dataset += batch_index * n * 3` / `temp += batch_index * n` / `idxs += batch_index * m`。prune 路径传 `n = N_points != N`，导致 B>1 时非零 batch 用 N_points 当 stride 读错偏移。

## 修复

prune 路径由单次 `fps_launcher_with_stream(B, N_points, num_points, xyz, temp, idx, stream)` 改为逐 batch 循环、每次单 batch（B=1 → kernel `batch_index == 0`），插件用真实 N 算 pointer 偏移：

```cpp
launch_fill_kernel(temp, static_cast<int>(B) * N_points, 1e10f, stream);
if (N_points > 0) {
    for (int b = 0; b < B; ++b) {
        fps_launcher_with_stream(1, N_points, num_points,
            xyz + static_cast<size_t>(b) * N * 3,
            temp + static_cast<size_t>(b) * N_points,
            idx + static_cast<size_t>(b) * M,
            stream);
    }
}
launch_fill_unselected(idx, idx, sel, B, N, N_points, M, num_points, stream);
```

- exact 路径（n==N）本就正确，不动
- `fill_unselected_kernel` 不用改（grid=B，内部 `bsel = sel + b*N` / `bidx = idx + b*M` 本就对）
- `N_points > 0` 守卫堵 `opt_n_threads(0)` 的 `log(0)` UB 边角
- workspace / serialize 均不变（engine 二进制兼容）

## 回归验证

测试脚本 `/tmp/opencode/test_fpsprune_batch.py`（golden/verify 两阶段，.so 全局加载故分进程跑）：

**矩阵**: 4 配置 × 8 点云
- `s4_k075` (stride=4, N=2750, keep_rate=0.75) — 主路径 N_points > M
- `s2_k04` (stride=2, N=5500, keep_rate=0.40) — prune 路径 N_points < M（原 bug 触发区）
- `s2_k025` (stride=2, N=5500, keep_rate=0.25) — prune 路径 N_points << M
- `s4_k10` (stride=4, N=2750, keep_rate=1.0) — exact 路径

| 验证项 | 结果 |
|---|---|
| B=1 bit 级回归（修复前 .so golden vs 修复后） | **32/32 PASS**（4 配置 × 8 点云逐字节一致） |
| B=2 逐 batch 对拍（每 batch == 单独 B=1 跑该点云） | **4/4 PASS** |
| B=3 逐 batch 对拍 | **4/4 PASS** |
| B=8 逐 batch 对拍 | **4/4 PASS** |
| 现役 FPS 零 diff（`git status` fps_kernel.cu / fps_plugin.cpp 无 M/??） | **通过** |
| ti10 acc（10 文件 0000068..0000077，fp32） | **0.9707**（== 修复前 0.970691） |
| 全量 acc（339 文件 sorted，fp32） | **0.955761**（== 修复前 0.955761 逐位一致） |
| 全量 NaN/Inf logit subcloud | 0 |

## 结论

修复后 B>1 输出与 B=1 逐 batch 完全一致，B=1 主路径逐字节不变（bit 级回归 + 双口径 acc 复测均逐位吻合）。改动仅 `fpsprune_plugin.cpp` 一处，现役 fps_kernel.cu / fps_plugin.cpp 零改动（回退通道不动）。
