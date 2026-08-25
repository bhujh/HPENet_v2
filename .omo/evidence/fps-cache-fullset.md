# fps vs fps_cache 全量数据集实测证据

- 时间: 2026-08-19
- 数据: `data/RadarClassi/radarfullwl/raw/*.ply` 全量 339 文件（含训练/验证/测试划分之外）
- 引擎: deploy/fps_algo_fps_{fp32,fp16}.engine、fps_algo_fps_cache_{fp32,fp16}.engine（现役 vs 等价缓存实现）
- GPU: NVIDIA L20 (46GB, 空闲) | TRT 8.6.1.6 | CUDA 11.8 | PyTorch 2.2.2
- 口径: 单文件 wall-time = preprocess(voxelize) + subcloud prep + engine + voxel-voting scatter-mean；engine 延迟 = H2D+compute+D2H (CUDA event, 逐 subcloud 累加)；acc = voting 后 per-point argmax vs PLY label
- 预处理镜像 deploy/trt_inference.py: voxel_size=0.3, gravity_dim=2, min_n pad=1024, feat_stats_area5.pth 归一化, seed(100) shuffle
- 每配置独立进程（规避同进程多 engine 段错误）

## CSV 数据

- 落盘目录: `/tmp/opencode/fps_cache_fullset/out/`
- `fps_fp32.csv` — 339 行 × 列 [file, N, n_subcloud, pipe_ms, engine_ms, engine_pct, engine_min_ms, engine_max_ms, acc]
- `fps_cache_fp32.csv` — 339 行 × 列 [file, N, n_subcloud, pipe_ms, engine_ms, engine_pct, engine_min_ms, engine_max_ms, acc]
- `fps_fp16.csv` — 339 行 × 列 [file, N, n_subcloud, pipe_ms, engine_ms, engine_pct, engine_min_ms, engine_max_ms, acc]
- `fps_cache_fp16.csv` — 339 行 × 列 [file, N, n_subcloud, pipe_ms, engine_ms, engine_pct, engine_min_ms, engine_max_ms, acc]
- 逐文件 argmax pred: `/tmp/opencode/fps_cache_fullset/preds/{config}/`
- PLY 缓存 (coord/feat/label npz): `/tmp/opencode/fps_cache_fullset/cache/` (PLY 读取 mean 36.5ms/文件, 建缓存 14.7s)

## 精度汇总 (per-file per-point acc)

| config | mean | median | P99 | min |
|---|---|---|---|---|
| fps_fp32 | 0.9569 | 0.9597 | 0.9897 | 0.8822 |
| fps_cache_fp32 | 0.9569 | 0.9597 | 0.9897 | 0.8824 |
| fps_fp16 | 0.9569 | 0.9598 | 0.9897 | 0.8822 |
| fps_cache_fp16 | 0.9569 | 0.9597 | 0.9897 | 0.8822 |

## fps vs fps_cache 逐文件一致性 (argmax pred 逐点比较)

- fp32: 完全一致文件 321/339；逐点匹配率 mean=99.9990%, 最低文件 99.9682% (max dev 0.0318%)
- fp16: 完全一致文件 194/339；逐点匹配率 mean=99.9898%, 最低文件 99.9135% (max dev 0.0865%)

fp32 vs fp16 同算法逐点一致性:
- fps: 完全一致 168/339, mean=99.9867%
- fps_cache: 完全一致 168/339, mean=99.9870%

## 时序汇总 (每文件 ms)

| config | pipe mean | pipe med | pipe P99 | eng mean | eng med | eng P99 |
|---|---|---|---|---|---|---|
| fps_fp32 | 72.59 | 69.31 | 134.46 | 63.31 | 60.06 | 119.61 |
| fps_cache_fp32 | 55.33 | 52.15 | 102.57 | 47.42 | 45.06 | 89.90 |
| fps_fp16 | 72.27 | 68.63 | 134.81 | 62.50 | 59.10 | 118.07 |
| fps_cache_fp16 | 56.34 | 53.39 | 106.17 | 47.00 | 44.47 | 88.96 |

### engine 延迟 per-subcloud (engine_ms / n_subcloud, 全集分布)

| config | per-subcloud eng mean | median | P99 |
|---|---|---|---|
| fps_fp32 | 10.107 | 10.475 | 14.312 |
| fps_cache_fp32 | 7.575 | 7.822 | 10.636 |
| fps_fp16 | 9.985 | 10.415 | 14.091 |
| fps_cache_fp16 | 7.516 | 7.843 | 10.503 |

## 提速比 fps vs fps_cache

| precision | pipe med ratio (fps/fps_cache) | pipe 提速 | eng med ratio | eng 提速 |
|---|---|---|---|---|
| fp32 | 1.306 | 23.4% | 1.338 | 25.3% |
| fp16 | 1.279 | 21.8% | 1.332 | 24.9% |

## fp16 数值异常

- 全部 4 配置 × 339 文件: 0 个 NaN/Inf logit subcloud（fp16 无溢出）

## 已知口径差异

- 训练文件 (0000001-00000N) 语义与 test 划分文件不同，cls1 比例更高，单文件 acc 更低（min ~0.882）
- ti10 锚点 (test 20% 尾部 10 文件 0000068-77) 的 7.22/5.49ms 为 **per-subcloud** engine 延迟（49 subcloud runs）
- 校准：本脚本在同一 10 文件上 per-subcloud engine median fps=7.340ms / fps_cache=5.625ms，与锚点一致（-23.7% vs 锚点 -24.0%）——计时口径对齐
- 全集每配置 engine 调用: 2089.0 次
- 全集 acc mean=0.9569 低于 test 子集锚点 0.9741：全集含训练/验证文件，cls1 分布更不均（单文件 cls1 比例 0.13-0.21），且含 27 个 subcloudN>6500 的大文件
