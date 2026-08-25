# F3 QA：fps / samplefps / flashfps 三算法真实动手验证

- 时间: 2026-08-19
- 环境: python=/home/wangpeng/miniforge3/envs/hpenet/bin/python | torch=2.2.2+cu118 | TRT=8.6.1.6 | GPU=NVIDIA L20（CUDA_VISIBLE_DEVICES=1，GPU0 被占用）
- 引擎: `deploy/fps_algo_{fps,samplefps,flashfps}_fp32.engine`（task-6 构建产物，完整网络 engine，pos(1,-1,3)/x(1,5,-1) → output(1,2,-1)）
- 数据: `data/RadarClassi/radarfullwl/raw`（339 文件）

## 1. 三算法完整推理（真实测试文件）

### 1.1 `deploy/trt_inference.py` 全管线跑（--num_files 3，测试集首 3 文件 0000068/69/70）

| engine | 0000068 acc | 0000069 acc | 0000070 acc | mean acc | per-file 耗时 | exit |
|---|---|---|---|---|---|---|
| fps fp32 | 0.9854 | 0.9446 | 0.9862 | 0.9721 | 0.040s | 0 |
| samplefps fp32 | 0.9854 | 0.9446 | 0.9862 | 0.9721 | 0.201s | 0 |
| flashfps fp32 | 0.9814 | 0.9425 | 0.9798 | 0.9679 | 0.080s | 0 |

- 三 engine 均完整跑通 voxel-voting 推理，无崩溃；acc 与 task-6 对应值一致（samplefps == fps 逐文件，平局归因；flashfps 近似语义）
- 结论行 `Done!` + `EXIT_CODE=0`

### 1.2 输出 shape + 简单统计（等价推理路径，每 engine 独立进程）

| engine | subcloud 数(3文件) | per-subcloud 输出 shape | 合并 logits shape | 全部有限 | logits min/max | mean/std | per-file acc mean |
|---|---|---|---|---|---|---|---|
| fps | 15 | (1,2,N_sub) 首个 [2,3523] | (2,56443) | True | -30.56 / 29.83 | -0.041 / 7.135 | 0.9721 |
| samplefps | 15 | (1,2,N_sub) 首个 [2,3523] | (2,56443) | True | -30.57 / 29.84 | -0.041 / 7.135 | 0.9721 |
| flashfps | 15 | (1,2,N_sub) 首个 [2,3523] | (2,56443) | True | -30.24 / 29.52 | -0.042 / 7.208 | 0.9679 |

- 输出 shape 正确（1,2,N）；全部 `np.isfinite` 通过；值域有限（|logits|<31）

## 2. 重跑 tests_fps_algos.py 全绿

命令: `python deploy/tests_fps_algos.py --configs fps,samplefps,flashfps_k1 --precision fp32 --n 1024,2750 --frames 2 --reps 10`

- 自检: 注入损坏索引 / stride 错配 均检出，`自检结论: ALL PASS`
- 对拍矩阵（idx一致率 / 序列精确率 / median ms）:

| 配置 | 精度 | 输入 | N=1024 | N=2750 |
|---|---|---|---|---|
| fps | fp32 | random | 100.0% / 100.0% / 0.264ms | 100.0% / 100.0% / 0.867ms |
| fps | fp32 | real | 100.0% / 100.0% / 0.265ms | 100.0% / 100.0% / 0.866ms |
| samplefps | fp32 | random | 100.0% / 100.0% / 1.379ms | 100.0% / 100.0% / 3.793ms |
| samplefps | fp32 | real | 100.0% / 100.0% / 1.307ms | 100.0% / 100.0% / 4.082ms |
| flashfps_k1 | fp32 | random | 100.0% / 100.0% / 1.357ms | 100.0% / 100.0% / 3.795ms |
| flashfps_k1 | fp32 | real | 100.0% / 100.0% / 1.293ms | 100.0% / 100.0% / 4.080ms |

- `总耗时 13.8s`，`=== 结论: ALL READY CONFIGS PASS ===`，EXIT=0
- 注：flashfps_k075 未在本次运行中（任务指定仅 fps,samplefps,flashfps_k1）；其低一致率是预期（harness 用全 N 精确 FPS 判据，prune 语义不同），权威判据见 §3

## 3. 重跑 tests_flashfps_prune.py ALL PASS

命令: `python -u deploy/tests_flashfps_prune.py --seed 7 --n 1024 --frames 2`

- keep_rate=1.0: FlashFPS==SampleFPS 逐索引 True，shape=(1,256) PASS
- keep_rate<1（0.75/0.5/0.25，N=1024 M=256）: shape_ok / prefix_exact / tail_ascending / verify_fps_prefix 全 True，PASS
- 确定性（N=2750, k=0.75/0.5）: run1==run2 True PASS
- 退化（k=0.05: N_points=51<M=256 → full_M_perm；k=0.001: idx==arange(M)）: PASS
- 真实雷达帧 k=0.75 抽查: prefix_exact=True tail_ascending=True verify_fps_prefix=100.0% PASS
- `=== 结论: ALL PASS ===`，EXIT=0

## 4. nsys 抽查（fps engine，1 文件 0000068）

命令: `nsys profile -t cuda -o /tmp/opencode/f3_fps.nsys-rep python deploy/trt_inference.py --engine deploy/fps_algo_fps_fp32.engine --num_files 1 --warmup 0`
（warmup 0 消除随机输入的 FPS kernel 干扰；trace 大小 463KB，nsys 2022.4.2）

`nsys stats --report gpukernsum` 结果（FPS kernel 段）:

| kernel | instances | total (ns) | avg (ns) | med (ns) |
|---|---|---|---|---|
| furthest_point_sampling_kernel<1024> | 12 | 20,297,122 | 1,691,427 | 1,690,920 |
| furthest_point_sampling_kernel<512> | 6 | 1,859,161 | 309,860 | 309,711 |
| furthest_point_sampling_kernel<256> | 6 | 723,615 | 120,603 | 120,608 |
| **FPS 合计** | **24** | **22,879,898** | — | — |

- 0000068 子云数 = 6（各 N=3523）→ 24 launch = 6 子云 × 4 采样级，与 evidence（×52/13 子云 = 4/子云）结构一致
- **per-subcloud FPS = 22.88ms / 6 ≈ 3.81ms**，与 evidence 8.1ms/subcloud 同量级（evidence 的 N≈3800-4200 更大，本文件 N=3523 更小 → 更短，一致）
- FPS kernel 存在性 + 耗时量级均确认

## 5. 观察（NOTES）

1. **同进程多 engine 共存段错误**：将 fps+samplefps+flashfps 三个 TRTSession 放同一 python 进程连续推理时，flashfps 阶段 core dump（`timeout: the monitored command dumped core`）。每 engine 独立进程（trt_inference.py 原生路径 + QA 脚本）均稳定无崩溃。疑为 FlashFPS 插件在多 engine 同进程场景的全局状态/上下文冲突，建议后续插件侧核查；不阻塞三算法正确性结论。
2. `tests_fps_algos.py` 重跑会按其设计覆写 `.omo/evidence/task-5-fps-samplefps-flashfps.md`（本次已重新生成，内容同本文件 §2）。
3. cuDNN 版本警告（TRT 链接 8.9.0、加载 8.7.0）为环境既有，不影响结果。

## 6. Verdict

**PASS-WITH-NOTES**

- 三 engine 完整推理 ✓（exit 0，shape 正确，值域有限，acc 与 task-6 一致）
- tests_fps_algos.py 三配置全绿 ✓（ALL READY CONFIGS PASS）
- tests_flashfps_prune.py ALL PASS ✓
- nsys 抽查 ✓（furthest_point_sampling_kernel 存在，~3.8ms/subcloud，与 evidence 同量级）
- 备注：同进程多 engine 共存时 flashfps 段错误（各 engine 独立进程正常），见 §5.1
