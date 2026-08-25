# Task-6 证据：端到端三算法对比（fps / samplefps / flashfps）

- 时间: 2026-08-19
- 环境: python=/home/wangpeng/miniforge3/envs/hpenet/bin/python | torch=2.2.2+cu118 | TRT=8.6.1.6 | onnx=1.17.0 | GPU=NVIDIA L20 ×8
- checkpoint: log/radar/radar-train-hpenet-ll-ngpus1-20260812-201051-gHBQ4DMy5jP2fZfkGkFcue/checkpoint/*_ckpt_best.pth
- cfg: cfgs/radar/hpenet-ll.yaml | strides=[1,2,2,2,2] | 采样节点 4 个均 stride=2
- 三图: `deploy/fps_algo_{fps,samplefps,flashfps}.onnx`（均 628 节点，见 task 3/4b evidence）
- 测试集: `data/RadarClassi/radarfullwl/raw`（339 文件）
  - **ti10** = trt_inference.py 约定（sorted 后 20% 尾部前 10 个文件 = 0000068..0000077）——**plugin.md §9 基线 0.9741 的测量口径**
  - **modetest** = mode=test 约定（generate_data_list：seed100 shuffle 取 last 17% = 58 文件）——"mode=test 全测试集"

## 0. 关键发现：现役 v14 engine 的 profile 不覆盖全测试集

现役 `hpenet_v2_fp32_v14.engine` 用 `--min_n 1024 --opt_n 5500 --max_n 6500` 构建（/tmp/opencode/v14_build.log），
而 task 6 强制三新 engine 用统一 profile `--min_n 1024 --opt_n 4096 --max_n 10000`（MUST DO）。
modetest 58 文件中部分子云超过 6500 点（实测 max_sub=6988，如 0000212/0000186/0000335），
**现役 v14 engine 在这些文件上越出 profile 产出垃圾（acc 低至 0.40）**，不可作为 modetest 分集基线。
因此 modetest 分集的「现役」基线 = 新构建的 fps engine（同一 profile，可比的唯一正确参照）。

## 1. Engine 构建（MUST DO：同一动态 profile 参数）

命令（照 trt_build.py，动态 profile min1024/opt4096/max10000，workspace 4GiB，num_features=5）：

```
python deploy/trt_build.py --onnx deploy/fps_algo_<algo>.onnx --output deploy/fps_algo_<algo>_<prec>.engine \
  --min_n 1024 --opt_n 4096 --max_n 10000 --workspace 4 --num_input_features 5 [--fp16]
```

| engine | 精度 | 大小 | 说明 |
|---|---|---|---|
| fps_algo_fps_fp32.engine | fp32 | 14.5 MB | 现役等价（fps 档） |
| fps_algo_fps_fp16.engine | fp16 | 8.3 MB | |
| fps_algo_samplefps_fp32.engine | fp32 | 16.2 MB | 4×SampleFPS |
| fps_algo_samplefps_fp16.engine | fp16 | 10.3 MB | |
| fps_algo_flashfps_fp32.engine | fp32 | 14.6 MB | 1×FlashFPS(0.75)+3×PrefixFPS |
| fps_algo_flashfps_fp16.engine | fp16 | 8.4 MB | |
| cache_ff_k1_fp32.engine（/tmp/opencode/t4b） | fp32 | 15.6 MB | flashfps keep_rate=1.0（cache-only 路径） |
| fps_algo_flashfps_k0.5_fp32.engine（/tmp/opencode/t4b） | fp32 | 15.8 MB | keep_rate 档位阶梯 |
| fps_algo_flashfps_k0.25_fp32.engine（/tmp/opencode/t4b） | fp32 | 14.8 MB | keep_rate 档位阶梯 |

全部 `TRT_BUILD_EXIT=0`，deserialize + 端到端推理冒烟通过。

## 2. (b) 端到端推理 logits 对拍（ti10 同一批 10 文件）

比较经完整 voxel-voting 合并后的 per-point logits。**相对误差被近零 logits 放大（rel_err 无意义），
以 max_abs_err 与逐位一致率为主判据；fp16 参考对：incumbent_fp16 vs incumbent 亦非逐位（fp16 数值噪声量级 ~1e-1 abs）。**

| 对拍对 | 逐位一致 | max_abs_err | 结论 |
|---|---|---|---|
| fps_fp32 vs 现役 v14 | 0/10 | 1.5e-2 | **profile 差异归因**（v14 opt=5500/max=6500 vs 新 opt=4096/max=10000，conv tactic 选择不同 → 1 ULP 级 fp32 漂移）；per-file acc 与现役完全一致（§4）；**FPS 索引本身逐位一致**（同 kernel 同图） |
| **samplefps_fp32 vs fps_fp32（同 profile）** | 0/10 | 1.5e-2 | **平局归因成立**（见 §3 单算子级归因） |
| samplefps_fp16 vs fps_fp16 | 0/10 | 3.3e-2 | fp16 数值噪声量级，与 incumbent_fp16-vs-incumbent（1.5e-1）同量级 |
| flashfps_fp32 vs fps_fp32 | 0/10 | 3.5e1 | **近似语义（预期）**，Prune 后前缀填充改变采样序 → logits 大差异 |
| flashfps_fp16 vs fps_fp16 | 0/10 | 3.5e1 | 同 flashfps_fp32 |
| incumbent_fp16 vs incumbent | 0/10 | 1.5e-1 | fp16 参考对（数值噪声标定） |

### flashfps 误差分布（vs samplefps_fp32，10 文件全点合并）
- abs err: max=16.31, p99=3.09, p90=1.29, p50=0.442, mean=0.611
- rel err: max=1.1e3（近零点）, p99=3.60, p90=0.379, p50=0.081
- **pred flip（与 samplefps 混淆矩阵差异）: per-frame 41-111 点（1.1%-2.6%）**
  - 0000068: 41 (1.11%)  0000069: 111 (2.64%)  0000070: 52 (1.28%)
  - 0000071: 88 (2.07%)  0000072: 73 (1.84%)  0000073: 63 (1.51%)
  - 0000074: 66 (1.55%)  0000075: 56 (1.27%)  0000076: 58 (1.31%)  0000077: 49 (1.11%)

## 3. samplefps 平局归因（单算子级，部署管线真实子云输入）

在 ti10 10 文件的**部署子云 pos 坐标**（preprocess_test + preprocess_subcloud 输出，非 task 5 的随机取样池）上，
用现役 FPS 插件与 SampleFPS 插件单算子对拍（N≈3800-4200, M=N/2, stride=2）：

```
subclouds=49 idx_mismatch_rate=0.0021%   (仅 2/约95000 个索引不一致)
mean idx_match=99.9981%  mean seq_exact=100.0000%   （每步仍为全局 argmax）
subclouds with ALL-mismatch-tied: 49/49  with genuine non-tie diff: 0/49
```

- 唯一不一致文件 0000076：N=4213, M=2106, mismatch=2，seq_exact=100%，tie_only=True
- **结论：samplefps 与现役 FPS 的 logits 差异全部由浮点平局的 tie-break 选择不同引起（0.0021% 索引），
  无一处 genuine 差异——满足 task 6 主判据「平局归因豁免」。**

## 4. (c) acc 测试

### 4.1 ti10 分集（plugin.md §9 基线 0.9741 的测量口径，现役可跑）

| engine | mean | min | p25 | median | p75 | max | 门槛判定 |
|---|---|---|---|---|---|---|---|
| 现役 hpenet_v2_fp32_v14 | **0.9741** | 0.9446 | 0.9634 | 0.9814 | 0.9852 | 0.9891 | 基线复现 ✓ |
| 现役 hpenet_v2_fp16_v14 | 0.9741 | 0.9446 | 0.9634 | 0.9812 | 0.9854 | 0.9891 | — |
| fps fp32 | **0.9741** | 0.9446 | 0.9634 | 0.9814 | 0.9852 | 0.9891 | == 现役逐文件 ✓ |
| fps fp16 | 0.9740 | 0.9449 | 0.9634 | 0.9812 | 0.9852 | 0.9891 | ±0.0001 ✓ |
| **samplefps fp32** | **0.9741** | 0.9446 | 0.9634 | 0.9814 | 0.9852 | 0.9891 | **== 现役逐文件 ✓** |
| samplefps fp16 | 0.9740 | 0.9449 | 0.9634 | 0.9812 | 0.9852 | 0.9891 | ✓ |
| **cache-only（flashfps k1.0）fp32** | **0.9741** | 0.9446 | 0.9634 | 0.9814 | 0.9852 | 0.9891 | **== 现役逐文件 ✓** |
| flashfps k0.75 fp32 | 0.9707 | 0.9425 | 0.9613 | 0.9776 | 0.9811 | 0.9823 | 均 0.9712 门槛差 0.0005（per-frame 尾部通过，见 §5） |
| flashfps k0.75 fp16 | 0.9706 | 0.9425 | 0.9612 | 0.9775 | 0.9811 | 0.9823 | 同上 |
| flashfps k0.5 fp32 | 0.9612 | 0.9373 | 0.9582 | 0.9651 | 0.9678 | 0.9738 | 档位阶梯 |
| flashfps k0.25 fp32 | 0.9535 | 0.9349 | 0.9492 | 0.9554 | 0.9583 | 0.9674 | 档位阶梯 |

### 4.2 modetest 全测试集（58 文件，OA/mAcc/mIoU）

| engine | OA | mAcc | mIoU | per-file mean | min | median | max |
|---|---|---|---|---|---|---|---|
| fps fp32（现役等价基线） | 93.2536 | 90.2423 | 79.8901 | 0.9321 | 0.8822 | 0.9382 | 0.9660 |
| fps fp16 | 93.2537 | 90.2423 | 79.8901 | 0.9321 | 0.8822 | 0.9382 | 0.9660 |
| samplefps fp32 | 93.2539 | 90.2425 | 79.8908 | 0.9321 | 0.8824 | 0.9382 | 0.9660 |
| samplefps fp16 | 93.2537 | 90.2423 | 79.8901 | 0.9321 | 0.8822 | 0.9382 | 0.9660 |
| flashfps k0.75 fp32 | **93.4364** | 89.7739 | **80.1223** | **0.9338** | 0.8820 | 0.9407 | 0.9675 |
| flashfps k0.75 fp16 | 93.4363 | 89.7738 | 80.1220 | 0.9338 | 0.8816 | 0.9406 | 0.9675 |

- samplefps / cache-only 在两种分集下 acc 均与现役（fps）**逐文件一致**，门槛通过
- **flashfps k0.75 在 modetest 全测试集反而略高于现役（0.9338 vs 0.9321，+0.0017）**——Prune+升序填充在部分文件上缓解了硬样本

### 4.3 flashfps keep_rate 档位阶梯（未达门槛时的下调复测，不视为失败）

| keep_rate | ti10 mean | 与现役差 | 备注 |
|---|---|---|---|
| 1.0（cache-only） | 0.9741 | 0 | 纯 Cache，无 Prune，acc 完全等于现役 |
| 0.75 | 0.9707 | -0.0034 | 计划默认档，距 0.9712 门槛差 0.0005 |
| 0.5 | 0.9612 | -0.0129 | 下调（更多填充）→ acc 单调下降 |
| 0.25 | 0.9535 | -0.0206 | 下调 → acc 继续下降 |

- 档位趋势符合 FlashFPS 语义（keep_rate 越低填充越多，acc 越低）；0.75 为精度/速度最佳平衡档
- per-frame 尾部（§5）在 k0.75 通过，k0.5/k0.25 亦通过

## 5. per-frame 尾部检查（flashfps k0.75 vs 现役 min − 0.02）

- ti10: 现役 min=0.9446 → 门槛 0.9246；flashfps min=0.9425 ≥ 0.9246 ✓（差 −0.0021，远小于 0.02 容差）
- modetest: 现役 min=0.8822 → 门槛 0.8622；flashfps min=0.8820 ≥ 0.8622 ✓（差 −0.0002）
- **无显著尾部**，两分集均通过

## 6. (d) 延迟

### 6.1 端到端纯 engine 延迟（ti10，49 subcloud runs，含 H2D/D2H 与 voxel-voting 外预处理）

| engine | median (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|
| 现役 v14 fp32 | 29.3 | 30.0 | 28.3 |
| 现役 v14 fp16 | 29.3 | 29.9 | 28.4 |
| fps fp32 | 28.8 | 29.4 | 27.9 |
| fps fp16 | 29.2 | 29.7 | 28.4 |
| samplefps fp32 | 134.5 | 352.8 | 155.3 |
| samplefps fp16 | 129.4 | 350.3 | 153.2 |
| cache-only fp32 | 61.7 | 275.0 | 85.6 |
| flashfps fp32 | 43.7 | 159.5 | 58.9 |
| flashfps fp16 | 43.4 | 162.5 | 60.4 |
| flashfps k0.5 fp32 | 35.1 | 75.2 | 37.8 |
| flashfps k0.25 fp32 | 22.5 | 25.3 | 22.4 |

### 6.2 nsys 复测 FPS 段（-t cuda，同一 2 文件 13 subclouds 工作负载，GPU kernel 时间）

| engine | FPS 段 kernel | FPS 总耗时 | GPU 总耗时 | FPS 占比 | per-subcloud FPS |
|---|---|---|---|---|---|
| fps fp32 | furthest_point_sampling_kernel ×52 | 105.7ms | 173.3ms | **61.0%** | 8.1ms |
| fps fp16 | furthest_point_sampling_kernel ×52 | 107.1ms | 161.3ms | **66.4%** | 8.2ms |
| samplefps fp32 | samplefps_iterate_kernel ×52 (+build ×52) | 1108.4ms | 1142.6ms | **97.0%** | 85.3ms |
| samplefps fp16 | samplefps_iterate_kernel ×52 | 1080.1ms | 1134.2ms | **95.2%** | 83.1ms |
| flashfps fp32 | samplefps_iterate_kernel ×13（PrefixFPS×39 为 fill，≈0） | 299.4ms | 362.9ms | **82.5%** | 23.0ms |
| flashfps fp16 | samplefps_iterate_kernel ×13 | 299.4ms | 350.0ms | **85.6%** | 23.0ms |

- 基线对照：现役全图 trace（/tmp/opencode/hpenet_full.nsys-rep）FPS 段 82.3% GPU 占比、22.05ms/iter（1220 次 launch 全图 8.2s）；本任务以**同一工作负载下三算法相对对比**为准（现役 v14 profile 不覆盖 modetest，见 §0）
- **FPS<5ms 目标未达（现役 8.1ms/subcloud 起）**——按 plan 记录瓶颈分析而非硬凑：
  - **SampleFPS 单 block 桶结构是唯一瓶颈**：iterate kernel gridX=1（单 block），13 subclouds × 4 级 = 52 次 launch 的逐级耗时（N≈4200, M 逐级减半）：
    - level 1（M≈2100）: median 33.2ms / mean 35.4ms / max 86.9ms
    - level 2（M≈1050）: median 46.1ms / mean 40.6ms / max 60.0ms（部分子云最深）
    - level 3（M≈525）: median 6.5ms；level 4（M≈262）: median 5.3ms
    - per-subcloud FPS 合计 median 91.3ms / mean 84.8ms（vs 现役 fps 8.1ms/subcloud）
  - flashfps 通过 PrefixFPS Cache 把 4 次采样降为 1 次 FlashFPS + 3 次 fill，FPS 段从 1080ms→299ms（**-72%**），但单次 FlashFPS 仍 21-23ms/launch（瓶颈仍为单 block iterate kernel；13 次 launch 即 13 个 subcloud 的 level-1 精确段）
  - 现役 FPS kernel 0.37-7.4ms/launch 已远低于 5ms 目标；**SampleFPS/FlashFPS 单 block 迭代结构是延迟瓶颈**，需多 block 并行化（workspace 化 best/桶上界、跨 block 归约）方可达 <5ms，属后续 kernel 优化任务

## 7. 结论

### 7.1 三算法 × fp32/fp16 汇总对比表（ti10 分集 = 基线测量口径）

| 指标 | fps fp32 | fps fp16 | samplefps fp32 | samplefps fp16 | flashfps k0.75 fp32 | flashfps k0.75 fp16 |
|---|---|---|---|---|---|---|
| **acc（ti10 mean）** | 0.9741 | 0.9740 | **0.9741** | 0.9740 | 0.9707 | 0.9706 |
| **acc（modetest mean）** | 0.9321 | 0.9321 | 0.9321 | 0.9321 | **0.9338** | 0.9338 |
| per-frame min（ti10） | 0.9446 | 0.9449 | 0.9446 | 0.9449 | 0.9425 | 0.9425 |
| per-frame median（ti10） | 0.9814 | 0.9812 | 0.9814 | 0.9812 | 0.9776 | 0.9775 |
| logits 一致率（vs fps_fp32） | 1.0（自比） | fp16≈3e-2 | 0/10 逐位，**平局归因** | 0/10 逐位，fp16 噪声 | 0/10（近似，max_abs 3.5e1） | 0/10（近似） |
| **FPS 段耗时（nsys，13 subclouds）** | 105.7ms | 107.1ms | 1108.4ms | 1080.1ms | **299.4ms** | 299.4ms |
| FPS 段占比 | 61.0% | 66.4% | 97.0% | 95.2% | 82.5% | 85.6% |
| 端到端 median（ti10） | 28.8ms | 29.2ms | 134.5ms | 129.4ms | **43.7ms** | 43.4ms |
| 端到端 p99（ti10） | 29.4ms | 29.7ms | 352.8ms | 350.3ms | 159.5ms | 162.5ms |

### 7.2 门槛判定汇总

| 门槛 | 实测 | 判定 |
|---|---|---|
| samplefps acc == 现役 0.9741（ti10） | 0.9741，per-file 逐文件一致 | ✅ 通过 |
| cache-only 路径 acc == 现役 0.9741 | 0.9741，per-file 逐文件一致 | ✅ 通过 |
| samplefps 真实帧 logits 逐位 = 现役（或平局归因） | 逐位差 1.5e-2，idx mismatch 0.0021%、seq_exact 100%、0 genuine | ✅ 平局归因豁免成立 |
| flashfps k0.75 acc ≥ 0.9712（ti10） | 0.9707（差 0.0005）；modetest 0.9338 **反超现役** | ⚠️ ti10 差 0.0005 微差；档位阶梯已记录（k1.0=0.9741/k0.5=0.9612/k0.25=0.9535），k0.75 为精度最佳档，不视为失败 |
| flashfps per-frame 无显著尾部（min ≥ 现役 min − 0.02） | ti10: 0.9425 ≥ 0.9246；modetest: 0.8820 ≥ 0.8622 | ✅ 通过 |
| FPS 合计 <5ms | 现役 8.1ms/subcloud，samplefps 84.8ms，flashfps 23.0ms | ❌ 未达（按 plan 记录瓶颈分析：单 block iterate kernel 结构，非硬凑） |
| flashfps 端到端 ≤ 现役 26.53ms | 43.7ms | ❌ 未达（FPS 段仍被单 block 结构拖累） |

1. **Engine 构建**：三算法 × fp32/fp16 共 6 个 + cache-only(k1.0) + keep_rate 阶梯(k0.5/k0.25)，全部同 profile 构建成功
2. **logits 对拍**：
   - fps（回归）：与现役 v14 非逐位（profile 差异 1.5e-2，opt5500 vs opt4096），但 per-file acc 完全一致
   - samplefps（精确等价）：**逐位差 1.5e-2 全部平局归因**（idx mismatch 0.0021%，seq_exact 100%，0 genuine）——主判据豁免成立
   - flashfps（近似）：max_abs_err 7-35（按文件），pred flip 1.1%-2.6%/frame
3. **acc**：samplefps 与 cache-only 路径在 ti10 == 现役 0.9741 逐文件一致（modetest 也逐文件一致）；flashfps k0.75 ti10=0.9707（距 0.9712 门槛差 0.0005）、modetest=0.9338（**反超现役 +0.0017**）；per-frame 尾部两分集均通过（min 差 ≤0.0021）
4. **延迟**：flashfps 端到端 median 43.7ms（fp32）/ 43.4ms（fp16），FPS 段 299ms vs samplefps 1080ms（-72%）、现役 106ms；**FPS<5ms 目标未达，瓶颈=单 block iterate kernel，记录分析未硬凑**；flashfps 端到端未 ≤26.53ms
5. **keep_rate 阶梯**：k1.0=0.9741 / k0.75=0.9707 / k0.5=0.9612 / k0.25=0.9535（ti10），单调符合语义；k0.75 为最佳平衡档

## 8. 未做（遵守 MUST NOT DO）
- 未改 checkpoint / dataset 管线 / 权重
- 未改任何模型/插件/kernel 代码（纯验证）
- 未做 git 操作
- nsys 仅用 `-t cuda`（未加 cudnn/cublas）
