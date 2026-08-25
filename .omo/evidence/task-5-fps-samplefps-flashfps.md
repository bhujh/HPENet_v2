# Task-5 证据：FPS 算法族单算子级对拍（deploy/tests_fps_algos.py）

- 时间: 2026-08-20 15:10:18
- 命令: python deploy/tests_fps_algos.py --configs fps,samplefps,flashfps_k1,flashfps_k075,prefixfps --precision fp32 --n 2750 --frames 5 --seed 0
- 环境: python=/home/wangpeng/miniforge3/envs/hpenet/bin/python | torch=2.2.2+cu118 | TRT=8.6.1 | onnx=1.17.0 | GPU=NVIDIA L20
- 参考约束: numpy float32 (dx*dx+dy*dy)+dz*dz 左结合非融合逐步，eps=0 bit-exact；随机输入坐标去重；①失败转②归因平局（float32 精确 ==）
- 判据: fp32 idx 逐索引一致；fp16 相对 half 舍入输入逐索引一致；每步全局 argmax 序列精确性；CUDA event 计时 warmup 20+median
- 总耗时: 45.8s | 自检: PASS | PrefixFPS 动态 N: PASS | PrefixFPS cache 语义: PASS

## 对拍矩阵

| 配置 | 精度 | 输入 | N=2750 |
|---|---|---|---|
| 现役 FPS (hpenet::FPS) (fps) | fp32 | random | 100.0% / 100.0% / 0.874ms |
| 现役 FPS (hpenet::FPS) (fps) | fp32 | real | 100.0% / 100.0% / 0.874ms |
| SampleFPS (samplefps) | fp32 | random | 100.0% / 100.0% / 3.777ms |
| SampleFPS (samplefps) | fp32 | real | 100.0% / 100.0% / 3.966ms |
| FlashFPS keep_rate=1.0 (flashfps_k1) | fp32 | random | 100.0% / 100.0% / 3.778ms |
| FlashFPS keep_rate=1.0 (flashfps_k1) | fp32 | real | 100.0% / 100.0% / 3.965ms |
| FlashFPS keep_rate=0.75 (flashfps_k075) | fp32 | random | 100.0% / 100.0% / 2.743ms |
| FlashFPS keep_rate=0.75 (flashfps_k075) | fp32 | real | 100.0% / 100.0% / 2.865ms |
| PrefixFPS (prefixfps) | fp32 | random | 100.0% / 100.0% / 0.015ms |
| PrefixFPS (prefixfps) | fp32 | real | 100.0% / 100.0% / 0.015ms |

## 各配置明细

### fps
  fp32  N=2750  random: idx= 100.0% seq= 100.0% mismatch=0 tie=none median=0.874ms out_shape=(1, 687)
  fp32  N=2750  real  : idx= 100.0% seq= 100.0% mismatch=0 tie=none median=0.874ms out_shape=(1, 687)

### samplefps
  fp32  N=2750  random: idx= 100.0% seq= 100.0% mismatch=0 tie=none median=3.777ms out_shape=(1, 687)  vs现役 idx= 100.0% mismatch=0 tie=none
  fp32  N=2750  real  : idx= 100.0% seq= 100.0% mismatch=0 tie=none median=3.966ms out_shape=(1, 687)  vs现役 idx= 100.0% mismatch=0 tie=none

### flashfps_k1
  fp32  N=2750  random: idx= 100.0% seq= 100.0% mismatch=0 tie=none median=3.778ms out_shape=(1, 687)  vs现役 idx= 100.0% mismatch=0 tie=none
  fp32  N=2750  real  : idx= 100.0% seq= 100.0% mismatch=0 tie=none median=3.965ms out_shape=(1, 687)  vs现役 idx= 100.0% mismatch=0 tie=none

### flashfps_k075
  fp32  N=2750  random: idx= 100.0% seq= 100.0% mismatch=0 tie=none median=2.743ms out_shape=(1, 687)
  fp32  N=2750  real  : idx= 100.0% seq= 100.0% mismatch=0 tie=none median=2.865ms out_shape=(1, 687)

### prefixfps
  fp32  N=2750  random: idx= 100.0% seq= 100.0% mismatch=0 tie=none median=0.015ms out_shape=(1, 687)
  fp32  N=2750  real  : idx= 100.0% seq= 100.0% mismatch=0 tie=none median=0.015ms out_shape=(1, 687)

## Deferred acceptance
- prefixfps 动态 N 形状验证（1024/6500）: PASS
- (b2) PrefixFPS cache 语义对拍: PASS
