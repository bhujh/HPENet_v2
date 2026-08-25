# Task-3 证据：samplefps_op + PrefixFPS 插件 + fps_algo 接线

- 时间: 2026-08-19 04:15
- 环境: python=/home/wangpeng/miniforge3/envs/hpenet/bin/python | torch=2.2.2+cu118 | TRT=8.6.1 | onnx=1.17.0 | GPU=NVIDIA L20
- checkpoint: log/radar/radar-train-hpenet-ll-ngpus1-20260812-201051-gHBQ4DMy5jP2fZfkGkFcue/checkpoint/*_ckpt_best.pth
- cfg: cfgs/radar/hpenet-ll.yaml | strides=[1,2,2,2,2]（导出图中 4 个采样节点 stride=2）

## 1. 交付物

### (a) `deploy/onnx_ops/samplefps_op.py`（新）
- `SampleFPSOp` symbolic 发 `hpenet::SampleFPS`，attrs **仅 `stride_i`**（keep_rate_f 归 flashfps_op）
- forward 回落 openpoints `furthest_point_sample`（torch 侧语义）；`make_samplefps_op(stride)` 工厂

### (b) PrefixFPS 轻量插件（Cache 载体，动态形状安全）
- `deploy/trt_plugins/src/prefixfps_plugin.cpp` + `deploy/trt_plugins/src/prefixfps_kernel.cu` + `include/prefixfps_plugin.h` + `include/prefixfps_kernel.h`（新）
- type="PrefixFPS" version="1"，属性仅 `stride`(int32)
- getOutputDimensions：`kFLOOR_DIV(输入 N, stride)` 动态推导 Mℓ（照 fps_plugin.cpp:23-34）
- enqueue：`idx[b*M+i]=i`（arange 填充，零距离计算，B 循环支持；小 fill kernel 放 prefixfps_kernel.cu）
- **workspace=0**（getWorkspaceSize 返回 0，不申请距离缓冲）
- supportsFormatCombination：pos0 FLOAT / pos1 INT32（照 fps_plugin.cpp:36-43）
- serialize/deserialize 只存 stride（int）
- `deploy/onnx_ops/prefixfps_op.py`（新）：symbolic 发 `hpenet::PrefixFPS` attrs `stride_i`；torch 侧 forward **返回 `arange(M, dtype=torch.int32)`**（Cache 语义取前缀，非真 FPS）
- 注册：plugin_registry.cpp 追加 `#include "prefixfps_plugin.h"` + `REGISTER_TENSORRT_PLUGIN(PrefixFPSPluginCreator)`（第 8 个，FlashFPS 之后）；CMakeLists.txt 追加 `src/prefixfps_plugin.cpp` + `src/prefixfps_kernel.cu`

### (c) fps_algo 接线
- `deploy/onnx_backend.py`: `patch_model_for_onnx(model, fps_algo='fps')`
  - `'fps'` → make_fps_op（**默认，零行为变化**）
  - `'samplefps'` → 各级 make_samplefps_op(stride)（复用 onnx_backend.py:210-213 的 sample_fn 替换点）
  - `'flashfps'` → raise NotImplementedError（task 4 收尾）
- `deploy/onnx_export.py`: 加 argparse `--fps-algo`（choices fps/samplefps，默认 fps），透传 patch_model_for_onnx

## 2. 构建

```
cd deploy/trt_plugins/build && make -j  →  100% Built target hpenet_plugins
```
registry 探针（libhpenet_plugins.so 重编后）：FPS/SampleFPS/FlashFPS/PrefixFPS 均注册 → True。

## 3. 图 diff 摘要

### fps 默认路径回归（MUST DO）
机械比对（任务指定 one-liner）：
```
python -c "import onnx; a=onnx.load('deploy/hpenet_v2_plugin.onnx'); b=onnx.load('fps_algo_fps.onnx'); assert [n.op_type for n in a.graph.node]==[n.op_type for n in b.graph.node]; ..."
→ PASS
```
- `deploy/hpenet_v2_plugin.onnx`（628 节点）vs `fps_algo_fps.onnx`（628 节点）：**op_type 序列逐节点一致**
- 变更前基线导出（未改代码）与 hpenet_v2_plugin.onnx 同样逐节点一致 → 变更零行为变化

### samplefps 图（MUST DO）
`fps_algo_samplefps.onnx`（628 节点）hpenet 域节点：
```
SampleFPS{stride:2} BallQueryGroup{radius:10} BallQueryDP{radius:20}   (encoder.1)
SampleFPS{stride:2} BallQueryGroup{radius:20} BallQueryDP{radius:40}   (encoder.2)
SampleFPS{stride:2} BallQueryGroup{radius:40} BallQueryDP{radius:80}   (encoder.3)
SampleFPS{stride:2} BallQueryGroup{radius:80} BallQueryDP{radius:160}  (encoder.4)
ThreeInterp ×5 (decoder)
```
- **4 个 `hpenet::SampleFPS` 节点**（stride=2）✅
- 非 hpenet 部分与 fps 图逐节点一致（仅 FPS→SampleFPS 替换）✅
- TRT 全模型 parse PASS，4 个 SampleFPS 插件层正确落在 encoder.1-4 各级 ✅

## 4. PrefixFPS 动态形状运行时断言（MUST DO）

`/tmp/opencode/t3/test_prefixfps_dyn.py`（照 tests_fps_algos.py 的 test_prefixfps_dynamic_shape）：
- 动态 profile：min=1024 / opt=3762 / max=6500（照 trt_build.py 的 min1024/opt4096/max10000 量级）
- N=1024：out_shape=(1, 256) == (1, 1024//4) PASS；idx==arange(256) PASS
- N=6500：out_shape=(1, 1625) == (1, 6500//4) PASS；idx==arange(1625) PASS
- B=2 静态：`idx[b*M+i]==i` PASS（B 循环支持）
- 构建期 getOutputDimensions 只做符号推导，上述为实际 enqueue 运行断言

## 5. 结论
- fps 默认路径回归：**通过**（导出图与 hpenet_v2_plugin.onnx 逐节点一致）
- PrefixFPS 动态形状运行时断言：**通过**（N=1024/6500 两档 profile 实际跑）
- libhpenet_plugins.so 重编：**通过**
- flashfps 档：未接线（task 4 收尾），onnx_backend 对 'flashfps' raise NotImplementedError

## 6. 未做（遵守 MUST NOT DO）
- 未改现役 fps_op.py / 其它插件 / hpenetv2.py
- Cache 未用 ONNX Slice/arange 常量实现（prefixfps 插件动态推导 Mℓ）
- 未删 traceable_random_fps
- 未做 git 操作
