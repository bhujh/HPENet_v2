# Task-4b 证据：flashfps 档 wiring（FlashFPS + PrefixFPS 图级 Cache）

- 时间: 2026-08-19
- 环境: python=/home/wangpeng/miniforge3/envs/hpenet/bin/python | torch=2.2.2+cu118 | TRT=8.6.1.6 | onnx=1.17.0 | GPU=NVIDIA L20
- checkpoint: log/radar/radar-train-hpenet-ll-ngpus1-20260812-201051-gHBQ4DMy5jP2fZfkGkFcue/checkpoint/*_ckpt_best.pth
- cfg: cfgs/radar/hpenet-ll.yaml | strides=[1,2,2,2,2] | 采样节点 4 个均 stride=2（encoder.1-4）

## 1. 交付物

### (a) `deploy/onnx_backend.py` — `patch_model_for_onnx(model, fps_algo='flashfps')` 档
- 'flashfps' 档不再 raise NotImplementedError：
  - 第 1 个采样节点（encoder stage 1）→ `make_flashfps_op(stride, keep_rate=0.75)` → `hpenet::FlashFPS`
  - 第 2-4 个采样节点（encoder stage 2-4）→ `make_prefixfps_op(stride)` → `hpenet::PrefixFPS`
- **判据（模块名 + 遍历顺序，已在代码注释写明）**：HPENetV2Encoder.encoder 是
  nn.Sequential，采样 SetAbstraction 恒为每个 stage 首个子模块（命名末两段 =
  `<stage>.<subidx>`，subidx==0，如 `encoder.encoder.1.0`）；head stage（stage==0，
  stride=1, is_head）无 sample_fn，stride>1 的 stage 1-4 各有 1 个采样节点。
  stage==1 → FlashFPS（prune），stage>=2 → PrefixFPS（cache）。named_modules()
  深度优先按注册序 → stage 索引升序，判据与遍历顺序一致。
- 实测 patch 计数：`Patched 1 SA.sample_fn → FlashFPS(keep_rate=0.75) + 3 SA.sample_fn → PrefixFPS`
  （模块名核对：encoder.encoder.1.0→FlashFPS keep_rate=0.75；encoder.encoder.2/3/4.0→PrefixFPS）

### (b) `deploy/onnx_export.py` — `--fps-algo` choices 扩为 `fps/samplefps/flashfps`
- choices=['fps', 'samplefps', 'flashfps']，默认 'fps'（零行为变化），透传 patch_model_for_onnx

### (c) 修复 trace 期形状 bug（必要最小修复，不碰 C++ 插件/kernel）
- `deploy/onnx_ops/prefixfps_op.py` forward 由 `arange(M)`（1-D）改为
  `arange(M).unsqueeze(0).expand(B, -1)`（(B, M)）：
  - 现役 FPS/SampleFPS 的 `furthest_point_sample` 返回 (B, M)；PrefixFPS TRT 插件
    getOutputDimensions 亦输出 (B, M)——torch 侧 forward 形状必须与插件输出约定一致，
    否则 trace 期下游 `idx.unsqueeze(-1).expand(-1,-1,3)` 在 1-D arange 上报错
    `RuntimeError: The expanded size of the tensor (-1) isn't allowed in a leading, non-existing dimension 0`
  - 导出图 PrefixFPS 节点输出形状由 trace 决定；(B, M) 与运行时插件输出一致，无形状漂移
- **未改** `samplefps_kernel.cu / flashfps_plugin.cpp / prefixfps_plugin.cpp / prefixfps_kernel.cu`
  等任何插件与 kernel（MUST NOT DO 遵守）

## 2. 图结构验证（flashfps 交付图）

导出命令（照 task 3 evidence 的 onnx_export.py 调用，--fps-algo flashfps）：
```
python deploy/onnx_export.py --checkpoint <ckpt> --cfg cfgs/radar/hpenet-ll.yaml \
  --output /tmp/opencode/t4b/fps_algo_flashfps.onnx --fps-algo flashfps --no_simplify
```
图：`fps_algo_flashfps.onnx`（628 节点，与 fps/samplefps 图同规模）

hpenet 域节点（17 个）：
```
FlashFPS{stride:2, keep_rate:0.75}  BallQueryGroup{radius:10} BallQueryDP{radius:20}   (encoder.1)
PrefixFPS{stride:2}                 BallQueryGroup{radius:20} BallQueryDP{radius:40}   (encoder.2)
PrefixFPS{stride:2}                 BallQueryGroup{radius:40} BallQueryDP{radius:80}   (encoder.3)
PrefixFPS{stride:2}                 BallQueryGroup{radius:80} BallQueryDP{radius:160}  (encoder.4)
ThreeInterp ×5 (decoder)
```
- **1 个 `hpenet::FlashFPS`（keep_rate_f→`keep_rate`=0.75，stride=2）+ 3 个 `hpenet::PrefixFPS`（stride=2）** ✅
- 非 hpenet 部分（611 节点）与 fps 图逐节点一致（op_type + domain）✅
- 其它 hpenet 节点（BallQueryGroup ×4 / BallQueryDP ×4 / ThreeInterp ×5）与 fps/samplefps 图一致 ✅

## 3. fps 默认路径回归（MUST DO）

```
python -c "import onnx; a=onnx.load('deploy/hpenet_v2_plugin.onnx'); b=onnx.load('/tmp/opencode/t4b/fps_algo_fps.onnx'); assert [n.op_type for n in a.graph.node]==[n.op_type for n in b.graph.node] and len(a.graph.node)==len(b.graph.node)==628"
→ PASS
```
- `hpenet_v2_plugin.onnx` vs 本次重导 `fps_algo_fps.onnx`：628 节点 op_type 序列逐节点一致 ✅
- flashfps 档改动对 fps 档零影响（fps 档走原 `make_fps_op` 分支，代码路径未变）

## 4. TRT 全模型 parse + engine build

### parse（flashfps 交付图，libhpenet_plugins.so 预加载）
```
PARSE OK. num_layers = 1210
  plugin: /model/encoder.1/encoder.1.0/FlashFPS
  plugin: /model/encoder.2/encoder.2.0/PrefixFPS
  plugin: /model/encoder.3/encoder.3.0/PrefixFPS
  plugin: /model/encoder.4/encoder.4.0/PrefixFPS
  + BallQueryGroup ×4 / BallQueryDP ×4 / ThreeInterp ×5
```
- **4 个采样插件层 = 1× FlashFPS + 3× PrefixFPS** ✅

### engine build
```
python deploy/trt_build.py --onnx /tmp/opencode/t4b/fps_algo_flashfps.onnx \
  --output /tmp/opencode/t4b/fps_algo_flashfps.engine \
  --min_n 1024 --opt_n 4096 --max_n 10000 --workspace 4 --num_input_features 5
```
- 结果：`Engine saved: /tmp/opencode/t4b/fps_algo_flashfps.engine (15.3 MB)`，`TRT_BUILD_EXIT=0`
- 反序列化冒烟：deserialize 成功（插件 serialize/deserialize 字节序对称，FlashFPS/PrefixFPS
  经 engine 序列化往返可加载）
- 端到端推理冒烟（TRTSession，动态 N）：N=1024/4096/5500 三档输出 shape=(1,2,N)、
  值域有限且合理，全部 PASS
- flashfps 图完整 parse → build → 序列化 → 反序列化 → 端到端推理全链路通过 ✅

## 5. cache-only 语义验证（前缀等价性直接实证）

独立脚本：`/tmp/opencode/t4b/cache_only_semantics.py`
- 导出 flashfps k=1.0 图（第 1 级 FlashFPS keep_rate=1.0 精确路径 + 3×PrefixFPS）与全 SampleFPS 图
- 图结构断言：flashfps-k1 图 1×FlashFPS(keep_rate=1.0) + 3×PrefixFPS(stride=2)，非 hpenet 部分与 samplefps 图逐节点一致
- **逐级 FPS 输出索引对比**（ORT 无法执行 hpenet::* 自定义算子，逐级对比在导出同源的 patch 模型上
  执行——torch 侧 forward 与 trace 进图的算子完全一致；两图相同输入）

```
=== CACHE-ONLY SEMANTICS: ALL PASS ===
  N=4096 level 1: M=2048 samplefps==flashfps=True  (精确路径)
  N=4096 level 2: M=1024 samplefps==flashfps=True prefix(==arange)=True PASS
  N=4096 level 3: M=512  samplefps==flashfps=True prefix(==arange)=True PASS
  N=4096 level 4: M=256  samplefps==flashfps=True prefix(==arange)=True PASS
  N=5500 level 1: M=2750 samplefps==flashfps=True
  N=5500 level 2: M=1375 samplefps==flashfps=True prefix(==arange)=True PASS
  N=5500 level 3: M=687  samplefps==flashfps=True prefix(==arange)=True PASS
  N=5500 level 4: M=343  samplefps==flashfps=True prefix(==arange)=True PASS
```
- 第 1 级：flashfps k=1.0 精确 FPS 与 samplefps 逐索引一致
- **第 2-4 级：samplefps 图 FPS 输出 == flashfps 图 PrefixFPS 输出 == arange(M_ℓ)** ——
  前缀等价性直接实证：深层精确 FPS 输出 = 第 1 级 FPS 序的前缀（各级种子=首点索引 0、
  下级输入=上级输出原序、min 距离增量 float32 精确）
- 双 N 档（4096/5500）全过

### keep_rate ≥ 0.5 约束核验（数学保证）
- keep_rate·M₁ ≥ M₂ 时第 2 级前缀落在精确 FPS 段
- N=4096：M₁=2048, M₂=1024；keep_rate·M₁ = 0.75×2048 = 1536 ≥ 1024 ✅
- N=5500：M₁=2750, M₂=1375；keep_rate·M₁ = 2062 ≥ 1375 ✅
- 默认 0.75 满足约束

## 6. 结论
- `patch_model_for_onnx(model, fps_algo='flashfps')` 可用，导出 flashfps 图：1×FlashFPS(keep_rate=0.75) + 3×PrefixFPS ✅
- 非 hpenet 部分与 fps 图逐节点一致 ✅；fps 档回归零变化 ✅
- TRT parse 4 个采样插件层（FlashFPS + 3×PrefixFPS）✅；engine build + deserialize + 端到端推理全过 ✅
- cache-only 语义（前缀等价性）N=4096/5500 逐索引实证全过 ✅
- keep_rate=0.75 满足 keep_rate·M₁ ≥ M₂ ✅

## 7. 未做（遵守 MUST NOT DO）
- 未改 samplefps/flashfps/prefixfps 插件与 kernel（C++ 侧零改动）
- 未改 fps 与 samplefps 档行为（回归验证通过）
- 未做 git 操作
