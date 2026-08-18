# HPENet V2 — Project Knowledge Base

**Generated:** 2026-05-29 | **Last updated:** 2026-08-17 | **Branch:** deploy (main 仍为默认远端分支，近期工作在 deploy，有未提交暂存改动)

## OVERVIEW

Point cloud deep learning for radar + lidar segmentation. Built on OpenPoints (MMCV-style registry). Primary task: custom radar binary segmentation.

## STRUCTURE
```
HPENet_v2/
├── openpoints/           # Core lib — models, dataset, cpp, utils → ./openpoints/AGENTS.md
│   └── cpp/              # CUDA C++ extensions → ./openpoints/cpp/AGENTS.md
├── cfgs/                 # YAML config cascade → ./cfgs/AGENTS.md
├── examples/             # Entry points: seg, cls, partseg + profile.py
├── deploy/               # GIT SUBMODULE (ONNX/TensorRT 导出 + 5 个 C++ 推理工程，详见 DEPLOY 节)
├── script_me/            # Custom radar launch scripts
├── script/               # General launch scripts (upstream)
├── data/                 # Datasets (RadarClassi radarfull/radarfullwl, S3DIS)；raw 被 gitignore，processed/ 有文件被 git 追踪
├── docs/                 # Upstream mkdocs site (mkdocs.yml at root)
├── log/                  # Training logs (radar/s3dis, wandb-style)，gitignored
├── results/              # Radar test outputs (untracked)
└── .omo/ + .codegraph    # Agent 工具状态 (untracked)
```

## ENVIRONMENT

- **Conda**: `hpenet` (Python 3.10, PyTorch 2.2.2+cu118, CUDA 11.8)。注意 `install.sh` 是过时的上游脚本（创建 py3.7 env `openpoints` + torch 1.10.1+cu113），实际环境是手动搭建的，勿直接重跑。
- **Install**: `source install.sh` — **PyTorch MUST be installed first** (line 28), then `requirements.txt` (line 31), then CUDA extensions (lines 34-53: pointnet2_batch L34 `install`, subsampling L39 `build_ext --inplace`, pointops L45 `install`, chamfer_dist L50 / emd L52 `install --user`). Reversing this order breaks all extension builds.
- **GPU arch**: `TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"` (install.sh line 11) — **missing 8.6 (RTX 30xx), 8.9 (RTX 40xx)**. Add `8.6;8.9` before building for newer GPUs.
- **CUDA extensions**: 5 个（含 pointops）。pointnet2_batch/pointops/chamfer_dist 已编译 (cpython-310 .so)；subsampling in-place；**emd 未编译**（系统中找不到 emd_cuda .so）。`.so` files end up in different locations.
- **No CI/CD** — builds are manual. No `.github/workflows/`.
- **Submodule**: `.gitmodules` 存在 — **`deploy/` 是 git 子模块** (http://192.168.50.3:30000/wangpeng/deploy.git，内部有未提交的暂存改动)。`openpoints/` 不是子模块，直接签入；`install.sh` 里的 submodule 命令是死引用。`update.sh` = `git pull --recurse-submodules`。
- **Git 状态注意**: 根仓库与 deploy 子模块均有未提交暂存改动 (cfgs/radar/default.yaml, plugin.md, script_me/*, deploy 内 C++ 工程)。另有 `openpoints/.gitignore` 残留未解决的 merge 冲突标记。
- **SECURITY**: `sers` / `sers.pub`（SSH 私钥+公钥）被提交进了 git 仓库 — 待用户决定是否清除。

## ENTRY POINTS

All via `examples/*/main.py` with YAML config + dot-notation CLI overrides:

| Task | Command |
|------|---------|
| Segmentation (radar/S3DIS/ScanNet) | `python examples/segmentation/main.py --cfg cfgs/{task}/{model}.yaml` |
| Classification (ModelNet40/ScanObjectNN) | `python examples/classification/main.py --cfg cfgs/{task}/{model}.yaml` |
| Part seg (ShapeNetPart) | `python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/{model}.yaml` |

**CLI overrides**: `mode=test`, `wandb.use_wandb=False`, `batch_size=8`, `model.encoder_args.width=128`
**Path hack**: seg 与 cls 的 `main.py` 首行 `import __init__`（触发 `sys.path.insert(0, '../../')`，勿删）。**shapenetpart/main.py 例外** — 用的是内联 `sys.path.insert`（lines 24-25），无 `import __init__`。
**Classification 结构**: `main.py` 是 67 行薄分发器，委托 `examples/classification/{train,pretrain}.py`，支持 `mode=pretrain`。
**其他脚本入口**（未纳入上表）: `examples/profile.py`（FLOPs/吞吐量剖析）、`examples/segmentation/main_debug.py`（alpha/lambda 调试 fork）、`examples/segmentation/vis_results.py`（OBJ 可视化）、根目录 `debug_compare_val_test.py`。`script/profile_flops.sh` 用了错误的 `--cfgs` 旗标（应为 `--cfg`），按原样跑会失败。

## MODEL BUILDING

MMCV-style: `@MODELS.register_module()` → `build_model_from_cfg(cfg)` reads `cfg.NAME`.
Segmentation: `BaseSeg(encoder_args → decoder_args → cls_args)`. Decoder inherits `encoder_channel_list` automatically.
For HPENet V2 specifically: `cfg.model.NAME: BaseSeg` → `encoder_args.NAME: HPENetV2Encoder` → `decoder_args.NAME: HPENetV2Decoder` → `cls_args.NAME: SegHead`

## RADAR DATASET (Custom)

- Binary segmentation (valid/invalid), PLY input: `x, y, z, mag, rcs, snr, v, label`（按字段名加载）
- `feature_keys: x,heights` — radar features (mag/rcs/snr/v = 4 dims) + z-height (1 dim) = 5 dims（`get_features_by_keys` 拼接，`data_util.py:177`）
- `in_channels: 5` 仅 hpenet-ll 匹配 5 维输入；**其余变体 (s/b/l/xl/xxl) 仍写 in_channels: 4，与 5 维数据管线不匹配（stem conv 会报错）— 属过时配置**
- Voxel size: 0.3, `voxel_max: 4608` (train/val), `None` (test) — 旧文档写 3000 已过时
- `dataset.train.loop: 10` — each sample seen 10× per epoch (multiplies `__len__`)
- Code in `openpoints/dataset/radar/s3disRadar.py`（`RadarClassi`，83/17 随机 split seed=100，feat/z 归一化用 feat_stats）— **misleading filename**；同目录 `s3disRadar_block.py`（孤儿，全注释）与 `s3disRadar_sphere.py`
- Launch: `script_me/main_segmentation_train.sh`, `script_me/main_segmentation_test.sh` — **当前均指向 hpenet-ll.yaml**
- **变体表**（`cfgs/radar/`，全部 BaseSeg → HPENetV2Encoder/Decoder/SegHead）:

| variant | width | blocks | strides | in_ch | radius | norm |
|---------|-------|--------|---------|-------|--------|------|
| s | 32 | [1,1,1,1,1] | [1,4,4,4,4] | 4 | 0.1 | in |
| b | 32 | [1,2,3,2,2] | [1,4,4,4,4] | 4 | 0.1 | in |
| l | 32 | [1,3,5,3,3] | [1,4,4,4,4] | 4 | 0.1 | in |
| **ll (active)** | 32 | [1,3,5,3,3] | **[1,2,2,2,2]** | **5** | **10** | **bn** |
| xl | 64 | [1,4,7,4,4] | [1,4,4,4,4] | 4 | 0.1 | in |
| xxl | 64 | [1,5,9,5,5] | [1,4,4,4,4] | 4 | 0.1 | in |

- **BN 死层问题已修复** (commit e850cbc, 2026-08-12): 原 7/8 `rel_pos.conv.0.1` BN `running_var=5.61e-45` 下溢、eval 输出 ~316× 放大。修复 = hpenet-ll 改 strides [1,4,4,4,4]→[1,2,2,2,2]、radius 0.3→10（降低 FPS/ball_query 采样下 dp=0 概率）+ warmup 10→15。旧 checkpoint 仍带病 BN，勿直接 eval。

## ANTI-PATTERNS (DO NOT)

- **Stale pretrained_path**: `cfgs/default.yaml` line 61 has commented-out hardcoded path — always override via CLI: `--pretrained_path /actual/path`
- **`.cuda()` in datasets**: `scanobjectnn.py:63`, `shapenetpart.py:244`, `matterport3d.py:137` call `.cuda()` inside `__init__` 数据预处理（非 `__getitem__`，旧文档写错）— breaks multi-GPU / 无 GPU 环境。
- **`.cuda()` in MODEL code** (同问题，模型侧): `curvenet.py:777`, `Stratified_transformer.py:18,28,66,294,339`, `cls_base.py:55`, `simpleview_util.py:249,251`, `debug_invvit.py:31-42`
- **Hardcoded absolute paths**: `cfgs/scannet/default.yaml:5`, `cfgs/shapenetpart/default.yaml:4`, `openpoints/dataset/semantic_kitti/utils/data_prepare_semantickitti.py:21,23`, `data_prepare_semantic3d.py:15`（cfgs/semantic_kitti/ 目录已不存在，旧引用过时）, `deploy/cmake_tensor.sh:22,33`（含 scp 目标 adas@192.168.137.40）
- **Dead code**: ~570 lines commented-out across `point_transformer_gpu.py`（文件是活的，仅注释掉的 transform 类 Chromatic*/Zoom/RGBtoHSV）、`s3dis_block.py` 与 `s3disRadar_block.py`（孤儿文件，整个类被注释）、`debug_invvit.py`（孤儿）、`DistillBaseSeg` class. Do not reference or restore without review.
- **Known BUGs**: `models/layers/kmeans.py:46` (centroids < K), `models/backbone/curvenet.py:775` (unlabeled), `scheduler/cosine_lr.py:78` ("seems not correct")
- **DO NOT USE**: `models/backbone/pointnextPyG.py` — entire file flagged "under development"
- **Multi-GPU test unsupported**: `examples/segmentation/main.py:361` — guard `if cfg.world_size < 2`. Testing bypasses DataLoader entirely.
- **HPENetV2Decoder.forward 硬编码 5 个 encoder stage**（`hpenetv2.py:590-604` 直接索引 f[1..5]）— 改 stage 数会崩。

## TEST CONVENTIONS

- **No unit tests**. No `pytest`, no CI. Only one `unittest.TestCase` file: `openpoints/cpp/chamfer_dist/test.py`.
- **"Test" = model evaluation**: `mode=test` runs inference with voxel voting, outputs mIoU/OA/mAcc to CSV. No automated correctness checks.
- **Debugging**: `debug_compare_val_test.py` compares val vs test data pipelines. Manual.
- **S3DIS benchmark**: `examples/segmentation/test_s3dis_6fold.py` — 6-fold cross-validation script.
- **Manual scripts (非测试框架)**: `cpp/emd/test_emd_loss.py`（EMD 手验）、`semantic_kitti/utils/nearest_neighbors/test.py`（KNN bench）、`semantic_kitti/utils/6_fold_cv.py`（离线 6-fold，硬编码 /data 路径）。

## DEPLOY

- **deploy/ 是 git 子模块**（内部有未提交的暂存改动，勿在未经允许时执行 git 操作）
- Python 侧: `deploy/onnx_export.py` → `deploy/onnx_inference.py` (has `deploy.md` with full docs)；`deploy/trt_build.py` → `deploy/trt_inference.py` (TensorRT FP16/FP32)。辅助库 `onnx_backend.py`（FPS/ball_query 已改为自定义 ONNX 算子）、`trt_utils.py`（TRTSession 自动 ctypes 加载 libhpenet_plugins.so）、`common.py`。Shell 包装: `onnx_deployment.sh`, `trt_deployment.sh`, `trt_cpp.sh`, `cmake_tensor.sh`。
- **自定义算子/插件迁移中**（`trt_plugin_tip.md`）: `onnx_ops/`（fps_op.py, ballquery_op.py 自定义 ONNX 算子）+ `trt_plugins/`（C++ libhpenet_plugins.so: fps/ballquery TRT plugin）。目标 linux-x86/aarch64/win-x86, CUDA≥11.4, TRT≥8.5.3。
- **5 个 C++ 推理工程**（均产 `hpenet_trt_infer` / `hpenet_onnx_infer`；build/ output/ lib/ 均被 gitignore）:
  - `CPP_trt` — 原始 C++ CLI 版（--engine/--stats/--data_dir），唯一带 gtest CUDA 测试（test_voxelize/fnv_hash/scatter_mean.cu）
  - `CPP_trt1` — extern "C" C-API 封装版，入口 `src/test.c`（main.cpp 未接 CMake），feat5 模型 (RAW_FEAT_DIM=4/FEAT_DIM=5)，FP16，radarfullwl 硬编码路径
  - `CPP_trt2` — 旧 3-feat 变体，FP32，radarfull，入口 main.cpp
  - `CPP_trt3` — **当前工作流**: = CPP_trt1 + trt_plugins 子目录 + cuDNN；仅 `src/trt_engine.cpp` 不同于 v1（加 initLibHPENetPlugins()）
  - `CPP_onnx` — ONNX Runtime CPU 推理（无 CUDA/TRT），Eigen 头文件已提交，libonnxruntime.so 未提交；`onnx_cppdeploy.md` + `verify.py` 对拍
- Build: `cd deploy/CPP_trt3 && mkdir build && cd build && cmake .. -DTENSORRT_ROOT=/usr/local/TensorRT-8.6.1.6 -DCMAKE_CUDA_ARCHITECTURES="80;86;89" -DCMAKE_BUILD_TYPE=Release && make -j`
- Pre-built `.onnx` / `.engine` in `deploy/`: `hpenet_v2_plugin.onnx`, `hpenet_v2_fp32.engine`, `onnx_model_feat5_bn(_sim).onnx` 等。

## HPENet vs HPENet V2

Config files use `hpenet-*.yaml` but code module is `hpenetv2.py`. Both refer to HPENet V2. V1 is not in this repo.

## Language

以后agent回答我的问题都用汉语。

## Agent Restrictions

未经我明确允许，agent 禁止执行任何 git 操作（包括但不限于 commit、push、reset、rebase、merge）。

## Chinese Documentation

`模型学习混乱修复.md` — training debugging log. `模型部署.md` — deployment guide. `cuDNN段错误排查记录.md` — cuDNN segfault 排查. `plugin.md` — TRT 插件工程文档 (74KB). `README_hpenetv2.md` — HPENet V2 说明. All at project root.
