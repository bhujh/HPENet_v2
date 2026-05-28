# HPENet V2 — Project Knowledge Base

**Generated:** 2026-05-29 | **Commit:** ba7c195 | **Branch:** main

## OVERVIEW

Point cloud deep learning for radar + lidar segmentation. Built on OpenPoints (MMCV-style registry), PyTorch 1.10.1, CUDA 11.3. Primary task: custom radar binary segmentation.

## STRUCTURE
```
HPENet_v2/
├── openpoints/           # Core lib — models, dataset, cpp, utils → ./openpoints/AGENTS.md
│   └── cpp/              # CUDA C++ extensions → ./openpoints/cpp/AGENTS.md
├── cfgs/                 # YAML config cascade → ./cfgs/AGENTS.md
├── examples/             # Entry points: seg, cls, partseg
├── deploy/               # ONNX export + TensorRT (has own deploy.md)
├── script_me/            # Custom radar launch scripts
└── script/               # General launch scripts (upstream)
```

## ENVIRONMENT

- **Conda**: `openpoints` (Python 3.7, PyTorch 1.10.1, CUDA 11.3)
- **Install**: `source install.sh` — **PyTorch MUST be installed first** (line 28), then `requirements.txt` (line 31), then CUDA extensions (lines 34-53). Reversing this order breaks all extension builds.
- **GPU arch**: `TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"` — **missing 8.6 (RTX 30xx), 8.9 (RTX 40xx)**. Add `8.6;8.9` before building for newer GPUs.
- **Extension builds are inconsistent**: `pointnet2_batch` uses `install`, `subsampling` uses `build_ext --inplace`, `chamfer_dist/emd` use `install --user`. `.so` files end up in different locations.
- **No CI/CD** — builds are manual. No `.github/workflows/`.
- **Submodule**: `openpoints/` is NOT a git submodule (no `.gitmodules`). Checked in directly despite `install.sh` referencing submodule commands.

## ENTRY POINTS

All via `examples/*/main.py` with YAML config + dot-notation CLI overrides:

| Task | Command |
|------|---------|
| Segmentation (radar/S3DIS/ScanNet) | `python examples/segmentation/main.py --cfg cfgs/{task}/{model}.yaml` |
| Classification (ModelNet40/ScanObjectNN) | `python examples/classification/main.py --cfg cfgs/{task}/{model}.yaml` |
| Part seg (ShapeNetPart) | `python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/{model}.yaml` |

**CLI overrides**: `mode=test`, `wandb.use_wandb=False`, `batch_size=8`, `model.encoder_args.width=128`
**Path hack**: Each `examples/*/main.py` starts with `import __init__` — triggers `sys.path.insert(0, '.../../')`. Do NOT remove.

## MODEL BUILDING

MMCV-style: `@MODELS.register_module()` → `build_model_from_cfg(cfg)` reads `cfg.NAME`.
Segmentation: `BaseSeg(encoder_args → decoder_args → cls_args)`. Decoder inherits `encoder_channel_list` automatically.
For HPENet V2 specifically: `cfg.model.NAME: BaseSeg` → `encoder_args.NAME: HPENetV2Encoder` → `decoder_args.NAME: HPENetV2Decoder` → `cls_args.NAME: SegHead`

## RADAR DATASET (Custom)

- Binary segmentation (valid/invalid), PLY input: `x, y, z, rcs, snr, v, label`
- `feature_keys: x,heights` — radar features + z-height
- Voxel size: 0.1 (vs 0.04 for S3DIS), InstanceNorm (vs BatchNorm for S3DIS)
- `dataset.train.loop: 10` — each sample seen 10× per epoch (multiplies `__len__`)
- Code in `openpoints/dataset/radar/s3disRadar.py` — **misleading filename** (named after S3DIS but is radar)
- Launch: `script_me/main_segmentation_train.sh`, `script_me/main_segmentation_test.sh`

## ANTI-PATTERNS (DO NOT)

- **Stale pretrained_path**: `cfgs/default.yaml` line 61 has commented-out hardcoded path — always override via CLI: `--pretrained_path /actual/path`
- **`.cuda()` in datasets**: `scanobjectnn.py:63`, `shapenetpart.py:244`, `matterport3d.py:137` call `.cuda()` inside `__getitem__` — breaks multi-GPU. Use `.to(device)` instead.
- **Hardcoded absolute paths**: `cfgs/scannet/default.yaml:5`, `cfgs/shapenetpart/default.yaml:4`, 5 files under `semantic_kitti/` — override with `data_root` config.
- **Dead code**: ~800 lines commented-out across `point_transformer_gpu.py`, `s3dis_block.py`, `s3disRadar_block.py`, `debug_invvit.py`, `DistillBaseSeg` class. Do not reference or restore without review.
- **Known BUGs**: `models/layers/kmeans.py:46` (centroids < K), `models/backbone/curvenet.py:775` (unlabeled), `scheduler/cosine_lr.py:78` ("seems not correct")
- **DO NOT USE**: `models/backbone/pointnextPyG.py` — entire file flagged "under development"
- **Multi-GPU test unsupported**: `examples/segmentation/main.py:361` — guard `if cfg.world_size < 2`. Testing bypasses DataLoader entirely.

## TEST CONVENTIONS

- **No unit tests**. No `pytest`, no CI. Only one `unittest.TestCase` file: `openpoints/cpp/chamfer_dist/test.py`.
- **"Test" = model evaluation**: `mode=test` runs inference with voxel voting, outputs mIoU/OA/mAcc to CSV. No automated correctness checks.
- **Debugging**: `debug_compare_val_test.py` compares val vs test data pipelines. Manual.
- **S3DIS benchmark**: `examples/segmentation/test_s3dis_6fold.py` — 6-fold cross-validation script.

## DEPLOY

- `deploy/onnx_export.py` → `deploy/onnx_inference.py` (has `deploy.md` with full docs)
- `deploy/trt_build.py` → `deploy/trt_inference.py` (TensorRT FP16/FP32 engines)
- Pre-built `.onnx` and `.engine` files in `deploy/`

## HPENet vs HPENet V2

Config files use `hpenet-*.yaml` but code module is `hpenetv2.py`. Both refer to HPENet V2. V1 is not in this repo.

## Language

以后agent回答我的问题都用汉语。

## Agent Restrictions

未经我明确允许，agent 禁止执行任何 git 操作（包括但不限于 commit、push、reset、rebase、merge）。

## Chinese Documentation

`模型学习混乱修复.md` — training debugging log. `模型部署.md` — deployment guide. Both at project root.
