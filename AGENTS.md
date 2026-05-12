# AGENTS.md — HPENet V2

## Environment

- **Conda env**: `openpoints` (Python 3.7, PyTorch 1.10.1, CUDA 11.3)
- **Install**: `source install.sh` — builds CUDA C++ extensions (`pointnet2_batch`, `subsampling`, `pointops`, `chamfer_dist`, `emd`). Must run **after** PyTorch is installed.
- **Git**: This repo uses `openpoints/` as a submodule. Run `git submodule update --init --recursive` if imports are broken.
- **VSCode**: Uses conda as the default env manager (`.vscode/settings.json`).

## Entry Points

All training/inference is launched via `examples/*/main.py` with a YAML config:

| Task | Command |
|------|---------|
| Scene segmentation (S3DIS, ScanNet, radar) | `python examples/segmentation/main.py --cfg cfgs/{task}/{model}.yaml` |
| Classification (ModelNet40, ScanObjectNN) | `python examples/classification/main.py --cfg cfgs/{task}/{model}.yaml` |
| Part segmentation (ShapeNetPart) | `python examples/shapenetpart/main.py --cfg cfgs/shapenetpart/{model}.yaml` |

**CLI overrides** use dot-notation: `mode=test`, `wandb.use_wandb=False`, `batch_size=8`.

## Config Cascade (3-level, automatic)

When you pass `--cfg cfgs/radar/hpenet-xl.yaml`, the system auto-loads:
```
cfgs/default.yaml          → cfgs/radar/default.yaml          → cfgs/radar/hpenet-xl.yaml
```
Later files override earlier ones via recursive dict merge. You never need to specify `--cfg` multiple times.

## Model Building

MMCV-style registry (`openpoints/utils/registry.py`): models registered with `@MODELS.register_module()`, built from config via `build_model_from_cfg(cfg)` which reads `cfg.NAME` to look up the class.

For segmentation: `BaseSeg` wraps `encoder_args` → `decoder_args` → `cls_args`. The decoder inherits `encoder_channel_list` from the encoder automatically.

## Radar Dataset (Custom)

This is a **custom** task in `openpoints/dataset/radar/` and `cfgs/radar/`:
- Binary semantic segmentation (2 classes: valid/invalid)
- Input format: PLY files with fields `x, y, z, rcs, snr, v, label`
- Feature keys: `x, heights` (radar features + z-height)
- Voxel size: 0.1 (larger than S3DIS 0.04)
- `dataset.train.loop: 10` means each sample is seen 10× per epoch
- Custom launch scripts: `script_me/main_segmentation_train.sh`, `script_me/main_segmentation_test.sh`
- Test script references checkpoint path explicitly; update `pretrained_path` when retraining.

## Gotchas

- **`examples/*/main.py`** each start with `import __init__` — a local `__init__.py` that adds the project root to `sys.path`. Don't remove it.
- **W&B** logging is on by default in most configs. Override with `wandb.use_wandb=False`.
- **Testing** bypasses DataLoader — uses raw file loading with voxel voting. Multi-GPU test is unsupported.
- **`dataset.train.loop`** multiplies `__len__` — controls epochs per pass, not repeat count in a loader.
- **Pretrained path** in `cfgs/default.yaml` (`pretrained_path`) is stale — always override via CLI.
- **GPUs**: Set `CUDA_VISIBLE_DEVICES=X` before the python command to control which GPU(s) are used.

## HPENet vs HPENet V2 Naming

Config files in `cfgs/` use `hpenet-*.yaml` but the code module is `hpenetv2.py`. Both refer to HPENet V2 models. Older HPENet v1 is not in this repo.
