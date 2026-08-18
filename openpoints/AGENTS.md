# openpoints/ — Core Library

## OVERVIEW

MMCV-style deep learning framework for point clouds. Models registered via `@MODELS.register_module()`, built from YAML config via `build_from_cfg(cfg)`. 212 Python files across 10 subdirectories.

## STRUCTURE
```
openpoints/
├── models/              # Backbones, segmentation/classification wrappers, layers
│   ├── backbone/        # 27 encoder models (HPENetV2, PointNeXt, PointVector, etc.)
│   ├── segmentation/    # BaseSeg, BasePartSeg, VariableSeg wrappers
│   ├── classification/  # BaseCls, DistillCls, ClsHead
│   ├── reconstruction/  # Masked point autoencoders
│   └── layers/          # Conv, attention, pooling, grouping primitives
├── dataset/             # 16 dataset loaders → REGISTERED via @DATASETS.register_module()
│   └── radar/           # Custom binary seg (2 classes, PLY with RCS/SNR)
├── cpp/                 # CUDA C++ extensions → ./cpp/AGENTS.md
├── utils/               # Registry, EasyConfig, distributed, logging
├── transforms/          # Point cloud augs (GPU: PointTransformerGPUTransform; CPU: 15 transforms)
├── loss/                # 8 loss classes (CrossEntropy, SmoothCrossEntropy, Poly1FocalLoss...)
├── optim/               # 30+ optimizers via string dispatch (NO registry)
└── scheduler/           # 6 schedulers via string dispatch
```

## REGISTRY PATTERN

4 registries, all in `openpoints/utils/registry.py` (294 lines):

| Registry | Location | `NAME` key |
|----------|----------|-----------|
| `MODELS` | `models/build.py` | `cfg.model.NAME`, `cfg.model.encoder_args.NAME`, etc. |
| `DATASETS` | `dataset/build.py` | `cfg.dataset.common.NAME` |
| `LOSS` | `loss/build.py` | `cfg.criterion_args.NAME` |
| `DataTransforms` | `transforms/transforms_factory.py` | Transform names |

**Flow**: `build_from_cfg(cfg)` → extracts `cfg['NAME']` → looks up class from registry → calls `cls(**remaining_kwargs)`.
**Optimizers/schedulers** use if/elif chains (string dispatch), NOT registry.

## MODEL BUILDING FLOW

```
build_model_from_cfg(cfg.model)
  ├── MODELS.build(cfg.model)              # cfg.model.NAME = "BaseSeg"
  │   ├── MODELS.build(encoder_args)       # NAME = "HPENetV2Encoder"
  │   ├── MODELS.build(decoder_args)       # NAME = "HPENetV2Decoder"
  │   └── MODELS.build(cls_args)           # NAME = "SegHead"
```

Decoder automatically receives `encoder_channel_list` from encoder output channels.
For classification: `BaseCls(encoder_args → cls_args)`.

## KEY UTILITIES

- **`EasyConfig`** (`utils/config.py`): dict subclass with `__getattr__` proxy. `load(fpath, recursive=True)` auto-discovers `default.yaml` ancestors. `update(opts)` parses dot-notation CLI overrides.
- **`Registry`** (`utils/registry.py`): MMCV-compatible. `register_module()` as decorator or programmatic. Supports hierarchical scope.
- **`get_dist_info`** (`utils/dist_utils.py`): Sets `rank`, `world_size`, distributed flags.

## DATASET CONVENTIONS

All datasets register with `@DATASETS.register_module()`. Each subdir is self-contained (1-6 files):
- **Radar** (`radar/s3disRadar.py`): `RadarClassi`, binary seg, `loop: 10` epoch multiplier. Misleadingly named after S3DIS. 同目录 `s3disRadar_block.py`（孤儿，整个类被注释）与 `s3disRadar_sphere.py`。PLY 按字段名加载 x,y,z,mag,rcs,snr,v,label；feat/z 用 feat_stats 归一化。
- **S3DIS** (`s3dis/`): 13-class, block+sphere sampling, `loop: 30`.
- **ScanNet** (`scannetv2/`): 20-class, RGB features.
- **ModelNet40** (`modelnet/`): 40-class cls, 1024/2048 point sampling.

**Gotcha**: Several datasets call `.cuda()` inside `__init__` 数据预处理 (scanobjectnn:63, shapenetpart:244, matterport3d:137). Breaks multi-GPU / 无 GPU 环境。

## KEY HPENET V2 SYMBOLS (CODE MAP)

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `HPENetV2Encoder` | class | `models/backbone/hpenetv2.py:389` | 5-stage PointNeXt 式编码器: SetAbstraction(FPS+ballquery+HPE) + InvResMLP; 宽度随 stride>1 翻倍, 输出 channel_list |
| `HPENetV2Decoder` | class | `models/backbone/hpenetv2.py:540` | FeaturePropogation×4 + BAFM 融合 + three_interpolation; **forward 硬编码 5 stage (f[1..5])** |
| `BaseSeg` | class | `models/segmentation/base_seg.py:15` | 容器: registry 建 encoder, 注入 encoder_channel_list 到 decoder |
| `SegHead` | class | `models/segmentation/base_seg.py:92` | 1D conv head, mlps=[in,in,num_classes]+dropout |
| `build_model_from_cfg` | func | `models/build.py:5` | `MODELS.build(cfg)` 分发 (40 callers incl. deploy) |
| `HPE` | class | `models/layers/position_encoding.py` | 相对位置编码 (encoder 内 rel_pos) |
| `BAFM` | class | `models/layers/channel_attention.py` | decoder 跨注意力融合模块 |

## WHERE TO LOOK

| Task | File |
|------|------|
| Add new backbone | `models/backbone/` — add .py + register in `__init__.py` |
| Add new dataset | `dataset/{name}/` — register with `@DATASETS.register_module()` |
| Add new loss | `loss/{name}.py` — register with `@LOSS.register_module()` |
| Add new CUDA op | `cpp/` — see `cpp/AGENTS.md` |
| Modify registry | `utils/registry.py` |
| Change config loading | `utils/config.py` — `EasyConfig` class |
