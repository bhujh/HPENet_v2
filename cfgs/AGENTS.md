# cfgs/ — Config System

## OVERVIEW

71 YAML files across 9 task directories. 3-level automatic cascade: `cfgs/default.yaml` → `cfgs/{task}/default.yaml` → `cfgs/{task}/{model}.yaml`. CLI dot-notation overrides apply last via `parse_known_args()`.
## 3-LEVEL CASCADE

```
python examples/segmentation/main.py --cfg cfgs/radar/hpenet-xl.yaml

EasyConfig.load("cfgs/radar/hpenet-xl.yaml", recursive=True):
  Level 1: cfgs/default.yaml              # Global: distributed, training, wandb
  Level 2: cfgs/radar/default.yaml        # Task: dataset, transforms, optimizer
  Level 3: cfgs/radar/hpenet-xl.yaml      # Model: architecture overrides
→ CLI: mode=test batch_size=8 wandb.use_wandb=False
```

Recursive dict merge — later files override at any depth. `default.yaml` auto-discovered by walking up from config path.
## TASK DIRECTORIES

| Dir | Task | Classes | Key params |
|-----|------|---------|------------|
| `radar/` | Custom binary seg | 2 | voxel=0.02, loop=10, norm: ll=bn/其余=in, feature_keys=x,heights, voxel_max=4608 |
| `s3dis/` | S3DIS room seg | 13 | voxel=0.04, loop=30, norm=bn, test_area=5 |
| `scannet/` | ScanNet v2 seg | 20 | voxel=0.02, loop=6, norm=bn, RGB |
| `modelnet40ply2048/` | ModelNet40 cls | 40 | 1024 pts, SmoothCrossEntropy, 600 epochs |
| `scanobjectnn/` | ScanObjectNN cls | 15 | SmoothCrossEntropy, cosine 250 epochs |
| `shapenetpart/` | ShapeNetPart seg | 50 parts | Poly1FocalLoss, multistep 300 epochs |
| `s3dis_pix4point/` | Pix4Point on S3DIS | — | Pretrained ImageNet/BERT |
| `shapenetpart_pix4point/` | Pix4Point on ShapeNetPart | — | ditto |
| `scanobjectnn_pix4point/` | Pix4Point on ScanObjectNN | — | ditto |
## MODEL CONFIG NAMING

- HPENet V2: `hpenet-{size}.yaml` where size ∈ {s, b, l, **ll (当前活跃, in_ch=5/strides=[1,4,4,4,4]/radius=5/norm=bn)**, xl, xxl}
- Baselines: `pointnext-{size}.yaml`, `pointnet++.yaml`, `dgcnn.yaml`, etc.
- Config structure for seg:
  ```yaml
  model:
    NAME: BaseSeg
    encoder_args: { NAME: HPENetV2Encoder, in_channels: 5, width: 32, blocks: [1,3,5,3,3] }
    decoder_args: { NAME: HPENetV2Decoder }
    cls_args: { NAME: SegHead, num_classes: 2 }
  ```
- 注意: 除 hpenet-ll 外其余变体 in_channels: 4，与 5 维 feature 管线不匹配（会报错）。
- YAML header comments show pre-computed GFLOPs/Params.

## TASK-SPECIFIC DIFFERENCES

| Param | Radar (ll) | Radar (其余) | S3DIS | ScanNet | ModelNet40 |
|-------|-------|-------|-------|---------|------------|
| `radius` | 5 | 0.1 | 0.1 | 0.05 | 0.15 |
| `in_channels` | 5 | 4 | 4 | 7 (RGB) | 3 (xyz) |
| `norm` | `bn` | `in` | `bn` | `bn` | `bn` |
| `nsample` | 32 | 32 | 32 | 32 | 32 |

## CLI OVERRIDES

Dot-notation via `parse_known_args()` + `EasyConfig.update(opts)`:
```bash
mode=test                                    # train/val/test/resume
wandb.use_wandb=False                        # disable W&B
batch_size=8                                 #
model.encoder_args.width=128                 # nested override
--pretrained_path /path/to/ckpt_best.pth     # MUST override (default is stale)
```

## GOTCHAS
- **Stale `pretrained_path`**: `cfgs/default.yaml` line 61 has commented-out hardcoded path. Always override via CLI.
- **Pix4Point configs**: Hardcoded `pretrained/imagenet/mae_s.pth` paths — may not exist locally.
- **`dataset.train.loop`**: Multiplies `__len__()` — NOT a DataLoader repeat. `loop: 10` = 10× per epoch.
- **Test mode**: Override `mode=test` — bypasses DataLoader, uses raw file loading with voxel voting. Multi-GPU unsupported.
