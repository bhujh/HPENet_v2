"""
对比 per-epoch val 路径和 test 路径的数据差异，定位验证指标异常的根因。
用法: python debug_compare_val_test.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples/segmentation'))
# fallback: ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import logging
import torch
from plyfile import PlyData
from easydict import EasyDict as edict
from tqdm import tqdm

# 基础设置
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 路径设置
data_root = "data/RadarClassi/radarfull"
raw_root = os.path.join(data_root, "raw")
processed_root = os.path.join(data_root, "processed")
voxel_size = 0.1
test_area = 5
gravity_dim = 2
num_classes = 2
feature_keys = "x,heights"

# ----------------------------------------------------------------------
# 1. 选择同一个样本来对比
# ----------------------------------------------------------------------
data_list = sorted(os.listdir(raw_root))
splitCount = int(len(data_list) * 0.83)
val_files = [f for f in data_list if int(f[:-4]) > splitCount]

# 选第一个 val 样本 (0000250.ply)
sample_name = val_files[0]
sample_path = os.path.join(raw_root, sample_name)
logger.info(f"=== 对比样本: {sample_name} ===")

# ----------------------------------------------------------------------
# 2. 加载统计值
# ----------------------------------------------------------------------
stats_file = os.path.join(processed_root, f"feat_stats_area{test_area}.pth")
stats = torch.load(stats_file)
feat_mean = stats['feat_mean']
feat_std = stats['feat_std']
z_mean = stats['z_mean']
z_std = stats['z_std']

logger.info(f"\n归一化统计值 (来自 {stats_file}):")
logger.info(f"  feat_mean: {feat_mean.tolist()}")
logger.info(f"  feat_std:  {feat_std.tolist()}")
logger.info(f"  z_mean:    {z_mean.item():.4f}")
logger.info(f"  z_std:     {z_std.item():.4f}")

# ----------------------------------------------------------------------
# 3. 加载数据 - 路径A: Dataset __getitem__ (presample=True, per-epoch val)
# ----------------------------------------------------------------------
logger.info("\n" + "="*70)
logger.info("路径A: Dataset __getitem__ (per-epoch val, presample=True)")
logger.info("="*70)

from openpoints.dataset.radar.s3disRadar import RadarClassi
from openpoints.transforms.point_transform_cpu import PointsToTensor
from openpoints.transforms.point_transformer_gpu import PointCloudXYZAlign
from openpoints.transforms.transforms_factory import Compose

# 先确保 pickle 存在
pkl_path = os.path.join(processed_root, f'radar_val_area{test_area}_{voxel_size:.3f}_None.pkl')
assert os.path.exists(pkl_path), f"Pickle 不存在: {pkl_path}"

# 模拟 validate 的 transform 流程 (与 val 配置一致)
transform_val = Compose([PointsToTensor(), PointCloudXYZAlign(gravity_dim=gravity_dim)])

# 加载数据集
dataset_val = RadarClassi(
    data_root=data_root,
    test_area=test_area,
    voxel_size=voxel_size,
    voxel_max=None,
    split='val',
    transform=transform_val,
    loop=1,
    presample=True,
    variable=False,
    shuffle=False,
)

# 取第一个样本
sample_idx = 0
logger.info(f"取第 {sample_idx} 个样本")

# 手动模拟 __getitem__ 里的 transform + heights + normalize
from openpoints.dataset import get_features_by_keys

raw_data_A = dataset_val[sample_idx]
logger.info(f"  Dataset __getitem__ 返回 (已应用 transform + normalize):")
for key in ['pos', 'x', 'y', 'heights']:
    if key in raw_data_A:
        val = raw_data_A[key]
        logger.info(f"    {key}: shape={val.shape}, dtype={val.dtype}, min={val.min().item():.4f}, max={val.max().item():.4f}, mean={val.float().mean().item():.4f}")

# 检查标签分布
labels_A = raw_data_A['y'].numpy()
unique_A, counts_A = np.unique(labels_A, return_counts=True)
logger.info(f"  标签分布: {dict(zip(unique_A, counts_A))}")

# 模拟 validate 中的 get_features_by_keys (需要加 batch dim)
raw_data_A_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in raw_data_A.items()}
x_A = get_features_by_keys(raw_data_A_batch, feature_keys)
logger.info(f"  模型输入 features (after get_features_by_keys): shape={x_A.shape}, min={x_A.min().item():.4f}, max={x_A.max().item():.4f}, mean={x_A.mean().item():.4f}")

# ----------------------------------------------------------------------
# 4. 加载数据 - 路径B: test 路径 (load_data)
# ----------------------------------------------------------------------
logger.info("\n" + "="*70)
logger.info("路径B: test 路径 (load_data + multi-voxel)")
logger.info("="*70)

from openpoints.dataset.data_util import voxelize

# load_data 函数 (从 main.py 复制)
def load_data_radar(data_path):
    plydata = PlyData.read(data_path)
    data = plydata["vertex"].data
    required = ["x", "y", "z", "rcs", "snr", "v", "label"]
    for f in required:
        if f not in data.dtype.names:
            raise ValueError(f"字段 '{f}' 不存在于 PLY 文件中")
    coord = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
    feat = np.column_stack((data["rcs"], data["snr"], data["v"])).astype(np.float32)
    label = data["label"].astype(np.float32)
    coord -= coord.min(0)

    debug_raw_label_unique, debug_raw_label_counts = np.unique(label.astype(np.int32), return_counts=True)
    logger.info(f"  原始标签分布: {dict(zip(debug_raw_label_unique, debug_raw_label_counts))}")
    logger.info(f"  原始 coord min: {coord.min(0)}, max: {coord.max(0)}")
    logger.info(f"  原始 feat min: {feat.min(0)}, max: {feat.max(0)}")

    idx_points = []
    idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
    logger.info(f"  voxelize mode=1: 共 {count.max()} 个 sub-cloud, voxel_idx range [{voxel_idx.min()},{voxel_idx.max()}]")
    for i in range(count.max()):
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
        idx_part = idx_sort[idx_select]
        np.random.shuffle(idx_part)
        idx_points.append(idx_part)

    coord = np.nan_to_num(coord, nan=0.0)
    feat = np.nan_to_num(feat, nan=0.0)
    label = np.nan_to_num(label, nan=0.0)
    return coord, feat, label, idx_points, voxel_idx

coord_B, feat_B, label_B, idx_points_B, voxel_idx_B = load_data_radar(sample_path)
logger.info(f"  load_data 返回: coord={coord_B.shape}, feat={feat_B.shape}, label={label_B.shape}")
logger.info(f"  子云数量: {len(idx_points_B)}, 每个子云点数: {len(idx_points_B[0]) if idx_points_B else 'N/A'}")

# 取第一个 sub-cloud 对比 (test 会做 multi-view voting, 这里取单 view)
idx_part_B = idx_points_B[0]
coord_part_B = coord_B[idx_part_B].copy()
coord_part_B -= coord_part_B.min(0)
feat_part_B = feat_B[idx_part_B].copy() if feat_B is not None else None
label_part_B = label_B[idx_part_B].copy()

logger.info(f"\n  第一个 sub-cloud (re-centered):")
logger.info(f"    coord_part: min={coord_part_B.min(0)}, max={coord_part_B.max(0)}")
logger.info(f"    feat_part:  min={feat_part_B.min(0)}, max={feat_part_B.max(0)}")

# 模拟 test 中的 transform 流程 (同样的 transform)
data_B = {'pos': coord_part_B}
if feat_part_B is not None:
    data_B['x'] = feat_part_B
pipe_transform = Compose([PointsToTensor(), PointCloudXYZAlign(gravity_dim=gravity_dim)])
data_B = pipe_transform(data_B)
data_B = pipe_transform(data_B)

if 'heights' not in data_B:
    data_B['heights'] = torch.from_numpy(coord_part_B[:, gravity_dim:gravity_dim+1].astype(np.float32))

# 归一化
data_B['x'] = (data_B['x'] - feat_mean.to(data_B['x'].device)) / feat_std.to(data_B['x'].device).clamp(min=1e-5)
data_B['heights'] = (data_B['heights'] - z_mean.to(data_B['heights'].device)) / z_std.to(data_B['heights'].device).clamp(min=1e-5)

logger.info(f"\n  归一化后 (test 路径):")
for key in ['pos', 'x', 'heights']:
    val = data_B[key]
    logger.info(f"    {key}: shape={val.shape}, min={val.min().item():.4f}, max={val.max().item():.4f}, mean={val.mean().item():.4f}")

# 模拟 get_features_by_keys
data_B_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in data_B.items()}
x_B = get_features_by_keys(data_B_batch, feature_keys)
logger.info(f"  模型输入 features (after get_features_by_keys): shape={x_B.shape}, min={x_B.min().item():.4f}, max={x_B.max().item():.4f}, mean={x_B.mean().item():.4f}")

# ----------------------------------------------------------------------
# 5. 逐通道对比
# ----------------------------------------------------------------------
logger.info("\n" + "="*70)
logger.info("逐通道对比: 路径A (val) vs 路径B (test)")
logger.info("="*70)

common_n = min(x_A.shape[2], x_B.shape[1])  # channels=N, points=A(N) or B(N)
logger.info(f"  路径A feature channels: {x_A.shape[1]} x {x_A.shape[2]} points")
logger.info(f"  路径B feature channels: {x_B.shape[1]} x {x_B.shape[2]} points")

channel_names = ['rcs', 'snr', 'v', 'heights']
for ch in range(4):
    if ch < x_A.shape[1] and ch < x_B.shape[1]:
        val_A = x_A[0, ch, :].cpu().numpy()
        val_B = x_B[0, ch, :].cpu().numpy()
        logger.info(f"\n  通道 {ch} ({channel_names[ch]}):")
        logger.info(f"    路径A: min={val_A.min():.4f}, max={val_A.max():.4f}, mean={val_A.mean():.4f}, std={val_A.std():.4f}")
        logger.info(f"    路径B: min={val_B.min():.4f}, max={val_B.max():.4f}, mean={val_B.mean():.4f}, std={val_B.std():.4f}")
        diff = np.abs(val_A.mean() - val_B.mean())
        logger.info(f"    均值差异: {diff:.6f}")

# 额外检查: pos 坐标差异
logger.info(f"\n  pos 坐标对比:")
logger.info(f"    路径A pos shape: {raw_data_A['pos'].shape}, min z: {raw_data_A['pos'][:, 2].min().item():.4f}")
logger.info(f"    路径B pos shape: {data_B['pos'].shape}, min z: {data_B['pos'][:, 2].min().item():.4f}")

# 额外的原始数据对比 - 把路径A的数据反归一化，看原始 feat 是否一致
logger.info(f"\n  === 反归一化检查 ===")
feat_A_raw = raw_data_A['x'].cpu() * feat_std + feat_mean
logger.info(f"  路径A raw feat (反归一化): min={feat_A_raw.min(0).values.tolist()}, max={feat_A_raw.max(0).values.tolist()}, mean={feat_A_raw.mean(0).tolist()}")

# ----------------------------------------------------------------------
# 6. 模型推理对比
# ----------------------------------------------------------------------
logger.info("\n" + "="*70)
logger.info("模型推理对比")
logger.info("="*70)

from openpoints.models import build_model_from_cfg

# ----------------------------------------------------------------------
# 6. 模型推理对比 (两个 checkpoint)
# ----------------------------------------------------------------------
logger.info("\n" + "="*70)
logger.info("模型推理对比")
logger.info("="*70)

from openpoints.models import build_model_from_cfg
from openpoints.utils.config import EasyConfig

raw_cfg = {
    'model': {
        'NAME': 'BaseSeg',
        'encoder_args': {
            'NAME': 'HPENetV2Encoder',
            'blocks': [1, 4, 7, 4, 4],
            'strides': [1, 4, 4, 4, 4],
            'sa_layers': 1,
            'sa_use_res': False,
            'width': 64,
            'in_channels': 4,
            'expansion': 1,
            'radius': 0.1,
            'nsample': 32,
            'aggr_args': {'feature_type': 'dp_fj', 'reduction': 'max'},
            'group_args': {'NAME': 'ballquery', 'normalize_dp': True},
            'conv_args': {'order': 'conv-norm-act'},
            'act_args': {'act': 'relu', 'inplace': True},
            'norm_args': {'norm': 'bn'},
        },
        'decoder_args': {'NAME': 'HPENetV2Decoder'},
        'cls_args': {
            'NAME': 'SegHead',
            'num_classes': 2,
            'in_channels': None,
            'norm_args': {'norm': 'bn'},
        },
    }
}

cfg = EasyConfig()
cfg.update(raw_cfg)

model = build_model_from_cfg(cfg.model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
logger.info(f"模型构建完成, device={device}")

def to_device(data_dict):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}

ckpt_paths = [
    ("正常训练", "log/radar/radar-train-hpenet-xl-ngpus1-20260511-015625-ZvxCNqdVPfRMuRfDRztkUx/checkpoint/radar-train-hpenet-xl-ngpus1-20260511-015625-ZvxCNqdVPfRMuRfDRztkUx_ckpt_best.pth"),
    ("问题训练", "log/radar/radar-train-hpenet-xl-ngpus1-20260511-151542-GM9GDJ866DzLXub3vohKoA/checkpoint/radar-train-hpenet-xl-ngpus1-20260511-151542-GM9GDJ866DzLXub3vohKoA_ckpt_best.pth"),
]

for run_name, ckpt_path in ckpt_paths:
    if not os.path.exists(ckpt_path):
        logger.warning(f"  {run_name} ckpt 不存在: {ckpt_path}")
        continue

    logger.info(f"\n--- {run_name}: {os.path.basename(ckpt_path)} ---")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    # 路径A 推理
    with torch.no_grad():
        data_A_fwd = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in raw_data_A_batch.items()}
        data_A_fwd['x'] = x_A.clone()
        data_A_fwd = to_device(data_A_fwd)
        logits_A = model(data_A_fwd)
        pred_A = logits_A.argmax(dim=1).squeeze()
        pred_counts_A = torch.bincount(pred_A, minlength=num_classes)
        tp_A = torch.bincount(pred_A[raw_data_A['y'] == 0], minlength=num_classes)
        logger.info(f"  路径A (val): pred class0={pred_counts_A[0].item()} class1={pred_counts_A[1].item()}, "
                     f"GT class0={1291} class1={260}")

    # 路径B 推理
    with torch.no_grad():
        data_B_fwd = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data_B_batch.items()}
        data_B_fwd['x'] = x_B.clone()
        data_B_fwd = to_device(data_B_fwd)
        logits_B = model(data_B_fwd)
        pred_B = logits_B.argmax(dim=1).squeeze()
        pred_counts_B = torch.bincount(pred_B, minlength=num_classes)
        logger.info(f"  路径B (test): pred class0={pred_counts_B[0].item()} class1={pred_counts_B[1].item()}")

logger.info("\n" + "="*70)
logger.info("调试脚本执行完毕")
logger.info("="*70)

# ----------------------------------------------------------------------
# 7. 检查训练数据 path (presample=False, 含 crop_pc)
# ----------------------------------------------------------------------
logger.info("\n" + "="*70)
logger.info("训练数据路径检查 (presample=False + crop_pc)")
logger.info("="*70)

# 取第一个训练样本
train_files = [f for f in data_list if int(f[:-4]) <= splitCount]
train_sample = train_files[0]
logger.info(f"训练样本: {train_sample}")

from openpoints.dataset.data_util import crop_pc

train_dataset = RadarClassi(
    data_root=data_root,
    test_area=test_area,
    voxel_size=voxel_size,
    voxel_max=24000,  # 与配置一致
    split='train',
    transform=transform_val,  # 用 val transform 避免随机增强干扰
    loop=1,
    presample=False,
    variable=False,
    shuffle=True,
)

train_data = train_dataset[0]
logger.info(f"\n训练 __getitem__ 返回:")
for key in ['pos', 'x', 'y', 'heights']:
    if key in train_data:
        val = train_data[key]
        logger.info(f"  {key}: shape={val.shape}, dtype={val.dtype}, min={val.min().item():.4f}, max={val.max().item():.4f}, mean={val.float().mean().item():.4f}")

# 标签分布
train_labels = train_data['y'].numpy()
t_unique, t_counts = np.unique(train_labels, return_counts=True)
logger.info(f"  训练标签分布: {dict(zip(t_unique, t_counts))}")

# 对比原始 PLY 原始标签分布
ply_path = os.path.join(raw_root, train_sample)
plydata = PlyData.read(ply_path)
raw_labels = plydata["vertex"].data["label"].astype(np.int32)
raw_u, raw_c = np.unique(raw_labels, return_counts=True)
logger.info(f"  原始 PLY 标签分布: {dict(zip(raw_u, raw_c))}")

# 检查反归一化后的特征
train_feat_raw = train_data['x'].cpu() * feat_std + feat_mean
logger.info(f"  训练 raw feat (反归一化): min={train_feat_raw.min(0).values.tolist()}, max={train_feat_raw.max(0).values.tolist()}, mean={train_feat_raw.mean(0).tolist()}")

# 关键: 用正常模型和问题模型分别在训练数据上推理
for run_name, ckpt_path in ckpt_paths:
    if not os.path.exists(ckpt_path):
        continue
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    train_data_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in train_data.items()}
    train_data_batch['x'] = get_features_by_keys(train_data_batch, feature_keys)
    train_data_batch = to_device(train_data_batch)
    
    with torch.no_grad():
        logits = model(train_data_batch)
        pred = logits.argmax(dim=1).squeeze()
        pred_counts = torch.bincount(pred, minlength=num_classes)
        logger.info(f"  {run_name} 训练数据预测: class0={pred_counts[0].item()}, class1={pred_counts[1].item()}")

# ----------------------------------------------------------------------
# 8. 检查 crop_pc 前后数据变化
# ----------------------------------------------------------------------
logger.info("\n" + "="*70)
logger.info("crop_pc 前后对比检查")
logger.info("="*70)

# 直接加载一个训练样本，不经过 dataset
ply_path = os.path.join(raw_root, train_sample)
plydata = PlyData.read(ply_path)
data_d = plydata["vertex"].data
cdata_raw = np.column_stack((
    data_d["x"], data_d["y"], data_d["z"],
    data_d["rcs"], data_d["snr"], data_d["v"],
    data_d["label"]
)).astype(np.float32)

# 手动模拟 __getitem__ 训练路径
cdata_centered = cdata_raw.copy()
cdata_centered[:, :3] -= np.min(cdata_centered[:, :3], 0)
cdata_centered = np.nan_to_num(cdata_centered, nan=0.0)
coord, feat, label = cdata_centered[:, :3], cdata_centered[:, 3:6], cdata_centered[:, 6:7]

logger.info(f"crop_pc 前: points={coord.shape[0]}")
logger.info(f"  coord min: {coord.min(0)}, max: {coord.max(0)}")
logger.info(f"  labels: {np.unique(label, return_counts=True)}")

coord_after, feat_after, label_after = crop_pc(
    coord.copy(), feat.copy(), label.copy(),
    'train', voxel_size, 24000,
    downsample=True, variable=False, shuffle=True)

logger.info(f"\ncrop_pc 后: points={coord_after.shape[0]}")
logger.info(f"  coord min: {coord_after.min(0)}, max: {coord_after.max(0)}")
logger.info(f"  labels: {np.unique(label_after, return_counts=True)}")
logger.info(f"  feat min: {feat_after.min(0)}, max: {feat_after.max(0)}")

# 检查标签是否被 crop_pc 损坏
n_voxelized = len(np.unique(coord_after, axis=0))
logger.info(f"\n  去重后的唯一点数: {n_voxelized}/{coord_after.shape[0]}")

# 关键: 对比 crop_pc 前后标签分布
label_raw_counts = np.bincount(cdata_raw[:, 6].astype(np.int32), minlength=2)
label_before_counts = np.bincount(label.astype(np.int32).squeeze(), minlength=2)
label_after_counts = np.bincount(label_after.astype(np.int32).squeeze(), minlength=2)
logger.info(f"\n标签分布: raw=[{label_raw_counts[0]}, {label_raw_counts[1]}], "
             f"centered=[{label_before_counts[0]}, {label_before_counts[1]}], "
             f"after crop_pc=[{label_after_counts[0]}, {label_after_counts[1]}]")
