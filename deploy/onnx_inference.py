"""
ONNX inference pipeline for HPENet V2 radar semantic segmentation.

Usage:
    python deploy/onnx_inference.py [--onnx PATH] [--data_dir PATH]

Features:
    - Full voxel voting pipeline identical to the PyTorch test code
    - Compares ONNX output with original PyTorch model for validation
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from plyfile import PlyData
from tqdm import tqdm

from openpoints.dataset.data_util import voxelize, get_features_by_keys
from openpoints.utils.config import EasyConfig
from openpoints.models.build import build_model_from_cfg
from deploy.onnx_backend import patch_model_for_onnx


# ---------------------------------------------------------------------------
#  Data loading (identical to examples/segmentation/main.py:load_data)
# ---------------------------------------------------------------------------

def load_data_ply(data_path):
    """Load a single radar PLY file.

    Returns:
        coord: (N, 3)  xyz
        feat:  (N, 3)  rcs, snr, v
        label: (N,)    ground truth
    """
    plydata = PlyData.read(data_path)
    vertex = plydata["vertex"].data

    required = ["x", "y", "z", "rcs", "snr", "v", "label"]
    for f in required:
        if f not in vertex.dtype.names:
            raise ValueError(f"Field '{f}' not found in {data_path}")

    data = np.column_stack((
        vertex["x"], vertex["y"], vertex["z"],
        vertex["rcs"], vertex["snr"], vertex["v"],
        vertex["label"],
    )).astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)

    coord = data[:, :3]
    feat = data[:, 3:6]
    label = data[:, 6]
    return coord, feat, label


def preprocess_test(coord, feat, voxel_size=0.1):
    """Mimics the test-time preprocessing: voxelize → split into sub-clouds.

    Returns:
        coord, feat: original arrays (shifted)
        idx_points: list of index arrays for each sub-cloud
        voxel_idx, reverse_idx_part, reverse_idx_sort: for vote merging
    """
    coord = coord - coord.min(0)

    idx_points = []
    voxel_idx_out = None
    reverse_idx_part = None
    reverse_idx_sort = None

    if voxel_size is not None:
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            np.random.shuffle(idx_part)
            idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(coord.shape[0]))

    coord = np.nan_to_num(coord, nan=0.0)
    feat = np.nan_to_num(feat, nan=0.0)
    return coord, feat, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort


# ---------------------------------------------------------------------------
#  Feature normalization
# ---------------------------------------------------------------------------

def load_stats(stats_file):
    """Load feature normalization statistics."""
    stats = torch.load(stats_file, map_location='cpu')
    return (
        stats['feat_mean'],   # (3,)  rcs, snr, v
        stats['feat_std'],    # (3,)
        stats['z_mean'],      # scalar
        stats['z_std'],       # scalar
    )


# ---------------------------------------------------------------------------
#  ONNX inference
# ---------------------------------------------------------------------------

def run_onnx_inference(session, pos, x):
    """Run a single ONNX inference.

    Args:
        session: onnxruntime.InferenceSession
        pos: (1, N, 3) numpy
        x:   (1, 4, N) numpy  –  combined & normalized features

    Returns:
        logits: (1, 2, N) numpy
    """
    outputs = session.run(None, {'pos': pos, 'x': x})
    return outputs[0]


def run_pytorch_inference(model, pos, x):
    """Run a single PyTorch inference.

    Args:
        model: BaseSeg (patched)
        pos: (1, N, 3) tensor
        x:   (1, 4, N) tensor

    Returns:
        logits: (1, 2, N) tensor
    """
    with torch.no_grad():
        out = model({'pos': pos, 'x': x})
    return out


# ---------------------------------------------------------------------------
#  Full inference pipeline on one point cloud
# ---------------------------------------------------------------------------

def infer_one_cloud_onnx(session, coord, feat, idx_points, feat_mean, feat_std,
                         z_mean, z_std, gravity_dim=2):
    """Run ONNX inference on all sub-clouds of one point cloud.

    Returns:
        all_logits: (total_points, 2)
        idx_points_flat: flat indices for scatter merge
    """
    all_logits = []
    all_idx = []

    for idx_part in idx_points:
        # Select and center sub-cloud
        coord_part = coord[idx_part].copy()
        coord_part -= coord_part.min(0)

        feat_part = feat[idx_part].copy()

        # Convert to tensor and apply PointCloudXYZAlign
        pos_t = torch.from_numpy(coord_part).float()
        # XYZAlign: mean-center xy, z-min
        pos_t = pos_t - pos_t.mean(dim=0, keepdim=True)
        pos_t[:, gravity_dim] -= pos_t[:, gravity_dim].min()

        feat_t = torch.from_numpy(feat_part).float()  # (N, 3)
        heights_t = pos_t[:, gravity_dim:gravity_dim + 1]  # (N, 1)

        # Normalize
        feat_t = (feat_t - feat_mean) / feat_std.clamp(min=1e-5)
        heights_t = (heights_t - z_mean) / z_std.clamp(min=1e-5)

        # Combine features: cat + transpose → (4, N)
        x_combined = torch.cat([feat_t, heights_t], dim=-1)  # (N, 4)

        # Add batch dim: (1, N, 3) and (1, 4, N)
        pos_batch = pos_t.unsqueeze(0).numpy().astype(np.float32)
        x_batch = x_combined.unsqueeze(0).transpose(1, 2).contiguous().numpy().astype(np.float32)

        # ONNX inference
        logits_np = run_onnx_inference(session, pos_batch, x_batch)  # (1, 2, N)
        logits_t = torch.from_numpy(logits_np[0])  # (2, N_part)

        all_logits.append(logits_t)
        all_idx.append(torch.from_numpy(idx_part).long())

    # Merge with scatter mean
    all_logits_cat = torch.cat(all_logits, dim=1)  # (2, total_in_batches)
    all_logits_cat = all_logits_cat.transpose(0, 1)  # (total, 2)
    idx_flat = torch.cat(all_idx, dim=0)  # (total,)

    from torch_scatter import scatter
    merged = scatter(all_logits_cat, idx_flat, dim=0, reduce='mean')  # (N_orig, 2)
    return merged


def infer_one_cloud_pytorch(model, coord, feat, idx_points, feat_mean, feat_std,
                            z_mean, z_std, gravity_dim=2):
    """Same as infer_one_cloud_onnx but using the PyTorch model."""
    all_logits = []
    all_idx = []

    for idx_part in idx_points:
        coord_part = coord[idx_part].copy()
        coord_part -= coord_part.min(0)

        feat_part = feat[idx_part].copy()

        pos_t = torch.from_numpy(coord_part).float()
        pos_t = pos_t - pos_t.mean(dim=0, keepdim=True)
        pos_t[:, gravity_dim] -= pos_t[:, gravity_dim].min()

        feat_t = torch.from_numpy(feat_part).float()
        heights_t = pos_t[:, gravity_dim:gravity_dim + 1]

        feat_t = (feat_t - feat_mean) / feat_std.clamp(min=1e-5)
        heights_t = (heights_t - z_mean) / z_std.clamp(min=1e-5)

        x_combined = torch.cat([feat_t, heights_t], dim=-1)

        pos_batch = pos_t.unsqueeze(0)
        x_batch = x_combined.unsqueeze(0).transpose(1, 2).contiguous()

        logits = run_pytorch_inference(model, pos_batch, x_batch)  # (1, 2, N)
        logits = logits.squeeze(0)  # (2, N_part)

        all_logits.append(logits)
        all_idx.append(torch.from_numpy(idx_part).long())

    all_logits_cat = torch.cat(all_logits, dim=1)
    all_logits_cat = all_logits_cat.transpose(0, 1)
    idx_flat = torch.cat(all_idx, dim=0)

    from torch_scatter import scatter
    merged = scatter(all_logits_cat, idx_flat, dim=0, reduce='mean')
    return merged


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='HPENet V2 ONNX Inference')
    parser.add_argument('--onnx', type=str, default='deploy/onnx_model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--checkpoint', type=str,
                        default='log/radar/radar-train-hpenet-l-ngpus1-20260515-013127-HXWMALkaAC4GiUWjNV5c3g/checkpoint/radar-train-hpenet-l-ngpus1-20260515-013127-HXWMALkaAC4GiUWjNV5c3g_ckpt_best.pth',
                        help='Path to PyTorch checkpoint (for comparison)')
    parser.add_argument('--data_dir', type=str,
                        default='data/RadarClassi/radarfull/raw',
                        help='Directory of test PLY files')
    parser.add_argument('--stats_file', type=str,
                        default='data/RadarClassi/radarfull/processed/feat_stats_area5.pth',
                        help='Feature statistics file')
    parser.add_argument('--num_files', type=int, default=3,
                        help='Number of files to test (use -1 for all)')
    parser.add_argument('--compare', action='store_true', default=True,
                        help='Compare ONNX vs PyTorch outputs')
    args = parser.parse_args()

    print('=' * 60)
    print('HPENet V2 ONNX Inference')
    print('=' * 60)

    # Load ONNX
    print(f'\n[1/4] Loading ONNX model: {args.onnx}')
    import onnxruntime as ort
    session = ort.InferenceSession(
        args.onnx,
        providers=['CPUExecutionProvider'],
    )
    print(f'  Providers: {session.get_providers()}')
    for inp in session.get_inputs():
        print(f'  Input: {inp.name} shape={inp.shape} type={inp.type}')
    for out in session.get_outputs():
        print(f'  Output: {out.name} shape={out.shape} type={out.type}')

    # Load stats
    print(f'\n[2/4] Loading feature stats: {args.stats_file}')
    feat_mean, feat_std, z_mean, z_std = load_stats(args.stats_file)
    print(f'  feat_mean: {feat_mean.tolist()}')
    print(f'  feat_std:  {feat_std.tolist()}')
    print(f'  z_mean:    {z_mean.item():.4f}')
    print(f'  z_std:     {z_std.item():.4f}')

    # Load PyTorch model for comparison
    pt_model = None
    if args.compare:
        print('\n[3a/4] Loading PyTorch model for comparison...')
        cfg = EasyConfig()
        cfg.load('cfgs/radar/hpenet-l.yaml', recursive=True)
        if cfg.model.get('in_channels', None) is None:
            cfg.model.in_channels = cfg.model.encoder_args.in_channels

        pt_model = build_model_from_cfg(cfg.model)
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_sd[k] = v
        pt_model.load_state_dict(new_sd, strict=True)
        pt_model = patch_model_for_onnx(pt_model)
        pt_model.eval()
        print('  PyTorch model loaded and patched.')

    # Get data files
    print(f'\n[3/4] Listing test files from: {args.data_dir}')
    all_files = sorted([
        f for f in os.listdir(args.data_dir)
        if f.endswith('.ply')
    ])
    np.random.seed(100)
    np.random.shuffle(all_files)
    n_total = len(all_files)
    # Take last 17% as test (matching generate_data_list in main.py)
    test_files = all_files[int(n_total * 0.83):]
    if args.num_files > 0:
        test_files = test_files[:args.num_files]
    print(f'  Total PLY files: {n_total}')
    print(f'  Test files (last 17%): {len(test_files)}')
    print(f'  Testing on: {len(test_files)} files')

    # Run inference
    print(f'\n[4/4] Running inference...')
    gravity_dim = 2

    for cloud_idx, fname in enumerate(tqdm(test_files, desc='Inference')):
        data_path = os.path.join(args.data_dir, fname)
        coord, feat, label = load_data_ply(data_path)
        coord, feat, idx_points, _, _, _ = preprocess_test(
            coord, feat, voxel_size=0.1,
        )

        # ONNX inference
        logits_onnx = infer_one_cloud_onnx(
            session, coord, feat, idx_points,
            feat_mean, feat_std, z_mean, z_std, gravity_dim,
        )
        pred_onnx = logits_onnx.argmax(dim=1).numpy()
        label_t = torch.from_numpy(label.astype(np.int64))

        # Accuracy
        acc = (torch.from_numpy(pred_onnx) == label_t).float().mean().item()
        print(f'  [{cloud_idx:3d}] {fname:30s}  acc={acc:.4f}')

        # Compare with PyTorch
        if pt_model is not None:
            logits_pt = infer_one_cloud_pytorch(
                pt_model, coord, feat, idx_points,
                feat_mean, feat_std, z_mean, z_std, gravity_dim,
            )
            pred_pt = logits_pt.argmax(dim=1)
            acc_pt = (pred_pt == label_t).float().mean().item()

            # Compare logits
            diff = (logits_onnx - logits_pt).abs().max().item()
            pred_match = (pred_onnx == pred_pt.numpy()).mean()
            print(f'         PT  acc={acc_pt:.4f}  '
                  f'logit_diff={diff:.6f}  pred_match={pred_match:.4f}')

    print('\nDone!')


if __name__ == '__main__':
    main()
