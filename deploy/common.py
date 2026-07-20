"""
Shared data loading and preprocessing functions for HPENet V2 deployment.

Extracted from trt_inference.py / onnx_inference.py to avoid duplication.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from plyfile import PlyData

from openpoints.dataset.data_util import voxelize


# ---------------------------------------------------------------------------
#  Data loading & preprocessing (identical to trt_inference.py)
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

    required = ["x", "y", "z", "mag", "rcs", "snr", "v", "label"]
    for f in required:
        if f not in vertex.dtype.names:
            raise ValueError(f"Field '{f}' not found in {data_path}")

    data = np.column_stack((
        vertex["x"], vertex["y"], vertex["z"],
        vertex["mag"], vertex["rcs"], vertex["snr"], vertex["v"],
        vertex["label"],
    )).astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)

    coord = data[:, :3]
    feat = data[:, 3:-1]
    label = data[:, -1]
    return coord, feat, label


def preprocess_test(coord, feat, voxel_size=0.3):
    """Voxelize → split into sub-clouds (one point per voxel per shift)."""
    coord = coord - coord.min(0)

    idx_points = []
    voxel_idx = None
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


def load_stats(stats_file):
    """Load feature normalization statistics."""
    stats = torch.load(stats_file, map_location="cpu")
    return (
        stats["feat_mean"],   # (3,)
        stats["feat_std"],    # (3,)
        stats["z_mean"],      # scalar
        stats["z_std"],       # scalar
    )


def preprocess_subcloud(coord, feat, idx_part, feat_mean, feat_std,
                        z_mean, z_std, gravity_dim=2):
    """Prepare a single sub-cloud for model inference.

    Returns:
        pos_batch: (1, N_part, 3) float32 numpy
        x_batch:   (1, 4, N_part) float32 numpy
    """
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

    pos_batch = pos_t.unsqueeze(0).contiguous().numpy().astype(np.float32)
    x_batch = x_combined.unsqueeze(0).transpose(1, 2).contiguous().numpy().astype(np.float32)
    return pos_batch, x_batch
