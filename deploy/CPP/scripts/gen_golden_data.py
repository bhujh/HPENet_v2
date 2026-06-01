#!/usr/bin/env python3
"""Generate golden test data for CUDA kernel unit tests.

Produces reproducible test vectors for:
  1. FNV64-1A hash
  2. Voxelization (hash-based)
  3. Scatter mean

Output: .bin files (raw binary for C++ fread) + .npy (for Python verification).
All data goes under <output_dir>/ (= deploy/CPP/tests/data/ by default).
"""

import argparse
import os
import sys

import numpy as np

# ── path hack: reach project root ──────────────────────────────────────────
_PROJ_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, _PROJ_ROOT)

# ── imports from openpoints ────────────────────────────────────────────────
from openpoints.dataset.data_util import voxelize, fnv_hash_vec

# ═══════════════════════════════════════════════════════════════════════════
#  FNV64-1A  test vectors
# ═══════════════════════════════════════════════════════════════════════════

def gen_fnv_test(output_dir: str, seed: int = 42, num_points: int = 1000):
    rng = np.random.RandomState(seed)
    voxel_size = 0.1

    # random point coordinates  (N, 3)  float32
    coord = rng.rand(num_points, 3).astype(np.float32)

    # compute voxel coordinates  (used internally by fnv_hash_vec)
    discrete_coord = np.floor(coord / voxel_size)

    # FNV64-1A hash
    hashed = fnv_hash_vec(discrete_coord)  # (N,) uint64

    # ── save ────────────────────────────────────────────────────────────
    coord.tofile(os.path.join(output_dir, "fnv_coord.bin"))
    hashed.astype(np.uint64).tofile(os.path.join(output_dir, "fnv_hash.bin"))

    # also save a reference .npy for Python-side verification
    np.save(os.path.join(output_dir, "fnv_coord.npy"), coord)
    np.save(os.path.join(output_dir, "fnv_hash.npy"), hashed)

    print(f"  [FNV]  {num_points} points → fnv_coord.bin, fnv_hash.bin")


# ═══════════════════════════════════════════════════════════════════════════
#  Voxelization  test vectors
# ═══════════════════════════════════════════════════════════════════════════

def gen_voxelize_test(output_dir: str, seed: int = 42, num_points: int = 1000):
    rng = np.random.RandomState(seed)
    voxel_size = 0.1

    # use same coordinates as FNV test for consistency
    coord = rng.rand(num_points, 3).astype(np.float32)

    # voxelize with mode=1 (val mode → full outputs)
    idx_sort, voxel_idx, count = voxelize(
        coord, voxel_size=voxel_size, hash_type="fnv", mode=1
    )

    # reconstruct per-voxel point-index list  (idx_points)
    idx_sort = idx_sort.astype(np.int64)
    voxel_idx = voxel_idx.astype(np.int64)
    count = count.astype(np.int64)

    idx_points_list = np.split(idx_sort, np.cumsum(count[:-1]))

    # ── save ────────────────────────────────────────────────────────────
    coord.tofile(os.path.join(output_dir, "voxel_coord.bin"))
    idx_sort.tofile(os.path.join(output_dir, "voxel_idx_sort.bin"))
    voxel_idx.tofile(os.path.join(output_dir, "voxel_voxel_idx.bin"))
    count.tofile(os.path.join(output_dir, "voxel_count.bin"))

    # save idx_points as a numpy .npy containing an object array of arrays
    np.save(
        os.path.join(output_dir, "voxel_idx_points.npy"),
        np.array(idx_points_list, dtype=object),
    )

    # also save reference .npy files
    np.save(os.path.join(output_dir, "voxel_coord.npy"), coord)
    np.save(os.path.join(output_dir, "voxel_idx_sort.npy"), idx_sort)
    np.save(os.path.join(output_dir, "voxel_voxel_idx.npy"), voxel_idx)
    np.save(os.path.join(output_dir, "voxel_count.npy"), count)

    print(f"  [Voxelize]  {num_points} points, {len(count)} voxels → "
          f"voxel_*.bin, voxel_idx_points.npy")


# ═══════════════════════════════════════════════════════════════════════════
#  Scatter-mean  test vectors
# ═══════════════════════════════════════════════════════════════════════════

def _scatter_mean_numpy(src: np.ndarray, index: np.ndarray, dim: int = 0):
    """Pure-numpy scatter_mean (reference implementation)."""
    assert src.ndim == 2
    assert index.ndim == 1
    assert src.shape[dim] == index.shape[0]

    unique_idx = np.unique(index)
    out_shape = list(src.shape)
    out_shape[dim] = len(unique_idx)
    result = np.zeros(out_shape, dtype=src.dtype)

    for i, u in enumerate(unique_idx):
        mask = index == u
        if dim == 0:
            result[i] = src[mask].mean(axis=0)
        else:
            result[:, i] = src[:, mask].mean(axis=1)
    return result


def gen_scatter_test(output_dir: str, seed: int = 42):
    rng = np.random.RandomState(seed)
    N = 64          # number of source elements
    D = 2           # feature dimension
    num_unique = 5  # number of unique index groups

    src = rng.randn(N, D).astype(np.float32)
    index = np.sort(rng.randint(0, num_unique, size=N).astype(np.int64))

    # ── scatter mean (pure numpy reference) ─────────────────────────────
    result = _scatter_mean_numpy(src, index, dim=0)

    # ── save ────────────────────────────────────────────────────────────
    src.tofile(os.path.join(output_dir, "scatter_src.bin"))
    index.tofile(os.path.join(output_dir, "scatter_idx.bin"))
    result.astype(np.float32).tofile(os.path.join(output_dir, "scatter_result.bin"))

    # reference .npy
    np.save(os.path.join(output_dir, "scatter_src.npy"), src)
    np.save(os.path.join(output_dir, "scatter_idx.npy"), index)
    np.save(os.path.join(output_dir, "scatter_result.npy"), result)

    print(f"  [Scatter]  {N} × {D} → scatter_*.bin  ({num_unique} unique groups)")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate golden test data for CUDA kernel unit tests"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "..", "tests", "data"),
        help="Output directory for generated test vectors (default: deploy/CPP/tests/data)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)

    print(f"Generating golden test data in: {out}\n")

    gen_fnv_test(out, seed=args.seed)
    gen_voxelize_test(out, seed=args.seed)
    gen_scatter_test(out, seed=args.seed)

    print(f"\nDone.  All golden data saved to {out}")

    # ── quick listing ───────────────────────────────────────────────────
    print(f"\nOutput files:")
    for f in sorted(os.listdir(out)):
        fpath = os.path.join(out, f)
        size = os.path.getsize(fpath)
        print(f"  {f:30s} {size:>8d} bytes")
