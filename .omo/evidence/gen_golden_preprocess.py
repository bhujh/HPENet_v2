#!/usr/bin/env python3
"""Generate preprocess_test golden data for C++ sub-cloud comparison."""
import argparse, os, sys, pickle, hashlib
import numpy as np

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJ_ROOT)

from deploy.common import preprocess_test, load_data_ply

def gen_preprocess_golden(ply_path: str, output_dir: str, seed: int = 100, voxel_size: float = 0.3):
    """Generate golden idx_points from preprocess_test for one PLY file."""
    coord, feat, label = load_data_ply(ply_path)
    coord, feat, idx_points, _, _, _ = preprocess_test(
        coord, feat, voxel_size=voxel_size, seed=seed
    )
    # idx_points is a list of numpy arrays (one per sub-cloud)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "golden_idx_points.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(idx_points, f)

    # metadata
    n_subclouds = len(idx_points)
    sizes = [len(ip) for ip in idx_points]
    h = hashlib.sha256(str([ip.tolist() for ip in idx_points]).encode()).hexdigest()[:16]

    print(f"  File: {os.path.basename(ply_path)}")
    print(f"  Points: {coord.shape[0]}, Sub-clouds: {n_subclouds}")
    print(f"  Sub-cloud sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")
    print(f"  SHA256 (first 16): {h}")
    print(f"  Golden saved to: {out_path}")

    # also save metadata
    meta = {
        "ply": ply_path, "seed": seed, "voxel_size": voxel_size,
        "n_points": int(coord.shape[0]), "n_subclouds": n_subclouds,
        "subcloud_sizes": sizes, "sha256": h
    }
    with open(os.path.join(output_dir, "golden_meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)
    return idx_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Path to input PLY file")
    parser.add_argument("--output_dir", default=".omo/evidence/golden")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--voxel_size", type=float, default=0.3)
    args = parser.parse_args()
    gen_preprocess_golden(args.ply, args.output_dir, args.seed, args.voxel_size)
