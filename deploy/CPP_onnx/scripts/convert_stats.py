#!/usr/bin/env python3
"""Convert PyTorch .pth feature statistics to plain JSON for C++ consumption.

Usage:
    python3 convert_stats.py --input feats_stats.pth --output stats.json
"""
import os
import sys

# Add project root to path (mirrors examples/segmentation/__init__.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import argparse
import json
import torch


def main():
    parser = argparse.ArgumentParser(description="Convert .pth stats to JSON")
    parser.add_argument("--input", type=str, required=True, default="data/RadarClassi/radarfullwl/processed/feat_stats_area5.pth", help="Path to .pth stats file")
    parser.add_argument("--output", default="stats_feat5.json", help="Output JSON path")
    args = parser.parse_args()

    stats = torch.load(args.input, map_location="cpu", weights_only=True)
    output = {
        "feat_mean": stats["feat_mean"].tolist(),  # [3] -> [f1,f2,f3]
        "feat_std":  stats["feat_std"].tolist(),   # [3]
        "z_mean":    stats["z_mean"].item(),        # scalar
        "z_std":     stats["z_std"].item(),         # scalar
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Written {args.output}")


if __name__ == "__main__":
    main()
