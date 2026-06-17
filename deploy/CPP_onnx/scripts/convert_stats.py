#!/usr/bin/env python3
"""Convert PyTorch .pth feature statistics to plain JSON for C++ consumption.

Usage:
    python3 convert_stats.py --input feats_stats.pth --output stats.json
"""

import argparse
import json
import torch


def main():
    parser = argparse.ArgumentParser(description="Convert .pth stats to JSON")
    parser.add_argument("--input", required=True, help="Path to .pth stats file")
    parser.add_argument("--output", default="stats.json", help="Output JSON path")
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
