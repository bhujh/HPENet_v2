#!/usr/bin/env python3
"""Convert PyTorch .pth stats file to JSON for C++ consumption."""

import argparse
import json
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pth feature stats to JSON"
    )
    parser.add_argument("--input", required=True, help="Path to .pth stats file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    stats = torch.load(args.input, map_location="cpu")

    output = {
        "feat_mean": stats["feat_mean"].tolist(),  # list of 3 floats
        "feat_std": stats["feat_std"].tolist(),    # list of 3 floats
        "z_mean": float(stats["z_mean"]),           # scalar
        "z_std": float(stats["z_std"]),             # scalar
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Stats written to {args.output}")


if __name__ == "__main__":
    main()
