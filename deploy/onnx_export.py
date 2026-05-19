"""
Export HPENet V2 radar segmentation model to ONNX format.

Usage:
    python deploy/onnx_export.py [--checkpoint PATH] [--output PATH] [--device cpu|cuda]

Requires the model modules to be importable (run from project root with
the openpoints conda environment activated).
"""

import os
import sys

# Add project root to path (mirrors examples/segmentation/__init__.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np

from openpoints.utils.config import EasyConfig
from openpoints.models.build import build_model_from_cfg
from deploy.onnx_backend import patch_model_for_onnx


class HPENetONNXWrapper(nn.Module):
    """Wraps BaseSeg model so the forward takes two tensors instead of a dict."""

    def __init__(self, base_seg_model):
        super().__init__()
        self.model = base_seg_model

    def forward(self, pos, x):
        return self.model({'pos': pos, 'x': x})


def load_config():
    """Load the 3-level config cascade."""
    cfg = EasyConfig()
    cfg.load('cfgs/radar/hpenet-l.yaml', recursive=True)
    return cfg


def load_checkpoint(model, checkpoint_path):
    """Load weights from a training checkpoint into the model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = None
    for key in ('model', 'state_dict'):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None:
        state_dict = checkpoint

    # Strip 'module.' prefix (DDP wrapper)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    return model


def export_onnx(wrapped_model, output_path, num_points=30000):
    """Trace and export the model to ONNX.

    Args:
        wrapped_model: HPENetONNXWrapper instance
        output_path: path to write the .onnx file
        num_points: dummy input point count (max voxel size from config)
    """
    wrapped_model.eval()
    wrapped_model.cpu()

    B, N, C_in = 1, num_points, 4  # 4 = rcs, snr, v, z_height
    device = 'cpu'

    pos = torch.randn(B, N, 3, device=device)
    x = torch.randn(B, C_in, N, device=device)

    # Only mark the point-count dimension as dynamic.
    # Keeping batch=1 static avoids InstanceNorm1d "unknown channel size" errors
    # in the ONNX exporter.
    dynamic_axes = {
        'pos': {1: 'npoints'},
        'x': {2: 'npoints'},
        'output': {2: 'npoints'},
    }

    print(f"  Exporting with dummy input: pos={tuple(pos.shape)}, x={tuple(x.shape)}")
    print(f"  Dynamic axes: batch, npoints")

    torch.onnx.export(
        wrapped_model,
        (pos, x),
        output_path,
        input_names=['pos', 'x'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"  Saved to: {output_path}")

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model check: PASSED")
    return output_path


def test_pytorch_forward(model, wrapped_model):
    """Run a forward pass to verify patched model produces sensible output."""
    B, N = 1, 1024
    pos = torch.randn(B, N, 3)
    x = torch.randn(B, 4, N)

    with torch.no_grad():
        out = wrapped_model(pos, x)

    print(f"  Test forward: input pos={tuple(pos.shape)} x={tuple(x.shape)}")
    print(f"  Output shape: {tuple(out.shape)}")
    print(f"  Output value range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    if torch.isnan(out).any():
        raise RuntimeError("Output contains NaN! Model may have issues.")
    if torch.isinf(out).any():
        raise RuntimeError("Output contains Inf! Model may have issues.")

    return out


def main():
    parser = argparse.ArgumentParser(description='Export HPENet V2 to ONNX')
    parser.add_argument('--checkpoint', type=str,
                        default='log/radar/radar-train-hpenet-l-ngpus1-20260515-013127-HXWMALkaAC4GiUWjNV5c3g/checkpoint/radar-train-hpenet-l-ngpus1-20260515-013127-HXWMALkaAC4GiUWjNV5c3g_ckpt_best.pth',
                        help='Path to the checkpoint .pth file')
    parser.add_argument('--output', type=str,
                        default='deploy/onnx_model.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--num_points', type=int, default=30000,
                        help='Maximum number of points for dummy export input (voxel_max)')
    parser.add_argument('--no_patch', action='store_true',
                        help='Skip ONNX operator patching (debug only – will fail with CUDA ops)')
    args = parser.parse_args()

    print('=' * 60)
    print('HPENet V2 → ONNX Export')
    print('=' * 60)

    # 1. Load config
    print('\n[1/5] Loading config cascade...')
    cfg = load_config()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    print(f'  Model: {cfg.model.NAME}')
    print(f'  Encoder: {cfg.model.encoder_args.NAME}')
    print(f'  Input channels: {cfg.model.encoder_args.in_channels}')
    print(f'  Num classes: {cfg.model.cls_args.num_classes}')

    # 2. Build model
    print('\n[2/5] Building model...')
    model = build_model_from_cfg(cfg.model)

    # 3. Load checkpoint
    print(f'\n[3/5] Loading checkpoint: {args.checkpoint}')
    model = load_checkpoint(model, args.checkpoint)
    model.eval()

    # 4. Patch for ONNX
    if not args.no_patch:
        print('\n[4/5] Patching custom CUDA ops for ONNX compatibility...')
        model = patch_model_for_onnx(model)
    else:
        print('\n[4/5] Skipping patch (--no_patch). Export will likely FAIL.')

    # Wrap for ONNX-friendly inputs
    wrapped_model = HPENetONNXWrapper(model)

    # Quick test
    print('\n[5/5] Testing & exporting...')
    out = test_pytorch_forward(model, wrapped_model)

    # Export
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    export_onnx(wrapped_model, args.output, num_points=args.num_points)

    print(f'\nDone! ONNX model: {args.output}')
    print(f'  File size: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB')


if __name__ == '__main__':
    main()
