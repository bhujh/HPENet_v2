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
import onnx

from openpoints.utils.config import EasyConfig
from openpoints.models.build import build_model_from_cfg
from deploy.onnx_backend import patch_model_for_onnx


# Input feature count for the radar model. MUST be kept in sync across
# export_onnx (dummy input), test_pytorch_forward (test input), and
# simplify_onnx (folding shape). Change here and all three update.
NUM_INPUT_FEATURES = 5


class HPENetONNXWrapper(nn.Module):
    """Wraps BaseSeg model so the forward takes two tensors instead of a dict."""

    def __init__(self, base_seg_model):
        super().__init__()
        self.model = base_seg_model

    def forward(self, pos, x):
        return self.model({'pos': pos, 'x': x})


def load_config(cfg_path:str):
    """Load the 3-level config cascade."""
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
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


def export_onnx(wrapped_model, output_path, num_points=3500):
    """Trace and export the model to ONNX.

    Args:
        wrapped_model: HPENetONNXWrapper instance
        output_path: path to write the .onnx file
        num_points: dummy input point count (max voxel size from config)
    """
    wrapped_model.eval()
    wrapped_model.cpu()

    B, N, C_in = 1, num_points, NUM_INPUT_FEATURES  # rcs, snr, v, z_height, +1
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
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"  Saved to: {output_path}")

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model check: PASSED")
    return output_path


def simplify_onnx(input_path, output_path, num_points=3500,
                  num_features=NUM_INPUT_FEATURES):
    """Run onnx-simplifier on the exported model.

    Uses a concrete input shape to fold constants on the dynamic 'npoints'
    dimension. The simplified output still keeps npoints dynamic.

    Returns output_path on success, or None if simplification failed. A None
    return is non-fatal — the raw exported model remains usable.
    """
    from onnxsim import simplify

    print(f"  Simplifying: {input_path}")
    onnx_model = onnx.load(input_path)

    # Concrete shapes let onnxsim run a real forward pass to fold constants.
    # Batch=1 (static) matches the export setting — InstanceNorm safe.
    input_shapes = {
        'pos': [1, num_points, 3],
        'x':   [1, num_features, num_points],
    }

    # Guard: input_shapes keys must match actual graph input names, otherwise
    # onnxsim silently ignores unmatched keys and falls back to symbolic mode.
    actual_inputs = [inp.name for inp in onnx_model.graph.input]
    missing = set(input_shapes.keys()) - set(actual_inputs)
    if missing:
        print(f"  [WARN] input_shapes keys {sorted(missing)} not found in graph "
              f"inputs {actual_inputs}.")
        print(f"  Simplification may fall back to symbolic mode (may fail check).")

    try:
        model_sim, check = simplify(onnx_model,
                                    overwrite_input_shapes=input_shapes)
    except Exception as e:  # noqa: BLE001 — onnxsim failure modes are varied
        print(f"  [WARN] onnxsim simplify() raised: {e}")
        print(f"  Skipping simplification — raw model is still usable.")
        return None

    if not check:
        print(f"  [WARN] onnxsim validity check FAILED (simplified != original).")
        print(f"  Keeping raw model: {input_path}")
        return None

    onnx.save(model_sim, output_path)
    onnx.checker.check_model(onnx.load(output_path))
    print(f"  Simplified model check: PASSED")
    print(f"  Saved to: {output_path}")

    # Numerical equivalence check on the same random input.
    import onnxruntime as ort

    pos_np = np.random.randn(1, num_points, 3).astype(np.float32)
    x_np = np.random.randn(1, num_features, num_points).astype(np.float32)
    try:
        sess_raw = ort.InferenceSession(input_path, providers=['CPUExecutionProvider'])
        sess_sim = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        out_raw = sess_raw.run(None, {'pos': pos_np, 'x': x_np})[0]
        out_sim = sess_sim.run(None, {'pos': pos_np, 'x': x_np})[0]
        max_diff = np.abs(out_raw - out_sim).max()
        print(f"  Numerical equivalence (max abs diff): {max_diff:.2e}")
        if max_diff > 1e-3:
            print(f"  [WARN] Diff > 1e-3, simplified model may diverge from raw.")
    except Exception as e:  # noqa: BLE001 — ORT inference failure is non-fatal
        print(f"  [WARN] Numerical check skipped (ORT inference failed): {e}")

    return output_path


def test_pytorch_forward(model, wrapped_model):
    """Run a forward pass to verify patched model produces sensible output."""
    B, N = 1, 3500
    pos = torch.randn(B, N, 3)
    x = torch.randn(B, NUM_INPUT_FEATURES, N)

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
                        default='log/radar/radar-train-hpenet-ll-ngpus1-20260625-144233-c5U2epnpA9JLFW53JxxUSj/checkpoint/radar-train-hpenet-ll-ngpus1-20260625-144233-c5U2epnpA9JLFW53JxxUSj_ckpt_best.pth',
                        help='Path to the checkpoint .pth file')
    parser.add_argument('--cfg', type=str,
                        default='cfgs/radar/hpenet-ll.yaml',
                        help='Path to the cfg .yaml file')
    parser.add_argument('--output', type=str,
                        default='deploy/onnx_model_feat5.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--num_points', type=int, default=3500,
                        help='Maximum number of points for dummy export input (voxel_max)')
    parser.add_argument('--no_patch', action='store_true',
                        help='Skip ONNX operator patching (debug only – will fail with CUDA ops)')
    parser.add_argument('--no_simplify', action='store_true',
                        help='Skip onnx-simplifier post-processing (simplify runs by default)')
    parser.add_argument('--simplified_output', type=str, default=None,
                        help='Output path for simplified ONNX (default: <output>_sim.onnx)')
    args = parser.parse_args()

    print('=' * 60)
    print('HPENet V2 → ONNX Export')
    print('=' * 60)

    # 1. Load config
    print('\n[1/5] Loading config cascade...')
    cfg = load_config(args.cfg)
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    print(f'  Model: {cfg.model.NAME}')
    print(f'  Encoder: {cfg.model.encoder_args.NAME}')
    print(f'  Input channels: {cfg.model.encoder_args.in_channels}')
    print(f'  Num classes: {cfg.model.cls_args.num_classes}')

    # 2. Build model
    print('\n[2/5] Building model...')
    model = build_model_from_cfg(cfg.model)
    print(model)
    # os.abort()

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

    raw_size = os.path.getsize(args.output) / 1024 / 1024
    print(f'\nDone! ONNX model: {args.output}')
    print(f'  File size: {raw_size:.1f} MB')

    if args.no_simplify:
        print(f'\nSimplification skipped (--no_simplify).')
    else:
        print('\n--- Simplifying ONNX model (onnx-simplifier) ---')
        if args.simplified_output:
            sim_path = args.simplified_output
        elif args.output.endswith('.onnx'):
            sim_path = args.output[:-5] + '_sim.onnx'
        else:
            sim_path = args.output + '_sim.onnx'
        result = simplify_onnx(args.output, sim_path,
                               num_points=args.num_points,
                               num_features=NUM_INPUT_FEATURES)
        if result is not None:
            sim_size = os.path.getsize(result) / 1024 / 1024
            delta_pct = (sim_size - raw_size) / raw_size * 100
            print(f'  Size: {raw_size:.1f} MB → {sim_size:.1f} MB '
                  f'({delta_pct:+.1f}%)')
            if delta_pct >= 0:
                os.remove(result)
                print(f'  [INFO] Simplification produced no size benefit '
                      f'({delta_pct:+.1f}%). Removed {result}.')
                print(f'  Use raw model: {args.output}')
            else:
                print(f'  Simplified model: {result}')
        else:
            print(f'  Simplification skipped, raw model remains: {args.output}')


if __name__ == '__main__':
    main()
