"""
Build TensorRT engine from ONNX model.

Uses the NEW TensorRT 8.6 API:
  - NetworkDefinitionCreationFlag.EXPLICIT_BATCH
  - builder.create_optimization_profile() → profile.set_shape()
  - config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, ...)
  - config.set_flag(BuilderFlag.FP16) for FP16 support

Usage:
    # FP32 engine
    python deploy/trt_build.py

    # FP16 engine (2x speedup)
    python deploy/trt_build.py --fp16

    # Custom profile
    python deploy/trt_build.py --min_n 64 --opt_n 4096 --max_n 30000
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tensorrt as trt
from deploy.trt_utils import setup_trt_env, _TRT_LIB_PATH, _CUDA_LIB_PATH


def build_engine(onnx_path, engine_path, fp16=False,
                 min_n=1024, opt_n=4096, max_n=30000,
                 workspace_gb=2):
    """Build a serialized TensorRT engine from an ONNX model.

    Args:
        onnx_path:   Path to input .onnx file.
        engine_path: Path to output .engine file.
        fp16:        Enable FP16 inference optimization.
        min_n:       Minimum number of points per input.
        opt_n:       Optimal number of points (used for kernel tuning).
        max_n:       Maximum number of points per input.
        workspace_gb: GPU workspace in GiB.

    Returns:
        bytes: The serialized engine.
    """
    logger = trt.Logger(trt.Logger.WARNING)

    # ---------- 1. Create builder & network ----------
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger)

    # ---------- 2. Parse ONNX ----------
    print(f"  Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        errors = []
        for i in range(parser.num_errors):
            err = parser.get_error(i)
            errors.append(f"    [{err.code()}] {err.desc()}")
        raise RuntimeError(
            f"ONNX parsing failed with {parser.num_errors} error(s):\n" + "\n".join(errors)
        )

    print(f"  ONNX parsed OK.  Inputs: {network.num_inputs}, outputs: {network.num_outputs}")

    # ---------- 3. Dynamic shape optimization profile ----------
    # Only 'npoints' dimension is dynamic (batch=1, channels=3/4 fixed)
    profile = builder.create_optimization_profile()

    # pos: (1, N, 3) — dynamic on dim 1
    profile.set_shape("pos",
                      (1, min_n, 3),
                      (1, opt_n, 3),
                      (1, max_n, 3))

    # x: (1, 4, N) — dynamic on dim 2
    profile.set_shape("x",
                      (1, 4, min_n),
                      (1, 4, opt_n),
                      (1, 4, max_n))

    # ---------- 4. Builder configuration ----------
    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))

    # Enable TF32 (available on L20 / Ampere+)
    if hasattr(config, 'set_flag') and hasattr(trt.BuilderFlag, 'TF32'):
        # TF32 is enabled by default on Ampere; set it explicitly
        pass

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode: ENABLED")

    print(f"  Dynamic profile: min_n={min_n}, opt_n={opt_n}, max_n={max_n}")
    print(f"  Workspace: {workspace_gb} GiB")
    print(f"  Building engine (this may take several minutes)...")

    # ---------- 5. Build ----------
    serialized_network = builder.build_serialized_network(network, config)
    if serialized_network is None:
        raise RuntimeError("Engine build failed (returned None). "
                           "This may be caused by an unsupported ONNX op, "
                           "insufficient workspace, or driver mismatch.")

    # TensorRT 8.6: build_serialized_network returns IHostMemory
    serialized = serialized_network if isinstance(serialized_network, bytes) else bytes(serialized_network)

    # ---------- 6. Save ----------
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = len(serialized) / 1024 / 1024
    print(f"  Engine saved: {engine_path} ({size_mb:.1f} MB)")
    return serialized


def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine from ONNX model (TRT 8.6.1+)"
    )
    parser.add_argument("--onnx", type=str, default="deploy/onnx_model.onnx",
                        help="Path to input ONNX model")
    parser.add_argument("--output", type=str, default=None,
                        help="Output engine path (default: trt_model_fp{32,16}.engine)")
    parser.add_argument("--fp16", action="store_true",
                        help="Enable FP16 inference (approx 2x speedup)")
    parser.add_argument("--min_n", type=int, default=1024,
                        help="Minimum point count for dynamic profile")
    parser.add_argument("--opt_n", type=int, default=4096,
                        help="Optimal point count for dynamic profile")
    parser.add_argument("--max_n", type=int, default=30000,
                        help="Maximum point count for dynamic profile")
    parser.add_argument("--workspace", type=int, default=2,
                        help="GPU workspace in GiB")
    args = parser.parse_args()

    if args.output is None:
        suffix = "fp16" if args.fp16 else "fp32"
        args.output = f"deploy/trt_model_{suffix}.engine"

    print("=" * 60)
    print("HPENet V2 → TensorRT Engine Builder (TRT 8.6)")
    print("=" * 60)
    print(f"  ONNX:   {args.onnx}")
    print(f"  Output: {args.output}")
    print(f"  FP16:   {args.fp16}")
    print(f"  Profile: min_n={args.min_n}, opt_n={args.opt_n}, max_n={args.max_n}")

    # Build environment requires TRT libraries
    setup_trt_env()

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.output,
        fp16=args.fp16,
        min_n=args.min_n,
        opt_n=args.opt_n,
        max_n=args.max_n,
        workspace_gb=args.workspace,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
