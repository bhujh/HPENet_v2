import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deploy.common import load_data_ply, preprocess_test, load_stats, preprocess_subcloud

import time
import argparse
import numpy as np
import torch

from tqdm import tqdm

from openpoints.dataset.data_util import voxelize
from openpoints.utils.config import EasyConfig
from openpoints.models.build import build_model_from_cfg
from deploy.onnx_backend import patch_model_for_onnx
from deploy.trt_utils import setup_trt_env, TRTSession

# 确认测试模式时是否有min_n>1024的限制，
def pad_subcloud(pos, x, min_n):
    """Pad to min_n points by replicating the last point."""
    N = pos.shape[1]
    if N >= min_n:
        return pos, x
    pad = min_n - N
    pos_pad = np.concatenate(
        [pos, np.tile(pos[:, -1:, :], (1, pad, 1))], axis=1
    ).astype(np.float32)
    x_pad = np.concatenate(
        [x, np.tile(x[:, :, -1:], (1, 1, pad))], axis=2
    ).astype(np.float32)
    return pos_pad, x_pad


def infer_one_cloud_trt(session, coord, feat, idx_points, feat_mean, feat_std,
                        z_mean, z_std, min_n=1024, gravity_dim=2):
    """TRT inference on all sub-clouds of one point cloud."""
    all_logits = []
    all_idx = []

    for idx_part in idx_points:
        pos_batch, x_batch = preprocess_subcloud(
            coord, feat, idx_part, feat_mean, feat_std, z_mean, z_std, gravity_dim,
        )

        N_true = pos_batch.shape[1]
        pos_batch, x_batch = pad_subcloud(pos_batch, x_batch, min_n)

        logits = session.run(pos_batch, x_batch)  # (1, 2, N_padded)

        # Trim padding
        if N_true < min_n:
            logits = logits[:, :, :N_true]

        all_logits.append(torch.from_numpy(logits[0]))  # (2, N_part)
        all_idx.append(torch.from_numpy(idx_part).long())

    # Merge with scatter mean
    all_logits_cat = torch.cat(all_logits, dim=1)
    all_logits_cat = all_logits_cat.transpose(0, 1)
    idx_flat = torch.cat(all_idx, dim=0)

    from torch_scatter import scatter
    merged = scatter(all_logits_cat, idx_flat, dim=0, reduce="mean")
    return merged


def infer_one_cloud_onnx(session, coord, feat, idx_points, feat_mean, feat_std,
                         z_mean, z_std, gravity_dim=2):
    """ONNX Runtime inference on all sub-clouds."""
    all_logits = []
    all_idx = []

    for idx_part in idx_points:
        pos_batch, x_batch = preprocess_subcloud(
            coord, feat, idx_part, feat_mean, feat_std, z_mean, z_std, gravity_dim,
        )

        outputs = session.run(None, {"pos": pos_batch, "x": x_batch})
        logits = outputs[0]

        all_logits.append(torch.from_numpy(logits[0]))
        all_idx.append(torch.from_numpy(idx_part).long())

    all_logits_cat = torch.cat(all_logits, dim=1).transpose(0, 1)
    idx_flat = torch.cat(all_idx, dim=0)

    from torch_scatter import scatter
    merged = scatter(all_logits_cat, idx_flat, dim=0, reduce="mean")
    return merged


def main():
    parser = argparse.ArgumentParser(description="HPENet V2 TensorRT Inference")
    parser.add_argument("--engine", type=str, default="deploy/trt_model_feat5_fp32.engine",
                        help="Path to TensorRT engine")
    parser.add_argument("--onnx", type=str, default="deploy/onnx_model_feat5.onnx",
                        help="Path to ONNX model (for comparison)")
    parser.add_argument("--checkpoint", type=str,
                        default="log/radar/radar-train-hpenet-ll-ngpus1-20260625-144233-c5U2epnpA9JLFW53JxxUSj/checkpoint/radar-train-hpenet-ll-ngpus1-20260625-144233-c5U2epnpA9JLFW53JxxUSj_ckpt_best.pth",
                        help="Path to PyTorch checkpoint (for comparison)")
    parser.add_argument('--cfgPath', type=str,
                        default='cfgs/radar/hpenet-ll.yaml',
                        help='Path to the cfg .yaml file')
    parser.add_argument("--data_dir", type=str,
                        default="data/RadarClassi/radarfullwl/raw",
                        help="Directory of test PLY files")
    parser.add_argument("--stats_file", type=str,
                        default="data/RadarClassi/radarfullwl/processed/feat_stats_area5.pth",
                        help="Feature statistics file")
    parser.add_argument("--num_files", type=int, default=10,
                        help="Number of test files (use -1 for all)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare TRT vs ONNX vs PyTorch outputs")
    parser.add_argument("--min_n", type=int, default=1024,
                        help="Minimum sub-cloud size (smaller clouds padded)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup runs")
    args = parser.parse_args()

    print("=" * 60)
    print("HPENet V2 TensorRT Inference")
    print("=" * 60)

    # --- Env ---
    setup_trt_env()
    # load config param
    cfg = EasyConfig()
    cfg.load(args.cfgPath, recursive=True)
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    # --- Load TRT ---
    print(f"\n[1/4] Loading TRT engine: {args.engine}")
    trt_session = TRTSession(args.engine)
    print(trt_session)

    # --- Warmup ---
    print(f"\n[2/4] Warmup ({args.warmup} runs)...")
    warmup_pos = np.random.randn(1, args.min_n, 3).astype(np.float32)
    warmup_x = np.random.randn(1, cfg.model.encoder_args.in_channels, args.min_n).astype(np.float32)
    for _ in range(args.warmup):
        trt_session.run(warmup_pos, warmup_x)
    print("  Warmup done.")


    onnx_session = None


    # --- Load data files ---
    print(f"\n[4/4] Loading test files from: {args.data_dir}")
    all_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".ply")])
    np.random.seed(100)
    np.random.shuffle(all_files)
    n_total = len(all_files)
    test_files = all_files[int(n_total * 0.83):]
    if args.num_files > 0:
        test_files = test_files[:args.num_files]
    print(f"  Total PLY files: {n_total}")
    print(f"  Test files: {len(test_files)}")

    # --- Load stats ---
    feat_mean, feat_std, z_mean, z_std = load_stats(args.stats_file)
    gravity_dim = 2

    # --- Run inference ---
    print(f"\n{'='*80}")
    header = f"  {'File':<15s} {'TRT_acc':>8s}  {'TRT_time':>10s}"
    if onnx_session:
        header += f"  {'ONNX_acc':>8s}  {'PT_acc':>8s}  {'PredMatch':>10s}"
    print(header)
    print("-" * 80)

    total_trt_time = 0.0
    trt_accs = []
    num_with_gt = 0

    for cloud_idx, fname in enumerate(tqdm(test_files, desc="Inference")):
        data_path = os.path.join(args.data_dir, fname)
        coord, feat, label = load_data_ply(data_path)
        print("coord:",coord.shape)
        coord, feat, idx_points, _, _, _ = preprocess_test(
            coord, feat, voxel_size=cfg.model.encoder_args.radius,
        )
        label_t = torch.from_numpy(label.astype(np.int64))

        # TRT inference
        t0 = time.time()
        logits_trt = infer_one_cloud_trt(
            trt_session, coord, feat, idx_points,
            feat_mean, feat_std, z_mean, z_std, min_n=args.min_n,
        )
        trt_time = time.time() - t0
        total_trt_time += trt_time
        pred_trt = logits_trt.argmax(dim=1)
        print("pred_trt:",pred_trt.shape)
        acc_trt = (pred_trt == label_t).float().mean().item()
        trt_accs.append(acc_trt)
        num_with_gt += 1

        line = f"  {fname:<15s} {acc_trt:>8.4f}  {trt_time:>9.3f}s"

        if onnx_session:
            logits_onnx = infer_one_cloud_onnx(
                onnx_session, coord, feat, idx_points,
                feat_mean, feat_std, z_mean, z_std,
            )
            pred_onnx = logits_onnx.argmax(dim=1)
            acc_onnx = (pred_onnx == label_t).float().mean().item()


            match = (pred_trt == pred_onnx).float().mean().item()
            line += f"  {acc_onnx:>8.4f}  {match:>9.4f}"

        tqdm.write(line)

    print("-" * 80)
    print(f"  Mean TRT accuracy: {np.mean(trt_accs):.4f}")
    print(f"  Mean TRT time:     {total_trt_time/num_with_gt:.3f}s per file")
    print(f"  Total TRT time:    {total_trt_time:.1f}s for {num_with_gt} files")

    print("\nDone!")


if __name__ == "__main__":
    main()