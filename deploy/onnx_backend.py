"""
Pure PyTorch implementations of custom CUDA operations for ONNX export.

All functions are traceable by torch.jit.trace / torch.onnx.export.
Replaces the 5 custom CUDA ops from openpoints/cpp/pointnet2_batch/:

    furthest_point_sample → traceable_random_fps
    ball_query            → traceable_ball_query
    grouping_operation    → traceable_grouping_operation
    three_nn              → (inside traceable_three_interpolation)
    three_interpolate     → (inside traceable_three_interpolation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ONNXInstanceNorm1d(nn.Module):
    """InstanceNorm replacement that is ONNX-exportable.

    Handles 1D (B,C,N), 2D (B,C,H,W), and 3D (B,C,D,H,W) inputs
    by normalizing over all spatial dimensions (dims 2..).
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, track_running_stats=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x: (B, C, ...) — normalize over all spatial dims
        ndim = x.dim()
        # Compute mean and var across spatial dims (2 to ndim-1)
        spatial_dims = list(range(2, ndim))
        mean = x.mean(dim=spatial_dims, keepdim=True)  # (B, C, 1, ...)
        var = ((x - mean) ** 2).mean(dim=spatial_dims, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Reshape weight/bias for broadcasting
        shape = [1, -1] + [1] * (ndim - 2)
        return x_norm * self.weight.view(*shape) + self.bias.view(*shape)


def traceable_random_fps(xyz, npoint):
    """Deterministic strided sampling to replace FPS (ONNX traceable).

    Takes the first `npoint` points from the input (each sub-cloud is already
    shuffled during voxelize, so this is effectively random).  Uses arange to
    avoid torch.randint (which requires constant high) and sort/argsort
    (which uses TopK that hits TRT's K ≤ 3840 limit).

    Args:
        xyz: (B, N, 3) point coordinates
        npoint: int, number of points to sample (can be dynamic/traced)

    Returns:
        idx: (B, npoint) indices, dtype=long
    """
    B, N, _ = xyz.shape
    # torch.arange with dynamic N is supported in ONNX opset 11+ via Range op
    idx = torch.arange(N, device=xyz.device, dtype=torch.long)
    idx = idx.unsqueeze(0).expand(B, -1)
    return idx[:, :npoint].contiguous()


def traceable_ball_query(radius, nsample, xyz, new_xyz):
    """Ball query using torch.cdist + topk (ONNX traceable).

    Handles the case where N < nsample (deep encoder stages with very
    few points) by padding the distance matrix.

    Args:
        radius: float, ball radius (must be a constant)
        nsample: int, maximum number of neighbors (must be a constant)
        xyz: (B, N, 3) support point coordinates
        new_xyz: (B, npoint, 3) query point coordinates

    Returns:
        idx: (B, npoint, nsample) neighbor indices, dtype=long.
             Out-of-radius / out-of-range positions are filled with index 0.
    """
    B, N, _ = xyz.shape
    npoint = new_xyz.shape[1]

    dist = torch.cdist(new_xyz, xyz)  # (B, npoint, N)

    large_val = torch.tensor(1e10, dtype=dist.dtype, device=dist.device)

    # Mask points beyond radius
    dist_masked = torch.where(dist <= radius, dist, large_val)

    # Pad with nsample extra columns of large_val so that N+nsample >= nsample
    # always holds.  This prevents topk(k=nsample) from failing when N < nsample
    # (which happens in deep encoder stages).
    padding = large_val.expand(B, npoint, nsample)
    dist_padded = torch.cat([dist_masked, padding], dim=-1)  # (B, npoint, N+nsample)

    _, idx = torch.topk(dist_padded, k=nsample, dim=-1, largest=False)

    # Reset indices that point to padding columns (>= N) to 0
    idx = torch.where(idx >= N, torch.zeros_like(idx), idx)

    # Reset entries that have no valid neighbor within radius to 0
    valid_count = (dist <= radius).sum(dim=-1, keepdim=True)  # (B, npoint, 1)
    pos_range = torch.arange(nsample, device=xyz.device).view(1, 1, nsample)
    out_of_radius = pos_range >= valid_count  # (B, npoint, nsample)
    idx = torch.where(out_of_radius, torch.zeros_like(idx), idx)

    return idx.long()


def traceable_grouping_operation(features, idx):
    """Group features by indices using torch.gather (ONNX traceable).

    Args:
        features: (B, C, N) feature tensor
        idx: (B, npoint, nsample) or (B, npoint) index tensor

    Returns:
        grouped: (B, C, npoint, nsample) or (B, C, npoint)
    """
    idx_flat = idx.reshape(idx.shape[0], -1)  # (B, npoint * nsample)
    idx_flat = idx_flat.unsqueeze(1).expand(-1, features.shape[1], -1)  # (B, C, npoint*nsample)
    grouped = torch.gather(features, 2, idx_flat.to(torch.int64))
    return grouped.reshape(idx.shape[0], features.shape[1], *idx.shape[1:])


def traceable_three_interpolation(unknown_xyz, known_xyz, know_feat):
    """Three-nearest-neighbor linear interpolation (ONNX traceable).

    Replaces the combined three_nn + three_interpolate CUDA pipeline
    with pure torch.cdist + gather.

    Args:
        unknown_xyz: (B, N, 3)  target point coordinates
        known_xyz:   (B, M, 3)  source point coordinates
        know_feat:   (B, C, M)  source features

    Returns:
        interpolated: (B, C, N)
    """
    B, C, M = know_feat.shape
    _, N, _ = unknown_xyz.shape

    # Compute pairwise distances
    dist = torch.cdist(unknown_xyz, known_xyz)  # (B, N, M)

    # Get 3 nearest neighbors
    dist_top3, idx_top3 = torch.topk(dist, k=3, dim=-1, largest=False)  # (B, N, 3)

    # Compute inverse distance weights
    dist_recip = 1.0 / (dist_top3 + 1e-8)  # (B, N, 3)
    norm = torch.sum(dist_recip, dim=-1, keepdim=True)  # (B, N, 1)
    weight = dist_recip / norm  # (B, N, 3)

    # Gather 3 neighbor features: (B, C, M) → (B, C, N, 3)
    idx_flat = idx_top3.reshape(B, -1)  # (B, N * 3)
    idx_flat = idx_flat.unsqueeze(1).expand(-1, C, -1)  # (B, C, N * 3)
    gathered = torch.gather(know_feat, 2, idx_flat.to(torch.int64))  # (B, C, N * 3)
    gathered = gathered.reshape(B, C, N, 3)  # (B, C, N, 3)

    # Weighted sum: (B, C, N, 3) * (B, 1, N, 3) → sum over dim=-1 → (B, C, N)
    weight = weight.unsqueeze(1)  # (B, 1, N, 3)
    interpolated = torch.sum(gathered * weight, dim=-1)  # (B, C, N)

    return interpolated


def patch_model_for_onnx(model):
    """Replace custom CUDA ops with pure-PyTorch equivalents in-place.

    Patches both:
    1. Module-level function references (resolved at call time)
    2. SetAbstraction.sample_fn attributes (captured at __init__ time)

    Args:
        model: a BaseSeg model with HPENetV2Encoder + HPENetV2Decoder

    Returns:
        model: the same model (modified in-place)
    """
    import openpoints.models.backbone.hpenetv2 as hpenetv2_mod
    import openpoints.models.layers.group as group_mod

    # Patch module-level functions resolved at call time
    hpenetv2_mod.grouping_operation = traceable_grouping_operation
    hpenetv2_mod.three_interpolation = traceable_three_interpolation
    group_mod.ball_query = traceable_ball_query
    group_mod.grouping_operation = traceable_grouping_operation

    # Walk model and replace sample_fn (captured at __init__ time)
    patched_count = 0
    for _name, module in model.named_modules():
        if hasattr(module, 'sample_fn'):
            module.sample_fn = traceable_random_fps
            patched_count += 1

    print(f"  Patched {patched_count} SetAbstraction.sample_fn: FPS → random sampling")

    # Replace InstanceNorm1d with ONNX-exportable version
    norm_patched = _replace_instance_norm(model)
    print(f"  Replaced {norm_patched} InstanceNorm1d → ONNXInstanceNorm1d")

    # Patch InvResMLP_block.forward and ResBlock.forward to remove
    # dynamic shape comparisons that create incompatible If nodes in ONNX.
    # The condition `f.shape[-1] == identity.shape[-1] and self.use_res`
    # is always True for stride-1 blocks, so we remove the check entirely.
    block_patched = _patch_residual_blocks(model)
    print(f"  Patched {block_patched} residual blocks (removed dynamic If)")

    # Remove .squeeze(-1) from BaseSeg.forward — it's a no-op when N > 1
    # but ONNX traces it as an If node with incompatible branch shapes.
    _patch_base_seg_squeeze(model)
    print(f"  Patched BaseSeg.forward (removed squeeze If)")

    return model


def _patch_base_seg_squeeze(model):
    """Replace BaseSeg.forward to remove .squeeze(-1) which causes ONNX If."""
    import openpoints.models.segmentation.base_seg as base_seg_mod

    model_cls = type(model)
    if model_cls.__name__ != 'BaseSeg':
        return

    encoder = model.encoder
    decoder = model.decoder
    head = model.head

    def forward(self, data):
        p, f = encoder.forward_seg_feat(data)
        if decoder is not None:
            f = decoder(p, f)  # removed .squeeze(-1)
        if head is not None:
            f = head(f)
        return f

    model.forward = forward.__get__(model)


def _patch_residual_blocks(module):
    """Replace InvResMLP_block.forward / ResBlock.forward with trace-safe versions."""
    import openpoints.models.backbone.hpenetv2 as hv2

    count = 0
    for _name, child in module.named_modules():
        clsname = type(child).__name__
        if clsname == 'InvResMLP_block':
            use_res = child.use_res
            child.forward = _make_invresmlp_block_forward(child, use_res)
            count += 1
        elif clsname == 'ResBlock':
            use_res = child.use_res
            child.forward = _make_resblock_forward(child, use_res)
            count += 1
    return count


def _make_invresmlp_block_forward(module, use_res):
    """Create a deterministic forward method without dynamic If."""
    convs = module.convs
    pwconv = module.pwconv
    act = module.act

    def forward(self, pf):
        p, f, dp, idx_dp, pe = pf
        identity = f
        f = convs([p, f, dp, idx_dp, pe])
        f = pwconv(f)
        if use_res:
            f = f + identity
        f = act(f)
        return [p, f, dp, idx_dp, pe]

    return forward.__get__(module)


def _make_resblock_forward(module, use_res):
    """Create a deterministic forward method without dynamic If."""
    convs = module.convs
    act = module.act

    def forward(self, pf):
        p, f = pf
        identity = f
        f = convs([p, f])
        if use_res:
            f = f + identity
        f = act(f)
        return [p, f]

    return forward.__get__(module)


def _replace_instance_norm(module):
    """Recursively replace nn.InstanceNorm1d/2d/3d with ONNXInstanceNorm1d."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            replacement = ONNXInstanceNorm1d(
                child.num_features,
                eps=child.eps,
            )
            # Copy weights (may be None when affine=False)
            if child.weight is not None:
                replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            setattr(module, name, replacement)
            count += 1
        else:
            count += _replace_instance_norm(child)
    return count
