# openpoints/cpp/ — CUDA C++ Extensions

## OVERVIEW

5 compiled C++/CUDA extensions providing point cloud ops. Built via `python setup.py install` during `install.sh`. Must build AFTER PyTorch (depends on `torch.utils.cpp_extension`). 注意 install.sh 本身是过时上游脚本（py3.7），实际环境为 py3.10 手动构建。

**当前编译状态** (py3.10 .so): pointnet2_batch / pointops / chamfer_dist / subsampling 已编译；**emd 未编译**（系统内无 emd_cuda .so）。

## EXTENSIONS

| Extension | Dir | What it does | Build | CUDA? |
|-----------|-----|-------------|-------|-------|
| `pointnet2_batch` | `pointnet2_batch/` | Ball query, group points, interpolation, FPS sampling | `install` | Yes |
| `pointops` | `pointops/` | 7 ops: aggregation, ball query, grouping, interpolation, KNN, sampling, subtraction | `install` | Yes |
| `subsampling` | `subsampling/` | Grid subsampling (CPU, nanoflann) | `build_ext --inplace` | No |
| `chamfer_dist` | `chamfer_dist/` | Chamfer distance for reconstruction | `install --user` | Yes |
| `emd` | `emd/` | Earth Mover Distance | `install --user` | Yes |

## POINTOPS STRUCTURE (largest)

```
pointops/
├── functions/pointops.py    # Python API façade — 7 ops unified
└── src/
    ├── pointops_api.cpp      # PyTorch binding entry
    ├── aggregation/          # {op}_cuda.cpp + {op}_cuda_kernel.cu + .h
    ├── ballquery/            # same 3-file template
    ├── grouping/             # ...
    ├── interpolation/
    ├── knnquery/
    ├── sampling/
    └── subtraction/
```

Each op follows a rigid 3-file template. To add a new op: copy template, rename, add binding in `pointops_api.cpp`, register in `functions/pointops.py`.

## BUILD GOTCHAS

- **Missing GPU archs**: `TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"` — missing **8.6 (RTX 3090), 8.9 (RTX 4090), 9.0 (H100)**. Add before building for newer GPUs.
- **Inconsistent build**: `pointnet2_batch` → `install`, `subsampling` → `build_ext --inplace`, `chamfer_dist/emd` → `install --user`. `.so` files end up in different locations.
- **`subsampling` is CPU-only**: No CUDA. Extra flag `-D_GLIBCXX_USE_CXX11_ABI=0`.
- **Only one unittest**: `chamfer_dist/test.py` uses `unittest.TestCase`. Others are manual scripts.
- **nanoflann duplicated**: Header-only lib in 3 locations. Don't modify one without checking others.

## ADDING A NEW EXTENSION

1. Create dir with `setup.py` using `CUDAExtension` (or `distutils` for CPU)
2. Follow 3-file template: `{name}_cuda.cpp` + `{name}_cuda_kernel.cu` + `{name}_cuda_kernel.h`
3. Add build step to `install.sh`
4. Verify `.so` loads with a quick import test
